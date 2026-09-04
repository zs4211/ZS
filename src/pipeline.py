"""
雷达三维风场反演的数据处理流水线。

读取 CINRAD WRS 产品文件（通过 PyCINRAD 的 read_auto），转换为 PyART Radar 对象，
执行质量控制，并将数据网格化到笛卡尔网格上，为 PyDDA 三维变分分析风场反演做好准备。

工作流程:
  1. 使用 cinrad.io.read_auto 读取 RSTM 产品文件
  2. 构建多仰角 PyART Radar 对象（径向速度 + 反射率因子）
  3. 质量控制：退模糊、去斑滤波、衰减订正
  4. 使用 pyart.map.grid_from_radars 网格化到笛卡尔坐标系
  5. 保存为 CF/Radial NetCDF 格式，供 PyDDA 使用
"""

import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import warnings


class RadarDataPipeline:
    """处理 CINRAD WRS RSTM 数据的流水线。

    将原始雷达产品文件转换为适合 PyDDA 三维变分分析风场反演的
    网格化 PyART 对象。

    工作流程:
      1. discover_files(): 扫描数据目录，按产品类型组织文件
      2. build_volume(): 使用 cinrad.io.read_auto 解析 RSTM 文件
      3. sweeps_to_pyart_radar(): 将多仰角扫描组装为 PyART Radar 对象
      4. quality_control(): 去斑滤波 + 速度退模糊 + 极端值滤波
      5. grid_radar_data(): 使用 Cressman 加权方案网格化到笛卡尔坐标
      6. match_times(): 查找两部雷达之间的时间匹配对（用于双多普勒分析）

    产品类型映射 (PRODUCT_MAP):
      019/020 — Zc 反射率因子（高分辨率）
      026/027 — Vc 径向速度（高分辨率）
      158     — ZDRc 差分反射率
      160     — RHO 相关系数
      161     — PDP 差分相位
      162     — KDP 差分传播相位常数
      19/20   — 低分辨率反射率因子
      26/27   — 低分辨率径向速度
    """

    # WRS RSTM 格式中的产品组织:
    # 目录代码 -> (产品类型, 分辨率级别)
    # 每个目录下的第 01-06 层映射到实际的仰角
    PRODUCT_MAP = {
        "019": ("reflectivity", "high"),    # Zc (反射率因子)
        "020": ("reflectivity", "high"),    # Zc
        "026": ("velocity", "high"),        # Vc (径向速度)
        "027": ("velocity", "high"),        # Vc
        "026": ("velocity", "high"),        # Vc
        "158": ("zdr", "high"),             # ZDRc (差分反射率)
        "160": ("rho", "high"),             # RHO (相关系数)
        "161": ("phidp", "high"),           # PDP (差分相位)
        "162": ("kdp", "high"),             # KDP (差分传播相位常数)
        "19":  ("reflectivity", "low"),     # REF (低分辨率反射率)
        "20":  ("reflectivity", "low"),     # REF
        "26":  ("velocity", "low"),         # VEL (低分辨率径向速度)
        "27":  ("velocity", "low"),         # VEL
    }

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # 步骤1: 读取并组织文件
    # ------------------------------------------------------------------

    def discover_files(self, station: str, product_type: str = "velocity",
                       resolution: str = "high") -> Dict:
        """
        发现指定测站和产品类型的所有 RSTM 文件。

        返回以 (时间戳, 层号, 仰角编号) 为键、文件路径为值的字典。
        """
        ppi_dir = self.data_dir / station / "PPI"
        if not ppi_dir.exists():
            raise FileNotFoundError(f"PPI directory not found: {ppi_dir}")

        # 查找匹配目标产品类型的目录
        target_dirs = []
        for d in sorted(ppi_dir.iterdir()):
            if not d.is_dir():
                continue
            info = self.PRODUCT_MAP.get(d.name.upper(), (None, None))
            if info[0] == product_type and info[1] == resolution:
                target_dirs.append(d)

        # 收集所有文件，按 (时间戳, 层号) 组织
        files = defaultdict(dict)  # (timestamp, layer) -> {elev_code: path}
        for elev_dir in target_dirs:
            for f in elev_dir.iterdir():
                if f.name == "ProductIndex":
                    continue
                parts = f.name.split("_")
                if len(parts) < 4:
                    continue
                ts = parts[1]
                layer = parts[3]
                files[(ts, layer)][elev_dir.name] = f

        return dict(files)

    def build_volume(self, station: str, timestamp: str,
                     product_type: str = "velocity",
                     resolution: str = "high") -> List:
        """
        为单个时间戳构建多仰角雷达体扫数据。

        使用 cinrad.io.read_auto 解析 RSTM 文件，将所有仰角扫描
        收集到一个扫描字典列表中。

        返回包含以下键的字典列表: radar (PyART), elevation, data_type。
        """
        from cinrad.io import read_auto

        all_files = self.discover_files(station, product_type, resolution)
        sweeps = []

        # 查找该时间戳下的所有层
        for (ts, layer), elev_files in sorted(all_files.items()):
            if ts != timestamp:
                continue

            for elev_code, fpath in elev_files.items():
                try:
                    pup = read_auto(str(fpath))
                    ds = pup.get_data()

                    # 从元数据中确定实际仰角
                    elevation = float(ds.attrs.get("elevation",
                                        pup.params.get("elevation", 0.0)))

                    # 提取数据变量（排除坐标变量）
                    data_vars = [v for v in ds.data_vars
                                if v not in ("longitude", "latitude", "height")]
                    if not data_vars:
                        continue
                    var_name = data_vars[0]
                    field_data = ds[var_name].values

                    # 获取坐标
                    azimuth = ds["azimuth"].values
                    range_vals = ds["distance"].values

                    print(f"  [{station}] {product_type} elev={elevation:.1f}° "
                          f"({layer}/{elev_code}) shape={field_data.shape}, "
                          f"pname={pup.pname}")

                    sweeps.append({
                        "elevation": elevation,
                        "layer": layer,
                        "elev_code": elev_code,
                        "field_data": field_data,
                        "field_name": pup.pname,
                        "azimuth": azimuth,
                        "range": range_vals,
                        "station": station,
                        "station_lat": pup.stationlat,
                        "station_lon": pup.stationlon,
                        "station_alt": float(pup.radarheight),
                        "scan_time": pup.scantime,
                        "vcp": pup.task_name,
                        "nyquist": self._extract_nyquist(pup, elevation),
                        "filepath": str(fpath),
                    })
                except Exception as e:
                    print(f"  WARNING: failed to read {fpath.name}: {e}")

        return sorted(sweeps, key=lambda s: s["elevation"])

    @staticmethod
    def _extract_nyquist(pup, elevation: float) -> float:
        """
        从 RSTM 任务扫描配置中读取当前仰角的 Nyquist 速度。

        同一仰角可能包含双 PRF 的两条配置；PUP 的 ``Vc`` 是校正速度，
        此处记录较大的有效 Nyquist 值作为该层仪器参数。绝不再从观测
        极值反推 Nyquist，因为折叠/已校正速度的极值不代表仪器上限。
        """
        config = getattr(pup, "scan_config", None)
        if config is not None and getattr(config.dtype, "names", None):
            if "elev" in config.dtype.names and "nyquist_spd" in config.dtype.names:
                elev = np.asarray(config["elev"], dtype=float)
                nyquist = np.asarray(config["nyquist_spd"], dtype=float)
                valid = np.isfinite(elev) & np.isfinite(nyquist) & (nyquist > 0)
                if valid.any():
                    delta = np.abs(elev[valid] - float(elevation))
                    nearest = delta.min()
                    if nearest <= 0.25:
                        candidates = nyquist[valid][delta <= nearest + 0.02]
                        return float(candidates.max())
        warnings.warn(
            f"Nyquist velocity missing for elevation {elevation:.2f}; "
            "using conservative 30 m/s fallback",
            RuntimeWarning,
        )
        return 30.0

    # ------------------------------------------------------------------
    # 步骤2: 转换为 PyART Radar 对象
    # ------------------------------------------------------------------

    def sweeps_to_pyart_radar(self, sweeps: List[Dict]) -> "pyart.core.Radar":
        """
        将扫描字典列表转换为单个 PyART Radar 对象。

        Radar 对象包含所有仰角扫描。按 PyART 要求的格式逐层存储
        各扫描的场数据。
        """
        import pyart

        if not sweeps:
            raise ValueError("No sweeps to convert")

        # 按仰角排序，保留所有扫描（026 和 027 目录在相同仰角下包含不同旋转方向的数据）
        sweeps = sorted(sweeps, key=lambda s: (s["elevation"], s["elev_code"]))

        n_sweeps = len(sweeps)
        total_rays = sum(len(s["azimuth"]) for s in sweeps)

        # 确定最大距离库数（使用第一个扫描的距离库数）
        n_gates = len(sweeps[0]["range"])

        # 预分配所有 numpy 数组
        scan_time = sweeps[0]["scan_time"]
        # RSTM products expose an absolute, timezone-aware scan time.  PyART
        # stores ray times as offsets from the epoch in ``time['units']``.
        # Keep zero offsets for these PUP products (per-ray acquisition times
        # are not present), but use the real volume time as the epoch.
        if scan_time.tzinfo is None:
            scan_time = scan_time.replace(tzinfo=timezone.utc)
        scan_time_utc = scan_time.astimezone(timezone.utc)
        time_origin = scan_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        _time_data = np.zeros(total_rays, dtype=np.float64)
        _range_data = sweeps[0]["range"].astype(np.float64) * 1000.0  # km -> m
        azimuth_data = np.zeros(total_rays, dtype=np.float32)
        elevation_data = np.zeros(total_rays, dtype=np.float32)
        nyquist_data = np.zeros(total_rays, dtype=np.float32)
        sweep_start_ray = np.zeros(n_sweeps, dtype=np.int32)
        sweep_end_ray = np.zeros(n_sweeps, dtype=np.int32)
        fixed_angle_data = np.zeros(n_sweeps, dtype=np.float64)
        sweep_number_data = np.arange(n_sweeps, dtype=np.int32)

        # 场数据数组
        field_name = sweeps[0]["field_name"]
        field_data_all = np.ma.zeros((total_rays, n_gates), dtype=np.float32)
        field_data_all.mask = np.ones((total_rays, n_gates), dtype=bool)

        ray_start = 0
        for sweep_idx, sweep in enumerate(sweeps):
            n_rays = len(sweep["azimuth"])
            n_bins = min(len(sweep["range"]), n_gates)
            ray_end = ray_start + n_rays

            # cinrad 存储的方位角是弧度制 - 转换为度以满足 PyART 要求
            az_rad = sweep["azimuth"]
            if np.max(az_rad) < 7.0:  # 弧度范围 0-2π
                az_deg = np.degrees(az_rad)
            else:
                az_deg = az_rad.astype(np.float32)
            azimuth_data[ray_start:ray_end] = az_deg
            elevation_data[ray_start:ray_end] = sweep["elevation"]
            nyquist_data[ray_start:ray_end] = sweep["nyquist"]
            sweep_start_ray[sweep_idx] = ray_start
            sweep_end_ray[sweep_idx] = ray_end - 1
            fixed_angle_data[sweep_idx] = sweep["elevation"]
            _time_data[ray_start:ray_end] = 0.0

            # 复制场数据（仅将 NaN 值掩膜处理）
            # 注意：不能屏蔽值为 0 的数据——零径向速度是真实观测，
            # 出现在风向与雷达波束垂直的位置、正负速度对的分界线、
            # 以及中气旋速度偶极子中心附近，屏蔽会破坏涡度计算
            fdata = sweep["field_data"][:n_rays, :n_bins]
            mask = np.isnan(fdata)
            field_data = np.ma.array(fdata.astype(np.float32), mask=mask)
            field_data_all[ray_start:ray_end, :n_bins] = field_data

            ray_start = ray_end

        # 根据产品名称确定场元数据标准名称
        if field_name.upper() in ("VC", "VEL", "V"):
            standard_name = "radial_velocity_of_scatterers_away_from_instrument"
            long_name = "Radial Velocity"
            units = "m/s"
        elif field_name.upper() in ("ZC", "REF", "Z"):
            standard_name = "equivalent_reflectivity_factor"
            long_name = "Reflectivity"
            units = "dBZ"
        else:
            standard_name = field_name.lower()
            long_name = field_name
            units = ""

        fields = {
            field_name: {
                "data": field_data_all,
                "units": units,
                "standard_name": standard_name,
                "long_name": long_name,
                "_FillValue": -9999.0,
            }
        }

        # 雷达元数据
        s = sweeps[0]
        # 构建雷达仪器参数（奈奎斯特速度）
        instrument_parameters = {
            "nyquist_velocity": {
                "data": nyquist_data,
                "units": "m/s",
                "standard_name": "nyquist_velocity",
            }
        }

        radar = pyart.core.Radar(
            time={"data": _time_data, "units": f"seconds since {time_origin}",
                   "calendar": "gregorian"},
            _range={"data": _range_data, "units": "meters"},
            fields=fields,
            metadata={"vcp": s.get("vcp", "VCP11D"),
                       "original_container": "CINRAD_WRS_RSTM",
                       "velocity_is_corrected": field_name.upper() == "VC"},
            scan_type="ppi",
            latitude={"data": np.array([s["station_lat"]], dtype=np.float64),
                       "units": "degrees_north"},
            longitude={"data": np.array([s["station_lon"]], dtype=np.float64),
                        "units": "degrees_east"},
            altitude={"data": np.array([s["station_alt"]], dtype=np.float64),
                       "units": "meters"},
            altitude_agl=None,
            azimuth={"data": azimuth_data, "units": "degrees"},
            elevation={"data": elevation_data, "units": "degrees"},
            fixed_angle={"data": fixed_angle_data, "units": "degrees"},
            sweep_number={"data": sweep_number_data},
            sweep_mode={"data": np.array(["ppi"] * n_sweeps, dtype="S3")},
            sweep_start_ray_index={"data": sweep_start_ray},
            sweep_end_ray_index={"data": sweep_end_ray},
            target_scan_rate=None,
            rays_are_indexed=None,
            ray_angle_res=None,
            instrument_parameters=instrument_parameters,
            radar_calibration=None,
        )

        return radar

    # ------------------------------------------------------------------
    # 步骤3: 质量控制
    # ------------------------------------------------------------------

    def quality_control(self, radar: "pyart.core.Radar") -> "pyart.core.Radar":
        """
        对雷达数据应用质量控制，包括:
        - 去斑滤波（移除孤立噪声像素）
        - 速度退模糊（基于区域分割）
        - 极端值滤波（剔除物理上不可能的值）

        返回质量控制后的 Radar 对象。
        """
        import pyart

        field_name = list(radar.fields.keys())[0]

        # 步骤3a: 去斑滤波 - 移除孤立的噪声像素
        radar = self._despeckle(radar, field_name)

        # 步骤3b: 速度退模糊
        if (field_name.upper() in ("VEL", "V", "VELOCITY")
                and not radar.metadata.get("velocity_is_corrected", False)):
            radar = self._dealias_velocity(radar, field_name)
        elif field_name.upper() == "VC":
            print("  Velocity field Vc is already corrected; skipping duplicate dealiasing")

        # 步骤3c: 过滤极端值
        radar = self._filter_extremes(radar, field_name)

        return radar

    @staticmethod
    def _despeckle(radar, field_name: str):
        """移除孤立斑点噪声。

        对每个仰角扫描应用中值滤波（3x3 窗口），保留数据区域的连续性。
        正确处理 MaskedArray：对有效值应用中值滤波，保持原有掩膜不变。
        """
        from scipy.ndimage import median_filter

        data = radar.fields[field_name]["data"]
        for sweep in range(radar.nsweeps):
            s_start = radar.sweep_start_ray_index["data"][sweep]
            s_end = radar.sweep_end_ray_index["data"][sweep] + 1
            sweep_data = data[s_start:s_end]
            if sweep_data.size == 0:
                continue

            if np.ma.is_masked(sweep_data):
                # MaskedArray: 用 NaN 填充掩膜区域，滤波后恢复掩膜
                filled = np.ma.filled(sweep_data, np.nan)
                filtered = median_filter(filled, size=(3, 3))
                # 滤波后的 NaN 代表掩膜区域的扩散，重新构建掩膜
                new_mask = np.isnan(filtered) | sweep_data.mask
                radar.fields[field_name]["data"][s_start:s_end] = np.ma.array(
                    np.nan_to_num(filtered), mask=new_mask)
            else:
                # 普通数组: 直接滤波
                filtered = median_filter(sweep_data, size=(3, 3))
                radar.fields[field_name]["data"][s_start:s_end] = filtered
        return radar

    @staticmethod
    def _dealias_velocity(radar, field_name: str):
        """基于区域分割的 CINRAD 数据速度退模糊。

        使用 PyART 的区域退模糊算法。dealias_region_based 会在雷达对象中
        添加一个带 _dealiased 后缀的新字段，原速度场不会被自动替换。
        因此需要显式将退模糊结果覆盖回原字段，确保后续网格化使用正确数据。

        算法原理: 在径向速度场中搜索折叠区域边界，通过比较相邻像素的
        速度梯度来判断是否发生折叠，然后对折叠区域整体加上或减去
        2*Nyquist 的整数倍来恢复真实速度。
        """
        import pyart

        nyquist = radar.instrument_parameters["nyquist_velocity"]["data"][0]

        try:
            # 执行退模糊, 返回退模糊后的场字典
            corrected = pyart.correct.dealias_region_based(
                radar,
                vel_field=field_name,
                nyquist_vel=float(nyquist),
                centered=True,
                skip_checks=False,
            )
            # 将退模糊后的速度场覆盖原字段
            radar.add_field(field_name, corrected, replace_existing=True)
            print(f"  Velocity dealiasing applied and field replaced "
                  f"(Nyquist={nyquist:.1f} m/s)")
        except Exception as e:
            print(f"  Dealiasing warning (field not replaced): {e}")

        return radar

    @staticmethod
    def _filter_extremes(radar, field_name: str):
        """
        过滤极端的、物理上不可能的值。

        对于 S 波段 CINRAD，径向速度的合理范围为 ±100 m/s；
        反射率因子的合理范围为 -20 到 80 dBZ。超出这些范围的值
        通常是由二次回波、地物杂波或仪器故障造成的。
        """
        data = radar.fields[field_name]["data"]

        if field_name.upper() in ("VC", "VEL", "V"):
            # S 波段 CINRAD 的速度应在 ±100 m/s 范围内
            radar.fields[field_name]["data"] = np.ma.masked_where(
                np.abs(data) > 100.0, data)
        elif field_name.upper() in ("ZC", "REF", "Z"):
            # dBZ 应在 -20 到 80 之间
            radar.fields[field_name]["data"] = np.ma.masked_where(
                (data < -20) | (data > 80), data)

        return radar

    # ------------------------------------------------------------------
    # 步骤4: 笛卡尔网格化
    # ------------------------------------------------------------------

    def grid_radar_data(self, radars: List["pyart.core.Radar"],
                        grid_shape: Tuple[int, int, int] = (20, 201, 201),
                        grid_limits: Tuple[float, ...] = (
                            # (z_min, z_max), (y_min, y_max), (x_min, x_max)
                            (0.5, 15000.0),  # z 单位为米
                            (-150000.0, 150000.0),  # y 单位为米 (南北)
                            (-150000.0, 150000.0),  # x 单位为米 (东西)
                        ),
                        fields_to_grid: Optional[List[str]] = None,
                        ) -> "pyart.core.Grid":
        """
        使用 PyART 将雷达数据网格化到笛卡尔坐标系。

        使用 pyart.map.grid_from_radars 进行多雷达网格化，默认采用
        Cressman 距离加权插值方案。网格原点设为两部雷达之间的中点，
        以最小化边缘的投影误差。
        """
        import pyart

        if fields_to_grid is None:
            # 默认使用第一部雷达的第一个场
            fields_to_grid = [list(radars[0].fields.keys())[0]]

        # 构建网格坐标
        grid_z = np.linspace(grid_limits[0][0], grid_limits[0][1],
                             grid_shape[0])
        grid_y = np.linspace(grid_limits[1][0], grid_limits[1][1],
                             grid_shape[1])
        grid_x = np.linspace(grid_limits[2][0], grid_limits[2][1],
                             grid_shape[2])

        # 将网格原点设在两部雷达的中点，确保对称的覆盖范围
        mean_lat = np.mean([r.latitude["data"][0] for r in radars])
        mean_lon = np.mean([r.longitude["data"][0] for r in radars])
        print(f"  Grid origin: ({mean_lat:.4f}°N, {mean_lon:.4f}°E)")
        print(f"  Grid shape: {grid_shape}, limits: z={grid_limits[0]}m, "
              f"xy=±{grid_limits[1][1]/1000:.0f}km")

        # 执行网格化
        grid = pyart.map.grid_from_radars(
            radars,
            grid_shape=grid_shape,
            grid_limits=grid_limits,
            fields=fields_to_grid,
            grid_origin=(mean_lat, mean_lon),
            gridding_algo="map_gates_to_grid",
            weighting_function="Cressman",
            roi_func="dist_beam",
            constant_roi=3000.0,         # [修复] 增大到 1.5km
            copy_field_data=True,
            center_origin=(0.0, 0.0),
        )

        return grid

    # ------------------------------------------------------------------
    # 步骤5: 双多普勒时间匹配
    # ------------------------------------------------------------------

    @staticmethod
    def match_times(station1: str, station2: str,
                    data_dir: Path,
                    product_type: str = "velocity",
                    resolution: str = "high",
                    max_diff_minutes: float = 2.0) -> List[Tuple[str, str, float]]:
        """
        查找两部雷达之间适合双多普勒分析的时间匹配对。

        两部雷达的扫描时间差应在 max_diff_minutes 以内。对于业务
        本项目默认要求时间差不超过 2 分钟，以减小快速演变对流中的
        非同时观测误差；调用方可以显式调整该阈值。

        返回 (ts_station1, ts_station2, time_diff_minutes) 的列表。
        """
        ppi1 = data_dir / station1 / "PPI"
        ppi2 = data_dir / station2 / "PPI"
        if not ppi1.is_dir() or not ppi2.is_dir():
            return []

        # 扫描每部测站的时间戳
        def get_timestamps(ppi_dir, product_type, resolution):
            tss = set()
            for d in ppi_dir.iterdir():
                if not d.is_dir():
                    continue
                info = RadarDataPipeline.PRODUCT_MAP.get(d.name, (None, None))
                if info[0] != product_type or info[1] != resolution:
                    continue
                for f in d.iterdir():
                    if f.name == "ProductIndex":
                        continue
                    parts = f.name.split("_")
                    if len(parts) >= 2:
                        tss.add(parts[1])
            return sorted(tss)

        ts1 = get_timestamps(ppi1, product_type, resolution)
        ts2 = get_timestamps(ppi2, product_type, resolution)

        # 解析时间戳字符串
        dts1 = [(datetime.strptime(t, "%Y%m%d%H%M%SZ"), t) for t in ts1]
        dts2 = [(datetime.strptime(t, "%Y%m%d%H%M%SZ"), t) for t in ts2]

        # 在容差范围内匹配（贪心算法，每个站2的时间戳最多用一次）
        matches = []
        used_s2 = set()
        for dt1, t1 in dts1:
            best = min(dts2, key=lambda x: abs(x[0] - dt1))
            dt2, t2 = best
            diff = abs((dt1 - dt2).total_seconds()) / 60.0
            if diff <= max_diff_minutes and t2 not in used_s2:
                matches.append((t1, t2, diff))
                used_s2.add(t2)

        return sorted(matches, key=lambda x: x[0])

    # ------------------------------------------------------------------
    # 主流水线执行函数
    # ------------------------------------------------------------------

    def run(self, station1: str = "Z9317", station2: str = "Z9543",
            timestamp1: Optional[str] = None, timestamp2: Optional[str] = None,
            output_dir: str = "output",
            product_type: str = "velocity",
            resolution: str = "high",
            grid_shape: Tuple[int, int, int] = (20, 201, 201),
            grid_limits: Tuple = (
                (500.0, 15000.0),
                (-150000.0, 150000.0),
                (-150000.0, 150000.0),
            )):
        """
        运行完整流水线，处理一个时间匹配的雷达体扫数据对。

        参数:
            station1, station2: 雷达站号
            timestamp1, timestamp2: 指定时间戳（为 None 时自动匹配）
            output_dir: 网格化 NetCDF 输出目录
            product_type: 'velocity'（径向速度）或 'reflectivity'（反射率因子）
            resolution: 'high'（高分辨率）或 'low'（低分辨率）
            grid_shape: (nz, ny, nx) 网格维度
            grid_limits: ((z_min, z_max), (y_min, y_max), (x_min, x_max))
        """
        import pyart

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 步骤0: 时间匹配
        matches = None
        if timestamp1 is None or timestamp2 is None:
            print("=== Step 0: Time Matching ===")
            matches = self.match_times(
                station1, station2, self.data_dir,
                product_type=product_type, resolution=resolution)
            if not matches:
                print("No matching timestamps found!")
                return None

            # 选择时间差最小的最佳匹配对
            matches.sort(key=lambda x: x[2])
            timestamp1, timestamp2, diff = matches[0]
            print(f"  Selected: {station1}={timestamp1}, "
                  f"{station2}={timestamp2} (diff={diff:.1f} min)")

            if len(matches) > 1:
                print(f"  {len(matches)} total matches available")
                for t1, t2, d in matches[:5]:
                    print(f"    {t1} <-> {t2} (±{d:.1f} min)")

        # 步骤1+2: 构建雷达体扫数据
        print(f"\n=== Step 1+2: Building Radar Volumes ===")
        print(f"  {station1} at {timestamp1}:")
        sweeps1 = self.build_volume(
            station1, timestamp1, product_type, resolution)
        print(f"  -> {len(sweeps1)} sweeps across "
              f"{len(set(s['elevation'] for s in sweeps1))} elevations")

        print(f"  {station2} at {timestamp2}:")
        sweeps2 = self.build_volume(
            station2, timestamp2, product_type, resolution)
        print(f"  -> {len(sweeps2)} sweeps across "
              f"{len(set(s['elevation'] for s in sweeps2))} elevations")

        if len(sweeps1) == 0 or len(sweeps2) == 0:
            print("ERROR: No sweeps found for one or both stations!")
            return None

        # 转换为 PyART Radar 对象
        print(f"\n  Converting to PyART Radar objects...")
        radar1 = self.sweeps_to_pyart_radar(sweeps1)
        radar2 = self.sweeps_to_pyart_radar(sweeps2)
        print(f"  {station1}: {radar1.nsweeps} sweeps, {radar1.nrays} rays")
        print(f"  {station2}: {radar2.nsweeps} sweeps, {radar2.nrays} rays")

        radars = [radar1, radar2]

        # 步骤3: 质量控制
        print(f"\n=== Step 3: Quality Control ===")
        radars_qc = []
        for i, (radar, stn) in enumerate(zip(radars, [station1, station2])):
            print(f"  {stn}:")
            radar_qc = self.quality_control(radar)
            radars_qc.append(radar_qc)

        # 步骤4: 笛卡尔网格化
        print(f"\n=== Step 4: Cartesian Gridding ===")
        grid = self.grid_radar_data(
            radars_qc,
            grid_shape=grid_shape,
            grid_limits=grid_limits,
        )

        # 步骤5: 保存输出
        print(f"\n=== Step 5: Saving Output ===")
        field_name = list(grid.fields.keys())[0]
        output_file = output_path / (
            f"gridded_{product_type}_{station1}_{station2}_"
            f"{timestamp1}_{timestamp2}.nc"
        )
        pyart.io.write_grid(str(output_file), grid)
        print(f"  Saved gridded data to: {output_file}")
        print(f"  Field: {field_name}")
        print(f"  Shape: {grid.fields[field_name]['data'].shape}")
        print(f"  Gridded data ready for PyDDA 3DVAR!")

        return {
            "radars": radars_qc,
            "grid": grid,
            "timestamp1": timestamp1,
            "timestamp2": timestamp2,
            "output_file": output_file,
            "matches": matches if timestamp1 != timestamp2 else None,
        }
