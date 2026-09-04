"""
CR组合反射率加载与地面叠加模块。

通过cinrad库读取CINRAD雷达CR产品（产品号037），将1840×1840的
笛卡尔反射率网格插值到分析网格的水平面上，支持多站最大值合成。

CR组合反射率是各仰角反射率因子的垂直最大值投影，用于识别
对流核心的水平和垂直结构，是判断中气旋强度的重要参考。
"""
import sys
import numpy as np
from pathlib import Path
from collections import namedtuple

sys.path.insert(0, str(Path(__file__).parent.parent))

CartesianData = namedtuple("CartesianData", ["lon", "lat", "data"])


def load_cr_reflectivity(grid, station_codes, timestamps, data_dir="data",
                         tol_seconds=180):
    """
    加载多个雷达站的CR组合反射率产品，按最大值法则合成。

    参数：
        grid: AnalysisGrid 分析网格对象
        station_codes: 雷达站编号列表，例如 ["Z9317", "Z9543"]
        timestamps: 与站点一一对应的时间戳字符串
        data_dir: 原始数据根目录
        tol_seconds: 时间容差，当前未使用（CR文件使用精确文件名匹配）

    返回：
        形状为 (nx, ny) 的二维dBZ反射率数组

    异常：
        RuntimeError: 如果没有成功加载任何CR数据

    合成策略：
        对于同一地理位置的多个站点观测值，取最大值（即最强回波）。
        这一策略基于以下考虑：强对流核心的反射率是业务预报中最关键的指标，
        使用最大值可以确保不遗漏任何雷达探测到的强回波区。
    """
    from datetime import datetime
    from cinrad.io import read_auto
    from src.gridder import interp_cartesian_to_grid

    composite = np.full((grid.nx, grid.ny), np.nan, dtype=np.float32)
    any_data = False

    for stn, ts_str in zip(station_codes, timestamps):
        cr_dir = Path(data_dir) / stn / "CR" / "037"
        if not cr_dir.is_dir():
            print(f"  [CR] {stn}: 目录不存在 {cr_dir}")
            continue

        # 按时间容差搜索最接近的CR文件
        ts_dt = datetime.strptime(ts_str.rstrip("Z"), "%Y%m%d%H%M%S")
        best_file = None
        best_diff = tol_seconds + 1
        for f in sorted(cr_dir.glob(f"{stn}_*_CR_00_037")):
            parts = f.name.split("_")
            if len(parts) >= 2:
                try:
                    ft = datetime.strptime(parts[1].rstrip("Z"), "%Y%m%d%H%M%S")
                    diff = abs((ft - ts_dt).total_seconds())
                    if diff < best_diff:
                        best_diff = diff
                        best_file = f
                except ValueError:
                    continue

        if best_file is None:
            print(f"  [CR] {stn}: 未找到 {tol_seconds}s 内的CR文件")
            continue

        print(f"  [CR] {stn}: {best_file.name} (时间差={best_diff:.0f}s)")

        try:
            pup = read_auto(str(best_file))
            ds = pup.get_data()
        except Exception as e:
            print(f"  [CR] {stn}: 读取失败 - {e}")
            continue

        cr_data = np.array(ds["CR"].values, dtype=np.float32)
        cr_lon = np.array(ds["longitude"].values, dtype=np.float64)
        cr_lat = np.array(ds["latitude"].values, dtype=np.float64)

        # 屏蔽无效值：NaN以及噪声（<-20 dBZ 通常是地物杂波）
        mask = np.isnan(cr_data) | (cr_data < -20)
        cr_data[mask] = np.nan

        # 构建CartesianData对象并插值到分析网格
        cart = CartesianData(lon=cr_lon, lat=cr_lat, data=cr_data)
        try:
            refl_2d = interp_cartesian_to_grid(cart, grid)
        except Exception as e:
            print(f"  [CR] Interpolation failed for {stn}: {e}")
            continue

        valid_mask = np.isfinite(refl_2d)
        n_valid = valid_mask.sum()
        if n_valid > 0:
            any_data = True
            print(f"  [CR] {stn}: {n_valid} valid cells, "
                  f"range [{np.nanmin(refl_2d):.0f}, {np.nanmax(refl_2d):.0f}] dBZ")

            # 最大值合成：在复合数组中保留每个网格点的最大dBZ值
            composite = np.where(
                valid_mask & (refl_2d > composite) | (valid_mask & np.isnan(composite)),
                refl_2d, composite)

    if not any_data:
        raise RuntimeError(
            f"反射率CR数据不可用: no CR data for stations {station_codes} "
            f"at timestamps {timestamps}")

    n_composite = np.isfinite(composite).sum()
    print(f"  [CR] Final composite: {n_composite} valid cells, "
          f"range [{np.nanmin(composite):.0f}, {np.nanmax(composite):.0f}] dBZ")

    return composite
