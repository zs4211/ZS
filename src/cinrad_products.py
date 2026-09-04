"""
CINRAD 二级产品读取模块。

读取雷达自带算法生成的各类气象产品，用于:
  1. M/TVS 产品 → 验证双多普勒反演的中气旋检测精度
  2. VWP 产品  → 构建垂直风廓线背景场（替代/补充 ERA5）
  3. SRM 产品  → 风暴相对速度（已去除平移风，直接展示旋转）
  4. TOPS 产品 → 回波顶高，作为质量连续积分的上边界约束
  5. VIL 产品  → 垂直累积液态水含量，辅助识别强上升区

产品代码说明 (CINRAD WRS RSTM 格式):
  060 - M    (Mesocyclone)           中气旋
  061 - TVS  (Tornado Vortex Signature) 龙卷涡旋特征
  048 - VWP  (VAD Wind Profile)      VAD 风廓线
  056 - SRM  (Storm Relative Motion) 风暴相对速度
  041 - TOPS (Echo Tops)             回波顶高
  057 - VIL  (Vertically Integrated Liquid) 垂直累积液态水
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple


# ══════════════════════════════════════════════════════════════
#  M — 中气旋产品
# ══════════════════════════════════════════════════════════════

def load_mesocyclones(station: str, timestamp: str,
                      data_dir: str = "data",
                      tol_seconds: float = 600) -> List[Dict]:
    """
    加载雷达自带中气旋检测产品 (M, 代码 060)。

    返回列表，每个元素包含:
      - station, time: 测站和时间
      - lat, lon: 中气旋中心经纬度
      - range_km, azimuth_deg: 极坐标位置
      - height_m, top_m, base_m: 高度/顶高/底高
      - radius_m: 中气旋半径
      - avg_shear, max_shear: 平均/最大切变 (10⁻³ s⁻¹)
      - avg_rv, max_rv: 平均/最大旋转速度 (m/s)
    """
    from cinrad.io import read_auto

    m_dir = Path(data_dir) / station / "M" / "060"
    if not m_dir.is_dir():
        return []

    ts_clean = timestamp.rstrip("Z")
    m_file = m_dir / f"{station}_{ts_clean}Z_M_00_060"
    if not m_file.exists():
        # 时间容差搜索
        ts_dt = datetime.strptime(ts_clean, "%Y%m%d%H%M%S")
        best, best_diff = None, 999
        for f in m_dir.glob(f"{station}_*_M_00_060"):
            parts = f.name.split("_")
            if len(parts) >= 2:
                try:
                    ft = datetime.strptime(parts[1].rstrip("Z"), "%Y%m%d%H%M%S")
                    diff = abs((ft - ts_dt).total_seconds())
                    if diff < best_diff:
                        best_diff = diff; best = f
                except ValueError:
                    continue
        if best is None or best_diff > tol_seconds:
            return []
        m_file = best

    source_timestamp = m_file.name.split("_")[1]
    requested_dt = datetime.strptime(ts_clean, "%Y%m%d%H%M%S")
    source_dt = datetime.strptime(
        source_timestamp.rstrip("Z"), "%Y%m%d%H%M%S"
    )
    time_offset_seconds = abs((source_dt - requested_dt).total_seconds())

    pup = read_auto(str(m_file))
    ds = pup.get_data()
    n = len(ds["feature_id"].values)

    results = []
    for i in range(n):
        results.append({
            "station": station,
            "time": source_timestamp,
            "time_offset_seconds": time_offset_seconds,
            "lat": float(ds["latitude"].values[i]),
            "lon": float(ds["longitude"].values[i]),
            "azimuth_deg": float(ds["meso_azimuth"].values[i]),
            "range_m": float(ds["meso_range"].values[i]),
            "height_m": float(ds["meso_height"].values[i]),
            "top_m": float(ds["meso_top"].values[i]),
            "base_m": float(ds["meso_base"].values[i]),
            "radius_m": float(ds["meso_radius"].values[i]),
            "avg_shear": float(ds["meso_avgshr"].values[i]),
            "max_shear": float(ds["meso_mxtanshr"].values[i]),
            "avg_rv": float(ds["meso_avgrv"].values[i]),
            "max_rv": float(ds["meso_mxrv"].values[i]),
            "diameter_m": float(ds["meso_azdia"].values[i]),
        })
    return results


# ══════════════════════════════════════════════════════════════
#  TVS — 龙卷涡旋特征
# ══════════════════════════════════════════════════════════════

def load_tvs(station: str, timestamp: str,
             data_dir: str = "data") -> List[Dict]:
    """加载龙卷涡旋特征产品 (TVS, 代码 061)。"""
    from cinrad.io import read_auto

    tvs_dir = Path(data_dir) / station / "TVS" / "061"
    if not tvs_dir.is_dir():
        return []

    ts_clean = timestamp.rstrip("Z")
    for f in sorted(tvs_dir.glob(f"{station}_*_TVS_00_061")):
        parts = f.name.split("_")
        if len(parts) >= 2 and parts[1].rstrip("Z") == ts_clean:
            pup = read_auto(str(f))
            ds = pup.get_data()
            n = len(ds["tvs_id"].values)
            return [{
                "station": station, "time": timestamp,
                "lat": float(ds["latitude"].values[i]),
                "lon": float(ds["longitude"].values[i]),
                "azimuth_deg": float(ds["tvs_azimuth"].values[i]),
                "range_m": float(ds["tvs_range"].values[i]),
                "lldv_mps": float(ds["tvs_lldv"].values[i]),
                "avg_dv_mps": float(ds["tvs_avgdv"].values[i]),
                "max_dv_mps": float(ds["tvs_mxdv"].values[i]),
                "depth_m": float(ds["tvs_depth"].values[i]),
                "base_m": float(ds["tvs_base"].values[i]),
                "top_m": float(ds["tvs_top"].values[i]),
                "max_shear": float(ds["tvs_mxshr"].values[i]),
            } for i in range(n)]
    return []


# ══════════════════════════════════════════════════════════════
#  VWP — VAD 风廓线
# ══════════════════════════════════════════════════════════════

def load_vwp(station: str, timestamp: str = None,
             data_dir: str = "data",
             tol_seconds: float = 600) -> Optional[Dict]:
    """
    加载 VAD 风廓线产品 (VWP, 代码 048)。

    返回字典:
      - height_m: (30,) 高度数组 (m)
      - wind_dir: (30,) 风向 (度)
      - wind_speed: (30,) 风速 (m/s)
      - rms: (30,) 拟合均方根误差 (m/s)
    或 None（无数据时）。

    VAD (Velocity Azimuth Display) 技术通过拟合单雷达全方位径向速度
    的正弦曲线来反演水平风廓线，不依赖双多普勒几何。
    """
    from cinrad.io import read_auto

    vwp_dir = Path(data_dir) / station / "VWP" / "048"
    if not vwp_dir.is_dir():
        return None

    # 时间匹配：精确或容差搜索
    best_file = None
    best_diff = float("nan")
    if timestamp:
        ts_dt = datetime.strptime(timestamp.rstrip("Z"), "%Y%m%d%H%M%S")
        best_diff = tol_seconds + 1
        for f in sorted(vwp_dir.glob(f"{station}_*_VWP_00_048")):
            parts = f.name.split("_")
            if len(parts) >= 2:
                try:
                    ft = datetime.strptime(parts[1].rstrip("Z"), "%Y%m%d%H%M%S")
                    diff = abs((ft - ts_dt).total_seconds())
                    if diff < best_diff:
                        best_diff = diff; best_file = f
                except ValueError:
                    continue
    else:
        # 无时间戳：取第一个文件
        files = sorted(vwp_dir.glob(f"{station}_*_VWP_00_048"))
        best_file = files[0] if files else None

    if best_file is None:
        return None

    source_timestamp = best_file.name.split("_")[1]

    pup = read_auto(str(best_file))
    ds = pup.get_data()
    wdir = ds["wind_direction"].values
    wspd = ds["wind_speed"].values
    rms = ds["rms"].values

    if wdir.ndim > 1 and wdir.shape[0] > 1:
        best_mode = np.argmin(np.nanmean(rms, axis=1))
        wdir = wdir[best_mode]; wspd = wspd[best_mode]; rms = rms[best_mode]
    else:
        wdir = wdir.ravel(); wspd = wspd.ravel(); rms = rms.ravel()

    heights = np.arange(0.3, 9.1, 0.3)[:len(wdir)] * 1000

    valid = (wspd > 0) & (rms < 10)
    u = np.zeros_like(wspd); v = np.zeros_like(wspd)
    u[valid] = -wspd[valid] * np.sin(np.radians(wdir[valid]))
    v[valid] = -wspd[valid] * np.cos(np.radians(wdir[valid]))

    return {
        "station": station, "time": source_timestamp,
        "time_offset_seconds": best_diff,
        "source_file": str(best_file),
        "height_m": heights,
        "wind_dir": wdir, "wind_speed": wspd, "rms": rms,
        "u": u, "v": v, "valid": valid,
    }


def vwp_to_background(vwp_data: Dict, grid_z_m: np.ndarray,
                      nz: int, ny: int, nx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 VWP 风廓线插值到三维网格的所有格点。

    对每个高度层，VWP 的 u, v 值复制到整个水平面。
    高度方向线性插值。
    """
    if vwp_data is None:
        u_bg = np.zeros((nz, ny, nx), dtype=np.float32)
        v_bg = np.zeros((nz, ny, nx), dtype=np.float32)
        return u_bg, v_bg

    h_vwp = vwp_data["height_m"]
    u_vwp = vwp_data["u"]
    v_vwp = vwp_data["v"]
    valid = vwp_data["valid"]

    # 只用有效层插值
    if valid.sum() >= 2:
        h_valid = h_vwp[valid]
        u_valid = u_vwp[valid]
        v_valid = v_vwp[valid]
    else:
        u_bg = np.zeros((nz, ny, nx), dtype=np.float32)
        v_bg = np.zeros((nz, ny, nx), dtype=np.float32)
        return u_bg, v_bg

    u_profile = np.interp(grid_z_m, h_valid, u_valid,
                          left=u_valid[0], right=u_valid[-1])
    v_profile = np.interp(grid_z_m, h_valid, v_valid,
                          left=v_valid[0], right=v_valid[-1])

    u_bg = np.tile(u_profile[:, np.newaxis, np.newaxis], (1, ny, nx)).astype(np.float32)
    v_bg = np.tile(v_profile[:, np.newaxis, np.newaxis], (1, ny, nx)).astype(np.float32)

    return u_bg, v_bg


# ══════════════════════════════════════════════════════════════
#  SRM — 风暴相对速度
# ══════════════════════════════════════════════════════════════

def load_srm(station: str, timestamp: str,
             data_dir: str = "data") -> Optional[Dict]:
    """
    加载风暴相对速度产品 (SRM, 代码 056)。

    SRM 已从径向速度中减去了风暴平移速度，因此能直接显示
    中气旋的旋转偶极子（正负速度对）。

    返回字典:
      - data: (360, 920) SRM 速度场 (m/s)
      - azimuth: (360,) 方位角 (度)
      - range_km: (920,) 距离 (km)
      - lon, lat: (360, 920) 经纬度网格
    """
    from cinrad.io import read_auto

    srm_dir = Path(data_dir) / station / "SRM" / "056"
    if not srm_dir.is_dir():
        return None

    ts_clean = timestamp.rstrip("Z")
    for f in sorted(srm_dir.glob(f"{station}_*_SRM_02_056")):
        parts = f.name.split("_")
        if len(parts) >= 2 and parts[1].rstrip("Z") == ts_clean:
            pup = read_auto(str(f))
            ds = pup.get_data()
            return {
                "station": station, "time": timestamp,
                "data": ds["SRM"].values.astype(np.float32),
                "azimuth": ds["azimuth"].values,
                "range_km": ds["distance"].values,
                "lon": ds["longitude"].values if "longitude" in ds else None,
                "lat": ds["latitude"].values if "latitude" in ds else None,
            }
    return None


# ══════════════════════════════════════════════════════════════
#  TOPS — 回波顶高
# ══════════════════════════════════════════════════════════════

def load_tops(station: str, timestamp: str,
              data_dir: str = "data") -> Optional[np.ndarray]:
    """加载回波顶高 (TOPS, 041)，返回 (1840, 1840) km 数组。"""
    from cinrad.io import read_auto

    tops_dir = Path(data_dir) / station / "TOPS" / "041"
    if not tops_dir.is_dir():
        return None

    ts_clean = timestamp.rstrip("Z")
    for f in sorted(tops_dir.glob(f"{station}_*_TOPS_00_041")):
        parts = f.name.split("_")
        if len(parts) >= 2 and parts[1].rstrip("Z") == ts_clean:
            pup = read_auto(str(f))
            ds = pup.get_data()
            return ds["ET"].values.astype(np.float32)
    return None


# ══════════════════════════════════════════════════════════════
#  VIL — 垂直累积液态水
# ══════════════════════════════════════════════════════════════

def load_vil(station: str, timestamp: str,
             data_dir: str = "data") -> Optional[np.ndarray]:
    """加载 VIL (057)，返回 (1840, 1840) kg/m² 数组。"""
    from cinrad.io import read_auto

    vil_dir = Path(data_dir) / station / "VIL" / "057"
    if not vil_dir.is_dir():
        return None

    ts_clean = timestamp.rstrip("Z")
    for f in sorted(vil_dir.glob(f"{station}_*_VIL_00_057")):
        parts = f.name.split("_")
        if len(parts) >= 2 and parts[1].rstrip("Z") == ts_clean:
            pup = read_auto(str(f))
            ds = pup.get_data()
            return ds["VIL"].values.astype(np.float32)
    return None
