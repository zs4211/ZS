"""
雷达站点集中式注册表。

所有站点的元数据（站号、名称、经纬度、海拔）在此统一定义。
其余模块通过 get_station() / get_radar_site() 查询，不再硬编码。

数据结构:
  RadarSite — 供 gridder.py 坐标变换和 visualize_3d.py 可视化使用的站点元数据
  StationInfo — 不可变的站点完整信息（含中文名称）

添加新站点（二选一）:
  1. 在 STATIONS 字典中新增一条记录（推荐，可指定中文名）
  2. 不修改代码 — 若站号未注册，自动从 data/<站号>/PPI/ 中的产品文件提取坐标
"""

from dataclasses import dataclass
from pathlib import Path


# ══════════════════════════════════════════════════════════════
#  站点元数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class RadarSite:
    """雷达站点元数据（供 gridder 和可视化模块使用）。

    Attributes
    ----------
    code : str
        站点代号（如 Z9317、Z9543）。
    name : str
        站点中文名称。
    lat, lon : float
        站点经纬度（WGS84，度）。
    height : float
        站点海拔高度（米）。
    """
    code: str
    name: str
    lat: float
    lon: float
    height: float


@dataclass(frozen=True)
class StationInfo:
    """站点完整信息（不可变）。

    Attributes
    ----------
    code : str
        站号，如 "Z9317"。
    name : str
        中文名称，如 "沧州"。
    lat : float
        纬度 (WGS84)。
    lon : float
        经度 (WGS84)。
    height : float
        海拔高度 (米)。
    """
    code: str
    name: str
    lat: float
    lon: float
    height: float


# ══════════════════════════════════════════════════════════════
#  已注册站点（CINRAD S 波段雷达网络）
# ══════════════════════════════════════════════════════════════

STATIONS = {
    "Z9317": StationInfo(code="Z9317", name="沧州", lat=38.279, lon=116.805, height=9.0),
    "Z9543": StationInfo(code="Z9543", name="滨州", lat=37.383, lon=117.933, height=18.0),
    "Z9532": StationInfo(code="Z9532", name="青岛", lat=35.989, lon=120.230, height=92.0),
    "Z9539": StationInfo(code="Z9539", name="临沂", lat=35.250, lon=118.421, height=224.0),
}


# ══════════════════════════════════════════════════════════════
#  站点查询接口
# ══════════════════════════════════════════════════════════════

def _auto_detect(code: str) -> StationInfo:
    """从未注册站点的数据文件中自动提取坐标和高度。

    通过 cinrad 读取该站点 PPI 目录下的第一个产品文件，
    从元数据中获取经纬度和雷达天线高度。
    """
    data_dir = Path("data") / code / "PPI"
    if not data_dir.is_dir():
        raise KeyError(
            f"未知站点: {code}，且 data/{code}/PPI/ 目录不存在。"
            f"已知站点: {list(STATIONS.keys())}")

    from cinrad.io import read_auto

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        files = [f for f in subdir.iterdir() if f.name != "ProductIndex"]
        if not files:
            continue
        try:
            pup = read_auto(str(files[0]))
            info = StationInfo(
                code=code,
                name=code,
                lat=float(pup.stationlat),
                lon=float(pup.stationlon),
                height=float(pup.radarheight),
            )
            STATIONS[code] = info  # 缓存以供后续查询
            return info
        except Exception:
            continue

    raise KeyError(f"无法从 data/{code}/PPI/ 读取站点元数据")


def get_station(code: str) -> StationInfo:
    """按站号查询站点信息（未注册则自动探测）。"""
    if code in STATIONS:
        return STATIONS[code]
    return _auto_detect(code)


def get_radar_site(code: str) -> RadarSite:
    """返回 RadarSite 对象（供可视化等模块使用）。"""
    s = get_station(code)
    return RadarSite(code=s.code, name=s.name, lat=s.lat, lon=s.lon, height=s.height)


def list_stations() -> list:
    """列出所有已注册站点的站号列表。"""
    return list(STATIONS.keys())
