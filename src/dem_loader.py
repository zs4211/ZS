"""
SRTM数字高程模型(DEM)加载模块。

下载SRTM3（3弧秒分辨率，约90米）高程瓦片，解析.hgt二进制格式，
并在与雷达分析网格匹配的ENU坐标系下构建PyVista结构化网格表面。

瓦片规格：
    每个瓦片覆盖 1°×1° 经纬度范围，SRTM3的3弧秒采样间距产生1201×1201个采样点。
    数据来源：SRTM GL3全球高程数据。
"""
import io
import zipfile
import numpy as np
from pathlib import Path

import requests
import pyvista as pv

from src.gridder import wgs84_to_enu


# 每个瓦片为1度×1度，SRTM3 = 3弧秒间距 → 1201×1201个采样点
SRTM_TILE_SIZE = 1201
SRTM_URL = "https://srtm.kurviger.de/SRTM3/Eurasia/N{lat:02d}E{lon:03d}.hgt.zip"
NO_DATA_VALUE = -32768


def _tile_key(lat, lon):
    """返回包含坐标(lat, lon)的1×1度瓦片标识符。"""
    return f"N{int(np.floor(lat)):02d}E{int(np.floor(lon)):03d}"


def _tiles_for_domain(lat_min, lat_max, lon_min, lon_max):
    """列出覆盖指定经纬度边界框所需的全部SRTM瓦片标识符。"""
    tiles = set()
    for ilat in range(int(np.floor(lat_min)), int(np.ceil(lat_max))):
        for ilon in range(int(np.floor(lon_min)), int(np.ceil(lon_max))):
            tiles.add(_tile_key(ilat, ilon))
    return sorted(tiles)


def _download_tile(tile_key, cache_dir):
    """
    下载单个SRTM .hgt.zip瓦片，解压提取.hgt文件，并缓存到本地。

    如果瓦片已在缓存目录中存在，则直接返回路径，避免重复下载。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / f"{tile_key}.hgt"
    if cache_path.exists():
        return str(cache_path)

    lat = int(tile_key[1:3])
    lon = int(tile_key[4:8])
    url = SRTM_URL.format(lat=lat, lon=lon)

    print(f"  [DEM] Downloading {tile_key} from {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = zf.namelist()
        if not names:
            raise RuntimeError(f"Empty zip for {tile_key}")
        raw = zf.read(names[0])

    # 根据文件大小自动识别SRTM3(1201)或SRTM1(3601)分辨率
    expected = SRTM_TILE_SIZE * SRTM_TILE_SIZE * 2
    if abs(len(raw) - expected) > 4:
        # 瓦片可能是3601×3601（SRTM1）；两种分辨率都接受
        alt = 3601 * 3601 * 2
        if abs(len(raw) - alt) <= 4:
            actual_n = 3601
        else:
            raise RuntimeError(
                f"Unexpected tile size for {tile_key}: {len(raw)} bytes "
                f"(expected {expected} or {alt})")
    else:
        actual_n = SRTM_TILE_SIZE

    with open(cache_path, "wb") as f:
        f.write(raw)
    print(f"  [DEM] Cached {tile_key} ({actual_n}x{actual_n}, {len(raw)} bytes)")
    return str(cache_path)


def _load_tile_cache(grid, cache_dir):
    """
    下载覆盖分析网格域的所有SRTM瓦片，并返回 {瓦片标识: 高程数据} 字典。

    对于下载失败的瓦片（如海面区域），会打印警告并以海平面(0米)替代。
    """
    lat_2d = grid.lat_grid[:, :, 0]
    lon_2d = grid.lon_grid[:, :, 0]
    lat_min, lat_max = float(np.nanmin(lat_2d)), float(np.nanmax(lat_2d))
    lon_min, lon_max = float(np.nanmin(lon_2d)), float(np.nanmax(lon_2d))

    tiles = _tiles_for_domain(lat_min, lat_max, lon_min, lon_max)
    print(f"  [DEM] Grid domain: lat [{lat_min:.2f}, {lat_max:.2f}], "
          f"lon [{lon_min:.2f}, {lon_max:.2f}]")
    print(f"  [DEM] Tiles needed: {len(tiles)} — {', '.join(tiles[:6])}"
          f"{'...' if len(tiles) > 6 else ''}")

    cache = {}
    for key in tiles:
        try:
            path = _download_tile(key, cache_dir)
            cache[key] = _parse_hgt(path)
        except Exception as e:
            print(f"  [DEM] Tile {key} unavailable ({e}), treating as sea level")
    if not cache:
        raise RuntimeError(f"DEM数据加载失败: no DEM tiles available for domain")
    return cache


def _parse_hgt(filepath):
    """
    解析原始SRTM .hgt文件为二维numpy数组。

    数据存储顺序：行为从北到南（第0行对应瓦片北边缘），列为从西到东。
    使用大端格式(>i2)读取16位有符号整数，无效值标记为-32768。
    """
    raw = Path(filepath).read_bytes()
    # 根据文件大小反推网格尺寸
    n = int(np.sqrt(len(raw) // 2))
    if n * n * 2 != len(raw):
        raise RuntimeError(f"Unexpected .hgt size for {filepath}: {len(raw)} bytes")
    data = np.frombuffer(raw, dtype=">i2").reshape(n, n).astype(np.float32)
    data[data <= NO_DATA_VALUE] = np.nan
    return data


def _query_elevation(lat, lon, tile_cache):
    """
    在WGS84坐标(lat, lon)处通过双线性插值查询高程（米）。

    双线性插值说明：
        先确定坐标落在瓦片的哪个网格单元内，计算行、列方向的小数偏移量
        (wr, wc)，然后在四个角点值之间做加权平均：
        elevation = (1-wr)[(1-wc)*v00 + wc*v01] + wr[(1-wc)*v10 + wc*v11]
        这能有效消除阶梯状伪影，得到平滑的地形过渡。
    """
    tile_lat = int(np.floor(lat))
    tile_lon = int(np.floor(lon))
    key = _tile_key(tile_lat, tile_lon)
    if key not in tile_cache:
        return np.nan

    data = tile_cache[key]
    n = data.shape[0]
    # 第0行 = 瓦片北边缘 (tile_lat + 1)
    # 第0列 = 瓦片西边缘 (tile_lon)
    row_f = (tile_lat + 1.0 - lat) * (n - 1)
    col_f = (lon - tile_lon) * (n - 1)

    r0 = int(np.floor(row_f))
    c0 = int(np.floor(col_f))
    r1 = min(r0 + 1, n - 1)
    c1 = min(c0 + 1, n - 1)
    r0 = max(r0, 0)
    c0 = max(c0, 0)

    wr = row_f - r0
    wc = col_f - c0

    v00 = data[r0, c0]
    v10 = data[r1, c0]
    v01 = data[r0, c1]
    v11 = data[r1, c1]

    # 双线性插值：先在列方向插值，再在行方向插值
    top = v00 * (1 - wc) + v01 * wc
    bot = v10 * (1 - wc) + v11 * wc
    return float(top * (1 - wr) + bot * wr)


def build_dem_surface(grid, cache_dir="data/dem_cache"):
    """
    在ENU坐标系下构建DEM地形的PyVista结构化网格表面。

    参数：
        grid: 风场AnalysisGrid对象
        cache_dir: SRTM瓦片缓存的本地目录

    返回：
        (surface_mesh: pv.StructuredGrid, elevation_2d_km: np.ndarray)
        surface_mesh的z坐标为SRTM高程（公里）。
        elevation_2d_km的形状为(nx, ny)，供叠加其他图层使用。

    处理说明：
        对于水域或瓦片边缘间隙处的网格单元，高程设为0米（海平面）。
    """
    tile_cache = _load_tile_cache(grid, cache_dir)

    lat_2d = grid.lat_grid[:, :, 0]
    lon_2d = grid.lon_grid[:, :, 0]
    nx, ny = lat_2d.shape

    ref_lat = float(grid.origin_lat)
    ref_lon = float(grid.origin_lon)

    elevation_m = np.zeros((nx, ny), dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            val = _query_elevation(lat_2d[i, j], lon_2d[i, j], tile_cache)
            if not np.isfinite(val):
                # 水域或无效网格单元：设为海平面（0米）
                val = 0.0
            elevation_m[i, j] = val

    elevation_km = elevation_m / 1000.0
    print(f"  [DEM] Elevation range: [{elevation_km.min():.3f}, "
          f"{elevation_km.max():.3f}] km")

    # 构建地形表面的ENU坐标
    x_enu = grid.X[:, :, 0].copy()
    y_enu = grid.Y[:, :, 0].copy()
    z_enu = elevation_km

    surface = pv.StructuredGrid(x_enu, y_enu, z_enu)
    return surface, elevation_km, tile_cache
