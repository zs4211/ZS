"""
地理边界加载模块：GADM县级行政区划边界 + Natural Earth河流水系。

下载GADM 4.1版本的中国县级(level-2)行政区划shapefile，
并通过cartopy加载Natural Earth 10m精度河流数据。
所有几何要素被裁剪到雷达分析网格范围，并重投影至ENU坐标系，
z坐标依附于DEM高程以确保地形贴合。
"""
import os
import zipfile
import io
import numpy as np
from pathlib import Path

from src.proj_setup import configure_proj

configure_proj()

import requests
import shapefile
import pyvista as pv
from shapely.geometry import Polygon, LineString, box
from shapely.ops import unary_union

import cartopy.feature as cfeature
from cartopy.io import shapereader

from src.gridder import wgs84_to_enu

GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_CHN_shp.zip"


def _download_and_extract(cache_dir):
    """
    下载GADM中国行政区划zip文件并解压shapefile组件到缓存目录。

    如果目标文件已存在则跳过下载，利用本地缓存加速。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / "gadm41_CHN_2.shp"
    if marker.exists():
        return str(cache_dir / "gadm41_CHN_2")

    print("  [GIS] Downloading GADM 4.1 China county boundaries ...")
    resp = requests.get(GADM_URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if name.startswith("gadm41_CHN_2."):
                zf.extract(name, cache_dir)
    print(f"  [GIS] Extracted GADM to {cache_dir}")
    return str(cache_dir / "gadm41_CHN_2")


def _domain_bbox(grid):
    """返回分析网格域的经纬度边界框 (lon_min, lon_max, lat_min, lat_max)。"""
    lat_2d = grid.lat_grid[:, :, 0]
    lon_2d = grid.lon_grid[:, :, 0]
    return (float(np.nanmin(lon_2d)), float(np.nanmax(lon_2d)),
            float(np.nanmin(lat_2d)), float(np.nanmax(lat_2d)))


def _polyline_to_enu_segments(coords_ll, ref_lat, ref_lon, bbox, dem_querier):
    """
    将一条折线（由(lon, lat)元组列表表示）裁剪到域边界框内，
    并为每个位于边界内的连续线段返回ENU坐标系下的(N, 3)点数组。

    算法流程：
        1. 遍历折线顶点，识别进入/离开边界框的分段点
        2. 对于完全在域内的连续子段，使用wgs84_to_enu转换为ENU坐标
        3. 通过dem_querier回调查询每个顶点的高程，使线条贴合地形

    参数：
        coords_ll: [(lon, lat), ...] 经纬度坐标列表
        ref_lat, ref_lon: ENU参考原点
        bbox: (lon_min, lon_max, lat_min, lat_max) 裁剪边界
        dem_querier: 可调用对象 (lat, lon) -> elevation_m

    返回：
        ENU坐标系统下的线段列表，每个线段为(N, 3)的numpy数组
    """
    if len(coords_ll) < 2:
        return []

    lon_min, lon_max, lat_min, lat_max = bbox

    # 快速剔除：检查整条折线的边界框是否与域相交
    lons = [pt[0] for pt in coords_ll]
    lats = [pt[1] for pt in coords_ll]
    if max(lons) < lon_min or min(lons) > lon_max or max(lats) < lat_min or min(lats) > lat_max:
        return []

    segments = []
    current_seg = []
    for pt in coords_ll:
        lon, lat = pt[0], pt[1]
        in_bounds = lon_min <= lon <= lon_max and lat_min <= lat <= lat_max

        if in_bounds:
            if current_seg is None:
                current_seg = []
            current_seg.append(pt)
        else:
            if current_seg and len(current_seg) >= 2:
                segments.append(current_seg)
            current_seg = None

    if current_seg and len(current_seg) >= 2:
        segments.append(current_seg)

    # 将裁剪后的线段转换为ENU坐标系，并附着DEM高程
    enu_segments = []
    for seg in segments:
        seg_lons = np.array([pt[0] for pt in seg])
        seg_lats = np.array([pt[1] for pt in seg])
        seg_alts = np.zeros_like(seg_lats)

        east, north, up = wgs84_to_enu(seg_lats, seg_lons, seg_alts, ref_lat, ref_lon)
        east = np.asarray(east).ravel()
        north = np.asarray(north).ravel()

        # 查询每个顶点处的DEM高程，使边界线贴合真实地形
        z_km = np.zeros(len(seg), dtype=np.float32)
        for k in range(len(seg)):
            elev_m = dem_querier(seg_lats[k], seg_lons[k])
            z_km[k] = (elev_m if np.isfinite(elev_m) else 0.0) / 1000.0

        pts = np.column_stack([east, north, z_km])
        enu_segments.append(pts)

    return enu_segments


def load_county_boundaries(grid, cache_dir, dem_querier):
    """
    加载GADM二级(县级)行政区划边界，裁剪至分析网格范围，转换为ENU坐标。

    返回：
        线段列表，每个元素为形状(N, 3)的numpy数组，单位为公里(ENU)。

    数据处理：
        GADM shapefile中的多边形由多条折线(polyline)组成，通过parts索引分割。
        每条折线独立处理，确保跨越网格边界的部分被正确裁剪。
    """
    shp_base = _download_and_extract(cache_dir)
    sf = shapefile.Reader(shp_base)
    bbox = _domain_bbox(grid)
    ref_lat = float(grid.origin_lat)
    ref_lon = float(grid.origin_lon)

    print(f"  [GIS] GADM features: {len(sf.shapes())}, "
          f"bbox: [{bbox[0]:.3f}, {bbox[1]:.3f}] x [{bbox[2]:.3f}, {bbox[3]:.3f}]")

    all_lines = []
    for shape_idx, shape_record in enumerate(sf.iterShapeRecords()):
        geom = shape_record.shape
        parts = list(geom.parts) + [len(geom.points)]

        for p in range(len(parts) - 1):
            seg_pts = geom.points[parts[p]:parts[p + 1]]
            enu_segs = _polyline_to_enu_segments(seg_pts, ref_lat, ref_lon, bbox, dem_querier)
            all_lines.extend(enu_segs)

        if (shape_idx + 1) % 500 == 0:
            print(f"    ... processed {shape_idx + 1}/{len(sf.shapes())} features")

    print(f"  [GIS] County boundary segments in domain: {len(all_lines)}")
    return all_lines


def load_rivers(grid, dem_querier):
    """
    通过cartopy加载Natural Earth 10m精度河流数据，裁剪至分析域，转换为ENU坐标。

    返回：
        线段列表，每个元素为形状(N, 3)的numpy数组，单位为公里(ENU)。

    数据来源：
        Natural Earth 10m cultural vectors，包含全球主要河流系统。
    """
    bbox = _domain_bbox(grid)
    ref_lat = float(grid.origin_lat)
    ref_lon = float(grid.origin_lon)

    print("  [GIS] Loading Natural Earth 10m rivers via cartopy ...")
    rivers = cfeature.RIVERS.with_scale("10m")
    geoms = list(rivers.geometries())

    all_lines = []
    for geom in geoms:
        coords = _geom_to_coords(geom)
        enu_segs = _polyline_to_enu_segments(coords, ref_lat, ref_lon, bbox, dem_querier)
        all_lines.extend(enu_segs)

    print(f"  [GIS] River segments in domain: {len(all_lines)}")
    return all_lines


def _geom_to_coords(geom):
    """
    从shapely几何对象(LineString或MultiLineString)中提取坐标列表。

    对于MultiLineString(多条线组成的几何体)，返回其中最长的线串，
    以确保可视化中河流的连续性。
    """
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return list(geom.coords)
    # 处理MultiLineString或GeometryCollection
    coords = []
    for g in geom.geoms if hasattr(geom, 'geoms') else [geom]:
        if isinstance(g, LineString) and not g.is_empty:
            coords.append(list(g.coords))
    if not coords:
        return []
    # 如果含多条线串，返回最长的一条以保持视觉连续性
    longest = max(coords, key=len)
    return longest


def load_city_labels(grid, cache_dir="data/shapefile_cache", dem_querier=None):
    """
    从GADM二级数据中提取位于分析域内的市/县名称及其中心点。

    参数：
        grid: AnalysisGrid 分析网格
        cache_dir: GADM shapefile缓存目录
        dem_querier: 可调用对象 (lat, lon) -> elevation_m，默认返回0

    返回：
        (positions, names):
        positions为 [(cx_km, cy_km), ...] ENU坐标的中心点列表
        names为城市/县名称字符串列表

    命名优先级：
        优先使用中文名称 (NL_NAME_2字段)，其次使用拼音名称 (NAME_2字段)。
        对于中文名称，以"|"分隔时取最后一部分（通常是最具体的行政区划名）。
    """
    if dem_querier is None:
        dem_querier = lambda lat, lon: 0.0

    shp_base = _download_and_extract(cache_dir)
    sf = shapefile.Reader(shp_base)
    bbox = _domain_bbox(grid)
    ref_lat = float(grid.origin_lat)
    ref_lon = float(grid.origin_lon)
    lon_min, lon_max, lat_min, lat_max = bbox

    positions = []
    names = []
    field_names = [f[0] for f in sf.fields[1:]]  # 跳过删除标记字段
    # 优先使用中文名 (NL_NAME_2)，退而使用拼音 (NAME_2)
    name_idx = None
    is_chinese = False
    for candidate in ["NL_NAME_2", "NAME_2"]:
        if candidate in field_names:
            name_idx = field_names.index(candidate)
            is_chinese = (candidate == "NL_NAME_2")
            break

    for shape_record in sf.iterShapeRecords():
        # 通过边界框快速剔除远离分析域的区域
        shp_bbox = shape_record.shape.bbox  # (lon_min, lat_min, lon_max, lat_max)
        if (shp_bbox[2] < lon_min or shp_bbox[0] > lon_max or
            shp_bbox[3] < lat_min or shp_bbox[1] > lat_max):
            continue

        # 计算区域的近似中心点（取多边形顶点均值）
        geom = shape_record.shape
        points = np.array(geom.points)
        if len(points) < 3:
            continue
        # 近似质心：多边形顶点坐标的算术平均
        clon = points[:, 0].mean()
        clat = points[:, 1].mean()

        if not (lon_min <= clon <= lon_max and lat_min <= clat <= lat_max):
            continue

        # 将中心点转换为ENU坐标
        east, north, up = wgs84_to_enu(
            np.array([clat]), np.array([clon]), np.array([0.0]), ref_lat, ref_lon)
        cx, cy = float(np.asarray(east).item()), float(np.asarray(north).item())

        # 获取名称：对于中文名称，以"|"分隔时取最后部分
        name = ""
        if name_idx is not None and name_idx < len(shape_record.record):
            raw = str(shape_record.record[name_idx])
            if is_chinese and "|" in raw:
                raw = raw.split("|")[-1]
            name = raw.strip()

        if name:
            positions.append((cx, cy))
            names.append(name)

    print(f"  [GIS] City labels in domain: {len(names)}")
    return positions, names


def load_all_boundaries(grid, cache_dir="data/shapefile_cache", dem_querier=None):
    """
    加载县级行政边界和河流水系，返回合并后的线段列表。

    参数：
        grid: AnalysisGrid 分析网格
        cache_dir: GADM shapefile缓存目录
        dem_querier: 可调用对象 (lat, lon) -> elevation_m(米)，None则默认0米

    返回：
        合并后的(N, 3) ENU点数组列表（公里）
    """
    if dem_querier is None:
        dem_querier = lambda lat, lon: 0.0

    counties = load_county_boundaries(grid, cache_dir, dem_querier)
    rivers = load_rivers(grid, dem_querier)

    return counties + rivers
