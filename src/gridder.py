"""
坐标变换与三维分析网格构建模块。

核心功能:
  1. WGS84 大地坐标 → ENU（东-北-天）局部笛卡尔坐标转换
  2. 三维分析网格 (AnalysisGrid) 数据结构定义
  3. CR组合反射率笛卡尔产品插值到分析网格

坐标系说明:
  - WGS84 (BLH): 大地经纬度 + 海拔高度，雷达数据的原生坐标系
  - ECEF (XYZ): 地心地固直角坐标，WGS84椭球体上的三维笛卡尔坐标
  - ENU (东北天): 以网格原点为参考的局部笛卡尔坐标，x=东, y=北, z=天
    所有分析计算均在 ENU 坐标系下进行，单位统一为公里(km)

坐标转换方法:
  WGS84 → ECEF: 使用 WGS84 椭球参数 (a=6378.137km, f=1/298.257)
  ECEF → ENU: 通过参考点的经纬度构造旋转矩阵，将 ECEF 差向量投影到局部切平面
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class AnalysisGrid:
    """用于双多普勒三维风场反演的笛卡尔分析网格。

    将所有雷达数据转换到统一的 ENU 笛卡尔坐标系下，
    每个网格点同时存储 ENU 坐标 (x, y, z) 和对应的
    经纬度近似值，方便地理信息叠加。

    Attributes
    ----------
    x, y, z : np.ndarray
        一维坐标轴（公里），x=东, y=北, z=高度。
    X, Y, Z : np.ndarray
        三维网格坐标 (nx, ny, nz)，由 meshgrid(x, y, z, indexing='ij') 生成。
    lon_grid, lat_grid : np.ndarray
        每个网格点的经纬度近似值，形状与 X, Y 相同。
    origin_lat, origin_lon : float
        ENU 参考原点的经纬度（度）。
    dx, dy, dz : float
        网格间距（公里）。
    nx, ny, nz : int
        各维度的网格点数。
    """
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    lon_grid: np.ndarray
    lat_grid: np.ndarray
    origin_lat: float
    origin_lon: float
    dx: float
    dy: float
    dz: float
    nx: int
    ny: int
    nz: int


def wgs84_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt=0.0):
    """将WGS84大地坐标转换为东北天(ENU)坐标系，结果单位为公里。

    算法分为两步:
      1. WGS84 (BLH) → ECEF (XYZ):
         使用 WGS84 椭球参数计算卯酉圈曲率半径 N，
         然后通过 N 将经纬度转换为地心地固坐标:
           X = (N + alt) · cosφ · cosλ
           Y = (N + alt) · cosφ · sinλ
           Z = (N · (1-e²) + alt) · sinφ

      2. ECEF (XYZ) → ENU (东北天):
         以参考点为原点，将 ECEF 差值通过旋转矩阵投影到局部切平面:
           [east]   [-sinλ      cosλ        0    ] [ΔX]
           [north] = [-sinφ·cosλ -sinφ·sinλ cosφ] [ΔY]
           [up   ]   [ cosφ·cosλ  cosφ·sinλ sinφ] [ΔZ]

    Parameters
    ----------
    lat, lon : np.ndarray 或 float
        纬度和经度（度）。
    alt : np.ndarray 或 float
        海拔高度（公里）。
    ref_lat, ref_lon : float
        参考原点的经纬度（度）。
    ref_alt : float
        参考原点的高度（公里），默认 0。

    Returns
    -------
    east, north, up : np.ndarray
        东、北、天方向分量（公里）。
    """
    # WGS84 椭球参数
    a = 6378.137           # 长半轴 (km)
    f = 1.0 / 298.257223563  # 扁率
    e2 = 2 * f - f * f     # 第一偏心率平方

    # 角度转弧度
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    ref_lat_r = np.radians(ref_lat)
    ref_lon_r = np.radians(ref_lon)

    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    sin_ref_lat = np.sin(ref_lat_r)
    cos_ref_lat = np.cos(ref_lat_r)

    # 卯酉圈曲率半径: N = a / sqrt(1 - e²·sin²φ)
    N = a / np.sqrt(1.0 - e2 * sin_lat ** 2)
    N_ref = a / np.sqrt(1.0 - e2 * sin_ref_lat ** 2)

    # 大地坐标 → 地心地固坐标 (ECEF)
    X = (N + alt) * cos_lat * np.cos(lon_r)
    Y = (N + alt) * cos_lat * np.sin(lon_r)
    Z = (N * (1 - e2) + alt) * sin_lat

    X_ref = (N_ref + ref_alt) * cos_ref_lat * np.cos(ref_lon_r)
    Y_ref = (N_ref + ref_alt) * cos_ref_lat * np.sin(ref_lon_r)
    Z_ref = (N_ref * (1 - e2) + ref_alt) * sin_ref_lat

    # ECEF 差向量
    dX = X - X_ref
    dY = Y - Y_ref
    dZ = Z - Z_ref

    # ECEF → ENU 旋转矩阵
    sin_phi = sin_ref_lat
    cos_phi = cos_ref_lat
    sin_lam = np.sin(ref_lon_r)
    cos_lam = np.cos(ref_lon_r)

    east = -sin_lam * dX + cos_lam * dY
    north = -sin_phi * cos_lam * dX - sin_phi * sin_lam * dY + cos_phi * dZ
    up = cos_phi * cos_lam * dX + cos_phi * sin_lam * dY + sin_phi * dZ

    return east, north, up


def interp_cartesian_to_grid(cart, grid):
    """将CR组合反射率笛卡尔产品插值到分析网格的水平面上。

    CR数据存储在规则的经纬度网格上（CINRAD CR产品为 1840×1840），
    使用 scipy RegularGridInterpolator 进行线性插值，
    将反射率值映射到分析网格的地面层。

    参数：
        cart: 包含 lon(1D)、lat(1D)、data(2D) 的 CartesianData 对象
        grid: AnalysisGrid 分析网格

    返回：
        形状为 (grid.nx, grid.ny) 的二维反射率数组 (dBZ)。
    """
    from scipy.interpolate import RegularGridInterpolator

    cr_lon = np.asarray(cart.lon, dtype=np.float64)
    cr_lat = np.asarray(cart.lat, dtype=np.float64)

    # cinrad 返回的 CR 数据 dims=["latitude", "longitude"]，即 shape=(nlat, nlon)
    # 转置为 (nlon, nlat)，使 axis=0 对应经度、axis=1 对应纬度，
    # 与 RegularGridInterpolator((cr_lon, cr_lat), ...) 的约定一致
    cr_data = np.asarray(cart.data, dtype=np.float64).T  # (nlat, nlon) → (nlon, nlat)

    # RegularGridInterpolator 要求坐标严格升序
    # CINRAD CR 产品的纬度通常是降序（北→南），需要翻转
    if cr_lat[0] > cr_lat[-1]:
        cr_lat = cr_lat[::-1]
        cr_data = cr_data[:, ::-1]  # 翻转纬度维度（axis=1）

    # 清洗无效数据：NaN 和 <-20 dBZ 的值（地物杂波）设为 NaN
    cr_data_clean = np.where(
        np.isfinite(cr_data) & (cr_data > -20),
        cr_data, np.nan
    )

    # 创建二维规则网格插值器
    interp = RegularGridInterpolator(
        (cr_lon, cr_lat), cr_data_clean,
        bounds_error=False, fill_value=np.nan
    )

    # 在分析网格的地面层（z=0 平面）上查询
    grid_lon = grid.lon_grid[:, :, 0]  # (nx, ny)
    grid_lat = grid.lat_grid[:, :, 0]  # (nx, ny)

    # 构建查询点并执行插值
    qpts = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
    result = interp(qpts)

    return result.reshape((grid.nx, grid.ny))
