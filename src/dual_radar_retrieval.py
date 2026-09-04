"""
双多普勒雷达直接风场合成。

通过在每个网格点上联立求解两部雷达的径向速度方程，实现传统的
双雷达风场反演。这为三维变分分析提供了一个物理上有意义的初始风场，
能加速收敛并提高强风切变区域（如中气旋）的反演精度。

算法原理 (Ray et al. 1980, Armijo 1969):
  1. 对每个网格点，计算来自两部雷达的单位波束方向向量
  2. 在 w=0 的假设下，求解 2×2 线性方程组得到水平风场 (u, v)
     [bx1  by1] [u]   [Vr1 - bz1·w]
     [bx2  by2] [v] = [Vr2 - bz2·w]
     解的行列式为 det = bx1·by2 - by1·bx2，
     当两部雷达的交叉波束角接近 90° 时行列式最大，解最稳定
  3. 通过从区域顶部向下积分质量连续方程，计算垂直速度 w
     ∂u/∂x + ∂v/∂y + ∂w/∂z = 0  →  ∂w/∂z = -(∂u/∂x + ∂v/∂y)
  4. 可选的迭代过程：用更新后的 w 重新求解 (u, v)，反复迭代直至收敛
"""

import numpy as np
from typing import Tuple

from src.gridder import wgs84_to_enu


def dual_radar_retrieve(
    grid1, grid2, radar1, radar2,
    vel_field: str = "Vc",
    min_cross_beam_angle_deg: float = 15.0,
    w_top: float = 0.0,
    max_iterations: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """双多普勒雷达直接风场合成：从两个网格化雷达体扫数据中反演三维风场。

    算法步骤（逐网格点执行）:

    步骤 A — 波束几何计算:
      对每个网格点 (x, y, z)，计算从雷达到该点的单位波束向量
      (bx, by, bz)。这是径向速度投影方程的核心几何量:
        Vr = u·bx + v·by + w·bz

    步骤 B — 2×2 线性系统求解 (u, v):
      对于两部雷达 i = {1, 2}:
        bx_i·u + by_i·v = Vr_i - bz_i·w    (将 w 项移到右侧)
      用克莱姆法则求解:
        det = bx1·by2 - by1·bx2
        当 det → 0 时（即交叉波束角 → 0° 或 180°），解不稳定。
        通过 min_cross_beam_angle 阈值剔除这些点。

    步骤 C — 质量连续方程积分求 w:
      从区域顶部 (k = nz-1) 开始，已知上边界条件 w_top，
      逐层向下积分:
        w[k-1] = w[k] + [du/dx + dv/dy]·dz

    步骤 D — 迭代:
      用更新后的 w 重新代入 2×2 系统求解 (u, v)，
      重复步骤 B-C 直至收敛或达到最大迭代次数。

    Parameters
    ----------
    grid1, grid2 : pyart.core.Grid
        每部雷达的网格化径向速度数据，对齐到相同的笛卡尔网格。
    radar1, radar2 : pyart.core.Radar
        原始 PyART Radar 对象（用于获取测站位置信息）。
    vel_field : str
        Grid 对象中速度场的名称，默认 "Vc"。
    min_cross_beam_angle_deg : float
        保证良态求解的最小交叉波束角（度）。
        推荐值: 20°（保守），30°（业务标准）。
    w_top : float
        区域顶部的假定垂直速度 (m/s)，作为质量连续积分的上边界条件。
    max_iterations : int
        w 反馈迭代次数（0 = 仅一次求解，已足够作为初始猜测场）。

    Returns
    -------
    u, v, w : np.ndarray, 形状 (nz, ny, nx)
        反演的三个风场分量 (m/s)。
    valid_mask : np.ndarray (bool), 形状 (nz, ny, nx)
        True 表示该网格点上有有效的双多普勒解。
    cross_beam_angle : np.ndarray, 形状 (nz, ny, nx)
        两条水平波束的锐夹角（度，范围 0~90）。
    """
    # ---- 网格几何信息（PyART 约定: 单位米） ----
    x = np.asarray(grid1.x["data"], dtype=np.float64)   # 东西方向, 长度 nx
    y = np.asarray(grid1.y["data"], dtype=np.float64)   # 南北方向, 长度 ny
    z = np.asarray(grid1.z["data"], dtype=np.float64)   # 垂直方向, 长度 nz

    nz, ny, nx = len(z), len(y), len(x)
    dx = x[1] - x[0] if nx > 1 else 1.0
    dy = y[1] - y[0] if ny > 1 else 1.0
    dz = z[1] - z[0] if nz > 1 else 500.0

    origin_lat = float(grid1.origin_latitude["data"][0])
    origin_lon = float(grid1.origin_longitude["data"][0])
    origin_alt_m = float(grid1.origin_altitude["data"][0])

    # ---- 计算雷达在网格 ENU 坐标系中的位置 ----
    # wgs84_to_enu 的单位是 km，需要转换为 m
    def _radar_enu(radar):
        r_lat = float(radar.latitude["data"][0])
        r_lon = float(radar.longitude["data"][0])
        r_alt_m = float(radar.altitude["data"][0])
        e_km, n_km, u_km = wgs84_to_enu(
            np.array([r_lat]), np.array([r_lon]),
            np.array([r_alt_m / 1000.0]),
            origin_lat, origin_lon, origin_alt_m / 1000.0)
        return (float(e_km.item()) * 1000.0,
                float(n_km.item()) * 1000.0,
                float(u_km.item()) * 1000.0)

    r1_e, r1_n, r1_u = _radar_enu(radar1)
    r2_e, r2_n, r2_u = _radar_enu(radar2)

    # ---- 网格化径向速度（无数据处已掩膜） ----
    vr1_full = grid1.fields[vel_field]["data"]
    vr2_full = grid2.fields[vel_field]["data"]
    vr1 = np.ma.getdata(vr1_full).astype(np.float64)
    vr2 = np.ma.getdata(vr2_full).astype(np.float64)
    mask1 = np.ma.getmaskarray(vr1_full) if np.ma.is_masked(vr1_full) else np.isnan(vr1)
    mask2 = np.ma.getmaskarray(vr2_full) if np.ma.is_masked(vr2_full) else np.isnan(vr2)

    # ---- 预计算波束几何信息 (nz, ny, nx) ----
    print(f"[DualRadar] 计算波束几何: {nz}x{ny}x{nx} 网格...")
    print(f"            雷达1 ENU: ({r1_e/1000:.1f}, {r1_n/1000:.1f}, {r1_u/1000:.1f}) km")
    print(f"            雷达2 ENU: ({r2_e/1000:.1f}, {r2_n/1000:.1f}, {r2_u/1000:.1f}) km")

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")   # 形状均为 (nz, ny, nx)

    # 从每部雷达到每个网格点的距离向量
    dx1 = X - r1_e; dy1 = Y - r1_n; dz1 = Z - r1_u
    rng1 = np.sqrt(dx1**2 + dy1**2 + dz1**2)

    dx2 = X - r2_e; dy2 = Y - r2_n; dz2 = Z - r2_u
    rng2 = np.sqrt(dx2**2 + dy2**2 + dz2**2)

    # 单位波束向量 (bx = 东分量, by = 北分量, bz = 垂直分量)
    with np.errstate(divide="ignore", invalid="ignore"):
        bx1, by1, bz1 = dx1 / rng1, dy1 / rng1, dz1 / rng1
        bx2, by2, bz2 = dx2 / rng2, dy2 / rng2, dz2 / rng2

    # ---- 交叉波束角质量掩膜 ----
    # 使用水平投影后的单位方向计算锐夹角，避免仰角的 cos(phi) 因子
    # 被错误地混入用户设置的“水平交叉角”阈值。
    det_raw = bx1 * by2 - by1 * bx2
    h1 = np.hypot(dx1, dy1)
    h2 = np.hypot(dx2, dy2)
    with np.errstate(divide="ignore", invalid="ignore"):
        dot_h = (dx1 * dx2 + dy1 * dy2) / (h1 * h2)
        cross_beam_angle = np.degrees(
            np.arccos(np.clip(np.abs(dot_h), 0.0, 1.0))
        )
    geom_mask = cross_beam_angle >= min_cross_beam_angle_deg

    valid_mask = ~mask1 & ~mask2 & geom_mask & (rng1 > 0) & (rng2 > 0)
    n_valid = valid_mask.sum()
    n_total = nz * ny * nx
    print(f"            双多普勒有效点: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")

    # ---- 初始化输出阵 (nz, ny, nx) ----
    u = np.zeros((nz, ny, nx), dtype=np.float32)
    v = np.zeros((nz, ny, nx), dtype=np.float32)
    w = np.zeros((nz, ny, nx), dtype=np.float32)

    # ---- 在每个有效网格点求解 2×2 线性方程组 ----
    # 方程组:
    #   [bx1  by1] [u]   [Vr1 - bz1·w]
    #   [bx2  by2] [v] = [Vr2 - bz2·w]
    #
    # 解（克莱姆法则）:
    #   u = ( by2·(Vr1 - bz1·w) - by1·(Vr2 - bz2·w) ) / d
    #   v = ( bx1·(Vr2 - bz2·w) - bx2·(Vr1 - bz1·w) ) / d

    def _solve_uv(vr1, vr2, bx1, by1, bz1, bx2, by2, bz2, w_cur, det_val, idx):
        """在有效网格点处使用克莱姆法则求解水平风场 (u, v)。"""
        u[idx] = ((by2[idx] * (vr1[idx] - bz1[idx] * w_cur[idx])
                   - by1[idx] * (vr2[idx] - bz2[idx] * w_cur[idx]))
                  / det_val)
        v[idx] = ((bx1[idx] * (vr2[idx] - bz2[idx] * w_cur[idx])
                   - bx2[idx] * (vr1[idx] - bz1[idx] * w_cur[idx]))
                  / det_val)

    idx = np.where(valid_mask)
    d = det_raw[idx]  # 有符号行列式，用于正确求解

    # 第一次求解: 假设 w = 0
    _solve_uv(vr1, vr2, bx1, by1, bz1, bx2, by2, bz2, w, d, idx)

    # ---- 通过质量连续方程计算垂直速度 ----
    # 不可压缩质量连续方程: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
    # 因此 ∂w/∂z = -(∂u/∂x + ∂v/∂y)
    # 从顶部向下积分: w[k-1] = w[k] + (du/dx + dv/dy)·dz
    #
    # 物理含义: 如果某一层存在水平辐合 (du/dx + dv/dy < 0)，
    # 则由质量连续方程可知，必须有垂直速度向上递增来补偿质量损失

    def _compute_w(u, v):
        dudx = np.zeros_like(u)
        dvdy = np.zeros_like(v)
        # 只在中心点及两侧邻点都有效时计算中心差分。禁止跨越
        # “无观测=0”边界求梯度，否则会产生虚假辐合并污染整列 w。
        x_stencil = np.zeros_like(valid_mask)
        y_stencil = np.zeros_like(valid_mask)
        x_stencil[:, :, 1:-1] = (
            valid_mask[:, :, :-2]
            & valid_mask[:, :, 1:-1]
            & valid_mask[:, :, 2:]
        )
        y_stencil[:, 1:-1, :] = (
            valid_mask[:, :-2, :]
            & valid_mask[:, 1:-1, :]
            & valid_mask[:, 2:, :]
        )
        x_values = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)
        y_values = (v[:, 2:, :] - v[:, :-2, :]) / (2 * dy)
        dudx[:, :, 1:-1] = np.where(x_stencil[:, :, 1:-1], x_values, 0.0)
        dvdy[:, 1:-1, :] = np.where(y_stencil[:, 1:-1, :], y_values, 0.0)
        derivative_valid = x_stencil & y_stencil
        divergence = np.where(derivative_valid, dudx + dvdy, 0.0)
        w_new = np.zeros_like(u)
        w_new[-1, :, :] = np.where(valid_mask[-1], w_top, 0.0)
        # 从顶部向下逐层积分
        for k in range(nz - 2, -1, -1):
            column_valid = (
                derivative_valid[k]
                & valid_mask[k]
                & valid_mask[k + 1]
            )
            w_new[k] = np.where(
                column_valid,
                w_new[k + 1] + divergence[k] * dz,
                0.0,
            )
        return w_new

    w = _compute_w(u, v)

    # ---- 迭代: 用更新的 w 重新求解 (u, v) ----
    for iteration in range(max_iterations):
        u_prev, v_prev = u.copy(), v.copy()
        _solve_uv(vr1, vr2, bx1, by1, bz1, bx2, by2, bz2, w, d, idx)
        w = _compute_w(u, v)

        du_max = np.abs(u - u_prev).max()
        dv_max = np.abs(v - v_prev).max()
        print(f"            Iter {iteration+1}: max|Δu|={du_max:.2f}, "
              f"max|Δv|={dv_max:.2f}")
        if du_max < 0.01 and dv_max < 0.01:
            break

    # ---- 掩膜掉无效网格点（设为 0） ----
    u[~valid_mask] = 0.0
    v[~valid_mask] = 0.0
    w[~valid_mask] = 0.0

    # ---- 输出统计摘要 ----
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    w_valid = w[valid_mask]
    if len(u_valid) > 0:
        print(f"[DualRadar] u: [{u_valid.min():.1f}, {u_valid.max():.1f}] m/s")
        print(f"[DualRadar] v: [{v_valid.min():.1f}, {v_valid.max():.1f}] m/s")
        print(f"[DualRadar] w: [{w_valid.min():.2f}, {w_valid.max():.2f}] m/s")

    return u, v, w, valid_mask, cross_beam_angle.astype(np.float32)
