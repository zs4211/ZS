"""
三维流线生成模块：从反演风场中识别中气旋区域并生成螺旋流线。

核心功能：
    1. 构建三维风场插值器，用于在任意空间位置查询风速矢量
    2. 使用四阶龙格-库塔(RK4)方法追踪流线，实现高精度轨迹积分
    3. 计算涡度(vorticity)和螺旋度(helicity)用于中气旋识别
    4. 在中气旋候选位置周围生成同心环状种子点的螺旋流线
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def wind_interpolator(u, v, w, x, y, z):
    """
    创建三维风场的插值器函数。

    参数：
        u, v, w: 三维风场分量数组，可以是展平的一维或三维形式
        x, y, z: 一维坐标轴

    返回：
        三个 RegularGridInterpolator 对象，分别对应 u、v、w 分量

    使用线性插值，边界外返回0（静止风），确保流线在离开分析域时自然终止。
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    if u.ndim == 1:
        shape = (len(x), len(y), len(z))
        u = u.reshape(shape)
        v = v.reshape(shape)
        w = w.reshape(shape)

    return (
        RegularGridInterpolator((x, y, z), u, bounds_error=False, fill_value=0.0),
        RegularGridInterpolator((x, y, z), v, bounds_error=False, fill_value=0.0),
        RegularGridInterpolator((x, y, z), w, bounds_error=False, fill_value=0.0),
    )


def trace_streamline(u_field, v_field, w_field, seed, x_grid, y_grid, z_grid,
                     step_size=0.2, max_steps=200, direction='both',
                     w_scale=1.0, speed_min=0.5, return_speeds=False):
    """使用四阶龙格-库塔(RK4)方法追踪单条三维流线。

    参数：
        u_field, v_field, w_field: RegularGridInterpolator 风速分量插值器
        seed: (x, y, z) 流线起点
        step_size: 积分步长（公里）
        max_steps: 最大积分步数
        direction: 'forward' / 'backward' / 'both'
        w_scale: 垂直速度放大系数（仅影响积分路径，不影响返回的速度值）
        speed_min: 速度低于此阈值（m/s）时终止积分
        return_speeds: 若为True，返回 (pts, speeds) 元组，speeds为真实物理风速

    返回：
        (N, 3) 数组，或 (pts, speeds) 元组（当 return_speeds=True）
    """
    x_min, x_max = x_grid[0], x_grid[-1]
    y_min, y_max = y_grid[0], y_grid[-1]
    z_min, z_max = z_grid[0], z_grid[-1]

    # 预分配查询缓冲区，避免每步创建新数组（关键性能优化）
    _xi = np.empty((1, 3), dtype=np.float64)

    def _query(px, py, pz):
        _xi[0, 0] = px; _xi[0, 1] = py; _xi[0, 2] = pz
        return (float(u_field(_xi)[0]),
                float(v_field(_xi)[0]),
                float(w_field(_xi)[0]) * w_scale)

    def _in_bounds(px, py, pz):
        return (x_min <= px <= x_max and
                y_min <= py <= y_max and
                z_min <= pz <= z_max)

    def _integrate(dt_sign):
        pts = np.empty((max_steps + 1, 3), dtype=np.float64)
        spd = np.empty(max_steps + 1, dtype=np.float64) if return_speeds else None
        n = 0
        px, py, pz = seed

        for _ in range(max_steps):
            if not _in_bounds(px, py, pz):
                break

            # k1 = 当前位置的速度
            vu, vv, vw = _query(px, py, pz)
            s = np.sqrt(vu*vu + vv*vv + vw*vw)
            if s < 1e-6:
                break

            pts[n, 0] = px; pts[n, 1] = py; pts[n, 2] = pz
            if spd is not None:
                spd[n] = np.sqrt(vu*vu + vv*vv + (vw / w_scale)**2)
            n += 1

            if s < speed_min:
                break

            dt = dt_sign * step_size / max(s, 1e-3)
            k1u, k1v, k1w = vu, vv, vw

            k2u, k2v, k2w = _query(px + dt*k1u/2, py + dt*k1v/2, pz + dt*k1w/2)
            k3u, k3v, k3w = _query(px + dt*k2u/2, py + dt*k2v/2, pz + dt*k2w/2)
            k4u, k4v, k4w = _query(px + dt*k3u, py + dt*k3v, pz + dt*k3w)

            px += dt * (k1u + 2*k2u + 2*k3u + k4u) / 6
            py += dt * (k1v + 2*k2v + 2*k3v + k4v) / 6
            pz += dt * (k1w + 2*k2w + 2*k3w + k4w) / 6

        return pts[:n], (spd[:n] if spd is not None else None)

    if direction == 'both':
        fwd_pts, fwd_spd = _integrate(1.0)
        bwd_pts, bwd_spd = _integrate(-1.0)
        n_fwd = len(fwd_pts)
        n_bwd = len(bwd_pts)
        total = n_fwd + n_bwd

        result = np.empty((total, 3), dtype=np.float64)
        result[:n_bwd] = bwd_pts[::-1]
        result[n_bwd:] = fwd_pts

        if return_speeds:
            result_spd = np.empty(total, dtype=np.float64)
            result_spd[:n_bwd] = bwd_spd[::-1]
            result_spd[n_bwd:] = fwd_spd
            return result, result_spd
        return result

    elif direction == 'forward':
        pts, spd = _integrate(1.0)
        return (pts, spd) if return_speeds else pts
    else:
        pts, spd = _integrate(-1.0)
        if return_speeds:
            return pts[::-1], spd[::-1]
        return pts[::-1]


def trace_streamline_fast(u_arr, v_arr, w_arr, x_arr, y_arr, z_arr, seed,
                          step_size=0.2, max_steps=200, direction='both',
                          w_scale=1.0, speed_min=0.5, return_speeds=False):
    """trace_streamline 的快速版本，使用内联三线性插值代替 RegularGridInterpolator。

    核心性能优化：
        - 直接整数除法计算网格索引（规则网格，O(1)，替代二分查找 O(log N)）
        - 内联三线性插值（避免 scipy RegularGridInterpolator 的函数调用开销）
        - 预分配 numpy 数组存储结果（避免 list append 开销）

    参数和返回值与 trace_streamline 相同。
    """
    nx, ny, nz = len(x_arr), len(y_arr), len(z_arr)
    dx = x_arr[1] - x_arr[0]
    dy = y_arr[1] - y_arr[0]
    dz = z_arr[1] - z_arr[0]
    x0, y0, z0 = x_arr[0], y_arr[0], z_arr[0]
    x_last, y_last, z_last = x_arr[-1], y_arr[-1], z_arr[-1]

    # 预计算倒数，避免每步做除法
    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dz = 1.0 / dz

    def _query(px, py, pz):
        """直接三线性插值查询 (u,v,w) —— 热路径，已极致优化。"""
        # 边界检查（内联，避免函数调用）
        if px < x0 or px > x_last or py < y0 or py > y_last or pz < z0 or pz > z_last:
            return 0.0, 0.0, 0.0

        ix_f = (px - x0) * inv_dx
        iy_f = (py - y0) * inv_dy
        iz_f = (pz - z0) * inv_dz

        ix = int(ix_f)
        iy = int(iy_f)
        iz = int(iz_f)

        # 确保在 [0, n-2] 范围内（边界附近可能有浮点舍入）
        if ix < 0 or ix >= nx - 1 or iy < 0 or iy >= ny - 1 or iz < 0 or iz >= nz - 1:
            return 0.0, 0.0, 0.0

        xd = ix_f - ix
        yd = iy_f - iy
        zd = iz_f - iz

        # 手动展开三线性插值，避免循环和临时分配
        c000 = u_arr[ix, iy, iz]
        c100 = u_arr[ix+1, iy, iz]
        c010 = u_arr[ix, iy+1, iz]
        c110 = u_arr[ix+1, iy+1, iz]
        c001 = u_arr[ix, iy, iz+1]
        c101 = u_arr[ix+1, iy, iz+1]
        c011 = u_arr[ix, iy+1, iz+1]
        c111 = u_arr[ix+1, iy+1, iz+1]
        c00 = c000 + (c100 - c000) * xd
        c01 = c001 + (c101 - c001) * xd
        c10 = c010 + (c110 - c010) * xd
        c11 = c011 + (c111 - c011) * xd
        c0 = c00 + (c10 - c00) * yd
        c1 = c01 + (c11 - c01) * yd
        u_val = c0 + (c1 - c0) * zd

        c000 = v_arr[ix, iy, iz]
        c100 = v_arr[ix+1, iy, iz]
        c010 = v_arr[ix, iy+1, iz]
        c110 = v_arr[ix+1, iy+1, iz]
        c001 = v_arr[ix, iy, iz+1]
        c101 = v_arr[ix+1, iy, iz+1]
        c011 = v_arr[ix, iy+1, iz+1]
        c111 = v_arr[ix+1, iy+1, iz+1]
        c00 = c000 + (c100 - c000) * xd
        c01 = c001 + (c101 - c001) * xd
        c10 = c010 + (c110 - c010) * xd
        c11 = c011 + (c111 - c011) * xd
        c0 = c00 + (c10 - c00) * yd
        c1 = c01 + (c11 - c01) * yd
        v_val = c0 + (c1 - c0) * zd

        c000 = w_arr[ix, iy, iz]
        c100 = w_arr[ix+1, iy, iz]
        c010 = w_arr[ix, iy+1, iz]
        c110 = w_arr[ix+1, iy+1, iz]
        c001 = w_arr[ix, iy, iz+1]
        c101 = w_arr[ix+1, iy, iz+1]
        c011 = w_arr[ix, iy+1, iz+1]
        c111 = w_arr[ix+1, iy+1, iz+1]
        c00 = c000 + (c100 - c000) * xd
        c01 = c001 + (c101 - c001) * xd
        c10 = c010 + (c110 - c010) * xd
        c11 = c011 + (c111 - c011) * xd
        c0 = c00 + (c10 - c00) * yd
        c1 = c01 + (c11 - c01) * yd
        w_val = c0 + (c1 - c0) * zd

        return u_val, v_val, w_val * w_scale

    def _integrate(dt_sign):
        pts = np.empty((max_steps + 1, 3), dtype=np.float64)
        spd = np.empty(max_steps + 1, dtype=np.float64) if return_speeds else None
        n = 0
        px, py, pz = seed

        for _ in range(max_steps):
            # k1 = 当前位置的速度
            vu, vv, vw = _query(px, py, pz)
            s = vu*vu + vv*vv + vw*vw
            if s < 1e-12:
                break
            s = np.sqrt(s)

            pts[n, 0] = px; pts[n, 1] = py; pts[n, 2] = pz
            if spd is not None:
                spd[n] = np.sqrt(vu*vu + vv*vv + (vw / w_scale)**2)
            n += 1

            if s < speed_min:
                break

            dt = dt_sign * step_size / max(s, 1e-3)
            k1u, k1v, k1w = vu, vv, vw

            k2u, k2v, k2w = _query(px + dt*k1u/2, py + dt*k1v/2, pz + dt*k1w/2)
            k3u, k3v, k3w = _query(px + dt*k2u/2, py + dt*k2v/2, pz + dt*k2w/2)
            k4u, k4v, k4w = _query(px + dt*k3u, py + dt*k3v, pz + dt*k3w)

            px += dt * (k1u + 2*k2u + 2*k3u + k4u) / 6
            py += dt * (k1v + 2*k2v + 2*k3v + k4v) / 6
            pz += dt * (k1w + 2*k2w + 2*k3w + k4w) / 6

        return pts[:n], (spd[:n] if spd is not None else None)

    if direction == 'both':
        fwd_pts, fwd_spd = _integrate(1.0)
        bwd_pts, bwd_spd = _integrate(-1.0)
        n_fwd = len(fwd_pts)
        n_bwd = len(bwd_pts)
        total = n_fwd + n_bwd

        result = np.empty((total, 3), dtype=np.float64)
        result[:n_bwd] = bwd_pts[::-1]
        result[n_bwd:] = fwd_pts

        if return_speeds:
            result_spd = np.empty(total, dtype=np.float64)
            result_spd[:n_bwd] = bwd_spd[::-1]
            result_spd[n_bwd:] = fwd_spd
            return result, result_spd
        return result

    elif direction == 'forward':
        pts, spd = _integrate(1.0)
        return (pts, spd) if return_speeds else pts
    else:
        pts, spd = _integrate(-1.0)
        if return_speeds:
            return pts[::-1], spd[::-1]
        return pts[::-1]


def generate_meso_streamlines(u, v, w, x_grid, y_grid, z_grid,
                              meso_positions,
                              n_rings=4, seeds_per_ring=15,
                              ring_min_r=3.0, ring_max_r=12.0,
                              z_floor=0.5, z_ceil=2.0,
                              min_length=12, max_steps=300,
                              step_size=0.3, w_scale=None):
    """
    在中气旋位置周围的低层同心环上生成种子点并追踪螺旋流线。

    种子策略：
        在每个中气旋候选位置周围生成 n_rings 个同心环，每个环上等间距分布
        seeds_per_ring 个种子点。种子高度在 [z_floor, z_ceil] 公里范围内随机分布。
        这种多层多角度的种子布局能够捕获气流从各个方向流入 → 辐合 → 上升的
        完整三维结构。

    参数：
        u, v, w: 三维风场数组
        x_grid, y_grid, z_grid: 坐标轴
        meso_positions: [(mx, my, mz), ...] 中气旋中心位置列表
        n_rings: 每个中气旋的种子环数
        seeds_per_ring: 每个环的种子点数
        ring_min_r, ring_max_r: 环半径范围（公里），控制入流采样的空间尺度
        z_floor, z_ceil: 种子高度范围（公里），覆盖低层入流区
        min_length: 最小流线长度（点数），过滤掉太短的无效流线
        max_steps, step_size: RK4积分的步数和步长
        w_scale: 垂直速度放大系数。None时自动计算，使 max(w_scaled) ≈ 0.35*max(水平风速)
                 手动指定可控制螺旋上升的视觉表现力

    返回：
        streamlines: 流线列表，每条为 (N, 3) 数组
        speeds: 速度列表，每条为 (N,) 数组（真实物理风速，非缩放值）
    """
    interp_u, interp_v, interp_w = wind_interpolator(u, v, w, x_grid, y_grid, z_grid)

    # 自动平衡垂直放大系数：使最大垂直速度约为最大水平速度的35%
    # 这样在视觉上能清晰展示上升气流，同时保持合理的物理比例
    if w_scale is None:
        vh = np.sqrt(np.nanmax(u**2 + v**2))
        wm = np.nanmax(np.abs(w))
        if wm > 1e-3:
            w_scale = 0.35 * vh / wm
        else:
            w_scale = 1.0
        w_scale = np.clip(w_scale, 1.5, 8.0)
        print(f"    Auto w_scale = {w_scale:.2f}  (Vh_max={vh:.1f}, w_max={wm:.1f} m/s)")

    streamlines = []
    speeds = []
    rng = np.random.RandomState(42)

    for (mx, my, _) in meso_positions:
        for ring_i in range(n_rings):
            r = ring_min_r + (ring_max_r - ring_min_r) * ring_i / max(n_rings - 1, 1)
            for s in range(seeds_per_ring):
                # 在每个环上均匀分布种子，并添加小量随机扰动避免完全对称
                angle = 2 * np.pi * s / seeds_per_ring + rng.uniform(-0.1, 0.1)
                z_seed = z_floor + rng.uniform(0, z_ceil - z_floor)

                seed = (mx + r * np.cos(angle),
                        my + r * np.sin(angle),
                        z_seed)

                pts = trace_streamline(
                    interp_u, interp_v, interp_w,
                    seed, x_grid, y_grid, z_grid,
                    step_size=step_size, max_steps=max_steps,
                    direction='both', w_scale=w_scale
                )

                if len(pts) >= min_length:
                    all_pts = np.array(pts)
                    # 使用真实物理风速（未缩放）进行颜色映射
                    speed_vals = np.sqrt(
                        interp_u(all_pts)**2 +
                        interp_v(all_pts)**2 +
                        interp_w(all_pts)**2
                    )
                    streamlines.append(all_pts)
                    speeds.append(speed_vals)

    return streamlines, speeds


def detect_vortex_region(u, v, w, grid_shape, dx=1.0, dy=1.0, dz=0.5,
                         vorticity_threshold=0.005):
    """
    通过垂直涡度 ζ = dv/dx - du/dy 检测涡旋区域。

    使用中心差分法计算水平风场的空间导数：
    dv/dx ≈ [v(i+1,j,k) - v(i-1,j,k)] / (2*dx)
    du/dy ≈ [u(i,j+1,k) - u(i,j-1,k)] / (2*dy)

    正值涡度对应气旋式旋转（北半球逆时针），负值对应反气旋式旋转。
    此处取绝对值，对两种旋转方向都进行检测。
    """
    nx, ny, nz = grid_shape
    dvdx = np.zeros_like(v)
    dudy = np.zeros_like(u)
    dvdx[1:-1, :, :] = (v[2:, :, :] - v[:-2, :, :]) / (2 * dx)
    dudy[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dy)
    vorticity = dvdx - dudy
    return np.abs(vorticity) > vorticity_threshold, vorticity


def compute_helicity(u, v, w, grid_shape, dx=1.0, dy=1.0, dz=0.5):
    """
    计算风暴相对螺旋度(SRH)密度。

    螺旋度 H = u·ω_x + v·ω_y + w·ω_z = V · (∇ × V)
    其中 ω = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy) 为涡度矢量。

    在气象学中，螺旋度是衡量气流旋转潜势的重要指标。
    高螺旋度区域通常与旋转上升气流（中气旋）相关联，
    是超级单体风暴和龙卷风形成的关键前兆因子。

    所有空间导数均使用中心差分格式计算，边界处为0。
    """
    nx, ny, nz = grid_shape
    dwdy = np.zeros_like(w)
    dvdz = np.zeros_like(v)
    dudz = np.zeros_like(u)
    dwdx = np.zeros_like(w)
    dvdx = np.zeros_like(v)
    dudy = np.zeros_like(u)

    # 中心差分计算各偏导数
    dwdy[:, 1:-1, :] = (w[:, 2:, :] - w[:, :-2, :]) / (2 * dy)
    dvdz[:, :, 1:-1] = (v[:, :, 2:] - v[:, :, :-2]) / (2 * dz)
    dudz[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dz)
    dwdx[1:-1, :, :] = (w[2:, :, :] - w[:-2, :, :]) / (2 * dx)
    dvdx[1:-1, :, :] = (v[2:, :, :] - v[:-2, :, :]) / (2 * dx)
    dudy[:, 1:-1, :] = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dy)

    # 涡度矢量的三个分量
    omega_x = dwdy - dvdz
    omega_y = dudz - dwdx
    omega_z = dvdx - dudy
    # 螺旋度 = 速度矢量与涡度矢量的点积
    helicity = u * omega_x + v * omega_y + w * omega_z
    return helicity, omega_x, omega_y, omega_z


def identify_meso_candidates(helicity, vortex_mask, x, y, z,
                             top_n=5, min_volume=10):
    """
    从螺旋度和涡度信息中识别中气旋候选区域。

    识别策略：
        1. 取螺旋度95百分位作为阈值，筛选高螺旋度区域
        2. 与涡旋掩膜做交集，确保既有旋转又有上升
        3. 使用连通域标记算法(connected-component labeling)将相邻高值区聚类
        4. 对每个连通域计算其质心坐标、最大螺旋度和体积
        5. 按最大螺旋度降序排列，返回前 top_n 个候选

    参数：
        helicity: 螺旋度数组 (nx, ny, nz)
        vortex_mask: 涡旋布尔掩膜
        x, y, z: 坐标轴
        top_n: 返回的候选数量上限
        min_volume: 最小体积（网格点数），过滤噪声

    返回：
        候选列表 [(cx, cy, cz, max_helicity, volume), ...]
    """
    from scipy.ndimage import label

    hel_threshold = np.percentile(helicity[np.isfinite(helicity)], 95)
    high_hel = (helicity > hel_threshold) & vortex_mask
    labeled, n_features = label(high_hel)

    candidates = []
    for feat_id in range(1, n_features + 1):
        region_mask = labeled == feat_id
        vol = region_mask.sum()
        if vol < min_volume:
            continue
        x_idx, y_idx, z_idx = np.where(region_mask)
        candidates.append((
            x[x_idx].mean(), y[y_idx].mean(), z[z_idx].mean(),
            helicity[region_mask].max(), vol
        ))
    candidates.sort(key=lambda c: c[3], reverse=True)
    return candidates[:top_n]
