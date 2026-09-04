"""
基于 PyDDA 的三维变分分析风场反演。

接收双多普勒雷达网格化数据，使用三维变分分析 (3DVAR) 方法
反演完整的三维风场 (u, v, w)。

══════════════════════════════════════════════════════════════
工作流程
══════════════════════════════════════════════════════════════
  1. 将每部雷达分别网格化到统一的笛卡尔网格
  2. 构建初始猜测风场（VWP 风廓线 + 双多普勒直接合成融合）
  3. 运行 PyDDA 三维变分分析优化
  4. 保存结果并生成诊断图

══════════════════════════════════════════════════════════════
三维变分分析 (3DVAR) 算法详解
══════════════════════════════════════════════════════════════

3DVAR 是一类反问题求解方法，目标是通过最小化一个标量代价函数 J(x)
来找到最优的三维风场估计 x = (u, v, w)。

【1. 数学问题表述】
  给定:
    - 两部雷达的径向速度观测 Vr₁, Vr₂（网格化到统一的笛卡尔网格）
    - 初始猜测风场 (u_bg, v_bg, w_bg)
    - 雷达波束几何信息（每个网格点的方位角、仰角）

  求解:
    (u*, v*, w*) = argmin J(u, v, w)

  其中 J 是包含多个约束项的标量代价函数。

【2. 代价函数 J 的完整形式】

  J(u,v,w) = Jo + Jb + Jm + Js

  ┌─────────────────────────────────────────────────────────┐
  │ 2.1  观测项 Jo (Observation term)                       │
  │                                                         │
  │   Jo = Co · Σᵢ Σⱼ [Vr_obs(xⱼ, rᵢ) - Vr_ret(xⱼ, rᵢ)]²  │
  │                                                         │
  │   其中:                                                  │
  │     i = {1, 2} 表示两部雷达                              │
  │     j 遍历第 i 部雷达的所有有效观测网格点                │
  │     Vr_obs 是雷达实际观测的径向速度                      │
  │     Vr_ret 是反演风场投影到雷达波束方向上的径向速度      │
  │     Co  是观测项权重                                     │
  │                                                         │
  │   径向速度前向算子（观测算子 H）:                         │
  │     Vr = u·bx + v·by + w·bz                             │
  │                                                         │
  │   其中 (bx, by, bz) 是从雷达到网格点的单位波束方向向量: │
  │     bx = (x - x_radar) / R    (东分量)                  │
  │     by = (y - y_radar) / R    (北分量)                  │
  │     bz = (z - z_radar) / R    (垂直分量)                │
  │     R  = sqrt[(x-x_r)² + (y-y_r)² + (z-z_r)²]  (斜距) │
  │                                                         │
  │   物理含义: 惩罚反演风场在雷达波束方向上的投影与实际     │
  │   观测径向速度之间的差异。这是数据拟合的核心项。         │
  │   权重 Co 越大 → 越信任观测 → 反演结果更贴近雷达数据      │
  │   权重 Co 过小 → 反演结果对观测不敏感，退化为背景场       │
  │   权重 Co 过大 → 对观测噪声敏感，产生虚假的小尺度结构     │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │ 2.2  背景项 Jb (Background term)                       │
  │                                                         │
  │   Jb = Cb · Σ [u(x) - u_bg(x)]² + [v - v_bg]² + [w]²   │
  │                                                         │
  │   物理含义: 惩罚反演风场与初始猜测场之间的偏离。          │
  │   这一项的作用类似于贝叶斯框架中的先验分布，              │
  │   防止反演结果在缺乏观测约束的区域（如双雷达覆盖盲区）    │
  │   产生不合理的风场值。                                   │
  │                                                         │
  │   权重 Cb 越大 → 越依赖初始猜测场 → 反演结果更平滑、保守 │
  │   权重 Cb 过小 → 初始猜测场几乎不起约束作用               │
  │                                                         │
  │   注意: 垂直速度 w 的背景项始终为零，因为背景场通常       │
  │   无法提供可靠的垂直速度估计（大尺度模式的 w 误差很大）。  │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │ 2.3  质量连续项 Jm (Mass continuity constraint)         │
  │                                                         │
  │   Jm = Cm · Σ [∂u/∂x + ∂v/∂y + ∂w/∂z + ε·w]²           │
  │                                                         │
  │   其中 ε·w 是密度分层修正项（anelastic近似）:            │
  │                                                         │
  │   可压缩大气的精确质量连续方程为:                         │
  │     ∂ρ/∂t + ∇·(ρV) = 0                                  │
  │                                                         │
  │   对于天气尺度以下的运动（M < 0.3），密度时间变化可忽略: │
  │     ∇·(ρ₀V) ≈ 0                                          │
  │                                                         │
  │   展开: ∂(ρ₀u)/∂x + ∂(ρ₀v)/∂y + ∂(ρ₀w)/∂z = 0         │
  │   即:  ρ₀(∂u/∂x + ∂v/∂y + ∂w/∂z) + w·(∂ρ₀/∂z) = 0     │
  │                                                         │
  │   定义 ε = (1/ρ₀)·(∂ρ₀/∂z) ≈ -1/H_ρ（密度标高倒数）     │
  │   →  ∂u/∂x + ∂v/∂y + ∂w/∂z + ε·w = 0                  │
  │                                                         │
  │   物理含义: 保证反演风场满足大气质量守恒定律。            │
  │   这对于垂直速度的求解至关重要——水平辐合必然伴随         │
  │   上升运动，水平辐散伴随下沉运动。                       │
  │                                                         │
  │   权重 Cm 越大 → 风场越严格满足质量连续方程              │
  │   权重 Cm 越小 → 可能出现质量不守恒的"物理上不可能"的风场 │
  │                                                         │
  │   在中气旋反演中，Cm 是最关键的约束项之一，               │
  │   因为它将水平旋转与垂直上升直接耦合起来。                │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │ 2.4  平滑项 Js (Smoothness / regularization terms)      │
  │                                                         │
  │   Js = Cx·Σ(∂²u/∂x²)² + Cy·Σ(∂²u/∂y²)² + Cz·Σ(∂²u/∂z²)²│
  │      + Cx·Σ(∂²v/∂x²)² + Cy·Σ(∂²v/∂y²)² + Cz·Σ(∂²v/∂z²)²│
  │      + Cx·Σ(∂²w/∂x²)² + Cy·Σ(∂²w/∂y²)² + Cz·Σ(∂²w/∂z²)²│
  │                                                         │
  │   使用二阶空间导数的平方和（二阶 Tikhonov 正则化），     │
  │   惩罚风场中剧烈的小尺度振荡。                            │
  │                                                         │
  │   物理含义:                                               │
  │     - 抑制由观测噪声和欠定问题引起的虚假小尺度波动        │
  │     - Cx, Cy, Cz 分别控制 x/y/z 方向的平滑强度            │
  │     - 值越大 → 反演风场越平滑 → 可能丢失真实的小尺度结构   │
  │     - 值越小 → 保留更多细节 → 但可能引入噪声伪结构         │
  │                                                         │
  │   双多普勒风场反演中典型的权重配置:                        │
  │     Cx=Cy=1e-4, Cz=1e-4 (水平/垂直平滑相等)              │
  └─────────────────────────────────────────────────────────┘

【3. 最优化求解方法】

  3DVAR 代价函数的最小化是一个大规模无约束优化问题。
  PyDDA 使用以下方法求解:

  3.1 问题规模
    对于典型的 20×201×201 网格:
      - 待求解变量: 3 × 20 × 201 × 201 ≈ 2.4×10⁶ 个
      - 代价函数和梯度需在每个迭代步计算

  3.2 优化算法: L-BFGS-B（有限内存拟牛顿法）

    L-BFGS-B 是拟牛顿法的一种高效实现，适用于大规模有界约束优化。

    牛顿法基本思想: 用二次模型局部近似代价函数
      J(x + Δx) ≈ J(x) + ∇J(x)ᵀ·Δx + ½·Δxᵀ·H(x)·Δx

     其中 ∇J 是梯度向量，H 是 Hessian 矩阵（二阶导数矩阵）。
     精确牛顿步: Δx = -H⁻¹·∇J

    拟牛顿法: 不显式计算 H（对百万变量而言 H 是 10¹² 元素量级），
     而是通过梯度历史逐步构建 H 的近似逆矩阵。

    L-BFGS 的 "Limited-memory" 特性:
      - 仅存储最近 m 步的梯度和位移向量（通常 m=10~20）
      - 内存: O(m·n) vs 全 BFGS 的 O(n²)
      - 适用于本项目百万级变量的场景

  3.3 收敛判据
    当以下任一条件满足时迭代终止:
      - |J_new - J_old| / J_old < wind_tol（代价函数下降量小于阈值）
      - |∇J|_∞ < gtol（梯度无穷范数小于阈值，已达到临界点）
      - 迭代次数达到 max_iterations

  3.4 梯度计算
    代价函数 J 对每个风场分量 (u, v, w) 的梯度通过以下方式计算:
      - 观测项: 解析梯度（前向算子是线性的: Vr = u·bx + v·by + w·bz）
      - 背景项: 解析梯度（二次型）
      - 质量连续项: 离散差分（中心差分近似 ∂/∂x, ∂/∂y, ∂/∂z）
      - 平滑项: 离散二阶差分

    实际计算中，梯度在整个三维网格上逐点计算，形成与待求解变量
    同维度的梯度向量，供 L-BFGS-B 使用。

【4. 预处理和后处理】

  4.1 低通滤波 (filter_window, filter_order)
    每步迭代后对风场应用 Butterworth 低通滤波器:
      - 目的: 抑制网格尺度（2Δx）噪声，保留中气旋等中尺度结构（~5-10km）
      - 默认滤波窗口 5，避免在约 1.5 km 网格上抹除中气旋尺度结构
      - 这是隐式正则化，与代价函数中的显式平滑项互补

  4.2 垂直速度后处理
    由于 w 分量仅通过质量连续方程间接约束（无直接观测），
    反演结果的 w 往往含有较大的不确定性。后处理步骤:
      - NaN 替换为 0（w 的约束不足，容易在无数据区产生 NaN）
      - 物理范围裁剪（|w| < 30 m/s）
══════════════════════════════════════════════════════════════
代价函数权重参考值（针对中气旋反演的推荐配置）
══════════════════════════════════════════════════════════════
  Co = 100.0   — 高度信任观测（雷达数据质量好）
  Cb = 0.01    — 弱背景约束（初猜场仅起引导作用）
  Cm = 1000.0  — 强质量连续约束（对 w 的可靠性至关重要）
  Cx=Cy=Cz = 1e-4 — 弱平滑（保留中气旋的尖锐涡度梯度）
"""

import sys
import unittest.mock as mock
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from src.proj_setup import configure_proj

configure_proj()
import pyart

# ---- 在导入时修补 PyDDA 的示例文件下载 ----
# PyDDA 在导入时会尝试下载示例数据文件，但此项目不需要它们。
# 通过用虚拟 Grid 对象替换 sample_files 模块来阻止下载。
_dummy_grid = None


def _make_dummy_grid():
    """创建一个虚拟 PyART Grid 对象，用于阻止 PyDDA 示例文件下载。"""
    global _dummy_grid
    if _dummy_grid is not None:
        return _dummy_grid
    _dummy_grid = pyart.core.Grid(
        time={"data": np.array([0.0])},
        fields={},
        metadata={},
        origin_latitude={"data": np.array([0.0])},
        origin_longitude={"data": np.array([0.0])},
        origin_altitude={"data": np.array([0.0])},
        x={"data": np.array([-1.0, 0.0, 1.0])},
        y={"data": np.array([-1.0, 0.0, 1.0])},
        z={"data": np.array([500.0, 1000.0])},
    )
    return _dummy_grid


from scipy.ndimage import distance_transform_edt

_patched = False
if not _patched:
    dg = _make_dummy_grid()
    sys.modules["pydda.tests.sample_files"] = mock.MagicMock()
    sys.modules["pydda.tests.sample_files"].EXAMPLE_RADAR0 = dg
    sys.modules["pydda.tests.sample_files"].EXAMPLE_RADAR1 = dg
    _patched = True


def _build_blend_weight(valid_mask, grid1, blend_sigma_km=8.0):
    """
    基于距离的融合权重场。

    在双多普勒有效区内部权重为 1.0，远离有效区的区域权重平滑衰减到 0.0。
    使用欧几里得距离变换 + 高斯衰减实现平滑过渡，避免风场拼接边界处
    出现人为的不连续（虚假散度/涡度）。

    Parameters
    ----------
    valid_mask : np.ndarray (bool), 形状 (nz, ny, nx)
        双多普勒有效点的布尔掩膜。
    grid1 : pyart.core.Grid
        分析网格（用于获取水平网格间距）。
    blend_sigma_km : float
        高斯衰减的特征宽度 (km)。控制双雷达修正量向 VWP 背景场
        过渡的平滑程度。值越大过渡越平缓。

    Returns
    -------
    weight : np.ndarray (float32), 形状 (nz, ny, nx)
        融合权重，范围 [0, 1]。
    """
    nz, ny, nx = valid_mask.shape
    dx_km = (grid1.x["data"][1] - grid1.x["data"][0]) / 1000.0
    dy_km = (grid1.y["data"][1] - grid1.y["data"][0]) / 1000.0

    # 对每个 z 层分别计算水平距离场，然后合并
    weight = np.zeros((nz, ny, nx), dtype=np.float32)

    for iz in range(nz):
        layer = valid_mask[iz].astype(np.uint8)  # (ny, nx)
        if layer.sum() == 0:
            continue

        # 欧几里得距离变换: 有效区外部每个点到最近有效点的距离 (像素)
        # sampling=(dy_km, dx_km) 使距离单位变为 km
        dist_px = distance_transform_edt(1 - layer, sampling=(dy_km, dx_km))

        # 高斯衰减: w = exp(-d² / (2·σ²))
        weight[iz] = np.exp(-0.5 * (dist_px / blend_sigma_km) ** 2)

    # 有效区内部权重严格设为 1.0
    weight[valid_mask] = 1.0

    return weight


class WindRetrieval:
    """
    基于 PyDDA 的三维变分分析双多普勒风场反演。

    接收各雷达的 PyART Radar 对象，将其网格化到统一的笛卡尔网格上，
    然后运行三维变分分析来反演完整的三维风场 (u, v, w)。

    三维变分分析的代价函数由以下各项组成:
      - 观测项 (Co): 惩罚反演风场与雷达径向速度观测之间的差异
      - 背景项 (Cb): 惩罚与初始猜测场的偏离
      - 质量连续项 (Cm): 强制执行 anelastic 质量连续方程 (∂u/∂x + ∂v/∂y + ∂w/∂z ≈ 0)
      - 平滑项 (Cx, Cy, Cz): 强制执行空间平滑性，抑制小尺度噪声

    参数
    ----------
    radar1, radar2 : pyart.core.Radar
        来自两个测站的质量控制后的雷达对象。
    vel_field : str
        雷达对象中速度场的名称。
    grid_origin : tuple (lat, lon)
        统一笛卡尔网格的原点（经纬度）。
    grid_shape : tuple (nz, ny, nx)
        网格维度。
    grid_limits : tuple
        ((z_min, z_max), (y_min, y_max), (x_min, x_max))，单位：米。
    """

    def __init__(
        self,
        radar1: "pyart.core.Radar",
        radar2: "pyart.core.Radar",
        reflectivity_radar1: "pyart.core.Radar" = None,
        reflectivity_radar2: "pyart.core.Radar" = None,
        vel_field: str = "Vc",
        grid_origin: Tuple[float, float] = None,
        grid_shape: Tuple[int, int, int] = (20, 201, 201),
        grid_limits: Tuple = (
            (500.0, 15000.0),
            (-150000.0, 150000.0),
            (-150000.0, 150000.0),
        ),
        station1: str = "Z9317",
        station2: str = "Z9543",
        data_dir: str = "data",
    ):
        self.radar1 = radar1
        self.radar2 = radar2
        self.reflectivity_radars = (reflectivity_radar1, reflectivity_radar2)
        self.vel_field = vel_field
        self.station1 = station1
        self.station2 = station2
        self.data_dir = Path(data_dir)
        self.radar_times = (
            pyart.util.datetime_from_radar(radar1),
            pyart.util.datetime_from_radar(radar2),
        )
        self.radar_time_offset_seconds = abs(
            (self.radar_times[0] - self.radar_times[1]).total_seconds()
        )

        # 统一网格原点（默认取两部雷达位置的中点，
        # 以最小化投影误差并保证对称的覆盖范围）
        if grid_origin is None:
            lat1 = radar1.latitude["data"][0]
            lon1 = radar1.longitude["data"][0]
            lat2 = radar2.latitude["data"][0]
            lon2 = radar2.longitude["data"][0]
            grid_origin = ((lat1 + lat2) / 2, (lon1 + lon2) / 2)

        self.grid_origin = grid_origin
        self.grid_shape = grid_shape
        self.grid_limits = grid_limits

        # 结果存储
        self.grid1: Optional[pyart.core.Grid] = None
        self.grid2: Optional[pyart.core.Grid] = None
        self.u: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None
        self.retrieved_grid: Optional[pyart.core.Grid] = None
        self.radar_count: Optional[np.ndarray] = None
        self.dual_valid_mask: Optional[np.ndarray] = None
        self.cross_beam_angle: Optional[np.ndarray] = None
        self.solver_invalid_mask: Optional[np.ndarray] = None
        self.background_profiles = None
        self.background_info = None
        self.radial_velocity_residuals = []
        self.radial_velocity_rmse = []
        self.continuity_residual: Optional[np.ndarray] = None
        self.quality_metrics = {}
        self.retrieval_config = {}

    # ------------------------------------------------------------------
    # 步骤1: 将每部雷达分别网格化到统一笛卡尔网格
    # ------------------------------------------------------------------

    def grid_radars(self) -> Tuple["pyart.core.Grid", "pyart.core.Grid"]:
        """
        将每部雷达分别网格化到相同的笛卡尔网格上。

        PyDDA 要求每部雷达有独立的 Grid 对象，但所有网格
        必须对齐到相同的坐标系。
        """
        print("[Grid] Gridding each radar to common Cartesian grid...")
        print(f"       Origin: ({self.grid_origin[0]:.4f}°N, "
              f"{self.grid_origin[1]:.4f}°E)")
        print(f"       Shape: {self.grid_shape}")

        grids = []
        for i, radar in enumerate([self.radar1, self.radar2]):
            stn = radar.metadata.get("vcp", f"radar{i+1}")
            print(f"       Radar {i+1} ({radar.nrays} rays)...")

            grd = pyart.map.grid_from_radars(
                (radar,),
                grid_shape=self.grid_shape,
                grid_limits=self.grid_limits,
                fields=[self.vel_field],
                grid_origin=self.grid_origin,
                gridding_algo="map_gates_to_grid",
                weighting_function="Cressman",
                roi_func="dist_beam",
                constant_roi=2000.0,  # 2km ROI，扩大双雷达覆盖
                copy_field_data=True,
            )
            grids.append(grd)
            n_valid = (~grd.fields[self.vel_field]["data"].mask).sum()
            print(f"         Valid grid points: {n_valid}")

            refl_radar = self.reflectivity_radars[i]
            if refl_radar is not None:
                refl_source_field = list(refl_radar.fields.keys())[0]
                refl_grid = pyart.map.grid_from_radars(
                    (refl_radar,),
                    grid_shape=self.grid_shape,
                    grid_limits=self.grid_limits,
                    fields=[refl_source_field],
                    grid_origin=self.grid_origin,
                    gridding_algo="map_gates_to_grid",
                    weighting_function="Cressman",
                    roi_func="dist_beam",
                    constant_roi=2000.0,
                    copy_field_data=True,
                )
                refl_data = refl_grid.fields[refl_source_field]["data"]
                grd.add_field(
                    "reflectivity",
                    {
                        "data": refl_data,
                        "units": "dBZ",
                        "standard_name": "equivalent_reflectivity_factor",
                        "long_name": "Observed radar reflectivity",
                    },
                    replace_existing=True,
                )
                n_refl = np.ma.count(refl_data)
                print(f"         Reflectivity valid grid points: {n_refl}")

        self.grid1, self.grid2 = grids
        mask1 = np.ma.getmaskarray(grids[0].fields[self.vel_field]["data"])
        mask2 = np.ma.getmaskarray(grids[1].fields[self.vel_field]["data"])
        self.radar_count = (~mask1).astype(np.uint8) + (~mask2).astype(np.uint8)
        return grids

    # ------------------------------------------------------------------
    # 步骤2: 构建初始风场（VWP 风廓线 + 双多普勒直接合成融合）
    # ------------------------------------------------------------------

    def make_initial_wind(
        self,
        min_cross_beam_angle_deg: float = 20.0,
        w_top: float = 0.0,
        max_iterations: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构建三维变分分析的初始猜测场。

        采用 VWP 风廓线 + 双多普勒直接合成融合策略:
          1. 加载 VWP (VAD Wind Profile) 作为大尺度背景风廓线（覆盖 100% 网格点）
          2. 运行双多普勒直接合成，在波束交叠区获得高分辨率风场
          3. 融合：双雷达覆盖区内使用双雷达修正（保留中尺度细节），
             其余区域保持 VWP 背景风场
          4. 在双雷达区域边界进行高斯平滑过渡，避免风场不连续

        如果 VWP 不可用，则回退为纯双雷达初猜场。

        背景风廓线 (VWP) 说明:
          VAD (Velocity Azimuth Display) 技术通过拟合单雷达全方位
          径向速度的正弦曲线来反演水平风廓线。相比 ERA5 再分析数据，
          VWP 的优势在于:
            - 零时间差：与雷达体扫同步，代表真实大气状态
            - 无需额外数据下载
            - 对中尺度特征（如低空急流）的刻画更精细

        Parameters
        ----------
        min_cross_beam_angle_deg : float
            双雷达波束间保证良态求解的最小夹角（度）。
        w_top : float
            区域顶部的假定垂直速度 (m/s)，作为质量连续积分的上边界条件。
        max_iterations : int
            垂直速度反馈迭代次数（0 = 单次求解，已足够作为初猜场）。

        Returns
        -------
        u_init, v_init, w_init : np.ndarray, 形状 (nz, ny, nx)
        """
        from src.dual_radar_retrieval import dual_radar_retrieve

        if self.grid1 is None or self.grid2 is None:
            self.grid_radars()

        nz, ny, nx = self.grid_shape

        # ---- 第1步: 加载 VWP 背景风廓线 ----
        # 只使用参与反演的两部雷达，按时间最近原则选择 VWP。
        # 之前的实现遍历所有注册站点且不传时间戳，可能选到
        # 不同站点、不同时次的错误风廓线作为背景场。
        u_bg = np.zeros((nz, ny, nx), dtype=np.float32)
        v_bg = np.zeros((nz, ny, nx), dtype=np.float32)
        w_bg = np.zeros((nz, ny, nx), dtype=np.float32)
        has_bg = False

        try:
            from src.cinrad_products import load_vwp, vwp_to_background
            # 从雷达对象中提取扫描时间，传递给 VWP 做时间匹配
            _t1 = pyart.util.datetime_from_radar(self.radar1)
            ts_vwp = _t1.strftime("%Y%m%d%H%M%S")

            # 只查询参与反演的两部雷达
            best_vwp, best_stn, best_n = None, None, 0
            for stn in [self.station1, self.station2]:
                vwp = load_vwp(
                    stn,
                    timestamp=ts_vwp + "Z",
                    data_dir=str(self.data_dir),
                    tol_seconds=600,
                )
                if vwp is None:
                    continue
                n = vwp["valid"].sum()
                if n > best_n:
                    best_vwp, best_stn, best_n = vwp, stn, n

            if best_vwp and best_n >= 3:
                z_m = np.asarray(self.grid1.z["data"])
                u_bg, v_bg = vwp_to_background(best_vwp, z_m, nz, ny, nx)
                self.background_profiles = (
                    u_bg[:, 0, 0].copy(),
                    v_bg[:, 0, 0].copy(),
                    z_m.copy(),
                )
                self.background_info = {
                    "background_station": best_stn,
                    "background_nominal_time": best_vwp.get("time", ""),
                    "background_time_offset_seconds": best_vwp.get(
                        "time_offset_seconds", np.nan
                    ),
                }
                has_bg = True
                print(f"\n[Init] VWP 背景风廓线 ({best_stn}, "
                      f"时间偏移={best_vwp.get('time','?')}): "
                      f"{best_n} 有效层, "
                      f"u=[{u_bg.min():.1f},{u_bg.max():.1f}], "
                      f"v=[{v_bg.min():.1f},{v_bg.max():.1f}] m/s")
            else:
                print(f"[Init] VWP 不可用 (两站有效层均<3)，"
                      f"回退为纯双雷达模式")
        except Exception as e:
            print(f"[Init] VWP 加载失败 ({e})，回退为纯双雷达模式")

        # ---- 第2步: 双多普勒直接合成 ----
        print(f"\n[Init] 运行双多普勒直接合成...")
        u_dr, v_dr, w_dr, valid_mask, cross_beam_angle = dual_radar_retrieve(
            self.grid1, self.grid2,
            self.radar1, self.radar2,
            vel_field=self.vel_field,
            min_cross_beam_angle_deg=min_cross_beam_angle_deg,
            w_top=w_top,
            max_iterations=max_iterations,
        )
        self.dual_valid_mask = valid_mask
        self.cross_beam_angle = cross_beam_angle

        # 限制双雷达 w 在物理合理范围（防散度积分发散）
        w_dr = np.clip(w_dr, -30.0, 30.0)

        n_dr = valid_mask.sum()
        print(f"[Init] 双雷达覆盖: {n_dr}/{nz*ny*nx} ({100*n_dr/(nz*ny*nx):.1f}%) 网格点")

        # ---- 第3步: 融合背景场 + 双雷达 ----
        # 融合策略: 增量分析 (Incremental Analysis Update)
        #
        # 基本思想: 在双雷达波束交叉区，双多普勒直接合成的风场精度高、
        # 分辨率高；在此区域之外，VWP 风廓线提供了可靠的背景估计。
        # 融合公式:
        #   V_final(x) = V_bg(x) + W(x) · [V_dr(x) - V_bg(x)]
        #
        # 其中 W(x) 是空间平滑的融合权重场:
        #   - 双雷达有效区内: W = 1.0（完全信任双雷达结果）
        #   - 远离有效区: W → 0.0（平滑过渡到纯背景场）
        #   - 过渡带: 高斯衰减 exp(-d²/2σ²), σ = blend_sigma_km
        #
        # 这种增量方法的优势:
        #   1. 保留双雷达区的高分辨率中尺度结构（如中气旋旋转）
        #   2. 双雷达盲区有物理上合理的背景值（而非零）
        #   3. 边界平滑过渡，不引入虚假的散度和涡度
        if has_bg:
            u_init = u_bg.copy()
            v_init = v_bg.copy()
            w_init = w_bg.copy()

            if n_dr > 0:
                # 对双雷达结果进行空间平滑，消除孤立极值
                # 高斯滤波核 σ=2.0 网格点 ≈ 3km，保留中气旋尺度(5-10km)结构
                from scipy.ndimage import gaussian_filter
                # 用有效样本权重归一化，避免零填充在观测边界把风速压低。
                smooth_weight = gaussian_filter(
                    valid_mask.astype(np.float32), sigma=2.0,
                    mode="constant", cval=0.0,
                )
                smooth_weight = np.maximum(smooth_weight, 1e-6)

                def _valid_smooth(values):
                    numerator = gaussian_filter(
                        np.where(valid_mask, values, 0.0), sigma=2.0,
                        mode="constant", cval=0.0,
                    )
                    return numerator / smooth_weight

                u_dr_smooth = _valid_smooth(u_dr)
                v_dr_smooth = _valid_smooth(v_dr)
                w_dr_smooth = _valid_smooth(w_dr)
                u_dr = np.where(valid_mask, u_dr_smooth, u_dr)
                v_dr = np.where(valid_mask, v_dr_smooth, v_dr)
                w_dr = np.where(valid_mask, w_dr_smooth, w_dr)

                # 计算增量: ΔV = V_dr - V_bg
                # 关键修复: 只在双雷达有效区内计算增量，无效区设为 0。
                # 之前的实现在无效区有 u_dr=0，导致 u_delta = -u_bg，
                # 然后在过渡带 u_init = u_bg + (-u_bg)·weight = u_bg·(1-weight)
                # → 将背景风场人为压向零，产生虚假散度/涡度。
                u_delta = np.where(valid_mask, u_dr - u_bg, 0.0)
                v_delta = np.where(valid_mask, v_dr - v_bg, 0.0)
                w_delta = np.where(valid_mask, w_dr - w_bg, 0.0)

                # 构建平滑权重场: 在双雷达区边界让修正量平滑过渡到 0
                # _build_blend_weight 使用欧几里得距离变换 + 高斯衰减
                weight = _build_blend_weight(valid_mask, self.grid1)

                # 增量融合: V_final = V_bg + W · ΔV
                u_init = u_init + u_delta * weight
                v_init = v_init + v_delta * weight
                w_init = w_init + w_delta * weight

                print(f"[Init] VWP背景 + 双雷达融合完成")
                print(f"       u_delta RMS: {np.sqrt(np.mean(u_delta[valid_mask]**2)):.1f} m/s")
                print(f"       v_delta RMS: {np.sqrt(np.mean(v_delta[valid_mask]**2)):.1f} m/s")
        else:
            # 纯双雷达模式（覆盖区外为 0）
            u_init, v_init, w_init = u_dr, v_dr, w_dr

        print(f"[Init] 初猜场: u=[{u_init.min():.1f},{u_init.max():.1f}], "
              f"v=[{v_init.min():.1f},{v_init.max():.1f}], "
              f"w=[{w_init.min():.2f},{w_init.max():.2f}] m/s")

        return u_init, v_init, w_init

    # ------------------------------------------------------------------
    # 步骤3: 运行三维变分分析
    # ------------------------------------------------------------------

    def retrieve(
        self,
        u_init: Optional[np.ndarray] = None,
        v_init: Optional[np.ndarray] = None,
        w_init: Optional[np.ndarray] = None,
        Co: float = 1.0,
        Cb: float = 1.0,
        Cm: float = 256.0,
        Cx: float = 1.0,
        Cy: float = 1.0,
        Cz: float = 1.0,
        wind_tol: float = 0.5,
        max_iterations: int = 50,
        engine: str = "scipy",
        filter_window: int = 5,
        filter_order: int = 3,
        min_bca: float = 20.0,
        **kwargs,
    ) -> "pyart.core.Grid":
        """运行 PyDDA 三维变分分析风场反演。

        调用 PyDDA 的 get_dd_wind_field()，底层使用 L-BFGS-B 拟牛顿法
        最小化代价函数 J(u,v,w) = Jo + Jb + Jm + Js。

        ┌──────────────────────────────────────────────────────┐
        │ 计算流程明细                                         │
        ├──────────────────────────────────────────────────────┤
        │                                                      │
        │ 第1步 — PyART Grid → xarray Dataset 格式转换         │
        │   PyDDA 内部使用 xarray 作为标准数据容器。            │
        │   read_from_pyart_grid() 将 PyART Grid 转换为         │
        │   带维度标签的 xarray Dataset，便于索引和梯度计算。   │
        │                                                      │
        │ 第2步 — 构建代价函数                                  │
        │   PyDDA 根据传入的权重参数在内存中构建完整的          │
        │   代价函数表达式及其梯度计算图。各约束项的贡献:        │
        │                                                      │
        │   Jo = Co/2 · Σ (Vr_ret_i - Vr_obs_i)²              │
        │      遍历两部雷达的所有有效观测网格点                │
        │                                                      │
        │   Jb = Cb/2 · Σ [(u-u_bg)²+(v-v_bg)²+w²]            │
        │      覆盖全部网格点(无观测区域由初猜场填充)          │
        │                                                      │
        │   Jm = Cm/2 · Σ [∂u/∂x+∂v/∂y+∂w/∂z+εw]²            │
        │      使用二阶中心差分近似散度                        │
        │                                                      │
        │   Js = Cx/2·Σ(∂²u/∂x²)² + ... + Cz/2·Σ(∂²w/∂z²)²  │
        │      使用二阶中心差分近似二阶导数（拉普拉斯平滑）    │
        │                                                      │
        │ 第3步 — L-BFGS-B 迭代优化                             │
        │   每一步迭代:                                         │
        │     a) 计算当前风场的代价函数值 J_k                   │
        │     b) 计算梯度 ∇J(u_k, v_k, w_k)——对百万级变量       │
        │        的梯度向量                                     │
        │     c) 用有限内存 BFGS 公式构建 Hessian 近似 H_k     │
        │     d) 求解搜索方向 p_k = -H_k⁻¹·∇J_k               │
        │     e) 线搜索确定步长 α_k（满足 Wolfe 条件）         │
        │     f) 更新风场: (u,v,w)_{k+1} = (u,v,w)_k + α_k·p_k│
        │     g) Savitzky-Golay 低通滤波（抑制网格尺度噪声）  │
        │     h) 检查收敛: |J_{k+1}-J_k|/J_k < wind_tol?      │
        │                                                      │
        │ 第4步 — 提取反演结果                                  │
        │   从 PyDDA 返回的 xarray Dataset 中提取 u, v, w      │
        │   风场分量，转换为 numpy 数组，NaN 替换为 0。         │
        │                                                      │
        └──────────────────────────────────────────────────────┘

        参数
        ----------
        u_init, v_init, w_init : np.ndarray, 形状 (nz, ny, nx)
            初始猜测风场（若为 None 则自动调用 make_initial_wind()）。
        Co : float
            观测项权重。越大表示越信任雷达径向速度数据。
            推荐: 1.0（默认）, 对高信噪比数据可用 10~100。
        Cb : float
            背景场约束权重。控制反演结果对初猜场的依赖程度。
            推荐: 0.01~1.0，中气旋分析建议较小值以充分引入双多普勒修正。
        Cm : float
            质量连续约束权重。强制执行 anelastic ∇·(ρ₀V) ≈ 0。
            推荐: 256~1000，强约束对垂直速度的准确性至关重要。
        Cx, Cy, Cz : float
            x/y/z 方向的二阶平滑权重（Tikhonov 正则化参数）。
            推荐: 1e-4 ~ 1.0，保留中气旋细节用较小值。
        wind_tol : float
            收敛容差 (m/s)。代价函数相对下降量 < wind_tol 时停止迭代。
            推荐: 0.1~0.5 m/s。
        max_iterations : int
            最大 L-BFGS-B 迭代次数。推荐: 50~150。
            对于初期测试可用 20~50，业务级反演建议 ≥100。
        engine : str
            优化引擎，仅支持 "scipy"（基于 scipy.optimize.fmin_l_bfgs_b）。
            Jax/TensorFlow 引擎在当前环境中不可用。
        filter_window : int
            Savitzky-Golay 滤波窗口。推荐 5；窗口过大会显著压低局地涡度。
        filter_order : int
            Savitzky-Golay 多项式阶数。推荐: 3。
        min_bca : float
            PyDDA 观测权重允许的最小双雷达交叉波束角（度）。必须与
            初猜场和输出质量掩膜使用同一阈值。

        Returns
        -------
        pyart.core.Grid
            包含 u, v, w, wind_speed, divergence, vorticity 六个物理量
            的完整反演结果。
        """
        from pydda.retrieval import get_dd_wind_field
        from pydda.io.read_grid import read_from_pyart_grid

        if self.grid1 is None:
            self.grid_radars()

        # ---- 第1步: 保存网格几何信息（深拷贝，防 PyDDA 原地修改） ----
        # PyDDA 内部会对 Grid 对象的字典结构进行原地修改（如添加/重命名字段）。
        # 为在反演后仍能正确构建输出 Grid，需要提前保存几何信息。
        grd = self.grid1
        self._time_dict = {"data": np.array(grd.time["data"]), "units": str(grd.time.get("units", ""))}
        self._x_dict = {"data": np.array(grd.x["data"]), "units": "meters"}
        self._y_dict = {"data": np.array(grd.y["data"]), "units": "meters"}
        self._z_dict = {"data": np.array(grd.z["data"]), "units": "meters"}
        self._olat = {"data": np.array(grd.origin_latitude["data"]), "units": "degrees_north"}
        self._olon = {"data": np.array(grd.origin_longitude["data"]), "units": "degrees_east"}
        self._oalt = {"data": np.array(grd.origin_altitude["data"]), "units": "meters"}

        if u_init is None:
            u_init, v_init, w_init = self.make_initial_wind()

        print(f"\n[3DVAR] 启动 PyDDA 三维变分分析...")
        print(f"        Co={Co}, Cb={Cb}, Cm={Cm}")
        print(f"        Cx={Cx}, Cy={Cy}, Cz={Cz}")
        print(f"        tol={wind_tol} m/s, max_iter={max_iterations}")

        # ---- 第2步: 反射率处理 ----
        # PyDDA 使用实测反射率估算水凝物终端下落速度。缺少反射率时
        # 不能用常数场静默替代，否则会把系统偏差带入垂直速度反演。
        for g in [self.grid1, self.grid2]:
            if "reflectivity" not in g.fields:
                raise RuntimeError(
                    "Observed reflectivity is required for terminal fall-speed "
                    "correction; provide matching 019/020 PPI volumes"
                )
        refl_field = "reflectivity"

        # ---- 第3步: PyART Grid → xarray Dataset 格式转换 ----
        # PyDDA 使用 xarray 作为其内部数据容器，需要将 PyART Grid 对象转换。
        # read_from_pyart_grid 将 Grid 的 fields/metadata/coordinates 映射到
        # 带有维度标签 (x, y, z) 的 xarray Dataset，其中:
        #   - Data variables: 各场 (u, v, w, Vc 等)
        #   - Coordinates: x, y, z, time
        #   - Attributes: 网格原点的经纬度等元数据
        new_g1 = read_from_pyart_grid(self.grid1)
        new_g2 = read_from_pyart_grid(self.grid2)

        nz, ny, nx = self.grid_shape
        n_valid_1 = (~self.grid1.fields[self.vel_field]["data"].mask).sum()
        n_valid_2 = (~self.grid2.fields[self.vel_field]["data"].mask).sum()
        print(f"        Grid: {nz}x{ny}x{nx}")
        print(f"        Radar1 valid pts: {n_valid_1}, Radar2 valid pts: {n_valid_2}")

        # ---- 第4步: 调用 PyDDA 核心求解器 ----
        # get_dd_wind_field 内部执行完整的 3DVAR 求解流程:
        #   1. 构建代价函数 J = Jo + Jb + Jm + Js 的符号表达式
        #   2. 使用 scipy.optimize.fmin_l_bfgs_b 进行迭代优化
        #   3. 每个迭代步: 计算 J → 计算 ∇J → BFGS更新 → 线搜索 → 滤波
        #
        # L-BFGS-B 的关键参数（由 PyDDA 内部设置）:
        #   factr = 1e7 → 控制收敛精度（与 wind_tol 相关）
        #   pgtol = 1e-5 → 梯度容差
        #   maxcor = 10 → L-BFGS 中存储的向量对数量（内存 vs 收敛速度的平衡）
        background_kwargs = {}
        if self.background_profiles is not None:
            u_back, v_back, z_back = self.background_profiles
            background_kwargs = {
                "u_back": u_back,
                "v_back": v_back,
                "z_back": z_back,
            }

        self.retrieval_config = {
            "Co": Co, "Cb": Cb, "Cm": Cm,
            "Cx": Cx, "Cy": Cy, "Cz": Cz,
            "wind_tol": wind_tol,
            "max_iterations": max_iterations,
            "filter_window": filter_window,
            "filter_order": filter_order,
            "min_bca": min_bca,
            "max_bca": 180.0 - min_bca,
            "engine": engine,
        }

        result = get_dd_wind_field(
            [new_g1, new_g2],
            u_init=u_init,      # 初猜 u 场
            v_init=v_init,      # 初猜 v 场
            w_init=w_init,      # 初猜 w 场
            engine=engine,       # "scipy" = L-BFGS-B
            Co=Co,              # 观测项权重
            Cb=Cb,              # 背景项权重
            Cm=Cm,              # 质量连续项权重
            Cx=Cx, Cy=Cy, Cz=Cz,  # 平滑项权重
            wind_tol=wind_tol,   # 收敛容差
            max_iterations=max_iterations,  # 最大迭代次数
            vel_name=self.vel_field,         # 速度场在 Dataset 中的名称
            refl_field=refl_field,           # 反射率场 (None=禁用下落速度修正)
            filter_window=filter_window,     # 低通滤波窗口
            filter_order=filter_order,       # 低通滤波阶数
            min_bca=min_bca,                 # PyDDA 观测权重的最小交叉波束角
            max_bca=180.0 - min_bca,         # 与急性交叉角阈值保持对称
            **background_kwargs,
            **kwargs,
        )

        # ---- 第5步: 从 PyDDA 输出中提取风场 ----
        # PyDDA 返回 (new_grid_list, parameters) 元组:
        #   new_grid_list[0]: 第一部雷达对应的 xarray Dataset，包含反演的 u, v, w
        #   parameters: 优化过程诊断信息（迭代次数、最终代价等）
        new_grid_list, parameters = result
        retrieved_ds = new_grid_list[0]

        # 提取三个风场分量并去除多余的维度
        self.u = np.array(retrieved_ds["u"].values.squeeze())
        self.v = np.array(retrieved_ds["v"].values.squeeze())
        self.w = np.array(retrieved_ds["w"].values.squeeze())
        self.solver_invalid_mask = ~(
            np.isfinite(self.u) & np.isfinite(self.v) & np.isfinite(self.w)
        )
        # 保留无效位置的掩膜用于输出；填充值只用于后续差分计算。
        self.u = np.nan_to_num(self.u, nan=0.0)
        self.v = np.nan_to_num(self.v, nan=0.0)
        self.w = np.nan_to_num(self.w, nan=0.0)

        # ---- 第6步: 独立计算可审计的拟合与质量连续诊断 ----
        self.radial_velocity_residuals = []
        self.radial_velocity_rmse = []
        for i in range(len(parameters.vrs)):
            vr_obs = np.asarray(parameters.vrs[i], dtype=float)
            az = np.asarray(parameters.azs[i], dtype=float)
            el = np.asarray(parameters.els[i], dtype=float)
            fall = np.asarray(parameters.wts[i], dtype=float)
            weight = np.asarray(parameters.weights[i], dtype=float)
            vr_model = (
                np.cos(el) * np.sin(az) * self.u
                + np.cos(el) * np.cos(az) * self.v
                + np.sin(el) * (self.w - np.abs(fall))
            )
            valid = (
                (weight > 0) & np.isfinite(vr_obs) & np.isfinite(vr_model)
                & (vr_obs > -9000) & (~self.solver_invalid_mask)
            )
            residual = np.ma.array(vr_obs - vr_model, mask=~valid)
            self.radial_velocity_residuals.append(residual)
            rmse = float(np.sqrt(np.mean(np.square(residual.compressed())))) \
                if residual.count() else float("nan")
            self.radial_velocity_rmse.append(rmse)

        dx = float(self._x_dict["data"][1] - self._x_dict["data"][0])
        dy = float(self._y_dict["data"][1] - self._y_dict["data"][0])
        dz = float(self._z_dict["data"][1] - self._z_dict["data"][0])
        self.continuity_residual = (
            np.gradient(self.u, dx, axis=2)
            + np.gradient(self.v, dy, axis=1)
            + np.gradient(self.w, dz, axis=0)
            - self.w / 10000.0
        ).astype(np.float32)

        quality_valid = (~self.solver_invalid_mask)
        if self.dual_valid_mask is not None:
            quality_valid &= self.dual_valid_mask
        w_values = np.abs(self.w[quality_valid])
        continuity_values = np.abs(self.continuity_residual[quality_valid])
        w_abs_p99 = float(np.percentile(w_values, 99)) \
            if w_values.size else float("nan")
        continuity_abs_p95 = float(np.percentile(continuity_values, 95)) \
            if continuity_values.size else float("nan")
        quality_pass = bool(
            np.isfinite(w_abs_p99)
            and np.isfinite(continuity_abs_p95)
            and w_abs_p99 <= 15.0
            and continuity_abs_p95 <= 0.005
        )
        self.quality_metrics = {
            "quality_gate_pass": int(quality_pass),
            "vertical_velocity_abs_p99": w_abs_p99,
            "continuity_residual_abs_p95": continuity_abs_p95,
            "quality_gate_limits": "w_abs_p99<=15m/s; continuity_abs_p95<=0.005s-1",
        }

        print("        Radial-velocity RMSE: " + ", ".join(
            f"radar{i + 1}={value:.2f} m/s"
            for i, value in enumerate(self.radial_velocity_rmse)
        ))

        self.retrieved_grid = self._build_output_grid()

        print(f"        Completed.")
        print(f"        u: [{self.u.min():.1f}, {self.u.max():.1f}] m/s")
        print(f"        v: [{self.v.min():.1f}, {self.v.max():.1f}] m/s")
        print(f"        w: [{self.w.min():.2f}, {self.w.max():.2f}] m/s")

        return self.retrieved_grid

    def _build_output_grid(self) -> "pyart.core.Grid":
        """
        构建包含 u, v, w 风场分量的 PyART Grid 对象。

        同时计算并附加以下诊断物理量:
          - wind_speed: 水平风速 (sqrt(u^2 + v^2))
          - divergence: 水平散度 (du/dx + dv/dy)，用于识别辐合/辐散区
          - vorticity: 相对涡度 (dv/dx - du/dy)，用于诊断中气旋等旋转系统

        散度和涡度使用中心差分计算，以便进行中尺度天气系统的诊断分析。
        """
        # 使用预先保存的几何信息（PyDDA 会修改原始网格字典）
        time_dict = self._time_dict
        x_dict = self._x_dict
        y_dict = self._y_dict
        z_dict = self._z_dict
        olat = self._olat
        olon = self._olon
        oalt = self._oalt

        fields = {}
        radar_count = self.radar_count
        if radar_count is None:
            radar_count = np.zeros_like(self.u, dtype=np.uint8)
        dual_mask = self.dual_valid_mask
        if dual_mask is None:
            dual_mask = radar_count == 2
        solver_invalid = self.solver_invalid_mask
        if solver_invalid is None:
            solver_invalid = np.zeros_like(self.u, dtype=bool)
        wind_mask = (radar_count == 0) | solver_invalid
        diagnostic_mask = (~dual_mask) | solver_invalid
        for name, data, units, long_name in [
            ("u", self.u, "m/s", "U-wind (eastward)"),
            ("v", self.v, "m/s", "V-wind (northward)"),
            ("w", self.w, "m/s", "W-wind (upward)"),
        ]:
            fields[name] = {
                "data": np.ma.array(data.astype(np.float32), mask=wind_mask),
                "units": units,
                "standard_name": f"{name}_wind",
                "long_name": long_name,
                "_FillValue": -9999.0,
            }

        # 附加诊断量：水平风速、散度、涡度
        dx = x_dict["data"][1] - x_dict["data"][0]
        dy = y_dict["data"][1] - y_dict["data"][0]

        speed = np.sqrt(self.u**2 + self.v**2).astype(np.float32)
        fields["wind_speed"] = {"data": np.ma.array(speed, mask=wind_mask), "units": "m/s", "_FillValue": -9999.0,
                                 "standard_name": "wind_speed", "long_name": "Horizontal Wind Speed"}

        # 水平散度: ∇_h·V_h = ∂u/∂x + ∂v/∂y
        # 使用 numpy.gradient 进行二阶中心差分
        # 物理含义:
        #   负值 = 辐合 (convergence) → 气流汇聚 → 强迫上升 → 利于对流发展
        #   正值 = 辐散 (divergence) → 气流发散 → 伴随下沉运动
        #   中气旋低层典型特征: 强辐合 (−10⁻³ ~ −10⁻² s⁻¹)
        dudx = np.gradient(self.u, axis=2) / dx
        dvdy = np.gradient(self.v, axis=1) / dy
        divergence = (dudx + dvdy).astype(np.float32)
        fields["divergence"] = {"data": np.ma.array(divergence, mask=diagnostic_mask), "units": "s-1", "_FillValue": -9999.0,
                                 "standard_name": "divergence", "long_name": "Horizontal Divergence"}

        # 相对涡度: ζ = ∂v/∂x - ∂u/∂y（垂直分量）
        # 使用 numpy.gradient 进行二阶中心差分
        # 物理含义:
        #   正值 (cyclonic) = 气旋式旋转（北半球逆时针）→ 中气旋的核心特征
        #   负值 (anticyclonic) = 反气旋式旋转（北半球顺时针）
        #   强中气旋典型值: |ζ| > 0.005 s⁻¹（即 5×10⁻³ s⁻¹）
        # 注意: 这里是相对涡度（relative vorticity），
        #   未包含地球自转行星涡度 f ≈ 10⁻⁴ s⁻¹（中纬度科氏参数）
        dvdx = np.gradient(self.v, axis=2) / dx
        dudy = np.gradient(self.u, axis=1) / dy
        vorticity = (dvdx - dudy).astype(np.float32)
        fields["vorticity"] = {"data": np.ma.array(vorticity, mask=diagnostic_mask), "units": "s-1", "_FillValue": -9999.0,
                                "standard_name": "relative_vorticity", "long_name": "Relative Vorticity"}

        fields["radar_count"] = {
            "data": radar_count.astype(np.uint8),
            "units": "1",
            "long_name": "Number of radars contributing radial velocity",
        }
        fields["dual_doppler_mask"] = {
            "data": dual_mask.astype(np.uint8),
            "units": "1",
            "long_name": "Valid dual-Doppler geometry and observations",
        }
        if self.cross_beam_angle is not None:
            fields["cross_beam_angle"] = {
                "data": np.ma.array(
                    self.cross_beam_angle.astype(np.float32),
                    mask=radar_count < 2,
                ),
                "units": "degree",
                "long_name": "Acute horizontal radar crossing angle",
                "_FillValue": -9999.0,
            }
        for i, residual in enumerate(self.radial_velocity_residuals):
            fields[f"radial_velocity_residual_radar{i + 1}"] = {
                "data": residual.astype(np.float32),
                "units": "m/s",
                "long_name": f"Observed minus retrieved radial velocity for radar {i + 1}",
                "_FillValue": -9999.0,
            }
        if self.continuity_residual is not None:
            fields["continuity_residual"] = {
                "data": np.ma.array(
                    self.continuity_residual,
                    mask=diagnostic_mask,
                ),
                "units": "s-1",
                "long_name": "Anelastic mass-continuity residual",
                "_FillValue": -9999.0,
            }
        if all("reflectivity" in g.fields for g in (self.grid1, self.grid2)):
            refl1 = np.ma.filled(
                self.grid1.fields["reflectivity"]["data"], np.nan
            )
            refl2 = np.ma.filled(
                self.grid2.fields["reflectivity"]["data"], np.nan
            )
            with np.errstate(all="ignore"):
                reflectivity = np.nanmax(np.stack([refl1, refl2]), axis=0)
            fields["reflectivity"] = {
                "data": np.ma.masked_invalid(reflectivity.astype(np.float32)),
                "units": "dBZ",
                "standard_name": "equivalent_reflectivity_factor",
                "long_name": "Maximum gridded reflectivity from both radars",
                "_FillValue": -9999.0,
            }

        # 使用保存的几何信息构建输出网格
        out_grid = pyart.core.Grid(
            time=time_dict,
            fields=fields,
            metadata={
                "retrieval_method": "PyDDA_3DVAR",
                "station1": self.station1,
                "station2": self.station2,
                "radar1_nominal_time": self.radar_times[0].isoformat(),
                "radar2_nominal_time": self.radar_times[1].isoformat(),
                "radar_time_offset_seconds": self.radar_time_offset_seconds,
                "solver_status": "completed; PyDDA scipy engine exposes no convergence flag",
                "radial_velocity_rmse_radar1": (
                    self.radial_velocity_rmse[0]
                    if len(self.radial_velocity_rmse) > 0 else np.nan
                ),
                "radial_velocity_rmse_radar2": (
                    self.radial_velocity_rmse[1]
                    if len(self.radial_velocity_rmse) > 1 else np.nan
                ),
                **(self.background_info or {}),
                **self.quality_metrics,
                **self.retrieval_config,
            },
            origin_latitude=olat,
            origin_longitude=olon,
            origin_altitude=oalt,
            x=x_dict,
            y=y_dict,
            z=z_dict,
        )

        return out_grid

    # ------------------------------------------------------------------
    # 保存结果
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        """
        将反演得到的风场保存为 NetCDF 文件。

        如果 PyART 写入失败（通常由于 Grid 对象被 PyDDA 修改后结构不完整），
        则回退为将原始 u, v, w 数组保存为 .npz 格式。
        """
        import traceback as _tb
        if self.retrieved_grid is None:
            raise RuntimeError("No retrieval results. Call retrieve() first.")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        try:
            pyart.io.write_grid(filepath, self.retrieved_grid)
            print(f"[Save] Wind field saved to: {filepath}")
        except Exception:
            print(f"[Save] ERROR saving grid, attempting field-by-field save...")
            # 回退方案：将 u, v, w 保存为原始 numpy 数组
            np.savez(filepath.replace('.nc', '.npz'),
                     u=self.u, v=self.v, w=self.w)
            print(f"[Save] Raw wind fields saved to: {filepath.replace('.nc', '.npz')}")
            _tb.print_exc()

    # ------------------------------------------------------------------
    # 快速可视化
    # ------------------------------------------------------------------

    def plot(self, level: int = None, save_path: Optional[str] = None):
        """
        在指定垂直高度层快速绘制反演风场。

        自动选择涡度最强的层（优先在下半区域内搜索，因为中气旋等
        强旋转特征通常出现在对流层中低层），然后绘制：
          (a) 水平风速填色图 + 风矢量箭头
          (b) 相对涡度填色图

        参数
        ----------
        level : int
            垂直层索引（默认：自动选择涡度最强的层）。
        save_path : str, 可选
            若提供，将图像保存至该路径。
        """
        import matplotlib.pyplot as plt

        if self.retrieved_grid is None:
            raise RuntimeError("No retrieval results. Call retrieve() first.")

        # 选择风速/涡度信号最强的 z 层
        vort_full = self.retrieved_grid.fields["vorticity"]["data"]
        speed_full = self.retrieved_grid.fields["wind_speed"]["data"]

        if level is None:
            # 在下半区域内（中气旋通常所在的层次）搜索涡度绝对值最大的层
            nz = self.grid_shape[0]
            search_range = slice(0, max(1, nz * 2 // 3))
            level = int(np.argmax(np.nanmax(np.abs(np.array(vort_full[search_range])),
                                             axis=(1, 2))))
            level = max(0, min(level, nz - 1))
            print(f"[Plot] Auto-selected z-level {level} (max |vorticity|)")

        z_m = self.retrieved_grid.z["data"][level]
        x = self.retrieved_grid.x["data"] / 1000.0  # km
        y = self.retrieved_grid.y["data"] / 1000.0  # km

        speed = np.array(speed_full[level])
        u_lvl = np.array(self.u[level])
        v_lvl = np.array(self.v[level])

        # 对风矢量箭头进行下采样，使图更清晰
        step = max(1, self.grid_shape[1] // 30)
        sl = slice(step // 2, None, step)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 子图1：水平风速 + 风场矢量
        im = axes[0].pcolormesh(x, y, speed, cmap="viridis", shading="auto")
        q = axes[0].quiver(x[sl], y[sl], u_lvl[sl, sl], v_lvl[sl, sl],
                           color="white", scale=150, width=0.003)
        axes[0].set_title(f"Wind Speed + Vectors at z={z_m:.0f} m")
        axes[0].set_xlabel("East-West [km]")
        axes[0].set_ylabel("North-South [km]")
        axes[0].set_aspect("equal")
        plt.colorbar(im, ax=axes[0], label="m/s")

        # 子图2: 相对涡度（使用对称红蓝色阶，正值为气旋式旋转）
        vort = np.array(vort_full[level]) * 1e3  # 转换为 10⁻³ s⁻¹ 便于显示
        vmax = max(abs(np.nanmin(vort)), abs(np.nanmax(vort)))
        if vmax < 0.01:
            vmax = 5.0  # 数据较平时使用默认范围
        im2 = axes[1].pcolormesh(x, y, vort, cmap="RdBu_r",
                                  shading="auto", vmin=-vmax, vmax=vmax)
        axes[1].set_title(f"Relative Vorticity (×10³) at z={z_m:.0f} m")
        axes[1].set_xlabel("East-West [km]")
        axes[1].set_aspect("equal")
        plt.colorbar(im2, ax=axes[1], label="10⁻³ s⁻¹")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Plot] Saved to: {save_path}")

        return fig
