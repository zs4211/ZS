#!/usr/bin/env python
"""
双多普勒雷达三维风场反演主入口。

工作流程:
  1. 通过 PyCINRAD 读取 CINRAD 径向速度数据并进行质量控制
  2. 将每部雷达的数据网格化到统一的笛卡尔坐标系
  3. 运行 PyDDA 三维变分分析反演 u, v, w 风场分量
  4. 将结果保存为 NetCDF 格式并生成诊断图

用法:
  python run_retrieval.py                          # 使用最佳时间匹配的雷达对
  python run_retrieval.py --timestamp1 20240613103000Z --timestamp2 20240613102947Z
  python run_retrieval.py --match-all              # 处理所有可用的雷达对
  python run_retrieval.py --list-matches           # 列出所有可用的时间匹配对
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path

# 将 src 目录加入搜索路径以便本地导入
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import RadarDataPipeline
from src.retrieval import WindRetrieval


def _plot_wind_from_grid(grid, save_path):
    """从已有的 PyART Grid 生成诊断图（风速+涡度）。"""
    import numpy as np
    import matplotlib.pyplot as plt

    u = np.array(grid.fields["u"]["data"])
    v = np.array(grid.fields["v"]["data"])
    speed = np.array(grid.fields["wind_speed"]["data"])
    vort = np.array(grid.fields["vorticity"]["data"]) * 1e3  # → 10⁻³ s⁻¹

    nz = len(grid.z["data"])
    # 选涡度最强的层
    search_range = slice(0, max(1, nz * 2 // 3))
    level = int(np.argmax(np.nanmax(np.abs(vort[search_range]), axis=(1, 2))))
    level = max(0, min(level, nz - 1))

    z_m = grid.z["data"][level]
    x = grid.x["data"] / 1000.0
    y = grid.y["data"] / 1000.0

    step = max(1, len(y) // 30)
    sl = slice(step // 2, None, step)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].pcolormesh(x, y, speed[level], cmap="viridis", shading="auto")
    axes[0].quiver(x[sl], y[sl], u[level, sl, sl], v[level, sl, sl],
                   color="white", scale=150, width=0.003)
    axes[0].set_title(f"Wind Speed + Vectors at z={z_m:.0f} m")
    axes[0].set_xlabel("East-West [km]")
    axes[0].set_ylabel("North-South [km]")
    axes[0].set_aspect("equal")
    plt.colorbar(im, ax=axes[0], label="m/s")

    vmax = max(abs(np.nanmin(vort[level])), abs(np.nanmax(vort[level])), 5.0)
    im2 = axes[1].pcolormesh(x, y, vort[level], cmap="RdBu_r",
                              shading="auto", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"Relative Vorticity (x10^-3) at z={z_m:.0f} m")
    axes[1].set_xlabel("East-West [km]")
    axes[1].set_aspect("equal")
    plt.colorbar(im2, ax=axes[1], label="10^-3 s^-1")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="基于 PyDDA/三维变分分析的双多普勒雷达三维风场反演")

    # 输入参数
    parser.add_argument("--station1", default="Z9317")
    parser.add_argument("--station2", default="Z9543")
    parser.add_argument("--timestamp1", default=None)
    parser.add_argument("--timestamp2", default=None)
    parser.add_argument("--product", default="velocity", choices=["velocity"],
                        help="反演观测量；当前仅允许径向速度")
    parser.add_argument("--resolution", default="high",
                        choices=["high", "low"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument(
        "--max-time-diff", type=float, default=2.0,
        help="两站体扫允许的最大时间差（分钟，默认=2）",
    )

    # 网格参数
    parser.add_argument("--grid-nz", type=int, default=20)
    parser.add_argument("--grid-ny", type=int, default=201)
    parser.add_argument("--grid-nx", type=int, default=201)
    parser.add_argument("--grid-zmin", type=float, default=500.0)
    parser.add_argument("--grid-zmax", type=float, default=15000.0)
    parser.add_argument("--grid-xymax", type=float, default=150000.0)

    # 三维变分分析参数（针对双多普勒反演调优）
    parser.add_argument("--Co", type=float, default=100.0,
                        help="观测约束权重（默认=100）")
    parser.add_argument("--Cb", type=float, default=0.01,
                        help="背景场约束权重（默认=0.01）")
    parser.add_argument("--Cm", type=float, default=1000.0,
                        help="质量连续约束权重（默认=1000）")
    parser.add_argument("--Cx", type=float, default=1e-4,
                        help="X方向平滑度权重（默认=1e-4）")
    parser.add_argument("--Cy", type=float, default=1e-4,
                        help="Y方向平滑度权重（默认=1e-4）")
    parser.add_argument("--Cz", type=float, default=1e-4,
                        help="Z方向平滑度权重（默认=1e-4）")
    parser.add_argument("--wind-tol", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--min-cross-angle", type=float, default=20.0,
                        help="双雷达初始场的最小交叉波束角（度，默认=20）")
    parser.add_argument("--dual-radar-iter", type=int, default=0,
                        help="双雷达初始场的垂直速度反馈迭代次数（默认=0，对初猜场已足够）")
    parser.add_argument("--filter-window", type=int, default=5,
                        help="Savitzky-Golay 低通滤波窗口（默认=5）")
    parser.add_argument("--filter-order", type=int, default=3,
                        help="低通滤波阶数（默认=3）")

    # 运行模式
    parser.add_argument("--list-matches", action="store_true")
    parser.add_argument("--match-all", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-only", action="store_true",
                        help="仅从已有的反演结果生成图表")

    args = parser.parse_args()

    if args.max_time_diff <= 0:
        parser.error("--max-time-diff must be positive")
    if not 0 < args.min_cross_angle < 90:
        parser.error("--min-cross-angle must be between 0 and 90 degrees")
    if bool(args.timestamp1) != bool(args.timestamp2):
        parser.error("--timestamp1 and --timestamp2 must be supplied together")

    # ---- 模式：列出时间匹配对 ----
    pipeline = RadarDataPipeline(data_dir=args.data_dir)
    if args.list_matches:
        matches = pipeline.match_times(
            args.station1, args.station2, pipeline.data_dir,
            product_type=args.product, resolution=args.resolution,
            max_diff_minutes=args.max_time_diff)
        print(f"Found {len(matches)} matching time pairs:")
        for t1, t2, d in matches:
            print(f"  {t1} <-> {t2} (±{d:.1f} min)")
        return 0

    # ---- 模式：仅从已有结果生成图表 ----
    if args.plot_only:
        print("Plot-only mode: re-plotting from existing wind3d_*.nc ...")
        output_dir = Path(args.output_dir)
        nc_files = sorted(output_dir.glob("wind3d_*.nc"))
        if not nc_files:
            print("No wind3d_*.nc files found in output directory.")
            return 1
        import pyart
        import matplotlib.pyplot as plt
        for nc_file in nc_files:
            try:
                g = pyart.io.read_grid(str(nc_file))
                png_file = nc_file.with_suffix(".png")
                _plot_wind_from_grid(g, str(png_file))
                plt.close("all")
                print(f"  → {png_file.name}")
            except Exception as e:
                print(f"  ERROR {nc_file.name}: {e}")
        return 0

    # 这些参数只参与实际反演，列配对和重画已有结果不应受其影响。
    if args.filter_window <= args.filter_order:
        parser.error("--filter-window must be greater than --filter-order")
    if args.filter_window % 2 == 0:
        parser.error("--filter-window must be an odd integer")
    if args.filter_window > min(args.grid_nz, args.grid_ny, args.grid_nx):
        parser.error(
            "--filter-window cannot exceed the smallest grid dimension "
            f"({min(args.grid_nz, args.grid_ny, args.grid_nx)})"
        )

    # ---- 设置网格配置 ----
    grid_shape = (args.grid_nz, args.grid_ny, args.grid_nx)
    grid_limits = (
        (args.grid_zmin, args.grid_zmax),
        (-args.grid_xymax, args.grid_xymax),
        (-args.grid_xymax, args.grid_xymax),
    )

    # ---- 确定时间匹配对 ----
    time_pairs = []
    if args.timestamp1 and args.timestamp2:
        try:
            dt1 = datetime.strptime(args.timestamp1, "%Y%m%d%H%M%SZ")
            dt2 = datetime.strptime(args.timestamp2, "%Y%m%d%H%M%SZ")
        except ValueError as exc:
            parser.error(f"timestamps must use YYYYMMDDHHMMSSZ: {exc}")
        diff = abs((dt1 - dt2).total_seconds()) / 60.0
        if diff > args.max_time_diff:
            parser.error(
                f"specified scans differ by {diff:.2f} min, exceeding "
                f"--max-time-diff={args.max_time_diff:.2f} min"
            )
        time_pairs = [(args.timestamp1, args.timestamp2, diff)]
    elif args.match_all:
        matches = pipeline.match_times(
            args.station1, args.station2, pipeline.data_dir,
            product_type=args.product, resolution=args.resolution,
            max_diff_minutes=args.max_time_diff)
        time_pairs = matches
        print(f"Processing {len(matches)} time pairs...")
    else:
        # 自动选择时间差最小的最佳匹配对
        matches = pipeline.match_times(
            args.station1, args.station2, pipeline.data_dir,
            product_type=args.product, resolution=args.resolution,
            max_diff_minutes=args.max_time_diff)
        if not matches:
            print("No matching timestamps found!")
            return 1
        matches.sort(key=lambda x: x[2])
        time_pairs = [matches[0]]
        print(f"Auto-selected best pair: {matches[0][0]} <-> {matches[0][1]} "
              f"(diff={matches[0][2]:.1f} min)")

    # ---- 处理每个时间匹配对 ----
    n_success = 0
    for idx, (ts1, ts2, diff) in enumerate(time_pairs):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(time_pairs)}] {ts1} <-> {ts2} (±{diff:.1f} min)")
        print(f"{'='*60}")

        try:
            # 步骤1: 读取雷达数据并进行质量控制
            print("\n>>> Step 1: Reading and QC...")
            sweeps1 = pipeline.build_volume(
                args.station1, ts1, args.product, args.resolution)
            sweeps2 = pipeline.build_volume(
                args.station2, ts2, args.product, args.resolution)
            refl_sweeps1 = pipeline.build_volume(
                args.station1, ts1, "reflectivity", args.resolution)
            refl_sweeps2 = pipeline.build_volume(
                args.station2, ts2, "reflectivity", args.resolution)

            if (len(sweeps1) == 0 or len(sweeps2) == 0
                    or len(refl_sweeps1) == 0 or len(refl_sweeps2) == 0):
                print("SKIP: Missing velocity or reflectivity volume for one station")
                continue

            radar1 = pipeline.sweeps_to_pyart_radar(sweeps1)
            radar2 = pipeline.sweeps_to_pyart_radar(sweeps2)
            refl_radar1 = pipeline.sweeps_to_pyart_radar(refl_sweeps1)
            refl_radar2 = pipeline.sweeps_to_pyart_radar(refl_sweeps2)
            print(f"  {args.station1}: {radar1.nsweeps} sweeps, {radar1.nrays} rays")
            print(f"  {args.station2}: {radar2.nsweeps} sweeps, {radar2.nrays} rays")

            radar1 = pipeline.quality_control(radar1)
            radar2 = pipeline.quality_control(radar2)
            refl_radar1 = pipeline.quality_control(refl_radar1)
            refl_radar2 = pipeline.quality_control(refl_radar2)

            # 步骤2: 网格化 + 双雷达初始风场
            print(f"\n>>> Step 2: Gridding + dual-radar initial wind...")
            vel_field = list(radar1.fields.keys())[0]  # 使用QC后的首个场名
            retrieval = WindRetrieval(
                radar1=radar1,
                radar2=radar2,
                reflectivity_radar1=refl_radar1,
                reflectivity_radar2=refl_radar2,
                vel_field=vel_field,
                grid_shape=grid_shape,
                grid_limits=grid_limits,
                station1=args.station1,
                station2=args.station2,
                data_dir=args.data_dir,
            )

            # VWP 背景风廓线 + 双雷达修正 → 融合初猜场
            u_init, v_init, w_init = retrieval.make_initial_wind(
                min_cross_beam_angle_deg=args.min_cross_angle,
                max_iterations=args.dual_radar_iter,
            )

            # 运行三维变分分析
            print(f"\n>>> Step 3: 3DVAR optimization...")
            retrieved_grid = retrieval.retrieve(
                u_init=u_init,
                v_init=v_init,
                w_init=w_init,
                Co=args.Co,
                Cb=args.Cb,
                Cm=args.Cm,
                Cx=args.Cx,
                Cy=args.Cy,
                Cz=args.Cz,
                wind_tol=args.wind_tol,
                max_iterations=args.max_iter,
                filter_window=args.filter_window,
                filter_order=args.filter_order,
                min_bca=args.min_cross_angle,
            )

            # 步骤4: 保存结果
            print(f"\n>>> Step 4: Saving...")
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            nc_file = output_path / f"wind3d_{ts1}_{ts2}.nc"
            retrieval.save(str(nc_file))

            # 生成诊断图
            if not args.no_plot:
                plot_file = output_path / f"wind3d_{ts1}_{ts2}.png"
                retrieval.plot(save_path=str(plot_file))

            print(f"\n  Finished {ts1} <-> {ts2}")
            n_success += 1

        except Exception as e:
            print(f"\n  ERROR processing {ts1}/{ts2}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Done. Successfully processed {n_success}/{len(time_pairs)} time pair(s).")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    return 0 if n_success == len(time_pairs) and n_success > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
