"""
三维中气旋风场可视化系统（基于PyVista）。

加载 run_retrieval.py 生成的三维变分风场反演结果 (wind3d_*.nc)，
自动检测双雷达有效区内的气旋式正涡度候选，生成螺旋流线，
并渲染包含多层地理信息叠加的交互式三维场景。

可视化层次结构（从下到上）:
  Layer 0: 二维地图底面（cartopy 地理底图作为纹理映射到地面平面）
  Layer 1: 反射率 CR CAPPI 半透明叠加（彩色 dBZ 色标）
  Layer 2: 地理边界与城市名标注（县界、河流、地名标签，三维点标注）
  Layer 3: 中气旋标记与雷达站点（白色球体标识）
  3D 层: 涡度等值面 + 三维螺旋流线管（彩色风速映射）

用法:
  python visualize_3d.py                          # 自动选择最新文件
  python visualize_3d.py output/wind3d_xxx.nc     # 指定文件
  python visualize_3d.py --list                   # 列出可用文件
  python visualize_3d.py --no-display --output screenshot.png
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.proj_setup import configure_proj

configure_proj()
import numpy as np
import pyart
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from src.gridder import AnalysisGrid


# ══════════════════════════════════════════════════════════════
#  MesoVisualizer（基于PyVista的三维渲染器）
# ══════════════════════════════════════════════════════════════

import pyvista as pv

def generate_basemap_texture(grid, boundary_lines=None,
                              city_positions=None, city_names=None,
                              save_path="basemap_temp.png"):
        """
        生成包含所有地理要素的二维底图纹理。

        将 cartopy 自然地理要素 + GADM 县界 + 城市名全部渲染到一张
        高分辨率图片中，作为 PyVista 的地面纹理。避免了多图层 3D
        渲染时的 z-fighting、可见性、坐标缩放不一致等问题。

        返回:
            basemap 图片的文件路径
        """
        lon_min = float(grid.lon_grid.min())
        lon_max = float(grid.lon_grid.max())
        lat_min = float(grid.lat_grid.min())
        lat_max = float(grid.lat_grid.max())

        lon_margin = (lon_max - lon_min) * 0.05
        lat_margin = (lat_max - lat_min) * 0.05
        lon_min -= lon_margin; lon_max += lon_margin
        lat_min -= lat_margin; lat_max += lat_margin

        lat_center = (lat_min + lat_max) / 2
        cos_lat = np.cos(np.radians(lat_center))
        width_ratio = (lon_max - lon_min) * cos_lat / (lat_max - lat_min)
        fig_width = 12
        fig_height = max(8, min(16, fig_width / width_ratio if width_ratio > 0 else 12))

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=200)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.set_facecolor('#f0f0f0')

        # 自然地理要素（cartopy Natural Earth）
        ax.add_feature(cfeature.OCEAN, facecolor="#666dbb", edgecolor='none')
        ax.add_feature(cfeature.LAND, facecolor="#e8e8e87c", edgecolor='none')
        ax.add_feature(cfeature.LAKES, facecolor='#dddddd', edgecolor='none')
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#444444')
        ax.add_feature(cfeature.RIVERS, edgecolor='#6699bb', linewidth=0.8)

        # GADM 县级边界线（叠加在 cartopy 底图上）
        if boundary_lines:
            for pts in boundary_lines:
                if len(pts) < 2:
                    continue
                # pts 是 ENU 坐标 (km)，需要转回 lon/lat
                lons, lats = [], []
                for pt in pts:
                    e_km, n_km = pt[0], pt[1]
                    lat_i = grid.origin_lat + n_km / 111.32
                    lon_i = grid.origin_lon + e_km / (111.32 * np.cos(np.radians(grid.origin_lat)))
                    lons.append(lon_i)
                    lats.append(lat_i)
                ax.plot(lons, lats, color='#333333', linewidth=0.8,
                       transform=ccrs.PlateCarree(), alpha=0.7)

        # 城市名称标注
        if city_positions and city_names:
            for (cx, cy), name in zip(city_positions, city_names):
                lat_c = grid.origin_lat + cy / 111.32
                lon_c = grid.origin_lon + cx / (111.32 * np.cos(np.radians(grid.origin_lat)))
                ax.text(lon_c, lat_c, name, fontsize=9, color="#222222CC",
                       ha='center', va='bottom', weight='bold',
                       transform=ccrs.PlateCarree())

        

        # 经纬度网格线
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='#cccccc', alpha=0.6)
        gl.top_labels = False; gl.right_labels = False

        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.05,
                   facecolor='white')
        plt.close()
        return save_path

class MesoVisualizer:
    """中气旋风场三维可视化器。"""

    def __init__(self, grid, background='#1a1a2e', z_scale=1.5, off_screen=False):
        self.grid = grid
        self.z_scale = z_scale
        self.plotter = pv.Plotter(window_size=[1400, 900], off_screen=off_screen)
        self.plotter.set_background(background)
        self.plotter.enable_depth_peeling()
        self.plotter.enable_anti_aliasing()

    def _z(self, z_val):
        """对高度值应用垂直夸张变换。"""
        return z_val * self.z_scale

    def _pts_3d(self, pts):
        """将点坐标数组的z分量按垂直夸张倍数缩放。"""
        out = pts.copy()
        out[:, 2] *= self.z_scale
        return out

    # ── 图层1: DEM地形 ──

    def add_dem_terrain(self, surface_mesh):
        """
        添加灰蓝色DEM地形曲面，使用Phong光照模型增强三维起伏感。

        Phong光照参数说明：
            - specular(镜面反射): 0.3 — 控制地表高光强度，模拟湿润/岩石表面的光泽
            - specular_power(镜面反射指数): 20 — 控制高光集中度，值越大高光斑越锐利
            - diffuse(漫反射): 0.8 — 控制表面基础亮度，使地形在不同角度都可见
            - ambient(环境光): 0.3 — 模拟散射光，确保阴影区不完全黑暗
        """
        self.plotter.add_mesh(
            surface_mesh,
            color=[0.53, 0.60, 0.67],
            specular=0.3, specular_power=20,
            diffuse=0.8, ambient=0.3,
            lighting=True, smooth_shading=True,
            name='dem_terrain', show_scalar_bar=False,
        )
    # ── 图层2: 反射率CAPPI（半透明叠加） ──

    # 组合反射率色标（15级，-5 ~ 65 dBZ）
    _REFL_LEVELS = np.array([-5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
                            dtype=np.float32)
    _REFL_COLORS = np.array([
        [0, 255, 255],    #  -5 dBZ — 青 (Cyan)
        [180, 130, 220],  #   0 dBZ — 浅紫 (Light purple)
        [0, 0, 205],      #   5 dBZ — 中蓝 (Medium blue)
        [30, 144, 255],   #  10 dBZ — 亮蓝 (Bright blue)
        [0, 255, 127],    #  15 dBZ — 浅绿 (Light green)
        [0, 255, 0],      #  20 dBZ — 鲜绿 (Bright green)
        [0, 160, 0],      #  25 dBZ — 深绿 (Dark green)
        [173, 255, 47],   #  30 dBZ — 黄绿 (Yellow-green)
        [255, 255, 0],    #  35 dBZ — 黄 (Yellow)
        [200, 180, 0],    #  40 dBZ — 橄榄黄 (Olive/Khaki)
        [255, 160, 122],  #  45 dBZ — 桃色/浅肉色 (Peach/Salmon)
        [255, 140, 0],    #  50 dBZ — 橙 (Orange)
        [255, 0, 0],      #  55 dBZ — 红 (Red)
        [230, 50, 230],   #  60 dBZ — 淡紫/洋红 (Light magenta)
        [160, 0, 200],    #  65 dBZ — 紫 (Purple)
    ], dtype=np.float32) / 255.0

    def add_reflectivity_cappi(self, refl_2d, z_offset=0.05):
        """
        在地图底面上方添加反射率叠加层。

        使用 15 级色标：青→浅紫→蓝→亮蓝→绿→黄绿→黄→橄榄黄→
        桃色→橙→红→洋红→紫。覆盖 -5 ~ 65 dBZ，低于 -5 dBZ 透明。
        """
        x = self.grid.X[:, :, 0]
        y = self.grid.Y[:, :, 0]
        z_vals = np.full_like(x, z_offset * self.z_scale)

        refl_clean = np.nan_to_num(refl_2d, nan=-99.0)
        nx, ny = refl_clean.shape

        # 将 dBZ 截断到色标有效范围
        dbz_clipped = np.clip(refl_clean, -5.0, 65.0)

        rgba = np.zeros((nx, ny, 4), dtype=np.float32)

        # 对 R/G/B 三通道在色标等级间线性插值
        for c in range(3):
            rgba[:, :, c] = np.interp(
                dbz_clipped, self._REFL_LEVELS, self._REFL_COLORS[:, c])

        # Alpha: -5 dBZ=0.12 → 45 dBZ=1.0，线性渐变
        rgba[:, :, 3] = np.clip((dbz_clipped + 5.0) / 50.0, 0.12, 1.0)

        rf_plane = pv.StructuredGrid(x, y, z_vals)
        rgba_flat = rgba.reshape(nx * ny, 4, order='F')
        rf_plane.point_data['rgba'] = rgba_flat

        self.plotter.add_mesh(
            rf_plane, scalars='rgba', rgba=True,
            show_scalar_bar=False,
            name='reflectivity_cappi', lighting=False, ambient=1.0,
        )

    # ── 图层3: 地理边界 ──

    def add_geographic_boundaries(self, boundary_lines,
                                  z_offset=0.03):
        """在地图底面上方添加白色县界和河流线条。"""
        if not boundary_lines:
            return
        for i, pts in enumerate(boundary_lines):
            if len(pts) < 2:
                continue
            # z_offset: ENU km，_pts_3d会统一乘z_scale
            pts = pts.copy()
            pts[:, 2] = z_offset
            pts_scaled = self._pts_3d(pts.astype(np.float32))
            poly = pv.PolyData()
            poly.points = pts_scaled
            npts = len(pts_scaled)
            lines_arr = np.zeros(npts + 1, dtype=np.int64)
            lines_arr[0] = npts
            lines_arr[1:] = np.arange(npts)
            poly.lines = lines_arr
            self.plotter.add_mesh(
                poly, color='white', line_width=4.0,
                opacity=0.95, name=f'geo_boundary_{i}',
                render_lines_as_tubes=True, lighting=False,
            )

    def add_basemap_plane(self, texture_path):
        """
        在 z=0 平面创建一个矩形网格，并将地图纹理贴上去。
        使用 PyVista 的 Plane 函数自动生成纹理坐标。

        纹理映射说明：
            底图由 cartopy 生成的灰度地图作为纹理贴到地面平面上，
            提供海洋、陆地、行政边界等参考信息。
        """
        # 获取 ENU 范围
        x_min, x_max = self.grid.x[0], self.grid.x[-1]
        y_min, y_max = self.grid.y[0], self.grid.y[-1]

        # 创建矩形平面（默认在 xy 平面，法向 z）
        plane = pv.Plane(
            center=(0, 0, -0.01),
            direction=(0, 0, 1),
            i_size=x_max - x_min,
            j_size=y_max - y_min,
            i_resolution=1,
            j_resolution=1,
        )
        # 平移至正确的 x,y 位置
        plane.points[:, 0] += (x_min + x_max) / 2
        plane.points[:, 1] += (y_min + y_max) / 2

        # 读取纹理
        texture = pv.read_texture(texture_path)

        # 添加到场景
        self.plotter.add_mesh(
            plane,
            texture=texture,
            lighting=False,
            name='basemap_plane',
            show_scalar_bar=False,
            opacity=1.0,
        )

        # 可选：添加一个半透明的网格线框，便于定位
        # self.plotter.add_mesh(mesh, style='wireframe', color='black', line_width=1, opacity=0.3)

    # ── 城市标签 ──
    def add_city_labels(self, city_positions, city_names, z_offset=10.0):
        """在地图底面上方添加城市名称标签（白色，始终可见）。"""
        if not city_positions:
            return
        pts_3d = []
        names = []
        for (cx, cy), name in zip(city_positions, city_names):
            pts_3d.append([cx, cy, self._z(z_offset)])
            names.append(name)
        pts_arr = np.array(pts_3d, dtype=np.float32)
        self.plotter.add_point_labels(
            pts_arr, names,
            font_size=12, text_color='black',
            point_size=0, shape_opacity=0.0,
            always_visible=True, name='city_labels',
            fill_shape=False,
        )
    

    def add_wind_vector_slice(self, u_2d, v_2d, speed_2d, z_level=2.0,
                               skip=6, scale=2.5):
        """
        在指定高度层添加彩色风矢量切片。

        风矢量显示真实的反演水平风场（方向=风向，长度=风速，颜色=风速），
        放置在涡度最强的多个高度层上。多层面叠加后能清晰展示中气旋的
        旋转结构随高度的变化。

        相比流线，风矢量切片的优势：
          - 不依赖风场的空间连续性（每个箭头独立）
          - 不会因局部噪声而偏离到错误路径
          - 能同时展示旋转、辐合/辐散、以及风速分布
        """
        x = self.grid.x[::skip]
        y = self.grid.y[::skip]
        nx_sub, ny_sub = len(x), len(y)

        Xg, Yg = np.meshgrid(x, y, indexing='ij')
        u_sub = u_2d[::skip, ::skip]
        v_sub = v_2d[::skip, ::skip]
        s_sub = speed_2d[::skip, ::skip]

        z_3d = np.full(Xg.size, self._z(z_level))

        pts = np.column_stack([Xg.ravel(), Yg.ravel(), z_3d])
        vec = np.column_stack([u_sub.ravel(), v_sub.ravel(), np.zeros(Xg.size)])

        valid = (np.isfinite(vec[:, 0]) & np.isfinite(vec[:, 1]) &
                 (np.sqrt(vec[:, 0]**2 + vec[:, 1]**2) > 0.3))

        if valid.sum() > 2:
            cloud = pv.PolyData(pts[valid])
            cloud['vectors'] = vec[valid].astype(np.float32)
            cloud['speed'] = s_sub.ravel()[valid].astype(np.float32)
            arrows = cloud.glyph(
                orient='vectors', scale='vectors', factor=scale * 0.3,
            )
            self.plotter.add_mesh(
                arrows, scalars='speed', cmap='turbo',
                clim=[0, 25],
                show_scalar_bar=False,
                name=f'wind_slice_{z_level:.0f}',
                lighting=False, ambient=1.0,
            )

    def add_ground_wind_arrows(self, u_2d, v_2d, skip=3, scale=2.0):
        """
        在地面平面上添加白色风矢量箭头。

        使用PyVista的glyph(字形)功能将矢量数据转换为三维箭头几何体。
        箭头方向 = 风向，箭头长度 = 风速。
        skip参数控制采样间隔以避免箭头过于密集。
        """
        x = self.grid.x[::skip]
        y = self.grid.y[::skip]
        Xg, Yg = np.meshgrid(x, y, indexing='ij')
        u_sub = u_2d[::skip, ::skip]
        v_sub = v_2d[::skip, ::skip]

        pts = np.column_stack([Xg.ravel(), Yg.ravel(), np.zeros(Xg.size)])
        vec = np.column_stack([u_sub.ravel(), v_sub.ravel(), np.zeros(u_sub.size)])

        valid = np.isfinite(vec[:, 0]) & np.isfinite(vec[:, 1])
        speeds_2d = np.sqrt(vec[valid, 0]**2 + vec[valid, 1]**2)
        valid &= speeds_2d > 0.5

        if valid.sum() > 2:
            cloud = pv.PolyData(pts[valid])
            cloud['vectors'] = vec[valid].astype(np.float32)
            arrows = cloud.glyph(orient='vectors', scale='vectors', factor=scale * 0.4)
            self.plotter.add_mesh(arrows, color='white', name='wind_arrows',
                                 lighting=False)

    def add_reflectivity_isosurface(self, refl_data, levels=(30, 40, 50, 60)):
        """
        添加反射率等值面轮廓（带垂直夸张）。

        等值面算法说明：
            使用Marching Cubes算法从三维标量场中提取指定值对应的等值面。
            对于反射率场，30dBZ对应弱对流边界，40dBZ对应中等到强对流，
            50dBZ对应强冰雹信号，60dBZ对应极端天气。
        """
        grid_pv = pv.ImageData()
        grid_pv.dimensions = [self.grid.nx + 1, self.grid.ny + 1, self.grid.nz + 1]
        grid_pv.origin = [self.grid.x[0], self.grid.y[0], self._z(self.grid.z[0])]
        grid_pv.spacing = [self.grid.dx, self.grid.dy, self.grid.dz * self.z_scale]

        refl_clean = np.nan_to_num(refl_data, nan=-10.0)
        refl_ordered = refl_clean.ravel(order='F')
        grid_pv.cell_data['reflectivity'] = refl_ordered.astype(np.float32)

        colors = {25: 'green', 30: 'cyan', 40: 'yellow', 50: 'orange', 60: 'red'}
        for level in levels:
            try:
                contour = grid_pv.contour([level], scalars='reflectivity')
                if contour.n_points > 0:
                    self.plotter.add_mesh(
                        contour, color=colors.get(level, 'white'),
                        opacity=0.12 + level / 300, name=f'iso_{level}dBZ'
                    )
            except Exception:
                pass

    def add_vorticity_isosurface(self, vorticity_3d, levels=None):
        """
        添加涡度等值面——展示真实反演的中气旋三维结构。

        这是整个可视化中最核心的三维结构展示层。涡度等值面直接来自
        反演数据，不需要任何人造模型。这里只显示北半球气旋式
        （正）垂直涡度，它能显示中气旋的:
          - 柱状旋转结构（垂直涡度管）
          - 空间范围和形状不对称性
          - 随高度的强度变化

        使用 Marching Cubes 算法从三维涡度标量场提取等值面。
        半透明渲染使内部结构可见。

        参数
        ----------
        vorticity_3d : np.ndarray, 形状 (nx, ny, nz)
            三维涡度场 (1/s)，仅正值参与等值面提取。
        levels : list, 可选
            等值面水平（1/s）。默认根据涡度分布自动选择。
        """
        vort_cyclonic = np.where(
            np.isfinite(vorticity_3d) & (vorticity_3d > 0),
            vorticity_3d,
            0.0,
        )
        if levels is None:
            signal = vort_cyclonic[vort_cyclonic > 1e-6]
            if signal.size == 0:
                return
            vmax = float(np.percentile(signal, 99.5))
            if vmax < 0.001:
                return
            # 自动选层：高值→内部核心，低值→外围边界
            levels = sorted([vmax * 0.3, vmax * 0.5, vmax * 0.7])

        grid_pv = pv.ImageData()
        grid_pv.dimensions = [self.grid.nx + 1, self.grid.ny + 1, self.grid.nz + 1]
        grid_pv.origin = [self.grid.x[0], self.grid.y[0], self._z(self.grid.z[0])]
        grid_pv.spacing = [self.grid.dx, self.grid.dy, self.grid.dz * self.z_scale]

        vort_flat = vort_cyclonic.ravel(order='F')
        grid_pv.cell_data['cyclonic_vorticity'] = vort_flat.astype(np.float32)

        # 使用红-白-蓝渐变：正涡度（气旋式）= 红色，负涡度 = 蓝色
        colors = {0: 'red', 1: 'orange', 2: 'yellow'}
        for i, level in enumerate(levels):
            try:
                contour = grid_pv.contour([level], scalars='cyclonic_vorticity')
                if contour.n_points > 0:
                    self.plotter.add_mesh(
                        contour,
                        color=colors.get(i, 'white'),
                        opacity=0.25 + 0.15 * i,  # 内层更不透明
                        name=f'vort_iso_{i}',
                        show_scalar_bar=False,
                        specular=0.5, specular_power=15,
                        smooth_shading=True,
                    )
                    print(f"  [涡度等值面] level={level:.5f} 1/s, "
                          f"顶点数={contour.n_points}")
            except Exception as e:
                pass

    def add_radar_sites(self, stations):
        """在地面层添加雷达站点标记（白色球体 + 站名标签）。"""
        for name, site in stations.items():
            from src.gridder import wgs84_to_enu
            e, n, u = wgs84_to_enu(
                np.array([site.lat]), np.array([site.lon]),
                np.array([site.height / 1000.0]),
                self.grid.origin_lat, self.grid.origin_lon
            )
            ex, ny = float(e.item()), float(n.item())
            sphere = pv.Sphere(radius=2.0, center=(ex, ny, 0.2))
            self.plotter.add_mesh(sphere, color='white', name=f'radar_{name}')
            self.plotter.add_point_labels(
                [(ex, ny, 1.0)], [name],
                font_size=14, text_color='white',
                point_size=0, shape_opacity=0,
                always_visible=True
            )

    def add_streamlines(self, streamlines, speeds, tube_radius=0.15):
        """
        添加流线管并进行颜色映射。

        使用turbo色标，固定风速范围0~30 m/s以保持不同场景间的可比性。
        流线由线段构成管状几何体(tube)，使三维结构更加直观。
        """
        if not streamlines:
            print("  No streamlines to add")
            return

        vmin, vmax = 0.0, 30.0 # 将风场色标固定在 0 ~ 30 m/s

        # [修复] 动态计算色标范围
        all_speeds_flat = []
        for spd in speeds:
            all_speeds_flat.extend(spd.tolist())
        if all_speeds_flat:
            vmax = min(np.percentile(all_speeds_flat, 95), 40.0)
            vmax = max(vmax, 10.0)  # 最小范围 10 m/s
        else:
            vmax = 30.0
        vmin = 0.0
        
        for i, (pts, spd) in enumerate(zip(streamlines, speeds)):
            if len(pts) < 2:
                continue

            pts_scaled = self._pts_3d(pts.astype(np.float32))

            poly = pv.PolyData()
            poly.points = pts_scaled
            npts = len(pts_scaled)
            lines_arr = np.zeros(npts + 1, dtype=np.int64)
            lines_arr[0] = npts
            lines_arr[1:] = np.arange(npts)
            poly.lines = lines_arr
            poly['speed'] = spd.astype(np.float32)

            try:
                tube = poly.tube(radius=tube_radius, n_sides=8)
                self.plotter.add_mesh(
                    tube, scalars='speed', cmap='turbo',
                    clim=[vmin, vmax],
                    show_scalar_bar=(i == 0),
                    scalar_bar_args={
                        'title': 'Wind Speed [m/s]',
                        'vertical': True,
                        'position_x': 0.85, 'position_y': 0.15,
                        'width': 0.06, 'height': 0.4,
                        'title_font_size': 18, 'label_font_size': 14,
                        'color': 'white',
                    },
                    name=f'streamline_{i}'
                )
            except Exception:
                self.plotter.add_mesh(poly, color='yellow', line_width=3,
                                     name=f'streamline_{i}')

    def add_meso_markers(self, positions, z_offset=0.06):
        """在中气旋中心位置添加白色标记球体。"""
        if not positions:
            return
        for i, (mx, my, _) in enumerate(positions):
            center = np.array([mx, my, z_offset * self.z_scale], dtype=np.float32)
            sphere = pv.Sphere(radius=2.0, center=center)
            self.plotter.add_mesh(
                sphere, color='white',
                ambient=1.0, diffuse=0.3,
                specular=0.2, specular_power=10,
                name=f'meso_marker_{i}', show_scalar_bar=False,
            )

    def add_bounding_box(self):
        """添加半透明边框立方体，提供空间参考框架。

        绘制分析域的外边界框及等间距的地面网格线，帮助判断三维空间位置。
        """
        x0, x1 = self.grid.x[0], self.grid.x[-1]
        y0, y1 = self.grid.y[0], self.grid.y[-1]
        z0, z1 = self._z(self.grid.z[0]), self._z(self.grid.z[-1])

        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])
        edges = [
            (0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ]
        for a, b in edges:
            line = pv.Line(corners[a], corners[b])
            self.plotter.add_mesh(line, color='gray', line_width=1,
                                 opacity=0.4, name=f'bbox_{a}_{b}')

        for xi in np.linspace(x0, x1, 6):
            line = pv.Line([xi, y0, z0], [xi, y1, z0])
            self.plotter.add_mesh(line, color='gray', line_width=0.5, opacity=0.3)
        for yi in np.linspace(y0, y1, 6):
            line = pv.Line([x0, yi, z0], [x1, yi, z0])
            self.plotter.add_mesh(line, color='gray', line_width=0.5, opacity=0.3)

    def add_axes(self):
        """添加坐标轴及标签（东、北、高度）。"""
        self.plotter.add_axes(
            xlabel='East [km]', ylabel='North [km]',
            zlabel=f'Height [km] (x{self.z_scale})',
            line_width=2
        )

    def add_title(self, time_str=""):
        """在视图左上角添加标题文字。"""
        self.plotter.add_text(
            f'Dual-Doppler Rotation Candidates{time_str}',
            font_size=16, color='white', position='upper_left'
        )

    def set_camera(self, azimuth=-60, elevation=20, distance=None):
        """设置相机视角以获得良好的三维透视效果。

        默认使用等距视图(isometric)，方位角-60度(从南偏东方向观察)，
        俯仰角20度，便于同时观察水平结构和垂直发展。
        """
        self.plotter.camera_position = 'iso'
        self.plotter.camera.azimuth = azimuth
        self.plotter.camera.elevation = elevation
        if distance:
            self.plotter.camera.distance = distance

    def show(self):
        """显示交互式窗口。"""
        self.add_axes()
        self.add_bounding_box()
        self.plotter.show_grid(color='#666688')
        self.plotter.show(title='Dual-Doppler Rotation Candidates',
                         interactive=True, auto_close=False)

    def save_screenshot(self, filepath):
        """保存截图（离屏渲染，无需显示窗口）。"""
        self.add_axes()
        self.add_bounding_box()
        self.plotter.screenshot(filepath, return_img=False)

    def close(self):
        """关闭渲染窗口。"""
        self.plotter.close()


# ══════════════════════════════════════════════════════════════
#  数据加载与网格构建
# ══════════════════════════════════════════════════════════════

def _parse_timestamps(filepath):
    """
    从wind3d文件名中提取两个雷达站的时间戳字符串。

    文件名格式：wind3d_YYYYMMDDHHMMSSZ_YYYYMMDDHHMMSSZ.nc
    返回 (ts1, ts2)，若提取失败则返回 (None, None)。
    """
    fname = Path(filepath).stem
    parts = fname.split("_")
    ts_list = [p for p in parts if len(p) == 15 and p.endswith("Z")]
    if len(ts_list) >= 2:
        return ts_list[0], ts_list[1]
    elif len(ts_list) == 1:
        return ts_list[0], ts_list[0]
    return None, None


def grid_from_pyart(g):
    """
    从PyART Grid对象构建AnalysisGrid。
    将坐标从米转换为公里。
    """
    x_m = g.x["data"]
    y_m = g.y["data"]
    z_m = g.z["data"]

    x = x_m / 1000.0
    y = y_m / 1000.0
    z = z_m / 1000.0

    dx = (x[1] - x[0]) if len(x) > 1 else 1.0
    dy = (y[1] - y[0]) if len(y) > 1 else 1.0
    dz = (z[1] - z[0]) if len(z) > 1 else 0.5

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    olat = float(g.origin_latitude["data"][0])
    olon = float(g.origin_longitude["data"][0])

    cos_ref = np.cos(np.radians(olat))
    deg_per_km_lat = 1.0 / 111.32
    deg_per_km_lon = 1.0 / (111.32 * cos_ref)
    lat_grid = olat + Y * deg_per_km_lat
    lon_grid = olon + X * deg_per_km_lon

    return AnalysisGrid(
        x=x, y=y, z=z, X=X, Y=Y, Z=Z,
        lon_grid=lon_grid, lat_grid=lat_grid,
        origin_lat=olat, origin_lon=olon,
        dx=dx, dy=dy, dz=dz,
        nx=len(x), ny=len(y), nz=len(z),
    )


def load_wind_data(filepath):
    """
    加载三维变分反演的风场数据，返回数据字典和AnalysisGrid。

    自动从 nc 文件元数据中读取 station1/station2，若缺失则回退为
    根据文件名时间戳的默认映射（ts1→Z9317, ts2→Z9543）。

    数据转置说明：
        PyART存储顺序为 (nz, ny, nx)，即高度-纬度-经度。
        流线模块期望的输入顺序为 (nx, ny, nz)，即经度-纬度-高度。
        因此在加载后进行转置以满足后续处理的需要。
    """
    print(f"[加载] {filepath}")
    g = pyart.io.read_grid(str(filepath))

    def _field(name):
        return np.ma.filled(g.fields[name]["data"], np.nan).astype(float)

    u = _field("u")          # (nz, ny, nx)
    v = _field("v")
    w = _field("w")
    wind_speed = _field("wind_speed")
    vorticity = _field("vorticity")
    if "dual_doppler_mask" in g.fields:
        dual_mask = np.asarray(
            g.fields["dual_doppler_mask"]["data"], dtype=bool
        )
    else:
        dual_mask = np.isfinite(vorticity)
    reflectivity = _field("reflectivity") if "reflectivity" in g.fields else None
    continuity = (
        _field("continuity_residual")
        if "continuity_residual" in g.fields else None
    )

    grid = grid_from_pyart(g)

    # 转置: PyART (nz, ny, nx) → 流线模块期望 (nx, ny, nz)
    u_t = np.transpose(u, (2, 1, 0))
    v_t = np.transpose(v, (2, 1, 0))
    w_t = np.transpose(w, (2, 1, 0))
    wind_speed_t = np.transpose(wind_speed, (2, 1, 0))
    vorticity_t = np.transpose(vorticity, (2, 1, 0))
    dual_mask_t = np.transpose(dual_mask, (2, 1, 0))
    reflectivity_t = (
        np.transpose(reflectivity, (2, 1, 0))
        if reflectivity is not None else None
    )
    continuity_t = (
        np.transpose(continuity, (2, 1, 0))
        if continuity is not None else None
    )

    ts1, ts2 = _parse_timestamps(filepath)

    # 从 nc 元数据读取站点号
    station1 = g.metadata.get("station1", "Z9317")
    station2 = g.metadata.get("station2", "Z9543")

    print(f"  网格: {grid.nx}x{grid.ny}x{grid.nz}")
    print(f"  站点: {station1} (ts1) + {station2} (ts2)")
    print(f"  范围: x=[{grid.x[0]:.0f},{grid.x[-1]:.0f}]km, "
          f"y=[{grid.y[0]:.0f},{grid.y[-1]:.0f}]km, "
          f"z=[{grid.z[0]:.2f},{grid.z[-1]:.2f}]km")
    def _finite_range(values):
        finite = np.asarray(values)[np.isfinite(values)]
        return ((float(finite.min()), float(finite.max()))
                if finite.size else (float("nan"), float("nan")))

    u_range = _finite_range(u)
    v_range = _finite_range(v)
    w_range = _finite_range(w)
    vort_range = _finite_range(np.abs(vorticity))
    print(f"  u: [{u_range[0]:.1f},{u_range[1]:.1f}] m/s")
    print(f"  v: [{v_range[0]:.1f},{v_range[1]:.1f}] m/s")
    print(f"  w: [{w_range[0]:.2f},{w_range[1]:.2f}] m/s")
    print(f"  |vort|: [{vort_range[0]:.4f},{vort_range[1]:.4f}] 1/s")

    return {
        "u": u_t, "v": v_t, "w": w_t,
        "wind_speed": wind_speed_t, "vorticity": vorticity_t,
        "dual_doppler_mask": dual_mask_t,
        "reflectivity_3d": reflectivity_t,
        "continuity_residual": continuity_t,
        "metadata": dict(g.metadata),
        "grid": grid, "timestamps": (ts1, ts2),
        "station1": station1, "station2": station2,
    }


# ══════════════════════════════════════════════════════════════
#  中气旋检测
# ══════════════════════════════════════════════════════════════

def detect_mesocyclones(vorticity, grid, z_range=(1.0, 5.0),
                        min_separation_km=10.0, max_count=5,
                        min_vorticity=0.005, min_vertical_layers=3,
                        valid_mask=None, reflectivity=None,
                        min_reflectivity_dbz=30.0):
    """Detect observed, vertically coherent cyclonic-vorticity columns.

    Candidates must be inside valid dual-Doppler coverage and, when 3-D
    reflectivity is available, inside precipitation echo.  A small horizontal
    neighbourhood is used for the vertical-continuity check so a tilted
    mesocyclone column is not rejected merely because it changes grid cell.
    """
    z_mask = (grid.z >= z_range[0]) & (grid.z <= z_range[1])
    z_indices = np.where(z_mask)[0]

    if len(z_indices) == 0:
        print("  [警告] 指定高度范围内无数据，扩大搜索范围")
        z_indices = np.arange(len(grid.z))

    if valid_mask is None:
        valid_mask = np.isfinite(vorticity)
    eligible = np.asarray(valid_mask, dtype=bool) & np.isfinite(vorticity)
    if reflectivity is not None:
        eligible &= (
            np.isfinite(reflectivity)
            & (reflectivity >= min_reflectivity_dbz)
        )

    # 水平差分在网格边缘采用单边格式，且邻域被截断，容易产生假极值。
    # 至少留出约 3 km 的完整水平邻域后再参与候选搜索。
    neighbourhood_cells = max(
        1, int(np.ceil(3.0 / max(grid.dx, grid.dy)))
    )
    if grid.nx > 2 * neighbourhood_cells:
        eligible[:neighbourhood_cells, :, :] = False
        eligible[-neighbourhood_cells:, :, :] = False
    if grid.ny > 2 * neighbourhood_cells:
        eligible[:, :neighbourhood_cells, :] = False
        eligible[:, -neighbourhood_cells:, :] = False

    # Northern-Hemisphere mesocyclones are cyclonic (positive vertical
    # vorticity).  Do not mix anticyclonic shear maxima through abs().
    vort_work = np.where(
        eligible[:, :, z_indices],
        vorticity[:, :, z_indices],
        -np.inf,
    )

    sep_cells = max(1, int(min_separation_km / max(grid.dx, grid.dy)))

    meso_positions = []
    for rank in range(max_count):
        if not np.isfinite(vort_work).any():
            break
        flat_idx = np.nanargmax(vort_work)
        ix, iy, iz_local = np.unravel_index(flat_idx, vort_work.shape)
        iz = z_indices[iz_local]

        cx = float(grid.x[ix])
        cy = float(grid.y[iy])
        cz = float(grid.z[iz])
        peak = float(vorticity[ix, iy, iz])

        # 【修复1】检查涡度是否达到中气旋最小阈值
        if peak < min_vorticity:
            print(f"  [检测终止] 剩余最大气旋涡度={peak:.4f} < "
                  f"阈值 {min_vorticity} /s，停止搜索")
            break

        # 在 3 km 邻域内检查垂直连续性，允许涡旋柱随高度倾斜。
        i0c, i1c = max(0, ix - neighbourhood_cells), min(grid.nx, ix + neighbourhood_cells + 1)
        j0c, j1c = max(0, iy - neighbourhood_cells), min(grid.ny, iy + neighbourhood_cells + 1)
        vertical_count = 0
        for z_check in z_indices:
            layer = np.where(
                eligible[i0c:i1c, j0c:j1c, z_check],
                vorticity[i0c:i1c, j0c:j1c, z_check],
                -np.inf,
            )
            if np.max(layer, initial=-np.inf) >= min_vorticity * 0.5:
                vertical_count += 1
        
        if vertical_count < min_vertical_layers:
            print(f"  [跳过 #{rank+1}] 垂直连续性不足: "
                  f"{vertical_count}层 < {min_vertical_layers}层阈值")
            # 屏蔽该区域继续搜索下一个
            i0 = max(0, ix - sep_cells)
            i1 = min(vort_work.shape[0], ix + sep_cells + 1)
            j0 = max(0, iy - sep_cells)
            j1 = min(vort_work.shape[1], iy + sep_cells + 1)
            vort_work[i0:i1, j0:j1, :] = -np.inf
            continue

        meso_positions.append((cx, cy, cz, peak))
        print(f"  #{rank+1}: ({cx:.0f}, {cy:.0f}, {cz:.2f}) km, "
              f"cyclonic vort={peak:.4f} 1/s, 垂直连续层数={vertical_count}")

        # 在周围屏蔽，避免重复检测
        i0 = max(0, ix - sep_cells)
        i1 = min(vort_work.shape[0], ix + sep_cells + 1)
        j0 = max(0, iy - sep_cells)
        j1 = min(vort_work.shape[1], iy + sep_cells + 1)
        vort_work[i0:i1, j0:j1, :] = -np.inf

        if not np.isfinite(vort_work).any() or np.nanmax(vort_work) < min_vorticity:
            print(f"  [检测完成] 剩余涡度均低于阈值")
            break

    # 返回 (x, y, z) 元组列表，供流线种子点生成使用
    return [(cx, cy, cz) for cx, cy, cz, _ in meso_positions]


def assess_retrieval_quality(data, max_w_abs_p99=15.0,
                             max_continuity_abs_p95=0.005):
    """Screen out numerically unstable wind fields before vortex rendering."""
    valid = (
        np.asarray(data["dual_doppler_mask"], dtype=bool)
        & np.isfinite(data["w"])
    )
    w_values = np.abs(data["w"][valid])
    continuity = data.get("continuity_residual")
    continuity_values = (
        np.abs(continuity[valid & np.isfinite(continuity)])
        if continuity is not None else np.array([])
    )
    w_p99 = float(np.percentile(w_values, 99)) if w_values.size else np.nan
    continuity_p95 = (
        float(np.percentile(continuity_values, 95))
        if continuity_values.size else np.nan
    )
    issues = []
    if not np.isfinite(w_p99) or w_p99 > max_w_abs_p99:
        issues.append(f"|w| P99={w_p99:.2f} m/s > {max_w_abs_p99:.1f}")
    if continuity is not None and (
        not np.isfinite(continuity_p95)
        or continuity_p95 > max_continuity_abs_p95
    ):
        issues.append(
            f"|continuity residual| P95={continuity_p95:.5f} s-1 "
            f"> {max_continuity_abs_p95:.3f}"
        )
    return {
        "ok": not issues,
        "w_abs_p99": w_p99,
        "continuity_abs_p95": continuity_p95,
        "issues": issues,
    }


# ══════════════════════════════════════════════════════════════
#  涡度约束风场增强（保真增强，非人造替换）
# ══════════════════════════════════════════════════════════════

def enhance_vortex_region(u, v, w, vorticity, meso_positions, grid,
                           rot_gain=1.0, w_gain=1.0, radius_km=8.0):
    """
    直接使用原始反演风场进行流线追踪，确保物理自洽性。
    
    保留函数结构以便未来添加物理合理的增强（如涡旋模型约束）。
    """
    return u.copy(), v.copy(), w.copy()


# ══════════════════════════════════════════════════════════════
#  主流程管线
# ══════════════════════════════════════════════════════════════

def build_visualization(data, output_path=None, show=True,
                        z_scale=3.0, tube_radius=0.12,
                        dem_cache_dir="data/dem_cache",
                        shapefile_cache_dir="data/shapefile_cache",
                        data_dir="data"):
    """
    构建包含四层底图叠加的三维可视化场景。

    可视化层次结构（从下到上）：
        Layer 0: 二维地图底面（cartopy灰度地形图作为纹理映射）
        Layer 1: 反射率CR CAPPI半透明叠加（彩色dBZ色标）
        Layer 2: 地理边界与城市名标注（县界、河流、地名标签）
        Layer 3: 中气旋标记与雷达站点（白色球体标识）
        3D层: 地面风矢量箭头 + 三维螺旋流线管
    """
    u = data["u"]
    v = data["v"]
    w = data["w"]
    vorticity = data["vorticity"]
    grid = data["grid"]
    ts1, ts2 = data["timestamps"]
    station1 = data.get("station1", "Z9317")
    station2 = data.get("station2", "Z9543")
    dual_mask = data.get("dual_doppler_mask", np.isfinite(vorticity))
    reflectivity_3d = data.get("reflectivity_3d")

    quality = assess_retrieval_quality(data)
    print(
        f"\n[Quality] |w| P99={quality['w_abs_p99']:.2f} m/s, "
        f"|continuity residual| P95={quality['continuity_abs_p95']:.5f} s-1"
    )
    if not quality["ok"]:
        print("  [质量拒绝] 反演场数值不稳定，不生成中气旋流线:")
        for issue in quality["issues"]:
            print(f"    - {issue}")
        return None

    # ── 步骤1: 检测中气旋中心 ──
    print("\n[Step 1] 检测中气旋中心 (双雷达有效区内的气旋式涡度柱)...")
    meso_positions = detect_mesocyclones(
        vorticity, grid, z_range=(1.0, 8.0),
        valid_mask=dual_mask,
        reflectivity=reflectivity_3d,
    )

    if not meso_positions:
        print("  [错误] 未检测到中气旋!")
        return None

    print(f"\n[Step 2] 涡度约束增强 + 流线追踪...")

    from src.streamline import trace_streamline_fast

    # 掩膜归一化高斯平滑：只使用双雷达有效点，避免无资料填充值
    # 向有效区扩散并制造流线。
    from scipy.ndimage import gaussian_filter
    wind_valid = (
        np.asarray(dual_mask, dtype=bool)
        & np.isfinite(u) & np.isfinite(v) & np.isfinite(w)
    )
    smooth_weight = gaussian_filter(wind_valid.astype(float), sigma=1.5)

    def _smooth_valid(field):
        numerator = gaussian_filter(
            np.where(wind_valid, field, 0.0), sigma=1.5, mode='nearest'
        )
        smoothed = numerator / np.maximum(smooth_weight, 1.0e-6)
        return np.where(wind_valid, smoothed, 0.0)

    u_sm = _smooth_valid(u)
    v_sm = _smooth_valid(v)
    w_sm = _smooth_valid(w)

    all_streamlines = []
    all_speeds = []
    accepted_meso_positions = []

    for rank, (cx, cy, cz) in enumerate(meso_positions):
        # --- 1. 计算该涡旋的局部平均背景风（风暴平移速度）---
        ix = np.argmin(np.abs(grid.x - cx))
        iy = np.argmin(np.abs(grid.y - cy))
        iz = np.argmin(np.abs(grid.z - cz))

        # 取半径 20km，高度 6km 以下区域计算平均背景风
        r_win_km = 20
        dx_win = max(1, int(r_win_km / grid.dx))
        dy_win = max(1, int(r_win_km / grid.dy))
        x_slice = slice(max(0, ix - dx_win), min(grid.nx, ix + dx_win))
        y_slice = slice(max(0, iy - dy_win), min(grid.ny, iy + dy_win))
        z_slice = slice(0, max(1, int(6.0 / grid.dz)))  # 仅取 6km 以下

        u_mean = np.nanmean(u[x_slice, y_slice, z_slice])
        v_mean = np.nanmean(v[x_slice, y_slice, z_slice])
        print(f"  涡旋 #{rank+1} 平移风: U={u_mean:.1f}, V={v_mean:.1f} m/s")

        # --- 2. 切向风一致性检查（使用相对风）---
        tan_vals = []
        for dxi in [-2, -1, 0, 1, 2]:
            for dyi in [-2, -1, 0, 1, 2]:
                xi, yi = ix + dxi, iy + dyi
                if 0 <= xi < grid.nx and 0 <= yi < grid.ny:
                    rx, ry = grid.x[xi] - cx, grid.y[yi] - cy
                    r = np.sqrt(rx**2 + ry**2)
                    if r > 0.3:
                        u_rel = u[xi, yi, iz] - u_mean
                        v_rel = v[xi, yi, iz] - v_mean
                        tan = (-u_rel * ry + v_rel * rx) / r
                        if np.isfinite(tan):
                            tan_vals.append(tan)

        tan_pos = sum(1 for t in tan_vals if t > 0) if tan_vals else 0
        tan_ratio = tan_pos / len(tan_vals) if tan_vals else 0

        print(f"\n  涡旋 #{rank+1} ({cx:.0f},{cy:.0f},{cz:.1f})km:")
        print(f"    风: u={u[ix,iy,iz]:.1f} v={v[ix,iy,iz]:.1f} m/s, "
              f"|vort|={abs(vorticity[ix,iy,iz]):.4f}")
        print(f"    切向一致性: {tan_pos}/{len(tan_vals)} ({100*tan_ratio:.0f}%) "
              f"{'[真涡旋]' if tan_ratio >= 0.40 else '[风切变带，跳过]'}")

         # 跳过非气旋式旋转的假涡旋
        if tan_ratio < 0.40:
            print(f"    [跳过] 切向风一致性不足，可能为风切变带而非真涡旋")
            continue

        # --- 3. 构造风暴相对风场（直接数组，供快速插值使用）---
        u_rel_arr = u_sm - u_mean
        v_rel_arr = v_sm - v_mean
        w_rel_arr = w_sm  # w 不减去背景（上升气流本身是局地特征）

        # --- 4. 判断涡旋类型：上升气流 / 下沉气流(RFD) ---
        iz_core_start = max(0, iz - 3)
        iz_core_end = min(grid.nz, iz + 4)
        w_core_vals = w_sm[ix, iy, iz_core_start:iz_core_end]
        w_core_mean = float(np.mean(w_core_vals))
        if w_core_mean > 0.3:
            vortex_type = 'updraft'
        elif w_core_mean < -0.3:
            vortex_type = 'downdraft'
        else:
            vortex_type = 'mixed'
        print(f"    涡旋类型: {vortex_type} (核心平均W={w_core_mean:.2f} m/s)")

        # --- 5. 种子高度：仅在 |W|>0.3 m/s 的实际活跃层播种 ---
        z_active = []
        for k in range(grid.nz):
            if wind_valid[ix, iy, k] and abs(w_sm[ix, iy, k]) > 0.3:
                z_active.append(grid.z[k])
        if len(z_active) >= 3:
            z_seeds = np.linspace(min(z_active), max(z_active),
                                  min(6, len(z_active)))
        else:
            print(f"    [跳过] 仅 {len(z_active)} 个垂直活跃层，不能支持三维螺旋流线")
            continue
        print(f"    种子高度: {z_seeds[0]:.1f} ~ {z_seeds[-1]:.1f} km "
              f"({len(z_seeds)}层, 活跃层={len(z_active)})")

        candidate_streamlines = []
        candidate_speeds = []
        rng = np.random.RandomState(42 + rank)
        for z_seed in z_seeds:
            # 种子同心环：避开涡核(r<3km旋转弱)，聚焦强旋转区(r=3~9km)
            for ring_i in range(5):
                r = 3.0 + ring_i * 1.5    # 3.0, 4.5, 6.0, 7.5, 9.0 km
                for s in range(8):
                    angle = 2 * np.pi * s / 8 + rng.uniform(-0.08, 0.08)
                    seed = (cx + r * np.cos(angle),
                            cy + r * np.sin(angle),
                            float(z_seed))
                    seed_ix = int(np.argmin(np.abs(grid.x - seed[0])))
                    seed_iy = int(np.argmin(np.abs(grid.y - seed[1])))
                    seed_iz = int(np.argmin(np.abs(grid.z - seed[2])))
                    if not wind_valid[seed_ix, seed_iy, seed_iz]:
                        continue

                    all_pts, speed_vals = trace_streamline_fast(
                        u_rel_arr, v_rel_arr, w_rel_arr,
                        grid.x, grid.y, grid.z,
                        seed,
                        step_size=0.15, max_steps=400,
                        direction='both', w_scale=1.0, speed_min=0.3,
                        return_speeds=True,
                    )
                    if len(all_pts) < 15:
                         continue

                    # 修剪：切除超出涡旋半径(15km)的水平发散段
                    dist_to_center = np.sqrt(
                        (all_pts[:, 0] - cx)**2 + (all_pts[:, 1] - cy)**2)
                    max_radius = 15.0  # km
                    within = dist_to_center <= max_radius
                    if not within.any():
                        continue
                    first = np.argmax(within)
                    last = len(within) - np.argmax(within[::-1]) - 1
                    all_pts = all_pts[first:last + 1]
                    speed_vals = speed_vals[first:last + 1]

                    if len(all_pts) < 15:
                        continue
                    z_span = all_pts[:, 2].max() - all_pts[:, 2].min()
                    if z_span < 0.5:
                        continue

                    candidate_streamlines.append(all_pts)
                    candidate_speeds.append(speed_vals)

        if candidate_streamlines:
            accepted_meso_positions.append((cx, cy, cz))
            all_streamlines.extend(candidate_streamlines)
            all_speeds.extend(candidate_speeds)
        else:
            print("    [跳过] 没有流线同时满足有效覆盖、长度和垂直跨度要求")

    print(f"\n  总计: {len(all_streamlines)} 条有效流线")
    if not accepted_meso_positions:
        print("  [错误] 所有中气旋候选均未通过环流与三维流线一致性检查")
        return None
    meso_positions = accepted_meso_positions

    # 保存涡度数据供等值面使用
    vort_data_3d = vorticity
    streamlines, speeds = all_streamlines, all_speeds

    # ── 步骤4: 构建底图 + 三维视角 ──
    print("\n[Step 4] 3D 可视化...")

    # 先加载 GIS 数据（用于底图渲染）
    from src.shapefile_loader import load_all_boundaries, load_city_labels
    def dem_querier(lat, lon):
        return 0.0
    print("  [GIS] 加载县界和城市名...")
    boundaries = load_all_boundaries(grid, shapefile_cache_dir, dem_querier)
    city_positions, city_names = load_city_labels(
        grid, shapefile_cache_dir, dem_querier)
    print(f"  县界: {len(boundaries)} 条, 城市: {len(city_names)} 个")

    vis = MesoVisualizer(grid, z_scale=z_scale, off_screen=not show)

    # --- Layer 0: 综合底图（cartopy + GIS边界 + 城市名 + 雷达站）---
    print("  [Layer 0] 生成综合底图纹理...")
    basemap_path = generate_basemap_texture(
        grid, boundary_lines=boundaries,
        # 城市名由 3D add_city_labels 负责 (避免与底图纹理中的城市名重复)
        city_positions=None, city_names=None,
        save_path="basemap_temp.png")
    vis.add_basemap_plane(basemap_path)

    # 县界已在底图纹理中渲染，不再重复添加三维线条

    # 添加三维城市标签（叠加在底图之上，更醒目）
    vis.add_city_labels(city_positions, city_names, z_offset=0.12)

    # 构造雷达站点信息（从集中式注册表获取）并添加三维小球和标签
    from src.stations import get_radar_site
    radar_sites = {
        station1: get_radar_site(station1),
        station2: get_radar_site(station2),
    }
    vis.add_radar_sites(radar_sites)
    
    # --- Layer 1: 反射率半透明叠加 ---
    print("  [Layer 1] 反射率...")
    from src.reflectivity_cappi import load_cr_reflectivity
    refl_2d = load_cr_reflectivity(
        grid, [station1, station2], [ts1, ts2], data_dir)
    vis.add_reflectivity_cappi(refl_2d)

    # --- 涡度等值面 ---
    print("  添加涡度等值面...")
    vis.add_vorticity_isosurface(vort_data_3d)

    # --- 中气旋标记（白色=我们的反演，洋红=雷达M产品）---
    vis.add_meso_markers(meso_positions)

    from src.cinrad_products import load_mesocyclones
    from src.gridder import wgs84_to_enu
    m_markers = []
    for stn, ts in [(station1, ts1), (station2, ts2)]:
        for m in load_mesocyclones(stn, ts, data_dir=data_dir):
            e, n, _ = wgs84_to_enu(
                np.array([m["lat"]]), np.array([m["lon"]]), np.array([0.0]),
                grid.origin_lat, grid.origin_lon)
            m_markers.append((float(np.asarray(e).item()),
                            float(np.asarray(n).item()),
                            m["height_m"]/1000.0))
    if m_markers:
        print(f"  M产品中气旋: {len(m_markers)} 个 (洋红色)")
        for i, (cx, cy, cz) in enumerate(meso_positions, start=1):
            distances = [
                (np.hypot(cx - mx, cy - my), abs(cz - mz))
                for mx, my, mz in m_markers
            ]
            horizontal_km, vertical_km = min(distances, key=lambda item: item[0])
            matched = horizontal_km <= 15.0 and vertical_km <= 3.0
            print(
                f"  反演候选 #{i} 与最近M产品: 水平 {horizontal_km:.1f} km, "
                f"垂直 {vertical_km:.1f} km -> "
                f"{'匹配' if matched else '未匹配'}"
            )
        for i, (mx, my, mz) in enumerate(m_markers):
            center = np.array([mx, my, mz * z_scale], dtype=np.float32)
            cube = pv.Cube(center=center, x_length=3, y_length=3, z_length=1.5)
            vis.plotter.add_mesh(cube, color='magenta', name=f'm_marker_{i}',
                                ambient=1.0, diffuse=0.3)

    # --- 三维流线管 ---
    if streamlines:
        print(f"  添加 {len(streamlines)} 条三维流线管（涡旋相对风场）...")
        vis.add_streamlines(streamlines, speeds, tube_radius=tube_radius)
    else:
        print("  [警告] 未生成有效流线")

    # 标题
    time_label = f"  {ts1} | {ts2}" if ts1 else ""
    vis.add_title(time_label)
    vis.set_camera(azimuth=-60, elevation=25)

    # ── 输出 ──
    if output_path:
        vis.save_screenshot(output_path)
        print(f"\n[保存截图] → {output_path}")

    if show:
        print("  打开交互窗口 (关闭窗口退出)...")
        vis.show()
    else:
        vis.close()

    print("\n完成.")
    return vis

# ══════════════════════════════════════════════════════════════
#  命令行接口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="三维中气旋风场可视化 (PyVista 交互式)")
    parser.add_argument("input", nargs="?", default=None,
                        help="wind3d_*.nc 文件路径")
    parser.add_argument("--output", "-o", default=None,
                        help="截图保存路径")
    parser.add_argument("--no-display", action="store_true",
                        help="不显示交互窗口 (仅保存截图)")
    parser.add_argument("--list", action="store_true",
                        help="列出可用的 wind3d 文件")
    parser.add_argument("--z-scale", type=float, default=1.5,
                        help="垂直夸张倍数 (默认 1.5)")
    parser.add_argument("--tube-radius", type=float, default=0.12,
                        help="流线管半径 (默认 0.12)")
    parser.add_argument("--dem-cache-dir", default="data/dem_cache",
                        help="SRTM DEM 缓存目录 (默认 data/dem_cache)")
    parser.add_argument("--shapefile-cache-dir", default="data/shapefile_cache",
                        help="Shapefile 缓存目录 (默认 data/shapefile_cache)")
    parser.add_argument("--data-dir", default="data",
                        help="CINRAD 原始数据目录 (默认 data)")
    parser.add_argument("--station1", default=None,
                        help="覆盖 nc 文件中的站点1 (雷达站号)")
    parser.add_argument("--station2", default=None,
                        help="覆盖 nc 文件中的站点2 (雷达站号)")
    args = parser.parse_args()

    output_dir = Path("output")

    if args.list:
        files = sorted(output_dir.glob("wind3d_*.nc"))
        if not files:
            print("未找到 wind3d_*.nc 文件")
            return 1
        for f in files:
            print(f.name)
        return 0

    # 选择输入文件：指定文件或最新修改的文件
    if args.input:
        filepath = Path(args.input)
    else:
        files = sorted(output_dir.glob("wind3d_*.nc"))
        if not files:
            print("未找到 wind3d_*.nc 文件。请先运行 run_retrieval.py")
            return 1
        # 选择最近修改的文件
        filepath = max(files, key=lambda f: f.stat().st_mtime)
        print(f"[自动选择] {filepath}")

    data = load_wind_data(filepath)
    # CLI 覆盖 nc 文件中的站点号
    if args.station1:
        data["station1"] = args.station1
    if args.station2:
        data["station2"] = args.station2
    visualization = build_visualization(
        data,
        output_path=args.output,
        show=not args.no_display,
        z_scale=args.z_scale,
        tube_radius=args.tube_radius,
        dem_cache_dir=args.dem_cache_dir,
        shapefile_cache_dir=args.shapefile_cache_dir,
        data_dir=args.data_dir,
    )
    return 0 if visualization is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
