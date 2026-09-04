"""
radar-3d-retrieval: 基于 PyDDA/三维变分分析的 CINRAD 双多普勒雷达三维风场反演系统。

项目结构:
  run_retrieval.py   — 主入口：双多普勒三维风场反演
  visualize_3d.py    — 主入口：三维中气旋风场交互式可视化

  src/
  ├── pipeline.py              — 数据流水线（读取 CINRAD RSTM 格式，QC，网格化）
  ├── retrieval.py             — 三维变分分析风场反演（PyDDA 3DVAR）
  ├── dual_radar_retrieval.py  — 双多普勒直接风场合成（初始猜测场）
  ├── gridder.py               — WGS84↔ENU 坐标变换 + 分析网格构建
  ├── stations.py              — 雷达站点集中式注册表
  ├── cinrad_products.py       — CINRAD 二级产品读取（M/TVS/VWP/SRM 等）
  ├── reflectivity_cappi.py    — CR 组合反射率加载与插值
  ├── dem_loader.py            — SRTM 数字高程模型加载
  ├── shapefile_loader.py      — GADM 行政区划 + Natural Earth 河流加载
  ├── streamline.py            — 三维流线追踪（RK4 积分 + 中气旋检测）
  └── visualize/               — 三维可视化子模块（待拆分）

处理流程:
  1. pipeline.py: 通过 PyCINRAD (read_auto) 读取 CINRAD WRS RSTM 格式数据
  2. pipeline.py: 转换为 PyART Radar 对象并进行质量控制（退模糊等）
  3. retrieval.py: 将每部雷达数据网格化到统一的笛卡尔网格
  4. retrieval.py: 运行 PyDDA 三维变分分析，反演 u, v, w 风场分量
  5. visualize_3d.py: 加载反演结果，检测中气旋，生成三维交互式可视化
"""

from .pipeline import RadarDataPipeline
from .retrieval import WindRetrieval

__version__ = "0.2.0"
