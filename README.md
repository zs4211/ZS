# Radar 3D Retrieval — CINRAD 双多普勒雷达三维风场反演与中气旋可视化

基于 PyDDA 三维变分分析 (3DVAR) 的双多普勒雷达风场反演系统，支持从 CINRAD WRS RSTM 格式原始数据到三维交互式可视化的完整流程。

## 项目结构

```
radar-3d-retrieval/
├── run_retrieval.py            # [入口] 三维风场反演
├── visualize_3d.py             # [入口] 三维交互式可视化
├── requirements.txt            # 已验证环境的精确依赖版本
├── tests/                      # 解析风场与中气旋筛选闭环测试
├── README.md
├── analysis/
│   └── analyze_data_v2.py      # 数据探索分析工具
├── src/
│   ├── pipeline.py             # 数据流水线 (读取/QC/网格化)
│   ├── retrieval.py            # 3DVAR 风场反演 (含算法详解)
│   ├── dual_radar_retrieval.py # 双多普勒直接风场合成
│   ├── gridder.py              # WGS84↔ENU 坐标变换
│   ├── stations.py             # 雷达站点注册表
│   ├── cinrad_products.py      # CINRAD 二级产品读取
│   ├── reflectivity_cappi.py   # CR 组合反射率
│   ├── dem_loader.py           # SRTM 数字高程模型
│   ├── shapefile_loader.py     # GIS 行政区划/河流
│   └── streamline.py           # 三维流线追踪 (RK4)
├── data/                       # 原始雷达数据 (按站点组织)
└── output/                     # 反演结果输出 (.nc + .png)
```

## 环境安装

### 1. 安装 Python

使用 SciPy 优化引擎时支持 Python 3.9+；当前项目已在 Python 3.13 上完成端到端测试。Jax/TensorFlow 引擎不是必需依赖。

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

**常见问题：** 如果 `pip install pydda` 在 Windows 上失败（Jax 兼容性问题），可安装无 Jax 依赖的 PyDDA：

```bash
pip install pyart
pip install git+https://github.com/openradar/PyDDA.git
```

### 4. 验证安装

```bash
python -c "import pyart; import pydda; import cinrad; import pyvista; print('OK')"
```

## 数据准备

### 数据来源

本项目使用中国新一代天气雷达 (CINRAD) 的 **WRS RSTM 格式** 产品数据。数据文件为二进制格式，由雷达数据采集软件 (RPG/CP2) 生成，每个文件包含一个特定产品的一次扫描/体扫结果。

### 雷达数据目录结构

原始 CINRAD WRS RSTM 数据按以下结构放置：

```
data/
├── Z9317/                    # 站点1: 沧州
│   ├── PPI/                  # ★ 最核心: 体扫基数据
│   │   ├── 019/              # 反射率因子 Zc (高分辨率, 01-06层)
│   │   ├── 020/              # 反射率因子 Zc (高分辨率)
│   │   ├── 026/              # 径向速度 Vc (高分辨率) ★ 反演核心输入
│   │   ├── 027/              # 径向速度 Vc (高分辨率) ★ 反演核心输入
│   │   ├── 158/              # 差分反射率 ZDRc (双偏振)
│   │   ├── 160/              # 相关系数 RHO (双偏振)
│   │   ├── 161/              # 差分相位 PDP (双偏振)
│   │   ├── 162/              # 差分传播相位常数 KDP (双偏振)
│   │   ├── 19/               # 低分辨率反射率 (64/128/256字节头)
│   │   ├── 20/               # 低分辨率反射率
│   │   ├── 26/               # 低分辨率径向速度
│   │   └── 27/               # 低分辨率径向速度
│   ├── CR/                   # 组合反射率 (Combined Reflectivity)
│   │   ├── 037/              # 高分辨率 CR (1840×1840)
│   │   ├── 038/              # 高分辨率 CR
│   │   ├── 37/               # 低分辨率 CR
│   │   └── 38/               # 低分辨率 CR
│   ├── CAPPI/                # 等高平面位置显示
│   │   ├── 0110/             # 1.1km 高度层 CAPPI
│   │   └── 110/              # 1.1km 高度层 (低分辨率)
│   ├── M/                    # 中气旋产品 (Mesocyclone)
│   │   ├── 060/              # 高分辨率
│   │   └── 60/               # 低分辨率
│   ├── TVS/                  # 龙卷涡旋特征 (Tornado Vortex Signature)
│   ├── VWP/                  # VAD 风廓线 (Velocity Azimuth Display Wind Profile)
│   ├── SRM/                  # 风暴相对速度 (Storm Relative Motion)
│   ├── HCL/                  # 冰雹指数产品 (Hail)
│   ├── HI/                   # 混合扫描反射率 (Hybrid Scan Reflectivity)
│   ├── TOPS/                 # 回波顶高 (Echo Tops)
│   ├── VIL/                  # 垂直累积液态水 (Vertically Integrated Liquid)
│   ├── SS/                   # 风暴结构 (Storm Structure)
│   ├── STI/                  # 风暴追踪信息 (Storm Tracking Information)
│   ├── SHEAR/                # 风切变产品
│   ├── OHP/                  # 一小时降水量 (One-Hour Precipitation)
│   ├── THP/                  # 三小时降水量
│   ├── STP/                  # 风暴总降水量 (Storm Total Precipitation)
│   ├── ML/                   # 融化层产品 (Melting Layer)
│   ├── UAM/                  # 用户警报信息 (User Alert Message)
│   │   ├── hail/             # 冰雹警报
│   │   ├── heavyrain/        # 暴雨警报
│   │   ├── thunderstorm/     # 雷暴警报
│   │   └── tornado/          # 龙卷警报
│   └── WER/                  # 弱回波区 (Weak Echo Region)
├── Z9543/                    # 站点2: 滨州
│   └── (同上结构，部分产品可能缺失)
├── Z9532/                    # 站点3: 青岛
│   └── (同上)
├── Z9539/                    # 站点4: 临沂
│   └── (同上)
├── dem_cache/                # SRTM 高程瓦片 (程序自动下载缓存)
└── shapefile_cache/          # GADM 中国县级行政区划 (程序自动下载缓存)
```

### 文件命名规范

CINRAD WRS RSTM 文件命名格式:

```
<站号>_<时间戳>_<产品>_<层号>_<仰角/产品代码>

例: Z9317_20240613102400Z_PPI_01_026
     │      │              │   │   └─ 仰角代码: 026 = 速度 (高分辨率)
     │      │              │   └─ 层号: 01 (该仰角下的第1层)
     │      │              └─ 产品: PPI (平面位置显示)
     │      └─ 时间戳: 2024年06月13日 10:24:00 UTC
     └─ 雷达站号: Z9317 (沧州)
```

- **时间戳格式**: `YYYYMMDDHHMMSSZ`，末尾 `Z` 表示 UTC 时间
- **层号**: PPI 产品中 `01`~`06` 分别对应 6 个不同的实际仰角扫描
- **仰角代码**: 三位数 = 高分辨率，两位数 = 低分辨率

### 数据产品详细说明

#### ★ 风场反演必需的核心数据

| 产品 | 代码 | 目录 | 用途 | 典型文件大小 |
|------|------|------|------|-------------|
| **径向速度** (velocity) | 026, 027 | `PPI/026/`, `PPI/027/` | 3DVAR 风分量约束的核心输入，反演必需 | ~350 KB/文件 |
| **VAD 风廓线** (VWP) | 048 | `VWP/048/` | 构建 3DVAR 初猜场的背景风廓线 | ~10 KB/文件 |

**为什么需要 026 和 027 两个目录？**
当前数据中的 026 和 027 含互补及部分重复的速度仰角扫描。程序按产品头中的实际仰角合并它们；例如默认时次得到 9 个 sweep、6 个实际仰角，而不是把目录编号当成仰角。

#### 可视化辅助数据

| 产品 | 代码 | 目录 | 用途 |
|------|------|------|------|
| **组合反射率** (CR) | 037, 038 | `CR/037/` | visualize_3d.py 中的半透明彩色叠加层，展示对流核心水平结构 |
| **中气旋** (M) | 060 | `M/060/` | 雷达自带算法的中气旋检测结果，用于验证反演涡度检测精度 |
| **龙卷涡旋** (TVS) | 061 | `TVS/061/` | 雷达到的龙卷涡旋特征 |

#### 其他可用的二级产品

| 产品 | 代码 | 说明 |
|------|------|------|
| **CAPPI** | 0110, 110 | 等高平面位置显示，1.1km 高度层的反射率水平切片 |
| **SRM** | 056 | 风暴相对速度（已减去风暴平移速度），直接展示旋转偶极子 |
| **TOPS** | 041 | 回波顶高，可用于设置质量连续积分的上边界 |
| **VIL** | 057 | 垂直累积液态水含量，辅助识别强上升气流区 |
| **HI** | 059 | 混合扫描反射率，取各仰角在最低可用高度处的反射率拼合 |
| **HCL** | 164 | 冰雹指数（三层：冰雹概率/严重冰雹概率/最大冰雹尺寸） |
| **SHEAR** | 087 | 风切变产品，辅助识别中气旋 |
| **SS** | 062 | 风暴结构（风暴 ID、位置、移动方向/速度） |
| **STI** | 058 | 风暴追踪信息 |
| **OHP** | 078, 169 | 过去一小时累积降水量 (mm) |
| **THP** | 079, 170 | 过去三小时累积降水量 |
| **STP** | 080, 171 | 风暴过程总降水量 |
| **ML** | 166 | 融化层高度和厚度 |
| **WER** | 053 | 弱回波区检测（超级单体的经典特征） |
| **UAM** | - | 用户警报信息（冰雹/暴雨/雷暴/龙卷四类警报） |

### 数据需求清单

**最小化运行所需:**
```
data/
├── <站点A>/PPI/026/*        # 径向速度，至少一个时次的所有层
├── <站点A>/PPI/027/*        # 径向速度
├── <站点A>/PPI/019/*        # 同时次反射率，用于下落速度订正
├── <站点A>/PPI/020/*        # 同时次反射率
├── <站点B>/PPI/026/*        # 径向速度
├── <站点B>/PPI/027/*        # 径向速度
├── <站点B>/PPI/019/*        # 同时次反射率
└── <站点B>/PPI/020/*        # 同时次反射率
```

**完整可视化体验所需 (额外):**
```
data/<站点>/CR/037/*          # 组合反射率
data/<站点>/M/060/*           # 中气旋产品
data/<站点>/VWP/048/*         # VAD风廓线
```

**无需手动准备的（自动下载）:**
```
data/dem_cache/               # SRTM 高程瓦片 (~25MB)
data/shapefile_cache/         # GADM 中国县级边界 (~50MB)
```

### PPI 体扫数据的关键信息

每个 PPI 文件 (如 `*_PPI_01_026`) 包含:
- 一个完整 360° 扫描 (~360 根径向)
- 每根径向的距离库数依产品而异：当前速度产品约 460 个、反射率产品约 920 个
- 数据值: 径向速度 (m/s)，正值为远离雷达，负值为朝向雷达
- 元数据: 仰角、方位角数组、Nyquist 速度、VCP 模式

当前 PPI 产品子集覆盖 6 个实际低仰角（约 0.5°–5.2°）；默认时次合并 026/027 后共有 9 个 sweep。文件时间戳是产品标称时刻，项目无法从该 PUP 产品恢复每根径向的独立采样时间。

### 已注册雷达站点

| 站号 | 站名 | 纬度 | 经度 | 海拔 |
|------|------|------|------|------|
| Z9317 | 沧州 | 38.279°N | 116.805°E | 9m |
| Z9543 | 滨州 | 37.383°N | 117.933°E | 18m |
| Z9532 | 青岛 | 35.989°N | 120.230°E | 92m |
| Z9539 | 临沂 | 35.250°N | 118.421°E | 224m |

如需添加新站点，编辑 `src/stations.py` 中的 `STATIONS` 字典，或直接将数据放入 `data/<站号>/` 目录（程序会自动从产品文件头中提取坐标）。

### 双多普勒反演的站点配对限制

> **重要：** 当前数据集中各站点的数据来自不同的天气过程，时间不重叠。
> 只有以下站点对可以进行双多普勒风场反演：

| 可用配对 | 站点1 | 站点2 | 数据时段 | 基线距离 |
|---------|-------|-------|---------|---------|
| **配对 A** | Z9317 (沧州) | Z9543 (滨州) | 2024-06-13 10:00–12:00 UTC | ~141 km |
| **配对 B** | Z9539 (临沂) | Z9532 (青岛) | 2026-05-11 | ~183 km |


**如何切换配对：**

```bash
# 配对 A: 沧州 + 滨州 (默认)
python run_retrieval.py --station1 Z9317 --station2 Z9543

# 配对 B: 临沂 + 青岛
python run_retrieval.py --station1 Z9539 --station2 Z9532
```

程序通过时间戳匹配 (`--list-matches`) 可自动验证两站数据是否有重叠时段。
若使用不可配对的站点组合，会输出 `No matching timestamps found!`。

## 运行步骤

### 步骤 1: 风场反演 (`run_retrieval.py`)

#### 1.1 列出可用的时间匹配对

```bash
python run_retrieval.py --list-matches
```

输出示例:
```
Found 12 matching time pairs:
  Z9317=20240613102400Z <-> Z9543=20240613102428Z (diff=0.5 min)
  Z9317=20240613103000Z <-> Z9543=20240613102947Z (diff=0.2 min)
  ...
```

两部雷达的扫描时间差默认必须在 **2 分钟以内**，以减小强对流快速演变造成的非同时观测误差。可用 `--max-time-diff` 显式调整；手工指定的时次也会执行同一检查。

#### 1.2 处理单个时间对（自动选择最佳匹配）

```bash
# 自动选择时间差最小的配对
python run_retrieval.py
```

#### 1.3 指定时间对

```bash
python run_retrieval.py \
  --timestamp1 20240613103000Z \
  --timestamp2 20240613102947Z
```

#### 1.4 批量处理所有匹配对

```bash
python run_retrieval.py --match-all
```

#### 1.5 自定义网格和优化参数

```bash
python run_retrieval.py \
  --grid-nz 20 --grid-ny 201 --grid-nx 201 \
  --grid-zmin 500 --grid-zmax 15000 --grid-xymax 150000 \
  --Co 100 --Cb 0.01 --Cm 1000 \
  --max-iter 150 --wind-tol 0.5
```

### 步骤 2: 三维可视化 (`visualize_3d.py`)

#### 2.1 自动选择最新结果

```bash
python visualize_3d.py
```

#### 2.2 指定文件

```bash
python visualize_3d.py output/wind3d_20240613103000Z_20240613102947Z.nc
```

#### 2.3 列出可用文件

```bash
python visualize_3d.py --list
```

#### 2.4 离屏渲染（无 GUI 服务器/远程环境）

```bash
python visualize_3d.py --no-display --output result.png
```

#### 2.5 自定义视觉效果

```bash
python visualize_3d.py \
  --z-scale 3.0 \        # 垂直夸张倍数 (默认 1.5)
  --tube-radius 0.15      # 流线管半径 (默认 0.12)
```

## 完整参数参考

### `run_retrieval.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--station1` | Z9317 | 第一部雷达站号 |
| `--station2` | Z9543 | 第二部雷达站号 |
| `--timestamp1` | None | 站点1时间戳 (YYYYMMDDHHMMSSZ) |
| `--timestamp2` | None | 站点2时间戳 |
| `--product` | velocity | 反演观测量；仅允许 velocity，反射率会自动读取 |
| `--resolution` | high | 分辨率: high / low |
| `--data-dir` | data | 原始数据根目录 |
| `--output-dir` | output | 结果输出目录 |
| `--max-time-diff` | 2 | 两站体扫最大允许时差 (分钟) |
| `--grid-nz` | 20 | 垂直层数 |
| `--grid-ny` | 201 | Y方向格点数 |
| `--grid-nx` | 201 | X方向格点数 |
| `--grid-zmin` | 500 | 网格底部高度 (m) |
| `--grid-zmax` | 15000 | 网格顶部高度 (m) |
| `--grid-xymax` | 150000 | 水平范围半径 (m) |
| `--Co` | 100 | 观测项权重 (Jo) |
| `--Cb` | 0.01 | 背景项权重 (Jb) |
| `--Cm` | 1000 | 质量连续项权重 (Jm) |
| `--Cx`/`--Cy`/`--Cz` | 1e-4 | 平滑项权重 (Js) |
| `--max-iter` | 150 | L-BFGS-B 最大迭代次数 |
| `--wind-tol` | 0.5 | 收敛容差 (m/s) |
| `--min-cross-angle` | 20 | 最小交叉波束角 (度) |
| `--filter-window` | 5 | Savitzky-Golay 低通滤波窗口 (格点数) |
| `--filter-order` | 3 | 低通滤波阶数 |
| `--list-matches` | - | 列出时间匹配对 |
| `--match-all` | - | 处理所有匹配对 |
| `--no-plot` | - | 不生成诊断图 |

### `visualize_3d.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input` | 最新文件 | wind3d_*.nc 文件路径 |
| `--output` / `-o` | None | 截图保存路径 |
| `--no-display` | - | 离屏渲染，不显示窗口 |
| `--list` | - | 列出可用的 wind3d 文件 |
| `--z-scale` | 1.5 | 垂直夸张倍数 |
| `--tube-radius` | 0.12 | 流线管半径 |

## 参数调优指南

### 权重参数的物理含义

3DVAR 代价函数由四项组成: `J = Jo + Jb + Jm + Js`。每一项的权重控制该项在总代价中的相对重要性。详见 `src/retrieval.py` 模块文档。

### 针对不同场景的推荐配置

#### 场景 A: 中气旋精细分析（默认配置）

观测数据质量好、双雷达覆盖充分时，优先保留中尺度结构。

```bash
--Co 100 --Cb 0.01 --Cm 1000 --Cx 1e-4 --Cy 1e-4 --Cz 1e-4 --max-iter 150
```

- **Co 大**: 充分信任雷达观测，反演风场紧密贴合径向速度数据
- **Cb 小**: 初猜场仅起引导作用，允许双多普勒修正充分体现
- **Cm 大**: 严格满足质量连续，确保 w 的可靠性
- **Cx/Cy/Cz 小**: 弱平滑，保留中气旋的尖锐涡度梯度

#### 场景 B: 大尺度背景风场分析

关注天气尺度系统（锋面、低空急流等），要求风场平滑稳定。

```bash
--Co 10 --Cb 1.0 --Cm 256 --Cx 1e-2 --Cy 1e-2 --Cz 1e-2 --max-iter 80
```

#### 场景 C: 双雷达覆盖不足时的保守模式

两部雷达波束重叠区域小，大部分网格点缺乏双多普勒约束。

```bash
--Co 1 --Cb 1.0 --Cm 100 --Cx 1e-1 --Cy 1e-1 --Cz 1e-1 --max-iter 50
```

- **Cb 大**: 在缺乏观测的区域紧密跟随 VWP 背景场
- **Cx/Cy/Cz 大**: 强平滑抑制无数据区的随机波动

#### 场景 D: 快速测试/预览

快速检查数据质量和时间匹配，不需要精细结果。

```bash
--grid-nz 15 --grid-ny 101 --grid-nx 101 --max-iter 30
```

- 粗网格降低计算量（格点数从 80 万降至 15 万）
- 每次运行约 2-5 分钟

### 调优策略

1. **先粗后细**: 用粗网格和少迭代次数快速验证数据可用性，再细化
2. **先观测后约束**: 用较大的 Co 确保数据合理被拟合，再逐步增加平滑
3. **诊断关键指标**: 查看输出日志中的 `w: [min, max] m/s`，若 w 超过 ±15 m/s 说明质量连续约束不足（增大 Cm）或平滑不够（增大 Cx/Cy/Cz）
4. **网格间距匹配**: 默认 Δx≈1.5 km 时使用 `filter_window=5`。实测窗口 15 会明显压低局地涡度并漏检中气旋尺度结构；增大窗口前必须做敏感性试验。

## 输出文件

### 反演结果 (output/)

| 文件 | 格式 | 内容 |
|------|------|------|
| `wind3d_<ts1>_<ts2>.nc` | PyART Grid NetCDF | 风场、观测覆盖、反射率与质量诊断 |
| `wind3d_<ts1>_<ts2>.png` | PNG | 水平风速 + 涡度的诊断图 |

NetCDF 文件中的变量:

| 变量 | 单位 | 形状 | 说明 |
|------|------|------|------|
| `u` | m/s | (nz, ny, nx) | 东西向风分量 (向东为正) |
| `v` | m/s | (nz, ny, nx) | 南北向风分量 (向北为正) |
| `w` | m/s | (nz, ny, nx) | 垂直速度 (向上为正) |
| `wind_speed` | m/s | (nz, ny, nx) | 水平风速 √(u²+v²) |
| `divergence` | s⁻¹ | (nz, ny, nx) | 水平散度 ∂u/∂x+∂v/∂y |
| `vorticity` | s⁻¹ | (nz, ny, nx) | 相对涡度 ∂v/∂x−∂u/∂y |
| `reflectivity` | dBZ | (nz, ny, nx) | 两部雷达实测网格反射率最大值 |
| `radar_count` | 1 | (nz, ny, nx) | 每个格点的有效雷达数量 |
| `dual_doppler_mask` | 1 | (nz, ny, nx) | 双雷达观测及交叉角均有效的掩膜 |
| `cross_beam_angle` | degree | (nz, ny, nx) | 两部雷达水平波束锐夹角 |
| `radial_velocity_residual_radar1/2` | m/s | (nz, ny, nx) | 观测减去反演回代径向速度 |
| `continuity_residual` | s⁻¹ | (nz, ny, nx) | anelastic 质量连续方程残差 |

`w` 是由径向速度、下落速度订正和质量连续约束共同得到的估计量，不是雷达直接观测值。中气旋流线只在满足双雷达覆盖、反射率、涡度和垂直连续性条件时生成。

### 科学适用范围与限制

- 双多普勒只能在两站同时有有效径向速度、且交叉波束角合格的区域可靠约束水平风；`dual_doppler_mask` 之外不能用于中气旋定量判断。
- PUP 文件只提供产品标称时刻，不能恢复每根径向的独立采样时刻。程序默认把站间标称时差限制在 2 分钟，但没有做风暴平流时间订正。
- 垂直速度 `w` 是受边界条件、VWP 背景场、下落速度参数化及连续方程共同影响的间接估计，应结合 `continuity_residual` 和径向速度残差判断。
- PyDDA 的 SciPy 接口不返回可持久化的正式收敛标志；输出中的 `solver_status` 只表示求解过程执行完毕，不能替代残差检查和独立观测验证。
- 当前默认样例时次（Z9317 10:30:00Z + Z9543 10:29:47Z）的 4 个业务 M 产品中心均不在双雷达有效柱内，因此本项目会保守地不生成中气旋螺旋流线。要验证完整中气旋目标，需要目标风暴位于两站共同覆盖区内的同步体扫案例。
- M 产品与本项目不是同一检测算法：M 基于单雷达径向速度旋转/切变特征，本项目基于双雷达反演风场的垂直涡度柱。两者应作为相互独立的证据，并同时报告空间距离和双雷达几何有效性，不能把 M 当作强制拟合目标。
- 中气旋候选默认要求峰值格点反射率不低于 30 dBZ；20 dBZ 在当前案例中会接受两个仅有 20–25 dBZ 的弱回波旋转假阳性。
- 可视化前执行数值质量门控：双雷达有效区 `|w|` 的 99 分位不得超过 15 m/s，连续方程残差绝对值的 95 分位不得超过 0.005 s⁻¹。失败结果只保留作敏感性分析，不生成中气旋流线。
- 急性交叉角阈值 θ 会对称传给 PyDDA 为 `[θ, 180°−θ]`。Z9539/Z9532 的 07:09 时次中，两个强回波 M 中心交叉角仅 14°–15°；10°敏感性试验虽在 M 附近恢复出旋转，却产生约 ±41 m/s 的 `w` 并未通过质量门控，因此不能作为可信三维风场。
- 本项目目前适合研究与可视化验证，不应直接作为业务告警或定量风害结论。

### 运行自检

```bash
python -m unittest discover -v
```

### 可视化输出

可视化脚本会渲染交互式 PyVista 窗口，包含：
- **底图**: cartopy 自然地理要素 + GADM 县级行政边界 + 城市名
- **DEM 地形**: SRTM 高程 (自动下载)
- **反射率叠加**: 组合反射率 CR 的半透明彩色图层
- **雷达站点**: 白色球体 + 站名标注
- **涡度等值面**: 红-橙-黄三层半透明等值面
- **中气旋标记**: 白色球体 (反演检测) + 洋红色方块 (雷达 M 产品)
- **三维流线管**: turbo 色标的风速映射螺旋流线

## 故障排查

### 1. `No matching timestamps found!`

两部雷达没有在 `--max-time-diff`（默认 2 分钟）容差内的时间匹配对。检查:
- 数据日期是否一致
- PPI/026 和 PPI/027 目录中是否有数据文件
- 产品类型参数是否匹配 (`--product velocity --resolution high`)

### 2. `WARNING: failed to read <filename>`

PyCINRAD 无法解析该文件。可能原因:
- 文件损坏（检查文件大小，完整文件通常 > 300KB）
- RSTM 格式版本不兼容

### 3. 反演结果中 w 的值异常大 (±50 m/s)

质量连续约束不足。增大 `--Cm` 并增大平滑权重 `--Cx --Cy --Cz`。

### 4. PyVista 窗口卡住或无响应

三维流线计算可能耗时。先用 `--no-display --output test.png` 试跑一次生成截图。

### 5. `ImportError: No module named 'pydda'`

PyDDA 未正确安装。尝试:
```bash
pip install git+https://github.com/openradar/PyDDA.git
```

### 6. DEM/GIS 数据下载失败

防火墙或网络问题。可以:
- 手动下载 SRTM 瓦片放入 `data/dem_cache/`
- 手动下载 GADM 中国数据放入 `data/shapefile_cache/`
- 或在 `visualize_3d.py` 中跳过 DEM 层

## 引用

- **PyDDA**: Jackson et al. (2020) — PyDDA: A Pythonic Direct Data Assimilation framework for wind retrievals. *Journal of Open Source Software*
- **3DVAR 算法**: Potvin et al. (2012) — 3DVAR vs. traditional dual-Doppler wind retrievals. *Monthly Weather Review*
- **CINRAD 数据格式**: PyCINRAD 项目 — https://github.com/CyanideCN/PyCINRAD
- **GADM 地理边界**: https://gadm.org
- **SRTM 高程数据**: https://srtm.kurviger.de
