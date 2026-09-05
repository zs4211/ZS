# Step 5：低秩 Tucker 可行性诊断

结论：**科学黄灯**。当前统一 PyDDA 基线在共同可靠的小型连续子域上显示稳定的空间/高度低秩信号，并存在明显低于原始自由度、同时保持三分量的秩；但证据域很小，强时间压缩会显著损害 `w` 和局地涡度，完整结构缺测域没有真实缺失风场可供验证。因此结果足以支持下一步做小网格 OSSE/最小 Tucker 原型，但不足以直接支持全局实况 Tucker 或确定最终秩。

## 1. 冻结输入与统一输出

- 六个连续匹配体扫全部使用冻结配置重跑：`20×201×201`，`Co=100`、`Cb=0.01`、`Cm=1000`、`Cx=Cy=Cz=1e-4`、`wind_tol=0.5`、`max_iterations=150`、滤波窗口 `15`/阶数 `3`、BCA 门槛 `20°`、SciPy engine。
- 未修改 PyDDA、QC、滤波、BCA 门槛、重复 sweep 或反演参数。
- 每个 NetCDF 都包含同名 `u/v/w`、两部雷达 observation mask、dual-radar mask、BCA mask、finite-wind mask、Step 3 physics mask，以及输入时次、配置和冻结代码 commit。
- 六次质量门均通过。physics-mask 点数依次为 `3834, 4265, 4602, 5184, 5993, 6620`；两部雷达径向速度 RMSE 范围为 `3.50–4.16 m/s`。
- 旧的“全域有限风场”未参与本实验。

## 2. 诊断设计

- 全部窗口：K=3 四个、K=4 三个、K=5 两个，共 9 个。
- 张量顺序固定为 `(x,y,z,time,component)`，且 `rc=3`。
- A 类采用 Eulerian 与 storm-relative 都完整有效的公平共同 mask，再取最大轴对齐全有效长方体；每时次只有 `336–726` 个格点，范围约 `9–28.5 km`（水平）和 `2.29–3.82 km`（垂直）。这是本结论最重要的外推限制。
- B 类采用 physics-mask 并集包围盒上的缺测感知迭代 ST-HOSVD；缺失值以分量观测均值初始化，不计算或解释零填充谱。原始可用比例为 Eulerian `13.1%–18.5%`、storm-relative `10.1%–15.2%`。
- B 类固定留出约 10% 的可用空间—时间格点，三个风分量同时留出；仅把留出误差解释为已有观测上的 representation diagnostic，不能解释为真实缺失区恢复精度。
- storm-relative 位移直接使用冻结的 Step 4 NCC 结果，没有重新估计或调参。
- 垂直涡度使用 Step 3 相同的显式差分规则及完整 physics stencil；没有 NaN→0→gradient。计算性填充值只出现在完整有效 stencil 之外，输出指标始终由显式 mask 限定。

预注册秩规则包含 aggressive、compact、moderate 三档，以及 `rt=min(2/3,K)` 和 `rt=K` 对照；所有结果均记录 Tucker 参数量、参数占比和压缩倍数。

## 3. 奇异值谱

共同可靠子域上，各 mode 第一奇异值能量的九窗口统计为：

| 表示 | x | y | z | time | component |
|---|---:|---:|---:|---:|---:|
| Eulerian 均值（范围） | 82.9% (65.4–93.7) | 84.3% (59.5–96.2) | 80.4% (54.0–95.2) | 80.7% (55.7–93.5) | 83.1% (69.5–94.7) |
| Storm-relative 均值（范围） | 87.1% (76.4–95.9) | 87.5% (65.5–98.0) | 85.7% (63.3–97.6) | 93.5% (83.4–97.4) | 86.4% (77.0–97.0) |

99% 能量秩跨窗口范围：Eulerian 的 `x/y/z/time/component` 分别为 `2–3 / 2–3 / 2 / 2–4 / 3`；storm-relative 为 `2–3 / 2–3 / 2 / 2–3 / 2–3`。因此局部 x/y/z 谱存在一致衰减，time 谱的稳定性较弱，component mode 不能安全假定低于 3。

## 4. 截断重建的主要权衡

下表为 A 类九窗口均值；误差和保持率均相对各自表示下的冻结风场。

| 表示/秩规则 | 压缩倍数 | Fro 误差 | NRMSE(w) | 峰值涡度保持 | 正涡度环流保持 |
|---|---:|---:|---:|---:|---:|
| Eulerian compact temporal | 23.3× | 16.1% | 36.8% | 76.5% | 85.8% |
| Storm-relative compact temporal | 23.3× | 11.3% | 29.1% | 85.9% | 84.1% |
| Eulerian compact `rt=K` | 15.4× | 8.7% | 16.2% | 82.2% | 90.2% |
| Storm-relative compact `rt=K` | 15.4× | 6.9% | 16.6% | 88.5% | 86.4% |
| Eulerian moderate temporal | 4.8× | 4.2% | 10.7% | 98.2% | 99.3% |
| Storm-relative moderate temporal | 4.8× | 3.3% | 10.1% | 94.8% | 99.3% |
| Eulerian moderate `rt=K` | 3.9× | 1.2% | 3.0% | 99.0% | 99.7% |
| Storm-relative moderate `rt=K` | 3.9× | 1.1% | 2.7% | 98.2% | 99.0% |

关键负结果：K=5 时 compact temporal 的平均 `w` NRMSE 达到 Eulerian `60.4%`、storm-relative `38.8%`；改为 `rt=K` 后分别降为 `17.5%` 和 `17.3%`。因此第一版不应强压时间模。

compact 截断在个别窗口把峰值涡度压到参考值的 `48.2%`（Eulerian）或 `61.7%`（storm-relative），中心最大偏移分别达到约 `9.16 km` 和 `4.50 km`。moderate 档总体保持较好，但压缩只有约 4–6 倍。最大风速通常仍接近 100%，说明只看最大风速会漏掉局地梯度损伤。

涡旋柱连续性指标在这些很浅、很小的共同子域上几乎饱和，不能作为本轮的有效支持证据。

## 5. 结构缺测域

B 类最可信的固定 `compact rt=K` 留出结果为：

| 表示 | 留出相对误差 | RMSE(u/v/w) | 参数/拟合标量样本 |
|---|---:|---:|---:|
| Eulerian | 7.1% | 0.918 / 0.730 / 0.172 m/s | 0.316 |
| Storm-relative | 7.4% | 0.965 / 0.739 / 0.177 m/s | 0.397 |

moderate 模型的参数量超过拟合标量样本数（平均约 `1.28–1.97` 倍），并出现 `27.8%–36.5%` 的留出误差，属于过参数化，不能用其很低的训练点误差证明缺测恢复能力。

## 6. 七个问题的直接回答

1. **x/y/z/time 是否稳定衰减？** x/y/z 在九个小型共同子域上稳定衰减；z 的 99% 能量秩始终为 2。time 的衰减较窗口敏感，Eulerian 99% 秩为 2–4，证据较弱。
2. **是否有明显压缩且三分量保持良好的秩？** 有。`compact rt=K` 提供约 15.4× 压缩、6.9%–8.7% 总误差和约 16% 的 w NRMSE；若要求更保守，moderate 档约 3.9× 压缩且 w NRMSE 约 3%。
3. **是否系统损害 w？** 强压时间模时是，且 K 越大越明显；`rt=K` 或 moderate 秩可显著缓解，所以损害不是 Tucker 空间低秩的必然后果。
4. **是否削弱涡度、环流或移动涡旋？** aggressive/compact 秩会，且个别窗口很严重；moderate 秩总体能保持，但共同子域过小，中心和涡旋柱结论仍不稳健。
5. **storm-relative 是否更优？** 在 A 类公平子域的所有 45 个“窗口×秩”组合中，Frobenius 重建误差都更低，time rank-1 能量均值从 80.7% 升至 93.5%；但 B 类 `compact rt=K` 留出误差反而从 7.1% 升至 7.4%，且 Step 4 的时间不一致并未系统降低。双线性平移本身也带有平滑效应。因此只能报告“局部表示上数值更低秩”，不能声称 motion correction 已获得稳健科学收益。
6. **令 `rt=K` 后空间/高度低秩是否成立？** 成立，而且是当前最稳健的可行性信号；它显著改善 w，同时仍有约 14–19× 的 compact 压缩。
7. **绿/黄/红？** **黄灯**。低秩信号真实存在于当前诊断域，但域太小、基线受强平滑影响、缺测真值不存在、compact 秩仍损伤局地涡旋；进入下一步只能从小网格 OSSE 和保守 `rt=K` 原型开始，不能直接扩展为全局实况 P-RCG。

## 7. 输出

- `uniform_baseline/manifest.json`：六次统一反演配置、时次、mask 数量、RMSE。
- `diagnostics/step5_summary.json`：窗口域、代码与输入 SHA-256、预注册秩规则和缺测语义。
- `diagnostics/mode_spectra.csv`：五个 mode 的完整奇异值、累计能量、effective rank、90/95/99% 能量秩。
- `diagnostics/rank_reconstruction_metrics.csv`：A/B 两类域、全部秩、三分量误差和涡旋保持指标。
- `diagnostics/height_rmse.csv`：每高度、每分量、每窗口、每秩误差。
- `diagnostics/mode_rank1_energy.png`、`compression_tradeoff.png`：谱衰减与压缩—误差—涡度权衡图。
