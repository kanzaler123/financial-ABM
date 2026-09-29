# 可审计人工金融市场 ABM

这是一个面向研究用途的人工金融市场（Agent-Based Model，ABM）项目。
当前包含一阶段市场基准，以及二阶段的信息图、投资者 Bandit 和企业表格 Q-learning。
一阶段配置默认关闭 Logit；二阶段保持偏好和类型不变，只学习行动策略。

> **2026-09-26：Stage 1 正式验收 22/22 PASS；Stage 2 合成机制与工程验收通过。**
> 一阶段第六轮开发 320 次、全新 holdout 160 次、正式 800 次均为 22/22；
> 正式样本为 50 个新种子 × 16 场景，每条 1000 日预热 + 2500 日统计期。
> [正式决策](reports/stage1_iter6_20260926/formal/final_decision.json)记录
> `stage1_complete=true`；[分析与冻结说明](reports/stage1_iter6_20260926/analysis.md)
> 保留失败轮次、预注册和源码归档。二阶段通过 1000 人 × 250 日试跑，
> 详见[二阶段验收](reports/stage2_20260926/analysis.md)。
>
> 2026-07-30 的 `stage1-mechanism-acceptance-v1` 虽然得到 `16/16 PASS`，
> 但该协议主要检查完整模型的汇总指标，没有阻止异常机制彼此抵消。
> 因此这个结果只保留为历史诊断，**不再作为 Stage 1 完成或进入 Stage 2 的证据**。

## 当前真实性结论

当前版本是可复现的合成人工市场实验原型。一阶段冻结的统计门槛已经通过，
二阶段学习器通过已知最优解和工程契约测试；这些证据不等同于真实市场校准、
政策有效性或独立研究审稿。

| 历史 v1 场景            |            报告中位数 | 暴露的问题                 |
| ----------------------- | --------------------: | -------------------------- |
| `value_only`          | 收益 ACF(1)`-0.978` | 机械正负振荡，价值回归过强 |
| `trend_only`          |    日波动率`0.0006` | 市场几乎不能自行启动       |
| `independent_signals` |  收益 ACF(1)`0.544` | 收益高度可预测             |
| `fixed_participation` |  收益 ACF(1)`0.389` | 参与和下单过度同步         |
| `liquidity_stress`    | 收益 ACF(1)`-0.581` | 流动性冲击引发严重反转     |

上表是历史 v1 结果。v2 已经针对这些问题完成代码重构：订单改为目标仓位增量，
价格由聚合型准订单簿形成，流动性冲击逐步衰减，Agent 异步参与且具有持续性流动性需求。
第六轮已经通过长样本、多滞后 ACF、尾部、价差、深度和订单流的冻结验收。
其永久冲击按净股数和固定流动性尺度计算，避免变动深度对往返订单产生永久漂移。

## 快速启动

以下命令均在 **Windows PowerShell** 中执行。所有相对路径都以项目根目录
`C:\Users\kanzaler\Desktop\ABM` 为起点。

### 1. 第一次安装

```powershell
Set-Location C:\Users\kanzaler\Desktop\ABM
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

如果 `.venv` 已经存在，不要重复创建；只需在代码或依赖变化后重新执行最后一条安装命令。
安装完成后可用下面的命令确认桌面入口可用：

```powershell
.\.venv\Scripts\abm-desktop.exe --help
```

### 2. 启动本地 Web 实验室（推荐）

```powershell
Set-Location C:\Users\kanzaler\Desktop\ABM
Set-Location web
npm ci
npm run build
Set-Location ..
.\.venv\Scripts\abm-web.exe --open
```

服务只绑定 `127.0.0.1`，启动时生成本地会话令牌。浏览器界面可发现历史运行、启动新运行、
暂停/继续/单步、通过 SSE 观察遥测、回放完成运行，并查看机制报告。运行目录和产物不可覆盖；
关闭或断开浏览器不会改变 Harness 结果。

### 3. 原生桌面兼容入口

```powershell
.\.venv\Scripts\abm-desktop.exe `
  --mode replay `
  --run-dir runs\stage2_five_nodes_20260926
```

PySide6 桌面端继续保留为离线/兼容选项，并与 Web 复用同一 worker 控制逻辑。
选择 `Agents & information` 查看 VisPy 信息图、财富、动作概率、Q 值和奖励分量。
取消 `Follow latest` 后可按日期、交易者、企业和来源筛选。新建五节点实时运行：

```powershell
.\.venv\Scripts\abm-desktop.exe --mode live `
  --config configs\stage2_five_nodes.json --run-dir runs\my_stage2_live_01
```

### 4. 不打开界面，全速运行

```powershell
.\.venv\Scripts\abm-desktop.exe `
  --mode batch `
  --config configs\stage2.json `
  --run-dir runs\my_stage2_batch_01
```

三种启动模式的区别：

| 模式       | 是否运行新模拟 | 是否打开窗口 | `run-dir` 要求 |
| ---------- | -------------: | -----------: | ---------------- |
| `replay` |             否 |           是 | 指向已有完整结果 |
| `live`   |             是 |           是 | 必须是新目录     |
| `batch`  |             是 |           否 | 必须是新目录     |

本机已有依赖和示例结果时，可直接启动 Web 或打开五节点回放。

## 主要功能

- 使用 NumPy 数组保存交易者的现金、持仓、策略和风险厌恶等状态，避免为每个交易者创建重量级对象。
- 使用聚合型准订单簿形成 mid、bid、ask、spread、depth 和执行价格。
- 内置三类规则策略：价值型（value）、趋势型（trend）和噪声型（noise）。
- 订单定义为“平滑目标仓位－当前仓位”，不会重复提交完整目标仓位。
- 价值 Agent 异步累积并处理公开新闻，主观估值、订单和 OFI 构成独立信息通道。
- 订单流冲击拆分为永久与暂时分量：有符号订单流形成永久价格发现，未预期 OFI 只形成逐步衰减的暂时冲击。
- Agent 采用异步参与、持续性活跃状态、流动性需求和策略内持续共同信念。
- 显式双通道价格发现：公开新闻惊喜可直接进入报价，Agent 主观估值与订单流形成独立通道；做市商不读取或持续锚定潜在真实基本面水平。
- Logit 学习保留为一阶段对照；二阶段 Bandit/Q-learning 与 Logit 不同时启用。
- 检查现金守恒、股份守恒、非负现金、非负净财富、卖空边界和每日唯一结算编号。
- 保存配置、随机种子、代码版本、运行摘要、状态数组和逐日审计日志。
- 自动生成市场概览与微观结构诊断的 PNG/SVG 图表。
- FastAPI 本地服务提供受控 REST 与 SSE；React/TypeScript/ECharts 前端统一呈现 live/replay 和机制报告。
- 通过非阻塞 `multiprocessing.Queue` 向 Web 或 PySide6 观察者发送只读 `TelemetryEvent`。
- 支持 `batch`、`live`、`replay` 三种模式，以及暂停、继续、单步和播放速度控制。
- 将完整遥测写入 DuckDB/Parquet；显示队列满时可丢帧，但不丢失回放数据。
- 二阶段追加逐日落盘日志，保存主体观察、行动、结算、消息来源/路径/到达时间、学习表和奖励分量。
- 投资者每 20 日选择价值、趋势、逆向或减仓策略；企业每 60 日选择真实投入、披露投入或维持。
- 静态图支持信任衰减和传播延迟；动态图仅根据过去 20 日已执行订单更新边权，次日生效。

## 二阶段配置与证据

`configs/stage2.json` 为 1000 名投资者、6 家企业、250 日动态图；
`configs/stage2_five_nodes.json` 为 4 名投资者、1 家企业、120 日固定图。
Web 中选择这两个配置即可运行；`Agents & information` 提供日志播放、单步和筛选。

| 开关 | 行为 |
| --- | --- |
| `stage2_enabled=false` | 完整旁路二阶段，回到同参数的一阶段 |
| `bandit_enabled=false` | 单独保留原有固定规则交易者 |
| `company_learning_enabled=false` | 单独固定企业为维持现状 |
| 两个学习开关同时为 `false` | 包括企业/图在内全部旁路，精确恢复一阶段 |
| `graph_enabled=false` | 当日公开消息直接广播，去掉图的延迟与路径衰减 |
| `dynamic_graph_enabled=false` | 保留静态信息图 |

投资者和企业各使用 4 个三档状态变量（81 状态），分别按原交易者类型和企业规模共享学习表。
未满 20/60 日的末尾窗口只记录，不提前分配奖励。交易者奖励中的费用只扣一次；
企业五项奖励分别记录。企业生产、披露和核验参数属于合成机制，尚未做真实数据估计。
`factor_preferences` 是稳定的通用因子偏好占位属性，本阶段不把它解释为 ESG，也不让它改变金融奖励。

逐日 JSONL 在发送显示帧前落盘。正常结束后与 DuckDB/Parquet 一起保存；
中断时可从 `runs/.<run-name>.events.jsonl` 读取已经完整写入的日期，末尾不完整行不会进入回放。
这支持日志恢复，不是继续训练的状态检查点。每次新运行须使用新目录名。
完整方法、测试边界和结果指纹见[二阶段验收报告](reports/stage2_20260926/analysis.md)。

## 当前模型边界

当前版本只包含一个资产和一个聚合做市商，采用日频批次决策。Agent 异步产生目标仓位增量，
聚合型准订单簿根据 OFI、深度、永久/暂时冲击和公开信息更新报价，再由做市商统一结算。

当前版本暂不包含：

- 多资产市场和真实社交网络校准；
- PPO、LLM Agent 和第三阶段真实事件/因子接入；
- 实时交易、逐笔连续限价订单簿和生产级公网服务（现有 Web 仅为 loopback 研究工具）。

Kenneth French 49 数据和旧 29/10/10 协议保留为数量级与风格事实诊断，
不用于拟合历史价格路径，也不再单独决定阶段完成状态。

后续阶段的设计边界和门槛见[三阶段开发计划](docs/ABM三阶段开发计划_渐进式学习版.md)。

## 环境与安装说明

- Windows PowerShell 或兼容的命令行环境
- Python 3.12 或更高版本
- NumPy 2.x
- Matplotlib 3.10.x
- FastAPI、Uvicorn、DuckDB
- Node.js 24+ 与 npm（构建 React Web 前端）
- PySide6、pyqtgraph、VisPy（可选原生桌面端）

建议始终使用项目虚拟环境中的 Python，避免调用到其他环境的解释器。
首次安装命令见[快速启动](#快速启动)；安装后所有命令都显式使用
`.\.venv\Scripts\` 下的 Python 或项目入口。

检查依赖：

```powershell
.\.venv\Scripts\python.exe -m pip check
```

## 运行测试

```powershell
.\.venv\Scripts\python.exe -m pytest
```

测试覆盖配置、结算、Manifest、两阶段 Harness、因果信息传播、学习窗口、已知最优动作、串并行确定性和界面回放。测试通过只是代码和基准契约的证据，不替代真实市场校准或样本外研究。

## 运行当前开发回归

当前一阶段配置为 `stage1_structural_dev_v11.json`，开发协议为 v13。
已冻结的开发结果为 **22/22**；在当前源码上重跑只产生开发回归证据：

```powershell
.\.venv\Scripts\python.exe -m abm.mechanism_validation `
  --config configs\stage1_structural_dev_v11.json `
  --protocol configs\stage1_mechanism_dev_v13.json `
  --output reports\stage1_dev_v13_new `
  --workers 4
```

历史正式 v13 为 18/22，第五轮为 20/22，均保留原始失败结论；
第六轮 holdout v15 和正式 v15 均为 22/22。输出目录逐任务写入 `checkpoint.json`；
中断后可在相同源码、配置和协议下增加 `--resume`。身份不匹配时程序拒绝混用旧
checkpoint。

## 正式验收保护

正式 v15 的原始源码保存在 `reports/stage1_iter6_20260926/frozen_stage1_source.zip`。
二阶段已扩展当前源码，旧冻结清单会正确拒绝在当前工作树上运行；精确复现历史正式实验须使用该归档。
当前源码与归档的一条完整 3500 日路径、以及历史 Logit 开/关结果均已进行指纹比对。
下面保留历史冻结命令示例；新的研究验收必须另行预注册未使用的种子，不能把旧种子重跑称为新验收。

holdout 全部 PASS 前，**不得运行 50 个正式种子**。正式协议只有在后续开发和未见
holdout 全部 PASS、冻结文件生成后才能执行。CLI 同时要求匹配清单与显式解封：

```powershell
.\.venv\Scripts\abm-freeze-stage1.exe create `
  --config configs\stage1_structural_acceptance_v6.json `
  --protocol configs\stage1_mechanism_acceptance_v6.json `
  --output configs\stage1_mechanism_freeze_v6.json

.\.venv\Scripts\python.exe -m abm.mechanism_validation `
  --config configs\stage1_structural_acceptance_v6.json `
  --protocol configs\stage1_mechanism_acceptance_v6.json `
  --freeze-manifest configs\stage1_mechanism_freeze_v6.json `
  --unseal-formal `
  --output reports\stage1_mechanism_acceptance_v6 `
  --workers 4
```

缺少任何保护参数、冻结哈希不一致或协议不是 `stage=formal` 时，程序都会拒绝运行。
开发协议生成的 `final_decision.json` 始终保持 `stage1_complete=false`。

## 重现历史机制诊断

下面的命令重现 2026-07-30 的 32 个未见随机种子 × 10 个机制/对照场景。
种子与开发种子互斥，但原协议的判据不充分；即使命令输出 `PASS`，也不能据此宣布
Stage 1 完成（该协议文件已归档于 `reports/archive/`）：

```powershell
.\.venv\Scripts\python.exe -m abm.mechanism_validation `
  --config configs\stage1.json `
  --protocol configs\stage1_mechanism_acceptance_v1.json `
  --output reports\stage1_mechanism_acceptance_new
```

输出包括逐种子 CSV、完整 JSON、Markdown 决策报告和四面板机制归因图。
历史结果位于 `reports/archive/stage1_mechanism_acceptance_v1/`。

## 运行冻结统计校准

下面的命令运行 29 条训练路径和 10 条验证路径，每条 6500 个交易日。测试行业和测试种子
不会运行：

```powershell
.\.venv\Scripts\abm-calibrate.exe `
  --config configs\stage1.json `
  --protocol configs\stage1_calibration_v1.json `
  --archive data\raw\kenneth_french\20260729T073457Z\49_Industry_Portfolios_daily_CSV.zip `
  --split configs\french49_split_v1.json `
  --output reports\stage1_calibration_new.json
```

## 运行 Stage 0 合成夹具

仓库提供一个十日合成场景：

- 场景文件：`data/synthetic/stage0_10_day.json`
- 数据 Manifest：`data/manifests/stage0_10_day.json`

该夹具用于测试数据契约、恢复和哈希校验，不是 Stage 1 的正式验收配置。

## 运行 Stage 1 当前原型

规模配置 `configs/stage1.json` 为 10000 名交易者、10000 个交易日、前 1000 日
burn-in、随机种子 `20260729`。快速试跑请改用
`configs/stage1_structural_dev_v11.json`：

```powershell
.\.venv\Scripts\abm-run.exe `
  --config configs\stage1_structural_dev_v11.json `
  --output runs\stage1_structural_v11_new
```

也可以使用模块入口：

```powershell
.\.venv\Scripts\python.exe -m abm.runner `
  --config configs\stage1_structural_dev_v11.json `
  --output runs\stage1_structural_v11_new
```

输出目录必须不存在；程序不会覆盖已有运行结果。命令行最后会打印 JSON 格式的运行摘要，
包括交易日数、交易者数量、最终价格、守恒误差、学习更新次数、最终策略数量和结果指纹。

## 重新生成图表

如果运行目录已经存在，可以单独重新生成图表：

```powershell
.\.venv\Scripts\abm-plot.exe `
  --run-dir runs\stage1_structural_v2_new
```

图表包括：

- 市场价格与基本面价值；
- 每日成交量；
- 策略构成变化；
- 现金和股份守恒误差；
- mid 与执行价格、bid-ask spread、depth 和 OFI；
- 永久冲击与暂时冲击的分解。

## 原生桌面监控与回放

桌面端的完整启动命令见[快速启动](#快速启动)。`live` 模式中的 Harness
运行在独立进程，界面只读取遥测事件；显示队列满时允许丢弃界面帧，但完整事件仍会写入
`telemetry.duckdb` 和 `telemetry.parquet`。`replay` 只读取这些持久化事件，不会改变原运行结果。

需要限制界面刷新率时可增加 `--max-fps 5`。需要自动导出回放截图并立即退出时可执行：

```powershell
.\.venv\Scripts\abm-desktop.exe `
  --mode replay `
  --run-dir runs\stage1_structural_v2_dev_20260730 `
  --screenshot runs\stage1_structural_v2_dev_20260730\desktop_replay_copy.png
```

常见启动问题：

| 现象                        | 原因与处理                                                           |
| --------------------------- | -------------------------------------------------------------------- |
| 提示输出目录已经存在        | `live` 和 `batch` 不覆盖结果；换一个新的 `run-dir`             |
| 找不到`abm-desktop.exe`   | 重新执行`.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`    |
| `replay` 提示缺少遥测文件 | 传入包含`telemetry.duckdb` 或 `telemetry.parquet` 的完整运行目录 |
| PowerShell 当前不在项目目录 | 先执行`Set-Location C:\Users\kanzaler\Desktop\ABM`                 |

## Kenneth French 49 基准统计

固定分组位于 `configs/french49_split_v1.json`。运行统计时必须使用新的
输出文件，测试组只验证身份和密封状态，不读取指标：

```powershell
.\.venv\Scripts\abm-benchmark.exe `
  --archive data\raw\kenneth_french\<snapshot>\49_Industry_Portfolios_daily_CSV.zip `
  --split configs\french49_split_v1.json `
  --run-dir runs\stage1_plan_corrected_20260729 `
  --output reports\stage1_new\french49_benchmark.json
```

## 运行结果目录

一次完整运行通常会生成以下文件：

```text
runs/<run-name>/
├── config.json             # 本次运行实际使用的配置
├── run_manifest.json       # 随机种子、代码版本和运行元数据
├── summary.json            # 运行摘要与结果指纹
├── state_arrays.npz        # 价格、基本面、成交量和最终状态数组
├── daily_audit.jsonl       # 每个交易日一条结算审计记录
├── learning_audit.json     # 默认为空；启用 Logit 后记录更新
├── stage2_audit.json       # 二阶段学习表、次数、偏好与企业账本
├── events.jsonl            # 二阶段逐日持久化日志
├── telemetry.duckdb       # 完整市场与主体事件
├── telemetry.parquet
├── stage1_overview.png     # 市场概览图
├── stage1_overview.svg
├── stage1_microstructure.png # 价差、深度、OFI 和冲击分解
└── stage1_microstructure.svg
```

运行结果目录应视为不可变实验产物。需要重跑时请使用新的目录名，以保留不同配置、代码版本和随机种子的证据链。

## 配置说明

主要配置位于 `configs/stage1.json`：

| 配置项                               | 含义                                      |
| ------------------------------------ | ----------------------------------------- |
| `population_size`                  | 交易者数量                                |
| `trading_days`                     | 模拟交易日数量                            |
| `seed`                             | 随机种子                                  |
| `initial_price`                    | 初始市场价格                              |
| `initial_fundamental`              | 初始基本面价值                            |
| `price_formation`                  | 当前固定为聚合型准订单簿                  |
| `price_impact`                     | 未预期 OFI 的价格影响强度                 |
| `liquidity_scale`                  | 流动性尺度                                |
| `fundamental_process`              | Gaussian、Student-t 或 GARCH-t 基本面过程 |
| `public_news_price_pass_through`   | 公开信息进入做市商报价的比例；v11 验收值为 1.0 |
| `permanent_impact_basis`           | `participation` 保留旧路径；`net_shares` 为 v11 固定尺度永久冲击 |
| `base_activity_rate`               | Agent 每日基础参与率                      |
| `activity_rate_dispersion`         | Agent 基础参与率的个体异质性              |
| `activity_persistence`             | 市场活跃状态的持续性                      |
| `activity_volatility_sensitivity`  | 参与率对近期波动的响应                    |
| `common_signal_correlation`        | 同策略 Agent 的共同信念强度               |
| `common_signal_persistence`        | 共同信念冲击的时间持续性                  |
| `liquidity_volatility_sensitivity` | 做市深度对波动压力的收缩强度              |
| `transaction_cost_rate`            | 交易成本率                                |
| `target_position_fraction`         | 目标仓位占财富的上限                      |
| `value_no_trade_band`              | 价值策略无交易区间                        |
| `trend_short/long_lookback_*`      | 趋势策略异质短长窗口范围                  |
| `permanent_impact_fraction`        | 订单流冲击中的永久部分                    |
| `transient_impact_decay`           | 暂时冲击的逐期衰减率                      |
| `order_flow_memory`                | 做市商对可预测订单流的估计记忆            |
| `burn_in_days`                     | 不进入正式统计的预热期                    |
| `learning_enabled`                 | 一阶段 Logit 开关；二阶段启用时须为 `false` |
| `strategy_shares`                  | value、trend、noise 的初始比例            |
| `announcements`                    | 指定交易日的基本面公告                    |

配置修改后，应使用新的输出目录运行，并保留对应的 `config.json` 和 `summary.json`。

## 代码结构

```text
.
├── configs/       # Stage 0 和 Stage 1 配置
├── data/          # 合成数据与数据 Manifest
├── docs/          # 设计文档、开发计划和验收报告
├── runs/          # 仿真输出目录
├── src/abm/
│   ├── config.py        # 配置读取和校验
│   ├── schemas.py       # 核心数据结构与序列化
│   ├── population.py    # 数组化交易者群体
│   ├── policies.py      # 异步目标仓位与订单到达机制
│   ├── microstructure.py # 聚合型准订单簿与价格冲击分解
│   ├── settlement.py    # 订单执行与做市商结算
│   ├── learning.py      # 暂停使用的 Logit 策略模仿
│   ├── harness.py       # 日频市场 Harness 和审计输出
│   ├── manifest.py      # 文件哈希与运行 Manifest
│   ├── runner.py        # abm-run 命令行入口
│   ├── visualization.py # abm-plot 静态导出入口
│   ├── telemetry.py     # 遥测契约、队列与 DuckDB/Parquet
│   ├── run_service.py   # Web/桌面共享的独立 worker 状态机
│   ├── web.py           # loopback FastAPI、REST、SSE 与静态前端
│   ├── desktop.py       # PySide6/pyqtgraph 兼容桌面端
│   └── french49.py      # 官方数据快照解析与密封基准统计
├── web/             # React/TypeScript/Vite/ECharts 前端
└── tests/           # 自动化测试
```

核心控制原则是：`MarketHarness` 统一管理市场时钟和状态变更，策略只能返回动作，不能直接修改价格、现金或持仓。这样可以将模型行为、结算结果和审计记录分开验证。

## 研究使用注意事项

1. 相同配置和随机种子应产生相同结果指纹；修改代码或配置后必须记录新的 Manifest。
2. 不要把结构性测试、守恒检查或单次仿真结果直接表述为真实市场验证。
3. 真实数据接入前，应先冻结数据快照、来源、下载时间、可用时间和文件哈希。
4. 正式结构验收通过前，不得启用 Logit、ESG、LLM、强化学习或政策实验。

## 许可证

当前项目在 `pyproject.toml` 中标记为 `Proprietary`，具体使用范围以项目所有者的授权为准。
