# 可审计人工金融市场 ABM

这是一个面向研究用途的人工金融市场（Agent-Based Model，ABM）项目。
当前仓库实现的是第一阶段：单一资产、日频、固定种子可精确重放的市场基准与
Logit 策略模仿机制。
项目优先保证模型边界清晰、运行结果可复现、状态变化可审计，再逐步扩展到信息图传播、学习算法和真实数据。

> 当前状态：Stage 1 已于 2026-07-30 按“机制生成真实性”标准完成。
> 冻结后的 32 个未见种子 × 10 个机制场景通过 `16/16` 项正式验收；旧版
> `stage1-freeze-v1` 的 4/7 IQR 结果保留为历史拟合诊断，不再作为项目完成判据。

## 主要功能

- 使用 NumPy 数组保存交易者的现金、持仓、策略和风险厌恶等状态，避免为每个交易者创建重量级对象。
- 模拟单一资产的日频同步交易流程：观察、提交订单、价格冲击、做市商结算和状态审计。
- 内置三类规则策略：价值型（value）、趋势型（trend）和噪声型（noise）。
- 默认基本面使用透明 Gaussian 创新，公开信息只能通过 Agent 净需求影响价格；
  Student-t/GARCH-t 仅作为显式外生压力对照。
- Agent 采用异步参与、波动响应参与和策略内共同信念机制；这些机制均可独立关闭并消融。
- Agent 具有风险敏感度、趋势观察期和决策噪声异质性，并允许受杠杆约束的有限卖空。
- 每隔固定交易日使用部分异步 Logit 模仿机制，只更新指定比例的交易者。
- 检查现金守恒、股份守恒、非负现金、非负净财富、卖空边界和每日唯一结算编号。
- 保存配置、随机种子、代码版本、运行摘要、状态数组和逐日审计日志。
- 自动生成市场概览与学习诊断的 PNG/SVG 图表。
- 通过非阻塞 `multiprocessing.Queue` 向独立 PySide6 桌面进程发送只读 `TelemetryEvent`。
- 支持 `batch`、`live`、`replay` 三种模式，以及暂停、继续、单步和播放速度控制。
- 将完整遥测写入 DuckDB/Parquet；显示队列满时可丢帧，但不丢失回放数据。

## 当前模型边界

当前版本只包含一个资产和一个做市商，采用日频同步决策。交易者通过规则策略产生净需求，价格根据净需求和流动性参数产生有界对数收益，然后由做市商完成统一结算。

当前版本暂不包含：

- 多资产市场、交易网络和信息传播；
- Contextual Bandit、表格 Q-learning、PPO 或 LLM Agent；
- 实时交易、连续订单簿和生产级前端界面。

Kenneth French 49 数据和旧 29/10/10 协议保留为数量级与风格事实诊断，
不用于拟合历史价格路径，也不再单独决定阶段完成状态。

后续阶段的设计边界和门槛见[三阶段开发计划](docs/ABM三阶段开发计划_渐进式学习版.md)。

## 环境要求

- Windows PowerShell 或兼容的命令行环境
- Python 3.12 或更高版本
- NumPy 2.x
- Matplotlib 3.10.x
- PySide6、pyqtgraph、VisPy
- DuckDB

建议始终使用项目虚拟环境中的 Python，避免调用到其他环境的解释器。

## 安装

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

检查依赖：

```powershell
.\.venv\Scripts\python.exe -m pip check
```

## 运行测试

```powershell
.\.venv\Scripts\python.exe -m pytest
```

测试覆盖配置校验、数据结构、策略、订单结算、场景、Manifest、可视化和 Stage 1 Harness。测试通过只是代码和基准契约的证据，不替代真实市场校准或样本外研究。

## 运行 Stage 1 机制验收

正式验收使用 32 个未见随机种子和 10 个机制/对照场景。正式种子与开发种子互斥：

```powershell
.\.venv\Scripts\python.exe -m abm.mechanism_validation `
  --config configs\stage1.json `
  --protocol configs\stage1_mechanism_acceptance_v1.json `
  --output reports\stage1_mechanism_acceptance_new
```

输出包括逐种子 CSV、完整 JSON、Markdown 决策报告和四面板机制归因图。
现有正式结果位于 `reports/stage1_mechanism_acceptance_v1/`。

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

## 运行 Stage 1 验收仿真

默认配置为 1000 名交易者、250 个交易日、随机种子 `20260729`：

```powershell
.\.venv\Scripts\abm-run.exe `
  --config configs\stage1.json `
  --output runs\stage1_acceptance_new
```

也可以使用模块入口：

```powershell
.\.venv\Scripts\python.exe -m abm.runner `
  --config configs\stage1.json `
  --output runs\stage1_acceptance_new
```

输出目录必须不存在；程序不会覆盖已有运行结果。命令行最后会打印 JSON 格式的运行摘要，包括交易日数、交易者数量、最终价格、守恒误差、学习更新次数、最终策略数量和结果指纹。

## 重新生成图表

如果运行目录已经存在，可以单独重新生成图表：

```powershell
.\.venv\Scripts\abm-plot.exe `
  --run-dir runs\stage1_acceptance_new
```

图表包括：

- 市场价格与基本面价值；
- 每日成交量；
- 策略构成变化；
- 现金和股份守恒误差；
- Logit 更新时各策略的平均适应度与选择概率。

## 原生桌面监控与回放

无界面全速运行：

```powershell
.\.venv\Scripts\abm-desktop.exe `
  --mode batch `
  --config configs\stage1.json `
  --run-dir runs\stage1_batch
```

Harness 独立进程运行并实时监控：

```powershell
.\.venv\Scripts\abm-desktop.exe `
  --mode live `
  --config configs\stage1.json `
  --run-dir runs\stage1_live
```

从 DuckDB/Parquet 回放，不重新运行模型：

```powershell
.\.venv\Scripts\abm-desktop.exe `
  --mode replay `
  --run-dir runs\stage1_live
```

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
├── learning_audit.json     # Logit 学习更新记录
├── stage1_overview.png     # 市场概览图
├── stage1_overview.svg
├── stage1_learning.png     # 学习诊断图
└── stage1_learning.svg
```

运行结果目录应视为不可变实验产物。需要重跑时请使用新的目录名，以保留不同配置、代码版本和随机种子的证据链。

## 配置说明

主要配置位于 `configs/stage1.json`：

| 配置项 | 含义 |
|---|---|
| `population_size` | 交易者数量 |
| `trading_days` | 模拟交易日数量 |
| `seed` | 随机种子 |
| `initial_price` | 初始市场价格 |
| `initial_fundamental` | 初始基本面价值 |
| `price_impact` | 净需求对价格的影响强度 |
| `liquidity_scale` | 流动性尺度 |
| `fundamental_process` | Gaussian、Student-t 或 GARCH-t 基本面过程 |
| `public_news_price_pass_through` | 公开信息直接进入价格的比例；默认值为 0 |
| `base_activity_rate` | Agent 每日基础参与率 |
| `activity_volatility_sensitivity` | 参与率对近期波动的响应 |
| `common_signal_correlation` | 同策略 Agent 的共同信念强度 |
| `liquidity_volatility_sensitivity` | 做市深度对波动压力的收缩强度 |
| `transaction_cost_rate` | 交易成本率 |
| `trend_lookback` | 趋势策略的回看窗口 |
| `learning_interval` | Logit 策略模仿间隔 |
| `strategy_shares` | value、trend、noise 的初始比例 |
| `announcements` | 指定交易日的基本面公告 |

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
│   ├── policies.py      # value/trend/noise 规则策略
│   ├── settlement.py    # 订单执行与做市商结算
│   ├── learning.py      # Logit 策略模仿
│   ├── harness.py       # 日频市场 Harness 和审计输出
│   ├── manifest.py      # 文件哈希与运行 Manifest
│   ├── runner.py        # abm-run 命令行入口
│   ├── visualization.py # abm-plot 静态导出入口
│   ├── telemetry.py     # 遥测契约、队列与 DuckDB/Parquet
│   ├── desktop.py       # PySide6/pyqtgraph batch/live/replay 桌面端
│   └── french49.py      # 官方数据快照解析与密封基准统计
└── tests/          # 自动化测试
```

核心控制原则是：`MarketHarness` 统一管理市场时钟和状态变更，策略只能返回动作，不能直接修改价格、现金或持仓。这样可以将模型行为、结算结果和审计记录分开验证。

## 研究使用注意事项

1. 相同配置和随机种子应产生相同结果指纹；修改代码或配置后必须记录新的 Manifest。
2. 不要把结构性测试、守恒检查或单次仿真结果直接表述为真实市场验证。
3. 真实数据接入前，应先冻结数据快照、来源、下载时间、可用时间和文件哈希。
4. 扩展学习算法前，应保留规则策略和 Logit 机制作为可关闭、可比较的基准。

## 许可证

当前项目在 `pyproject.toml` 中标记为 `Proprietary`，具体使用范围以项目所有者的授权为准。
