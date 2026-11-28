# GitHub 展示层 · Financial ABM

独立于 `web/` 的静态站点，不连接 FastAPI，不在构建时运行模拟。两个页面：

| 页面 | 内容 |
|---|---|
| `_site/index.html`（首页） | 漂绿政策沙盘（计划阶段 G + MAS 机制升级 §§29–32）：首屏作战台回放面板 ② 的 14 年简化动态（企业球、决策频谱、事件流、核查日志，标注为非研究结果）；面板 ① 企业单期决策，面板 ② 简化动态（无市场反馈，非研究结果），面板 ③ 阶段 F 完整模型结果，面板 ④ MAS 机制设计确认性结果（7 项研究、2,250 次运行、40/41 Holm 支持，支持 Stackelberg 核查、内生鉴证相变/帕累托、有界学习与 Q-learning / AI-washing 交互浏览）。默认英文，可切换中文；ESG / AI-washing 切换 |
| `_site/stage1/index.html` | 阶段 1 展厅：历史 v13/v14 机制报告（原首页，内容不变），默认英文，可切换中文 |

## 预览

在项目根目录执行：

```sh
python showcase/build.py --output _site
python -m http.server 8080 --directory _site
```

访问 `http://localhost:8080`，也可以直接打开 `_site/index.html`。输出内联全部 CSS、JavaScript 与数据，没有 CDN、在线字体、遥测或外部图表库。构建只用 Python 3.12+ 标准库。

## 沙盘的数据与一致性

| 文件 | 来源 |
|---|---|
| `data/sandbox_firms.json` | [导出脚本](../reports/disclosure_stage_g_20261015/export.py)：新种子 51001 按阶段 F 主设定运行，第 1200 日快照 200 家合成企业；AI-washing 领域同样快照 |
| `data/stage_f.json` | 从已提交的 `confirmatory_v1_analysis.json` 复制，附 `experiment_id`、SHA-256 与提交号 |
| `reports/disclosure_mas_upgrade_20261125/mas_upgrade_analysis.json` | `build.py` 直接读取 §§29–32 确认性分析文件并按 SHA-256 校验，提取 7 项研究、2,250 次运行与 41 项 Holm 校正主检验供面板 ④ 展示 |
| `data/simplification.json` | [对照脚本](../reports/disclosure_stage_g_20261015/compare.py)：简化动态与完整模型的差距（“简化的代价”） |
| `sandbox/core.js` | 决策核心与简化动态，移植自 `src/abm/disclosure` 与 `sandbox/reference.py` |
| `sandbox/fixtures/*.json` | [夹具脚本](../reports/disclosure_stage_g_20261015/fixtures.py)：由 Python 模型代码生成的期望值 |

构建会核对 `stage_f.json` 与分析文件的 SHA-256、导出时的阶段 F 配置和 AI-washing 领域文件，任何一处变化都会拒绝构建。企业字段只允许模型自己的数值列，不含任何真实公司信息。

```sh
node showcase/sandbox/check_core.js          # JavaScript 与 Python 逐项一致（相对误差 ≤ 1e-12）
python -m unittest discover -s showcase -p 'test_*.py'
python -m pytest tests/test_sandbox_reference.py   # 夹具仍与当前模型代码一致（需要安装 abm）
```

`sandbox/reference.py` 放在 `src/abm` 之外：包内源码哈希是每个冻结实验身份的一部分。

## GitHub Pages

线上地址：[https://kanzaler123.github.io/financial-ABM/](https://kanzaler123.github.io/financial-ABM/)。Pages 已于 2026-10-16 启用（Source: GitHub Actions），`GitHub showcase` 工作流在 master 更新时部署 `_site`。未启用 Pages 时，CI 仍验证页面并提供名为 `github-showcase` 的离线产物；PR 只构建检查，不发布。

## 链接预览图

两个页面的 `<head>` 里都有英文的 Open Graph 与 X（Twitter）卡片标签，`og:image` 指向 `https://kanzaler123.github.io/financial-ABM/social-card.png`。构建时会把 `social/social-card.png`（1200×630）复制进 `_site`；如果尺寸不是 1200×630，构建会直接失败。图上只写研究问题，不放研究数字和结论，所以结论更新时不必重做这张图。

| 文件 | 内容 |
|---|---|
| `social/card.html` | 预览图源文件，用站点的配色与字体；右侧企业球取自 `.github/assets/disclosure-cover.svg`，只是结构示意 |
| `social/render.py` | 用无头 Chrome / Edge 把源文件渲染成 `social/social-card.png`（1200×630）和 `../.github/assets/social-preview.png`（1280×640，仓库社交预览图，需要在 Settings → General → Social preview 手动上传） |

改了 `card.html` 之后，运行 `python showcase/social/render.py`，然后提交两张 PNG。构建本身不会启动浏览器。现有图片是在 Windows 上渲染的；换到其他系统，字体会回落成别的字体，文字也会重新排版。

## 阶段 1 展厅的数据约定

`build.py` 从三个已提交的历史 v13/v14 `mechanism_report.json` 读取原始数据，保留报告代码版本、SHA-256 和源文件链接，不重算模拟。2026-08-14 正式归档为 18/22，开发与 holdout 各为 22/22；此后[冻结 v15](../reports/stage1_iter6_20260926/formal/final_decision.json)于 2026-09-26 正式验收 22/22。分布图使用跨种子中位数与 Q10–Q90，不是时序数据或置信区间。

## 修改入口

| 文件 | 内容 |
|---|---|
| `sandbox/index.html`、`styles.css` | 首页结构与样式 |
| `sandbox/i18n.js` | 中英文案；`test_showcase.py` 检查两种语言的键一致，并检查模板中没有未接入切换的中文 |
| `sandbox/app.js` | 政策调节、作战台回放、图表与交互（模型计算都在 `core.js`） |
| `sandbox/desk.js` | 作战台的绘制：企业球（canvas）、决策频谱、智能体角色与迷你走势线；只画图，不含文案和模型计算 |
| `stage1/` | 阶段 1 展厅的模板、样式与脚本；文案在 `stage1/i18n.js` |
| `social/` | 链接预览图的源文件与渲染脚本（见上一节） |
| `build.py` | 数据核对与离线单页构建 |
| `test_showcase.py` | 数据与静态构建回归；也检查链接预览标签和两张预览图的尺寸 |
