# 复现验收：只用公开仓库从零重建（2026-10-23）

**性质：复现验收，不是新实验。** 在一个全新目录里克隆 GitHub 上 `kanzaler123/financial-ABM` 的 master（`8afc3fc`），按锁定依赖建新虚拟环境，只用已提交的文件逐项重建测试、论文图表、论文 PDF、各报告表格和实验规格，并核对契约里记录的哈希与 `experiment_id`。重建时没有读取原作者工作副本和其中的本地档案。只有在第 5 节“用本地档案核对”中，为查明源码摘要对不上的原因，事后只读地查看了几份本地档案的 `experiment.json` 和工作副本中一个源文件的换行。也没有跑任何契约规模的仿真；除测试中的小规模运行外，唯一跑的模型是阶段 G 的简化沙盒（240 次，12 个进程）。所有步骤都在同一台 Windows 机器上完成，Linux 与 macOS 没有实测，下文凡涉及其他平台的说法都是从代码推断的。

## 结论

1. **从公开文件能重建、而且逐字节一致的：** 全部 12 个由汇总文件生成的报告表格（阶段 F 两份、衰减网格、四个反馈分解、20.21 稳健性、定向核查、鉴证、学习型企业重做、联合稳健性）；阶段 G 的简化对照（`simplification_tables.md` 与 `showcase/data/simplification.json`）；论文的 4 张 PNG 图；用 matplotlib 3.11.0 时另外 4 张 PDF 图也逐字节一致；7 个规格脚本写出的 12 份配置文件（其中阶段 F 的 2 份在 Windows 上写成 CRLF，转成 LF 后一致），以及用记录的 `domain_sha256` 生成的学习型企业重做的 2 份。全量测试 974 通过、1 跳过。
2. **能重建、但不逐字节一致的：** 用锁定的 matplotlib 3.11.1 重画的 4 张 PDF 图，绘图内容流与已提交的相同，只差字体子集与 Producer 字段（已提交的图是 3.11.0 画的）；论文 PDF 30 页的抽取文本与已提交的逐页相同，固定 `SOURCE_DATE_EPOCH` 后除日期字符串、`/ID` 和交叉引用偏移外逐字节相同。
3. **卡在本地档案（b）：** 各实验的 `*_analysis.json` 本身（要读 `summary.json`、记录和紧凑数组）、`finalize.py` 汇总、规格脚本中对阶段 F 档案的派生核对、论文中 V_hidden − B 的区间，以及契约第 21–23 节记录的“源码摘要”。最后一项**直接用公开文件对不上**。用本地档案的 `experiment.json` 查明了原因：冻结阶段 F 及以后各实验身份的工作副本中，`src/abm/mechanism_validation.py` 是 CRLF，git 里是 LF，内容相同。把这一个文件转成 CRLF 后，冻结前的提交就能逐字节重现记录的源码表、源码摘要和 `experiment_id`（第 5 节、问题 2）。
4. **卡在供应商数据（c）：** 领域文件 `configs/domains/csi300_huazheng_esg_quantile_copula_fixed_fine_saturated.json` 不在 git 中，所以阶段 F 及以后的所有完整仿真都不能重跑，除学习型企业重做之外的 `experiment_id` 都不能重算（`freeze_spec` 把领域文件的全部内容算进 `experiment_id`），7 个规格脚本中有 6 个在写出配置后的派生核对处退出，学习型企业重做的规格脚本在写出配置前就退出。领域文件本身要从华证评级和市场数据重建，采集脚本与哈希清单在 `reports/disclosure_public_data_20261011/`，原始数据不在 git 中。

## 1. 环境

| 项目 | 值 |
|---|---|
| 系统 | Windows 11 Home China 10.0.26200，Git Bash |
| Git | 2.55.0.windows.2；系统级 `core.autocrlf=true`（未改） |
| 克隆 | `git clone https://github.com/kanzaler123/financial-ABM.git`，master `8afc3fc`，3516 个文件，新目录 `C:\Users\kanzaler\repro_20261023\financial-ABM` |
| Python | 3.12.9（python.org 安装版，MSC v.1942 64 bit） |
| 依赖 | `python -m venv .venv` 后 `pip install -c requirements-lock.txt -e ".[all,dev]"`，用时 4 分 40 秒；`pip check`：No broken requirements found；`pip freeze` 与 `requirements-lock.txt` 的 42 个包完全相同（只差 CRLF） |
| TeX | MiKTeX 26.5，pdfTeX 1.40.29，latexmk 4.88 |
| 其他 | Node.js（一个测试要用）；一个独立的工具环境（matplotlib 3.11.0、numpy 2.5.1、pypdf 6.21.2，其余为当时最新版），只用于第 3 节的对照 |

在这个克隆里，`reports/disclosure_*/**` 设为 `-text`，2222 个文件在 git 里按原样存为 CRLF，检出后字节不变；`*.py`、`*.json`、`*.md` 设为 `eol=lf`，所以 `src/abm` 和 `configs/` 在 `autocrlf=true` 的机器上也检出为 LF。本次运行没有遇到 WMI 卡住，也不需要手动调用 `_no_wmi.py`：各分析脚本导入时会自己调用它，在非 Windows 平台上它什么也不做。

## 2. 测试

```
python -m pytest -p no:cacheprovider
974 passed, 1 skipped, 1 warning in 185.38s (0:03:05)
```

- 跳过的是 `tests/test_sandbox_reference.py::test_exported_domains_match_their_configuration_files` 中的 ESG 部分，理由是“the CSI 300 domain file is vendor-derived and not in the repository”，属于 (c)。
- 警告是 `StarletteDeprecationWarning`（starlette 的测试客户端对 httpx 的弃用提示），与研究代码无关。
- 同一提交在 GitHub Actions 上（Ubuntu，Python 3.12 与 3.13）的测试工作流结果为 success：<https://github.com/kanzaler123/financial-ABM/actions/runs/39956631808>。

## 3. 论文图表与 PDF

**图。** `python paper/make_figures.py` 退出码 0，用时约 2 秒，输出 `wrote figures to paper\figures`。脚本只读 `reports/disclosure_stage_f_20261014/confirmatory_v1_analysis.json` 和 `reports/disclosure_decay_grid_20261016/grid_v1_analysis.json`，这两份都已提交，和它的说明一致。

| 文件 | 锁定环境（matplotlib 3.11.1） | matplotlib 3.11.0 |
|---|---|---|
| `fig_*.png`（4 张） | 逐字节一致 | 逐字节一致 |
| `fig_*.pdf`（4 张） | 不一致；解压后的主绘图内容流相同，嵌入字体程序从约 53–89 KB 缩到 10–13 KB，Producer 为 `v3.11.1`（已提交的为 `v3.11.0`） | 逐字节一致 |

**论文。** 在 `paper/` 下运行 `latexmk -pdf -interaction=nonstopmode main.tex`，退出码 0，用时约 19 秒，30 页。日志里有一条 LaTeX 警告 `` `h' float specifier changed to `ht' ``，没有未定义引用或缺失文件。

- 用锁定环境重画的图编译：451,161 字节（已提交的为 525,769 字节，差在图里的字体子集）；用 pypdf 逐页抽取文本，30 页全部与已提交的 `main.pdf` 相同。
- 用已提交的图，再设 `SOURCE_DATE_EPOCH=1792731164`（2026-10-23 12:52:44 +08:00，即已提交 PDF 的 CreationDate）和 `FORCE_SOURCE_DATE=1` 编译：525,757 字节；把 CreationDate/ModDate、`/ID` 和交叉引用表的偏移屏蔽后与已提交的逐字节相同。剩下的差别是日期写成 `D:20261023045244Z` 而不是 `D:20261023125244+08'00'`，以及由文件名和时间生成的 `/ID`。

论文中的数字是手工录入的（`paper/README.md` 已写明来源）。除 V_hidden − B 的区间外，都能在已提交的报告 README、`*_tables.md` 或 `*_analysis.json` 中找到。V_hidden − B 的区间 −0.005 [−0.012, +0.001] 只出现在 `reports/disclosure_targeted_audits_20261020/README.md` 的正文里，不在任何已提交的机读文件中，它来自本地的 `summary.json`。

## 4. 报告表格

`python reports/reproduction_acceptance_20261023/rebuild_tables.py` 对每个目录读取已提交的分析文件，交给该目录自己的脚本里生成表格的函数（就是当初写出表格的代码），再与已提交的表格逐字节比较。每个目标在独立的解释器里跑，不读任何本地档案。结果在 [rebuild_tables.json](rebuild_tables.json)，用时约 7 秒：

| 表格 | 来源 | 结果 |
|---|---|---|
| `disclosure_stage_f_20261014/confirmatory_v1_tables.md` | `confirmatory_v1_analysis.json` + `tables.py` | 逐字节一致（CRLF，见问题 3） |
| `disclosure_stage_f_20261014/development_v1_tables.md` | `development_v1_analysis.json` + `tables.py` | 逐字节一致（CRLF，见问题 3） |
| `disclosure_decay_grid_20261016/grid_v1_tables.md` | `grid_v1_analysis.json` + `analyze.tables` | 逐字节一致 |
| `disclosure_feedback_counterfactual_20261016/tables.md` | `analysis.json` + `counterfactual.write_tables` | 逐字节一致 |
| `disclosure_feedback_strong_20261016/tables.md` | `analysis.json` + `strong.write_tables` | 逐字节一致 |
| `disclosure_feedback_channels_20261017/tables.md` | `analysis.json` + `channels.write_tables` | 逐字节一致 |
| `disclosure_feedback_channels_main_20261017/tables.md` | `analysis.json` + `channels_main.write_tables` | 逐字节一致 |
| `disclosure_robustness_20261018/robust_v1_tables.md` | `robust_v1_analysis.json` + `analyze.tables` | 逐字节一致 |
| `disclosure_targeted_audits_20261020/audits_v1_tables.md` | `audits_v1_analysis.json` + `analyze.tables` | 逐字节一致 |
| `disclosure_assurance_20261021/assurance_v1_tables.md` | `assurance_v1_analysis.json` + `analyze.tables` | 逐字节一致 |
| `disclosure_learning_redo_20261021/redo_v1_tables.md` | `redo_v1_analysis.json` + `analyze.tables_md` | 逐字节一致 |
| `disclosure_joint_robustness_20261023/joint_v1_tables.md` | `joint_v1_analysis.json` + `analyze.tables` | 逐字节一致 |

**阶段 G。** `python reports/disclosure_stage_g_20261015/compare.py` 退出码 0（12 个进程，8 个政策 × 衰减组合 × 30 个核查种子），输出 `wrote showcase/data/simplification.json` 和 `wrote reports/disclosure_stage_g_20261015/simplification_tables.md`，跑完后 `git status` 干净，即两份文件都逐字节一致。它只读已提交的 `showcase/data/sandbox_firms.json`（导出的 ESG 参数和 200 家快照企业），不需要领域文件。

各 `*_analysis.json` 本身不能从公开文件重算：每个 `analyze.py` 都要读运行档案中的 `summary.json`、逐次记录和紧凑数组（后几节还要读 `experiment.json`），这些档案只在原作者本地。四个反馈分解目录提交了逐种子的 `results.json`，其中 `channels_main.py` 在分析时会用 20.19 的 `results.json` 重算该格点的对比，但这些脚本的 `analyze` 子命令仍要读本地的 `work/` 记录。

## 5. 规格、配置哈希与 experiment_id

在克隆里依次运行 8 个规格脚本（它们原地改写 `configs/*.json`），之后用 `git status` 和 [check_specs.py](check_specs.py) 核对。结果在 [check_specs.json](check_specs.json)。

| 脚本 | 退出码 | 写出的配置 | 退出原因 |
|---|---:|---|---|
| `disclosure_stage_f_20261014/build_specs.py` | 0 | 2 份；在 Windows 上为 CRLF，转成 LF 后与已提交的相同 | — |
| `disclosure_decay_grid_20261016/build_spec.py` | 1 | 1 份，逐字节一致 | `check_derivation` → `freeze_spec` 找不到领域文件 |
| `disclosure_feedback_strong_20261016/build_spec.py` | 1 | 1 份，逐字节一致 | 同上 |
| `disclosure_robustness_20261018/build_spec.py` | 1 | 2 份，逐字节一致 | 同上（之后还要读阶段 F 确认性档案） |
| `disclosure_targeted_audits_20261020/build_spec.py` | 1 | 2 份，逐字节一致 | 同上 |
| `disclosure_assurance_20261021/build_spec.py` | 1 | 2 份，逐字节一致 | 同上 |
| `disclosure_joint_robustness_20261023/build_spec.py` | 1 | 2 份，逐字节一致 | 同上 |
| `disclosure_learning_redo_20261021/build_spec.py` | 1 | 无 | `spec()` 要读领域文件来算 `domain_sha256` |

学习型企业重做：把 `domain_sha256` 换成配置里记录的 `c3dd4824…f907` 后，`spec()` 生成的两份配置与已提交的逐字节相同，脚本自带的 `check()` 也通过（这一步用的是临时补丁，没有改脚本）。

与契约记录的比较：

| 规格 | 配置 SHA256 与契约 | experiment_id |
|---|---|---|
| 阶段 F 确认性（契约第 376 行） | 契约记录的 `86b6bc77…8832` 是 **CRLF** 文件的哈希；已提交的 LF 文件为 `4c16e210…ab01f` | 不能重算（c） |
| 阶段 F 开发、20.21 开发、21/22 开发 | 契约未记录配置哈希 | 不能重算（c） |
| 衰减网格、20.18、20.21、21、22、24 确认性与 24 开发 | 与契约记录的 LF 哈希一致 | 不能重算（c） |
| 23 确认性与开发 | 与契约记录的 LF 哈希一致 | **重算一致**：`b4430136…6062`、`74b6a503…728f` |

除学习型企业重做外，`experiment_id` 是 `freeze_spec` 对 `{spec, raw_domain, scenarios, source_hashes, python_version, numpy_version, scipy_version}` 的规范 JSON 取 SHA-256，其中 `raw_domain` 是领域文件的全部内容，所以没有供应商派生文件就算不出来。即使有了领域文件，也还需要 Python 3.12.9、numpy 2.5.1、scipy 1.18.1（已提交的档案记录的就是这几个版本），以及冻结时的 `src/abm`。阶段 F 之后 `src/abm` 又改过 4 次，所以重算阶段 F 和 20.x 各节的 `experiment_id` 要先检出当时的提交。还要把 `src/abm/mechanism_validation.py` 转成 CRLF，见下文。

**源码摘要：只用公开文件。** 契约第 21–23 节记录了“源码摘要（各文件源码哈希的规范 JSON 的 SHA256）”：`a94bed8b…`、`99966d0c…`、`7a073291…`。按 `abm.disclosure.runner.source_hashes()` 加 `canonical_json_bytes` 的定义，当前检出的 `src/abm`（52 个文件）得 `c4423d93…7246`。从 git 历史中 17 个改动过 `src/abm` 的提交的原始字节算，没有一个提交得出这三个值中的任何一个，包括第 21、22、23 节各自“在确认性运行之前”提交的 `3fced38`、`e6854a0`、`3d82afe`。把全部文件都转成 CRLF、换路径前缀、只取 `abm/disclosure`、换 JSON 序列化方式，也都对不上。

**源码摘要：用本地档案核对（2026-10-23）。** 读取原作者本地的确认性档案 `audits_v1/`、`assurance_v1/`、`redo_v1/` 中的 `experiment.json`，逐文件比较其 `source_hashes` 与 git 历史：

- 三份档案的源码表算出的摘要正是契约记录的三个值。也就是说，契约的记录与档案一致，定义也与 `source_hashes()` 相同。
- 三份源码表与冻结前的提交（`3fced38`、`e6854a0`、`3d82afe`）相比，52 个文件中只有 `abm/mechanism_validation.py` 不同。档案记录的哈希 `3c70fc90…` 是原作者工作副本中该文件的字节哈希；该文件在工作副本中是 CRLF（`git ls-files --eol` 显示 `i/lf w/crlf`，最后修改于 2026-10-11），转成 LF 后就是 git 里的 blob（`514bb588…`）。`git status` 不显示差异，因为 git 比较时会先规范化换行。工作副本的 `src/` 中只有这一个文件是 CRLF。
- 阶段 F 的开发与确认性档案、衰减网格，以及第 21、22、23、24 节的开发与确认性档案，记录的都是这个 CRLF 哈希。已提交的 `reports/disclosure_learning_protocol_checks_20261011/common_prepolicy_smoke/experiment.json` 也是，所以公开文件里也能看到这一点。
- 只把这一个文件转成 CRLF，`git show` 出来的源码就能逐字节重现各档案的源码表：阶段 F 确认性与衰减网格为 `2b08f65`，第 21 节为 `3fced38`，第 22 节为 `e6854a0`，第 23、24 节为 `3d82afe`。`check_specs.py` 会对每个提交同时算“原样”和“该文件转 CRLF”两种摘要，后者在 `3fced38`、`e6854a0`、`3d82afe` 上分别等于第 21、22、23 节的记录。
- 各档案的 `experiment_id` 都等于其 `experiment.json` 其余内容的规范 JSON 哈希。如果把源码表换成干净克隆的 LF 源码，阶段 F 确认性、衰减网格、第 21、22、24 节确认性的 `experiment_id` 分别变成 `d9ade0e0…`、`0777be1f…`、`b4ce98fc…`、`de9573b7…`、`6814b6ff…`，与记录的都不同。

因此，这些运行用的就是已提交的代码：唯一不同的文件只差换行，而且 `src/abm/disclosure` 中没有任何模块导入它。但仅凭干净克隆，即使有领域文件，用对了提交和版本，算出的 `experiment_id` 也与契约不同；对已有档案运行 `--resume` / `finalize.py` 时，`run_experiment` 会因为 `freeze_spec` 的结果与档案的 `experiment.json` 不同而报错“cannot resume with different inputs, source, or environment”。

作为参照，2026-10-11 的开发档案 `reports/disclosure_action_grid_development_20261011_v1/experiment.json` 提交了完整的 `source_hashes` 表，它与任何提交都不完全一致（最接近的 `92ff504` 也只有 37/52 个文件相同），说明那批早期开发运行用的是未单独提交的中间源码。

## 6. 分类

| 步骤 | 类别 | 原因与缺的具体文件 |
|---|---|---|
| 建环境、`pip check`、`pytest` | (a) | 974 通过；跳过的 1 项需要领域文件（c） |
| `paper/make_figures.py` | (a) | 只读两份已提交的分析文件；PDF 逐字节一致需要 matplotlib 3.11.0（问题 4） |
| `latexmk -pdf main.tex` | (a) | 文本逐页一致；逐字节一致需要固定 `SOURCE_DATE_EPOCH`，且只到日期、`/ID` 为止 |
| 论文数字的出处 | (a)，一项 (b) | V_hidden − B 的区间只在 README 正文里，机读来源是本地 `reports/disclosure_targeted_audits_20261020/audits_v1/summary.json` |
| 12 个 `*_tables.md` | (a) | 从已提交的分析文件逐字节重建 |
| 阶段 G 简化对照 | (a) | 只读 `showcase/data/sandbox_firms.json` |
| 配置文件（14 份） | (a) | 10 份由脚本直接写出后逐字节一致，阶段 F 的 2 份转成 LF 后一致；学习型企业重做的 2 份要用记录的 `domain_sha256` 代替读领域文件 |
| 学习型企业重做的 `experiment_id` | (a) | 它是规格本身的规范 JSON 哈希 |
| 各 `*_analysis.json`、`*_pipeline.json` | (b) | 缺 `reports/<目录>/<run>/` 下的 `summary.json`、记录、紧凑数组与 `experiment.json`，例如 `disclosure_stage_f_20261014/confirmatory_v1/`、`disclosure_robustness_20261018/robust_v1/`、`disclosure_targeted_audits_20261020/audits_v1/`、`disclosure_assurance_20261021/assurance_v1/`、`disclosure_learning_redo_20261021/redo_v1/`、`disclosure_joint_robustness_20261023/joint_v1/` 与各 `check_v1/` |
| `finalize.py` 汇总与报告渲染 | (b) | 同上；`--resume` 要求全部试次已完成 |
| 规格脚本的派生核对（20.21、21、22、24） | (b) + (c) | 先要领域文件，再要阶段 F 确认性档案的 `experiment.json` |
| 契约第 21–23 节的源码摘要 | (a)，需一步换行转换 | 原样对不上；把 `src/abm/mechanism_validation.py` 转成 CRLF 后，冻结前的提交重现三个记录值（第 5 节）。这个原因是用本地档案查明的 |
| 其余各节的 `experiment_id` | (c) | 缺 `configs/domains/csi300_huazheng_esg_quantile_copula_fixed_fine_saturated.json`（SHA256 `c3dd4824…f907`） |
| 阶段 F 及以后的完整仿真 | (c) | 同上；每个规格都引用这份领域文件 |
| 领域文件本身 | (c) | 要从华证评级、账面值与市值快照重建（`reports/disclosure_public_data_20261011/collect_*.py`、`build_csi300_joint_20261012.py`，再经 fixed_fine 与 saturated 两步），原始数据与中间文件都在 `.gitignore` 中，只提交了 `summary.json` 等哈希清单 |

## 7. 发现的问题

1. **完整仿真离不开供应商派生的领域文件。** 这是授权限制下的设计选择，各报告 README 的“复现”一节也写了。对外的影响是：别人用公开仓库能核对表格、图和配置，但不能独立核对任何一个仿真结果或 `experiment_id`（学习型企业重做的除外）。
2. **冻结身份绑定了一个文件的本地换行**（第 5 节）。原作者工作副本中的 `src/abm/mechanism_validation.py` 是 CRLF，而 `source_hashes()` 对文件字节取哈希，所以阶段 F 以来所有记录的源码表、源码摘要和 `experiment_id` 都依赖这个本地换行，干净克隆（LF）重现不了。代码内容相同，结果不受影响。**原作者决定分阶段处理（2026-10-23），已写入[契约第 8 节](../../docs/disclosure_research_contract.md#8-种子统计与预注册状态)：**
   - **当前：** 工作副本保持不变；全部历史哈希与 `experiment_id` 保留原值；旧档案不改写；运行器的源码一致性检查不放宽。
   - **旧档案的复现与续跑：** 在冻结时提交的独立副本中进行，不在日常工作副本里切换换行。工具是 [historical_source.py](historical_source.py)，见第 9 节。
   - **新实验：** 一律用 LF。冻结前必须通过 [check_clean_source.py](check_clean_source.py)，确认源码摘要和冻结身份与干净克隆一致，见第 9 节。
3. **阶段 F 的两个脚本按平台换行写文件。** `build_specs.py` 与 `tables.py` 调用 `write_text` 时没有传 `newline="\n"`，在 Windows 上写出 CRLF。所以契约第 376 行记录的配置 SHA256 是 CRLF 文件的哈希，而 git 里的配置是 LF（哈希 `4c16e210…`）；在 Linux 或 macOS 上重跑这两个脚本，配置的哈希与契约记录的不同，两份表格只在换行上与已提交的不同（`rebuild_tables.json` 的 `identical_after_crlf_to_lf` 一项可用来区分）。后来的脚本都已改用 `newline="\n"`。这两个脚本在哈希绑定的报告目录里，本分支没有改动，建议在契约第 376 行注明“（CRLF）”。
4. **已提交的论文 PDF 图不是用锁定的 matplotlib 画的。** 图是 matplotlib 3.11.0 画的，锁定文件写的是 3.11.1，两者的 PDF 后端对字体子集的处理不同。要么在锁定环境里重画图并重编论文，要么在 `paper/README.md` 写明版本。两者都会碰到 PR #26 正在修改的 `paper/main.pdf`，所以本分支没有改。
5. **学习型企业重做的规格脚本缺领域文件就不能写出配置。** 它读领域文件只为算 `domain_sha256`，而这个值已经记在配置里。其他 6 个脚本都是先写配置、后做派生核对，所以没有领域文件也能核对配置字节，但退出码是 1，各 README 也没有说明这一点。
6. **`experiment_id` 绑定 Python 补丁版本。** 身份里记录的是 `python_version` = `3.12.9`；`pyproject.toml` 只要求 `>=3.12`，锁定文件不约束 Python 版本，README 也没写。用其他 3.12.x 或 3.13 即使有领域文件，算出的 `experiment_id` 也会不同。
7. **`.gitignore` 漏了两类只应留在本地的文件**：供应商派生的领域文件 `configs/domains/csi300_*.json`，以及联合稳健性的档案 `reports/disclosure_joint_robustness_20261023/joint_v1/`、`check_v1/`、`work_*`。前者被误提交就是再分发供应商派生数据。**本分支已补上**；补完后没有任何已跟踪文件被忽略（`git ls-files -ci --exclude-standard` 为空）。
8. **Windows 专属部分。** `_no_wmi.py` 与 `abm.platform_tuning.disable_power_throttling` 在非 Windows 平台上什么也不做；各报告脚本说明中的命令写成 `.venv/Scripts/python.exe`，`OPERATIONS.md` 只给 PowerShell 写法。CI 在 Ubuntu 上通过了测试，但本次没有在 Linux 上跑第 3–5 节的重建。
9. **论文 PDF 不可逐字节重现**：pdfTeX 默认写入编译时间和由时间生成的 `/ID`。第 3 节给出了固定时间的做法；时区写法仍会不同。

## 8. 复现本报告

在一个新目录中：

```
git clone https://github.com/kanzaler123/financial-ABM.git
cd financial-ABM
python -m venv .venv
.venv/Scripts/python.exe -m pip install -c requirements-lock.txt -e ".[all,dev]"   # macOS / Linux: .venv/bin/python
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe paper/make_figures.py && git status --short paper/figures
(cd paper && latexmk -pdf main.tex)
.venv/Scripts/python.exe reports/reproduction_acceptance_20261023/rebuild_tables.py --out ../rebuild_tables.json
.venv/Scripts/python.exe reports/disclosure_stage_g_20261015/compare.py && git status --short
for f in reports/disclosure_*/build_spec*.py; do .venv/Scripts/python.exe $f; done; git status --short configs
.venv/Scripts/python.exe reports/reproduction_acceptance_20261023/check_specs.py --out ../check_specs.json
git checkout -- configs paper showcase
```

规格脚本中 7 个会以退出码 1 结束（第 5 节），这是预期结果。`check_specs.py` 在领域文件存在时会用 `freeze_spec` 重算其余规格的 `experiment_id`，并核对领域文件的 SHA256。两个脚本只写 `--out` 指定的文件。

## 9. 换行问题的处理工具

两个工具都只比较字节和完整的冻结身份，不做任何规范化或豁免；运行器自身的源码一致性检查不变。

**旧档案：[historical_source.py](historical_source.py)。** 它做四件事：
1. 用 `git archive` 把冻结时的提交导出到本工作副本之外的新目录，导出时不做换行转换；
2. 只把 `--crlf` 列出的文件转成 CRLF，默认只有 `src/abm/mechanism_validation.py`；
3. 在副本里用副本自己的 `source_hashes()`，与档案的源码表逐文件比较，并确认 `abm` 确实是从副本导入的；
4. 给了 `--domain` 时，把领域文件复制进副本，在副本中运行 `freeze_spec`，与档案的 `experiment.json` 逐字段比较，这正是 `--resume` 要做的比较。学习型企业重做的档案改为核对规格哈希、`experiment_id` 和源码摘要。

档案只读。2026-10-23 用原作者本地的档案和领域文件实测，结果如下：

| 档案 | `--commit` | 源码表 | 冻结身份 |
|---|---|---|---|
| 阶段 F `confirmatory_v1/` | `2b08f65` | 52 个文件一致 | 逐字段一致，`90ee8606…5d57` |
| 衰减网格 `grid_v1/` | `e2cec47` | 52 个文件一致 | 逐字段一致，`19385815…7740` |
| 第 21 节 `audits_v1/` | `3fced38` | 52 个文件一致 | 逐字段一致，`06bec148…2a32` |
| 第 22 节 `assurance_v1/` | `e6854a0` | 52 个文件一致 | 逐字段一致，`35c4f767…339a` |
| 第 23 节 `redo_v1/` | `3d82afe` | 52 个文件一致 | 规格哈希、`experiment_id`、源码摘要一致 |
| 第 24 节 `joint_v1/` | `be0f850` | 52 个文件一致 | 逐字段一致，`c856a0cd…6765` |
| 第 22 节，不恢复 CRLF（`--crlf` 为空） | `e6854a0` | 1 个文件不同 | 在 `experiment_id`、`source_hashes` 上不同，退出码 1 |
| 第 24 节，给错提交 | `3d82afe` | — | 该提交里没有这份规格，退出码 1 |

`--commit` 要用首次提交该规格的提交，这时规格和源码都已经提交。上表各档案对应的提交可用 `git log --diff-filter=A -- configs/<规格>` 查到。核对通过后，从副本目录运行档案自己的脚本（如 `finalize.py --resume`），设 `PYTHONPATH=<副本>/src`，`--output` 指向原档案。

```
python reports/reproduction_acceptance_20261023/historical_source.py --archive <档案目录> --commit <冻结提交> --dest <新目录> --domain <领域文件>
```

**新实验：[check_clean_source.py](check_clean_source.py)。** 冻结前在将要运行实验的工作副本中执行 `--spec <规格>`，以下各项都满足才通过：
- `src/abm` 下每个 `*.py` 都已跟踪，并且与 HEAD 的 blob 逐字节一致。只差换行的文件会单独标出，但同样不通过；
- 源码摘要与干净克隆的相同；
- 规格已提交且未改动；
- 在工作副本和 HEAD 的干净导出中运行 `freeze_spec`，得到逐字段相同的冻结身份。

实测结果：

| 位置 | 结果 |
|---|---|
| 本次的干净克隆，`--source-only` | 通过，摘要 `c4423d93…7246` |
| 本次的干净克隆，`--spec configs/disclosure_joint_robustness_v1.json`（临时复制领域文件，用后删除） | 通过，两边的 `experiment_id` 都是 `6814b6ff…63b0` |
| 原作者的日常工作副本（只读），同一规格 | 不通过：`mechanism_validation.py` 的换行与 HEAD 不同；冻结身份在 `experiment_id`、`source_hashes` 上不同（`c856a0cd…` 对 `6814b6ff…`） |

所以在日常工作副本中的这个文件改成 LF 之前，新实验要在干净克隆中冻结和运行。
