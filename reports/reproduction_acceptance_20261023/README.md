# 复现验收：只用公开仓库从零重建（2026-10-23）

**性质：复现验收，不是新实验。** 在一个全新目录里克隆 GitHub 上 `kanzaler123/financial-ABM` 的 master（`8afc3fc`），按锁定依赖建新虚拟环境，只用已提交的文件逐项重建测试、论文图表、论文 PDF、各报告表格和实验规格，并核对契约里记录的哈希与 `experiment_id`。没有读取原作者工作副本和其中的本地档案，也没有跑任何契约规模的仿真；除测试中的小规模运行外，唯一跑的模型是阶段 G 的简化沙盒（240 次，12 个进程）。所有步骤都在同一台 Windows 机器上完成，Linux 与 macOS 没有实测，下文凡涉及其他平台的说法都是从代码推断的。

## 结论

1. **从公开文件能重建、而且逐字节一致的：** 全部 12 个由汇总文件生成的报告表格（阶段 F 两份、衰减网格、四个反馈分解、20.21 稳健性、定向核查、鉴证、学习型企业重做、联合稳健性）；阶段 G 的简化对照（`simplification_tables.md` 与 `showcase/data/simplification.json`）；论文的 4 张 PNG 图；用 matplotlib 3.11.0 时另外 4 张 PDF 图也逐字节一致；7 个规格脚本写出的 12 份配置文件（其中阶段 F 的 2 份在 Windows 上写成 CRLF，转成 LF 后一致），以及用记录的 `domain_sha256` 生成的学习型企业重做的 2 份。全量测试 974 通过、1 跳过。
2. **能重建、但不逐字节一致的：** 用锁定的 matplotlib 3.11.1 重画的 4 张 PDF 图，绘图内容流与已提交的相同，只差字体子集与 Producer 字段（已提交的图是 3.11.0 画的）；论文 PDF 30 页的抽取文本与已提交的逐页相同，固定 `SOURCE_DATE_EPOCH` 后除日期字符串、`/ID` 和交叉引用偏移外逐字节相同。
3. **卡在本地档案（b）：** 各实验的 `*_analysis.json` 本身（要读 `summary.json`、记录和紧凑数组）、`finalize.py` 汇总、规格脚本中对阶段 F 档案的派生核对、论文中 V_hidden − B 的区间，以及契约第 21–23 节记录的“源码摘要”。最后一项**用公开文件对不上**：按 `source_hashes()` 的定义，HEAD 和 17 个改动过 `src/abm` 的提交都算不出这三个摘要，原因只能用本地档案的 `experiment.json` 查（见问题 2）。
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

除学习型企业重做外，`experiment_id` 是 `freeze_spec` 对 `{spec, raw_domain, scenarios, source_hashes, python_version, numpy_version, scipy_version}` 的规范 JSON 取 SHA-256，其中 `raw_domain` 是领域文件的全部内容，所以没有供应商派生文件就算不出来。即使有了领域文件，也还需要 Python 3.12.9、numpy 2.5.1、scipy 1.18.1（已提交的档案记录的就是这几个版本），以及冻结时的 `src/abm`。阶段 F 之后 `src/abm` 又改过 4 次，所以重算阶段 F 和 20.x 各节的 `experiment_id` 要先检出当时的提交。

**源码摘要。** 契约第 21–23 节记录了“源码摘要（各文件源码哈希的规范 JSON 的 SHA256）”：`a94bed8b…`、`99966d0c…`、`7a073291…`。按 `abm.disclosure.runner.source_hashes()` 加 `canonical_json_bytes` 的定义，当前检出的 `src/abm`（52 个文件）得 `c4423d93…7246`；从 git 历史中 17 个改动过 `src/abm` 的提交的原始字节算，没有一个提交得出这三个值中的任何一个，包括第 21、22、23 节各自“在确认性运行之前”提交的 `3fced38`、`e6854a0`、`3d82afe`。另外试过把文件转成 CRLF、换路径前缀、只取 `abm/disclosure`、换 JSON 序列化方式，在最近 25 个提交上都对不上。可见公开仓库没有给出能重现这三个摘要的定义或源码。作为参照，2026-10-11 的开发档案 `reports/disclosure_action_grid_development_20261011_v1/experiment.json` 提交了完整的 `source_hashes` 表，它与任何提交都不完全一致（最接近的 `92ff504` 也只有 37/52 个文件相同），说明那批早期开发运行用的是未单独提交的中间源码。

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
| 契约第 21–23 节的源码摘要 | (b) | 公开文件对不上（第 5 节）；要用本地档案的 `experiment.json` 中的 `source_hashes` 核对 |
| 其余各节的 `experiment_id` | (c) | 缺 `configs/domains/csi300_huazheng_esg_quantile_copula_fixed_fine_saturated.json`（SHA256 `c3dd4824…f907`） |
| 阶段 F 及以后的完整仿真 | (c) | 同上；每个规格都引用这份领域文件 |
| 领域文件本身 | (c) | 要从华证评级、账面值与市值快照重建（`reports/disclosure_public_data_20261011/collect_*.py`、`build_csi300_joint_20261012.py`，再经 fixed_fine 与 saturated 两步），原始数据与中间文件都在 `.gitignore` 中，只提交了 `summary.json` 等哈希清单 |

## 7. 发现的问题

1. **完整仿真离不开供应商派生的领域文件。** 这是授权限制下的设计选择，各报告 README 的“复现”一节也写了。对外的影响是：别人用公开仓库能核对表格、图和配置，但不能独立核对任何一个仿真结果或 `experiment_id`（学习型企业重做的除外）。
2. **契约第 21–23 节的源码摘要用公开文件重现不了**（第 5 节）。这三个摘要不进入 `experiment_id` 的计算，但契约用它们说明“与开发运行相同”。可能是记录时用的定义与 `source_hashes()` 不同，也可能是运行时导入的源码与提交的不同（第 20.21 条的 README 提到过，在工作树中运行时要让 `PYTHONPATH` 指向本工作树的 `src`）。原作者可以在本地用确认性档案的 `experiment.json` 中的 `source_hashes` 逐文件对比对应提交的 `git show <提交>:src/<路径>` 来查明。本分支只记录，没有改契约。
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
