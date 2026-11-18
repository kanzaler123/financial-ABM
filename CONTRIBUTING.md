# Contributing to Financial ABM

Thanks for your interest. This is a research codebase: an agent-based market in which investors, firms and a regulator interact, used to compare disclosure policies such as ESG transparency and fines. You can help without knowing the whole project. Most useful contributions touch one file.

[中文说明见文末](#中文简要说明)

## Ways to help

| You want to… | Start here | Typical size |
|---|---|---|
| Try a policy idea | Copy [`configs/disclosure_quickstart_v1.json`](configs/disclosure_quickstart_v1.json), change a scenario, run it, and share the report in an issue | 30 minutes |
| Model another disclosure problem | Copy [`configs/domains/esg.json`](configs/domains/esg.json) into a new domain file (carbon claims, earnings management, AI-washing…) and run the quickstart with it | an afternoon |
| Improve the in-browser desk | [`showcase/sandbox/`](showcase/sandbox): page text is in `i18n.js` (Chinese and English), page logic in `app.js` | small PR |
| Fix docs or translations | `README.md`, `README.zh-CN.md`, [`docs/`](docs) | small PR |
| Fix a bug | Issues labelled `bug`; add a test that fails before the fix | small PR |

Issues labelled [`good first issue`](https://github.com/kanzaler123/financial-ABM/labels/good%20first%20issue) are scoped so that you do not need to understand the whole model. Comment on one before you start so two people don't work on the same thing.

## Setup

Python 3.12 or newer.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS / Linux: source .venv/bin/activate
python -m pip install -c requirements-lock.txt -e ".[all,dev]"
python -m pytest
```

`[all]` includes the desktop GUI (PySide6) and the empirical-data dependencies.
For the test suite without the desktop GUI, install `".[plot,research,web,dev]"`;
only the Qt-specific tests then skip, while worker, telemetry and web tests still run.
For empirical estimation scripts, also install `".[empirical]"` (pandas and openpyxl).
The persistence and rating-channel mechanism tests use a public synthetic domain,
independently of licensed observations. The remaining vendor-backed showcase
comparison reports its skip explicitly when its source data is unavailable.

Check that the model runs (10–30 seconds):

```sh
python -m abm.disclosure.experiments --spec configs/disclosure_quickstart_v1.json --output runs/quickstart --workers 2
```

Check both completion and the validation results in the report. The quickstart
retains price-guard failures as diagnostic evidence; it is a development example,
not a calibrated acceptance run. Do not select new seeds to hide failed checks.

The [November 2026 review corrections](docs/review_corrections_20261116.md)
describe the statistical errata, their new recomputation entry point, and the
historical results that still require access to the original archives.

For the in-browser desk you need Node.js 22 as well:

```sh
node showcase/sandbox/check_core.js
python -m unittest discover -s showcase -p "test_*.py"
python showcase/build.py --output _site
python -m http.server 8080 --directory _site
```

## Research rules

These rules keep the results trustworthy. A pull request that breaks them will be asked to change, however good the code is.

1. **Settings come before results.** Do not tune parameters until a policy ranking comes out the way you expect. Confirmatory designs are written into the [research contract](docs/disclosure_research_contract.md) before they run. Exploratory runs are fine; label them as development.
2. **Failures stay.** Do not delete failed runs, failed checks or inconvenient reports. Add a note explaining them.
3. **Archived evidence is byte-exact.** Never edit files under `reports/` that have recorded SHA-256 hashes. On Windows, write such files with LF line endings.
4. **Every empirical number has a source.** New parameters cite a source and an evidence level, as in the [evidence table](docs/disclosure_empirical_magnitudes_20261012.md).
5. **The core stays domain-neutral.** Code under `src/abm/disclosure` must not contain ESG-specific vocabulary; domain words belong in the domain JSON. A test enforces this.
6. **No vendor data.** Huazheng, Eastmoney and other vendor data have no open redistribution licence. Commit collection scripts and hash manifests, not the data.
7. **The browser core matches Python.** If you change the firm decision rule in Python, update `showcase/sandbox/core.js` and its fixtures so that `check_core.js` still passes.

## Pull requests

- Keep each pull request to one change, and explain what it changes and why.
- Add or update tests. CI runs `python -m pytest`, the showcase tests and the web build.
- Every page text key must exist in both languages in `showcase/sandbox/i18n.js`; a test checks this.
- Results in README files must match the exported numbers; a test checks the stage F figures.
- By contributing you agree that your work is released under the [MIT licence](LICENSE).

## Questions and ideas

Open an issue with the matching template: bug report, policy scenario or new domain. For a question, use the question template. Please be kind and specific; this project follows the spirit of the [Contributor Covenant](https://www.contributor-covenant.org/).

## 中文简要说明

欢迎参与。不需要先理解整个项目，大多数有用的贡献只涉及一个文件：

- **试一个政策想法**：复制 `configs/disclosure_quickstart_v1.json`，改一个场景，运行后把报告贴到 issue 里。
- **换一个领域**：复制 `configs/domains/esg.json`，写碳排放声明、盈余管理、AI-washing 等新领域。
- **改进浏览器作战台**：文案在 `showcase/sandbox/i18n.js`（中英文都要有），逻辑在 `app.js`。
- **修文档、翻译或 bug**：修 bug 时请附一个修复前会失败的测试。

研究规则：设定先于结果，不为得到预期排序而调参；失败的运行和报告不删除；`reports/` 中记录了哈希的文件不得修改；经验参数要注明出处；核心代码不出现领域词汇；不提交供应商数据；修改 Python 决策规则时同步更新 `core.js` 并通过 `check_core.js`。开始做某个 `good first issue` 前，请先在下面留言。
