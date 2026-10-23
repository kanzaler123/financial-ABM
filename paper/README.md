# Working paper

`main.tex` is the English working paper on the stage F experiment (contract Section 20), its exploratory supplements, the targeted-audit experiment (contract Section 21), the assurance experiment (contract Section 22), the learning-firm redo (contract Section 23) and the joint robustness test (contract Section 24). The version of 20 October 2026 is the one first posted on SSRN; the version of 23 October 2026 adds the assurance experiment, the learning-firm redo and the joint robustness test, and describes the mean-reversion rates as data-motivated sensitivity values rather than upper bounds. Red bracketed text in the PDF marks placeholders; none remain.

Build from this directory:

```sh
python make_figures.py
latexmk -pdf main.tex
```

`make_figures.py` reads only committed analysis files under `reports/`. The committed figures were drawn with matplotlib 3.11.1, the version in `requirements-lock.txt`. The two tables of the targeted-audit section are typed from `reports/disclosure_targeted_audits_20261020/audits_v1_tables.md` and `audits_v1_analysis.json`; the intervals for V_hidden − B and J_hidden − B come from the run's `summary.json`, which stays local, and match the report's README. The two tables of the assurance section and its numbers are typed from `reports/disclosure_assurance_20261021/README.md` and `assurance_v1_analysis.json`; the numbers of the learning-firm redo come from `reports/disclosure_learning_redo_20261021/README.md` and `redo_v1_analysis.json`; the joint-test rows of the robustness table and the numbers of the paragraph "Both assumptions together" come from `reports/disclosure_joint_robustness_20261023/README.md` and `joint_v1_analysis.json`.

On 20 October 2026 every entry in `references.bib` was checked: journal and book entries against Crossref, with each DOI resolved at doi.org; `zhao2024esg` against the published PDF and the journal's table of contents; the Securities Law and the SSE guideline against their official texts. On 23 October 2026 the provisions added for the assurance section, Article 213(3) of the Securities Law and Article 58 of the SSE guideline, were checked against the official texts, and the Securities Law entry now names the article at each citation instead of in its title.
