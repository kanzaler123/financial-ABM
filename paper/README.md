# Working paper

`main.tex` is the English working paper on the stage F experiment (contract Section 20), its exploratory supplements, the targeted-audit experiment (contract Section 21) and the learning-firm redo (contract Section 23). The version of 20 October 2026 is the one prepared for SSRN; the version of 23 October 2026 adds the learning-firm redo, whose numbers come from `reports/disclosure_learning_redo_20261021/README.md` and `redo_v1_analysis.json`. Red bracketed text in the PDF marks placeholders; none remain.

Build from this directory:

```sh
python make_figures.py
latexmk -pdf main.tex
```

`make_figures.py` reads only committed analysis files under `reports/`. The two tables of the targeted-audit section are typed from `reports/disclosure_targeted_audits_20261020/audits_v1_tables.md` and `audits_v1_analysis.json`; the intervals for V_hidden − B and J_hidden − B come from the run's `summary.json`, which stays local, and match the report's README.

On 20 October 2026 every entry in `references.bib` was checked: journal and book entries against Crossref, with each DOI resolved at doi.org; `zhao2024esg` against the published PDF and the journal's table of contents; the Securities Law and the SSE guideline against their official texts.
