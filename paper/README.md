# Paper Reproduction Package

This directory contains the scripts required to reproduce the figures and descriptive composition statistics for the working paper:

> *Greenwashing Moves to Where Enforcement Is Weakest: Pre-Registered Experiments in an Artificial Market*
> Working paper available on [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7640081).

LaTeX manuscript sources, bibliographies, and submission drafts are maintained separately from this public code and evidence repository; generated figure binaries (`figures/*.pdf`, `figures/*.png`) are gitignored and rebuilt deterministically from the committed experiment archives under [`reports/`](../reports/).

## Reproduce all figures

From this directory (with the project virtual environment activated and `matplotlib` installed via `pip install -c requirements-lock.txt -e ".[plot,research]"`):

```sh
python make_figures.py
```

`make_figures.py` reads only committed JSON analysis files under `reports/`:

| Analysis file | Figures generated |
|---|---|
| `reports/disclosure_stage_f_20261014/confirmatory_v1_analysis.json` | `fig_frontier`, `fig_size` |
| `reports/disclosure_decay_grid_20261016/grid_v1_analysis.json` | `fig_threshold`, `fig_decay` |
| `reports/disclosure_robustness_20261018/robust_v1_analysis.json` | `fig_decay` |
| `reports/disclosure_joint_robustness_20261023/joint_v1_analysis.json` | `fig_decay` |
| `reports/disclosure_targeted_audits_20261020/audits_v1_analysis.json` | `fig_targeted` |
| `reports/disclosure_assurance_20261021/assurance_v1_analysis.json` | `fig_assurance` |
| `reports/disclosure_stage_h3_20261027/h3_v1_analysis.json` | `fig_regimes` |
| `reports/disclosure_stage_i_20261027/stage_i_v2_analysis.json` | `fig_regimes` |

Outputs are written to `paper/figures/` as both vector PDF and PNG (`fig_decay`, `fig_frontier`, `fig_threshold`, `fig_size`, `fig_targeted`, `fig_assurance`, and `fig_regimes`).

## Descriptive composition shares

```sh
python composition.py
```

`composition.py` derives the descriptive inflation-composition shares (by firm size tier and by prior undetected-inflation status) from the `describe` blocks of `audits_v1_analysis.json` and `assurance_v1_analysis.json`. Its final check—the share of assured firms' inflating decisions at the year's lowest-quality provider without natural decay (`97.4%`, 12,262 of 12,594)—reads the local assurance run archive when present.

## Evidence and reconciliation provenance

- **Pre-registered experiment archives (`reports/`):** Stored byte for byte (see [`.gitattributes`](../.gitattributes)); all primary experiments were pre-registered in [`docs/disclosure_research_contract.md`](../docs/disclosure_research_contract.md) before execution.
- **Post-review corrections:** See [`docs/review_corrections_20261116.md`](../docs/review_corrections_20261116.md) (contract §28) for the corrected H3 equal-budget contrasts (`V − B`, `J − B`), omitted H2/I3 auxiliary profile intervals, and the version boundary between the archived v1 assurance runs (`6f740df`) and the subsequent v2 protocol fix (`7cdb384`).
- **Model specification:** Formal equations and microstructure parameters correspond directly to [`src/abm/disclosure/`](../src/abm/disclosure/).
