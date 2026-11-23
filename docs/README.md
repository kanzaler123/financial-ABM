# Documentation Index

> **Overview.** This repository contains two lines of work:
> 1. **Single-asset artificial market (Stage 1–3)** with a frozen, formally accepted Stage 1 foundation (`v15`, 22/22 checks).
> 2. **Multi-agent disclosure game (`src/abm/disclosure/`)**: 1,000 heterogeneous investors (profit-maximizing vs. impact) trade 200 firms choosing between real improvement, report inflation (greenwashing), and maintaining under audits, fines, transparency, and strategic assurance providers. The disclosure domain (`esg`, `ai_washing`, `wba_social_2023`) is configured via JSON.
>
> Internal drafting plans, handoff notes, and manuscript sources are maintained privately; this index lists the public research specifications, audit records, and evidence reports.

## Research documents

| Document | Language | Purpose |
|---|---|---|
| [disclosure_research_contract.md](disclosure_research_contract.md) | Chinese (+ English §28 addendum) | Pre-registration contract: fixed experimental designs, hypotheses, seeds, and acceptance rules (Stages A–J, §1–§28) |
| [review_corrections_20261116.md](review_corrections_20261116.md) | English | Post-review statistical corrections (H2/I3 auxiliary profiles, H3 equal-budget contrasts) and v2 assurance protocol fixes (`7cdb384`) |
| [disclosure_empirical_magnitudes_20261012.md](disclosure_empirical_magnitudes_20261012.md) | Chinese | Statutory penalty, enforcement intensity, and ESG financing literature sources mapped to model units |
| [disclosure_inspection_propositions.md](disclosure_inspection_propositions.md) | Chinese | Analytical single-firm benchmarks (Propositions 1–2: lump-sum fine condition and transparency threshold) |
| [disclosure_fine_booking_20261020.md](disclosure_fine_booking_20261020.md) | Chinese | Legal and enforcement case study on per-case vs. per-report fine booking |
| [disclosure_information_design.md](disclosure_information_design.md) | Chinese | Minimum mechanism design and acceptance checks for the disclosure transparency policy |
| [disclosure_q_learning_design.md](disclosure_q_learning_design.md) | Chinese | Comparable tabular Q-learning design and training-budget protocol (coverage target unmet; kept for record) |
| [data_inventory.md](data_inventory.md) | Chinese | File list, SHA-256 hashes, and quality checks for locally stored licensed empirical datasets |
| [technical_report.md](technical_report.md) | English | Stage 1 single-asset market MSM calibration technical report |

## Project entry points

- [README.md](../README.md) (English default) · [README.zh-CN.md](../README.zh-CN.md) (Simplified Chinese)
- [paper/README.md](../paper/README.md) — Figure and descriptive-statistic reproduction scripts
- [OPERATIONS.md](../OPERATIONS.md) — Detailed operating manual for the Stage 1–3 single-asset application
- [CONTRIBUTING.md](../CONTRIBUTING.md) — Development setup and research rules
- [showcase/README.md](../showcase/README.md) — Browser policy desk build and verification

## Disclosure model code layout

Core modules live in [`src/abm/disclosure/`](../src/abm/disclosure/) and domain configurations in [`configs/domains/`](../configs/domains/). Tests enforce that core modules contain no domain-specific vocabulary.

| Module | Role |
|---|---|
| `config.py` | Market configuration (`MarketConfig`) |
| `firms.py`, `market.py`, `investors.py`, `harness.py` | Firm size/value, multi-asset quoting, investor imitation, daily clearing, and conservation checks |
| `domain.py` | Domain mapping: direction, scale, empirical distributions, size copula, costs, and lags |
| `game.py`, `policy.py`, `financing.py` | Firm actions, audits/fines/assurance, policy schedules, financing signals, and payoffs |
| `learning.py`, `training.py` | Tabular Q-learning, training loop, and state–action coverage diagnostics |
| `experiments.py`, `metrics.py`, `phase.py`, `report.py`, `runner.py` | Scenario grids, paired contrasts, metrics, phase diagrams, reporting, and archival runs |
| `validate.py`, `calibration.py` | Market validation and empirical moment/distribution fitting |

## Archived experiment reports

| Area | Reports |
|---|---|
| **Confirmatory policy & mechanism studies** | [Stage F: Policy instruments](../reports/disclosure_stage_f_20261014/README.md) · [Targeted audits (E.1)](../reports/disclosure_targeted_audits_20261020/README.md) · [Assurance providers (E.2)](../reports/disclosure_assurance_20261021/README.md) · [Learning-firm redo (D.3)](../reports/disclosure_learning_redo_20261021/README.md) · [Joint robustness](../reports/disclosure_joint_robustness_20261023/README.md) · [Counterfactual feedback channels](../reports/disclosure_feedback_channels_main_20261017/tables.md) |
| **Empirical calibration & regime studies** | [Huazheng panel dynamics (R1)](../reports/huazheng_panel_dynamics_20261026/README.md) · [Persistence estimation (H2)](../reports/disclosure_persistence_estimation_20261026/README.md) · [Persistence rerun (H3)](../reports/disclosure_stage_h3_20261027/README.md) · [Financing premium (I)](../reports/disclosure_financing_premium_20261027/README.md) · [Stage I model](../reports/disclosure_stage_i_20261027/README.md) · [Investor response (J)](../reports/disclosure_investor_response_20261028/README.md) |
| **Market layer & data qualification** | [60-world audit](../reports/disclosure_market_qualification_20261011/audit_report.md) · [Order-flow replay](../reports/disclosure_market_qualification_20261011/flow_replay/interpretation.md) · [Public data overview](../reports/disclosure_public_data_20261011/README.md) · [CSI 300 joint sample](../reports/disclosure_public_data_20261011/csi300_joint_20261009/README.md) · [Public reproduction acceptance](../reports/reproduction_acceptance_20261023/README.md) |
| **Single-asset foundation (Stage 1–3)** | [Stage 1 v15 formal acceptance](../reports/stage1_iter6_20260926/analysis.md) · [Stage 2 report](../reports/stage2_20260926/analysis.md) · [Stage 3 report](../reports/stage3_20261003/analysis.md) |

## What is excluded from the public repository

- **Licensed third-party datasets** (Huazheng, CNRDS, Wind, Eastmoney, BaoStock, CCDC curves, Tsay textbook data): no open redistribution licence. Collection scripts and SHA-256 manifests are preserved under `reports/` and [`data_inventory.md`](data_inventory.md).
- **Large regenerable run archives** (`.npz` paths, per-trial source zips, trained Q-tables, oversized audit JSONs) and **generated figure binaries** (`paper/figures/`): excluded via [`.gitignore`](../.gitignore) and [`paper/.gitignore`](../paper/.gitignore).
- **Manuscript sources and internal working notes**: maintained outside the public tree (see [`paper/README.md`](../paper/README.md)).
- Evidence directories under `reports/` are stored byte for byte (see [`.gitattributes`](../.gitattributes)) so recorded SHA-256 digests remain directly verifiable.
