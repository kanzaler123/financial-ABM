# Research corrections — 16 November 2026

This note corrects two issues found in the review of commit `6f740dfe333763e2b81b5835e77f4aa450e7a4e4`. It supersedes the affected interpretations, while preserving historical files under `reports/` byte for byte. Contract section 28 records the correction. It is a post-publication correction, not a new pre-registration or a new confirmatory result.

| Item | Status |
|---|---|
| Auxiliary distance-profile interval implementation | Corrected; public synthetic regression tests pass |
| Actual H2 and I v3 auxiliary intervals | **Await recomputation from the original sampled-grid archives** |
| H3 equal-budget point differences | Recomputed from the committed public scenario means below |
| H3 equal-budget paired confidence intervals | **Await the original per-seed H3 summary** |
| Joint empirical calibration | Still unresolved; this code correction does not fit a new model |

## 1. Auxiliary profile intervals used the wrong reference minimum

The archived estimator in [`estimate.py`](../reports/disclosure_persistence_estimation_20261026/estimate.py), function `profile_intervals`, defines the full-covariance distance `J* = g' Ω⁻¹ g`. It then subtracts `J*(theta)`, where `theta` was estimated using the different diagonal-weight criterion `J = g' W g`. In general, that point does not minimize `J*`.

Contract 25.15 requires `J*_profile − min J* ≤ 3.841458820694124`. Using the old point as the reference can accept parameter values that would be excluded using the actual minimum. Stage I v3 imports the same H2 helper, so its auxiliary profiles have the same issue.

The corrected pure function is [`distance_profile_intervals`](../src/abm/research_inference.py). It:

- recomputes the unrestricted bounded minimum of the **same** full-covariance distance before forming differences;
- keeps the original covariance fixed throughout that calculation;
- uses deterministic multiple starts, including supplied grid nodes, and a bounded fallback for failed line searches;
- rejects unsuccessful/nonfinite optimizations instead of using their coordinates as valid interval endpoints;
- supports parameter bounds and fixed nuisance parameters, includes the grid endpoints, and checks for a constrained profile below the unrestricted reference;
- records the reference point, both objective values, every profiled grid point, accepted segments, and whether the accepted range touches the parameter bounds.

The historical cutoff and 61-point grid are retained. These remain auxiliary grid profiles: endpoints are not interpolated, the interval envelope may contain gaps (reported separately), and the cutoff does not acquire a new boundary-adjusted or misspecification-robust coverage guarantee. Multiple starts provide a numerical search, not a proof of a global minimum.

### Regression evidence and affected results

In the public linear example in [`test_research_audit_fixes.py`](../tests/test_research_audit_fixes.py), the diagonal-weight estimate is `(0.2, 0)`. At that point `J* = 361.025641`; the minimum of `J*` is `83.333333`, at approximately `(-0.75, 0)`. On the same 61-point grid, the historical calculation accepts the envelope `[-1.666667, 0.2]`, while the corrected envelope is `[-0.8, -0.666667]`, agreeing with the analytic profile.

This example establishes the implementation error; it is **not** a replacement estimate for the empirical study. The committed H2 output contains `κ_T = [0.25, 0.845833…]` and `s_P = [0.1, 0.7]`. Those auxiliary ranges and the interpretation based on their width should not be reused pending recomputation. The sampled grid signals and grid moment surface are not in the public checkout; the existing point estimates alone cannot reconstruct corrected ranges.

The original point estimator, fit statistic, recovery test and policy contrasts do not read the profile interval output. This correction therefore does not automatically change their results. It also does not resolve their separately recorded fit and identification problems.

### Safe recomputation entry point

The old `reports/.../estimate.py` and `estimate_v3.py` are retained for historical reproduction. **Use the new entry point for corrected auxiliary profiles:**

```sh
# Existing sampled archives only: these commands do not launch simulations.
python scripts/recompute_research_corrections.py profiles --stage h2 --grid /path/to/grid_v1 --output runs/review_20261116/h2_profiles.json
python scripts/recompute_research_corrections.py profiles --stage i3 --grid /path/to/grid_v3 --output runs/review_20261116/i3_profiles.json
```

The command checks the recorded experiment identity and seed set, reconstructs the original interpolation from sampled signals, and verifies that moments at the archived point estimates match the committed estimation. It then recalculates only the auxiliary profiles. It records input hashes and the old ranges alongside the corrections. `--forms Q` restricts the calculation to Q if needed. Omit `--output` to print JSON.

Outputs must be new files outside `reports/` and the input archive. Existing outputs cannot be overwritten. An absent or mismatched archive causes an explicit error; it never triggers a fresh simulation or substitutes synthetic data for the original study.

## 2. H3's “same-budget” comparator was A2, a double-budget policy

The [H3 report, auxiliary analysis](../reports/disclosure_stage_h3_20261027/README.md) and contract 25.8 described `V − A2` and `J − A2` as comparisons against random audits with the same budget. This description is incorrect.

The [H3 specification](../configs/disclosure_stage_h3_v1.json) retains the baseline audit probability for V and J, changing only the allocation rule. A2 explicitly raises the audit probability to 0.02. The original targeted-audit design in [`paper/main.tex`](../paper/main.tex), section “Targeted audits”, defines the baseline as 0.01. The allocation mechanism preserves `n × audit_probability`, as documented in [`policy.py`](../src/abm/disclosure/policy.py). With 200 firms:

| Scenario | Audit probability budget per firm and period | Expected audits per period |
|---|---:|---:|
| B: random | 0.01 | 2 |
| V: value-targeted | 0.01 | 2 |
| J: jump-targeted | 0.01 | 2 |
| A2: random, doubled | 0.02 | 4 |

### Corrected point differences

Source: [`h3_v1_analysis.json`](../reports/disclosure_stage_h3_20261027/h3_v1_analysis.json), `describe.<scenario>.true_change`; all four means use 30 completed seeds. Metric: `true_change.equal_weight_sd`.

| Scenario | Public mean |
|---|---:|
| B | 0.057346261054776616 |
| V | 0.03641966316277364 |
| J | 0.04757899214744154 |
| A2 | 0.06733530554081593 |

| Comparison | Correct interpretation | Mean difference (SD) | Paired 95% interval |
|---|---|---:|---|
| V − B | Same expected number of audits | **−0.020926597892002977** | Pending per-seed archive |
| J − B | Same expected number of audits | **−0.009767268907335078** | Pending per-seed archive |
| V − A2 | Against twice as many random audits | −0.030915642378042292 | Historical interval applies only to this contrast |
| J − A2 | Against twice as many random audits | −0.019756313393374393 | Historical interval applies only to this contrast |

The point differences remain negative. Their uncertainty cannot be recovered from four means: the covariance of outcomes within each paired seed is required. No confidence interval, p-value or significance verdict is inferred here, and the A2 intervals must not be reused for the corrected contrasts. These auxiliary corrections do not replace H3's eight primary questions.

Reproduce the public calculation, or supply the original per-seed archive when available:

```sh
python scripts/recompute_research_corrections.py h3
python scripts/recompute_research_corrections.py h3 --archive /path/to/h3_v1 --output runs/review_20261116/h3_paired.json
```

Without `--archive`, confidence-interval fields are explicitly `null` with status `not_computable_from_scenario_means`. With an archive, the script checks matching experiment IDs, complete unique seed pairs and agreement with the published scenario means. It computes unadjusted paired t intervals by seed; missing, duplicated or inconsistent observations cause an error instead of silently dropping evidence.

## 3. Remaining empirical limits and current magnitudes

The [H2 report](../reports/disclosure_persistence_estimation_20261026/README.md) records rejection of all three candidate natural dynamics and a failed parameter recovery check. The [H3 report](../reports/disclosure_stage_h3_20261027/README.md) reruns the main comparisons under the closest candidate, Q, without claiming that it fits. Its main-block information-minus-fine effect is about **0.003 SD**, versus **0.205 SD** in the original setting; information-minus-baseline is **0.017 SD**, versus **0.331 SD**. With lasting improvement instead, the H3 `perm_` block reports **0.077 SD** for information-minus-fine. Thus a preserved direction is not an invariant policy magnitude.

The [Stage I report](../reports/disclosure_stage_i_20261027/README.md) records the difficulty of matching the financing association without ratings bunching at the upper bound and losing empirical persistence. The [Stage J report](../reports/disclosure_investor_response_20261028/README.md) records a stronger investor response in the model than in its data counterpart, with measurement differences stated. No empirical parameter or policy dose is changed by this correction. Joint calibration, identification of improvement persistence, out-of-sample evaluation and equal-cost policy comparisons remain separate research tasks.

## 4. Implementation corrections and reproducibility

New assurance runs keep check-day ledgers for firm fees and provider effort. A fee enters the firm's private outflow once; it is a transfer, while provider effort is included in the resource-cost denominator. Regulator costs remain unmodelled. Old assurance archives without these ledgers remain loadable, but missing cost totals are reported as unknown rather than zero.

Assurance checks now use a separate random stream derived from the seed and calendar day. The same firm's draw on the same day stays paired across assurance scopes and policy start dates. New manifests identify this as `independent_assignment_and_calendar_checks_v2`. **The historical assurance effects and intervals have not been rerun under this protocol**; source hashes distinguish new runs from the preserved evidence.

Two edge cases are also corrected: a zero imitation-update fraction performs no updates, and zero discounting together with a zero future-audit probability produces a finite expected-fine factor. Public synthetic regressions cover these changes, assurance accounting, archive compatibility and actual experiment resume.

Runtime fixes allow workers to exit when display queues are unconsumed, stream appended telemetry without rereading the whole JSONL history, and retain failed runs and their diagnostic downloads. These changes affect observation and failure handling; durable archives remain the source for replay. Public mechanism tests no longer require the licensed rating domain; a separate empirical dependency extra provides pandas and openpyxl for the estimation scripts.

## Source fingerprints and validation

| Preserved input | SHA-256 |
|---|---|
| `reports/disclosure_stage_h3_20261027/h3_v1_analysis.json` | `fcfeefa7817017fc85d68a55b0f664682130c10504eb5436224f41f4f8d0b6b2` |
| `reports/disclosure_persistence_estimation_20261026/estimation_v1.json` | `ff8aac22805f62ad0a2d987162f738557fa537c2314a95e771ef386214218f4e` |
| `reports/disclosure_stage_i_20261027/estimation_v3.json` | `aefd0698372bf8194cb84dffbec021541bc99b68b4842c540e731601f98a4dfe` |

Validation uses public synthetic cases for correlated covariance, boundary optima, fixed nuisance parameters, optimizer failure, invalid inputs, seed alignment, missing/duplicate records and protection of historical outputs:

```sh
python -m pytest tests/test_research_audit_fixes.py
```

No licensed-grid recomputation or new market simulation was performed for this correction.
