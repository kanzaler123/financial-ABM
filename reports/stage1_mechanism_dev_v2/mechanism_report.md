# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-dev-v1`
- Frozen seeds: `32`
- Decision: `PASS`

## Gate checks

- [x] `full_runs_without_price_cap`
- [x] `cash_conservation`
- [x] `share_conservation`
- [x] `full_volatility_in_daily_range`
- [x] `full_return_autocorrelation_near_zero`
- [x] `full_volatility_clustering_positive`
- [x] `full_heavier_than_gaussian`
- [x] `full_volume_volatility_relation_positive`
- [x] `full_price_discovery_bounded`
- [x] `value_strategy_anchors_price`
- [x] `trend_strategy_creates_persistence`
- [x] `logit_changes_strategy_ecology`
- [x] `common_beliefs_increase_tail_weight`
- [x] `adaptive_participation_reduces_return_predictability`
- [x] `liquidity_stress_increases_clustering`
- [x] `garch_control_is_labelled_exogenous`

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.013873 | 0.060682 | 0.072830 | 0.121781 | 0.386974 | 0.012611 |
| value_only | 0.109475 | -0.977928 | 0.876856 | -1.536617 | 0.978058 | 0.048845 |
| trend_only | 0.000613 | 0.118330 | 0.024197 | -0.086454 | 0.947737 | 0.088974 |
| noise_only | 0.059016 | 0.004781 | 0.125728 | -0.176451 | 0.759924 | 0.268390 |
| no_logit | 0.012847 | 0.129862 | 0.082802 | 0.064928 | 0.394006 | 0.012176 |
| independent_signals | 0.007984 | 0.547169 | 0.311568 | -0.025146 | 0.436436 | 0.010103 |
| fixed_participation | 0.007276 | 0.419491 | 0.157037 | -0.064600 | 0.442297 | 0.012681 |
| liquidity_stress | 0.036913 | -0.449390 | 0.513999 | 1.542022 | 0.638355 | 0.023074 |
| concentrated_wealth | 0.013861 | 0.068050 | 0.078823 | 0.185276 | 0.383063 | 0.013198 |
| garch_t_control | 0.010998 | -0.009643 | 0.095321 | 0.295353 | 0.421191 | 0.010530 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
