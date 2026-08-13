# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-acceptance-v1`
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
| full | 0.013608 | 0.051067 | 0.054426 | 0.084531 | 0.404241 | 0.012023 |
| value_only | 0.106655 | -0.977689 | 0.873447 | -1.477509 | 0.975966 | 0.047352 |
| trend_only | 0.000612 | 0.128397 | 0.012751 | 0.024593 | 0.942836 | 0.082967 |
| noise_only | 0.050011 | -0.015395 | 0.125079 | 0.081261 | 0.783185 | 0.365394 |
| no_logit | 0.012395 | 0.122208 | 0.063479 | 0.087951 | 0.408389 | 0.011911 |
| independent_signals | 0.007985 | 0.544267 | 0.308959 | 0.083012 | 0.485851 | 0.009586 |
| fixed_participation | 0.007180 | 0.388765 | 0.148263 | 0.037953 | 0.467081 | 0.012166 |
| liquidity_stress | 0.041730 | -0.580924 | 0.525752 | 0.569548 | 0.691750 | 0.019980 |
| concentrated_wealth | 0.014318 | 0.006643 | 0.047942 | 0.156965 | 0.439873 | 0.012048 |
| garch_t_control | 0.011896 | -0.029421 | 0.107891 | 0.367492 | 0.440386 | 0.009773 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
