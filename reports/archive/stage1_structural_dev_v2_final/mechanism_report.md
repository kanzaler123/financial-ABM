# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-development-v2`
- Frozen seeds: `5`
- Decision: `PASS`

## Gate checks

- [x] `full_runs_without_price_cap`
- [x] `cash_conservation`
- [x] `share_conservation`
- [x] `full_volatility_in_daily_range`
- [x] `full_return_autocorrelation_near_zero_across_lags`
- [x] `full_volatility_clustering_has_decay`
- [x] `full_heavier_than_gaussian`
- [x] `full_volume_volatility_relation_positive`
- [x] `full_price_discovery_bounded`
- [x] `single_strategy_markets_remain_active`
- [x] `ablations_avoid_pathological_return_predictability`
- [x] `fixed_population_has_no_strategy_turnover`
- [x] `persistent_order_flow_is_observable`
- [x] `spread_and_depth_are_finite`
- [x] `liquidity_stress_increases_clustering`
- [x] `garch_control_is_labelled_exogenous`

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.007411 | 0.025665 | 0.059139 | 0.109316 | 0.097777 | 0.025313 |
| value_only | 0.006755 | -0.181197 | 0.016224 | -0.012738 | 0.040958 | 0.024434 |
| trend_only | 0.006661 | 0.125075 | 0.053752 | 0.143727 | 0.031970 | 0.025390 |
| noise_only | 0.020537 | 0.165889 | 0.211725 | 3.936207 | 0.444560 | 0.034789 |
| no_logit | 0.007411 | 0.025665 | 0.059139 | 0.109316 | 0.097777 | 0.025313 |
| no_activity_persistence | 0.007423 | 0.023629 | 0.060664 | 0.107328 | 0.096636 | 0.025275 |
| independent_signals | 0.006734 | -0.058736 | 0.013613 | 0.015536 | 0.068911 | 0.024719 |
| fixed_participation | 0.008767 | -0.063183 | 0.072712 | 0.308066 | 0.133704 | 0.025595 |
| liquidity_stress | 0.010868 | -0.089827 | 0.139455 | 0.732749 | 0.171517 | 0.027538 |
| concentrated_wealth | 0.008047 | -0.023441 | 0.094798 | 0.290854 | 0.137093 | 0.025203 |
| garch_t_control | 0.006704 | 0.010428 | 0.125770 | 1.907860 | 0.087620 | 0.018938 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.063192 |
| Absolute-return ACF(5) | 0.036267 |
| Absolute-return ACF(20) | 0.014576 |
| Absolute-return ACF(50) | 0.018200 |
| Volume ACF(1) | 0.973578 |
| OFI ACF(1) | 0.884426 |
| OFI-return correlation | 0.101314 |
| Mean spread (bps) | 5.284607 |
| Three-sigma tail fraction | 0.004000 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
