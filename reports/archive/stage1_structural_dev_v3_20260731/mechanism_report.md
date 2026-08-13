# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-development-v3`
- Frozen seeds: `5`
- Decision: `FAIL`

## Gate checks

- [x] `full_runs_without_price_cap`
- [x] `cash_conservation`
- [x] `share_conservation`
- [x] `full_volatility_in_daily_range`
- [x] `full_return_autocorrelation_near_zero_across_lags`
- [x] `full_volatility_clustering_has_decay`
- [ ] `full_heavier_than_gaussian`
- [x] `full_volume_volatility_relation_positive`
- [x] `full_price_discovery_bounded`
- [x] `public_news_channel_has_paired_price_response`
- [ ] `agent_information_channel_has_paired_price_response`
- [x] `empty_information_baseline_is_weakest_discovery`
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
| full | 0.011651 | -0.052147 | 0.034889 | 0.050773 | 0.058119 | 0.029701 |
| no_direct_news | 0.006308 | 0.410332 | 0.213682 | 0.448395 | 0.026627 | 0.538620 |
| no_agent_information | 0.011470 | 0.045593 | 0.042483 | 0.032899 | 0.067588 | 0.027528 |
| no_information_channels | 0.006582 | 0.456785 | 0.229493 | 0.649848 | 0.019216 | 0.525186 |
| value_only | 0.011281 | -0.205104 | 0.045335 | -0.094822 | 0.074625 | 0.053172 |
| trend_only | 0.011160 | 0.043198 | 0.041394 | 0.043449 | 0.022632 | 0.057040 |
| noise_only | 0.020250 | 0.127555 | 0.167552 | 1.835259 | 0.393028 | 0.080508 |
| no_logit | 0.011651 | -0.052147 | 0.034889 | 0.050773 | 0.058119 | 0.029701 |
| no_activity_persistence | 0.011638 | -0.052002 | 0.034978 | 0.051545 | 0.062839 | 0.026574 |
| independent_signals | 0.011162 | -0.074809 | 0.015462 | -0.098308 | 0.032273 | 0.041441 |
| fixed_participation | 0.012384 | -0.090598 | 0.033382 | 0.122685 | 0.074607 | 0.027273 |
| liquidity_stress | 0.013909 | -0.038458 | 0.091229 | 0.286972 | 0.164712 | 0.052277 |
| concentrated_wealth | 0.012066 | -0.067305 | 0.052076 | 0.060132 | 0.078809 | 0.042427 |
| garch_t_control | 0.010146 | -0.027699 | 0.142264 | 3.395936 | 0.064888 | 0.029520 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.075903 |
| Absolute-return ACF(5) | 0.032003 |
| Absolute-return ACF(20) | 0.003189 |
| Absolute-return ACF(50) | 0.001456 |
| Volume ACF(1) | 0.971569 |
| OFI ACF(1) | 0.855270 |
| OFI-return correlation | 0.093119 |
| Mean spread (bps) | 5.449962 |
| Three-sigma tail fraction | 0.003000 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
