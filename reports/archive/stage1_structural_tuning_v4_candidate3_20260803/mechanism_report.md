# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-tuning-v4`
- Frozen seeds: `5`
- Decision: `FAIL`

## Gate checks

- [x] `full_runs_without_price_cap`
- [x] `cash_conservation`
- [x] `share_conservation`
- [x] `full_volatility_in_daily_range`
- [x] `full_return_autocorrelation_near_zero_across_lags`
- [ ] `full_volatility_clustering_has_decay`
- [x] `full_heavier_than_gaussian`
- [x] `endogenous_liquidity_feedback_increases_tail_weight`
- [x] `full_volume_volatility_relation_positive`
- [ ] `full_price_discovery_bounded`
- [x] `public_news_channel_has_paired_price_response`
- [ ] `agent_information_channel_has_paired_price_response`
- [x] `empty_information_baseline_is_weakest_discovery`
- [x] `single_strategy_markets_remain_active`
- [ ] `ablations_avoid_pathological_return_predictability`
- [x] `fixed_population_has_no_strategy_turnover`
- [x] `persistent_order_flow_is_observable`
- [x] `spread_and_depth_are_finite`
- [ ] `liquidity_stress_increases_clustering`
- [x] `garch_control_is_labelled_exogenous`

## Gate evidence

| Check | Observed | Threshold | Pass rate | Failed seeds |
|---|---:|---:|---:|---|
| `full_volatility_in_daily_range` | `0.012565203810315548` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.07816470392018425` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.08403461513500679` | `0.03` | 40.0% | 20260802, 20260803, 20260805 |
| `full_heavier_than_gaussian` | `0.5871409306327671` | `0.1` | 80.0% | 20260802 |
| `full_volume_volatility_relation_positive` | `0.207210527370104` | `0.05` | 80.0% | 20260802 |
| `full_price_discovery_bounded` | `0.06048937730634883` | `0.1` | 60.0% | 20260802, 20260805 |
| `endogenous_liquidity_feedback_increases_tail_weight` | `0.3797902475570649` | `0.05` | 80.0% | 20260802 |
| `public_news_marginal_response` | `6.032360290755126` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `4.747102883147674` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.052430136346205` | `1.0` | 80.0% | 20260801 |
| `agent_information_independent_response` | `1.0163985544952174` | `1.0` | 60.0% | 20260801, 20260805 |
| `empty_information_baseline_is_weakest_discovery` | `6.516394886649588` | `1.0` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.8705238004584911` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.012565 | -0.020570 | 0.084035 | 0.587141 | 0.207211 | 0.060489 |
| no_direct_news | 0.006601 | 0.413744 | 0.179819 | 0.541945 | 0.047875 | 0.528836 |
| no_agent_information | 0.011274 | 0.084385 | 0.044265 | 0.307260 | 0.061165 | 0.115854 |
| no_information_channels | 0.006808 | 0.450087 | 0.202239 | 0.556392 | 0.030002 | 0.549973 |
| value_only | 0.015214 | -0.377080 | 0.295609 | 1.991976 | 0.692387 | 0.064675 |
| trend_only | 0.010016 | 0.063936 | 0.067264 | 0.312185 | 0.027186 | 0.100469 |
| noise_only | 0.028452 | 0.140046 | 0.169742 | 2.762416 | 0.465632 | 0.060010 |
| no_logit | 0.012565 | -0.020570 | 0.084035 | 0.587141 | 0.207211 | 0.060489 |
| no_activity_persistence | 0.012511 | -0.020389 | 0.058196 | 0.535871 | 0.188400 | 0.057102 |
| independent_signals | 0.010205 | -0.046118 | 0.015797 | 0.015327 | 0.137024 | 0.104240 |
| fixed_participation | 0.013876 | -0.035468 | 0.065382 | 0.593975 | 0.237641 | 0.069587 |
| fixed_liquidity | 0.011411 | -0.020720 | 0.031494 | 0.207351 | 0.151413 | 0.084057 |
| liquidity_stress | 0.014690 | -0.002442 | 0.075589 | 0.685789 | 0.229615 | 0.091815 |
| concentrated_wealth | 0.013534 | -0.031181 | 0.056175 | 0.646870 | 0.208336 | 0.049000 |
| garch_t_control | 0.009931 | -0.004959 | 0.159458 | 2.230839 | 0.143985 | 0.052351 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.078165 |
| Absolute-return ACF(5) | 0.038568 |
| Absolute-return ACF(20) | 0.044675 |
| Absolute-return ACF(50) | 0.025391 |
| Volume ACF(1) | 0.963505 |
| OFI ACF(1) | 0.870524 |
| OFI-return correlation | 0.167012 |
| Mean spread (bps) | 6.260202 |
| Three-sigma tail fraction | 0.007000 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
