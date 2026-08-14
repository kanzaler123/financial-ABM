# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-holdout-v11`
- Frozen seeds: `10`
- Decision: `PASS`

## Gate checks

- [x] `full_runs_without_price_cap`
- [x] `cash_conservation`
- [x] `share_conservation`
- [x] `full_volatility_in_daily_range`
- [x] `full_return_autocorrelation_near_zero_across_lags`
- [x] `full_volatility_clustering_has_decay`
- [x] `full_heavier_than_gaussian`
- [x] `endogenous_liquidity_feedback_increases_tail_weight`
- [x] `full_volume_volatility_relation_positive`
- [x] `full_price_discovery_bounded`
- [x] `public_news_channel_has_paired_price_response`
- [x] `agent_information_channel_has_paired_price_response`
- [x] `empty_information_baseline_is_weakest_discovery`
- [x] `single_strategy_markets_remain_active`
- [x] `ablations_avoid_pathological_return_predictability`
- [x] `fixed_population_has_no_strategy_turnover`
- [x] `logit_imitation_fires_and_tracks_fitness`
- [x] `logit_market_remains_active_and_bounded`
- [x] `persistent_order_flow_is_observable`
- [x] `spread_and_depth_are_finite`
- [x] `liquidity_stress_increases_clustering`
- [x] `garch_control_is_labelled_exogenous`

## Gate evidence

| Check | Observed | Threshold | Pass rate | Failed seeds |
|---|---:|---:|---:|---|
| `full_volatility_in_daily_range` | `0.019166980256673263` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.11745359498789741` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.19326661549796897` | `0.03` | 90.0% | 20261348 |
| `full_heavier_than_gaussian` | `1.8283182854206919` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.36020063981512984` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.018912028695391957` | `0.1` | 100.0% | none |
| `endogenous_liquidity_feedback_increases_tail_weight` | `1.5932625765304151` | `0.05` | 100.0% | none |
| `public_news_marginal_response` | `8.635190542540103` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `5.611497173793705` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.6678947248434315` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.189939417507512` | `1.0` | 80.0% | 20261342, 20261346 |
| `empty_information_baseline_is_weakest_discovery` | `9.41201365155641` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.012872461435446687` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.025320831268165117` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.07023835284451942` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.08504659350971414` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.019646452049655858` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.06468378593870602` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.05207338865331862` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.057387496754668005` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.025774922451880726` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.09766793788134295` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `6.997539397542758` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.09920418798859076` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `2550.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.02327097964320783` | `{"maximum": 0.04, "minimum": 0.005}` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.4395521930426133` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.019167 | -0.028122 | 0.193267 | 1.828318 | 0.360201 | 0.018912 |
| no_direct_news | 0.018573 | 0.010305 | 0.562826 | 14.177879 | 0.735349 | 0.159871 |
| no_agent_information | 0.019134 | 0.008778 | 0.197273 | 1.954203 | 0.388614 | 0.038538 |
| no_information_channels | 0.018471 | 0.043257 | 0.564461 | 14.429562 | 0.742882 | 0.185241 |
| value_only | 0.012872 | -0.085047 | 0.029640 | 0.188918 | 0.358659 | 0.007264 |
| trend_only | 0.025321 | 0.019646 | 0.433835 | 14.218705 | 0.604728 | 0.085841 |
| noise_only | 0.070238 | 0.064684 | 0.609211 | 5.863535 | 0.780678 | 1.907885 |
| no_logit | 0.019167 | -0.028122 | 0.193267 | 1.828318 | 0.360201 | 0.018912 |
| no_activity_persistence | 0.019187 | -0.028715 | 0.210004 | 1.350962 | 0.376026 | 0.019384 |
| independent_signals | 0.011938 | -0.052073 | -0.011908 | 0.114427 | 0.032768 | 0.015212 |
| fixed_participation | 0.020202 | -0.057387 | 0.144639 | 0.801907 | 0.330827 | 0.019318 |
| fixed_liquidity | 0.016220 | -0.025775 | 0.072879 | 0.118754 | 0.210081 | 0.015697 |
| liquidity_stress | 0.041698 | -0.097668 | 0.305206 | 2.533505 | 0.542081 | 0.033658 |
| concentrated_wealth | 0.019179 | -0.027435 | 0.190606 | 1.465165 | 0.349456 | 0.019502 |
| garch_t_control | 0.017562 | -0.036388 | 0.382580 | 4.332836 | 0.550101 | 0.018538 |
| logit_learning | 0.023271 | -0.007968 | 0.219871 | 2.263962 | 0.424386 | 0.031987 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.117454 |
| Absolute-return ACF(5) | 0.169781 |
| Absolute-return ACF(20) | 0.090261 |
| Absolute-return ACF(50) | -0.025176 |
| Volume ACF(1) | 0.822951 |
| OFI ACF(1) | 0.439552 |
| OFI-return correlation | 0.521200 |
| Mean spread (bps) | 6.997539 |
| Three-sigma tail fraction | 0.009500 |
| Return skewness | 0.075978 |
| Leverage correlation, r(t) vs |r|(t+1) | -0.017794 |
| Crash-day fraction, r < -5% | 0.011000 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.304714 |
| Final wealth Gini | 0.058460 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
