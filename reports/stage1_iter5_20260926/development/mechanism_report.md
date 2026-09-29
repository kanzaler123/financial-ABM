# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-development-v12`
- Frozen seeds: `20`
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
| `full_volatility_in_daily_range` | `0.022431342024206505` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.14560441612176617` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.22062308101151668` | `0.03` | 100.0% | none |
| `full_heavier_than_gaussian` | `2.092742954480146` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.2935652294934239` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.04643095010118056` | `0.1` | 90.0% | 20261107, 20261115 |
| `endogenous_liquidity_feedback_increases_tail_weight` | `1.8927405951016463` | `0.05` | 100.0% | none |
| `public_news_marginal_response` | `17.228201881858517` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `10.390914470568479` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.416168573762425` | `1.0` | 80.0% | 20261102, 20261104, 20261106, 20261117 |
| `agent_information_independent_response` | `1.0671247161424895` | `1.0` | 80.0% | 20261103, 20261111, 20261112, 20261118 |
| `empty_information_baseline_is_weakest_discovery` | `16.75504896842041` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.013742631474982909` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.030376784776723554` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.019759441245619236` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.10325223290296548` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.006897923271491922` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.008583828659669963` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.07125596525853295` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.08470313464140306` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.03142048994043544` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.14244394431577306` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `7.147165048215363` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.1477007592479081` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `6300.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.026830343198420713` | `{"maximum": 0.04, "minimum": 0.005}` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.42581747253023017` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.022431 | -0.047067 | 0.220623 | 2.092743 | 0.293565 | 0.046431 |
| no_direct_news | 0.017781 | -0.023458 | 0.562875 | 13.810098 | 0.738095 | 0.795251 |
| no_agent_information | 0.022101 | 0.008295 | 0.224803 | 2.192166 | 0.311143 | 0.055706 |
| no_information_channels | 0.018418 | 0.005176 | 0.558292 | 14.046157 | 0.739674 | 0.812486 |
| value_only | 0.013743 | -0.103252 | 0.045278 | 0.165784 | 0.198917 | 0.040522 |
| trend_only | 0.030377 | -0.006898 | 0.433471 | 13.150786 | 0.580065 | 0.104968 |
| noise_only | 0.019759 | 0.008584 | 0.313265 | 13.428855 | 0.574002 | 0.172559 |
| no_logit | 0.022431 | -0.047067 | 0.220623 | 2.092743 | 0.293565 | 0.046431 |
| no_activity_persistence | 0.022447 | -0.045587 | 0.213225 | 2.087019 | 0.295025 | 0.043434 |
| independent_signals | 0.012665 | -0.071256 | 0.030605 | 0.064619 | 0.021945 | 0.091676 |
| fixed_participation | 0.023332 | -0.084703 | 0.149174 | 1.285580 | 0.262457 | 0.050119 |
| fixed_liquidity | 0.018634 | -0.031420 | 0.075441 | 0.252401 | 0.148864 | 0.046824 |
| liquidity_stress | 0.036763 | -0.142444 | 0.364926 | 4.591954 | 0.514255 | 0.055044 |
| concentrated_wealth | 0.022512 | -0.048988 | 0.212660 | 2.308107 | 0.292419 | 0.049370 |
| garch_t_control | 0.022101 | -0.053237 | 0.343253 | 5.183192 | 0.401214 | 0.057186 |
| logit_learning | 0.026830 | 0.078197 | 0.263627 | 2.846889 | 0.383082 | 0.052477 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.145604 |
| Absolute-return ACF(5) | 0.210638 |
| Absolute-return ACF(20) | 0.054686 |
| Absolute-return ACF(50) | -0.049572 |
| Volume ACF(1) | 0.887290 |
| OFI ACF(1) | 0.425817 |
| OFI-return correlation | 0.567866 |
| Mean spread (bps) | 7.147165 |
| Three-sigma tail fraction | 0.011000 |
| Return skewness | -0.047645 |
| Leverage correlation, r(t) vs |r|(t+1) | 0.010370 |
| Crash-day fraction, r < -5% | 0.017800 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.399453 |
| Final wealth Gini | 0.220456 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
