# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-holdout-v8`
- Frozen seeds: `10`
- Decision: `FAIL`

## Gate checks

- [x] `full_runs_without_price_cap`
- [x] `cash_conservation`
- [x] `share_conservation`
- [x] `full_volatility_in_daily_range`
- [x] `full_return_autocorrelation_near_zero_across_lags`
- [ ] `full_volatility_clustering_has_decay`
- [ ] `full_heavier_than_gaussian`
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
| `full_volatility_in_daily_range` | `0.012958544194199308` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.08955014566777433` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.04331789659292901` | `0.03` | 50.0% | 20261311, 20261313, 20261315, 20261319, 20261320 |
| `full_heavier_than_gaussian` | `0.1681438607609227` | `0.1` | 50.0% | 20261311, 20261313, 20261315, 20261316, 20261319 |
| `full_volume_volatility_relation_positive` | `0.17592002470641332` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.045104581615108996` | `0.1` | 80.0% | 20261311, 20261319 |
| `endogenous_liquidity_feedback_increases_tail_weight` | `0.10657335001750212` | `0.05` | 90.0% | 20261316 |
| `public_news_marginal_response` | `5.7516684742591835` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `5.1510941667769945` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.3120275742745373` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.178874941153299` | `1.0` | 100.0% | none |
| `empty_information_baseline_is_weakest_discovery` | `6.838767452541898` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.010736738930464059` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.012130999840550421` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.045830867877915536` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.045426417148945755` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.06677920278443181` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.04662984525716608` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.030908930826096005` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.045640905665524326` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.0064426288238563474` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.0764666087940731` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `6.597804124066073` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.12369599560667441` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `2550.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.015391308255585106` | `{"maximum": 0.04, "minimum": 0.005}` | 90.0% | 20261314 |
| `persistent_order_flow_is_observable` | `0.46763697858041664` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.012959 | -0.013830 | 0.043318 | 0.168144 | 0.175920 | 0.045105 |
| no_direct_news | 0.011427 | -0.002243 | 0.524837 | 11.125488 | 0.702863 | 0.349010 |
| no_agent_information | 0.012904 | 0.018046 | 0.043216 | 0.134292 | 0.164352 | 0.065123 |
| no_information_channels | 0.012042 | 0.068662 | 0.548368 | 10.408924 | 0.700141 | 0.415209 |
| value_only | 0.010737 | -0.045426 | 0.012825 | 0.022000 | 0.244437 | 0.031390 |
| trend_only | 0.012131 | 0.066779 | 0.120904 | 0.351916 | 0.194584 | 0.094146 |
| noise_only | 0.045831 | 0.046630 | 0.348221 | 2.105719 | 0.585284 | 0.389555 |
| no_logit | 0.012959 | -0.013830 | 0.043318 | 0.168144 | 0.175920 | 0.045105 |
| no_activity_persistence | 0.012954 | -0.014700 | 0.039924 | 0.138968 | 0.169524 | 0.045500 |
| independent_signals | 0.010578 | -0.030909 | 0.027699 | 0.113661 | 0.018310 | 0.047536 |
| fixed_participation | 0.013481 | -0.045641 | 0.040548 | 0.160987 | 0.174832 | 0.042562 |
| fixed_liquidity | 0.012150 | -0.006443 | 0.035010 | -0.008850 | 0.120890 | 0.049203 |
| liquidity_stress | 0.025118 | -0.076467 | 0.188657 | 1.211194 | 0.390442 | 0.035283 |
| concentrated_wealth | 0.013460 | -0.016339 | 0.042944 | 0.193737 | 0.201755 | 0.047985 |
| garch_t_control | 0.011399 | -0.002995 | 0.236901 | 3.157068 | 0.366774 | 0.039231 |
| logit_learning | 0.015391 | 0.005721 | 0.068040 | 0.500873 | 0.259097 | 0.078782 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.089550 |
| Absolute-return ACF(5) | 0.028880 |
| Absolute-return ACF(20) | 0.012370 |
| Absolute-return ACF(50) | -0.018411 |
| Volume ACF(1) | 0.863088 |
| OFI ACF(1) | 0.467637 |
| OFI-return correlation | 0.428593 |
| Mean spread (bps) | 6.597804 |
| Three-sigma tail fraction | 0.004000 |
| Return skewness | 0.011443 |
| Leverage correlation, r(t) vs |r|(t+1) | -0.004371 |
| Crash-day fraction, r < -5% | 0.000000 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.288854 |
| Final wealth Gini | 0.030612 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
