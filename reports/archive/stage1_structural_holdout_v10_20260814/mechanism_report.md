# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-holdout-v10`
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
| `full_volatility_in_daily_range` | `0.01763459078790524` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.13472947439701527` | `0.2` | 90.0% | 20261334 |
| `full_volatility_clustering_has_decay` | `0.20087468155109253` | `0.03` | 100.0% | none |
| `full_heavier_than_gaussian` | `2.627939605658267` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.3746228648374255` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.019492210167804204` | `0.1` | 100.0% | none |
| `endogenous_liquidity_feedback_increases_tail_weight` | `2.1245704712305695` | `0.05` | 100.0% | none |
| `public_news_marginal_response` | `12.707956035047275` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `9.924575106371709` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.3965395338810667` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.1919588107397918` | `1.0` | 90.0% | 20261333 |
| `empty_information_baseline_is_weakest_discovery` | `14.225825411513789` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.01276884978021863` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.024508684086761007` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.07760736543024829` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.0795307001641238` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.008920205759078774` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.10167557604318626` | `0.25` | 90.0% | 20261335 |
| `independent_signals_avoids_pathological_return_predictability` | `0.04656745835907172` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.0737698976375342` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.01743376494682371` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.09910025831941663` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `6.844269432349641` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.058406924061450885` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `2550.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.019790972320807534` | `{"maximum": 0.04, "minimum": 0.005}` | 90.0% | 20261336 |
| `persistent_order_flow_is_observable` | `0.43750819261376694` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.017635 | -0.044287 | 0.200875 | 2.627940 | 0.374623 | 0.019492 |
| no_direct_news | 0.018083 | -0.003611 | 0.570876 | 17.608025 | 0.740550 | 0.310525 |
| no_agent_information | 0.017566 | 0.005243 | 0.198171 | 2.177290 | 0.373051 | 0.031366 |
| no_information_channels | 0.019524 | -0.000273 | 0.547999 | 14.208841 | 0.732001 | 0.363764 |
| value_only | 0.012769 | -0.079531 | 0.012441 | 0.135077 | 0.305523 | 0.007706 |
| trend_only | 0.024509 | 0.008920 | 0.410228 | 12.765069 | 0.605497 | 0.092979 |
| noise_only | 0.077607 | 0.101676 | 0.716723 | 5.030241 | 0.786212 | 3.201682 |
| no_logit | 0.017635 | -0.044287 | 0.200875 | 2.627940 | 0.374623 | 0.019492 |
| no_activity_persistence | 0.017701 | -0.046022 | 0.198791 | 2.612507 | 0.373295 | 0.019336 |
| independent_signals | 0.012126 | -0.046567 | 0.009866 | 0.069163 | 0.038759 | 0.013041 |
| fixed_participation | 0.018346 | -0.073770 | 0.131948 | 1.545156 | 0.351819 | 0.020350 |
| fixed_liquidity | 0.015554 | -0.017434 | 0.062696 | 0.457687 | 0.233799 | 0.018269 |
| liquidity_stress | 0.034009 | -0.099100 | 0.267238 | 3.635258 | 0.532955 | 0.032762 |
| concentrated_wealth | 0.017341 | -0.048639 | 0.177312 | 2.821324 | 0.358928 | 0.019497 |
| garch_t_control | 0.017188 | -0.046031 | 0.314973 | 5.389313 | 0.505219 | 0.020410 |
| logit_learning | 0.019791 | -0.015847 | 0.207484 | 2.410139 | 0.408874 | 0.025733 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.134729 |
| Absolute-return ACF(5) | 0.184084 |
| Absolute-return ACF(20) | 0.074938 |
| Absolute-return ACF(50) | -0.022604 |
| Volume ACF(1) | 0.853969 |
| OFI ACF(1) | 0.437508 |
| OFI-return correlation | 0.504338 |
| Mean spread (bps) | 6.844269 |
| Three-sigma tail fraction | 0.011000 |
| Return skewness | 0.081293 |
| Leverage correlation, r(t) vs |r|(t+1) | -0.057931 |
| Crash-day fraction, r < -5% | 0.006500 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.250956 |
| Final wealth Gini | 0.058435 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
