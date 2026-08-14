# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-development-v8`
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
| `full_volatility_in_daily_range` | `0.015697992952273365` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.10863914177966799` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.1841176489169989` | `0.03` | 90.0% | 20261104, 20261113 |
| `full_heavier_than_gaussian` | `1.9174743902829574` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.3882902040366427` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.03878397154922465` | `0.1` | 95.0% | 20261104 |
| `endogenous_liquidity_feedback_increases_tail_weight` | `1.7829914838830552` | `0.05` | 95.0% | 20261104 |
| `public_news_marginal_response` | `6.027368281406137` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `5.895472130364713` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.3091625097352728` | `1.0` | 95.0% | 20261106 |
| `agent_information_independent_response` | `1.2124073427017898` | `1.0` | 85.0% | 20261103, 20261117, 20261119 |
| `empty_information_baseline_is_weakest_discovery` | `6.92989903033628` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.010889158726532254` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.019990866980875682` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.08575529285529666` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.038598617438196856` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.07080687416386909` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.09460510683953013` | `0.25` | 95.0% | 20261110 |
| `independent_signals_avoids_pathological_return_predictability` | `0.012848754555405153` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.03110653881944251` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.0006288960058416547` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.06935895596165342` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `6.794802342621047` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.11826878962434009` | `0.0` | 95.0% | 20261105 |
| `logit_imitation_fires_and_tracks_fitness` | `2550.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.019443649566624997` | `{"maximum": 0.04, "minimum": 0.005}` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.4547925205415348` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.015698 | -0.000595 | 0.184118 | 1.917474 | 0.388290 | 0.038784 |
| no_direct_news | 0.016297 | 0.006711 | 0.586760 | 16.736108 | 0.745307 | 0.289175 |
| no_agent_information | 0.015460 | 0.023086 | 0.184227 | 2.045052 | 0.386885 | 0.054435 |
| no_information_channels | 0.016709 | -0.011608 | 0.605485 | 17.350722 | 0.743030 | 0.371845 |
| value_only | 0.010889 | -0.038599 | 0.016337 | 0.022135 | 0.152418 | 0.024771 |
| trend_only | 0.019991 | 0.070807 | 0.406258 | 8.638906 | 0.568061 | 0.085216 |
| noise_only | 0.085755 | 0.094605 | 0.728437 | 1.589924 | 0.741234 | 3.065121 |
| no_logit | 0.015698 | -0.000595 | 0.184118 | 1.917474 | 0.388290 | 0.038784 |
| no_activity_persistence | 0.015728 | 0.002979 | 0.180931 | 1.861439 | 0.385410 | 0.037488 |
| independent_signals | 0.010681 | -0.012849 | 0.000178 | -0.005633 | 0.037871 | 0.043592 |
| fixed_participation | 0.016321 | -0.031107 | 0.131130 | 1.114497 | 0.337688 | 0.038343 |
| fixed_liquidity | 0.013407 | -0.000629 | 0.067693 | 0.184265 | 0.210432 | 0.041273 |
| liquidity_stress | 0.032264 | -0.069359 | 0.322693 | 2.769176 | 0.547139 | 0.040907 |
| concentrated_wealth | 0.015427 | -0.000346 | 0.203213 | 1.927161 | 0.386501 | 0.038921 |
| garch_t_control | 0.015388 | -0.014411 | 0.317210 | 6.213945 | 0.508462 | 0.041403 |
| logit_learning | 0.019444 | 0.020644 | 0.234266 | 2.645958 | 0.454315 | 0.064602 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.108639 |
| Absolute-return ACF(5) | 0.186065 |
| Absolute-return ACF(20) | 0.059067 |
| Absolute-return ACF(50) | -0.034708 |
| Volume ACF(1) | 0.843174 |
| OFI ACF(1) | 0.454793 |
| OFI-return correlation | 0.498825 |
| Mean spread (bps) | 6.794802 |
| Three-sigma tail fraction | 0.010000 |
| Return skewness | 0.071747 |
| Leverage correlation, r(t) vs |r|(t+1) | -0.009503 |
| Crash-day fraction, r < -5% | 0.003500 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.254107 |
| Final wealth Gini | 0.047768 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
