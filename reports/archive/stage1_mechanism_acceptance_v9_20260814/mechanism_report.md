# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-acceptance-v9`
- Frozen seeds: `50`
- Decision: `FAIL`

## Gate checks

- [ ] `full_runs_without_price_cap`
- [x] `cash_conservation`
- [x] `share_conservation`
- [x] `full_volatility_in_daily_range`
- [x] `full_return_autocorrelation_near_zero_across_lags`
- [x] `full_volatility_clustering_has_decay`
- [ ] `full_heavier_than_gaussian`
- [x] `endogenous_liquidity_feedback_increases_tail_weight`
- [x] `full_volume_volatility_relation_positive`
- [ ] `full_price_discovery_bounded`
- [x] `public_news_channel_has_paired_price_response`
- [ ] `agent_information_channel_has_paired_price_response`
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
| `full_volatility_in_daily_range` | `0.012173390499655982` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.06730601091169491` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.10339320104373134` | `0.03` | 80.0% | 20261214, 20261215, 20261216, 20261222, 20261225, 20261227, 20261236, 20261250, 20261256, 20261259 |
| `full_heavier_than_gaussian` | `0.7047119487415565` | `0.1` | 74.0% | 20261214, 20261215, 20261216, 20261222, 20261229, 20261236, 20261238, 20261240, 20261249, 20261250, 20261256, 20261257, 20261259 |
| `full_volume_volatility_relation_positive` | `0.25179982798359035` | `0.05` | 94.0% | 20261222, 20261250, 20261259 |
| `full_price_discovery_bounded` | `0.10918328313513172` | `0.1` | 42.0% | 20261214, 20261215, 20261216, 20261217, 20261218, 20261221, 20261222, 20261223, 20261225, 20261227, 20261229, 20261231, 20261233, 20261236, 20261237, 20261238, 20261240, 20261242, 20261244, 20261245, 20261246, 20261248, 20261249, 20261250, 20261251, 20261255, 20261256, 20261257, 20261259 |
| `endogenous_liquidity_feedback_increases_tail_weight` | `0.5259243971282583` | `0.05` | 86.0% | 20261216, 20261222, 20261236, 20261238, 20261246, 20261250, 20261259 |
| `public_news_marginal_response` | `6.471707017789933` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `6.632668610492155` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.061262927449322` | `1.0` | 74.0% | 20261214, 20261216, 20261218, 20261222, 20261228, 20261236, 20261237, 20261240, 20261246, 20261247, 20261249, 20261253, 20261259 |
| `agent_information_independent_response` | `1.054941382470012` | `1.0` | 68.0% | 20261211, 20261213, 20261214, 20261216, 20261217, 20261220, 20261221, 20261225, 20261226, 20261229, 20261231, 20261234, 20261250, 20261253, 20261256, 20261257 |
| `empty_information_baseline_is_weakest_discovery` | `7.3733192329900055` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.010354683339807557` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.013883431433875446` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.013071748648237392` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.018202293667146995` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.008256129308331018` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.009655918213548314` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.014977879600720804` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.032834365053671016` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.007601428969911402` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.04685866472763849` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `6.584943484872207` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.16480977936965008` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `6300.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.014083132566226515` | `{"maximum": 0.04, "minimum": 0.005}` | 96.0% | 20261211, 20261251 |
| `persistent_order_flow_is_observable` | `0.45879219565063445` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.012173 | -0.010054 | 0.103393 | 0.704712 | 0.251800 | 0.109183 |
| no_direct_news | 0.015273 | -0.001315 | 0.580217 | 17.616044 | 0.742685 | 0.777727 |
| no_agent_information | 0.012209 | 0.006675 | 0.099749 | 0.767746 | 0.260160 | 0.113656 |
| no_information_channels | 0.015984 | 0.025314 | 0.567712 | 16.490208 | 0.739624 | 0.873259 |
| value_only | 0.010355 | -0.018202 | 0.006990 | 0.001557 | 0.051545 | 0.122777 |
| trend_only | 0.013883 | 0.008256 | 0.272751 | 4.970900 | 0.420436 | 0.216062 |
| noise_only | 0.013072 | -0.009656 | 0.174692 | 2.621970 | 0.389629 | 2.791060 |
| no_logit | 0.012173 | -0.010054 | 0.103393 | 0.704712 | 0.251800 | 0.109183 |
| no_activity_persistence | 0.012204 | -0.008880 | 0.104468 | 0.641478 | 0.248838 | 0.114557 |
| independent_signals | 0.010348 | -0.014978 | 0.008698 | -0.005306 | 0.022883 | 0.137574 |
| fixed_participation | 0.012573 | -0.032834 | 0.091591 | 0.663749 | 0.264737 | 0.113823 |
| fixed_liquidity | 0.011600 | -0.007601 | 0.048003 | 0.101386 | 0.172625 | 0.134687 |
| liquidity_stress | 0.020120 | -0.046859 | 0.277168 | 3.492026 | 0.516186 | 0.092851 |
| concentrated_wealth | 0.012445 | -0.012676 | 0.097830 | 0.734478 | 0.253869 | 0.114661 |
| garch_t_control | 0.011571 | -0.014640 | 0.320448 | 6.970431 | 0.396783 | 0.106357 |
| logit_learning | 0.014083 | 0.005289 | 0.151568 | 1.646116 | 0.345307 | 0.135034 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.067306 |
| Absolute-return ACF(5) | 0.098004 |
| Absolute-return ACF(20) | 0.031638 |
| Absolute-return ACF(50) | 0.008806 |
| Volume ACF(1) | 0.894717 |
| OFI ACF(1) | 0.458792 |
| OFI-return correlation | 0.349875 |
| Mean spread (bps) | 6.584943 |
| Three-sigma tail fraction | 0.005600 |
| Return skewness | 0.009695 |
| Leverage correlation, r(t) vs |r|(t+1) | 0.001075 |
| Crash-day fraction, r < -5% | 0.000200 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.335478 |
| Final wealth Gini | 0.083847 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
