# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-development-v7`
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
| `full_volatility_in_daily_range` | `0.015368944028218442` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.09612967199808488` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.08921407580931075` | `0.03` | 80.0% | 20261104 |
| `full_heavier_than_gaussian` | `0.24843264606562698` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.2573366898484686` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.045146987272789964` | `0.1` | 80.0% | 20261104 |
| `endogenous_liquidity_feedback_increases_tail_weight` | `0.3929803069398434` | `0.05` | 100.0% | none |
| `public_news_marginal_response` | `7.461338212089066` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `6.902767719443075` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.4786525438184213` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.143581306893264` | `1.0` | 100.0% | none |
| `empty_information_baseline_is_weakest_discovery` | `9.641916388698451` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.011281432522242974` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.013761435796456183` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.1006415402507143` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.034171818927425664` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.04721009925984263` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.10270349503756622` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.029658974854986465` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.07131854131317047` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.012567427406402805` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.1248694221149684` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `9.809162955878687` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.11655986544230895` | `0.0` | 80.0% | 20261105 |
| `logit_imitation_fires_and_tracks_fitness` | `2550.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.022419904592538103` | `{"maximum": 0.03, "minimum": 0.005}` | 80.0% | 20261105 |
| `persistent_order_flow_is_observable` | `0.40484475976657325` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.015369 | -0.048185 | 0.089214 | 0.248433 | 0.257337 | 0.045147 |
| no_direct_news | 0.001824 | 0.347946 | 0.407388 | 2.416088 | 0.594857 | 0.440899 |
| no_agent_information | 0.015291 | -0.001900 | 0.091100 | 0.244416 | 0.250996 | 0.071573 |
| no_information_channels | 0.002364 | 0.411479 | 0.488163 | 4.737332 | 0.645954 | 0.494055 |
| value_only | 0.011281 | -0.034172 | -0.016258 | 0.049810 | 0.221147 | 0.028480 |
| trend_only | 0.013761 | 0.047210 | 0.131270 | 0.554174 | 0.297483 | 0.093830 |
| noise_only | 0.100642 | 0.102703 | 0.195100 | -1.139039 | 0.405573 | 1.828147 |
| no_logit | 0.015369 | -0.048185 | 0.089214 | 0.248433 | 0.257337 | 0.045147 |
| no_activity_persistence | 0.015368 | -0.047988 | 0.092328 | 0.253250 | 0.257897 | 0.044601 |
| independent_signals | 0.011179 | -0.029659 | -0.006879 | 0.048149 | 0.013881 | 0.035530 |
| fixed_participation | 0.015885 | -0.071319 | 0.086228 | 0.231746 | 0.249563 | 0.047366 |
| fixed_liquidity | 0.012190 | -0.012567 | 0.001719 | 0.103905 | 0.117808 | 0.056058 |
| liquidity_stress | 0.028938 | -0.124869 | 0.202864 | 1.203050 | 0.420277 | 0.099955 |
| concentrated_wealth | 0.015995 | -0.048813 | 0.071814 | 0.255476 | 0.244078 | 0.048425 |
| garch_t_control | 0.016877 | -0.062336 | 0.183698 | 1.251339 | 0.349272 | 0.022260 |
| logit_learning | 0.022420 | -0.027771 | 0.059446 | 0.242329 | 0.261584 | 0.100473 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.096130 |
| Absolute-return ACF(5) | 0.074421 |
| Absolute-return ACF(20) | -0.005210 |
| Absolute-return ACF(50) | 0.061341 |
| Volume ACF(1) | 0.923714 |
| OFI ACF(1) | 0.404845 |
| OFI-return correlation | 0.535261 |
| Mean spread (bps) | 9.809163 |
| Three-sigma tail fraction | 0.007000 |
| Return skewness | 0.112701 |
| Leverage correlation, r(t) vs |r|(t+1) | -0.009015 |
| Crash-day fraction, r < -5% | 0.000000 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.225201 |
| Final wealth Gini | 0.043479 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
