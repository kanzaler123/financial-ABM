# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-development-v8`
- Frozen seeds: `20`
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
| `full_volatility_in_daily_range` | `0.015479817957100524` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.09438441866840072` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.0894993028511066` | `0.03` | 65.0% | 20261104, 20261107, 20261109, 20261112, 20261113, 20261114, 20261119 |
| `full_heavier_than_gaussian` | `0.3293761873404242` | `0.1` | 85.0% | 20261107, 20261109, 20261120 |
| `full_volume_volatility_relation_positive` | `0.23881851330647152` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.03212859364253426` | `0.1` | 95.0% | 20261104 |
| `endogenous_liquidity_feedback_increases_tail_weight` | `0.31500994089454304` | `0.05` | 85.0% | 20261107, 20261109, 20261120 |
| `public_news_marginal_response` | `8.05040969182191` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `6.17031996603402` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.5702893709325587` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.2147573733121686` | `1.0` | 100.0% | none |
| `empty_information_baseline_is_weakest_discovery` | `9.665277798997131` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.011086899654316277` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.016513193994058185` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.08500275921988855` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.061727823095189246` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.04761008771535287` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.07360219210320114` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.025888137222040503` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.047336199502591506` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.0024460880194168543` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.10468183009522392` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `9.829049453698275` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.093961366725033` | `0.0` | 85.0% | 20261105, 20261110, 20261112 |
| `logit_imitation_fires_and_tracks_fitness` | `2550.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.022527515558546183` | `{"maximum": 0.04, "minimum": 0.005}` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.4083854744403149` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.015480 | -0.038647 | 0.089499 | 0.329376 | 0.238819 | 0.032129 |
| no_direct_news | 0.004002 | 0.141267 | 0.458939 | 6.039755 | 0.650859 | 0.306604 |
| no_agent_information | 0.015336 | 0.005379 | 0.093532 | 0.309361 | 0.237201 | 0.057284 |
| no_information_channels | 0.005619 | 0.076690 | 0.510724 | 7.183027 | 0.700525 | 0.367930 |
| value_only | 0.011087 | -0.061728 | 0.002096 | 0.047321 | 0.259988 | 0.017029 |
| trend_only | 0.016513 | 0.047610 | 0.214102 | 1.339117 | 0.389455 | 0.060724 |
| noise_only | 0.085003 | 0.073602 | 0.177408 | -0.756426 | 0.427445 | 1.441503 |
| no_logit | 0.015480 | -0.038647 | 0.089499 | 0.329376 | 0.238819 | 0.032129 |
| no_activity_persistence | 0.015516 | -0.038529 | 0.090491 | 0.323405 | 0.239592 | 0.031951 |
| independent_signals | 0.010875 | -0.025888 | 0.016759 | 0.011314 | 0.049296 | 0.028399 |
| fixed_participation | 0.016084 | -0.047336 | 0.077094 | 0.295786 | 0.227006 | 0.032219 |
| fixed_liquidity | 0.012162 | -0.002446 | 0.028551 | 0.083893 | 0.140061 | 0.038150 |
| liquidity_stress | 0.031530 | -0.104682 | 0.160741 | 0.700031 | 0.366418 | 0.045917 |
| concentrated_wealth | 0.016090 | -0.047379 | 0.073329 | 0.326466 | 0.242415 | 0.030880 |
| garch_t_control | 0.014117 | -0.038335 | 0.216316 | 1.466225 | 0.416671 | 0.024526 |
| logit_learning | 0.022528 | -0.025385 | 0.052522 | 0.252520 | 0.259394 | 0.078624 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.094384 |
| Absolute-return ACF(5) | 0.073606 |
| Absolute-return ACF(20) | 0.036557 |
| Absolute-return ACF(50) | 0.015236 |
| Volume ACF(1) | 0.916904 |
| OFI ACF(1) | 0.408385 |
| OFI-return correlation | 0.544093 |
| Mean spread (bps) | 9.829049 |
| Three-sigma tail fraction | 0.005000 |
| Return skewness | 0.041501 |
| Leverage correlation, r(t) vs |r|(t+1) | -0.008359 |
| Crash-day fraction, r < -5% | 0.002000 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.273889 |
| Final wealth Gini | 0.043716 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
