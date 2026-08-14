# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-holdout-v9`
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
| `full_volatility_in_daily_range` | `0.01736841379079368` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.12997588416710065` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.196255196392796` | `0.03` | 100.0% | none |
| `full_heavier_than_gaussian` | `2.2639786957639054` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.3697017875783374` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.038415310072493346` | `0.1` | 100.0% | none |
| `endogenous_liquidity_feedback_increases_tail_weight` | `1.7715609901960832` | `0.05` | 100.0% | none |
| `public_news_marginal_response` | `5.979014535423642` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `4.973359877739467` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.689246297582202` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.2453066748584511` | `1.0` | 100.0% | none |
| `empty_information_baseline_is_weakest_discovery` | `7.304076501905662` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.011030034149467557` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.019107509490250404` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.08711443001808494` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.0592713057753277` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.03827755337840172` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.1267670822957055` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.03369236391562676` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.04415764911148008` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.0013716169236477603` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.041541136093199274` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `6.841707106384488` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.09148921416335941` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `2550.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.021580238083772182` | `{"maximum": 0.04, "minimum": 0.005}` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.4719968195179693` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.017368 | -0.002945 | 0.196255 | 2.263979 | 0.369702 | 0.038415 |
| no_direct_news | 0.016112 | -0.008153 | 0.555134 | 15.017913 | 0.720205 | 0.157773 |
| no_agent_information | 0.017294 | 0.033209 | 0.181973 | 2.412774 | 0.387598 | 0.060562 |
| no_information_channels | 0.017840 | 0.027302 | 0.560403 | 15.572321 | 0.727453 | 0.222524 |
| value_only | 0.011030 | -0.059271 | 0.017818 | 0.073446 | 0.230279 | 0.012898 |
| trend_only | 0.019108 | 0.038278 | 0.354178 | 13.679029 | 0.529180 | 0.107778 |
| noise_only | 0.087114 | 0.126767 | 0.727862 | 1.592455 | 0.776303 | 2.578763 |
| no_logit | 0.017368 | -0.002945 | 0.196255 | 2.263979 | 0.369702 | 0.038415 |
| no_activity_persistence | 0.017373 | -0.010770 | 0.188956 | 2.227417 | 0.370101 | 0.038536 |
| independent_signals | 0.010871 | -0.033692 | 0.021608 | -0.049481 | 0.015190 | 0.023697 |
| fixed_participation | 0.018479 | -0.044158 | 0.140211 | 1.739357 | 0.349116 | 0.032526 |
| fixed_liquidity | 0.015274 | -0.001372 | 0.063064 | 0.314558 | 0.225121 | 0.024684 |
| liquidity_stress | 0.036104 | -0.041541 | 0.280933 | 2.713340 | 0.524023 | 0.038927 |
| concentrated_wealth | 0.017762 | -0.013337 | 0.183287 | 2.433576 | 0.369491 | 0.035945 |
| garch_t_control | 0.015076 | 0.002953 | 0.347650 | 5.604431 | 0.549298 | 0.026819 |
| logit_learning | 0.021580 | 0.006651 | 0.214052 | 2.804663 | 0.450691 | 0.056297 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.129976 |
| Absolute-return ACF(5) | 0.184513 |
| Absolute-return ACF(20) | 0.055327 |
| Absolute-return ACF(50) | -0.025429 |
| Volume ACF(1) | 0.837961 |
| OFI ACF(1) | 0.471997 |
| OFI-return correlation | 0.549167 |
| Mean spread (bps) | 6.841707 |
| Three-sigma tail fraction | 0.012500 |
| Return skewness | 0.137540 |
| Leverage correlation, r(t) vs |r|(t+1) | -0.011909 |
| Crash-day fraction, r < -5% | 0.005000 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.298918 |
| Final wealth Gini | 0.052469 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
