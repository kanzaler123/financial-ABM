# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-development-v5`
- Frozen seeds: `5`
- Decision: `FAIL`

## Gate checks

- [x] `full_runs_without_price_cap`
- [x] `cash_conservation`
- [x] `share_conservation`
- [x] `full_volatility_in_daily_range`
- [x] `full_return_autocorrelation_near_zero_across_lags`
- [ ] `full_volatility_clustering_has_decay`
- [ ] `full_heavier_than_gaussian`
- [ ] `endogenous_liquidity_feedback_increases_tail_weight`
- [ ] `full_volume_volatility_relation_positive`
- [ ] `full_price_discovery_bounded`
- [x] `public_news_channel_has_paired_price_response`
- [x] `agent_information_channel_has_paired_price_response`
- [x] `empty_information_baseline_is_weakest_discovery`
- [x] `single_strategy_markets_remain_active`
- [x] `ablations_avoid_pathological_return_predictability`
- [x] `fixed_population_has_no_strategy_turnover`
- [x] `persistent_order_flow_is_observable`
- [x] `spread_and_depth_are_finite`
- [ ] `liquidity_stress_increases_clustering`
- [x] `garch_control_is_labelled_exogenous`

## Gate evidence

| Check | Observed | Threshold | Pass rate | Failed seeds |
|---|---:|---:|---:|---|
| `full_volatility_in_daily_range` | `0.009956196708227609` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.06903027718124406` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `-0.020654277945362832` | `0.03` | 20.0% | 20261101, 20261102, 20261103, 20261104 |
| `full_heavier_than_gaussian` | `0.01721307405448913` | `0.1` | 20.0% | 20261101, 20261102, 20261103, 20261105 |
| `full_volume_volatility_relation_positive` | `-0.018655084609725796` | `0.05` | 20.0% | 20261101, 20261102, 20261104, 20261105 |
| `full_price_discovery_bounded` | `0.08420136763368984` | `0.1` | 60.0% | 20261102, 20261104 |
| `endogenous_liquidity_feedback_increases_tail_weight` | `-5.888008857635185e-05` | `0.05` | 0.0% | 20261101, 20261102, 20261103, 20261104, 20261105 |
| `public_news_marginal_response` | `5.65816115454963` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `4.988664606684389` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.1541415679985387` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.0512344111407355` | `1.0` | 100.0% | none |
| `empty_information_baseline_is_weakest_discovery` | `5.948053709442363` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.010114914641385633` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.009828954970102697` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.011551063543373458` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.012293297461273755` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.05432554035026424` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.0627311242374489` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.002494423680100423` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.005242808913448781` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.01606986927306508` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.01928833511981931` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `5.443005159564836` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `-0.0004714550910158836` | `0.0` | 20.0% | 20261101, 20261102, 20261104, 20261105 |
| `persistent_order_flow_is_observable` | `0.8768745886392355` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.009956 | 0.015551 | -0.020654 | 0.017213 | -0.018655 | 0.084201 |
| no_direct_news | 0.001599 | 0.596859 | 0.332099 | -0.133191 | 0.469695 | 0.476425 |
| no_agent_information | 0.009731 | 0.020235 | -0.020326 | 0.010069 | -0.023158 | 0.097180 |
| no_information_channels | 0.001710 | 0.624442 | 0.411473 | -0.135087 | 0.520594 | 0.500834 |
| value_only | 0.010115 | -0.012293 | -0.018997 | -0.006493 | 0.098467 | 0.070802 |
| trend_only | 0.009829 | 0.054326 | -0.021543 | -0.062494 | 0.053294 | 0.102938 |
| noise_only | 0.011551 | 0.062731 | -0.009853 | 0.174603 | 0.080653 | 0.103036 |
| no_logit | 0.009956 | 0.015551 | -0.020654 | 0.017213 | -0.018655 | 0.084201 |
| no_activity_persistence | 0.009966 | 0.013831 | -0.021335 | 0.012277 | -0.016748 | 0.084286 |
| independent_signals | 0.009844 | -0.002494 | -0.016708 | -0.040367 | -0.020970 | 0.084612 |
| fixed_participation | 0.010294 | -0.005243 | -0.022528 | 0.001245 | -0.005715 | 0.084288 |
| fixed_liquidity | 0.009950 | 0.016070 | -0.020585 | 0.014946 | -0.018799 | 0.084817 |
| liquidity_stress | 0.010603 | 0.019288 | -0.020733 | 0.046354 | 0.015300 | 0.076917 |
| concentrated_wealth | 0.009919 | 0.005640 | -0.017878 | 0.004262 | -0.013719 | 0.083265 |
| garch_t_control | 0.009231 | 0.004367 | 0.276395 | 6.694461 | 0.101969 | 0.028774 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.069030 |
| Absolute-return ACF(5) | 0.006414 |
| Absolute-return ACF(20) | -0.016617 |
| Absolute-return ACF(50) | -0.008143 |
| Volume ACF(1) | 0.963043 |
| OFI ACF(1) | 0.876875 |
| OFI-return correlation | 0.127537 |
| Mean spread (bps) | 5.443005 |
| Three-sigma tail fraction | 0.003000 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
