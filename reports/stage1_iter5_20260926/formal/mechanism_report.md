# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-acceptance-v14`
- Frozen seeds: `50`
- Decision: `FAIL`

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
| `full_volatility_in_daily_range` | `0.022471772489982278` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.1395218759363866` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.20904613017648227` | `0.03` | 100.0% | none |
| `full_heavier_than_gaussian` | `2.2001524933450693` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.2885826654987502` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.06077503738538613` | `0.1` | 70.0% | 20270203, 20270207, 20270209, 20270210, 20270212, 20270216, 20270227, 20270228, 20270233, 20270235, 20270240, 20270246, 20270247, 20270248, 20270250 |
| `endogenous_liquidity_feedback_increases_tail_weight` | `2.0442830860987753` | `0.05` | 100.0% | none |
| `public_news_marginal_response` | `12.445457084674093` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `11.033665091492455` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.1503360049216909` | `1.0` | 74.0% | 20270203, 20270208, 20270209, 20270211, 20270212, 20270216, 20270223, 20270226, 20270236, 20270240, 20270242, 20270247, 20270250 |
| `agent_information_independent_response` | `1.0784214488091366` | `1.0` | 88.0% | 20270201, 20270209, 20270221, 20270238, 20270239, 20270243 |
| `empty_information_baseline_is_weakest_discovery` | `14.026467518377492` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.013445719154325533` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.030119798423066314` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.02067091798178062` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.09175781088012419` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.020351346597883735` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.004576280498016321` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.06471186673514809` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.08193858011909935` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.030041949909294808` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.14272806796563003` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `7.176116711690968` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.15154738147976565` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `6300.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.027549350299585856` | `{"maximum": 0.04, "minimum": 0.005}` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.4264023475349353` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.022472 | -0.042694 | 0.209046 | 2.200152 | 0.288583 | 0.060775 |
| no_direct_news | 0.018200 | -0.025864 | 0.547613 | 14.820159 | 0.734524 | 0.920726 |
| no_agent_information | 0.022213 | 0.001281 | 0.208270 | 2.137585 | 0.289091 | 0.074057 |
| no_information_channels | 0.018535 | -0.006144 | 0.567678 | 13.613682 | 0.737716 | 0.943896 |
| value_only | 0.013446 | -0.091758 | 0.041241 | 0.077446 | 0.174733 | 0.064484 |
| trend_only | 0.030120 | -0.020351 | 0.427321 | 13.082719 | 0.565959 | 0.085083 |
| noise_only | 0.020671 | -0.004576 | 0.346247 | 12.848792 | 0.564482 | 0.246987 |
| no_logit | 0.022472 | -0.042694 | 0.209046 | 2.200152 | 0.288583 | 0.060775 |
| no_activity_persistence | 0.022496 | -0.042034 | 0.205087 | 2.261195 | 0.282439 | 0.061573 |
| independent_signals | 0.012644 | -0.064712 | 0.033028 | -0.023262 | 0.028456 | 0.102712 |
| fixed_participation | 0.023358 | -0.081939 | 0.140969 | 1.280716 | 0.244537 | 0.067870 |
| fixed_liquidity | 0.018601 | -0.030042 | 0.071975 | 0.179551 | 0.141751 | 0.053827 |
| liquidity_stress | 0.036791 | -0.142728 | 0.370713 | 4.521528 | 0.521402 | 0.062005 |
| concentrated_wealth | 0.022545 | -0.046378 | 0.207129 | 2.124439 | 0.294368 | 0.058851 |
| garch_t_control | 0.022053 | -0.050075 | 0.353734 | 5.668745 | 0.446739 | 0.052559 |
| logit_learning | 0.027549 | 0.082604 | 0.270923 | 3.125405 | 0.352326 | 0.062197 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.139522 |
| Absolute-return ACF(5) | 0.214551 |
| Absolute-return ACF(20) | 0.065503 |
| Absolute-return ACF(50) | -0.052470 |
| Volume ACF(1) | 0.883902 |
| OFI ACF(1) | 0.426402 |
| OFI-return correlation | 0.571389 |
| Mean spread (bps) | 7.176117 |
| Three-sigma tail fraction | 0.011600 |
| Return skewness | -0.040047 |
| Leverage correlation, r(t) vs |r|(t+1) | 0.008686 |
| Crash-day fraction, r < -5% | 0.017200 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.346901 |
| Final wealth Gini | 0.219428 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
