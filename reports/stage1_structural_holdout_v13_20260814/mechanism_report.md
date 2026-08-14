# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-holdout-v13`
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
| `full_volatility_in_daily_range` | `0.02146740013318825` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.1364080871820802` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.2466214774695223` | `0.03` | 100.0% | none |
| `full_heavier_than_gaussian` | `2.3418226323802265` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.4555410423018361` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.024338838766795597` | `0.1` | 100.0% | none |
| `endogenous_liquidity_feedback_increases_tail_weight` | `1.8496841594209819` | `0.05` | 100.0% | none |
| `public_news_marginal_response` | `5.702314772998571` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `5.390761839792912` | `1.0` | 90.0% | 20261369 |
| `agent_information_marginal_response` | `1.7142877584235698` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.5995898180592714` | `1.0` | 100.0% | none |
| `empty_information_baseline_is_weakest_discovery` | `7.685267595294072` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.013553386675923818` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.02607729995393037` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.06061171633999081` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.11381065409075768` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.042774610384511254` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.05586597841212426` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.057197111894316865` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.07356399332074137` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.02671523198694355` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.12707760950539326` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `7.028668861095776` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.1321513550967744` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `2550.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.02431894237041546` | `{"maximum": 0.04, "minimum": 0.005}` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.4487964004816337` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.021467 | -0.042415 | 0.246621 | 2.341823 | 0.455541 | 0.024339 |
| no_direct_news | 0.019149 | -0.016256 | 0.543919 | 13.694143 | 0.733710 | 0.143215 |
| no_agent_information | 0.020585 | 0.029663 | 0.256421 | 2.334062 | 0.458335 | 0.037998 |
| no_information_channels | 0.018062 | 0.006444 | 0.559845 | 12.426079 | 0.733587 | 0.218270 |
| value_only | 0.013553 | -0.113811 | 0.069559 | 0.253366 | 0.406588 | 0.006685 |
| trend_only | 0.026077 | 0.042775 | 0.455886 | 15.904612 | 0.626623 | 0.106857 |
| noise_only | 0.060612 | 0.055866 | 0.539097 | 7.512157 | 0.743070 | 2.662857 |
| no_logit | 0.021467 | -0.042415 | 0.246621 | 2.341823 | 0.455541 | 0.024339 |
| no_activity_persistence | 0.021672 | -0.035538 | 0.244642 | 2.035577 | 0.451484 | 0.024206 |
| independent_signals | 0.012346 | -0.057197 | 0.026042 | 0.007169 | 0.045422 | 0.011256 |
| fixed_participation | 0.023336 | -0.073564 | 0.192061 | 1.471735 | 0.414092 | 0.027017 |
| fixed_liquidity | 0.017859 | -0.026715 | 0.092107 | 0.309005 | 0.280272 | 0.016722 |
| liquidity_stress | 0.046543 | -0.127078 | 0.363048 | 3.523497 | 0.575126 | 0.039918 |
| concentrated_wealth | 0.022225 | -0.033441 | 0.241768 | 2.513046 | 0.463292 | 0.029506 |
| garch_t_control | 0.017895 | -0.023177 | 0.371530 | 5.205247 | 0.525084 | 0.019948 |
| logit_learning | 0.024319 | -0.019433 | 0.251778 | 2.862392 | 0.472536 | 0.037295 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.136408 |
| Absolute-return ACF(5) | 0.237132 |
| Absolute-return ACF(20) | 0.079011 |
| Absolute-return ACF(50) | -0.044869 |
| Volume ACF(1) | 0.861366 |
| OFI ACF(1) | 0.448796 |
| OFI-return correlation | 0.536234 |
| Mean spread (bps) | 7.028669 |
| Three-sigma tail fraction | 0.013000 |
| Return skewness | 0.182263 |
| Leverage correlation, r(t) vs |r|(t+1) | 0.000808 |
| Crash-day fraction, r < -5% | 0.015000 |
| Bubble-day fraction, log gap > 10% | 0.010500 |
| Maximum drawdown | 0.331474 |
| Final wealth Gini | 0.072127 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
