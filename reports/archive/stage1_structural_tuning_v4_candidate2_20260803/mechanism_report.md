# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-tuning-v4`
- Frozen seeds: `5`
- Decision: `FAIL`

## Gate checks

- [x] `full_runs_without_price_cap`
- [x] `cash_conservation`
- [x] `share_conservation`
- [x] `full_volatility_in_daily_range`
- [x] `full_return_autocorrelation_near_zero_across_lags`
- [ ] `full_volatility_clustering_has_decay`
- [x] `full_heavier_than_gaussian`
- [ ] `endogenous_liquidity_feedback_increases_tail_weight`
- [ ] `full_volume_volatility_relation_positive`
- [ ] `full_price_discovery_bounded`
- [x] `public_news_channel_has_paired_price_response`
- [ ] `agent_information_channel_has_paired_price_response`
- [x] `empty_information_baseline_is_weakest_discovery`
- [x] `single_strategy_markets_remain_active`
- [x] `ablations_avoid_pathological_return_predictability`
- [x] `fixed_population_has_no_strategy_turnover`
- [x] `persistent_order_flow_is_observable`
- [x] `spread_and_depth_are_finite`
- [x] `liquidity_stress_increases_clustering`
- [x] `garch_control_is_labelled_exogenous`

## Gate evidence

| Check | Observed | Threshold | Pass rate | Failed seeds |
|---|---:|---:|---:|---|
| `full_volatility_in_daily_range` | `0.011524865993254634` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.07436211255150027` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.03017088213620807` | `0.03` | 40.0% | 20260801, 20260802, 20260805 |
| `full_heavier_than_gaussian` | `0.2870023108096773` | `0.1` | 80.0% | 20260802 |
| `full_volume_volatility_relation_positive` | `0.14528925061003548` | `0.05` | 60.0% | 20260802, 20260805 |
| `full_price_discovery_bounded` | `0.0799103647247401` | `0.1` | 60.0% | 20260802, 20260805 |
| `endogenous_liquidity_feedback_increases_tail_weight` | `0.06901129075985368` | `0.05` | 60.0% | 20260802, 20260805 |
| `public_news_marginal_response` | `6.074398496806378` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `5.044542156776155` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.1433259656855141` | `1.0` | 80.0% | 20260802 |
| `agent_information_independent_response` | `1.0080862821488414` | `1.0` | 60.0% | 20260801, 20260805 |
| `empty_information_baseline_is_weakest_discovery` | `6.092487214685858` | `1.0` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.8714842619517802` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.011525 | -0.021517 | 0.030171 | 0.287002 | 0.145289 | 0.079910 |
| no_direct_news | 0.006602 | 0.409282 | 0.176555 | 0.450450 | 0.043561 | 0.535481 |
| no_agent_information | 0.010943 | 0.071040 | 0.038161 | 0.203882 | 0.050839 | 0.110556 |
| no_information_channels | 0.006808 | 0.450693 | 0.202239 | 0.556392 | 0.030221 | 0.549990 |
| value_only | 0.011189 | -0.207097 | 0.065408 | 0.207530 | 0.344849 | 0.081946 |
| trend_only | 0.010018 | 0.062370 | 0.066446 | 0.306964 | 0.023809 | 0.100863 |
| noise_only | 0.026250 | 0.142578 | 0.166404 | 3.217821 | 0.455277 | 0.062933 |
| no_logit | 0.011525 | -0.021517 | 0.030171 | 0.287002 | 0.145289 | 0.079910 |
| no_activity_persistence | 0.011468 | -0.022815 | 0.029507 | 0.234949 | 0.113269 | 0.073890 |
| independent_signals | 0.009986 | -0.046954 | 0.009952 | 0.009724 | 0.107504 | 0.100386 |
| fixed_participation | 0.013105 | -0.095127 | 0.085062 | 0.514083 | 0.193570 | 0.053810 |
| fixed_liquidity | 0.011213 | -0.020799 | 0.031360 | 0.177274 | 0.112983 | 0.089809 |
| liquidity_stress | 0.014378 | -0.024358 | 0.100042 | 0.806734 | 0.236697 | 0.068684 |
| concentrated_wealth | 0.012349 | -0.031517 | 0.030715 | 0.447790 | 0.147306 | 0.066273 |
| garch_t_control | 0.009572 | 0.007508 | 0.131429 | 2.138819 | 0.136695 | 0.057966 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.074362 |
| Absolute-return ACF(5) | 0.005788 |
| Absolute-return ACF(20) | 0.023362 |
| Absolute-return ACF(50) | 0.026840 |
| Volume ACF(1) | 0.965137 |
| OFI ACF(1) | 0.871484 |
| OFI-return correlation | 0.143097 |
| Mean spread (bps) | 5.690232 |
| Three-sigma tail fraction | 0.006000 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
