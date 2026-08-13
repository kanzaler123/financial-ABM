# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-development-v3`
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
- [x] `endogenous_liquidity_feedback_increases_tail_weight`
- [ ] `full_volume_volatility_relation_positive`
- [x] `full_price_discovery_bounded`
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
| `full_volatility_in_daily_range` | `0.011964770709675094` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.07790076368689844` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.02428704888296366` | `0.03` | 20.0% | 20260801, 20260802, 20260804, 20260805 |
| `full_heavier_than_gaussian` | `0.09504284067653801` | `0.1` | 40.0% | 20260801, 20260802, 20260803 |
| `full_volume_volatility_relation_positive` | `0.06506905569308534` | `0.05` | 60.0% | 20260802, 20260805 |
| `full_price_discovery_bounded` | `0.04618523731018915` | `0.1` | 100.0% | none |
| `endogenous_liquidity_feedback_increases_tail_weight` | `0.0` | `0.0` | 100.0% | none |
| `public_news_marginal_response` | `10.814154119940047` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `9.058154998614839` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.2048695999498766` | `1.0` | 60.0% | 20260802, 20260805 |
| `agent_information_independent_response` | `0.99612235865879` | `1.0` | 40.0% | 20260801, 20260804, 20260805 |
| `empty_information_baseline_is_weakest_discovery` | `10.772220708854352` | `1.0` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.8445952801423284` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.011965 | -0.055663 | 0.024287 | 0.095043 | 0.065069 | 0.046185 |
| no_direct_news | 0.006503 | 0.413791 | 0.201541 | 0.575183 | 0.048430 | 0.520516 |
| no_agent_information | 0.011857 | 0.044350 | 0.033786 | 0.121386 | 0.048618 | 0.052260 |
| no_information_channels | 0.006808 | 0.450693 | 0.202239 | 0.556392 | 0.030289 | 0.549990 |
| value_only | 0.011297 | -0.178611 | 0.036674 | -0.104373 | 0.059788 | 0.049638 |
| trend_only | 0.011185 | 0.041146 | 0.042223 | 0.136506 | 0.014348 | 0.056866 |
| noise_only | 0.022000 | 0.146815 | 0.129232 | 2.791130 | 0.364858 | 0.070373 |
| no_logit | 0.011965 | -0.055663 | 0.024287 | 0.095043 | 0.065069 | 0.046185 |
| no_activity_persistence | 0.011930 | -0.050163 | 0.026748 | 0.068757 | 0.048116 | 0.042955 |
| independent_signals | 0.011186 | -0.076189 | 0.007317 | -0.016464 | 0.032338 | 0.044087 |
| fixed_participation | 0.012787 | -0.077262 | 0.045714 | 0.107975 | 0.093821 | 0.034111 |
| fixed_liquidity | 0.011965 | -0.055663 | 0.024287 | 0.095043 | 0.065069 | 0.046185 |
| liquidity_stress | 0.014729 | -0.092633 | 0.094011 | 0.580258 | 0.192957 | 0.050004 |
| concentrated_wealth | 0.012459 | -0.057167 | 0.031075 | 0.140365 | 0.086400 | 0.036852 |
| garch_t_control | 0.010157 | -0.028419 | 0.124172 | 2.340613 | 0.121081 | 0.028953 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.077901 |
| Absolute-return ACF(5) | 0.001741 |
| Absolute-return ACF(20) | 0.009711 |
| Absolute-return ACF(50) | 0.005676 |
| Volume ACF(1) | 0.969437 |
| OFI ACF(1) | 0.844595 |
| OFI-return correlation | 0.130532 |
| Mean spread (bps) | 5.458442 |
| Three-sigma tail fraction | 0.004000 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
