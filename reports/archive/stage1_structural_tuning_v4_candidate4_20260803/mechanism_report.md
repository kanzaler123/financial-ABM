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
- [x] `endogenous_liquidity_feedback_increases_tail_weight`
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
| `full_volatility_in_daily_range` | `0.011575069374885495` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.07537606204712934` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.02729953289843285` | `0.03` | 40.0% | 20260801, 20260802, 20260805 |
| `full_heavier_than_gaussian` | `0.32240145878263604` | `0.1` | 80.0% | 20260802 |
| `full_volume_volatility_relation_positive` | `0.14342424424053052` | `0.05` | 60.0% | 20260802, 20260805 |
| `full_price_discovery_bounded` | `0.08285833250881154` | `0.1` | 60.0% | 20260802, 20260805 |
| `endogenous_liquidity_feedback_increases_tail_weight` | `0.1933109419311596` | `0.05` | 80.0% | 20260802 |
| `public_news_marginal_response` | `5.90077906007556` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `5.46290853976271` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.0745246365724865` | `1.0` | 60.0% | 20260801, 20260802 |
| `agent_information_independent_response` | `0.9752072129530844` | `1.0` | 20.0% | 20260802, 20260803, 20260804, 20260805 |
| `empty_information_baseline_is_weakest_discovery` | `5.8700298133172595` | `1.0` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.8688394023282853` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.011575 | 0.002662 | 0.027300 | 0.322401 | 0.143424 | 0.082858 |
| no_direct_news | 0.004997 | 0.398250 | 0.194231 | 0.557427 | 0.044000 | 0.575982 |
| no_agent_information | 0.010778 | 0.064099 | 0.043375 | 0.211621 | 0.052132 | 0.119210 |
| no_information_channels | 0.004888 | 0.433322 | 0.214175 | 0.566411 | 0.035633 | 0.533307 |
| value_only | 0.011140 | -0.134674 | 0.047334 | 0.279505 | 0.423411 | 0.067790 |
| trend_only | 0.009909 | 0.046122 | 0.035046 | 0.095388 | 0.004005 | 0.104857 |
| noise_only | 0.022941 | 0.144622 | 0.159648 | 3.862162 | 0.444432 | 0.126194 |
| no_logit | 0.011575 | 0.002662 | 0.027300 | 0.322401 | 0.143424 | 0.082858 |
| no_activity_persistence | 0.011552 | 0.006204 | 0.030549 | 0.267187 | 0.115436 | 0.073939 |
| independent_signals | 0.009950 | -0.028715 | 0.001490 | -0.034293 | 0.096634 | 0.106537 |
| fixed_participation | 0.012981 | -0.046566 | 0.051888 | 0.639247 | 0.216621 | 0.056895 |
| fixed_liquidity | 0.010640 | -0.002866 | 0.016323 | 0.120755 | 0.093901 | 0.099421 |
| liquidity_stress | 0.012993 | -0.016211 | 0.043631 | 0.516037 | 0.137571 | 0.066904 |
| concentrated_wealth | 0.012131 | -0.021895 | 0.036345 | 0.420364 | 0.144215 | 0.064283 |
| garch_t_control | 0.009224 | -0.001623 | 0.154253 | 2.667064 | 0.129450 | 0.061832 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.075376 |
| Absolute-return ACF(5) | 0.003813 |
| Absolute-return ACF(20) | 0.046160 |
| Absolute-return ACF(50) | 0.019632 |
| Volume ACF(1) | 0.963972 |
| OFI ACF(1) | 0.868839 |
| OFI-return correlation | 0.141987 |
| Mean spread (bps) | 6.756672 |
| Three-sigma tail fraction | 0.005000 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
