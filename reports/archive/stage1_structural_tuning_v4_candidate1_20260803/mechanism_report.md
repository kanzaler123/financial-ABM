# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-tuning-v4`
- Frozen seeds: `5`
- Decision: `FAIL`

## Gate checks

- [x] `full_runs_without_price_cap`
- [x] `cash_conservation`
- [x] `share_conservation`
- [x] `full_volatility_in_daily_range`
- [ ] `full_return_autocorrelation_near_zero_across_lags`
- [x] `full_volatility_clustering_has_decay`
- [x] `full_heavier_than_gaussian`
- [x] `endogenous_liquidity_feedback_increases_tail_weight`
- [x] `full_volume_volatility_relation_positive`
- [x] `full_price_discovery_bounded`
- [x] `public_news_channel_has_paired_price_response`
- [ ] `agent_information_channel_has_paired_price_response`
- [x] `empty_information_baseline_is_weakest_discovery`
- [x] `single_strategy_markets_remain_active`
- [ ] `ablations_avoid_pathological_return_predictability`
- [x] `fixed_population_has_no_strategy_turnover`
- [x] `persistent_order_flow_is_observable`
- [x] `spread_and_depth_are_finite`
- [x] `liquidity_stress_increases_clustering`
- [x] `garch_control_is_labelled_exogenous`

## Gate evidence

| Check | Observed | Threshold | Pass rate | Failed seeds |
|---|---:|---:|---:|---|
| `full_volatility_in_daily_range` | `0.014673881943014395` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.20960549552442667` | `0.2` | 40.0% | 20260801, 20260803, 20260804 |
| `full_volatility_clustering_has_decay` | `0.13504669707527972` | `0.03` | 100.0% | none |
| `full_heavier_than_gaussian` | `0.8530854276315538` | `0.1` | 80.0% | 20260802 |
| `full_volume_volatility_relation_positive` | `0.28077438248762876` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.058542999291924254` | `0.1` | 100.0% | none |
| `endogenous_liquidity_feedback_increases_tail_weight` | `0.7066120412177637` | `0.05` | 100.0% | none |
| `public_news_marginal_response` | `11.826160992106924` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `11.802410793833856` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `0.9755491792077474` | `1.0` | 40.0% | 20260801, 20260802, 20260804 |
| `agent_information_independent_response` | `1.0336338411841643` | `1.0` | 60.0% | 20260804, 20260805 |
| `empty_information_baseline_is_weakest_discovery` | `12.223920212733807` | `1.0` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.807684287126295` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.014674 | -0.209605 | 0.135047 | 0.853085 | 0.280774 | 0.058543 |
| no_direct_news | 0.006857 | 0.377010 | 0.171968 | 0.563314 | 0.058130 | 0.541806 |
| no_agent_information | 0.013471 | 0.074024 | 0.069612 | 0.552212 | 0.126372 | 0.043155 |
| no_information_channels | 0.007127 | 0.428733 | 0.178445 | 0.778742 | 0.064223 | 0.582896 |
| value_only | 0.058369 | -0.967758 | 0.971603 | 1.896271 | 0.970266 | 0.046275 |
| trend_only | 0.011297 | 0.061984 | 0.068510 | 0.274439 | 0.045710 | 0.070849 |
| noise_only | 0.033417 | 0.105036 | 0.176951 | 2.586970 | 0.477930 | 0.105708 |
| no_logit | 0.014674 | -0.209605 | 0.135047 | 0.853085 | 0.280774 | 0.058543 |
| no_activity_persistence | 0.015031 | -0.237962 | 0.145121 | 0.787616 | 0.282361 | 0.052823 |
| independent_signals | 0.012537 | -0.245959 | 0.065089 | 0.158058 | 0.147905 | 0.029371 |
| fixed_participation | 0.015335 | -0.194676 | 0.150868 | 1.042557 | 0.277415 | 0.057130 |
| fixed_liquidity | 0.012910 | -0.154118 | 0.051728 | 0.184029 | 0.130943 | 0.035035 |
| liquidity_stress | 0.016312 | -0.214808 | 0.189247 | 1.334464 | 0.308887 | 0.051429 |
| concentrated_wealth | 0.014971 | -0.136546 | 0.133335 | 1.045136 | 0.329290 | 0.066364 |
| garch_t_control | 0.012823 | -0.267588 | 0.220556 | 2.556959 | 0.169807 | 0.035408 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.209605 |
| Absolute-return ACF(5) | 0.101977 |
| Absolute-return ACF(20) | 0.029355 |
| Absolute-return ACF(50) | 0.047000 |
| Volume ACF(1) | 0.966449 |
| OFI ACF(1) | 0.807684 |
| OFI-return correlation | 0.203639 |
| Mean spread (bps) | 7.758293 |
| Three-sigma tail fraction | 0.008000 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
