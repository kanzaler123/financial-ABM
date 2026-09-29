# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-holdout-v14`
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
| `full_volatility_in_daily_range` | `0.022604585142024354` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.135079808492646` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.21570243168698439` | `0.03` | 100.0% | none |
| `full_heavier_than_gaussian` | `2.1488963789716307` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.3140117470247803` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.05584190256570476` | `0.1` | 90.0% | 20270102 |
| `endogenous_liquidity_feedback_increases_tail_weight` | `2.029383490550636` | `0.05` | 100.0% | none |
| `public_news_marginal_response` | `14.288445806851922` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `11.452355319527033` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.42509245360317` | `1.0` | 90.0% | 20270101 |
| `agent_information_independent_response` | `1.1334580690831693` | `1.0` | 100.0% | none |
| `empty_information_baseline_is_weakest_discovery` | `15.354553456026604` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.013597040708272619` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.030283119683752108` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.02005404729206655` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.09374247882161807` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.031061808487429454` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.012385127800055733` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.05864965540132779` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.09417837109126262` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.03863734470208385` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.16236066851423175` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `7.223279184235446` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.15168638614837776` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `6300.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.02772243229743763` | `{"maximum": 0.04, "minimum": 0.005}` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.41143145184626306` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.022605 | -0.055512 | 0.215702 | 2.148896 | 0.314012 | 0.055842 |
| no_direct_news | 0.018173 | -0.055733 | 0.570968 | 13.652688 | 0.740146 | 0.708808 |
| no_agent_information | 0.022353 | 0.000973 | 0.214309 | 2.172199 | 0.322983 | 0.074452 |
| no_information_channels | 0.018760 | 0.004142 | 0.555869 | 14.328304 | 0.740245 | 0.802732 |
| value_only | 0.013597 | -0.093742 | 0.063774 | 0.094729 | 0.202556 | 0.049409 |
| trend_only | 0.030283 | -0.031062 | 0.430308 | 13.606723 | 0.587538 | 0.073370 |
| noise_only | 0.020054 | -0.012385 | 0.325843 | 14.760474 | 0.569025 | 0.214481 |
| no_logit | 0.022605 | -0.055512 | 0.215702 | 2.148896 | 0.314012 | 0.055842 |
| no_activity_persistence | 0.022693 | -0.058222 | 0.213698 | 2.113626 | 0.310326 | 0.051017 |
| independent_signals | 0.012640 | -0.058650 | 0.033910 | 0.012008 | 0.015085 | 0.072685 |
| fixed_participation | 0.023460 | -0.094178 | 0.152732 | 1.271722 | 0.272536 | 0.056261 |
| fixed_liquidity | 0.018828 | -0.038637 | 0.077934 | 0.105968 | 0.153637 | 0.040427 |
| liquidity_stress | 0.036871 | -0.162361 | 0.376684 | 4.616801 | 0.537303 | 0.048722 |
| concentrated_wealth | 0.022922 | -0.059354 | 0.228201 | 2.554673 | 0.316782 | 0.052222 |
| garch_t_control | 0.022505 | -0.060172 | 0.381846 | 5.668899 | 0.472545 | 0.036387 |
| logit_learning | 0.027722 | 0.050513 | 0.253111 | 3.255892 | 0.396840 | 0.071595 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.135080 |
| Absolute-return ACF(5) | 0.198017 |
| Absolute-return ACF(20) | 0.061696 |
| Absolute-return ACF(50) | -0.049632 |
| Volume ACF(1) | 0.869512 |
| OFI ACF(1) | 0.411431 |
| OFI-return correlation | 0.576766 |
| Mean spread (bps) | 7.223279 |
| Three-sigma tail fraction | 0.010800 |
| Return skewness | -0.039030 |
| Leverage correlation, r(t) vs |r|(t+1) | 0.004552 |
| Crash-day fraction, r < -5% | 0.017200 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.355139 |
| Final wealth Gini | 0.220467 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
