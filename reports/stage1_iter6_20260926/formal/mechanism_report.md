# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-acceptance-v15`
- Frozen seeds: `50`
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
| `full_volatility_in_daily_range` | `0.020458197167813656` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.12706009427026788` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.15462586379702187` | `0.03` | 100.0% | none |
| `full_heavier_than_gaussian` | `1.1673759532965953` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.2488246341086222` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.0381315943810756` | `0.1` | 100.0% | none |
| `endogenous_liquidity_feedback_increases_tail_weight` | `0.9696376260897632` | `0.05` | 100.0% | none |
| `public_news_marginal_response` | `21.087387709054173` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `17.93174297245362` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.2389279604597632` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.0642680580825972` | `1.0` | 100.0% | none |
| `empty_information_baseline_is_weakest_discovery` | `22.43580040912316` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.01368965130424769` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.026038777117379768` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.018262423553399004` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.10374087417590443` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.04464769353438044` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.025526219197129225` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.07392827535233557` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.10059665829809863` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.03994385894098683` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.1756245875718644` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `7.072466028891809` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.16778401946421317` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `6300.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.02442592227763922` | `{"maximum": 0.04, "minimum": 0.005}` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.41600511877369517` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.020458 | -0.061933 | 0.154626 | 1.167376 | 0.248825 | 0.038132 |
| no_direct_news | 0.016172 | -0.047734 | 0.555145 | 13.902441 | 0.736544 | 0.816719 |
| no_agent_information | 0.020089 | -0.014484 | 0.154777 | 1.122937 | 0.256712 | 0.048167 |
| no_information_channels | 0.016670 | -0.028268 | 0.554116 | 12.642613 | 0.736761 | 0.870547 |
| value_only | 0.013690 | -0.103741 | 0.042491 | 0.048253 | 0.155955 | 0.027452 |
| trend_only | 0.026039 | -0.044648 | 0.375878 | 11.674121 | 0.512935 | 0.048664 |
| noise_only | 0.018262 | -0.025526 | 0.278121 | 9.053136 | 0.529111 | 0.119359 |
| no_logit | 0.020458 | -0.061933 | 0.154626 | 1.167376 | 0.248825 | 0.038132 |
| no_activity_persistence | 0.020429 | -0.062705 | 0.153715 | 1.118999 | 0.242856 | 0.038154 |
| independent_signals | 0.012772 | -0.073928 | 0.026457 | 0.032977 | 0.020098 | 0.036297 |
| fixed_participation | 0.021380 | -0.100597 | 0.105084 | 0.736778 | 0.214678 | 0.038238 |
| fixed_liquidity | 0.017619 | -0.039944 | 0.065898 | 0.179953 | 0.145617 | 0.037466 |
| liquidity_stress | 0.034027 | -0.175625 | 0.326313 | 3.534975 | 0.482219 | 0.044066 |
| concentrated_wealth | 0.020644 | -0.065712 | 0.154415 | 1.215029 | 0.241871 | 0.038167 |
| garch_t_control | 0.019503 | -0.072350 | 0.314267 | 3.862107 | 0.419087 | 0.037656 |
| logit_learning | 0.024426 | 0.043442 | 0.195020 | 2.016976 | 0.317395 | 0.072529 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.127060 |
| Absolute-return ACF(5) | 0.148801 |
| Absolute-return ACF(20) | 0.039249 |
| Absolute-return ACF(50) | -0.024828 |
| Volume ACF(1) | 0.860967 |
| OFI ACF(1) | 0.416005 |
| OFI-return correlation | 0.515150 |
| Mean spread (bps) | 7.072466 |
| Three-sigma tail fraction | 0.008800 |
| Return skewness | -0.021114 |
| Leverage correlation, r(t) vs |r|(t+1) | -0.001264 |
| Crash-day fraction, r < -5% | 0.011600 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.380784 |
| Final wealth Gini | 0.204833 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
