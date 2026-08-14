# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-development-v8`
- Frozen seeds: `20`
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
| `full_volatility_in_daily_range` | `0.01744720477003015` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.1045373962766864` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.17866958586028292` | `0.03` | 85.0% | 20261104, 20261107, 20261113 |
| `full_heavier_than_gaussian` | `2.1385390859885844` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.39164363064954466` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.01901895314264846` | `0.1` | 100.0% | none |
| `endogenous_liquidity_feedback_increases_tail_weight` | `1.9720782917839372` | `0.05` | 90.0% | 20261104, 20261107 |
| `public_news_marginal_response` | `10.539782752307687` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `7.9844543375153485` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.3955161937707201` | `1.0` | 95.0% | 20261104 |
| `agent_information_independent_response` | `1.1672555141041614` | `1.0` | 95.0% | 20261119 |
| `empty_information_baseline_is_weakest_discovery` | `13.260464344723953` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.012780322375880782` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.022857044049137604` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.08964434836845422` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.08992713015661197` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.04876670612916583` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.09624041100118613` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.034380431037058165` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.05539729180658082` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.02072630092863134` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.11898547731863793` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `6.881973817117615` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.12457929010703295` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `2550.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.020179104141931505` | `{"maximum": 0.04, "minimum": 0.005}` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.4298812006986757` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.017447 | -0.020965 | 0.178670 | 2.138539 | 0.391644 | 0.019019 |
| no_direct_news | 0.019171 | -0.000973 | 0.568320 | 14.353740 | 0.737012 | 0.296553 |
| no_agent_information | 0.016966 | 0.016959 | 0.179272 | 2.933230 | 0.383140 | 0.032062 |
| no_information_channels | 0.019886 | -0.004157 | 0.572951 | 15.133074 | 0.744733 | 0.344638 |
| value_only | 0.012780 | -0.089927 | 0.022527 | 0.147994 | 0.272894 | 0.009897 |
| trend_only | 0.022857 | 0.048767 | 0.380580 | 11.434475 | 0.559396 | 0.086315 |
| noise_only | 0.089644 | 0.096240 | 0.700750 | 3.212674 | 0.775635 | 3.385729 |
| no_logit | 0.017447 | -0.020965 | 0.178670 | 2.138539 | 0.391644 | 0.019019 |
| no_activity_persistence | 0.017453 | -0.016985 | 0.174616 | 1.945551 | 0.376848 | 0.019053 |
| independent_signals | 0.012015 | -0.034380 | -0.003868 | -0.002953 | 0.041215 | 0.018473 |
| fixed_participation | 0.018093 | -0.055397 | 0.130526 | 1.322928 | 0.329207 | 0.021535 |
| fixed_liquidity | 0.014891 | -0.020726 | 0.065666 | 0.209715 | 0.203083 | 0.018517 |
| liquidity_stress | 0.035937 | -0.118985 | 0.315808 | 3.055454 | 0.536777 | 0.032874 |
| concentrated_wealth | 0.017130 | -0.031309 | 0.199720 | 2.011001 | 0.386929 | 0.020094 |
| garch_t_control | 0.017036 | -0.036368 | 0.313324 | 6.461395 | 0.503466 | 0.021264 |
| logit_learning | 0.020179 | -0.000482 | 0.213470 | 2.577548 | 0.440915 | 0.036875 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.104537 |
| Absolute-return ACF(5) | 0.179221 |
| Absolute-return ACF(20) | 0.056448 |
| Absolute-return ACF(50) | -0.028534 |
| Volume ACF(1) | 0.844990 |
| OFI ACF(1) | 0.429881 |
| OFI-return correlation | 0.500076 |
| Mean spread (bps) | 6.881974 |
| Three-sigma tail fraction | 0.010000 |
| Return skewness | 0.067059 |
| Leverage correlation, r(t) vs |r|(t+1) | -0.013339 |
| Crash-day fraction, r < -5% | 0.006500 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.273646 |
| Final wealth Gini | 0.050911 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
