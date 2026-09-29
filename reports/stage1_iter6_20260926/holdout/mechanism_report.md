# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-holdout-v15`
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
| `full_volatility_in_daily_range` | `0.02036458664808547` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.14687680263150912` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.14548137581818166` | `0.03` | 100.0% | none |
| `full_heavier_than_gaussian` | `1.4204100898540855` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.2572018967535873` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.037611486985336964` | `0.1` | 100.0% | none |
| `endogenous_liquidity_feedback_increases_tail_weight` | `1.2340897761997538` | `0.05` | 100.0% | none |
| `public_news_marginal_response` | `20.507140224919056` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `17.79766221285442` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.241209044963969` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.0671673551133476` | `1.0` | 100.0% | none |
| `empty_information_baseline_is_weakest_discovery` | `21.916494227701627` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.013734499369532851` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.02612719273297153` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.017572464499835624` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.11601141509099686` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.06454098807194991` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.017011036780014407` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.08361926907914553` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.0990124567506051` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.04262696178541826` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.16667739610962523` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `7.048335321473061` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.16896091218377154` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `6300.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.024705189652371562` | `{"maximum": 0.04, "minimum": 0.005}` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.4014720222125001` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.020365 | -0.061728 | 0.145481 | 1.420410 | 0.257202 | 0.037611 |
| no_direct_news | 0.015908 | -0.053899 | 0.554940 | 14.083084 | 0.742997 | 0.781768 |
| no_agent_information | 0.020019 | -0.018582 | 0.137235 | 1.382027 | 0.256872 | 0.046741 |
| no_information_channels | 0.016635 | -0.025111 | 0.561707 | 13.931052 | 0.736760 | 0.832785 |
| value_only | 0.013734 | -0.116011 | 0.041572 | 0.141975 | 0.197149 | 0.026628 |
| trend_only | 0.026127 | -0.064541 | 0.389844 | 12.669064 | 0.526447 | 0.053503 |
| noise_only | 0.017572 | -0.017011 | 0.235438 | 7.807362 | 0.488738 | 0.119385 |
| no_logit | 0.020365 | -0.061728 | 0.145481 | 1.420410 | 0.257202 | 0.037611 |
| no_activity_persistence | 0.020426 | -0.063370 | 0.141649 | 1.383734 | 0.250324 | 0.037661 |
| independent_signals | 0.012770 | -0.083619 | 0.031926 | 0.016725 | 0.030157 | 0.034369 |
| fixed_participation | 0.021439 | -0.099012 | 0.089897 | 0.821407 | 0.226534 | 0.037636 |
| fixed_liquidity | 0.017627 | -0.042627 | 0.057549 | 0.238036 | 0.161150 | 0.037118 |
| liquidity_stress | 0.034147 | -0.166677 | 0.320109 | 3.551762 | 0.486307 | 0.043085 |
| concentrated_wealth | 0.020972 | -0.062771 | 0.147935 | 1.373535 | 0.261403 | 0.038099 |
| garch_t_control | 0.019489 | -0.066007 | 0.336811 | 4.903221 | 0.441739 | 0.037660 |
| logit_learning | 0.024705 | 0.036299 | 0.187919 | 2.211859 | 0.313747 | 0.065692 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.146877 |
| Absolute-return ACF(5) | 0.152657 |
| Absolute-return ACF(20) | 0.034831 |
| Absolute-return ACF(50) | -0.032893 |
| Volume ACF(1) | 0.854595 |
| OFI ACF(1) | 0.401472 |
| OFI-return correlation | 0.515805 |
| Mean spread (bps) | 7.048335 |
| Three-sigma tail fraction | 0.010000 |
| Return skewness | 0.023945 |
| Leverage correlation, r(t) vs |r|(t+1) | 0.010797 |
| Crash-day fraction, r < -5% | 0.011800 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.363706 |
| Final wealth Gini | 0.207210 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
