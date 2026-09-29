# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-development-v13`
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
| `full_volatility_in_daily_range` | `0.020344075590618384` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.13625023152164478` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.16216956074805688` | `0.03` | 100.0% | none |
| `full_heavier_than_gaussian` | `1.3633718338817253` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.2567847964618256` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.03750738883164874` | `0.1` | 100.0% | none |
| `endogenous_liquidity_feedback_increases_tail_weight` | `1.1475085773851128` | `0.05` | 100.0% | none |
| `public_news_marginal_response` | `20.667219730165463` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `17.721849888992462` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.239159856736599` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.0660169139679214` | `1.0` | 100.0% | none |
| `empty_information_baseline_is_weakest_discovery` | `22.06206717377057` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.013732045315341505` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.025885281530705097` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.018057516981102712` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.11021482733269952` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.032512212001796825` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.011884348381835627` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.08074412397336254` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.10087497457466629` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.04552947566650586` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.17240723936795666` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `7.041554153708034` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.1573319015399907` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `6300.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.024373109478589255` | `{"maximum": 0.04, "minimum": 0.005}` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.42238062442331314` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.020344 | -0.059206 | 0.162170 | 1.363372 | 0.256785 | 0.037507 |
| no_direct_news | 0.015804 | -0.037156 | 0.553559 | 14.218329 | 0.733394 | 0.768675 |
| no_agent_information | 0.020015 | -0.010827 | 0.165660 | 1.359632 | 0.262709 | 0.047195 |
| no_information_channels | 0.016547 | -0.032955 | 0.552581 | 13.058262 | 0.736709 | 0.819250 |
| value_only | 0.013732 | -0.110215 | 0.036962 | 0.098781 | 0.167375 | 0.026059 |
| trend_only | 0.025885 | -0.032512 | 0.360783 | 11.879467 | 0.513353 | 0.049157 |
| noise_only | 0.018058 | -0.011884 | 0.259685 | 10.510291 | 0.518949 | 0.155560 |
| no_logit | 0.020344 | -0.059206 | 0.162170 | 1.363372 | 0.256785 | 0.037507 |
| no_activity_persistence | 0.020344 | -0.060338 | 0.161684 | 1.380668 | 0.249785 | 0.037470 |
| independent_signals | 0.012759 | -0.080744 | 0.028266 | 0.064755 | 0.016336 | 0.034745 |
| fixed_participation | 0.021481 | -0.100875 | 0.114621 | 0.914572 | 0.224196 | 0.037811 |
| fixed_liquidity | 0.017646 | -0.045529 | 0.069279 | 0.237448 | 0.145335 | 0.037044 |
| liquidity_stress | 0.033675 | -0.172407 | 0.314831 | 3.387216 | 0.473831 | 0.042634 |
| concentrated_wealth | 0.020652 | -0.064712 | 0.163692 | 1.323835 | 0.256045 | 0.037079 |
| garch_t_control | 0.020075 | -0.073658 | 0.311748 | 4.311077 | 0.364737 | 0.038458 |
| logit_learning | 0.024373 | 0.041431 | 0.208389 | 2.145153 | 0.332113 | 0.070270 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.136250 |
| Absolute-return ACF(5) | 0.156665 |
| Absolute-return ACF(20) | 0.027009 |
| Absolute-return ACF(50) | -0.034487 |
| Volume ACF(1) | 0.875990 |
| OFI ACF(1) | 0.422381 |
| OFI-return correlation | 0.510199 |
| Mean spread (bps) | 7.041554 |
| Three-sigma tail fraction | 0.009400 |
| Return skewness | -0.005082 |
| Leverage correlation, r(t) vs |r|(t+1) | 0.014207 |
| Crash-day fraction, r < -5% | 0.010600 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.408745 |
| Final wealth Gini | 0.205549 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
