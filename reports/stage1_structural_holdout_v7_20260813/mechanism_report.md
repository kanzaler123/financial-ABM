# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-holdout-v7`
- Frozen seeds: `10`
- Decision: `FAIL`

## Gate checks

- [x] `full_runs_without_price_cap`
- [x] `cash_conservation`
- [x] `share_conservation`
- [x] `full_volatility_in_daily_range`
- [x] `full_return_autocorrelation_near_zero_across_lags`
- [ ] `full_volatility_clustering_has_decay`
- [ ] `full_heavier_than_gaussian`
- [ ] `endogenous_liquidity_feedback_increases_tail_weight`
- [x] `full_volume_volatility_relation_positive`
- [x] `full_price_discovery_bounded`
- [x] `public_news_channel_has_paired_price_response`
- [x] `agent_information_channel_has_paired_price_response`
- [x] `empty_information_baseline_is_weakest_discovery`
- [x] `single_strategy_markets_remain_active`
- [x] `ablations_avoid_pathological_return_predictability`
- [x] `fixed_population_has_no_strategy_turnover`
- [x] `logit_imitation_fires_and_tracks_fitness`
- [ ] `logit_market_remains_active_and_bounded`
- [x] `persistent_order_flow_is_observable`
- [x] `spread_and_depth_are_finite`
- [x] `liquidity_stress_increases_clustering`
- [x] `garch_control_is_labelled_exogenous`

## Gate evidence

| Check | Observed | Threshold | Pass rate | Failed seeds |
|---|---:|---:|---:|---|
| `full_volatility_in_daily_range` | `0.022649344128199647` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.1387552004038443` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.07183974785720934` | `0.03` | 70.0% | 20261305, 20261308, 20261310 |
| `full_heavier_than_gaussian` | `0.19226795741286962` | `0.1` | 60.0% | 20261303, 20261304, 20261308, 20261310 |
| `full_volume_volatility_relation_positive` | `0.2665588267638517` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.023988510171915618` | `0.1` | 100.0% | none |
| `endogenous_liquidity_feedback_increases_tail_weight` | `0.15399271478843568` | `0.05` | 70.0% | 20261303, 20261305, 20261310 |
| `public_news_marginal_response` | `6.846512640759446` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `6.11173240960979` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.5641160262052112` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.474623575386917` | `1.0` | 100.0% | none |
| `empty_information_baseline_is_weakest_discovery` | `9.644952023185581` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.011512111603152127` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.027934692146434577` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.06116252965202762` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.10781149511077334` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.03042567876052117` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.043429276970136516` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.05073464981325611` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.04631154944966087` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.008360482201841268` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.09770216451046718` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `10.55881129838792` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.039156556562407276` | `0.0` | 80.0% | 20261301, 20261306 |
| `logit_imitation_fires_and_tracks_fitness` | `2550.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.03017352448329455` | `{"maximum": 0.03, "minimum": 0.005}` | 50.0% | 20261301, 20261302, 20261306, 20261307, 20261308 |
| `persistent_order_flow_is_observable` | `0.41068859588413664` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.022649 | -0.035409 | 0.071840 | 0.192268 | 0.266559 | 0.023989 |
| no_direct_news | 0.008221 | 0.055261 | 0.457322 | 4.357642 | 0.653421 | 0.152882 |
| no_agent_information | 0.021781 | 0.029405 | 0.082314 | 0.191833 | 0.257923 | 0.040839 |
| no_information_channels | 0.006667 | 0.149482 | 0.519281 | 10.157539 | 0.699205 | 0.243999 |
| value_only | 0.011512 | -0.107811 | 0.023858 | 0.082930 | 0.389352 | 0.006650 |
| trend_only | 0.027935 | 0.030426 | 0.254825 | 1.514458 | 0.515227 | 0.061838 |
| noise_only | 0.061163 | 0.043429 | 0.042690 | -0.056121 | 0.370170 | 0.372365 |
| no_logit | 0.022649 | -0.035409 | 0.071840 | 0.192268 | 0.266559 | 0.023989 |
| no_activity_persistence | 0.022696 | -0.034789 | 0.071334 | 0.204049 | 0.265645 | 0.024041 |
| independent_signals | 0.011363 | -0.050735 | 0.045414 | 0.049710 | 0.053806 | 0.016405 |
| fixed_participation | 0.023175 | -0.046312 | 0.056049 | 0.173139 | 0.243440 | 0.024203 |
| fixed_liquidity | 0.014482 | -0.008360 | 0.062568 | 0.104141 | 0.170666 | 0.019977 |
| liquidity_stress | 0.048297 | -0.097702 | 0.118186 | 0.243633 | 0.327413 | 0.045334 |
| concentrated_wealth | 0.022873 | -0.053330 | 0.069346 | 0.316016 | 0.263322 | 0.025090 |
| garch_t_control | 0.016154 | -0.016348 | 0.229622 | 1.571949 | 0.393278 | 0.022277 |
| logit_learning | 0.030174 | 0.001613 | 0.033086 | 0.143805 | 0.248211 | 0.048375 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.138755 |
| Absolute-return ACF(5) | 0.068041 |
| Absolute-return ACF(20) | 0.014080 |
| Absolute-return ACF(50) | 0.044981 |
| Volume ACF(1) | 0.897870 |
| OFI ACF(1) | 0.410689 |
| OFI-return correlation | 0.655126 |
| Mean spread (bps) | 10.558811 |
| Three-sigma tail fraction | 0.005500 |
| Return skewness | -0.011395 |
| Leverage correlation, r(t) vs |r|(t+1) | -0.001386 |
| Crash-day fraction, r < -5% | 0.015500 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.372812 |
| Final wealth Gini | 0.055845 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
