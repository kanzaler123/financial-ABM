# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-holdout-v12`
- Frozen seeds: `10`
- Decision: `FAIL`

## Gate checks

- [ ] `full_runs_without_price_cap`
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
| `full_volatility_in_daily_range` | `0.02088437517201358` | `{"maximum": 0.03, "minimum": 0.005}` | 90.0% | 20261360 |
| `full_return_autocorrelation_near_zero_across_lags` | `0.12196833707427035` | `0.2` | 90.0% | 20261359 |
| `full_volatility_clustering_has_decay` | `0.22533259592323113` | `0.03` | 100.0% | none |
| `full_heavier_than_gaussian` | `3.5176598226720497` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.41191629382159983` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.0238331063458923` | `0.1` | 100.0% | none |
| `endogenous_liquidity_feedback_increases_tail_weight` | `3.1343045390117323` | `0.05` | 100.0% | none |
| `public_news_marginal_response` | `8.486432859269222` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `4.439217366045268` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `2.133375726605914` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.1408975758728883` | `1.0` | 90.0% | 20261356 |
| `empty_information_baseline_is_weakest_discovery` | `11.648434335952143` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.013106968685722049` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.02746720009511145` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.08107737835220004` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.09803827742861719` | `0.25` | 90.0% | 20261360 |
| `trend_only_avoids_pathological_return_predictability` | `0.08420503329494121` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.08486343549741196` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.05746036548691624` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.0721829743692291` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.029083613911690537` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.09978084521561104` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `7.0254547115069075` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.09991420317182326` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `2550.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.024696008331995246` | `{"maximum": 0.04, "minimum": 0.005}` | 90.0% | 20261360 |
| `persistent_order_flow_is_observable` | `0.4396189722686535` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.020884 | -0.043027 | 0.225333 | 3.517660 | 0.411916 | 0.023833 |
| no_direct_news | 0.019482 | -0.046035 | 0.577701 | 14.699929 | 0.731903 | 0.238760 |
| no_agent_information | 0.020540 | 0.030644 | 0.236306 | 3.080190 | 0.407886 | 0.052057 |
| no_information_channels | 0.019915 | -0.001226 | 0.546693 | 13.073934 | 0.739398 | 0.250622 |
| value_only | 0.013107 | -0.098038 | 0.065843 | 0.261399 | 0.394866 | 0.007730 |
| trend_only | 0.027467 | 0.084205 | 0.480948 | 13.820270 | 0.589744 | 0.089519 |
| noise_only | 0.081077 | 0.084863 | 0.745213 | 4.290808 | 0.760571 | 2.363281 |
| no_logit | 0.020884 | -0.043027 | 0.225333 | 3.517660 | 0.411916 | 0.023833 |
| no_activity_persistence | 0.020911 | -0.041156 | 0.225983 | 3.644328 | 0.404818 | 0.023642 |
| independent_signals | 0.012324 | -0.057460 | 0.033410 | -0.027433 | 0.053943 | 0.012961 |
| fixed_participation | 0.021965 | -0.072183 | 0.169852 | 1.913333 | 0.376042 | 0.022851 |
| fixed_liquidity | 0.017137 | -0.029084 | 0.078029 | 0.209822 | 0.269250 | 0.016917 |
| liquidity_stress | 0.045197 | -0.099781 | 0.348435 | 2.697808 | 0.565724 | 0.038089 |
| concentrated_wealth | 0.019860 | -0.046536 | 0.222395 | 3.078713 | 0.423374 | 0.025421 |
| garch_t_control | 0.021098 | -0.042208 | 0.387960 | 7.074426 | 0.562628 | 0.022960 |
| logit_learning | 0.024696 | -0.021999 | 0.242928 | 2.997966 | 0.446434 | 0.037426 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.121968 |
| Absolute-return ACF(5) | 0.231985 |
| Absolute-return ACF(20) | 0.088513 |
| Absolute-return ACF(50) | -0.029453 |
| Volume ACF(1) | 0.853152 |
| OFI ACF(1) | 0.439619 |
| OFI-return correlation | 0.517733 |
| Mean spread (bps) | 7.025455 |
| Three-sigma tail fraction | 0.012500 |
| Return skewness | 0.134684 |
| Leverage correlation, r(t) vs |r|(t+1) | -0.016575 |
| Crash-day fraction, r < -5% | 0.010500 |
| Bubble-day fraction, log gap > 10% | 0.001500 |
| Maximum drawdown | 0.340470 |
| Final wealth Gini | 0.065932 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
