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
| `full_volatility_in_daily_range` | `0.017302985171061616` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.10246495795386931` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.1681163476204161` | `0.03` | 85.0% | 20261104, 20261107, 20261113 |
| `full_heavier_than_gaussian` | `2.0869410071193024` | `0.1` | 100.0% | none |
| `full_volume_volatility_relation_positive` | `0.3813592897016597` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.01967310867410907` | `0.1` | 100.0% | none |
| `endogenous_liquidity_feedback_increases_tail_weight` | `1.9204802129146552` | `0.05` | 90.0% | 20261104, 20261107 |
| `public_news_marginal_response` | `11.477889105918358` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `9.151325646085947` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.3950131665686571` | `1.0` | 95.0% | 20261104 |
| `agent_information_independent_response` | `1.1817569841014257` | `1.0` | 85.0% | 20261101, 20261104, 20261116 |
| `empty_information_baseline_is_weakest_discovery` | `14.40868928679364` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.012801920070400968` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.021779394366291914` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.07239163898576453` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.09027290484143465` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.05576773300891413` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.060982526948066385` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.034460488555812824` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.05549378085500274` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.02072630092863134` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.11778645098618937` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `6.849867947478085` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.12605006966509968` | `0.0` | 95.0% | 20261105 |
| `logit_imitation_fires_and_tracks_fitness` | `2550.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.020221489581834333` | `{"maximum": 0.04, "minimum": 0.005}` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.4288468093342193` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.017303 | -0.020486 | 0.168116 | 2.086941 | 0.381359 | 0.019673 |
| no_direct_news | 0.017821 | -0.023847 | 0.569053 | 14.574700 | 0.740755 | 0.282480 |
| no_agent_information | 0.016834 | 0.017993 | 0.168202 | 2.463726 | 0.368094 | 0.028071 |
| no_information_channels | 0.017995 | 0.012999 | 0.578599 | 13.988546 | 0.741428 | 0.346862 |
| value_only | 0.012802 | -0.090273 | 0.021930 | 0.143200 | 0.272425 | 0.009852 |
| trend_only | 0.021779 | 0.055768 | 0.367543 | 6.971626 | 0.553557 | 0.067336 |
| noise_only | 0.072392 | 0.060983 | 0.541444 | 5.835903 | 0.704017 | 1.995251 |
| no_logit | 0.017303 | -0.020486 | 0.168116 | 2.086941 | 0.381359 | 0.019673 |
| no_activity_persistence | 0.017307 | -0.018466 | 0.163014 | 1.797717 | 0.361268 | 0.019640 |
| independent_signals | 0.012015 | -0.034460 | -0.004401 | -0.003272 | 0.041026 | 0.018424 |
| fixed_participation | 0.018013 | -0.055494 | 0.123409 | 1.179203 | 0.322692 | 0.021367 |
| fixed_liquidity | 0.014891 | -0.020726 | 0.065666 | 0.209715 | 0.203077 | 0.018517 |
| liquidity_stress | 0.035943 | -0.117786 | 0.308515 | 2.762346 | 0.533873 | 0.033063 |
| concentrated_wealth | 0.016990 | -0.025450 | 0.186740 | 2.052191 | 0.379705 | 0.020313 |
| garch_t_control | 0.016808 | -0.034119 | 0.304438 | 5.913923 | 0.498888 | 0.021028 |
| logit_learning | 0.020221 | 0.000271 | 0.209412 | 2.254273 | 0.438851 | 0.037228 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.102465 |
| Absolute-return ACF(5) | 0.173934 |
| Absolute-return ACF(20) | 0.053524 |
| Absolute-return ACF(50) | -0.027574 |
| Volume ACF(1) | 0.840927 |
| OFI ACF(1) | 0.428847 |
| OFI-return correlation | 0.503024 |
| Mean spread (bps) | 6.849868 |
| Three-sigma tail fraction | 0.009500 |
| Return skewness | 0.072241 |
| Leverage correlation, r(t) vs |r|(t+1) | -0.010155 |
| Crash-day fraction, r < -5% | 0.005500 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.273725 |
| Final wealth Gini | 0.050931 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
