# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-acceptance-v13`
- Frozen seeds: `50`
- Decision: `FAIL`

## Gate checks

- [ ] `full_runs_without_price_cap`
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
- [x] `logit_market_remains_active_and_bounded`
- [x] `persistent_order_flow_is_observable`
- [x] `spread_and_depth_are_finite`
- [x] `liquidity_stress_increases_clustering`
- [x] `garch_control_is_labelled_exogenous`

## Gate evidence

| Check | Observed | Threshold | Pass rate | Failed seeds |
|---|---:|---:|---:|---|
| `full_volatility_in_daily_range` | `0.013440175457645526` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.0672952264261438` | `0.2` | 98.0% | 20261251 |
| `full_volatility_clustering_has_decay` | `0.08910514094837782` | `0.03` | 74.0% | 20261214, 20261215, 20261216, 20261217, 20261222, 20261225, 20261227, 20261236, 20261238, 20261249, 20261250, 20261256, 20261259 |
| `full_heavier_than_gaussian` | `0.624146700740327` | `0.1` | 70.0% | 20261214, 20261215, 20261216, 20261217, 20261222, 20261225, 20261229, 20261231, 20261236, 20261238, 20261240, 20261249, 20261250, 20261256, 20261259 |
| `full_volume_volatility_relation_positive` | `0.24065630548919104` | `0.05` | 90.0% | 20261216, 20261222, 20261236, 20261250, 20261259 |
| `full_price_discovery_bounded` | `0.0430388660855579` | `0.1` | 86.0% | 20261214, 20261222, 20261225, 20261236, 20261250, 20261256, 20261259 |
| `endogenous_liquidity_feedback_increases_tail_weight` | `0.4417077824057287` | `0.05` | 74.0% | 20261214, 20261215, 20261216, 20261217, 20261222, 20261225, 20261236, 20261238, 20261246, 20261249, 20261250, 20261256, 20261259 |
| `public_news_marginal_response` | `14.561217469581933` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `15.37873779342587` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.1455898562615208` | `1.0` | 80.0% | 20261214, 20261215, 20261216, 20261218, 20261222, 20261236, 20261237, 20261240, 20261241, 20261244 |
| `agent_information_independent_response` | `1.0803539891089313` | `1.0` | 82.0% | 20261211, 20261213, 20261217, 20261228, 20261234, 20261237, 20261243, 20261249, 20261250 |
| `empty_information_baseline_is_weakest_discovery` | `16.825761878575218` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.011774390836407316` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.015137415170852458` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.015850947561221927` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.033536090573202004` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.013102329073217794` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.003724298924433175` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.02158711823932235` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.04368133767419384` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.016490911565528454` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.07516198592190494` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `6.626624896533217` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.17285446081369976` | `0.0` | 96.0% | 20261251, 20261259 |
| `logit_imitation_fires_and_tracks_fitness` | `6300.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.014794139985415608` | `{"maximum": 0.04, "minimum": 0.005}` | 96.0% | 20261211, 20261251 |
| `persistent_order_flow_is_observable` | `0.4546414711069435` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.013440 | -0.023405 | 0.089105 | 0.624147 | 0.240656 | 0.043039 |
| no_direct_news | 0.016121 | 0.001796 | 0.574811 | 16.348027 | 0.742277 | 0.813137 |
| no_agent_information | 0.013363 | 0.005394 | 0.083820 | 0.655263 | 0.234875 | 0.057963 |
| no_information_channels | 0.017948 | 0.028241 | 0.563768 | 15.183129 | 0.738145 | 0.880607 |
| value_only | 0.011774 | -0.033536 | 0.011111 | 0.046804 | 0.100254 | 0.050789 |
| trend_only | 0.015137 | 0.013102 | 0.217857 | 3.735603 | 0.382937 | 0.125081 |
| noise_only | 0.015851 | -0.003724 | 0.239472 | 6.084302 | 0.484566 | 0.945140 |
| no_logit | 0.013440 | -0.023405 | 0.089105 | 0.624147 | 0.240656 | 0.043039 |
| no_activity_persistence | 0.013443 | -0.022210 | 0.091569 | 0.580250 | 0.240212 | 0.041676 |
| independent_signals | 0.011590 | -0.021587 | 0.007663 | -0.013502 | 0.025320 | 0.059765 |
| fixed_participation | 0.013775 | -0.043681 | 0.083991 | 0.532209 | 0.254835 | 0.041035 |
| fixed_liquidity | 0.012789 | -0.016491 | 0.044307 | 0.131177 | 0.165553 | 0.053590 |
| liquidity_stress | 0.021341 | -0.075162 | 0.274176 | 4.168577 | 0.508020 | 0.045394 |
| concentrated_wealth | 0.013566 | -0.024884 | 0.089794 | 0.721167 | 0.245190 | 0.045840 |
| garch_t_control | 0.012462 | -0.017539 | 0.311908 | 6.912130 | 0.373958 | 0.043745 |
| logit_learning | 0.014794 | 0.004293 | 0.123235 | 1.257912 | 0.307551 | 0.060861 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.067295 |
| Absolute-return ACF(5) | 0.079794 |
| Absolute-return ACF(20) | 0.027923 |
| Absolute-return ACF(50) | 0.013730 |
| Volume ACF(1) | 0.896530 |
| OFI ACF(1) | 0.454641 |
| OFI-return correlation | 0.324552 |
| Mean spread (bps) | 6.626625 |
| Three-sigma tail fraction | 0.005200 |
| Return skewness | 0.009345 |
| Leverage correlation, r(t) vs |r|(t+1) | 0.002284 |
| Crash-day fraction, r < -5% | 0.001000 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.361235 |
| Final wealth Gini | 0.082194 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
