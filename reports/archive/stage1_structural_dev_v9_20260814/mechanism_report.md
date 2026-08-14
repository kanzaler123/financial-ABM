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
| `full_volatility_in_daily_range` | `0.012918391640429447` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.08352620013118356` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.08952610460075876` | `0.03` | 85.0% | 20261104, 20261107, 20261113 |
| `full_heavier_than_gaussian` | `0.5083198999055536` | `0.1` | 85.0% | 20261101, 20261102, 20261113 |
| `full_volume_volatility_relation_positive` | `0.2464430136778203` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.03682549811683922` | `0.1` | 95.0% | 20261104 |
| `endogenous_liquidity_feedback_increases_tail_weight` | `0.45167340749550466` | `0.05` | 90.0% | 20261104, 20261107 |
| `public_news_marginal_response` | `6.753021349353848` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `6.231548182160368` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.4338558145788136` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.1996692476459867` | `1.0` | 100.0% | none |
| `empty_information_baseline_is_weakest_discovery` | `8.88117915003628` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.010824463890327544` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.012282679637936132` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.08134010605835792` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.043598075205076525` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.0651356202540437` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.07619829776354246` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.011623785985792834` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.04148869151015386` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.0014685360348461421` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.07332032193370536` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `6.642698401671345` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.10960178516171792` | `0.0` | 100.0% | none |
| `logit_imitation_fires_and_tracks_fitness` | `2550.0` | `150.0` | 100.0% | none |
| `logit_market_remains_active_and_bounded` | `0.01598630771963798` | `{"maximum": 0.04, "minimum": 0.005}` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.46547615681212023` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.012918 | -0.009749 | 0.089526 | 0.508320 | 0.246443 | 0.036825 |
| no_direct_news | 0.011308 | 0.015965 | 0.540327 | 11.804323 | 0.693463 | 0.306612 |
| no_agent_information | 0.012817 | 0.018124 | 0.094226 | 0.496828 | 0.239408 | 0.059004 |
| no_information_channels | 0.011628 | 0.016588 | 0.557453 | 10.456403 | 0.705126 | 0.361098 |
| value_only | 0.010824 | -0.043598 | 0.000821 | 0.015378 | 0.198261 | 0.022794 |
| trend_only | 0.012283 | 0.065136 | 0.114815 | 0.675241 | 0.213410 | 0.078014 |
| noise_only | 0.081340 | 0.076198 | 0.414240 | -0.280653 | 0.601977 | 1.449845 |
| no_logit | 0.012918 | -0.009749 | 0.089526 | 0.508320 | 0.246443 | 0.036825 |
| no_activity_persistence | 0.012940 | -0.010164 | 0.086773 | 0.516480 | 0.245700 | 0.037701 |
| independent_signals | 0.010639 | -0.011624 | -0.004059 | -0.001382 | 0.042817 | 0.041306 |
| fixed_participation | 0.013449 | -0.041489 | 0.057606 | 0.403683 | 0.242434 | 0.036908 |
| fixed_liquidity | 0.012129 | -0.001469 | 0.026063 | 0.077425 | 0.141464 | 0.039276 |
| liquidity_stress | 0.024878 | -0.073320 | 0.202878 | 1.423111 | 0.425158 | 0.034887 |
| concentrated_wealth | 0.013218 | -0.018507 | 0.084884 | 0.577831 | 0.241184 | 0.037480 |
| garch_t_control | 0.013017 | -0.002234 | 0.259829 | 3.377270 | 0.430357 | 0.030125 |
| logit_learning | 0.015986 | 0.009544 | 0.146839 | 1.108876 | 0.327408 | 0.071188 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.083526 |
| Absolute-return ACF(5) | 0.080741 |
| Absolute-return ACF(20) | 0.012164 |
| Absolute-return ACF(50) | -0.015103 |
| Volume ACF(1) | 0.854165 |
| OFI ACF(1) | 0.465476 |
| OFI-return correlation | 0.428812 |
| Mean spread (bps) | 6.642698 |
| Three-sigma tail fraction | 0.006500 |
| Return skewness | 0.019706 |
| Leverage correlation, r(t) vs |r|(t+1) | -0.002229 |
| Crash-day fraction, r < -5% | 0.000000 |
| Bubble-day fraction, log gap > 10% | 0.000000 |
| Maximum drawdown | 0.250688 |
| Final wealth Gini | 0.032522 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
