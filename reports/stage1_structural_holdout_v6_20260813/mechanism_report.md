# Stage 1 mechanism validation

- Protocol: `stage1-mechanism-holdout-v6`
- Frozen seeds: `10`
- Decision: `FAIL`

## Gate checks

- [x] `full_runs_without_price_cap`
- [x] `cash_conservation`
- [x] `share_conservation`
- [x] `full_volatility_in_daily_range`
- [x] `full_return_autocorrelation_near_zero_across_lags`
- [ ] `full_volatility_clustering_has_decay`
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
- [x] `persistent_order_flow_is_observable`
- [x] `spread_and_depth_are_finite`
- [x] `liquidity_stress_increases_clustering`
- [x] `garch_control_is_labelled_exogenous`

## Gate evidence

| Check | Observed | Threshold | Pass rate | Failed seeds |
|---|---:|---:|---:|---|
| `full_volatility_in_daily_range` | `0.01726033728420243` | `{"maximum": 0.03, "minimum": 0.005}` | 100.0% | none |
| `full_return_autocorrelation_near_zero_across_lags` | `0.1242817795144327` | `0.2` | 100.0% | none |
| `full_volatility_clustering_has_decay` | `0.09499536973073394` | `0.03` | 60.0% | 20260901, 20260905, 20260907, 20260908 |
| `full_heavier_than_gaussian` | `0.4925520755961976` | `0.1` | 80.0% | 20260901, 20260903 |
| `full_volume_volatility_relation_positive` | `0.2895380325229302` | `0.05` | 100.0% | none |
| `full_price_discovery_bounded` | `0.025519675298997` | `0.1` | 100.0% | none |
| `endogenous_liquidity_feedback_increases_tail_weight` | `0.34464866471091704` | `0.05` | 100.0% | none |
| `public_news_marginal_response` | `7.802247118294673` | `1.0` | 100.0% | none |
| `public_news_independent_response` | `5.932510004091467` | `1.0` | 100.0% | none |
| `agent_information_marginal_response` | `1.5187526871249646` | `1.0` | 100.0% | none |
| `agent_information_independent_response` | `1.4121190157085213` | `1.0` | 100.0% | none |
| `empty_information_baseline_is_weakest_discovery` | `9.623412600082364` | `1.0` | 100.0% | none |
| `value_only_remains_active` | `0.010931059777943068` | `0.001` | 100.0% | none |
| `trend_only_remains_active` | `0.01791753952349251` | `0.002` | 100.0% | none |
| `noise_only_remains_active` | `0.05809592274985848` | `0.003` | 100.0% | none |
| `value_only_avoids_pathological_return_predictability` | `0.05626328484370505` | `0.25` | 100.0% | none |
| `trend_only_avoids_pathological_return_predictability` | `0.040558540108917004` | `0.25` | 100.0% | none |
| `noise_only_avoids_pathological_return_predictability` | `0.06420787391076119` | `0.25` | 100.0% | none |
| `independent_signals_avoids_pathological_return_predictability` | `0.02013878636080971` | `0.25` | 100.0% | none |
| `fixed_participation_avoids_pathological_return_predictability` | `0.05285597291564459` | `0.25` | 100.0% | none |
| `fixed_liquidity_avoids_pathological_return_predictability` | `0.0029786682427587695` | `0.25` | 100.0% | none |
| `liquidity_stress_avoids_pathological_return_predictability` | `0.09732351701139921` | `0.25` | 100.0% | none |
| `spread_and_depth_are_finite` | `10.077202728055788` | `{"maximum": 25.0, "minimum": 1.0}` | 100.0% | none |
| `liquidity_stress_increases_clustering` | `0.08385869110932148` | `0.0` | 100.0% | none |
| `persistent_order_flow_is_observable` | `0.41540370920948116` | `0.05` | 100.0% | none |

## Scenario medians

| Scenario | Volatility | Return ACF | |Return| ACF | Excess kurtosis | Volume-|return| corr | Price gap |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.017260 | -0.038313 | 0.094995 | 0.492552 | 0.289538 | 0.025520 |
| no_direct_news | 0.006654 | 0.007452 | 0.511872 | 9.639092 | 0.690602 | 0.199113 |
| no_agent_information | 0.016978 | 0.017569 | 0.100012 | 0.435135 | 0.289093 | 0.046928 |
| no_information_channels | 0.003681 | 0.170575 | 0.432454 | 4.589493 | 0.636229 | 0.277690 |
| value_only | 0.010931 | -0.056263 | 0.009096 | 0.198350 | 0.324996 | 0.009555 |
| trend_only | 0.017918 | 0.040559 | 0.245099 | 1.706518 | 0.420211 | 0.054016 |
| noise_only | 0.058096 | 0.064208 | 0.114836 | 0.157174 | 0.414881 | 0.560618 |
| no_logit | 0.017260 | -0.038313 | 0.094995 | 0.492552 | 0.289538 | 0.025520 |
| no_activity_persistence | 0.017283 | -0.038383 | 0.095515 | 0.509987 | 0.289423 | 0.025565 |
| independent_signals | 0.010878 | -0.020139 | 0.023033 | 0.122726 | 0.038119 | 0.019518 |
| fixed_participation | 0.017887 | -0.052856 | 0.080475 | 0.460482 | 0.281095 | 0.025718 |
| fixed_liquidity | 0.012390 | -0.002979 | 0.041179 | 0.173139 | 0.137775 | 0.024599 |
| liquidity_stress | 0.035885 | -0.097324 | 0.183500 | 0.685972 | 0.412365 | 0.044006 |
| concentrated_wealth | 0.017155 | -0.048151 | 0.098763 | 0.679708 | 0.304144 | 0.025495 |
| garch_t_control | 0.014881 | -0.038690 | 0.241164 | 1.761709 | 0.427589 | 0.022521 |

## Full-market structural metrics

| Metric | Median |
|---|---:|
| Maximum absolute return ACF, lags 1–20 | 0.124282 |
| Absolute-return ACF(5) | 0.058774 |
| Absolute-return ACF(20) | 0.057270 |
| Absolute-return ACF(50) | 0.038987 |
| Volume ACF(1) | 0.928114 |
| OFI ACF(1) | 0.415404 |
| OFI-return correlation | 0.614224 |
| Mean spread (bps) | 10.077203 |
| Three-sigma tail fraction | 0.006000 |

The GARCH-t scenario is an explicitly exogenous stress control. It is not counted as evidence that Agent interaction generated fat tails or volatility clustering.
