# MAS Upgrade Confirmatory Report (Contract §§29–31, 2026-11-25)

All three pre-registered confirmatory experiments were executed from frozen commit `ff10f5b5e2da9ee665afe1fd656ca90221b020e2` (source digest `8cc9fa92f1114b9d0b41d399c4c88c0fcc7fe5d5d103f1b7f10fe1ba2fb10cf7`) on three fresh, mutually disjoint 30-seed blocks (`71001–71030`, `72001–72030`, `73001–73030`) under the CSI 300 quantile-copula fixed-fine saturated domain (`configs/domains/csi300_huazheng_esg_quantile_copula_fixed_fine_saturated.json`, 1000 investors, 200 firms, 3500 trading days, burn-in 1200 days, policy window days 1200–3500, compact archival).

**Headline verdict: 18/18 primary pre-registered hypotheses are `supported` (two-sided paired $t$-tests across $n = 30$ independent seeds, Holm-adjusted $p \le 6.31 \times 10^{-5}$), and all 18/18 remain `supported` (Holm $p \le 9.57 \times 10^{-5}$) after excluding every seed with a market-layer price-cap hit.**

---

## 1. Archive Integrity & Execution Audit

| Experiment | Frozen `experiment_id` | Trials | Pre-Switch Bit-Exact Parity | Interrupted Attempts Archived | Core-Val Failures (`no_price_caps` only) | Clean Seeds |
|---|---|---:|---:|---:|---|---:|
| §29 Stackelberg (`71001–71030`) | `f8297bce3f57482b…` | 330/330 | 300/300 (0 mismatches) | 0 | 22 (`71022` [11], `71028` [11]) | 28 |
| §30 Empirical Belief (`72001–72030`) | `394a6af463c7c8b0…` | 240/240 | 210/210 (0 mismatches) | 12 (server restart; resumed via `--resume`) | 8 (`72006` [8]) | 29 |
| §31 Assurance & Remedies (`73001–73030`) | `ede705da4e1256c6…` | 450/450 | 420/420 (0 mismatches) | 0 | 31 (`73015` [1], `73022` [15], `73024` [15]) | 27 |

- Every `experiment.json`, `summary.json`, and `summary_manifest.json` SHA-256 digest passed verification (`audit_summary.json`).
- All core-validation failures are isolated `no_price_caps` hits (`1–2` hits per affected trial; `finite_positive_prices`, `finite_positive_intrinsic_values`, `cash_conservation`, `share_conservation`, and `stylized` gates all `PASS`). Per contract §§20.9, 24.10, and 31.4, all 30 seeds are retained in the primary analysis and clean-seed sensitivity is reported alongside.

---

## 2. §29 Joint Type-State Bayesian Dynamic Stackelberg Audit Mechanism (`71001–71030`, $n = 30$)

### Scenario Means

| Scenario | Eq-Wt $\Delta g$ (SD) | Cap-Wt $\Delta g$ (SD) | Mean Inflation (SD) | Wash Share | Small / Large Wash | Audit Hit Rate |
|---|---:|---:|---:|---:|---:|---:|
| `B` (1% random) | `-1.1930` | `-1.3496` | `0.888` | `20.9%` | `5.0% / 65.3%` | `26.3%` |
| `V` (public value-weighted) | `-1.3858` | `-1.2963` | `1.328` | `38.0%` | `35.6% / 47.7%` | `49.4%` |
| `J` (public jump-threshold) | `-1.3298` | `-1.4876` | `1.031` | `33.3%` | `13.8% / 78.3%` | `67.8%` |
| `A2` (2% random, 2× budget) | `-1.0725` | `-1.2065` | `0.672` | `14.0%` | `1.7% / 54.7%` | `21.1%` |
| **`S_obs` (public, history-only Bayes)** | **`-1.0726`** | **`-1.1688`** | **`0.650`** | **`14.1%`** | `3.7% / 44.7%` | `23.8%` |
| `S_obs_hid` (hidden optimal clearing) | `-1.2007` | `-1.3773` | `0.813` | `22.0%` | `5.2% / 70.0%` | **`92.1%`** |
| **`S_obs_sig` (public + $\Delta r$ Bayes)** | **`-1.0404`** | **`-1.1551`** | **`0.579`** | **`12.1%`** | `2.3% / 48.9%` | `34.1%` |
| **`S_obs_val` (public, cap-weighted)** | `-1.1321` | **`-1.1078`** | `0.765` | `17.9%` | `10.0% / 25.2%` | `24.2%` |
| `S_oracle_eq` (oracle upper bound) | `-0.9252` | `-0.9514` | `0.332` | `6.1%` | `0.7% / 27.4%` | `41.0%` |
| `S_oracle_val` (oracle cap-weighted) | `-0.9904` | `-0.9144` | `0.461` | `9.5%` | `7.0% / 5.6%` | `40.0%` |
| `S_oracle_hid` (oracle hidden clearing) | `-1.1853` | `-1.3414` | `0.871` | `20.6%` | `4.9% / 64.7%` | **`100.0%`** |

### Primary Holm-Adjusted Contrasts (`6/6 supported`) & Key Decompositions

| Contrast | Metric | Mean | 95% Unadjusted CI | Holm $p$ (Raw $p$) | Sign (`+/-`) | Clean $n=28$ Holm $p$ | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| **`S1`: `S_obs − B`** | `equal_weight_sd` | **`+0.1204`** | `[+0.1109, +0.1299]` | **`4.82e-21`** | `30/0` | `6.81e-20` | `supported` |
| **`S2a`: `S_obs − V`** | `equal_weight_sd` | **`+0.3132`** | `[+0.2893, +0.3371]` | **`2.53e-21`** | `30/0` | `1.35e-19` | `supported` |
| **`S2b`: `S_obs − J`** | `equal_weight_sd` | **`+0.2572`** | `[+0.2411, +0.2734]` | **`1.28e-23`** | `30/0` | `3.11e-22` | `supported` |
| **`S3`: `S_obs − S_obs_hid`** | `equal_weight_sd` | **`+0.1281`** | `[+0.1168, +0.1394]` | **`8.12e-20`** | `30/0` | `2.01e-19` | `supported` |
| **`S4`: `S_obs_val − V`** | `fixed_start_cap_weight_sd` | **`+0.1885`** | `[+0.1695, +0.2075]` | **`2.22e-18`** | `30/0` | `1.95e-17` | `supported` |
| **`S5`: `S_obs_sig − S_obs`** | `equal_weight_sd` | **`+0.0322`** | `[+0.0256, +0.0388]` | **`6.73e-11`** | `29/1` | `5.55e-10` | `supported` |
| *Aux*: `S_obs − A2` | `equal_weight_sd` | `-0.0001` | `[-0.0139, +0.0137]` | (`0.991`) | `15/15` | — | 1× matches 2× budget |
| *Aux*: `S_obs_sig − A2` | `equal_weight_sd` | **`+0.0321`** | `[+0.0202, +0.0440]` | (`5.78e-06`) | `26/4` | — | 1× beats 2× budget |
| *Aux*: `S_oracle_eq − S_obs` (VoI) | `equal_weight_sd` | `+0.1474` | `[+0.1277, +0.1671]` | (`2.01e-15`) | `30/0` | — | Total private VoI |
| *Aux*: `S_oracle_eq − S_obs_sig` (VoI) | `equal_weight_sd` | `+0.1152` | `[+0.0970, +0.1335]` | (`1.54e-13`) | `30/0` | — | Residual VoI (`S_obs_sig` captures 57.0% of oracle gain over `B`) |

---

## 3. §30 Zero-Leakage Online Empirical-Belief Bounded Rationality (`72001–72030`, $n = 30$)

### Scenario Means (`firm_policy="empirical_belief"`, `firm_belief_decay=0.15`)

| Scenario | Eq-Wt $\Delta g$ (SD) | Cap-Wt $\Delta g$ (SD) | Mean Inflation (SD) | Wash Share | Small / Large Wash |
|---|---:|---:|---:|---:|---:|
| `EB_B` | `-1.2123` | `-1.3804` | `1.006` | `24.3%` | `6.5% / 71.2%` |
| `EB_I` ($\tau = 0.6$) | `-0.8576` | `-0.8261` | `0.458` | `0.0%` | `0.0% / 0.0%` |
| `EB_F` (4× fine, learned online) | `-1.1053` | `-1.2594` | `0.854` | `16.6%` | `3.6% / 62.5%` |
| `EB_V` (value-weighted audits) | `-1.2785` | `-1.3179` | `1.114` | `31.0%` | `23.4% / 51.9%` |
| `EB_S_obs` (coarse-belief Stackelberg) | `-1.2619` | `-1.3328` | `1.062` | `29.0%` | `19.1% / 55.8%` |
| `EB_R16_bayes` (assigned, 16×) | `-0.8954` | `-0.9115` | `0.173` | `8.5%` | `2.5% / 26.1%` |
| `EB_F16_bayes` (firm choice, 16×) | `-0.9429` | `-0.9849` | `0.196` | `13.0%` | `3.8% / 39.0%` |
| **`EB_F100_bayes_smooth`** (firm, 100×, $\gamma=0.7$) | **`-0.8555`** | **`-0.8370`** | **`0.024`** | **`2.1%`** | `0.5% / 7.1%` |

### Primary Holm-Adjusted Contrasts (`6/6 supported`) & Key Diagnostics

| Contrast | Metric | Mean | 95% Unadjusted CI | Holm $p$ (Raw $p$) | Sign (`+/-`) | Clean $n=29$ Holm $p$ | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| **`L1`: `EB_I − EB_B`** | `equal_weight_sd` | **`+0.3547`** | `[+0.3212, +0.3883]` | **`1.19e-18`** | `30/0` | `4.92e-18` | `supported` |
| **`L2a`: `EB_V − EB_B`** | `wash_share_by_size.small` | **`+0.1691`** | `[+0.1396, +0.1986]` | **`3.17e-12`** | `30/0` | `1.12e-11` | `supported` |
| **`L2b`: `EB_V − EB_B`** | `wash_share_by_size.large` | **`-0.1928`** | `[-0.2224, -0.1631]` | **`2.89e-13`** | `0/30` | `1.14e-12` | `supported` |
| **`L3`: `EB_R16_bayes − EB_B`** | `equal_weight_sd` | **`+0.3170`** | `[+0.2814, +0.3525]` | **`9.79e-17`** | `30/0` | `3.71e-16` | `supported` |
| **`L4`: `EB_F16_bayes − EB_R16_bayes`** | `equal_weight_sd` | **`-0.0475`** | `[-0.0683, -0.0267]` | **`6.31e-05`** | `4/26` | `9.57e-05` | `supported` |
| **`L5`: `EB_F100_bayes_smooth − EB_F16_bayes`** | `equal_weight_sd` | **`+0.0874`** | `[+0.0728, +0.1020]` | **`1.73e-12`** | `30/0` | `6.14e-12` | `supported` |
| *Aux*: `L1b` `EB_F − EB_B` | `equal_weight_sd` | `+0.1070` | `[+0.0907, +0.1233]` | (`5.47e-14`) | `30/0` | — | Online fine learning |
| *Aux*: `L2c` `EB_V − EB_B` | `equal_weight_sd` | `-0.0662` | `[-0.0864, -0.0459]` | (`2.57e-07`) | `3/27` | — | Net equal-weight loss |
| *Aux*: `L2d` `EB_S_obs − EB_B` | `equal_weight_sd` | **`-0.0496`** | `[-0.0716, -0.0276]` | (`7.48e-05`) | `7/23` | — | Coarse-belief Stackelberg fragility |
| *Aux*: `L5b` `EB_F100_bayes_smooth − EB_I` | `equal_weight_sd` | `+0.0021` | `[-0.0056, +0.0098]` | (`0.577`) | `16/14` | — | Matches transparency |

---

## 4. §31 Partial-Observability Assurance Blindness & Runtime Hysteresis / Rotation Remedies (`73001–73030`, $n = 30$)

### Scenario Means

| Scenario | Eq-Wt $\Delta g$ (SD) | Cap-Wt $\Delta g$ (SD) | Mean Inf (SD) | Wash Share | Quality Gap `q_gap` | Excess Laxest `excLax` |
|---|---:|---:|---:|---:|---:|---:|
| `B` | `-1.1414` | `-1.3103` | `1.014` | `23.4%` | — | — |
| `I` ($\tau = 0.6$) | `-0.7731` | `-0.7261` | `0.439` | `0.0%` | — | — |
| `ALL_R16` (exact) | `-0.8141` | `-0.8178` | `0.142` | `8.1%` | `-0.0000` | `+33.0%` |
| `ALL_R100` (exact) | `-0.8590` | `-0.8714` | `0.072` | `10.9%` | `-0.0000` | `+4.2%` |
| `ALL_F16` (exact) | `-1.0561` | `-1.1607` | `0.575` | `26.2%` | `-0.0296` | `+61.1%` |
| `ALL_R16_bayes` | `-0.8228` | `-0.8236` | `0.186` | `9.7%` | `+0.0000` | `+31.7%` |
| `ALL_R100_bayes` | `-0.8742` | `-0.8873` | `0.122` | `14.2%` | `+0.0000` | `+13.3%` |
| `ALL_F16_bayes` | `-1.0687` | `-1.1688` | `0.605` | `26.7%` | **`-0.0269`** | **`+70.1%`** |
| `ALL_R16_smooth` ($\gamma=0.7$) | `-0.7794` | `-0.7850` | `0.116` | `3.7%` | `-0.0000` | `+22.0%` |
| `ALL_R100_smooth` ($\gamma=0.7$) | `-0.7789` | `-0.7614` | `0.025` | `1.5%` | `-0.0000` | `+17.2%` |
| **`ALL_R100_bayes_smooth`** ($\gamma=0.7$) | **`-0.7810`** | **`-0.7659`** | **`0.024`** | **`1.8%`** | `+0.0000` | `+3.7%` |
| `ALL_F16_smooth` ($\gamma=0.7$) | `-0.9412` | `-1.0308` | `0.251` | `15.4%` | `-0.0072` | `+74.7%` |
| `ALL_F100_smooth` ($\gamma=0.7$) | `-0.8022` | `-0.8068` | `0.037` | `4.1%` | `-0.0024` | `+71.1%` |
| **`ALL_F100_bayes_smooth`** ($\gamma=0.7$) | **`-0.7838`** | **`-0.7696`** | **`0.026`** | **`3.2%`** | **`-0.0012`** | `+68.4%` |
| **`ALL_ROT16_bayes`** (annual rotation) | **`-0.8098`** | **`-0.8077`** | `0.186` | `8.7%` | **`+0.0000`** | `+29.8%` |

### Primary Holm-Adjusted Contrasts (`6/6 supported`) & Key Remedies

| Contrast | Metric | Mean | 95% Unadjusted CI | Holm $p$ (Raw $p$) | Sign (`+/-`) | Clean $n=27$ Holm $p$ | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| **`N1`: `ALL_R16_bayes − B`** | `equal_weight_sd` | **`+0.3187`** | `[+0.2806, +0.3567]` | **`4.15e-16`** | `30/0` | `3.30e-14` | `supported` |
| **`N2`: `ALL_F16_bayes − ALL_R16_bayes`** | `equal_weight_sd` | **`-0.2460`** | `[-0.2793, -0.2127]` | **`8.54e-15`** | `0/30` | `4.80e-13` | `supported` |
| **`N3`: `ALL_F16_bayes − ALL_R16_bayes`** | `client_minus_unweighted_quality_gap` | **`-0.0269`** | `[-0.0321, -0.0217]` | **`3.69e-11`** | `0/30` | `8.49e-10` | `supported` |
| **`N4`: `ALL_R100_bayes − ALL_R16_bayes`** | `equal_weight_sd` | **`-0.0514`** | `[-0.0706, -0.0322]` | **`6.79e-06`** | `6/24` | `4.06e-05` | `supported` |
| **`N5`: `ALL_R100_bayes_smooth − ALL_R100_bayes`** | `equal_weight_sd` | **`+0.0931`** | `[+0.0828, +0.1035]` | **`9.20e-17`** | `30/0` | `1.39e-14` | `supported` |
| **`N6`: `ALL_F100_bayes_smooth − ALL_F16_bayes`** | `equal_weight_sd` | **`+0.2850`** | `[+0.2530, +0.3170]` | **`1.02e-16`** | `30/0` | `1.25e-14` | `supported` |
| *Aux*: `N3b` `excLax` (`F16_by − R16_by`) | `excess_laxest_wash_concentration` | `+0.3839` | `[+0.3416, +0.4262]` | (`1.23e-17`) | `30/0` | — | Tie-safe concentration |
| *Aux*: `N5c` `ALL_R100_smooth − ALL_R16_smooth` | `equal_weight_sd` | `+0.0006` | `[-0.0126, +0.0137]` | (`0.931`) | `16/14` | — | Non-monotonicity cured |
| *Aux*: `N6b` `ALL_F100_smooth − ALL_F16` | `equal_weight_sd` | `+0.2539` | `[+0.2156, +0.2922]` | (`4.44e-14`) | `30/0` | — | Exact hysteresis remedy |
| *Aux*: `N6c` `ALL_F16_smooth − ALL_F16` | `equal_weight_sd` | `+0.1149` | `[+0.0916, +0.1381]` | (`5.17e-11`) | `29/1` | — | Hysteresis at 16× |
| *Aux*: `N6d` `ALL_ROT16_bayes − ALL_F16_bayes` | `equal_weight_sd` | `+0.2589` | `[+0.2235, +0.2943]` | (`3.58e-15`) | `30/0` | — | Annual rotation remedy |
| *Aux*: `Exact_vs_Bayes_R100` | `equal_weight_sd` | `-0.0151` | `[-0.0247, -0.0056]` | (`3.06e-03`) | `10/20` | — | Confirms Prop. 6(2) blindness cost |
| *Aux*: `N6f` `ALL_F100_bayes_smooth − I` | `equal_weight_sd` | `-0.0106` | `[-0.0221, +0.0008]` | (`0.068`) | `9/21` | — | Recovers 97.3% of `I` |
