# MAS Mechanism-Boundary Extensions Confirmatory Report (Contract §32, 2026-11-26)

All four pre-registered confirmatory experiments in contract §32 were executed from frozen commit `6144298db2c3111c3c6c08d207c9b80ff3647e13` (source digest `7a916d264eac9ab09547e4961837a9aa4d7f346e598612cf696e9ea7902d3999`) on four fresh, mutually disjoint 30-seed blocks (`74001–77030`, zero overlap with all 2,914 prior seeds), totaling **1,230 confirmatory runs (0 interruptions)**.

**Headline verdict: 22/23 primary pre-registered hypotheses are `supported` (Holm-adjusted $p \le 8.88 \times 10^{-6}$) and 1/23 (`G5`) is honestly `unresolved` (Holm $p = 0.539$). Every single verdict (22 `supported`, 1 `unresolved`) is completely invariant after excluding every seed with a market-layer `no_price_caps` hit.**

---

## 1. Archive Integrity & Execution Audit (`audit_sec32.json`)

| Experiment | Frozen `experiment_id` | Trials | Pre-Switch Bit-Exact Parity | Interrupted | Core-Val Failures (`mechanical.no_price_caps` only) | Clean Seeds |
|---|---|---:|---:|---:|---|---:|
| §32A `Q(0.505, 0.423)` (`74001–74030`) | `de019a78da1c4509…` | 360/360 | 330/330 (0 mismatches) | 0 | **0** | 30 |
| §32B Phase & Pareto (`75001–75030`) | `b388700c1ce84638…` | 330/330 | 300/300 (0 mismatches) | 0 | 2 (`75021`) | 29 |
| §32C-1 Granularity (`76001–76030`) | `51e6ec0b972453be…` | 270/270 | 210/210 (0 mismatches) | 0 | 18 (`76016`, `76023`) | 28 |
| §32C-2 `ai_washing` (`77001–77030`) | `f9992a2311cf5d16…` | 270/270 | 240/240 (0 mismatches) | 0 | 98 (15 seeds; stronger `taste_median=0.22`) | 15 |

All SHA-256 digests (`experiment.json`, `summary.json`, `summary_manifest.json`) verified; zero `cash_conservation`, `share_conservation`, finiteness, or `stylized` failures across all 1,230 trials.

---

## 2. §32A Empirically Estimated `Q(κ_T = 0.505, s_P = 0.423)` Dynamics (`74001–74030`, $n = 30$)

| Contrast | Mean | 95% Unadjusted CI | Holm $p$ (Raw $p$) | Sign (`+/-`) | Clean $n=30$ | Verdict |
|---|---:|---|---:|---:|---:|---|
| **`Q1`: `Q_S_obs − Q_B`** | **`+0.0304`** | `[+0.0258, +0.0349]` | **`2.44e-13`** | `30/0` | identical | `supported` |
| **`Q2`: `Q_S_obs − Q_S_obs_hid`** (VoC; hit `38.3%` vs `87.5%`) | **`+0.0268`** | `[+0.0221, +0.0314]` | **`8.85e-12`** | `30/0` | identical | `supported` |
| **`Q3`: `Q_S_obs − Q_I`** (beats transparency `τ=0.6`!) | **`+0.0122`** | `[+0.0099, +0.0145]` | **`3.42e-11`** | `30/0` | identical | `supported` |
| **`Q4`: `Q_F16_bayes − Q_R16_bayes`** | **`-0.0156`** | `[-0.0212, -0.0101]` | **`3.17e-06`** | `2/28` | identical | `supported` |
| **`Q5`: `Q_R100_bayes_smooth − Q_R100_bayes`** | **`+0.0192`** | `[+0.0152, +0.0233]` | **`2.71e-10`** | `30/0` | identical | `supported` |
| **`Q6`: `Q_F100_bayes_smooth − Q_F16_bayes`** | **`+0.0489`** | `[+0.0395, +0.0583]` | **`4.74e-11`** | `30/0` | identical | `supported` |
| *Aux*: `Q_S_obs_sig − Q_S_obs` (high-noise $\sigma_r=0.351$ SNR boundary) | `-0.0174` | `[-0.0197, -0.0150]` | (`2.52e-15`) | `0/30` | — | Pure audit history dominates |
| *Aux*: `Q_F100_bayes_smooth − Q_I` / `Q_R100_bayes_smooth − Q_I` | **`+0.0363`** / **`+0.0438`** | `[+0.028, +0.044]` / `[+0.034, +0.054]` | (`3.45e-10` / `9.15e-10`) | `30/0` / `30/0` | — | >3× transparency's gain over `B` |

---

## 3. §32B Institutional Phase Transition & Quadratic Effort-Cost Pareto Frontier (`75001–75030`, $n = 30$)

### Sweep Means (Unrestricted Firm Choice + `detected_bayes`)

| Scenario | Eq-Wt $\Delta g$ (SD) | Mean Inf (SD) | Wash Share | `q_gap` | Total Resource (`bp`) | Verifier Effort (`model units`) |
|---|---:|---:|---:|---:|---:|---:|
| `B` | `-1.2099` | `0.845` | `20.5%` | — | `3.69` | — |
| `F100_g00` ($\gamma = 0.0$) | `-1.1664` | `0.425` | `22.1%` | `-0.0539` | `5.55` | `0.00436` |
| `F100_g03` ($\gamma = 0.3$) | `-1.1284` | `0.281` | `17.8%` | `-0.0319` | `4.83` | `0.00325` |
| `F100_g05` ($\gamma = 0.5$) | `-1.0400` | `0.124` | `10.8%` | `-0.0104` | `3.90` | `0.00327` |
| `F100_g07` ($\gamma = 0.7$) | `-0.9317` | `0.021` | `2.7%` | `-0.0011` | `2.66` | `0.00380` |
| `F100_g09` ($\gamma = 0.9$) | `-0.9067` | `0.015` | `0.0%` | `-0.0000` | `2.70` | `0.00825` |
| `F16_m000` ($a_{\min} = 0.00$) | `-1.1530` | `0.496` | `21.7%` | `-0.0188` | `5.42` | `0.00051` |
| `F16_m003` ($a_{\min} = 0.03 < a_{\text{ins}}$) | `-1.1446` | `0.393` | `21.7%` | `-0.0195` | `4.89` | `0.00057` |
| `F16_m006` ($a_{\min} = 0.06 > a_{\text{ins}}$) | `-1.0322` | `0.211` | `11.9%` | `-0.0058` | `3.92` | `0.00041` |
| **`F16_m015`** ($a_{\min} = 0.15$) | **`-0.8998`** | `0.070` | **`0.0%`** | `+0.0000` | **`2.45`** | **`0.00039`** |
| `F16_m025` ($a_{\min} = 0.25 \approx a^*$) | `-0.8991` | `0.049` | `0.0%` | `+0.0000` | `2.47` | `0.00103` |

### Primary Contrasts (`6/6 supported`)

| Contrast | Metric | Mean | 95% Unadjusted CI | Holm $p$ | Clean $n=29$ Holm $p$ | Verdict |
|---|---|---:|---|---:|---:|---|
| **`P1`: `F100_g05 − g00`** | `equal_weight_sd` | **`+0.1264`** | `[+0.1000, +0.1528]` | **`2.16e-10`** | `7.20e-10` | `supported` |
| **`P2`: `F100_g07 − g05`** | `equal_weight_sd` | **`+0.1083`** | `[+0.0768, +0.1397]` | **`9.37e-08`** | `1.47e-07` | `supported` |
| **`P3`: `F100_g07 − g00`** (resource free lunch) | `resource_cost_fraction` | **`-2.89 bp`** | `[-3.19, -2.58] bp` | **`2.10e-17`** | `1.11e-16` | `supported` |
| **`P4`: `F16_m006 − m003`** (crosses $a_{\text{ins}}=0.0346$) | `equal_weight_sd` | **`+0.1125`** | `[+0.0963, +0.1286]` | **`3.53e-14`** | `1.59e-13` | `supported` |
| **`P5`: `F16_m015 − m006`** (enters zero-wash regime) | `equal_weight_sd` | **`+0.1324`** | `[+0.1145, +0.1502]` | **`9.78e-15`** | `3.36e-14` | `supported` |
| **`P6`: `F16_m025 − F100_g07`** (Jensen effort advantage) | `assurance_effort_cost` | **`-0.00277`** (`-73%`) | `[-0.00281, -0.00273]` | **`3.77e-42`** | `6.58e-41` | `supported` |

---

## 4. §32C-1 Communication-Granularity Information-Design Curve (`76001–76030`, $n = 30$)

| Scenario / Contrast | Eq-Wt $\Delta g$ / Diff | Share of Full `S_obs` Gain | 95% CI | Holm $p$ (Raw $p$) | Clean $n=28$ Holm $p$ | Verdict |
|---|---:|---:|---|---:|---:|---|
| `NPV_B` ($G=1$, 0 bits) | `-1.1424` | `0.0%` | — | — | — | Baseline |
| `NPV_g2 − B` ($G=2$, 1 bit watchlist) | `+0.0386` | `30.7%` | `[+0.0295, +0.0477]` | (`1.44e-09`) | — | Aux |
| **`G1`: `NPV_g3 − B`** ($G=3$, 1.6 bits) | **`+0.0335`** | `26.7%` | `[+0.0283, +0.0388]` | **`4.34e-13`** | `7.12e-12` | `supported` |
| **`G2`: `NPV_g5 − g3`** ($G=5$, 2.3 bits; `+0.0790` vs `B`) | **`+0.0454`** | **`62.8%`** | `[+0.0416, +0.0493]` | **`4.54e-20`** | `9.40e-19` | `supported` |
| **`G3`: `NPV_g8 − g5`** ($G=8$, 3.0 bits; `+0.0906` vs `B`) | **`+0.0116`** | **`72.0%`** | `[+0.0074, +0.0158]` | **`8.88e-06`** | `3.71e-05` | `supported` |
| **`G4`: `NPV_S_obs − g8`** (exact; `+0.1258` vs `B`) | **`+0.0352`** | `100.0%` | `[+0.0285, +0.0419]` | **`3.40e-11`** | `1.54e-10` | `supported` |
| **`G5`: `EB_g2 − EB_S_obs`** (unquantified 2-band label) | `-0.0124` | — | `[-0.0530, +0.0283]` | `0.539` | `0.527` | `unresolved` |

---

## 5. §32C-2 Non-ESG Second-Domain Generalization (`ai_washing.json`, `77001–77030`, $n = 30$)

| Contrast | Metric | Mean | 95% Unadjusted CI | Holm $p$ (Raw $p$) | Sign (`+/-`) | Clean $n=15$ Holm $p$ | Verdict |
|---|---|---:|---|---:|---:|---:|---|
| **`A1`: `AI_S_obs − AI_B`** (`65.9% → 56.2%`) | `actions.wash` | **`-584.6`** | `[-637.2, -532.0]` | **`2.54e-19`** | `0/30` | `1.43e-09` | `supported` |
| **`A2`: `AI_S_obs − AI_V`** (`80.0% → 56.2%`) | `actions.wash` | **`-1432.5`** | `[-1523.0, -1342.0]` | **`1.54e-23`** | `0/30` | `1.83e-12` | `supported` |
| **`A3`: `AI_R16_bayes − AI_B`** | `equal_weight_sd` | **`+0.2797`** | `[+0.2183, +0.3410]` | **`3.12e-10`** | `30/0` | `4.82e-05` | `supported` |
| **`A4`: `AI_F16_bayes − AI_R16_bayes`** | `equal_weight_sd` | **`-0.1842`** | `[-0.2091, -0.1594]` | **`7.39e-15`** | `0/30` | `6.70e-07` | `supported` |
| **`A5`: `AI_F16_bayes − AI_R16_bayes`** | `q_gap` | **`-0.0978`** | `[-0.1070, -0.0886]` | **`6.41e-19`** | `1/29` | `2.13e-08` | `supported` |
| **`A6`: `AI_F100_bayes_smooth − AI_F16_bayes`** (`57.6% → 4.4%`) | `equal_weight_sd` | **`+0.3484`** | `[+0.3004, +0.3963]` | **`8.63e-15`** | `30/0` | `6.70e-07` | `supported` |
| *Aux*: `AI_S_obs − AI_S_obs_hid` (Hit-Rate Paradox: `60.9%` vs `82.5%`) | `actions.wash` | `-229.4` | `[-278.7, -180.0]` | (`2.08e-10`) | `1/29` | — | Replicates |
| *Aux*: `AI_F100_bayes_smooth − AI_I` | `equal_weight_sd` | **`+0.2677`** | `[+0.2243, +0.3111]` | (`2.64e-13`) | `30/0` | — | Crushes `τ=0.6` |
