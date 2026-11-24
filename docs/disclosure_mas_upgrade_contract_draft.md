# 预注册草案：MAS 理论、机制设计与有限理性升级（对应契约 §§29–31 草案）

> **状态：预注册草案（未冻结；CSI 300 领域文件已本地就位并完成 SHA-256 校验，待作者批准后合并入 [disclosure_research_contract.md](disclosure_research_contract.md) 再启动确认性 30 种子运行）。**
> 本草案严格遵守本项目“先契约后运行、全新不相交种子块、默认选项向后兼容、失败不删档”的研究规范。所有新种子段均与历史研究（`41001–57030`）完全不相交。

---

## §29（草案）联合类型-状态贝叶斯动态 Stackelberg 核查机制（Study 2 算法升级）

- **理论依据**：[disclosure_dynamic_propositions.md](disclosure_dynamic_propositions.md) 命题 5（$D(q)$ 严格凹性、反序定理与两状态 RMAB 动态清案 Bellman 凹包络）。
- **实现模块**：[`BayesianAuditFilter`](../src/abm/disclosure/bayesian_audit.py)（零私有信息泄露的持久成本类型 $\times$ Markov 未结案状态联合贝叶斯滤波 + Bellman 上凹包络水填求解器）、[`stackelberg_audit_probabilities`](../src/abm/disclosure/policy.py)（全信息先知上界），集成于 [`DisclosureMarket._stackelberg_probabilities`](../src/abm/disclosure/game.py)。
- **种子块**：
  - 已消耗开发诊断种子：`60901–60903`（CSI 300 开发基准验证）；
  - 开发流水线检查：`60101–60103`；
  - 确认性实验：30 个全新配对种子 `61001–61030`。
- **场景矩阵（均在等期望预算 $K = N\bar q = 2.0$ 下，含 `A2` 双倍预算参照与全信息先知上界）**：
  1. `B`：随机核查（1%）；
  2. `V`：按市值成比例核查（公开启发式）；
  3. `J`：按评分跳升核查（公开启发式）；
  4. `A2`：双倍随机核查（2%，$K=4.0$）；
  5. `S_obs`：**可落地纯核查历史联合贝叶斯动态 Stackelberg 核查（公开，等权）**——仅用公开市值、规模档、公开融资斜率、结构成本先验与自身历史核查结果（完全不依赖评分跳升，免疫卡线规避）；
  6. `S_obs_hid`：**无混杂隐藏规则最优贝叶斯清案对照（隐藏）**——同一联合贝叶斯信念下，因规则隐藏无威慑承诺，最大化期望清理价值；
  7. `S_obs_sig`：**可落地含公开报告信号贝叶斯动态 Stackelberg 核查（公开，等权）**——在 `S_obs` 基础上融合公开报告变动 $\Delta r_i$ 平滑似然更新；
  8. `S_obs_val`：**可落地市值加权联合贝叶斯动态 Stackelberg 核查（公开，市值加权）**；
  9. `S_oracle_eq` / `S_oracle_val` / `S_oracle_hid`：**全信息先知上界**（读取企业私有 $c_i$ 与隐状态 $x_i$，仅用于信息价值 VoI 与理论天花板分解，不作为可落地主检验）。
- **主要预注册检验（双侧配对 $t$ 检验，Holm 校正，政策窗口第 1200–3500 日，全部聚焦 100% 公开信息可落地算法）**：
  - `S1`（可落地机制超越随机核查）：`S_obs − B`（等权真实 ESG 变化，预测 $> 0$）；
  - `S2`（可落地机制超越公开启发式）：`S_obs − V` 与 `S_obs − J`（等权真实 ESG 变化，预测 $> 0$）；
  - `S3`（无混杂公开承诺价值）：`S_obs − S_obs_hid`（等权真实 ESG 变化，预测 $> 0$；对比最优隐藏清案的高命中率与最优公开承诺的高合规，呈现严格命中率悖论）；
  - `S4`（市值加权正面对决）：`S_obs_val − V`（市值加权真实 ESG 变化，预测 $> 0$）；
  - `S5`（公开报告贝叶斯滤波信息价值）：`S_obs_sig − S_obs`（等权真实 ESG 变化，预测 $> 0$）。
- **辅助分解指标**：
  - 预算倍增等效对比：`S_obs − A2` 与 `S_obs_sig − A2`（检验 1× 预算贝叶斯承诺是否匹敌或超越 2× 随机核查）；
  - 私有状态信息价值（VoI）：`S_oracle_eq − S_obs` 与 `S_oracle_eq − S_obs_sig`；
  - 信念校准质量：终点未结案后验 Brier 分数。

---

## §30（草案）有界理性企业正向验证：方差缩减差分奖励学习与同行社会模仿（Section 5.4 升级）

- **理论依据**：Agogino & Tumer (2008) 差分奖励（Difference Rewards）与 Sutton & Barto (2018) 基线方差缩减、Schlag (1998) 成对比例模仿。
- **实现模块**：[`src/abm/disclosure/bounded.py`](../src/abm/disclosure/bounded.py) 中的 `DifferenceRewardTracker`、`DifferenceRewardQLearner`（`firm_policy="diff_q"`）与 `PeerImitationPolicy`（`firm_policy="peer_imitation"`），采用与 Stage D.3 完全相同的离散动作网格 `firm_action_fractions=(0.04, 0.2, 1.0)` 与单期成本约束 `firm_cost_budget_fraction=0.01`。
- **种子块**：
  - 开发流水线检查：`60201–60203`；
  - 确认性实验：30 个全新配对种子 `62001–62030`。
- **场景矩阵（分别在 `diff_q` 与 `peer_imitation` 下运行）**：`B`、`I`、`F`、`V`、`ALL_R16`、`ALL_F16`（共 12 个场景 × 30 种子 = 360 次运行）。
- **门控与主要检验（Holm 校正）**：
  - **覆盖与方差门控**：核验 `diff_q` 在 `(tier, open_case)` 紧凑状态空间内的覆盖率（目标 100%，对比 Stage D.3 的 91/100 失败）及已结算差分奖励标准差相对原始账户噪声（190 bp）的压缩倍数；
  - `L1`（透明度效应）：`diff_q` 下 `I − B`（等权真实 ESG，预测 $> 0$）；
  - `L2`（跨档位移）：`diff_q` 下 `V − B` 的小企业漂绿决策占比变化与大企业漂绿决策占比变化（预测小升、大降）；
  - `L3`（强指派鉴证）：`diff_q` 下 `ALL_R16 − B`（等权真实 ESG，预测 $> 0$）；
  - `L4`（自选削弱）：`diff_q` 下 `ALL_F16 − ALL_R16`（等权真实 ESG，预测 $< 0$）。

---

## §31（草案）鉴证方不完全观测贝叶斯更新稳健性（Study 3 信息边界升级）

- **动机**：回应研究局限中关于鉴证方年末精确观测真实客户污染率 $s_k$ 的强假设（Study 3 信息结构边界）。
- **实现模块**：`PolicyRegime.assurer_observation="detected_bayes"` 与 [`AssuranceState.detected_bayes_share`](../src/abm/disclosure/game.py)，鉴证方仅根据自身发布前更正数 $C_k$ 与监管事后追责罚单数 $E_k$ 按有效观测暴露 $n_k [a_k + (1-a_k)\bar q]$ 做经验贝叶斯平滑更新；平局安全聚集指标实现于 [`metrics.py`](../src/abm/disclosure/metrics.py)（`assurance.excess_laxest_wash_concentration` 与 `assurance.client_minus_unweighted_quality_gap`）。
- **种子块**：
  - 开发检查：`60301–60303`；
  - 确认性实验：30 个全新配对种子 `63001–63030`。
- **场景矩阵（10 场景 × 30 种子 = 300 次运行，涵盖 `exact` v2 复核与 `detected_bayes` 对比；待 Phase 2 补充机制修复情景）**：
  - `B`、`I`；
  - `exact`（v2 协议复核）：`ALL_R16`、`ALL_R100`、`ALL_F16`；
  - `detected_bayes`：`ALL_R16_bayes`、`ALL_R100_bayes`、`ALL_F16_bayes`、`ALL_R1_bayes`、`LRG_R16_bayes`。
- **主要检验（Holm 校正）**：
  - `N1`（不完全观测下强责任指派有效）：`ALL_R16_bayes − B > 0`；
  - `N2`（逃逸周期稳健性）：`ALL_F16_bayes − ALL_R16_bayes < 0`；
  - `N3`（平局安全最宽松方超额聚集）：`ALL_F16_bayes` 的 `assurance.excess_laxest_wash_concentration` 显著高于 `ALL_R16_bayes`（预测 $> 0$；同步报告 `assurance.client_minus_unweighted_quality_gap < 0`）；
  - `N4`（非单调性信息敏感度，双侧无预设方向）：`ALL_R100_bayes − ALL_R16_bayes`（检验噪声更新是否阻尼或保留高责任 3-周期崩塌）。
