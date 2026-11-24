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

## §30（草案）零泄露三层有限理性阶梯：控制变量 SNR 边界与在线经验信念学习（Section 5.4 升级）

- **动机与三层阶梯设计**：
  - **L0（原始账户无模型 Q-learning，已有 Stage D.3）**：确立 0.07 bp 信号 vs. 190 bp 20 期账户回报噪声的 SNR 物理边界。
  - **L1（零泄露真实账户控制变量诊断）**：直接在真实融资费用现金流上减去同规模组均值（及一阶差分减同组均值），零参数泄露诊断显示：20 期融资回报标准差仅从 73.0 bp 降至 70.6 bp（一阶差分减同组均值仍达 0.84 bp，相对单期 ~0.0035 bp 信号仍低 240 倍），证明**企业特质基本面波动（>95% 方差）使跨截面基线相减同样无法让纯账户无模型 RL 在现实股价中介融资中生效**。
  - **L2（零泄露在线经验信念学习与折扣 Thompson 采样，`firm_policy="empirical_belief"`）**：保留与 `npv` 完全一致的连续动作空间（消除离散网格混杂）及企业本地财务评估能力，但**完全剥离对政策干预执法参数的理性预期**——企业初始先验仅取模拟前历史基线 `domain.baseline_*`，**绝不读取干预后的 `regime.audit_probability`、定向规则 `q_i`、`regime.fixed_fine`、`regime.penalty` 或真实鉴证质量 `assurance.quality`**，也无任何硬编码阈值，仅从已发生的公开执法事件在线更新：
    1. 折扣 Beta-Binomial 规模档核查概率 $\hat q_{\text{tier}}$（仅观测同档已抽查事件）；
    2. 指数平滑法定罚款 $\hat F_{\text{money}}$（仅观测已查处罚单）；
    3. 折扣 Beta-Binomial 鉴证纠错率 $\hat a_k$ 与**一致折扣 Thompson 采样**（仅观测已发生的发布前检查与更正，同时驱动自选签约与当期决策探测，解决“无人漂绿则无纠错观测”的删失反馈探索问题）。
- **实现模块**：[`EmpiricalBeliefLearner`](../src/abm/disclosure/bounded.py)，集成于 [`DisclosureMarket._bounded_decision`](../src/abm/disclosure/game.py)，配置字段 `firm_policy="empirical_belief"`、`firm_belief_decay=0.15`。
- **种子块**：
  - 已消耗开发诊断种子：`60901–60903`；
  - 开发流水线检查：`60201–60203`；
  - 确认性实验：30 个全新配对种子 `62001–62030`。
- **场景矩阵（8 场景 × 30 种子 = 240 次运行，均在 `empirical_belief` 下运行，鉴证场景与 §31 一致采用 `detected_bayes`）**：
  - `EB_B`、`EB_I`、`EB_F`（4× 法定罚款）、`EB_V`、`EB_S_obs`、`EB_R16_bayes`、`EB_F16_bayes`、`EB_F100_bayes_smooth`。
- **主要检验（双侧配对 $t$ 检验，Holm 校正，政策窗口第 1200–3500 日）**：
  - `L1`（透明度效应）：`EB_I − EB_B`（等权真实 ESG，预测 $> 0$）；
  - `L2`（无规则公式知晓下的跨档位移）：`EB_V − EB_B` 下小企业漂绿占比上升（预测 $> 0$）、大企业漂绿占比下降（预测 $< 0$），检验企业仅凭同行被查频率是否自发重现 `V` 的跨档位移；
  - `L3`（在线经验鉴证有效）：`EB_R16_bayes − EB_B`（等权真实 ESG，预测 $> 0$）；
  - `L4`（Thompson 采样自选逃逸与学习摩擦衰减）：`EB_F16_bayes − EB_R16_bayes`（等权真实 ESG，预测 $< 0$；同步报告漂绿占比上升与 `excess_laxest_wash_concentration` 升高，并检验双边学习相位滞后下逃逸幅度相对 `npv` 理性预期的衰减）；
  - `L5`（运行时滞回对有限理性企业稳健）：`EB_F100_bayes_smooth − EB_F16_bayes`（等权真实 ESG，预测 $> 0$）。
- **辅助机制检验**：
  - `L1b`（在线罚款学习）：`EB_F − EB_B > 0`（检验仅从已查处罚单在线学习 4× 罚款是否产生净改善）；
  - `L2b`（粗粒度信念下的 Stackelberg 承诺脆弱性）：`EB_S_obs − EB_B`（检验当企业忽略个体化公开承诺 $q_i^*$ 仅依赖粗粒度规模档经验频率时，个体化公开承诺增益是否衰减并向隐藏规则退化，确立 Stackelberg 机制的认知前提边界）。

---

## §31（草案）鉴证方不完全观测稳健性与运行时滞回制度修复（Study 3 信息边界与机制设计升级）

- **理论依据**：[disclosure_dynamic_propositions.md](disclosure_dynamic_propositions.md) 命题 3–4、命题 6（部分可观测 $50.5\times$ 逆向监控盲区与运行时审查滞回多周期止漂定理）。
- **实现模块**：
  - 部分可观测贝叶斯更新：`PolicyRegime.assurer_observation="detected_bayes"` 与 [`AssuranceState.detected_bayes_share`](../src/abm/disclosure/game.py)（仅用自身更正数 $C_k$ 与事后追责罚单数 $E_k$）；
  - 规范制度修复：`PolicyRegime.assurer_smoothing`（运行时审查滞回 $\gamma = 0.7$）、`PolicyRegime.assurer_min_quality`（法定最低质量相变对照）、`PolicyRegime.assurer_selection="rotating"`（年度强制轮换）；
  - 平局安全聚集与协方差指标：[`metrics.py`](../src/abm/disclosure/metrics.py) 中的 `assurance.client_minus_unweighted_quality_gap`（命题 4(2) 协方差缺口）与 `assurance.excess_laxest_wash_concentration`。
- **种子块**：
  - 已消耗开发诊断种子：`60901–60903`；
  - 开发流水线检查：`60301–60303`；
  - 确认性实验：30 个全新配对种子 `63001–63030`。
- **场景矩阵（14 场景 × 30 种子 = 420 次运行）**：
  - 基准与参照：`B`、`I`；
  - `exact` 复核：`ALL_R16`、`ALL_R100`、`ALL_F16`；
  - `detected_bayes` 稳健性：`ALL_R16_bayes`、`ALL_R100_bayes`、`ALL_F16_bayes`、`ALL_R1_bayes`；
  - 制度修复验证：`ALL_R16_smooth`、`ALL_R100_smooth`、`ALL_R100_bayes_smooth`、`ALL_F16_smooth`、`ALL_F100_smooth`、`ALL_ROT16_bayes`（共 15 场景时含 `ALL_R1_bayes`，或精简为 14 场景不含 `ALL_R1_bayes`，冻结前定稿）。
- **主要检验（Holm 校正）**：
  - `N1`（不完全观测下强责任指派有效）：`ALL_R16_bayes − B > 0`；
  - `N2`（逃逸周期对部分可观测性稳健）：`ALL_F16_bayes − ALL_R16_bayes < 0`；
  - `N3`（命题 4(2) 平局安全排序检验）：`ALL_F16_bayes` 的 `assurance.client_minus_unweighted_quality_gap` 严格 $< 0$ 且低于 `ALL_R16_bayes`（预测 $< 0$；同步报告 `excess_laxest_wash_concentration` 显著高于 `ALL_R16_bayes`）；
  - `N4`（监控盲区保留高责任非单调性）：`ALL_R100_bayes − ALL_R16_bayes`（检验逆向惯性下 3-周期是否维持）；
  - `N5`（运行时滞回治愈 3-周期与监控盲区）：`ALL_R100_bayes_smooth − ALL_R100_bayes > 0`；
  - `N6`（运行时滞回封堵自选逃逸口）：`ALL_F100_smooth − ALL_F16 > 0`（及 `ALL_F16_smooth − ALL_F16 > 0`、`ALL_ROT16_bayes − ALL_F16_bayes > 0`）。
