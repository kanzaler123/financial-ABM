# 多智能体动态与博弈命题：鉴证共演化与 Stackelberg 核查（Propositions 3–5）

> **状态：理论推导与数值核验。** 本文档承接单企业静态基准 [disclosure_inspection_propositions.md](disclosure_inspection_propositions.md)（Propositions 1–2），为 Study 2（定向核查）和 Study 3（第三方鉴证）中的多智能体涌现现象提供形式化数学刻画，并导出预算约束下的 Stackelberg 核查分配结构。全部解析阈值与不等式均在 [`tests/test_disclosure_dynamic_propositions.py`](../tests/test_disclosure_dynamic_propositions.py) 中与 [`src/abm/disclosure/financing.py`](../src/abm/disclosure/financing.py) 逐项数值核验。

---

## 1. 记号与公共前提

沿用 Propositions 1–2 的内点设定（$\sigma = 0, p = 0$，日度报告、未公开核查结果、无均值回归或给定持续性年金 $A_r > 0$）：

- 交易日周期：$P_d = 60$ 日，贴现率 $r_d = 0.05$/年，252 交易日/年，期贴现因子 $\beta = e^{-r_d P_d / 252} \approx 0.988166$，首次核查贴现因子 $\beta_1 = e^{-r_d (t_{a,1} - t_0)/252} = \beta$（在默认时序 $t_0 \equiv 0 \pmod{60}, t_w = t_0 + 20 \implies t_{a,1} = t_0 + 60$ 下）。
- 监管基础核查率：$\bar q \in (0,1)$（默认 $\bar q = 0.01$），每鉴证年含 $m = 4$ 个决策期（240 交易日）。
- 鉴证质量为 $a \in [0, 1]$ 时，[`assured_signal_factors`](../src/abm/disclosure/financing.py) 给出：
  $$A_w(a) = \frac{1}{252}\sum_{u=1}^{H}\mathbf{1}_{\{q(u)\ge t_w\}}(1-\bar q)^{n_a(u)}(1-a)^{n_c(u)} e^{-r_d u/252}, \qquad D(a) = \frac{\bar q(1-a)^{c_0}\beta_1}{1 - (1-\bar q)(1-a)\beta},$$
  其中默认时序下 $c_0 = 1$（新虚报在首次监管核查前先经历披露日的 1 次发布前检查）。
- 通用鉴证止漂门槛 $a^* \in (0,1)$ 定义为 $(1-\tau)A_w(a^*) = \sqrt{w}\,A_r$ 的唯一解；在默认参数（$\tau = 0.1, w = 0.2, \bar q = 0.01$）下，$a^* = 0.082707$。
- 对任意无鉴证核查概率 $q \in [0, 1]$，[`financing_discount_factors`](../src/abm/disclosure/financing.py) 给出：
  $$A_w^{\text{aud}}(q) = \frac{1}{252}\sum_{u=1}^{H}\mathbf{1}_{\{q(u)\ge t_w\}}(1-q)^{n_a(u)} e^{-r_d u/252}, \qquad D^{\text{aud}}(q) = \frac{\beta_1 q}{(1-\beta) + \beta q}.$$
  无罚款/已有未结案的核查止漂门槛 $q^* \in (0,1)$ 定义为 $(1-\tau)A_w^{\text{aud}}(q^*) = \sqrt{w}\,A_r$ 的唯一解；默认参数下 $q^* = 0.104924$。

---

## 2. 命题 3：指派鉴证下的罚款保险、无零漂绿不动点与高责任 3-周期非单调性

考虑随机指派给单一鉴证方 $k$ 的连续测度（或代表性异质分布）客户群，每家内点企业在质量 $a$ 下的洁净开案规模门槛为：
$$P^*(a) \equiv \frac{4 c F\, D(a)}{B^2\big[(1-\tau)^2 A_w(a)^2 / w - A_r^2\big]} \quad (a < a^*),$$
记当质量为 $a$ 时满足 $P_i > P^*(a)$（即无未结案时亦主动发起漂绿）的客户占比为 $\theta(a) \in [0,1]$，且对所有 $a \ge a^*$ 有 $\theta(a) = 0$。在每一期 $p \in \{1,\dots,m\}$ 中，携带虚报的报告在披露日经概率 $a$ 的发布前检查（在更正前计入受检虚报占比 $s_{k,y}$）与期末概率 $\bar q$ 的监管核查，因此单期存续率为 $\rho(a) \equiv (1-a)(1-\bar q)$。鉴证方每年初按短视最优反应更新质量：$a_{k,y+1} = \min(1, \lambda_L s_{k,y})$，其中 $\lambda_L = L\bar q / (2\kappa)$。

**命题 3（指派鉴证动力学）.** 设 $\tau < \tau^*(0)$（从而 $a^* > 0$）且 $\theta(0) > 0$。则：

1. **（弱责任罚款保险区间）** 当且仅当
   $$\left|\frac{D'(0)}{D(0)}\right| > \left|\frac{2 A_w'(0)}{A_w(0)}\right| \cdot \frac{(1-\tau)^2 A_w(0)^2 / w}{(1-\tau)^2 A_w(0)^2 / w - A_r^2}$$
   时，存在非空开区间 $(0, a_{\mathrm{ins}}) \subset (0, a^*)$，使得对任意 $a \in (0, a_{\mathrm{ins}})$ 均有 $P^*(a) < P^*(0)$。在默认参数下，左端为 $45.84$，右端为 $19.55 \times 1.305 = 25.51$，不等式严格成立；精确罚款保险区间为 $(0, 0.0346)$，门槛最低点在 $a_{\min} = 0.0156$ 处达到 $P^*(a_{\min}) / P^*(0) = 0.8728$。此外，若 $\lambda_L < a^*$（如 $1\times$ 法定责任 $\lambda_L = 0.0525 < 0.0827$），鉴证质量恒满足 $a_{k,y} \le \lambda_L < a^*$，永远无法达到止漂门槛。
2. **（零漂绿稳态的不存在性与内生质量回落）** 对任意有限责任斜率 $\lambda_L \in (0, \infty)$，耦合均值场映射 $(u_y, a_y) \mapsto (u_{y+1}, a_{y+1})$ **不存在**满足 $a_{\mathrm{eq}} \ge a^*$ 的不动点。进一步地，任意进入止漂年（$a_y \ge a^*$）的轨迹满足存量几何收缩
   $$u_{y+1} = \rho(a_y)^m u_y, \qquad s_y = \Phi_m(a_y)\, u_y, \qquad \Phi_m(a) \equiv \frac{1 - \rho(a)^m}{m(1-\rho(a))},$$
   因此至多经过有限个连续止漂年后必然出现 $s_y < a^*/\lambda_L \implies a_{y+1} < a^*$，触发漂绿复发。
3. **（高责任精确 3-周期与责任威慑的非单调性）** 设在复发年（$a_y = 0, u_y = 0$）中，高倾向客户 $\theta(0)$ 在每期决策日均立即发起漂绿（并在被 1% 抽查清除后于下期重新发起），于是 $s_y = \theta(0)$ 且年末留存 $u_{y+1} = (1-\bar q)\theta(0)$。若责任斜率满足
   $$\lambda_L \ge \lambda_L^{\mathrm{3c}} \equiv \frac{\max\big(1,\, m\,a^*/(1-\bar q)\big)}{\theta(0)},$$
   则该确定性映射从 $(u_0, a_0) = (0, 0)$ 起进入一个**精确的 3-周期轨道**：
   $$\big((u_0, a_0),\; (u_1, a_1),\; (u_2, a_2)\big) = \left((0, 0),\; \big((1-\bar q)\theta(0),\, 1\big),\; \Big(0,\, \min\!\Big(1,\, \frac{\lambda_L(1-\bar q)\theta(0)}{m}\Big)\Big)\right),$$
   其长期平均漂绿决策占比恒为 $\frac{1}{3}\theta(0) > 0$（即便 $\lambda_L \to \infty$ 亦不趋于零）。反之，若 $\theta(a)$ 在 $[0, a^*]$ 上连续且 $\theta(a^*) = 0$，则存在中等责任斜率 $\lambda_L^\dagger < \lambda_L^{\mathrm{3c}}$（保持质量在 $a^*$ 附近小幅波动而不发生彻底清空后的零质量崩塌），其长期平均漂绿占比严格低于 $\frac{1}{3}\theta(0)$。由此证明：**在年度短视鉴证更新下，长期漂绿率对鉴证方责任 $L$ 严格非单调**。

*证明.*
1. 由 $P^*(a) \propto D(a) / \big[(1-\tau)^2 A_w(a)^2/w - A_r^2\big]$ 取对数求导，在 $a=0$ 处有
   $$\left.\frac{d\ln P^*(a)}{da}\right|_{a=0} = \frac{D'(0)}{D(0)} - \frac{2(1-\tau)^2 A_w(0) A_w'(0)/w}{(1-\tau)^2 A_w(0)^2/w - A_r^2}.$$
   因 $D'(0) < 0$ 且 $A_w'(0) < 0$，当且仅当题设不等式成立时 $\left.\frac{d\ln P^*}{da}\right|_{a=0} < 0$，由连续性即存在 $(0, a_{\mathrm{ins}})$ 使 $P^*(a) < P^*(0)$。又因为 $s_{k,y} \in [0, 1]$，故 $a_{k,y+1} = \min(1, \lambda_L s_{k,y}) \le \lambda_L$。
2. 反证：设 $(u_{\mathrm{eq}}, a_{\mathrm{eq}})$ 为不动点且 $a_{\mathrm{eq}} \ge a^*$。此时该年内无企业发起或追加漂绿，故进入第 $p$ 期检查时的虚报存量占比为 $u_{p-1} = \rho(a_{\mathrm{eq}})^{p-1} u_{\mathrm{eq}}$，年末剩余存量满足 $u_{\mathrm{eq}} = \rho(a_{\mathrm{eq}})^m u_{\mathrm{eq}}$。因 $\rho(a_{\mathrm{eq}}) \le (1-a^*)(1-\bar q) < 1$，必有 $u_{\mathrm{eq}} = 0$，从而 $s_{\mathrm{eq}} = \Phi_m(a_{\mathrm{eq}})\cdot 0 = 0$，推出 $a_{\mathrm{eq}} = \min(1, \lambda_L \cdot 0) = 0 < a^*$，矛盾。对任意轨迹，若第 $y, y+1, \dots, y+t-1$ 年均维持 $a \ge a^*$，则 $u_{y+t} \le \rho(a^*) ^{m t} u_y \to 0$，故必在有限步内出现 $\lambda_L \Phi_m(a) u < a^*$。
3. 设 $\lambda_L \ge \lambda_L^{\mathrm{3c}}$，直接迭代三步：
   - **Year 0**（$u_0 = 0, a_0 = 0$）：每期恰有 $\theta(0)$ 比例的企业发起漂绿（并携带虚报接受检查），故 $s_0 = \theta(0)$，第 4 期期末审计后存量为 $u_1 = (1-\bar q)\theta(0)$，下年质量 $a_1 = \min(1, \lambda_L\theta(0)) = 1$（因 $\lambda_L \ge 1/\theta(0)$）。
   - **Year 1**（$u_1 = (1-\bar q)\theta(0), a_1 = 1 > a^*$）：全年零新漂绿；第 1 期检查观测到全部存量 $u_1$ 并以概率 $a_1 = 1$ 全部清除，第 2–4 期观测到 0 虚报，故 $s_1 = \frac{1}{m} u_1 = \frac{1-\bar q}{m}\theta(0)$，年末存量 $u_2 = 0$；下年质量 $a_2 = \min\!\big(1, \lambda_L \frac{1-\bar q}{m}\theta(0)\big) \ge a^*$（因 $\lambda_L \ge m a^* / [(1-\bar q)\theta(0)]$）。
   - **Year 2**（$u_2 = 0, a_2 \ge a^*$）：既无旧存量又无新漂绿，故 $s_2 = 0$、$u_3 = 0$，下年质量 $a_3 = \min(1, 0) = 0$，精确回到 $(u_0, a_0) = (0, 0)$。
   三步中仅 Year 0 发生漂绿（占比 $\theta(0)$），Year 1 与 Year 2 均为 0，故周期均值恒为 $\frac{1}{3}\theta(0)$。在默认参数下 $\frac{m a^*}{1-\bar q} = 0.3342 < 1$，故临界条件简化为 $\lambda_L \ge 1/\theta(0)$；对于 $\theta(0) \approx 0.19$，$\lambda_L^{\mathrm{3c}} \approx 5.26$，恰好对应 $100\times$ 责任（$\lambda_L = 5.25$）处的剧烈 3-周期（Study 3 仿真中 $100\times$ 漂绿率 $8.6\%$ 高于 $16\times$ 的 $6.2\%$）。对于中等 $\lambda_L$，取任意足够接近 $a^*$ 的目标质量 $\tilde a \in (a_{\mathrm{ins}}, a^*)$ 使 $\theta(\tilde a) < \frac{1}{3}\theta(0)$ 且令 $\lambda_L^\dagger = \tilde a / \tilde s(\tilde a)$（其中 $\tilde s(\tilde a) \in [\theta(\tilde a), 1]$ 为该质量下的稳态受检占比），则 $(\tilde u, \tilde a)$ 为不动点（或在局部小幅波动），其平均漂绿率严格低于 $\frac{1}{3}\theta(0)$。$\square$

---

## 3. 命题 4：自选鉴证下的客户排序、对称破缺与年度逃逸周期

考虑 $M \ge 2$ 家鉴证方，第 $y$ 年初质量向量为 $a_y = (a_{1,y}, \dots, a_{M,y}) \in [0, 1]^M$，记 $\underline{a}_y = \min_k a_{k,y}$。企业 $i$ 评估单期最优价值 $V_i^*(a) = \max\{V_{I,i}, V_{W,i}(a), 0\}$，并在严格改进时切换至 $k_i^* = \arg\max_k V_i^*(a_k)$（平局取最小索引）。

**命题 4（自选鉴证的聚集与逃逸周期）.** 在强制全员鉴证且允许企业按年自选鉴证方（`ALL_F`）的机制下：

1. **（单调排序与最宽松机构聚集）**
   - 对所有已携带未查出虚报的企业（$\pi_i = 0$），$V_{W,i}(a)$ 在 $[0, 1]$ 上严格递减；因此只要 $\underline{a}_y < a^*$，所有 $\pi_i = 0$ 企业均严格偏好最低质量机构 $\arg\min_k a_{k,y}$。
   - 对所有洁净企业（$\pi_i = 1$），若所有非最宽松机构均满足 $a_{k,y} \ge a_{\mathrm{ins}}$（特别是 $a_{k,y} \ge a^*$）且最宽松机构满足 $a_{k_-, y} = \underline{a}_y < a^*$（或 $\underline{a}_y \ge a_{\mathrm{ins}}$），则所有在 $\underline{a}_y$ 下有意漂绿的洁净企业亦严格偏好最低质量机构 $k_-$；同时所有在 $\underline{a}_y$ 下不漂绿的企业对所有鉴证方无差异（$V_i^*(a_k) \equiv V_{I,i}$），因严格改进切换规则而留在原机构。
   - 推论：当至少存在一家机构 $a_{k,y} \ge a^*$ 且存在唯一最低质量机构 $k_-$ 满足 $a_{k_-,y} < a^*$（且其余机构 $\ge a_{\mathrm{ins}}$）时，**100% 的当期漂绿决策集中于最低质量机构 $k_-$ 的客户群**（对应 Study 3 的 96.9% 与持久化 $Q$ 下的 95.6%）。
2. **（客户加权质量缺口）** 只要存在两家质量不同的机构 $a_{k_1, y} < a^* \le a_{k_2, y}$ 且有正测度潜在漂绿企业从高质机构迁入低质机构，当年的客户加权平均质量 $\bar a_y^{\mathrm{client}} = \sum_k \frac{n_{k,y}}{N} a_{k,y}$ 严格低于等权机构平均质量 $\bar a_y^{\mathrm{unw}} = \frac{1}{M}\sum_k a_{k,y}$（当初始各机构客户数相等且漂绿者仅从高质量迁往低质量时，质量与客户份额呈严格负协方差：$\bar a_y^{\mathrm{client}} - \bar a_y^{\mathrm{unw}} = M \cdot \mathrm{Cov}(n_{k,y}/N,\, a_{k,y}) < 0$）。这直接解释了 Study 3 中 `ALL_F16` 的等权质量（0.131）高于 `ALL_R16`（0.110），但客户实际面对质量仅 0.110 且漂绿率飙升至 21.0% 的现象。
3. **（无非对称稳态与年度逃逸周期）** 设强责任条件 $\lambda_L \cdot \frac{\theta(0)}{1/M + (1-1/M)\theta(0)} \ge a^*$ 成立（即只要全部漂绿企业聚集至单一机构 $k_-$，该机构的污染率便足以将其次年质量推至 $a^*$ 以上）。若 $M \ge 2$ 且在第 $y$ 年出现单宽松机构状态（$a_{k_-, y} < a^*$，而存在其他机构 $k' \neq k_-$ 满足 $a_{k', y} \ge a^*$ 且无残余虚报存量），则次年必有 $a_{k_-, y+1} \ge a^*$ 且 $a_{k', y+1} = 0 < a^*$：**宽松机构身份必然在机构之间跨年轮转，不存在稳定均衡，且每个周期的活跃漂绿决策数永不归零。**

*证明.*
1. 当 $\pi_i = 0$ 时，$V_{W,i}(a) = B_i^2(1-\tau)^2 A_w(a)^2 / (4wc_i)$，因 $A_w(a)$ 关于 $a$ 严格递减（$(1-a)^{n_c(u)}$ 逐项严格递减且 $n_c \ge 1$），故 $V_{W,i}(a)$ 在 $[0, 1]$ 上严格递减，且最优价值 $V_i^*(a) = \max\{V_{I,i}, V_{W,i}(a)\}$ 在 $[0, a^*]$ 上严格递减、在 $[a^*, 1]$ 上恒等于 $V_{I,i}$。当 $\pi_i = 1$ 时，若 $a_{k,y} \ge a^*$ 则 $V_i^*(a_{k,y}) = V_{I,i}$；若企业在 $\underline{a}_y$ 下选择漂绿，则 $V_{W,i}(\underline{a}_y) > V_{I,i} = V_i^*(a_{k,y})$；而在活跃区间 $[a_{\mathrm{ins}}, a^*)$ 上（命题 3(1) 保证 $P^*(a) > P^*(0)$ 且在 $V_{W,i}(a) > V_{I,i}$ 的点处 $V_{W,i}'(a) < 0$），最优价值 $V_i^*(a) = \max\{V_{I,i}, V_{W,i}(a)\}$ 在 $[a_{\mathrm{ins}}, 1]$ 上单调不增并在所有满足 $V_{W,i}(a) > V_{I,i}$ 的区间严格递减，故只要 $\underline{a}_y \notin (0, a_{\mathrm{ins}})$ 或所有其他机构 $\ge a_{\mathrm{ins}}$，即有 $V_i^*(\underline{a}_y) > V_i^*(a_k)$。
2. 设初始各机构各有 $N/M$ 家客户，其中不漂绿企业 $(1-\theta)$ 留守原机构，漂绿企业 $\theta$ 全部迁往最低质量机构 $k_-$，则机构 $k_-$ 客户占比为 $w_{k_-} = \frac{1-\theta}{M} + \theta > \frac{1}{M}$，其余每家机构占比 $w_k = \frac{1-\theta}{M} < \frac{1}{M}$。故
   $$\bar a_y^{\mathrm{client}} - \bar a_y^{\mathrm{unw}} = \theta\Big(a_{k_-,y} - \frac{1}{M}\sum_{k=1}^M a_{k,y}\Big) < 0,$$
   因为 $a_{k_-,y} = \min_k a_{k,y}$ 且各机构质量不完全相等。
3. 接上式，所有 $\theta(a_{k_-,y})$ 漂绿企业迁入 $k_-$ 后，$k_-$ 当年客户总占比为 $\frac{1-\theta}{M} + \theta$，其客户群污染率下界为 $s_{k_-,y} \ge \frac{\theta}{\frac{1-\theta}{M} + \theta}$。在强责任条件下 $\lambda_L s_{k_-,y} \ge a^*$，故 $a_{k_-,y+1} \ge a^*$。同时，上年处于 $a_{k',y} \ge a^*$ 且无旧存量的机构 $k'$ 本年零客户漂绿，$s_{k',y} = 0 \implies a_{k',y+1} = 0 < a^*$。因此次年最宽松机构严格切换为 $k' \neq k_-$，漂绿企业集体从 $k_-$ 逃逸至 $k'$，形成永不停止的年度逃逸周期。$\square$

---

## 4. 命题 5：预算约束核查的凹性失效与 Stackelberg 背包最优结构

固定市场状态 $(B_i, P_i, c_i, \pi_i)_{i=1}^N$，考察监管者在固定期望核查预算 $\sum_{i=1}^N q_i \le K = N\bar q$（$q_i \in [0, 1]$）下的公开承诺核查博弈（Stackelberg Audit Game）。企业 $i$ 观测到自身核查率 $q_i$ 后，当且仅当净漂绿盈余
$$\Delta V_i(q_i) \equiv V_{W,i}(q_i) - V_{I,i} = \frac{B_i^2}{4 c_i}\left[\frac{(1-\tau)^2 A_w^{\text{aud}}(q_i)^2}{w} - A_r^2\right] - \pi_i \frac{F}{P_i} D^{\text{aud}}(q_i) \le 0$$
时放弃漂绿并选择真实改善。

**命题 5（核查凹性与 Stackelberg 背包结构）.** 设 $p = 0, \sigma = 0, \tau < \tau^*(0)$。

1. **（$D^{\text{aud}}(q)$ 的严格凹性与唯一企业威慑门槛 $q_i^{\mathrm{det}}$）** 对所有 $q \in [0, 1]$：
   $$\frac{d D^{\text{aud}}}{d q} = \frac{\beta_1 (1-\beta)}{[(1-\beta) + \beta q]^2} > 0, \qquad \frac{d^2 D^{\text{aud}}}{d q^2} = -\frac{2\beta_1 \beta (1-\beta)}{[(1-\beta) + \beta q]^3} < 0.$$
   连同 $A_w^{\text{aud}}(q)$ 的严格递减性，$\Delta V_i(q)$ 在 $[0, 1]$ 上严格递减，故每家内点企业存在唯一的最小威慑核查概率 $q_i^{\mathrm{det}} \in [0, q^*]$ 使得企业 $i$ 放弃漂绿当且仅当 $q_i \ge q_i^{\mathrm{det}}$。其中：
   - 若 $\pi_i = 0$ 或 $F = 0$，则 $q_i^{\mathrm{det}} = q^*$（默认参数下 $q^* = 0.104924$）；
   - 若 $\pi_i = 1$ 且 $F > 0$，则 $q_i^{\mathrm{det}} \in [0, q^*)$ 关于市值 $P_i$ 严格递增，满足 $\lim_{P_i\to 0} q_i^{\mathrm{det}} = 0$ 且 $\lim_{P_i\to\infty} q_i^{\mathrm{det}} = q^*$。
2. **（按市值定向核查 V 的双渠道威慑失效）** 任取一对洁净企业 $(i, j)$ 满足 $P_i < P_j$。将核查概率 $\Delta q > 0$ 从小企业 $i$ 转移至大企业 $j$（如从均匀随机 $\bar q$ 转为按市值分配 $V$）时：
   - **边际威慑双衰减**：因同时有 $F/P_i > F/P_j$ 与 $(D^{\text{aud}})'(q_i) > (D^{\text{aud}})'(q_j)$（严格凹性），小企业损失的期望定额罚威慑 $\frac{F}{P_i}(D^{\text{aud}})'(q_i)\Delta q$ 严格大于大企业获得的威慑增量 $\frac{F}{P_j}(D^{\text{aud}})'(q_j)\Delta q$（在默认参数下，$(D^{\text{aud}})'(0) \approx 83.5$ 是 $(D^{\text{aud}})'(0.08) \approx 1.42$ 的 59 倍）；
   - **离散跨越损失**：只要 $q_i^V < q_i^{\mathrm{det}} \le \bar q$ 且 $q_j^V < q_j^{\mathrm{det}}$（在 Study 2 中，小企业核查率由 $1\%$ 跌至 $0.41\%$，而最大企业仅升至 $8.0\% < q^* = 10.49\%$），按市值分配严格降低该企业对的合规企业数（由 1 降至 0）。
3. **（单期 Stackelberg 最优核查的 0–1 背包刻画）** 设监管者最大化加权合规目标 $\sum_{i=1}^N \omega_i\, \mathbf{1}_{\{q_i \ge q_i^{\mathrm{det}}\}}$（$\omega_i > 0$；例如等权合规 $\omega_i \equiv 1$，等权真实改善收益 $\omega_i = \Delta_{I,i}$，或市值加权收益 $\omega_i = P_i \Delta_{I,i}$），约束为 $\sum_{i=1}^N q_i \le K$，$q_i \in [0, 1]$。则：
   - 任何最优解均可取“门槛精确匹配”形式：$q_i^* \in \{0, q_i^{\mathrm{det}}\}$（多给超过 $q_i^{\mathrm{det}}$ 的概率不增加当期威慑，少于 $q_i^{\mathrm{det}}$ 的概率完全浪费当期威慑），原问题精确等价于 **0–1 背包问题**：
     $$\max_{z \in \{0,1\}^N}\sum_{i=1}^N \omega_i z_i \quad \text{s.t.} \quad \sum_{i=1}^N q_i^{\mathrm{det}} z_i \le K.$$
   - 在**等权合规目标**（$\omega_i \equiv 1$）下，将所有尚未自动威慑（$q_i^{\mathrm{det}} > 0$）的企业按 $q_i^{\mathrm{det}}$ **升序排列**并依次分配 $q_i^* = q_i^{\mathrm{det}}$ 直至预算耗尽，在 $O(N\log N)$ 时间内给出**全局精确最优解**。该结构与按市值分配 $V$（给 $q_i^{\mathrm{det}}$ 最高的大企业最多核查、给 $q_i^{\mathrm{det}}$ 最低的小企业最少核查）**方向完全相反**。
   - 在一般正权重 $\omega_i$（如市值加权 $\omega_i = P_i \Delta_{I,i}$）下，按性价比 $\omega_i / q_i^{\mathrm{det}}$ 降序贪心分配在 $O(N\log N)$ 时间内给出连续松弛最优解（至多相差一个边界企业）；当超大企业 $P_i$ 极大而 $q_i^{\mathrm{det}} \le q^*$ 有界时，它把大企业直接推过 $q_i^{\mathrm{det}}$（彻底止漂），而不是像 $V$ 那样停留在 $q_i^V < q^*$ 的无效区间。

*证明.*
1. $D^{\text{aud}}(q) = \beta_1 q / [(1-\beta) + \beta q]$，直接求导即得一阶导为正、二阶导为负（$\beta \in (0, 1)$）。由于 $A_w^{\text{aud}}(q)$ 严格递减且 $A_w^{\text{aud}}(q^*) = \frac{\sqrt{w}}{1-\tau} A_r$，故 $\Delta V_i(q)$ 严格递减且 $\Delta V_i(q^*) = -\pi_i \frac{F}{P_i} D^{\text{aud}}(q^*) \le 0$。若 $\Delta V_i(0) \le 0$，则 $q_i^{\mathrm{det}} = 0$；若 $\Delta V_i(0) > 0$，由介值定理与严格单调性，存在唯一根 $q_i^{\mathrm{det}} \in (0, q^*]$。当 $\pi_i = 0$ 或 $F = 0$ 时 $\Delta V_i(q^*) = 0$，故 $q_i^{\mathrm{det}} = q^*$。当 $\pi_i = 1, F > 0$ 时，由隐函数定理 $\frac{\partial q_i^{\mathrm{det}}}{\partial P_i} = -\frac{\partial \Delta V_i / \partial P_i}{\partial \Delta V_i / \partial q} = \frac{F D^{\text{aud}}(q_i^{\mathrm{det}}) / P_i^2}{-\partial \Delta V_i / \partial q} > 0$；且 $P_i \to 0$ 时 $F/P_i \to \infty$ 使 $\Delta V_i(q) < 0$ 对任意固定 $q > 0$ 成立，故 $q_i^{\mathrm{det}} \to 0$；$P_i \to \infty$ 时 $F/P_i \to 0$，故 $q_i^{\mathrm{det}} \to q^*$。
2. 直接由 (1) 的严格凹性与 $P_i < P_j \implies F/P_i > F/P_j$ 得出。
3. 令 $z_i = \mathbf{1}_{\{q_i \ge q_i^{\mathrm{det}}\}} \in \{0, 1\}$。给定任意可行分配 $q$，将其替换为 $\tilde q_i = z_i q_i^{\mathrm{det}}$ 满足 $\tilde q_i \le q_i$，故仍满足预算约束 $\sum_i \tilde q_i \le \sum_i q_i \le K$ 且获得完全相同的目标值 $\sum_i \omega_i z_i$。因此最优分配必归约为 0–1 背包问题。当 $\omega_i \equiv 1$ 时，任意可行子集 $S$ 若未选最小代价元素，可用未选最小代价元素替换 $S$ 中最大代价元素而不违反预算且不减基数（标准交换论证），故按 $q_i^{\mathrm{det}}$ 升序贪心精确最优。一般权重下的连续松弛最优性由 Dantzig 贪心定理直接给出。$\square$

---

## 5. 适用范围与仿真衔接说明

1. **命题 3 与命题 4 属于连续均值场/内点基准**：完整多资产市场中存在有限样本离散随机性（$N=200, 4$ 家鉴证方，每机构 50 家企业）、价格与融资收益 $B_i$ 的内生波动以及报告满分上限 $C$ 的截断；这些有限样本噪声会使确定性 3-周期略微失相（从而在 30 种子平均图上呈现平滑振荡并出现 28/30 种子穿越 $a^*$），但不改变无零漂绿不动点、罚款保险区间、逃逸周期聚集与高责任非单调性的机制结构。
2. **命题 5 与多期动态市场的衔接**：单期背包分配仅考虑当期威慑，在多期市场中，未被当期威慑的企业（尤其携带未结案存量的大企业）仍可通过剩余预算 $K - \sum_{i \in S^*} q_i^{\mathrm{det}}$ 进行抽查清理（一旦查结旧案，$\pi_i$ 从 0 恢复为 1，次期 $q_i^{\mathrm{det}}$ 从 $q^*$ 大幅回落，产生动态滚动威慑红利）。Track 3 的多期 Stackelberg 核查算法即直接实现这一“背包威慑 + 剩余预算清存量”机制。
