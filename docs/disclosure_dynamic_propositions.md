# 多智能体动态与博弈命题：鉴证共演化、Stackelberg 核查与运行时滞回修复（Propositions 3–6）

> **状态：形式化推导与数值核验。** 本文档承接单企业静态基准 [disclosure_inspection_propositions.md](disclosure_inspection_propositions.md)（Propositions 1–2），为 Study 2（定向与贝叶斯动态核查）及 Study 3（第三方鉴证共演化、部分可观测性与制度修复）建立领域无关的规范多智能体（Normative MAS）理论框架。全部解析阈值、分岔条件、凹包络结构与轨道不等式均在 [`tests/test_disclosure_dynamic_propositions.py`](../tests/test_disclosure_dynamic_propositions.py) 中与 [`src/abm/disclosure/financing.py`](../src/abm/disclosure/financing.py)、[`src/abm/disclosure/bayesian_audit.py`](../src/abm/disclosure/bayesian_audit.py) 逐项数值核验。

---

## 1. 记号与公共前提

沿用 Propositions 1–2 的基准设定（$\sigma = 0, p = 0$，日度报告、未公开核查结果、无均值回归或给定持续性年金 $A_r > 0$）：

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

## 2. 命题 3：指派鉴证下的罚款保险、分岔失稳与高责任 3-周期非单调性

考虑随机指派给单一鉴证方 $k$ 的连续测度异质客户群，每家内点企业在质量 $a < a^*$ 下的洁净开案规模门槛为：
$$P^*(a) \equiv \frac{4 c F\, D(a)}{B^2\big[(1-\tau)^2 A_w(a)^2 / w - A_r^2\big]} \quad (a < a^*),$$
记当质量为 $a$ 时满足 $P_i > P^*(a)$（无未结案时亦主动发起漂绿）的客户占比为 $\theta(a) \in [0,1]$，且对所有 $a \ge a^*$ 有 $\theta(a) = 0$。单期虚报存续率为 $\rho(a) \equiv (1-a)(1-\bar q)$。短视鉴证方每年初更新质量：$a_{k,y+1} = \min(1, \lambda_L s_{k,y})$，其中 $\lambda_L = L\bar q / (2\kappa)$。

**命题 3（指派鉴证动力学）.** 设 $\tau < \tau^*(0)$（从而 $a^* > 0$）且 $\theta(0) > 0$。则：

1. **（弱责任罚款保险区间）** 当且仅当
   $$\left|\frac{D'(0)}{D(0)}\right| > \left|\frac{2 A_w'(0)}{A_w(0)}\right| \cdot \frac{(1-\tau)^2 A_w(0)^2 / w}{(1-\tau)^2 A_w(0)^2 / w - A_r^2}$$
   时，存在非空开区间 $(0, a_{\mathrm{ins}}) \subset (0, a^*)$，使得对任意 $a \in (0, a_{\mathrm{ins}})$ 均有 $P^*(a) < P^*(0)$。在默认参数下，左端 $45.84 >$ 右端 $25.51$，精确罚款保险区间为 $(0, 0.0346)$，门槛最低点在 $a_{\min} = 0.0156$ 处达到 $P^*(a_{\min}) / P^*(0) = 0.8728$。若 $\lambda_L < a^*$（如 $1\times$ 责任 $\lambda_L = 0.0525 < 0.0827$），恒有 $a_{k,y} \le \lambda_L < a^*$，永远无法触及止漂门槛。
2. **（零漂绿稳态的不存在性与内生回落）** 对任意有限 $\lambda_L \in (0, \infty)$，耦合映射 $(u_y, a_y) \mapsto (u_{y+1}, a_{y+1})$ **不存在**满足 $a_{\mathrm{eq}} \ge a^*$ 的不动点。任意进入止漂年（$a_y \ge a^*$）的轨迹满足几何收缩
   $$u_{y+1} = \rho(a_y)^m u_y, \qquad s_y = \Phi_m(a_y)\, u_y, \qquad \Phi_m(a) \equiv \frac{1 - \rho(a)^m}{m(1-\rho(a))},$$
   因此至多经过有限个连续止漂年后必出现 $\lambda_L s_y < a^* \implies a_{y+1} < a^*$，触发复发。
3. **（内点不动点倍周期分岔、高责任记忆擦除 3-周期与严格非单调性）**
   - **(a) 内点平衡与倍周期失稳条件**：若 $\theta(a)$ 在 $[a_{\mathrm{ins}}, a^*]$ 上连续可微、严格递减且 $\theta(a^*) = 0$，则对任意目标质量 $\tilde a \in (a_{\mathrm{ins}}, a^*)$，令 $\lambda_L(\tilde a) = \tilde a / \theta(\tilde a)$ 存在内点平衡态（高倾向子群 $\theta(\tilde a)$ 持续漂绿，其余客户持续洁净，稳态受检污染率 $s_{\mathrm{stat}}(\tilde a) = \theta(\tilde a)$）。然而在 $a < a^*$ 区域，一维简化映射 $a \mapsto \lambda_L \theta(a)$ 在 $\tilde a$ 处的导数为 $\lambda_L \theta'(\tilde a) = \tilde a \theta'(\tilde a) / \theta(\tilde a) \equiv \varepsilon_\theta(\tilde a) < 0$；当且仅当弹性 $|\varepsilon_\theta(\tilde a)| > 1$ 时该内点平衡经**倍周期（Flip）分岔失稳**并产生自激振荡（在校准对数正态分布下，所有 $\tilde a > 0.040$ 均满足 $|\varepsilon_\theta(\tilde a)| > 1$，直接解释了 Study 3 中 28/30 种子自发穿越 $a^*$ 循环的现象）。
   - **(b) 高责任完全记忆擦除精确 3-周期**：若责任斜率满足
     $$\lambda_L \ge \lambda_L^{\mathrm{3c}} \equiv \frac{\max\big(1,\, m\,a^*/(1-\bar q)\big)}{\theta(0)},$$
     则从 $(u_0, a_0) = (0, 0)$ 起，该映射进入**精确 3-周期轨道**：
     $$\big((u_0, a_0),\; (u_1, a_1),\; (u_2, a_2)\big) = \left((0, 0),\; \big((1-\bar q)\theta(0),\, 1\big),\; \Big(0,\, \min\!\Big(1,\, \frac{\lambda_L(1-\bar q)\theta(0)}{m}\Big)\Big)\right).$$
     因 Year 1 过冲至 $a_1 = 1$ 将未结案存量在首期**完全清空（$u_2 = 0$，记忆擦除）**，Year 2 观测 $s_2 = 0$，迫使 Year 3 彻底跌回 $a_3 = 0$，承受最大复发幅度 $\theta(0)$；长期平均漂绿率恒为 $\frac{1}{3}\theta(0) > 0$（即使 $\lambda_L \to \infty$ 亦不下降）。
   - **(c) 中等责任不完全清空防崩塌与全局轨道非单调性**：设 $\theta(a)$ 在 $[a_{\mathrm{ins}}, a^*]$ 上连续递减且 $\theta(a^*) = 0$。若存在中等斜率 $\lambda_L^\dagger \in (a^*/\theta(0),\, 1/\theta(0))$ 与阈值 $\underline a \in (a_{\mathrm{ins}}, a^*)$，使得从任意处于紧陷阱区 $a_y \in [\underline a, a^*), u_y \le \theta(\underline a)$ 的复发年出发，均有：(i) 次年过冲不触顶，$a_{y+1} = \lambda_L^\dagger s_y \in [a^*, 1)$；(ii) 留存残余信号支撑连续两止漂年，$a_{y+2} = \lambda_L^\dagger \Phi_m(a_{y+1}) u_{y+1} \ge a^*$；(iii) 第三年回落质量保持严格正值，$a_{y+3} = \lambda_L^\dagger \Phi_m(a_{y+2})\rho(a_{y+1})^m u_{y+1} \in [\underline a, a^*)$。则该 3 步返回映射将复发质量永久锁定在 $[\underline a, a^*)$ 内（永不跌回 $a = 0$），每个 3 年周期内至多 1 年复发且复发率 $\le \theta(\underline a) < \theta(0)$，因此长期平均漂绿率严格低于 $\frac{1}{3}\theta(\underline a) < \frac{1}{3}\theta(0)$。由此**不依赖内点不动点稳定性**地严格证明：在年度短视更新下，长期漂绿率对鉴证方责任 $L$ **全局严格非单调**。

*证明.*
1. 由 $P^*(a) \propto D(a)/[(1-\tau)^2 A_w(a)^2/w - A_r^2]$ 对 $a$ 取对数求导，在 $a = 0$ 处有 $\frac{d\ln P^*}{da}\big|_{0} = \frac{D'(0)}{D(0)} - \frac{2(1-\tau)^2 A_w(0)A_w'(0)/w}{(1-\tau)^2 A_w(0)^2/w - A_r^2} < 0$，由连续性得 $(0, a_{\mathrm{ins}})$；又因 $s_{k,y} \in [0, 1]$，故 $a_{k,y+1} \le \lambda_L$。
2. 反证：若 $(u_{\mathrm{eq}}, a_{\mathrm{eq}})$ 为不动点且 $a_{\mathrm{eq}} \ge a^*$，则 $u_{\mathrm{eq}} = \rho(a_{\mathrm{eq}})^m u_{\mathrm{eq}}$，因 $\rho(a^*) < 1$ 必有 $u_{\mathrm{eq}} = 0 \implies s_{\mathrm{eq}} = 0 \implies a_{\mathrm{eq}} = 0 < a^*$，矛盾。有限时间回落由 $\rho(a^*)^{mt} \to 0$ 直接得出。
3. (a) 在 $a < a^*$ 下，满足 $P_i > P^*(a)$ 的企业每期皆主动发起/延续漂绿（即便被查处亦于次期立即重开），故其受检污染概率恒为 1，其余洁净企业恒为 0，得 $s_{\mathrm{stat}}(a) = \theta(a)$；Jacobian 特征值即 $\frac{d}{da}[\lambda_L\theta(a)]\big|_{\tilde a} = \varepsilon_\theta(\tilde a)$，故 $|\varepsilon_\theta(\tilde a)| > 1$ 时按标准一维分岔定理失去局部渐近稳定性。(b) 对 $\lambda_L \ge \lambda_L^{\mathrm{3c}}$ 从 $(0, 0)$ 显式迭代三步：$a_0=0 \implies s_0=\theta(0), u_1=(1-\bar q)\theta(0), a_1=\min(1, \lambda_L\theta(0))=1$；$a_1=1 \implies \rho(1)=0, s_1=\frac{u_1}{m}, u_2=0, a_2 \ge a^*$；$u_2=0, a_2\ge a^* \implies s_2=0, u_3=0, a_3=0$，三步均值恒为 $\frac{1}{3}\theta(0)$。(c) 设三步陷阱条件在 $[\underline a, a^*)$ 上成立（在校准异质分布中，于 $25\times$ 责任 $\lambda_L^\dagger = 1.313$ 下，自 $(0,0)$ 仅 1 个周期即进入 $a_{3k} \in [0.055, 0.058] > a_{\mathrm{ins}} = 0.0346$，复发率 $\approx 0.094 < 0.19$，长期均值 $3.01\% < 6.33\%$）：每 3 步中至多第 $3k$ 年满足 $a_{3k} < a^*$，且该年所有漂绿决策受 $a_{3k} \ge \underline a$ 约束（新开案比例 $\theta(a_{3k}) \le \theta(\underline a)$，旧存量 $u_{3k}$ 经两止漂年几何衰减满足 $u_{3k} \ll \theta(\underline a)$），第 $3k+1, 3k+2$ 年质量均 $\ge a^*$ 故零漂绿，三步均值严格低于 $\frac{1}{3}\theta(0)$。$\square$

---

## 3. 命题 4：自选鉴证下的客户排序、协方差质量缺口与年度逃逸周期

考虑 $M \ge 2$ 家鉴证方，第 $y$ 年初质量向量 $a_y = (a_{1,y}, \dots, a_{M,y}) \in [0, 1]^M$，记 $\underline a_y = \min_k a_{k,y}$。企业 $i$ 评估单期最优价值 $V_i^*(a) = \max\{V_{I,i}, V_{W,i}(a), 0\}$，在严格改进时切换至 $k_i^* = \arg\max_k V_i^*(a_k)$。

**命题 4（自选鉴证的聚集、协方差缺口与逃逸周期）.** 在强制全员鉴证且允许按年自选（`ALL_F`）的机制下：

1. **（单调排序与最宽松机构聚集）** 对所有未结案企业（$\pi_i = 0$），$V_{W,i}(a)$ 在 $[0, 1]$ 上严格递减，只要 $\underline a_y < a^*$ 即严格偏好 $\arg\min_k a_{k,y}$；对洁净企业（$\pi_i = 1$），只要非最宽松机构满足 $a_{k,y} \ge a_{\mathrm{ins}}$ 且在 $\underline a_y$ 下有意漂绿，亦严格偏好最宽松机构 $k_-$，而不漂绿企业无差异留守原机构。当存在唯一最宽松机构 $a_{k_-,y} < a^*$ 且其余机构 $\ge a^*$ 时，**100% 当期漂绿决策集中于 $k_-$**。
2. **（客户加权质量协方差缺口）** 记机构 $k$ 的客户份额为 $w_{k,y} = n_{k,y}/N$。客户实际面对的加权质量与机构等权均值满足精确协方差恒等式：
   $$\bar a_y^{\mathrm{client}} - \bar a_y^{\mathrm{unw}} = \sum_{k=1}^M \left(w_{k,y} - \frac{1}{M}\right) a_{k,y} = M \cdot \mathrm{Cov}(w_{k,y},\, a_{k,y}).$$
   每当漂绿企业从高质量机构迁往低质量机构导致质量与客户份额呈负相关时，严格有 $\bar a_y^{\mathrm{client}} < \bar a_y^{\mathrm{unw}}$（平局安全，直接作为 [`metrics.py`](../src/abm/disclosure/metrics.py) 中的 `client_minus_unweighted_quality_gap` 检验指标）。
3. **（无非对称稳态与年度逃逸周期）** 在强责任条件 $\lambda_L \cdot \frac{\theta(0)}{1/M + (1-1/M)\theta(0)} \ge a^*$ 与短视更新（$\gamma = 0$）下，任意单宽松机构状态（$a_{k_-,y} < a^*$，其余 $k' \neq k_-$ 满足 $a_{k',y} \ge a^*$ 且无残余存量）在次年必翻转至 $a_{k_-,y+1} \ge a^*$ 且 $a_{k',y+1} = 0 < a^*$：**最宽松机构跨年轮转，不存在稳定均衡，且每期活跃漂绿永不归零。**

*证明.* 与单调性、负协方差展开及次年 $s_{k_-,y} \ge \frac{\theta}{(1-\theta)/M + \theta} \implies a_{k_-,y+1} \ge a^*$、$s_{k',y} = 0 \implies a_{k',y+1} = 0$ 一致。$\square$

---

## 4. 命题 5：核查双凹反序定理与多期两状态 RMAB-Stackelberg 动态索引

固定市场状态 $(B_i, P_i, c_i, \pi_i)_{i=1}^N$ 与固定期望核查预算 $\sum_{i=1}^N q_i \le K = N\bar q$（$q_i \in [0, 1]$）。在内点情形（未触及单期幅度上限 $M$，且 $p=0, \sigma=0$）下，企业 $i$ 的净漂绿盈余为
$$\Delta V_i(q_i) = \frac{B_i^2}{4 c_i}\left[\frac{(1-\tau)^2 A_w^{\text{aud}}(q_i)^2}{w} - A_r^2\right] - \pi_i \frac{F}{P_i} D^{\text{aud}}(q_i);$$
当存在幅度上限 $M < \infty$ 时，按截断步长 $d_I = \min(\frac{B_i A_r}{2c_i}, M), d_W(q) = \min(\frac{B_i(1-\tau)A_w^{\text{aud}}(q)}{2w c_i}, M)$ 定义的 $\Delta V_i^{(M)}(q)$ 仍关于 $q$ 严格递减（因包络定理 $\frac{d V_W^{(M)}}{dq} = B_i(1-\tau)(A_w^{\text{aud}})'(q) d_W(q) - \pi_i\frac{F}{P_i}(D^{\text{aud}})'(q) < 0$）。

**命题 5（双凹反序定理与多期 RMAB-Stackelberg 凹包络分配）.** 设 $p = 0, \sigma = 0, \tau < \tau^*(0)$。

1. **（$D^{\text{aud}}(q)$ 严格凹性与唯一威慑门槛 $q_i^{\mathrm{det}}$）** 对所有 $q \in [0, 1]$，$(D^{\text{aud}})'(q) = \frac{\beta_1(1-\beta)}{[(1-\beta)+\beta q]^2} > 0$ 且 $(D^{\text{aud}})''(q) = -\frac{2\beta_1\beta(1-\beta)}{[(1-\beta)+\beta q]^3} < 0$（默认参数下 $(D^{\text{aud}})'(0) / (D^{\text{aud}})'(0.08) \approx 59$）。每家企业存在唯一威慑门槛 $q_i^{\mathrm{det}} \in [0, q^*]$：当 $\pi_i = 0$ 或 $F = 0$ 时 $q_i^{\mathrm{det}} = q^*$；当 $\pi_i = 1, F > 0$ 时 $q_i^{\mathrm{det}}$ 随 $P_i$ 严格递增且满足 $\lim_{P_i\to 0} q_i^{\mathrm{det}} = 0$、$\lim_{P_i\to\infty} q_i^{\mathrm{det}} = q^*$。
2. **（双凹反序定理：按市值启发式 `V` 与最优序严格反序）**
   - 对任意一对洁净企业 $P_i < P_j$，将预算 $\Delta q$ 从小企业 $i$ 移至大企业 $j$ 时，因 $F/P_i > F/P_j$ 与 $(D^{\text{aud}})'(q_i) > (D^{\text{aud}})'(q_j)$ 同向叠加，引发边际威慑双衰减与离散跨越损失。
   - 在单期加权合规目标 $\max \sum_{i=1}^N \omega_i \mathbf{1}_{\{q_i \ge q_i^{\mathrm{det}}\}}$（$\sum q_i \le K$）下，最优分配精确归约为门槛匹配 0–1 背包问题；当 $\omega_i \equiv 1$ 时，全局精确最优解按 $q_i^{\mathrm{det}}$ **升序**分配（优先锁定洁净小/中企业），与按市值降序分配的启发式 `V` **方向完全相反**。
3. **（多期部分可观测 RMAB-Stackelberg Bellman 凹包络分配）** 将每家企业 $i$ 的持久成本类型 $\varepsilon_k$ 与未结案二元状态 $x_{i,t} \in \{0, >0\}$ 建模为弱耦合多臂不稳定赌博机（RMAB），其联合后验为 $(w^{\mathrm{clean}}_{i,k}, w^{\mathrm{dirty}}_{i,k})$。记清理一个未结案旧案所释放的折现跨期预算红利（以边际影子价格 $\mu = 1/q^*$ 计价）为
   $$\Delta J_{i,k} = \frac{\beta(1 - \beta^{H_p})}{1 - \beta}\left(1 - \frac{q^{\mathrm{det,clean}}_{i,k}}{q^*}\right)_+.$$
   则公开承诺下两状态 Bellman 动作价值函数为
   $$Q^{\mathrm{clean}}_{i,k}(q) = d^{\mathrm{clean}}_{i,k}(q)\big(1 + \beta\Delta J_{i,k}\big) + \big(1 - d^{\mathrm{clean}}_{i,k}(q)\big)\,q\,\beta\Delta J_{i,k}, \qquad Q^{\mathrm{dirty}}_{i,k}(q) = d^{\mathrm{dirty}}_{i,k}(q) + q\,\beta\Delta J_{i,k},$$
   期望收益 $U_i(q) = \omega_i \sum_k [w^{\mathrm{clean}}_{i,k} Q^{\mathrm{clean}}_{i,k}(q) + w^{\mathrm{dirty}}_{i,k} Q^{\mathrm{dirty}}_{i,k}(q)]$。对其上凹包络执行全局斜率降序水填（$O(NG\log(NG))$，实现于 [`BayesianAuditFilter.allocate`](../src/abm/disclosure/bayesian_audit.py)），给出拉格朗日松弛精确最优解，并自动涌现三层优先级结构：
   - **第一优先（保护洁净小/中企业）**：洁净企业斜率 $\sim (1+\beta\Delta J)/q^{\mathrm{clean}} \gg 1/q^* + \beta\Delta J$，彻底杜绝 `V` 的跨档位移；
   - **第二优先（在 $q^*$ 处兼顾止漂与清旧案）**：对高概率未结案企业，在 $[0, q^*]$ 上斜率为 $p^{\mathrm{dirty}}_i(1/q^* + \beta\Delta J)$，严格高于 $[q^*, 1]$ 上的纯清理斜率 $p^{\mathrm{dirty}}_i \beta\Delta J$——与“将预算集中于少数企业给 $q=1$”相比，以等量期望清案速率同时多获得 $1/q^* \approx 9.5\times$ 当期止漂红利；
   - **第三优先（溢出清案）**：仅当洁净威慑与 $q^*$ 档全部饱和后，剩余预算沿 $[q^*, 1]$ 加速清案。
4. **（无混杂隐藏规则最优解与命中率悖论）** 当规则隐藏（`audit_rule_public=False`）时，企业按均匀 $\bar q$ 规划，故 $q_i$ 无当期威慑作用；此时贝叶斯最优隐藏规则退化为线性规划 $\max \sum_i \omega_i q_i \sum_k w^{\mathrm{dirty}}_{i,k}(1 + \Delta J_{i,k})$，将全部预算集中投向最可能带案的 $\lfloor K \rfloor$ 家企业（$q_i = 1$）。尽管该隐藏最优解可将核查命中率推高至 $92\%$–$98\%$，但其当期不阻止任何洁净企业开案，因而公开承诺价值 `S_obs − S_obs_hid` 严格为正。

*证明.* (1)–(2) 由包络定理、二阶导数及交换论证直接给出。(3) 由 $Q^{\mathrm{dirty}}_{i,k}(q)$ 在 $q^*$ 处的阶跃增量 $+1$ 得 $[0, q^*]$ 割线斜率严格大于 $[q^*, 1]$ 斜率 $1/q^*$；弱耦合约束 $\sum_i q_i \le K$ 的拉格朗日对偶将多企业问题分解为单企业 $\max_{q_i} [U_i(q_i) - \lambda q_i]$，最优解必落在各 $U_i$ 上凹包络的顶点及单一边界切线段上，由斜率降序水填精确求解。(4) 在 $q_i$ 不可观测时，$d^{\mathrm{clean}}, d^{\mathrm{dirty}}$ 与 $q_i$ 无关，目标函数关于 $q_i$ 严格线性，极点解即按斜率降序填满 $q_i = 1$。$\square$

---

## 5. 命题 6：部分可观测监控盲区与运行时滞回（Hysteresis）修复定理

考察 Study 3 中鉴证方在部分可观测信息结构（`assurer_observation="detected_bayes"`）下仅观测自身发布前更正数 $C_{k,y}$ 与监管事后追责罚单数 $E_{k,y}$ 的经验贝叶斯更新：
$$\hat s_{k,y+1} = \mathrm{clip}_{[0,1]}\!\left(\frac{\nu\,\hat s_{k,y} + C_{k,y} + E_{k,y}}{\nu + n_{k,y}\, p_{\mathrm{obs}}(a_{k,y})}\right), \qquad p_{\mathrm{obs}}(a) \equiv a + (1-a)\bar q,$$
以及带运行时审查滞回平滑 $\gamma \in [0, 1)$ 与法定最低质量下限 $a_{\min} \in [0, 1]$ 的质量更新规则（实现于 [`PolicyRegime`](../src/abm/disclosure/policy.py) 与 [`DisclosureMarket._assurance_year`](../src/abm/disclosure/game.py)）：
$$a_{k,y+1} = \max\!\Big(a_{\min},\; \gamma\, a_{k,y} + (1 - \gamma)\min(1,\, \lambda_L \hat s_{k,y+1})\Big).$$

**命题 6（部分可观测监控盲区与滞回修复定理）.** 设每家鉴证方当年检查次数为 $n_k > 0$，先验强度 $\nu > 0$，$\bar q \in (0, 1)$。则：

1. **（无偏暴露缩放与状态依赖惯性权重）** 在真实客户污染率 $s_{k,y}$ 给定下，$\mathbb{E}[C_{k,y} + E_{k,y} \mid s_{k,y}] = n_k p_{\mathrm{obs}}(a_{k,y}) s_{k,y}$。记最大似然估计 $\hat s^{\mathrm{ML}}_{k,y} = \frac{C_{k,y} + E_{k,y}}{n_k p_{\mathrm{obs}}(a_{k,y})}$，则贝叶斯更新精确等价于线性收缩：
   $$\hat s_{k,y+1} = \gamma_{\mathrm{eff}}(a_{k,y})\,\hat s_{k,y} + \big(1 - \gamma_{\mathrm{eff}}(a_{k,y})\big)\,\hat s^{\mathrm{ML}}_{k,y}, \qquad \gamma_{\mathrm{eff}}(a) \equiv \frac{\nu}{\nu + n_k [a + (1-a)\bar q]}.$$
2. **（逆向非对称惯性 / 监控盲区效应）** $\gamma_{\mathrm{eff}}(a)$ 关于 $a \in [0, 1]$ **严格递减**：
   $$\gamma_{\mathrm{eff}}'(a) = -\frac{\nu\, n_k (1 - \bar q)}{\big(\nu + n_k[a + (1-a)\bar q]\big)^2} < 0, \qquad \frac{\gamma_{\mathrm{eff}}(0)}{\gamma_{\mathrm{eff}}(1)} = \frac{\nu + n_k}{\nu + n_k \bar q} > 1.$$
   在基准参数（$n_k = 200, \bar q = 0.01, \nu = 2$）下，$\gamma_{\mathrm{eff}}(0) = 0.500$ 而 $\gamma_{\mathrm{eff}}(1) = 0.0099$，形成 **$50.5\times$ 逆向不对称**：鉴证方在松懈态（$a_k \approx 0$）陷入观测盲区、极慢察觉漂绿抬头，却在严审态（$a_k \approx 1$）几乎零惯性、一旦清空存量便迅速遗忘回落。
3. **（运行时审查滞回 $\gamma$ 的多周期止漂下界与双病理同治）** 在 `assurer_smoothing` $= \gamma \in (0, 1)$ 下：
   - **打破高责任 3-周期并恢复单调性**：每当某机构质量达到峰值 $a_{\mathrm{peak}} > a^*$，即使此后零违规（$\hat s = 0$），其质量按 $a_{y+t} \ge \gamma^t a_{\mathrm{peak}}$ 衰减，必然维持至少
     $$T(\gamma; a_{\mathrm{peak}}) = \left\lfloor \frac{\ln(a_{\mathrm{peak}} / a^*)}{\ln(1/\gamma)} \right\rfloor + 1$$
     个连续零漂绿止漂年（默认 $a^* = 0.0827, \gamma = 0.7$ 且高责任达到 $a_{\mathrm{peak}} \ge 0.30$ 时，$T \ge 4$；在 $a_{\mathrm{peak}} \to 1$ 时 $T(0.7) = 7$，将渐近复发频率由 $1/3$ 压缩至 $\le 1/8$，恢复随责任 $L$ 增大的单调威慑改善）。
   - **封堵自选鉴证逃逸口**：在 $M$ 家机构竞争的自选机制下，当 $M-1$ 家严审机构维持 $a_{k',y} \ge a_{\mathrm{peak}}$ 而最宽松机构 $k_-$ 吸收漂绿客户并在次年跃升至 $a_{k_-,y+1} = (1-\gamma)\min(1, \lambda_L s_{k_-,y}) \ge a^*$ 时，只要 $\gamma a_{\mathrm{peak}} \ge a^*$，次年**全部 $M$ 家机构同时满足 $\min_k a_{k,y+1} \ge a^*$**——逃逸口被同步封闭，打破命题 4(3) 的逐年逃逸周期。
4. **（法定最低质量下限 $a_{\min}$ 的相变临界性）** 强制下限 $a_{\min}$ 存在尖锐双阈值相变：若 $a_{\min} \in (0, a_{\mathrm{ins}})$，所有机构被迫落入命题 3(1) 的罚款保险区间，反而降低洁净企业开案门槛 $P^*(a_{\min}) < P^*(0)$；当且仅当 $a_{\min} \ge a^*$ 时，零漂绿状态 $(u_{\mathrm{eq}}, a_{\mathrm{eq}}) = (0, a_{\min})$ 成为全局吸引不动点，同时消除指派 3-周期与自选逃逸周期。

*证明.*
1. 单次携带虚报的报告在发布前被纠错的概率为 $a_k$，未被纠错且被监管抽查追责的概率为 $(1-a_k)\bar q$，两事件互斥，故总观测概率为 $p_{\mathrm{obs}}(a_k) = a_k + (1-a_k)\bar q$，直接取期望并代数整理即得线性收缩式。
2. 对 $\gamma_{\mathrm{eff}}(a)$ 求导即得严格负导数与比值公式。
3. 由非负目标项得 $a_{y+t} \ge \gamma^t a_{\mathrm{peak}}$，令 $\gamma^t a_{\mathrm{peak}} \ge a^*$ 解出 $t \le \ln(a_{\mathrm{peak}}/a^*)/\ln(1/\gamma)$，含峰值年共 $T(\gamma; a_{\mathrm{peak}})$ 年连续 $\ge a^*$。在自选机制下，若 $\gamma a_{\mathrm{peak}} \ge a^*$ 且 $a_{k_-,y+1} \ge a^*$，则对所有 $k \in \{1,\dots,M\}$ 均有 $a_{k,y+1} \ge a^*$，从而 $\min_k a_{k,y+1} \ge a^*$，当期全体企业均选择停止漂绿。
4. $a_{\min} \in (0, a_{\mathrm{ins}})$ 由命题 3(1) 得出；$a_{\min} \ge a^*$ 时恒有 $a_{k,y} \ge a^*$，故无新漂绿且旧存量按 $\rho(a_{\min})^{mt} \to 0$ 全局收敛至 0。$\square$
