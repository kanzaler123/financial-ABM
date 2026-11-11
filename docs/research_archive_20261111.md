# ABM 阶段性封存（2026-11-11）

Git 标签 `archive-2026-11-11` 固定本次研究快照。工作论文采用 10 月 29 日的 [PDF](../paper/main.pdf) 和 [LaTeX 源稿](../paper/main.tex)，对应公开主分支提交 `6f740dfe333763e2b81b5835e77f4aa450e7a4e4`。阶段 K 收尾基于 `fd6295a` 的实验代码和规格，结果记录到[研究契约第 28.39 条](disclosure_research_contract.md#28-阶段-k执法强度漂绿代理与企业成本参数2026-10-29-起草用户于同日批准此后固定先于任何本节的估计与运行)。

| 研究内容 | 结果入口 |
|---|---|
| 金融市场模型 | [阶段 1](../reports/stage1_iter6_20260926/analysis.md)、[阶段 2](../reports/stage2_20260926/analysis.md)、[阶段 3](../reports/stage3_20261003/analysis.md) |
| ESG 披露与漂绿主实验 | [阶段 F](../reports/disclosure_stage_f_20261014/README.md)、[自然衰减扫描](../reports/disclosure_decay_grid_20261016/README.md)、[反馈机制](../reports/disclosure_feedback_channels_main_20261017/README.md) |
| 定向核查、鉴证、学习企业 | [定向核查](../reports/disclosure_targeted_audits_20261020/README.md)、[鉴证](../reports/disclosure_assurance_20261021/README.md)、[学习型企业重做](../reports/disclosure_learning_redo_20261021/README.md) |
| 稳健性 | [单项稳健性](../reports/disclosure_robustness_20261018/README.md)、[联合稳健性](../reports/disclosure_joint_robustness_20261023/README.md) |
| 评级、融资与投资者反应 | [评级持续性估计](../reports/disclosure_persistence_estimation_20261026/README.md)、[确认性重跑](../reports/disclosure_stage_h3_20261027/README.md)、[融资溢价估计](../reports/disclosure_financing_premium_20261027/README.md)、[融资机制仿真](../reports/disclosure_stage_i_20261027/README.md)、[投资者反应](../reports/disclosure_investor_response_20261028/README.md) |
| 执法和企业成本 | [阶段 K 报告](../reports/disclosure_enforcement_costs_20261029/README.md)、[规模核查完整汇总](../reports/disclosure_enforcement_costs_20261029/sizeaudit_k3.json)、[390 次运行校验](../reports/disclosure_enforcement_costs_20261029/sizeaudit_v1_sizeaudit_check.json) |

本地封存包 `financial-ABM-2026-11-11.zip` 包含标签对应的仓库快照和 Git 历史、`reports/` 与 `runs/` 的完整研究输出、冻结配置和研究数据，以及论文版本产物。逐文件 SHA-256 清单和恢复说明随包保存。供应商数据、派生面板与逐次运行文件按研究契约在本地保管。公开复现范围见[复现验收报告](../reports/reproduction_acceptance_20261023/README.md)。

阶段 K 恢复点为规模核查可行性扫描。规则公开和不公开两种设定都达到起草重估的门槛。完整重估、整体拟合检验和使用新估计值的政策重跑留待恢复研究时决定。
