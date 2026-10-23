<p align="center">
  <img src=".github/assets/disclosure-cover.svg" alt="Financial ABM policy desk: policy meets markets. Who greenwashes, and why? A multi-agent artificial market for ESG transition and greenwashing. The firm sphere on the right is a structural schematic, not live data." width="100%">
</p>

<p align="center">
  <a href="https://github.com/kanzaler123/financial-ABM/actions/workflows/tests.yml"><img src="https://github.com/kanzaler123/financial-ABM/actions/workflows/tests.yml/badge.svg?branch=master" alt="Code tests (not model acceptance)"></a>
  <a href="https://github.com/kanzaler123/financial-ABM/actions/workflows/showcase.yml"><img src="https://github.com/kanzaler123/financial-ABM/actions/workflows/showcase.yml/badge.svg?branch=master" alt="Showcase build"></a>
</p>

<p align="center">
  <b>Turn a regulator's dial and watch 200 firms decide: improve for real, or greenwash?</b><br>
  An open-source agent-based market where investors, firms and a regulator learn and react, built to compare ESG policies that were never tried.
</p>

<p align="center"><a href="https://kanzaler123.github.io/financial-ABM/"><b>Try the live policy desk</b></a> · <a href="#quick-start"><b>Run it locally</b></a> · <a href="#what-we-found"><b>What we found</b></a> · <a href="CONTRIBUTING.md"><b>Contribute</b></a> · <a href="README.zh-CN.md"><b>中文</b></a></p>

<p align="center">
  <a href="https://kanzaler123.github.io/financial-ABM/"><img src=".github/assets/policy-desk-demo.gif" alt="The policy desk as transparency rises from 0.10 to 0.70: orange firms, which inflate their reports, thin out and disappear near the threshold, and the sphere turns blue, improving for real." width="88%"></a><br>
  <sub>Transparency rises from 0.10 to 0.70; orange firms (greenwashing) thin out and vanish near the threshold. Simplified replay in the browser, not a research result. <a href="https://kanzaler123.github.io/financial-ABM/">Open it and move the levers yourself.</a></sub>
</p>

## What we found

In a pre-registered experiment on 1,770 simulated markets, conditional on the model's assumptions:

- **Clearer sight beats heavier fines.** Raising transparency moves firms toward real improvement more than raising the fixed fine, at the doses tested.
- **Transparency has a threshold, but how sharp it is depends on costs.** If every firm faces the same cost of greenwashing, firms stop greenwashing altogether above it; if the cost differs across firms, greenwashing falls step by step as transparency rises. Heavier fines have diminishing bite.
- **Big firms greenwash more,** because a capped fine is negligible next to a large market cap, not because improving costs them more.
- **Nothing works without investors who care.** Without impact investors' taste for ESG, none of the policies has any effect.

Numbers, intervals and caveats are in [Stage F findings](#stage-f-findings) below.

> [!NOTE]
> **Research in progress.** The stage F confirmatory experiment is complete. Its conclusions are conditional on the model's assumptions, and policy-effect magnitudes depend on an uncalibrated natural decay, so they are reported as a range. Other items carry an evidence label.

## Quick start

**No install:** open the [live policy desk](https://kanzaler123.github.io/financial-ABM/), move a lever, then use **Copy a link to this policy** to share exactly what you see.

**Run the model** (Python 3.12+; a few minutes including install):

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS / Linux: source .venv/bin/activate
python -m pip install -c requirements-lock.txt -e ".[plot,research]"
python -m abm.disclosure.experiments --spec configs/disclosure_quickstart_v1.json --output runs/quickstart --workers 2
```

The run takes about 10–30 seconds: three policies (baseline, transparency 0.6, a proportional fine) on a small market of 40 firms and 4 seeds. Open `runs/quickstart/report.md` for paired contrasts with intervals and `policy_effects.png` for the chart.

**Change one thing and see it move.** In [`configs/disclosure_quickstart_v1.json`](configs/disclosure_quickstart_v1.json), set scenario `I` to `"transparency": 0.3` and run again with `--output runs/quickstart_tau03`. Transparency is now below the threshold: the information policy's gain over baseline falls from about +0.37 SD to about +0.07 SD, and the fine now does better. The quickstart uses the bundled, uncalibrated ESG domain at small scale, so it demonstrates the pipeline; it does not reproduce stage F.

Developers: `python -m pip install -c requirements-lock.txt -e ".[all,dev]"` then `python -m pytest`. The single-asset Stage 1–3 application (web and desktop observatory, SEC 8-K information factors) is described in the [Chinese README](README.zh-CN.md) and [OPERATIONS.md](OPERATIONS.md).

## Build on it

- **A new domain is a JSON file.** The disclosure game, a true attribute versus a reported one, is not specific to ESG: copy [`configs/domains/esg.json`](configs/domains/esg.json) and model carbon claims, earnings management or AI-washing.
- **A new policy experiment is a spec.** Copy the quickstart spec, add scenarios and paired comparisons, and the runner reports intervals.
- **Start small.** [CONTRIBUTING.md](CONTRIBUTING.md) explains the setup, the research rules and where to begin; issues labelled [`good first issue`](https://github.com/kanzaler123/financial-ABM/labels/good%20first%20issue) are scoped for newcomers.
- **Cite it:** [CITATION.cff](CITATION.cff).

## The question

Most ESG research asks backward-looking correlation questions: do high-ESG firms earn more, or borrow more cheaply? Those methods struggle to explain **how** a transition happens, and they cannot compare policies that were never tried.

This project builds an artificial financial market in which heterogeneous investors and firms interact, learn and evolve, and then treats policy as a controlled variable:

> **Under joint pressure from policy and markets, how efficient is the ESG transition, and what path does it take?**

1. **Which policy works better?** Information policies (transparency) versus enforcement policies (fines), compared at doses fixed in advance: transparency raised from 0.1 to 0.6, and the fixed fine raised from CNY 10 million to 4×, each with its own dose ladder. The doses are not matched on expected fine or implementation cost, so the comparison is a ranking at these doses, not which policy is better at equal cost.
2. **How strong is the market's own engine?** Can impact investors drive a transition by “voting with their feet”? Do policies amplify or replace that force?
3. **Do large and small firms differ?** Is there an “ESG divide” in how fast norms spread by firm size?
4. **When does the market fall into a greenwashing trap?** Phase diagrams separating transition success, failure and greenwashing traps.

## Agents

| Agent | Count | Behaviour |
|---|---|---|
| **Profit-maximizing investors (PM)** | part of 1,000 investors | Mean–variance allocation; ESG has no utility value |
| **Impact investors (II)** | part of 1,000 investors | Utility includes a taste for *reported* ESG. Investors imitate the better-performing type each quarter, so the II share evolves |
| **Firms** | 200, in large / mid / small tiers | Each period: **real improvement** (costly, delayed), **greenwashing** (cheap, fast, may be audited) or **maintain**. NPV-rational or Q-learning modes |
| **Regulator** | 1 | Transparency, audit probability, fines and subsidies; policies can switch on any date |
| **Market maker and noise flow** | — | Multi-asset quasi order book with price impact, volatility-dependent depth, and daily cash and share conservation |

The disclosure game, a true attribute versus a reported attribute, is not specific to ESG. **A domain is a JSON file**; the core code contains no domain vocabulary, and a test enforces that. Bundled domains: [ESG](configs/domains/esg.json), [AI-washing](configs/domains/ai_washing.json), [WBA social](configs/domains/wba_social_2023.json).

**Game-theoretic benchmarks.** Each layer maps to a classic game: firm vs regulator (inspection game), firm vs investors (signaling: pooling vs separating), investor population (evolutionary game and replicator dynamics), regulator as a Stackelberg leader. The simulation tests whether boundedly rational, learning agents reach these equilibria, and where they get stuck.

## Stage F findings

> Confirmatory experiment (2026-10-15): 59 scenarios × 30 paired seeds (41001–41030), 1,770 worlds at contract scale (1,000 investors, 200 firms, 3,500 trading days). The design, seeds and primary questions were written into the [research contract, §20](docs/disclosure_research_contract.md) and committed before the run (`2b08f65`). Firms use the continuous NPV rule. Every conclusion is conditional on the model's assumptions.

1. **Robust ordering: information > penalty > baseline.** This is a ranking at the tested doses (transparency 0.6 against a 4× fixed fine), which are not matched on expected fine or implementation cost; it does not mean information policy is better at equal cost. It holds at both ends of natural decay (0 and 0.2 SD/year), under both initialization controls and under the “equal cost” control, in which firms’ improvement cost does not vary with size. Exploratory checks run after stage F ([robustness report (Chinese)](reports/disclosure_robustness_20261018/README.md)) find the same ordering under mean reversion of true levels, one fixed fine per false report, and wash costs that differ across firms; under mean reversion, however, information beats penalty by only +0.003 to +0.006 SD.
2. **Magnitudes are a range.** Over the policy window (days 1200–3500), the information policy raises the true level over baseline by +0.04 to +0.33 SD, the penalty policy by +0.005 to +0.13 SD, and information beats penalty by +0.035 to +0.21 SD. The ends correspond to a natural decay of 0 and 0.2 SD/year without effort. That parameter is uncalibrated; pinning it down needs a verified rating panel (plan stage H). The range covers uniform decay only: under mean reversion at speeds bounded by the Huazheng panel, information beats baseline by +0.02 to +0.03 SD and penalty by close to 0, both below the lower end. Items 3–6 below exist only for the main setting (0.2 SD/year), so their magnitudes are the upper end.
3. **Policy works only through market preference.** Impact investors' preference raises the full 14-year true level by +0.79 SD [+0.74, +0.85]. Without it, firms neither improve nor greenwash, and none of the three policies has any effect.
4. **Transparency has a threshold.** When every firm has the same wash cost and real improvement lasts, greenwashing decisions stop entirely above τ* ≈ 0.57 and fines no longer matter; Proposition 2's threshold prediction held in all 16 grid cells. Undetected inflation accumulated before the switch does not disappear by itself. When wash costs differ across firms, each firm has its own threshold and greenwashing falls step by step with transparency (still 7% of decisions at 0.6); when true levels revert to the mean, the threshold rises to about 0.8.
5. **Fixed fines have diminishing marginal deterrence.** Going from no fine to CNY 10 million cuts the share of greenwashing decisions from 91% to 22%; under one fine per detected case, doubling, quadrupling and multiplying it by eight only reaches 16%, 13% and 13%, because a firm with an undetected case adds new inflation to that case and faces no new fixed fine. If every false report is fined separately, the 8× fine cuts the share to 0.8%; each doubling still adds less than the one before.
6. **The size gap comes from greenwashing, not costs.** Under the baseline policy, 66% of large firms' decisions are greenwashing against 6% for small firms, and large firms improve 0.33 SD less: a CNY 10 million fine is negligible against a large market cap. A proportional fine removes the size gradient (large firms greenwashing more than small ones), not greenwashing itself: under a 1× proportional fine, 72% of large firms’ decisions are still greenwashing, against 88% for small firms and 84% overall. At the equal-amount reference it is far weaker overall than the fixed fine (−0.34 SD).

**Limits.** Only NPV firms: the learning firms did not pass stage D, in the pre-registered redo either ([below](#learning-firms-stage-d3)). Audit intensity, targeted audits and the implementation costs of policy are not modelled. 37 runs hit the price cap; they stay in the statistics, and excluding them changes nothing. Full tables: [stage F report (Chinese)](reports/disclosure_stage_f_20261014/README.md).

**In-browser policy desk.** [`showcase/`](showcase/README.md) brings the firm decision layer into the browser. The home page opens in English (one click switches to Chinese) on a light policy desk: key stage F results and a live wire across the top, policy levers on the left, a firm sphere of the 200 synthetic firms in the middle, a decision ticket for one firm on the right, and below them agent cards (regulator, firm size classes, investors, the threshold τ*), a decision spectrogram and an audit trace. The desk replays panel ②, a 14-year simplified dynamic without market feedback, and is labelled as not a research result. Further down, panel ① shows how the 200 firms would choose this period under the selected policy, and panel ③ shows the stage F results above with their sources. The decision rule matches the Python model term by term (1,200 random inputs, relative error at most 10⁻¹²), and the page states how far the simplified dynamics are from the full model. An AI-washing switch shows the same mechanism in another domain. The page address carries the policy and the selected firm, so a copied link reopens the same desk. The stage 1 exhibit (`stage1/`) uses the same light style. Live: [https://kanzaler123.github.io/financial-ABM/](https://kanzaler123.github.io/financial-ABM/); to preview locally: `python showcase/build.py --output _site`, then open `_site/index.html`.

<details>
<summary><b>Early diagnostics (before stage B)</b></summary>

**[Fact]** checked against primary sources · **[Diagnosis]** reproducible analysis of model behaviour · **[Dev]** small-scale, few-seed observation · **[Open]** identified but unresolved

- **[Fact] Statutory fines barely register for large Chinese firms.** China's Securities Law (Art. 197(2)) caps fines for false disclosure at CNY 10 million, about 0.01% of the median CSI 300 market cap. The 2024 exchange sustainability-reporting guidelines provide only disciplinary measures, with no fines. Whether greenwashing in ESG reports falls under Art. 197 is unclear. [Evidence table](docs/disclosure_empirical_magnitudes_20261012.md)
- **[Diagnosis] Our firms never greenwashed, and we found out why.** In the development runs, the expected fine was 14.5× the gross gain from washing (median). Mapped to real magnitudes, the model's fine was about 50× the legal cap and the financing benefit 10–30× too small. Without the fine, washing beats real improvement in every recorded decision. [Diagnosis](reports/disclosure_wash_diagnosis_20261012/README.md)
- **[Diagnosis] The transparency threshold is all-or-nothing.** With no fine, every firm washes below transparency ≈ 0.565 and none does above it, a consequence of homogeneous discounting and timing that the formal design must address. Later tested (contract 20.21, group W): with wash costs that differ across firms, the decline becomes gradual.
- **[Dev] Impact investors push real improvement.** Across three initialization settings, the market-preference effect is positive: about +1.7 to +2.5 score points over the measurement window (3 seeds, wide intervals).
- **[Diagnosis, fixed in stage C] With zero taste the impact-investor share still rose, because of the imitation rule.** The rule lacked the random draw of a comparison partner; with Schlag's original rule there is no drift at zero taste. [Stage C report (Chinese)](reports/disclosure_imitation_diagnosis_20261013/README.md)

</details>

## Learning firms (stage D.3)

> Confirmatory redo (2026-10-23), pre-registered in the [research contract, §23](docs/disclosure_research_contract.md) and committed before implementation and runs (`8bb5d9e`). Tabular Q-learning firms learn from their realised accounts: 10 independent trainings × 2 state versions (A: the stage D.2 state; V: the cost tier replaced by the firm's cumulative real improvement) × 4 scenarios (B, I, F, IF), compared with an NPV firm on the same action grid (discrete NPV). 540/540 jobs completed. Full tables: [stage D.3 report (Chinese)](reports/disclosure_learning_redo_20261021/README.md).

- **Stage D is still not accepted.** 91/100 tables pass the coverage test. All 9 failures are post-policy tables, 8 of them in the large-firm group.
- **Training randomness dominates.** Within a scenario, the greenwashing share differs 14–18 times more between independently trained tables than within one table across validation seeds, so a single table cannot answer a policy question.
- **A greenwashing trap, not learned profit (P1 supported).** Under the information policy, learning firms greenwash in 23.7% of decisions against 3.8% for discrete NPV: +19.9 percentage points, Holm p = 0.031. The extra greenwashing loses a median of about 0.042 basis points of market value per decision against the NPV choice, far below the reward noise. The learners cannot learn their way out of it; they have not learned that greenwashing pays. The other four primary questions (P2–P5) are unresolved.
- **Seeing improvement cuts greenwashing, not into improvement** (auxiliary, uncorrected). With the V state, pre-policy tables trained from scratch greenwash 34.7 percentage points less, but those decisions become “maintain”, and real improvement does not rise.
- **Fragile, and outside the main conclusions.** Dropping the 5 replicates that contain a failing table leaves P1 unresolved too. Learning firms stay out of the stage F conclusions above.

## Roadmap

- [x] Single-asset market foundation (Stage 1): frozen v15 formal acceptance 22/22
- [x] Multi-asset market: 60 fixed worlds audited, 57 PASS / 3 FAIL (kept on record); order-flow replay reproduced 60/60 bit for bit
- [x] Empirical start: CSI 300 ESG scores, half-year book equity and market caps, 300/300 each
- [x] Greenwashing-condition diagnosis and evidence review
- [x] Firm game with capped lump-sum fines, plus two inspection-game propositions (stage B)
- [x] Investor evolution: imitation rule corrected against replicator dynamics (stage C)
- [x] Learning firms: compared, then redone with 10 independent trainings per scenario (stage D.3, 2026-10-23); coverage 91/100 tables, so they still do not pass acceptance and are excluded from the formal experiment (stage D)
- [ ] New agents: risk-based regulator, assurance providers, rating disagreement
- [x] Formal experiments: information vs enforcement, phase diagrams, size divide (fresh seeds, design frozen before running; stage F)
- [x] In-browser policy sandbox: [GitHub Pages](https://kanzaler123.github.io/financial-ABM/), v0.2.0 (2026-10-16); redesigned as a light policy desk, English first (2026-10-18)

## Research standards

Key settings are fixed in a [research contract](docs/disclosure_research_contract.md) before any policy experiment that uses them. Failures are kept, not deleted. Evidence directories are stored byte for byte, so the recorded SHA-256 values can be checked directly. Every empirical parameter cites its source and evidence level. Vendor data (Huazheng, Eastmoney and others) has no open redistribution licence and is not included; collection scripts and hash manifests are.

---

<sub>Financial ABM · Research prototype. Inspectable mechanisms, traceable evidence. Not investment advice.</sub>
