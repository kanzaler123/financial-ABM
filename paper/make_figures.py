"""Figures for the working paper, read from committed analysis files only.

Run from the repository root:

    python paper/make_figures.py

Inputs (all tracked in git):
  reports/disclosure_stage_f_20261014/confirmatory_v1_analysis.json
  reports/disclosure_decay_grid_20261016/grid_v1_analysis.json
  reports/disclosure_targeted_audits_20261020/audits_v1_analysis.json
  reports/disclosure_assurance_20261021/assurance_v1_analysis.json
  reports/disclosure_robustness_20261018/robust_v1_analysis.json
  reports/disclosure_joint_robustness_20261023/joint_v1_analysis.json
  reports/disclosure_stage_h3_20261027/h3_v1_analysis.json
  reports/disclosure_stage_i_20261027/stage_i_v2_analysis.json
  reports/disclosure_mas_upgrade_20261125/mas_upgrade_analysis.json

Outputs: paper/figures/fig_*.pdf (vector, for LaTeX) and fig_*.png (for
previews and social posts). The PNGs carry a note under the chart (setting,
window, intervals, fine references), because they travel without the LaTeX
caption. Colours follow one rule across the paper:
information policy blue, fixed (per-case) fines orange, proportional fines
aqua. The palette passed the dataviz validator (light mode, all pairs).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STAGE_F = ROOT / "reports/disclosure_stage_f_20261014/confirmatory_v1_analysis.json"
DECAY = ROOT / "reports/disclosure_decay_grid_20261016/grid_v1_analysis.json"
AUDITS = ROOT / "reports/disclosure_targeted_audits_20261020/audits_v1_analysis.json"
ASSURANCE = ROOT / "reports/disclosure_assurance_20261021/assurance_v1_analysis.json"
ROBUST = ROOT / "reports/disclosure_robustness_20261018/robust_v1_analysis.json"
JOINT = ROOT / "reports/disclosure_joint_robustness_20261023/joint_v1_analysis.json"
STAGE_H3 = ROOT / "reports/disclosure_stage_h3_20261027/h3_v1_analysis.json"
STAGE_I = ROOT / "reports/disclosure_stage_i_20261027/stage_i_v2_analysis.json"
MAS_UPGRADE = ROOT / "reports/disclosure_mas_upgrade_20261125/mas_upgrade_analysis.json"
OUT = ROOT / "paper/figures"

INFO = "#2a78d6"
FIXED = "#eb6834"
PROP = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4df"
BLUE_RAMP = ["#f3f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

METRIC = "true_change.equal_weight_sd"
NOTE_SIZE = 8

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 220,
})


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def grid_y(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name: str, note: str) -> None:
    """Write the PDF for LaTeX, then the PNG with the note drawn under the chart."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    add_note(fig, note)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def add_note(fig, note: str) -> None:
    """Wrap the note to the width of everything drawn so far and place it below."""
    box = fig.get_tightbbox(fig.canvas.get_renderer())  # inches
    width, height = fig.get_size_inches()
    chars = int(box.width * 72 / (0.45 * NOTE_SIZE))
    fig.text(box.x0 / width, (box.y0 - 0.1) / height, textwrap.fill(note, chars),
             ha="left", va="top", fontsize=NOTE_SIZE, color=INK_2, linespacing=1.35)


def contrast(stage_f: dict, name: str) -> tuple[float, float, float]:
    item = stage_f["secondary"][name][METRIC]
    return item["mean"], item["ci_low"], item["ci_high"]


def fig_decay(decay: dict) -> None:
    """Policy effects across the five uniform natural-decay settings."""
    points = decay["grid_points"]
    confirmatory = {0.0, 0.2}
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for key, colour, label in (("I_minus_B", INFO, "Information ($\\tau$ = 0.6) − baseline"),
                               ("F_minus_B", FIXED, "Fixed fine 4× − baseline")):
        xs, ys, lo, hi = [], [], [], []
        for p in points:
            c = decay["points"][str(p)]["contrasts"][key]["true_change"]
            xs.append(p), ys.append(c["mean"]), lo.append(c["ci_low"]), hi.append(c["ci_high"])
        ax.plot(xs, ys, color=colour, linewidth=1.5, zorder=2)
        ax.vlines(xs, lo, hi, color=colour, linewidth=1.2, zorder=2)
        for x, y in zip(xs, ys):
            filled = x in confirmatory
            ax.plot(x, y, marker="o", markersize=6, zorder=3, color=colour,
                    markerfacecolor=colour if filled else "white", markeredgewidth=1.3)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK_2, linewidth=0.6)
    ax.set_xticks(points)
    ax.set_xticklabels([f"{p:g}" for p in points])
    ax.set_xlim(-0.01, 0.21)
    ax.set_xlabel("Natural decay of the true level without effort (SD per year)")
    ax.set_ylabel("Change in true level vs. baseline (SD)")
    grid_y(ax)
    ax.plot([], [], marker="o", color=INK_2, markerfacecolor=INK_2, linestyle="none", label="Confirmatory (pre-registered)")
    ax.plot([], [], marker="o", color=INK_2, markerfacecolor="white", linestyle="none", label="Exploratory supplement")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3)
    save(fig, "fig_decay",
         "Policy window days 1200–3500; means over 30 paired seeds, bars are paired 95% intervals. "
         "Uniform decay only: if true levels revert to the mean instead (exploratory follow-up), "
         "information − baseline is +0.02 to +0.03 SD, below this range, and fixed fine − baseline about +0.02 SD.")


def fig_frontier(stage_f: dict) -> None:
    """Washing share against the change in true level, for every tested dose."""
    sc = stage_f["scenarios"]
    base_x = 100 * sc["B"]["wash"]
    # Dose labels: (scenario, label, label offset in points). Empty labels are
    # covered by a neighbour's label: tau 0.6-0.8 all sit at zero washing, and
    # the 4x and 8x fines nearly coincide. Fixed-fine labels sit left of their
    # points, below the information line that runs just above them.
    series = (
        ("Information (transparency $\\tau$)", INFO,
         [("B", "", None), ("I_0p3", "0.3", (6, 3)), ("I_0p5", "0.5", (6, 3)),
          ("I", "", None), ("I_0p7", "", None), ("I_0p8", "$\\tau$ = 0.6, 0.7, 0.8", (8, 4))]),
        ("Fixed fine per case", FIXED,
         [("F_0x", "0×", (0, 7)), ("B", "", None), ("F_2x", "2×", (-7, -2)),
          ("F", "", None), ("F_8x", "4×, 8×", (-7, -3))]),
        ("Proportional fine", PROP,
         [("P_1x", "1×", (0, 7)), ("P_2x", "2×", (0, 7)), ("P_4x", "4×", (0, 7))]),
    )
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    for name, colour, doses in series:
        xs, ys, lo, hi = [], [], [], []
        for scenario, _, _ in doses:
            xs.append(100 * sc[scenario]["wash"])
            if scenario == "B":
                ys.append(0.0), lo.append(0.0), hi.append(0.0)
            else:
                m, l, h = contrast(stage_f, f"{scenario}_minus_B")
                ys.append(m), lo.append(l), hi.append(h)
        ax.plot(xs, ys, color=colour, linewidth=1.5, zorder=2, label=name)
        ax.vlines(xs, lo, hi, color=colour, linewidth=1.2, zorder=2)
        for (scenario, dose, offset), x, y in zip(doses, xs, ys):
            if scenario == "B":
                continue
            ax.plot(x, y, marker="o", markersize=5.5, color=colour, zorder=3)
            if dose:
                ha = "right" if offset[0] < 0 else ("left" if offset[0] > 0 else "center")
                ax.annotate(dose, (x, y), xytext=offset, textcoords="offset points", ha=ha,
                            va="bottom" if offset[1] > 0 else "center", fontsize=8, color=INK_2)
    ax.plot(base_x, 0, marker="o", markersize=7, color=INK, zorder=4)
    ax.annotate("Baseline\n($\\tau$ = 0.1, fixed fine 1×)", (base_x, 0), xytext=(10, -22), textcoords="offset points",
                fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.6))
    ax.axhline(0, color=INK_2, linewidth=0.6)
    ax.set_xlim(-3, 97)
    ax.set_xlabel("Share of firm decisions that inflate the report, policy window (%)")
    ax.set_ylabel("Change in true level vs. baseline (SD)")
    grid_y(ax)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.4)
    save(fig, "fig_frontier",
         "Main setting of the pre-registered stage F experiment: natural decay 0.2 SD/year, the upper end of the "
         "tested range. Days 1200–3500; points are means over 30 paired seeds, bars are paired 95% intervals. "
         "Fixed fine 1× = CNY 10m per detected case. Proportional fine 1× = CNY 10m per SD of detected inflation "
         "for a firm of average value, much lighter per decision than the fixed fine.")


def fig_threshold(stage_f: dict) -> None:
    """Washing share on the w x tau grid with the predicted threshold."""
    cells = stage_f["tau_star_check"]
    ws = sorted({c["w"] for c in cells})
    taus = sorted({c["tau"] for c in cells})
    value = {(c["w"], c["tau"]): 100 * c["wash_share"] for c in cells}
    star = {c["w"]: c["tau_star"] for c in cells}
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    vmax = 50.0
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, w in enumerate(ws):
        for j, tau in enumerate(taus):
            v = value[(w, tau)]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=cmap(min(v, vmax) / vmax), edgecolor="white", linewidth=2))
            ax.text(j + 0.5, i + 0.5, f"{v:.0f}%" if v >= 0.5 else "0",
                    ha="center", va="center", fontsize=9, color="white" if v > 28 else INK)
    # Predicted boundary: firms stop inflating once tau >= tau*(w).
    for i, w in enumerate(ws):
        j = sum(t < star[w] for t in taus)
        ax.plot([j, j], [i, i + 1], color=INK, linewidth=2.2, solid_capstyle="butt", zorder=5)
        if i + 1 < len(ws):
            j_next = sum(t < star[ws[i + 1]] for t in taus)
            if j_next != j:
                ax.plot([j, j_next], [i + 1, i + 1], color=INK, linewidth=2.2, solid_capstyle="butt", zorder=5)
    ax.set_xlim(0, len(taus))
    ax.set_ylim(0, len(ws))
    ax.set_xticks([j + 0.5 for j in range(len(taus))])
    ax.set_xticklabels([f"{t:g}" for t in taus])
    ax.set_yticks([i + 0.5 for i in range(len(ws))])
    ax.set_yticklabels([f"$w$ = {w:g}  ($\\tau^*$ = {star[w]:.2f})" for w in ws])
    ax.set_xlabel("Transparency $\\tau$ from day 1201")
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("Share of decisions that inflate; line = predicted $\\tau^*(w)$", loc="left", fontsize=9, color=INK)
    save(fig, "fig_threshold",
         "Main setting (natural decay 0.2 SD/year); within a row every firm has the same wash cost ratio $w$. "
         "Shares of new decisions in days 1200–3500, means over 30 seeds; inflation already in reports does not "
         "disappear by itself. Exploratory follow-ups: with $w$ differing across firms, 7% of decisions still inflate "
         "at $\\tau$ = 0.6; if true levels revert to the mean, $\\tau^*(0.2)$ rises to about 0.8.")


def fig_size(stage_f: dict) -> None:
    """Washing share by size tier under fixed and proportional fines."""
    sc = stage_f["scenarios"]
    tiers = ("wash_large", "wash_mid", "wash_small")
    panels = (
        ("(a) Fixed fine per case\n1× = CNY 10m per detected case", FIXED,
         [("B", "1×"), ("F_2x", "2×"), ("F", "4×"), ("F_8x", "8×")]),
        ("(b) Proportional fine\n1× = CNY 10m per SD of inflation (average firm)", PROP,
         [("P_1x", "1×"), ("P_2x", "2×"), ("P_4x", "4×")]),
    )
    markers = ("o", "s", "^", "D")
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    for ax, (title, colour, doses) in zip(axes, panels):
        for k, (scenario, dose) in enumerate(doses):
            ys = [100 * sc[scenario][t] for t in tiers]
            ax.plot(range(3), ys, color=colour, linewidth=1.5, marker=markers[k], markersize=5.5, zorder=3)
            # Large firms are where the doses spread out, so label that end.
            ax.annotate(dose, (0, ys[0]), xytext=(-8, 0), textcoords="offset points", ha="right",
                        va="center", fontsize=8, color=INK_2)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Large", "Mid", "Small"])
        ax.set_xlim(-0.5, 2.2)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("Firm size tier")
        grid_y(ax)
    axes[0].set_ylabel("Share of decisions that inflate (%)")
    axes[0].set_ylim(0, 100)
    save(fig, "fig_size",
         "Main setting (natural decay 0.2 SD/year); days 1200–3500, means over 30 seeds. Per decision the "
         "proportional fine is much lighter than the fixed fine at the same multiple, so (a) against (b) compares "
         "the two designs at this reference only, not in general.")


def fig_targeted(audits: dict) -> None:
    """Targeted audits: public vs. hidden inflating rates by tier, and hit-rate vs. deterrence paradox."""
    desc = audits["describe"]
    prim = audits["primary"]
    sec = audits["secondary"]
    tiers = ("wash_share_large", "wash_share_mid", "wash_share_small")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.55, 3.15))

    # Panel (a): Within-tier inflating rates (%) — Random vs. Public & Hidden V / J
    x = [0, 1, 2]
    series = [
        ("B", "Random (B)", INK, ":", "o", True),
        ("V_hidden", "Value hidden ($V_{\\mathrm{hid}}$)", FIXED, "--", "o", False),
        ("V", "Value public (V)", FIXED, "-", "o", True),
        ("J_hidden", "Jump hidden ($J_{\\mathrm{hid}}$)", PROP, "--", "s", False),
        ("J", "Jump public (J)", PROP, "-", "s", True),
    ]
    for sc_key, label, col, ls, mk, filled in series:
        ys = [100.0 * desc[sc_key][t] for t in tiers]
        ax1.plot(x, ys, color=col, linestyle=ls, linewidth=1.45 if filled else 1.15,
                 marker=mk, markersize=5.2, markerfacecolor=col if filled else "white",
                 markeredgewidth=1.25, label=label, zorder=4 if filled else 2)
    ax1.annotate(
        "Small firms:\n4.8% \u2192 35.2%",
        xy=(2.0, 100.0 * desc["V"]["wash_share_small"] + 1.5), xytext=(1.52, 54.0),
        ha="center", va="bottom", fontsize=7.3, color=FIXED,
        arrowprops=dict(arrowstyle="->", color=FIXED, linewidth=0.75),
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Large (10%)", "Mid (30%)", "Small (60%)"])
    ax1.set_xlim(-0.25, 2.28)
    ax1.set_ylim(-2, 86)
    ax1.set_xlabel("Firm size tier")
    ax1.set_ylabel("Within-tier inflating rate (%)")
    ax1.set_title("(a) Public vs. hidden rules by size tier", loc="left", fontsize=8.8, color=INK)
    grid_y(ax1)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none",
               framealpha=0.92, fontsize=7.0, handlelength=1.8, handletextpad=0.35)

    # Panel (b): Hit-rate vs. deterrence paradox (Hit rate % vs. True ESG change vs. B)
    ax2.axhline(0, color=INK_2, linewidth=0.6)
    base_tc = desc["B"]["true_change"]
    points = [
        ("B", "B (random)", 100.0 * desc["B"]["hit_rate"], 0.0, None, None, INK, "o", True, (6, 5)),
        ("A2", "A2 (2\u00d7 random)", 100.0 * desc["A2"]["hit_rate"],
         sec["A2_minus_B"][METRIC]["mean"], sec["A2_minus_B"][METRIC]["ci_low"],
         sec["A2_minus_B"][METRIC]["ci_high"], INFO, "^", True, (7, -2)),
        ("V_hidden", "$V_{\\mathrm{hid}}$", 100.0 * desc["V_hidden"]["hit_rate"],
         desc["V_hidden"]["true_change"] - base_tc, None, None, FIXED, "o", False, (0, 8)),
        ("V", "V (public)", 100.0 * desc["V"]["hit_rate"],
         prim["E1_V_minus_B"]["mean"], prim["E1_V_minus_B"]["ci_low"],
         prim["E1_V_minus_B"]["ci_high"], FIXED, "o", True, (-6, -13)),
        ("J_hidden", "$J_{\\mathrm{hid}}$", 100.0 * desc["J_hidden"]["hit_rate"],
         sec["J_hidden_minus_B"][METRIC]["mean"], sec["J_hidden_minus_B"][METRIC]["ci_low"],
         sec["J_hidden_minus_B"][METRIC]["ci_high"], PROP, "s", False, (0, 8)),
        ("J", "J (public)", 100.0 * desc["J"]["hit_rate"],
         prim["E2_J_minus_B"]["mean"], prim["E2_J_minus_B"]["ci_low"],
         prim["E2_J_minus_B"]["ci_high"], PROP, "s", True, (-6, -13)),
    ]
    coords = {}
    for key, label, hx, ym, lo, hi, col, mk, filled, off in points:
        coords[key] = (hx, ym)
        if lo is not None and hi is not None:
            ax2.vlines(hx, lo, hi, color=col, linewidth=1.2, zorder=2)
        ax2.plot(hx, ym, marker=mk, markersize=5.8, color=col,
                 markerfacecolor=col if filled else "white", markeredgewidth=1.35, zorder=4)
        ha = "left" if off[0] > 0 else ("right" if off[0] < 0 else "center")
        ax2.annotate(label, (hx, ym), xytext=off, textcoords="offset points",
                     ha=ha, va="center", fontsize=7.3, color=col if col != INK else INK)

    # Arrows from hidden to public rules isolating strategic response
    ax2.annotate("", xy=(coords["V"][0] + 0.5, coords["V"][1] + 0.015),
                 xytext=(coords["V_hidden"][0] - 0.5, coords["V_hidden"][1] - 0.012),
                 arrowprops=dict(arrowstyle="->", color=FIXED, linewidth=0.95))
    ax2.text(46.8, -0.095, "Public\n\u22120.186", ha="right", va="center", fontsize=7.1, color=FIXED)
    ax2.annotate("", xy=(coords["J"][0] - 0.4, coords["J"][1] + 0.014),
                 xytext=(coords["J_hidden"][0] + 0.4, coords["J_hidden"][1] - 0.012),
                 arrowprops=dict(arrowstyle="->", color=PROP, linewidth=0.95))
    ax2.text(65.0, -0.072, "Evasion\n\u22120.131", ha="right", va="center", fontsize=7.1, color=PROP)

    ax2.set_xlim(15, 76)
    ax2.set_ylim(-0.255, 0.165)
    ax2.set_xlabel("Audit hit rate (% of audits finding inflation)")
    ax2.set_ylabel("Change in true ESG vs. random B (SD)")
    ax2.set_title("(b) Higher hit rates, lower deterrence", loc="left", fontsize=8.8, color=INK)
    grid_y(ax2)

    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_targeted",
         "Targeted-audit experiment (30 paired seeds, 42001\u201342030; high-headroom uniform-decay reference 0.2 SD/year). "
         "(a) Within-tier inflating decision rates show that hidden targeting tracks random audits B, whereas public V surges "
         "small-firm inflating from 4.8% to 35.2% and public J elevates inflating across all tiers. (b) Both public targeted "
         "rules raise audit hit rates sharply yet worsen true ESG relative to equal-budget random audits B (paired 95% intervals "
         "shown where pre-computed vs. B; V_hidden plots scenario-mean difference vs. B).")


def fig_assurance(ass: dict) -> None:
    """Coupled firm-assurer dynamics: non-monotonic liability curve, escape-cycle collapse, and quality-cycle trajectories."""
    import numpy as np
    from matplotlib.gridspec import GridSpec

    doses = ["0", "1", "4", "10", "16", "100"]
    xs = [0, 1, 2, 3, 4, 5.4]  # visual gap before 100x
    sec = ass["secondary"]
    desc = ass["describe"]
    base_tc = desc["B"]["true_change"]

    ys, lo, hi, wash_r = [], [], [], []
    for d in doses:
        c = sec[f"ALL_R{d}_minus_B"][METRIC]
        ys.append(c["mean"])
        lo.append(c["ci_low"])
        hi.append(c["ci_high"])
        wash_r.append(100.0 * desc[f"ALL_R{d}"]["wash_share"])

    # Firm-choice scenarios: scenario-mean difference vs B (paired CIs are against assigned R, not B)
    f_xs = [1, 4]
    f_ys = [desc["ALL_F1"]["true_change"] - base_tc, desc["ALL_F16"]["true_change"] - base_tc]
    f_wash = [100.0 * desc["ALL_F1"]["wash_share"], 100.0 * desc["ALL_F16"]["wash_share"]]
    i_mean = sec["I_minus_B"][METRIC]["mean"]
    c2 = ass["primary"]["C2_ALL_F16_minus_ALL_R16"]

    fig = plt.figure(figsize=(6.7, 5.35))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 0.95], hspace=0.44, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # Panel (a): True ESG change vs baseline
    ax1.axhline(0, color=INK_2, linewidth=0.6)
    ax1.axhline(i_mean, color=INFO, linestyle="--", linewidth=1.0, zorder=1)
    ax1.text(0.05, i_mean + 0.008, f"Information ($\\tau$=0.6): +{i_mean:.3f}", fontsize=7.5, color=INFO)
    ax1.plot(xs, ys, color=PROP, linewidth=1.6, zorder=2, label="Assigned providers (R)")
    ax1.vlines(xs, lo, hi, color=PROP, linewidth=1.2, zorder=2)
    ax1.plot(xs, ys, linestyle="none", marker="o", markersize=5.2, color=PROP, zorder=3)
    ax1.plot(f_xs, f_ys, linestyle="none", marker="D", markersize=5.8, color=FIXED,
             markerfacecolor="white", markeredgewidth=1.4, zorder=4, label="Firm choice (F)")
    ax1.annotate(
        "", xy=(4.0, f_ys[1] + 0.010), xytext=(4.0, ys[4] - 0.010),
        arrowprops=dict(arrowstyle="<->", color=FIXED, linewidth=1.1),
    )
    ax1.text(3.85, 0.148,
             f"Escape cycle\n\u2212{abs(c2['mean']):.3f} SD",
             ha="right", va="center", fontsize=7.4, color=FIXED,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
    ax1.annotate(
        "Quality-cycle\ndrop at 100×", xy=(xs[-1], ys[-1] - 0.012), xytext=(5.00, 0.145),
        ha="center", va="top", fontsize=7.3, color=INK_2,
        arrowprops=dict(arrowstyle="->", color=INK_2, linewidth=0.7),
    )
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{d}×" for d in doses])
    ax1.set_xlim(-0.35, 5.95)
    ax1.set_ylim(-0.02, 0.355)
    ax1.set_xlabel("Provider liability multiple (× CNY 5m)")
    ax1.set_ylabel("True ESG vs. baseline (SD)")
    ax1.set_title("(a) True ESG gain: peak at 16× & choice collapse", loc="left", fontsize=8.6, color=INK)
    grid_y(ax1)
    ax1.legend(loc="lower right", frameon=False, fontsize=7.5, handletextpad=0.3)

    # Panel (b): Inflating decision share (%)
    base_wash = 100.0 * desc["B"]["wash_share"]
    ax2.axhline(base_wash, color=INK_2, linestyle=":", linewidth=0.9, zorder=1)
    ax2.text(5.88, base_wash - 0.6, f"Baseline ({base_wash:.1f}%)",
             ha="right", va="top", fontsize=7.2, color=INK_2)
    ax2.plot(xs, wash_r, color=PROP, linewidth=1.6, marker="o", markersize=5.2, zorder=3,
             label="Assigned providers (R)")
    ax2.plot(f_xs, f_wash, linestyle="none", marker="D", markersize=5.8, color=FIXED,
             markerfacecolor="white", markeredgewidth=1.4, zorder=4, label="Firm choice (F)")
    ax2.annotate(
        "96.9% at laxest\nprovider (21.0%)", xy=(4.0, f_wash[1] + 0.5), xytext=(3.10, 24.5),
        ha="center", va="bottom", fontsize=7.4, color=FIXED,
        arrowprops=dict(arrowstyle="->", color=FIXED, linewidth=0.7),
    )
    ax2.annotate(
        "Rebound\n(8.6% vs. 6.2%)", xy=(xs[-1], wash_r[-1] + 0.4), xytext=(4.85, 11.8),
        ha="center", va="bottom", fontsize=7.4, color=INK_2,
        arrowprops=dict(arrowstyle="->", color=INK_2, linewidth=0.7),
    )
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{d}×" for d in doses])
    ax2.set_xlim(-0.35, 5.95)
    ax2.set_ylim(0, 29.0)
    ax2.set_xlabel("Provider liability multiple (× CNY 5m)")
    ax2.set_ylabel("Inflating decisions (%)")
    ax2.set_title("(b) Inflating share: insurance, rebound & shopping", loc="left", fontsize=8.6, color=INK)
    grid_y(ax2)

    # Panel (c): Endogenous annual quality cycles under ALL_R16 (30 paired seeds, assurance years 1–10)
    t6 = ass["predictions"]["T6"]
    series_dict = t6["yearly_mean_quality_by_seed"]
    mat = np.array([series_dict[s] for s in sorted(series_dict.keys())], dtype=float)
    years = np.arange(1, mat.shape[1] + 1)
    q25 = np.percentile(mat, 25, axis=0)
    q50 = np.median(mat, axis=0)
    q75 = np.percentile(mat, 75, axis=0)
    a_star = ass["predictions"]["T3"]["stop_quality"]

    for row in mat:
        ax3.plot(years, row, color=PROP, alpha=0.14, linewidth=0.75, zorder=1)
    ax3.fill_between(years, q25, q75, color=PROP, alpha=0.18, zorder=2, label="Interquartile band (30 seeds)")
    ax3.plot(years, q50, color=PROP, linewidth=1.7, marker="o", markersize=4.2, zorder=4, label="Cross-seed median")
    exemplar = series_dict["43019"]
    ax3.plot(years, exemplar, color=FIXED, linewidth=1.55, linestyle="-", marker="s", markersize=4.0,
             zorder=5, label="Representative oscillating seed (43019)")
    ax3.axhline(a_star, color=INK, linestyle="--", linewidth=1.05, zorder=3)
    ax3.text(10.12, a_star - 0.014, f"Deterrence threshold $a^* = {a_star:.4f}$",
             ha="right", va="top", fontsize=7.4, color=INK,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.0))
    ax3.annotate(
        "Clean year lowers $s_k$\n$\\Rightarrow a_k$ drops below $a^*$",
        xy=(3.0, exemplar[2] - 0.005), xytext=(4.75, 0.012),
        ha="center", va="bottom", fontsize=7.2, color=FIXED,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8),
        arrowprops=dict(arrowstyle="->", color=FIXED, linewidth=0.75),
    )
    ax3.annotate(
        "Renewed inflation spikes $s_k$\n$\\Rightarrow a_k$ tightens again",
        xy=(4.0, exemplar[3] + 0.006), xytext=(6.15, 0.255),
        ha="center", va="center", fontsize=7.2, color=FIXED,
        arrowprops=dict(arrowstyle="->", color=FIXED, linewidth=0.75),
    )
    ax3.set_xticks(years)
    ax3.set_xlim(0.75, 10.25)
    ax3.set_ylim(0.0, 0.365)
    ax3.set_xlabel("Assurance year after policy switch (240 trading days per year)")
    ax3.set_ylabel("Mean provider quality $a_k$")
    ax3.set_title("(c) Endogenous quality oscillation cycles under assigned 16× assurance (ALL_R16: 28 of 30 seeds cycle across $a^*$)",
                  loc="left", fontsize=8.6, color=INK)
    grid_y(ax3)
    ax3.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none",
               framealpha=0.92, fontsize=7.1, ncol=3, handletextpad=0.35, columnspacing=0.9)

    save(fig, "fig_assurance",
         "Coupled firm-assurer experiment (30 paired seeds, 43001–43030; high-headroom uniform-decay reference 0.2 SD/year). "
         "(a) True ESG vs. baseline B across liability multiples with paired 95% intervals for assigned providers and "
         "ALL_F16 − ALL_R16 = −0.199 SD annotated. (b) Inflating shares showing the 1× fine-insurance bump, 16× trough, "
         "100× quality-cycle rebound and 96.9% laxest-provider concentration under firm choice. (c) Annual mean provider quality "
         "under ALL_R16 across assurance years 1–10: 28 of 30 seeds cross above and subsequently fall below a* = 0.0827, "
         "reigniting inflation in all 30 seeds.")


def fig_regimes(stage_f: dict, rob: dict, joint: dict, h3: dict, stage_i: dict) -> None:
    """Phase boundaries: persistence dynamics (panel a) and financing-incentive saturation (panel b)."""
    p2_main = next(x for x in stage_f["primary"] if x["name"] == "P2_I_minus_F")
    decay0 = stage_f["secondary"]["decay0_I_minus_F"][METRIC]
    perm_q = h3["auxiliary"]["runner"]["perm_I_minus_perm_F"][METRIC]
    q_main = h3["primary"]["P1"]
    rev32 = rob["R"]["0.32"]["contrasts"]["I_minus_F"]["true_change"]
    rev50 = rob["R"]["0.5"]["contrasts"]["I_minus_F"]["true_change"]
    j_rr32 = joint["primary"]["J_rr0p32_I_minus_F"]

    rows = [
        ("Uniform decay $\\delta=0.2$\n(ref.; panel rejected)", p2_main["mean"], p2_main["ci_low"], p2_main["ci_high"], True),
        ("$Q$, permanent improvement\n(persistence study, $\\tau^*=0.57$)", perm_q["mean"], perm_q["ci_low"], perm_q["ci_high"], True),
        ("No decay $\\delta=0$\n(Study 1, $\\tau^*=0.57$)", decay0["mean"], decay0["ci_low"], decay0["ci_high"], True),
        ("Reversion $\\kappa=0.32$ + fine/rep.\n(joint test, $\\tau^*=0.77$)", j_rr32["mean"], j_rr32["ci_low"], j_rr32["ci_high"], False),
        ("Reversion $\\kappa=0.32$\n(exploratory, $\\tau^*=0.77$)", rev32["mean"], rev32["ci_low"], rev32["ci_high"], False),
        ("$Q$, transitory improvement\n(persistence study, $\\tau^*=0.83$)", q_main["mean"], q_main["ci_low"], q_main["ci_high"], False),
        ("Reversion $\\kappa=0.50$\n(exploratory, $\\tau^*=0.82$)", rev50["mean"], rev50["ci_low"], rev50["ci_high"], False),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.45), gridspec_kw={"width_ratios": [1.08, 1.0]})

    # Panel (a): Persistence phase boundary for I - F
    y_pos = list(range(len(rows) - 1, -1, -1))
    ax1.axvline(0, color=INK_2, linewidth=0.6)
    ax1.axhline(3.5, color=INK_2, linestyle=":", linewidth=0.8)
    for y, (label, m, lo, hi, crosses) in zip(y_pos, rows):
        col = INFO if crosses else FIXED
        ax1.hlines(y, lo, hi, color=col, linewidth=1.6, zorder=2)
        ax1.plot(m, y, marker="o", markersize=5.2, color=col, zorder=3)
        ax1.text(hi + 0.006, y, f"+{m:.3f}", va="center", ha="left", fontsize=7.5, color=INK)
    ax1.text(0.245, 5.15, "$\\tau=0.6 > \\tau^*$\n(inflation stops)", ha="right", va="center",
             fontsize=7.3, color=INFO)
    ax1.text(0.245, 1.5, "$\\tau=0.6 < \\tau^*$\n(inflation persists)", ha="right", va="center",
             fontsize=7.3, color=FIXED)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([r[0] for r in rows], fontsize=7.4)
    ax1.set_xlim(-0.008, 0.278)
    ax1.set_xlabel("Information $-$ fixed fine $4\\times$, true ESG (SD)")
    ax1.set_title("(a) Persistence phase boundary", loc="left", fontsize=8.8, color=INK)
    ax1.grid(axis="x", color=GRID, linewidth=0.5)
    ax1.set_axisbelow(True)

    # Panel (b): Financing-incentive saturation under Q (Stage I)
    keys = ["lambda0", "lambda_fe", "lambda_lo", "lambda_star", "lambda_hi"]
    lams = [10000.0 * stage_i["points"][k] for k in keys]  # basis points
    p8_at = stage_i["sensitivity"]["contrasts"]["P8"]["at"]
    p5_at = stage_i["sensitivity"]["contrasts"]["P5"]["at"]
    wash_b = [100.0 * stage_i["sensitivity"]["washing_shares"][k]["B"]["wash_share"] for k in keys]

    p8_m = [100.0 * p8_at[k]["mean"] for k in keys]
    p8_lo = [100.0 * p8_at[k]["ci_low"] for k in keys]
    p8_hi = [100.0 * p8_at[k]["ci_high"] for k in keys]
    p5_m = [100.0 * p5_at[k]["mean"] for k in keys]
    p5_lo = [100.0 * p5_at[k]["ci_low"] for k in keys]
    p5_hi = [100.0 * p5_at[k]["ci_high"] for k in keys]

    ax2.plot(lams, wash_b, color=INK_2, linestyle="--", linewidth=1.2, marker="^", markersize=4.5,
             zorder=2, label="Baseline inflating rate (%)")
    ax2.plot(lams, p8_m, color=PROP, linewidth=1.6, marker="o", markersize=5.2, zorder=3,
             label="Laxest-assurer share shift (pp)")
    ax2.vlines(lams, p8_lo, p8_hi, color=PROP, linewidth=1.2, zorder=3)
    ax2.plot(lams, p5_m, color=FIXED, linewidth=1.6, marker="s", markersize=4.8, zorder=3,
             label="Small-firm audit share shift (pp)")
    ax2.vlines(lams, p5_lo, p5_hi, color=FIXED, linewidth=1.2, zorder=3)

    ax2.annotate("Saturation &\nrank collapse", xy=(9.4, wash_b[1] + 0.5), xytext=(2.8, 93.0),
                 ha="center", va="center", fontsize=7.3, color=INK_2,
                 arrowprops=dict(arrowstyle="->", color=INK_2, linewidth=0.7))
    ax2.set_xlim(-1.8, 32.2)
    ax2.set_ylim(-3, 106)
    ax2.set_xlabel("Direct rating slope $\\lambda$ (bp per unit signal)")
    ax2.set_ylabel("Share shift (pp) or baseline inflating (%)")
    ax2.set_title("(b) Financing-incentive boundary (under $Q$)", loc="left", fontsize=8.8, color=INK)
    grid_y(ax2)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.01, 0.72), frameon=True,
               facecolor="white", edgecolor="none", framealpha=0.9,
               fontsize=7.1, handletextpad=0.35)

    fig.tight_layout(w_pad=1.4)
    save(fig, "fig_regimes",
         "Two structural phase boundaries from committed confirmatory and exploratory archives (30 paired seeds per point, "
         "paired 95% intervals). (a) True-ESG advantage I − F stays strictly positive across all persistence regimes but drops "
         "by one to two orders of magnitude when deliberate improvement fades faster than tau = 0.6 can clear tau*. "
         "(b) Under form Q (stage I), raising direct debt-rate sensitivity lambda drives baseline inflation from 21.4% to 93.5%, "
         "compressing cross-tier displacement shares while preserving every contrast sign.")


def fig_stackelberg(mas: dict) -> None:
    """Bayesian dynamic RMAB-Stackelberg audits (§29) and G-band communication granularity (§32C-1)."""
    s29 = mas["studies"]["sec29_stackelberg"]["scenarios"]
    s32c = mas["studies"]["sec32C1_granularity"]["scenarios"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 3.25), gridspec_kw={"width_ratios": [1.16, 1.0]})

    # Panel (a): Hit-rate vs. deterrence paradox 2.0 & VoC / VoI ladder (§29)
    ax1.axhline(0.0, color=INK_2, linewidth=0.6)
    pts = [
        ("B", "B (1% rand.)", INK, "o", True, (6, -10), "left"),
        ("V", "V (public)", FIXED, "o", True, (-6, -10), "right"),
        ("J", "J (public)", FIXED, "s", True, (-6, -10), "right"),
        ("A2", "A2 (2× rand.)", INK_2, "^", False, (0, 10), "center"),
        ("S_obs_hid", "$S_{\\mathrm{obs}}^{\\mathrm{hid}}$ (92%)", INFO, "o", False, (-6, -11), "right"),
        ("S_oracle_hid", "$S_{\\mathrm{orc}}^{\\mathrm{hid}}$ (100%)", PROP, "D", False, (-5, 13), "right"),
        ("S_obs", "$S_{\\mathrm{obs}}$ (1×)", INFO, "o", True, (6, -9), "left"),
        ("S_obs_sig", "$S_{\\mathrm{obs}}^{\\mathrm{sig}}$ (1×)", INFO, "s", True, (7, -2), "left"),
        ("S_oracle_eq", "$S_{\\mathrm{orc}}$ (upper bd.)", PROP, "D", True, (7, 0), "left"),
    ]
    coords = {}
    for key, label, col, mk, filled, off, ha in pts:
        sc = s29[key]
        hx = 100.0 * sc["hit_rate_mean"]
        if key == "B":
            ym, lo, hi = 0.0, None, None
        else:
            vb = sc["vs_baseline_eq"]
            ym, lo, hi = vb["mean"], vb["ci_low"], vb["ci_high"]
        coords[key] = (hx, ym)
        if lo is not None and hi is not None:
            ax1.vlines(hx, lo, hi, color=col, linewidth=1.25, zorder=2)
        ax1.plot(hx, ym, marker=mk, markersize=5.6, color=col,
                 markerfacecolor=col if filled else "white", markeredgewidth=1.35, zorder=4)
        ax1.annotate(label, (hx, ym), xytext=off, textcoords="offset points",
                     ha=ha, va="center", fontsize=7.0, color=col if col != INK_2 else INK)

    # Value of Commitment arrow: S_obs_hid -> S_obs
    ax1.annotate(
        "", xy=(coords["S_obs"][0] + 2.5, coords["S_obs"][1] - 0.008),
        xytext=(coords["S_obs_hid"][0] - 2.5, coords["S_obs_hid"][1] + 0.012),
        arrowprops=dict(arrowstyle="->", color=INFO, linewidth=1.0, connectionstyle="arc3,rad=0.14"),
    )
    ax1.text(73.0, 0.122, "Commitment (VoC)\n+0.128 SD", ha="center", va="center",
             fontsize=7.0, color=INFO)

    # Residual VoI arrow: S_obs_sig -> S_oracle_eq
    ax1.annotate(
        "", xy=(coords["S_oracle_eq"][0] - 0.7, coords["S_oracle_eq"][1] - 0.014),
        xytext=(coords["S_obs_sig"][0] + 0.7, coords["S_obs_sig"][1] + 0.014),
        arrowprops=dict(arrowstyle="->", color=PROP, linewidth=0.95, linestyle="--"),
    )
    ax1.text(30.5, 0.228, "VoI\n+0.115", ha="right", va="center", fontsize=6.9, color=PROP)

    ax1.set_xlim(7, 107)
    ax1.set_ylim(-0.245, 0.315)
    ax1.set_xlabel("Audit hit rate (% of audits finding inflation)")
    ax1.set_ylabel("True ESG vs. random B (SD)")
    ax1.set_title("(a) RMAB-Stackelberg VoC & VoI (§29)", loc="left", fontsize=8.6, color=INK)
    grid_y(ax1)

    # Panel (b): Communication-granularity curve (§32C-1)
    ax2.axhline(0.0, color=INK_2, linewidth=0.6)
    g_keys = ["NPV_B", "NPV_g2", "NPV_g3", "NPV_g5", "NPV_g8", "NPV_S_obs"]
    g_x = [0, 1, 2, 3, 4, 5]
    g_labels = ["1\n(B)", "2\n(1 bit)", "3\n(1.6 b)", "5\n(2.3 b)", "8\n(3.0 b)", "$N$\n(exact)"]
    ys, lo, hi = [], [], []
    for k in g_keys:
        if k == "NPV_B":
            ys.append(0.0), lo.append(0.0), hi.append(0.0)
        else:
            vb = s32c[k]["vs_baseline_eq"]
            ys.append(vb["mean"]), lo.append(vb["ci_low"]), hi.append(vb["ci_high"])
    ax2.plot(g_x, ys, color=INFO, linewidth=1.6, zorder=3, label="Forward-looking NPV (vs. B)")
    ax2.vlines(g_x[1:], lo[1:], hi[1:], color=INFO, linewidth=1.25, zorder=3)
    ax2.plot(g_x, ys, linestyle="none", marker="o", markersize=5.4, color=INFO, zorder=4)

    full_gain = ys[-1]
    for idx, off in ((1, (0, 8)), (3, (0, 8)), (4, (0, 8)), (5, (-4, 8))):
        pct = 100.0 * ys[idx] / full_gain
        ax2.annotate(f"{pct:.0f}%", (g_x[idx], ys[idx]), xytext=off, textcoords="offset points",
                     ha="center", va="bottom", fontsize=7.0, color=INFO)

    # Bounded-learner cognitive boundary (vs. EB_B)
    eb_x = [0, 1, 5]
    eb_ys = [0.0, s32c["EB_g2"]["vs_eb_baseline_eq"]["mean"], s32c["EB_S_obs"]["vs_eb_baseline_eq"]["mean"]]
    eb_lo = [0.0, s32c["EB_g2"]["vs_eb_baseline_eq"]["ci_low"], s32c["EB_S_obs"]["vs_eb_baseline_eq"]["ci_low"]]
    eb_hi = [0.0, s32c["EB_g2"]["vs_eb_baseline_eq"]["ci_high"], s32c["EB_S_obs"]["vs_eb_baseline_eq"]["ci_high"]]
    ax2.plot(eb_x, eb_ys, color=FIXED, linestyle="--", linewidth=1.25, zorder=2,
             label="Empirical-belief learner (vs. EB_B)")
    ax2.vlines(eb_x[1:], eb_lo[1:], eb_hi[1:], color=FIXED, linewidth=1.2, zorder=2)
    ax2.plot(eb_x, eb_ys, linestyle="none", marker="D", markersize=5.2, color=FIXED,
             markerfacecolor="white", markeredgewidth=1.35, zorder=4)
    ax2.text(3.0, -0.095, "Unquantified label fails ($G5$:\nunresolved, $p=0.54$)",
             ha="center", va="center", fontsize=6.9, color=FIXED,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.8))

    ax2.set_xticks(g_x)
    ax2.set_xticklabels(g_labels, fontsize=7.6)
    ax2.set_xlim(-0.35, 5.45)
    ax2.set_ylim(-0.145, 0.182)
    ax2.set_xlabel("Published risk bands $G$ ($\\log_2 G$ bits)")
    ax2.set_ylabel("True ESG vs. own random B (SD)")
    ax2.set_title("(b) Communication granularity (§32C-1)", loc="left", fontsize=8.6, color=INK)
    grid_y(ax2)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none",
               framealpha=0.9, fontsize=6.9, handletextpad=0.35)

    fig.tight_layout(w_pad=1.35)
    save(fig, "fig_stackelberg",
         "Bayesian dynamic RMAB-Stackelberg audits (§29, seeds 71001–71030) and G-band communication granularity (§32C-1, "
         "seeds 76001–76030; 30 paired seeds each, paired 95% intervals). (a) Pure-public history-only S_obs matches 2× "
         "random budget A2 at 1× budget and beats hidden Bayes clearing S_obs_hid (92.1% hit rate) by +0.128 SD; adding "
         "public report signals S_obs_sig beats A2 (+0.032 SD). (b) Incentive-compatible G-band menus capture 63% (G=5) "
         "and 72% (G=8, 3 bits) of exact S_obs gain for forward-looking firms, whereas unquantified 2-band labels fail to "
         "rescue empirical-belief learners (G5 unresolved, Holm p = 0.539).")


def fig_remedies(mas: dict) -> None:
    """Partial-observability assurance blindness, hysteresis/floor phase transitions, and Pareto frontier (§31, §32B)."""
    from matplotlib.gridspec import GridSpec

    s31 = mas["studies"]["sec31_assurance"]["scenarios"]
    s32b = mas["studies"]["sec32B_phase_pareto"]["scenarios"]

    fig = plt.figure(figsize=(6.75, 5.35))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.34)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Panel (a): §31 Partial observability (detected_bayes) & institutional remedies
    i_gain = s31["I"]["vs_baseline_eq"]["mean"]
    rows_a = [
        ("R100 + hyst. ($\\gamma$=0.7)", "ALL_R100_bayes_smooth", PROP, "o", True),
        ("F100 + hyst. (choice)", "ALL_F100_bayes_smooth", INFO, "D", True),
        ("ROT16 (annual rot.)", "ALL_ROT16_bayes", INFO, "^", True),
        ("R16 (assigned)", "ALL_R16_bayes", PROP, "o", False),
        ("R100 (3-cycle dip)", "ALL_R100_bayes", PROP, "s", False),
        ("F16 (choice escape)", "ALL_F16_bayes", FIXED, "D", False),
    ]
    ys_a = list(range(len(rows_a) - 1, -1, -1))
    ax1.axvline(0.0, color=INK_2, linewidth=0.6)
    ax1.axvline(i_gain, color=INFO, linestyle="--", linewidth=1.0, zorder=1)
    ax1.text(i_gain + 0.014, 0.24, f"$I$ ($\\tau$=0.6)\n+{i_gain:.3f}", ha="left", va="center",
             fontsize=6.8, color=INFO)
    for y, (label, key, col, mk, filled) in zip(ys_a, rows_a):
        vb = s31[key]["vs_baseline_eq"]
        w = 100.0 * s31[key]["wash_share"]
        ax1.hlines(y, vb["ci_low"], vb["ci_high"], color=col, linewidth=1.45, zorder=2)
        ax1.plot(vb["mean"], y, marker=mk, markersize=5.2, color=col,
                 markerfacecolor=col if filled else "white", markeredgewidth=1.3, zorder=4)
        ax1.text(vb["ci_high"] + 0.008, y, f"+{vb['mean']:.3f} ({w:.1f}%)",
                 ha="left", va="center", fontsize=6.7, color=INK,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.4))
    ax1.set_yticks(ys_a)
    ax1.set_yticklabels([r[0] for r in rows_a], fontsize=7.2)
    ax1.set_xlim(-0.01, 0.56)
    ax1.set_xlabel("True ESG vs. B (SD); label = gain (wash %)")
    ax1.set_title("(a) Partial-obs. remedies (§31, detected_bayes)", loc="left", fontsize=8.3, color=INK)
    ax1.grid(axis="x", color=GRID, linewidth=0.5)
    ax1.set_axisbelow(True)

    # Panel (b): §32B Hysteresis sweep (gain vs. total social resource cost)
    gammas = [0.0, 0.3, 0.5, 0.7, 0.9]
    g_keys = ["F100_g00", "F100_g03", "F100_g05", "F100_g07", "F100_g09"]
    g_m = [s32b[k]["vs_baseline_eq"]["mean"] for k in g_keys]
    g_lo = [s32b[k]["vs_baseline_eq"]["ci_low"] for k in g_keys]
    g_hi = [s32b[k]["vs_baseline_eq"]["ci_high"] for k in g_keys]
    g_res = [s32b[k]["resource_cost_bp"] for k in g_keys]
    b_res = s32b["B"]["resource_cost_bp"]

    ax2.plot(gammas, g_m, color=INFO, linewidth=1.6, marker="o", markersize=5.2, zorder=3,
             label="True ESG vs. B (SD)")
    ax2.vlines(gammas, g_lo, g_hi, color=INFO, linewidth=1.25, zorder=3)
    ax2.set_xticks(gammas)
    ax2.set_xlim(-0.06, 0.97)
    ax2.set_ylim(0.0, 0.375)
    ax2.set_xlabel("Runtime hysteresis weight $\\gamma$ (F100, choice)")
    ax2.set_ylabel("True ESG vs. B (SD)", color=INFO)
    ax2.tick_params(axis="y", colors=INFO)
    grid_y(ax2)

    ax2r = ax2.twinx()
    ax2r.spines["top"].set_visible(False)
    ax2r.spines["right"].set_visible(True)
    ax2r.spines["right"].set_color(FIXED)
    ax2r.plot(gammas, g_res, color=FIXED, linestyle="--", linewidth=1.4, marker="s", markersize=4.8,
              markerfacecolor="white", markeredgewidth=1.3, zorder=2, label="Resource cost (bp)")
    ax2r.axhline(b_res, color=INK_2, linestyle=":", linewidth=0.8)
    ax2r.text(0.01, b_res - 0.14, f"B ({b_res:.2f} bp)", ha="left", va="top", fontsize=6.6, color=INK_2)
    ax2r.annotate("−2.89 bp\nfree lunch", xy=(0.68, g_res[3]), xytext=(0.43, 2.35),
                  ha="center", va="center", fontsize=6.8, color=FIXED,
                  arrowprops=dict(arrowstyle="->", color=FIXED, linewidth=0.75))
    ax2r.set_ylim(1.8, 6.2)
    ax2r.set_ylabel("Total resource cost (bp)", color=FIXED)
    ax2r.tick_params(axis="y", colors=FIXED)
    ax2.set_title("(b) Hysteresis & resource free lunch (§32B)", loc="left", fontsize=8.3, color=INK)

    # Panel (c): §32B Statutory floor phase transition
    mins = [0.00, 0.03, 0.06, 0.15, 0.25]
    m_keys = ["F16_m000", "F16_m003", "F16_m006", "F16_m015", "F16_m025"]
    m_y = [s32b[k]["vs_baseline_eq"]["mean"] for k in m_keys]
    m_lo = [s32b[k]["vs_baseline_eq"]["ci_low"] for k in m_keys]
    m_hi = [s32b[k]["vs_baseline_eq"]["ci_high"] for k in m_keys]

    ax3.axvspan(0.0, 0.0346, color=FIXED, alpha=0.14, zorder=1)
    ax3.axvline(0.0346, color=FIXED, linestyle="--", linewidth=0.95, zorder=2)
    ax3.axvline(0.0827, color=INK, linestyle="--", linewidth=0.95, zorder=2)
    ax3.plot(mins, m_y, color=PROP, linewidth=1.6, marker="o", markersize=5.2, zorder=4)
    ax3.vlines(mins, m_lo, m_hi, color=PROP, linewidth=1.25, zorder=4)
    ax3.text(0.038, 0.355, "$a_{\\mathrm{ins}}$\n0.035", ha="left", va="top", fontsize=6.6, color=FIXED)
    ax3.text(0.087, 0.355, "$a^* = 0.0827$", ha="left", va="top", fontsize=6.7, color=INK)
    ax3.annotate("0% wash\nsaturated", xy=(0.15, m_y[3] - 0.008), xytext=(0.185, 0.165),
                 ha="center", va="center", fontsize=6.8, color=PROP,
                 arrowprops=dict(arrowstyle="->", color=PROP, linewidth=0.75))
    ax3.set_xlim(-0.012, 0.27)
    ax3.set_ylim(0.0, 0.375)
    ax3.set_xlabel("Statutory minimum quality $a_{\\min}$ (F16, choice)")
    ax3.set_ylabel("True ESG vs. B (SD)")
    ax3.set_title("(c) Statutory floor phase transition (§32B)", loc="left", fontsize=8.3, color=INK)
    grid_y(ax3)

    # Panel (d): §32B Quadratic effort-cost Pareto frontier
    h_eff = [1000.0 * s32b[k]["assurance_effort_cost"] for k in g_keys]
    f_eff = [1000.0 * s32b[k]["assurance_effort_cost"] for k in m_keys]
    ax4.plot(h_eff, g_m, color=INFO, linewidth=1.5, marker="o", markersize=5.0, zorder=3,
             label="Hysteresis $\\gamma$ ($a^*$ unknown)")
    ax4.plot(f_eff, m_y, color=PROP, linewidth=1.5, marker="s", markersize=5.0, zorder=4,
             label="Floor $a_{\\min}$ ($a^*$ known)")
    for idx, txt, off in ((0, "$\\gamma$=0", (-6, -2)), (1, "0.3", (-6, 1)), (2, "0.5", (-6, 0)), (3, "0.7", (6, -5)), (4, "0.9", (-2, -10))):
        ax4.annotate(txt, (h_eff[idx], g_m[idx]), xytext=off, textcoords="offset points",
                     ha="left" if off[0] > 0 else "right", va="center", fontsize=6.6, color=INFO)
    for idx, txt, off in ((2, "0.06", (6, 0)), (3, "0.15", (-2, 8)), (4, "0.25", (6, 4))):
        ax4.annotate(txt, (f_eff[idx], m_y[idx]), xytext=off, textcoords="offset points",
                     ha="center" if off[0] < 0 else "left", va="center", fontsize=6.6, color=PROP)
    ax4.annotate("", xy=(f_eff[4] + 0.20, m_y[4] - 0.010), xytext=(h_eff[3] - 0.15, g_m[3] + 0.005),
                 arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.9))
    ax4.text(2.45, 0.240, "Jensen: −73%\n(−90% at 0.15)", ha="center", va="center",
             fontsize=6.6, color=INK, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.5))
    ax4.set_xlim(-0.2, 8.95)
    ax4.set_ylim(0.0, 0.375)
    ax4.set_xlabel("Verifier effort cost ($\\times 10^{-3}$ model units)")
    ax4.set_ylabel("True ESG vs. B (SD)")
    ax4.set_title("(d) Effort-cost Pareto frontier (§32B)", loc="left", fontsize=8.3, color=INK)
    grid_y(ax4)
    ax4.legend(loc="center right", bbox_to_anchor=(1.0, 0.43), frameon=True, facecolor="white", edgecolor="none",
               framealpha=0.92, fontsize=6.5, handletextpad=0.3)

    save(fig, "fig_remedies",
         "Partial-observability assurance remedies (§31, seeds 73001–73030) and phase-transition / Pareto frontier sweeps "
         "(§32B, seeds 75001–75030; 30 paired seeds, paired 95% intervals, detected_bayes). (a) Under 50.5× monitoring "
         "blindness, runtime hysteresis (γ=0.7) cures both 100× 3-cycles and firm-choice escape cycles (recovering 97.3% "
         "of transparency I); annual rotation also closes escape. (b) Hysteresis cuts total resource cost by −2.89 bp "
         "(5.55 to 2.66 bp). (c) Statutory floor exhibits a sharp phase transition across a_ins = 0.0346 to a* = 0.0827. "
         "(d) Under quadratic effort cost, a fixed floor saves 73%–90% verifier effort over pulsed hysteresis when a* is known.")


def fig_bounded(mas: dict) -> None:
    """Zero-leakage online empirical-belief bounded rationality (§30): robustness vs. cognitive boundary."""
    s30 = mas["studies"]["sec30_empirical_belief"]["scenarios"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.65, 3.2), gridspec_kw={"width_ratios": [1.18, 1.0]})

    # Panel (a): Policy contrasts vs. EB_B
    rows = [
        ("F100 + hyst. (choice)", "EB_F100_bayes_smooth", PROP, "D", True),
        ("Information ($\\tau$=0.6)", "EB_I", INFO, "o", True),
        ("R16 assigned (bayes)", "EB_R16_bayes", PROP, "o", True),
        ("F16 choice (bayes)", "EB_F16_bayes", FIXED, "D", False),
        ("Fixed fine 4× (learned)", "EB_F", INK_2, "s", True),
        ("Stackelberg $S_{\\mathrm{obs}}$", "EB_S_obs", FIXED, "o", False),
        ("Value targeting (V)", "EB_V", FIXED, "o", True),
    ]
    ys = list(range(len(rows) - 1, -1, -1))
    ax1.axvline(0.0, color=INK_2, linewidth=0.6)
    for y, (label, key, col, mk, filled) in zip(ys, rows):
        vb = s30[key]["vs_baseline_eq"]
        ax1.hlines(y, vb["ci_low"], vb["ci_high"], color=col, linewidth=1.45, zorder=2)
        ax1.plot(vb["mean"], y, marker=mk, markersize=5.3, color=col,
                 markerfacecolor=col if filled else "white", markeredgewidth=1.35, zorder=4)
        if vb["mean"] >= 0:
            ax1.text(vb["ci_high"] + 0.010, y, f"+{vb['mean']:.3f}", ha="left", va="center", fontsize=7.0, color=INK)
        else:
            ax1.text(vb["ci_low"] - 0.010, y, f"\u2212{abs(vb['mean']):.3f}", ha="right", va="center", fontsize=7.0, color=FIXED)
    ax1.annotate("Coarse-belief\nfragility", xy=(-0.050, 1.0), xytext=(0.14, 0.9),
                 ha="left", va="center", fontsize=6.9, color=FIXED,
                 arrowprops=dict(arrowstyle="->", color=FIXED, linewidth=0.75))
    ax1.set_yticks(ys)
    ax1.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax1.set_xlim(-0.19, 0.485)
    ax1.set_xlabel("True ESG vs. baseline EB_B (SD)")
    ax1.set_title("(a) Online belief learning (§30, zero leakage)", loc="left", fontsize=8.5, color=INK)
    ax1.grid(axis="x", color=GRID, linewidth=0.5)
    ax1.set_axisbelow(True)

    # Panel (b): Within-tier inflating rates (%)
    x = [0, 1, 2]
    tiers = ("wash_share_large", "wash_share_mid", "wash_share_small")
    series = [
        ("EB_B", "Baseline (EB_B)", INK, ":", "o", True),
        ("EB_V", "Value target (EB_V)", FIXED, "-", "o", True),
        ("EB_F16_bayes", "Choice 16× (EB_F16)", INFO, "--", "D", False),
        ("EB_F100_bayes_smooth", "Hysteresis (EB_F100_sm)", PROP, "-", "D", True),
    ]
    for key, label, col, ls, mk, filled in series:
        vals = [100.0 * s30[key][t] for t in tiers]
        ax2.plot(x, vals, color=col, linestyle=ls, linewidth=1.5 if filled else 1.25,
                 marker=mk, markersize=5.1, markerfacecolor=col if filled else "white",
                 markeredgewidth=1.3, label=label, zorder=4 if filled else 3)
    ax2.annotate("Small: 6.5% → 23.4%\n(no formula known)",
                 xy=(2.0, 23.4 + 1.5), xytext=(1.42, 46.5),
                 ha="center", va="bottom", fontsize=6.9, color=FIXED,
                 arrowprops=dict(arrowstyle="->", color=FIXED, linewidth=0.75))
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Large (10%)", "Mid (30%)", "Small (60%)"])
    ax2.set_xlim(-0.25, 2.25)
    ax2.set_ylim(-2, 85)
    ax2.set_xlabel("Firm size tier")
    ax2.set_ylabel("Within-tier inflating rate (%)")
    ax2.set_title("(b) Tier displacement & hysteresis cure", loc="left", fontsize=8.5, color=INK)
    grid_y(ax2)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none",
               framealpha=0.92, fontsize=6.7, handletextpad=0.35)

    fig.tight_layout(w_pad=1.35)
    save(fig, "fig_bounded",
         "Zero-leakage online empirical-belief bounded rationality (§30, seeds 72001–72030, 30 paired seeds, paired 95% "
         "intervals). Firms never read post-switch policy parameters, updating audit, fine and assurance beliefs strictly "
         "from realised events via discounted Beta-Binomial posteriors and Thompson sampling. (a) Assurance hysteresis "
         "matches transparency I (+0.357 vs +0.355 SD), whereas coarse-belief Stackelberg S_obs degrades below EB_B "
         "(−0.050 SD). (b) Value targeting spontaneously displaces greenwashing from large to small firms (L2a/L2b).")


def fig_generalization(mas: dict) -> None:
    """Cross-regime (empirical Q, §32A) and cross-domain (non-ESG ai_washing, §32C-2) generalization."""
    s32a = mas["studies"]["sec32A_q_dynamics"]["scenarios"]
    s32c2 = mas["studies"]["sec32C2_ai_washing"]["scenarios"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 3.5), gridspec_kw={"width_ratios": [1.02, 1.08]})

    # Panel (a): Empirical persistence Q(0.505, 0.423) (§32A)
    qi = s32a["Q_I"]["vs_baseline_eq"]["mean"]
    rows_q = [
        ("R100 + hyst. ($\\gamma$=0.7)", "Q_R100_bayes_smooth", PROP, "o", True),
        ("F100 + hyst. (choice)", "Q_F100_bayes_smooth", PROP, "D", True),
        ("Stackelberg $S_{\\mathrm{obs}}$", "Q_S_obs", INFO, "o", True),
        ("R16 assigned", "Q_R16_bayes", PROP, "o", False),
        ("Information ($\\tau$=0.6)", "Q_I", INFO, "s", False),
        ("$S_{\\mathrm{obs}}^{\\mathrm{sig}}$ (noisy $\\Delta r$)", "Q_S_obs_sig", INK_2, "s", True),
        ("F16 choice", "Q_F16_bayes", FIXED, "D", False),
        ("$S_{\\mathrm{obs}}^{\\mathrm{hid}}$ (88% hit)", "Q_S_obs_hid", INK_2, "o", False),
        ("Jump target (J)", "Q_J", FIXED, "s", True),
        ("Value target (V)", "Q_V", FIXED, "o", True),
    ]
    yq = list(range(len(rows_q) - 1, -1, -1))
    ax1.axvline(0.0, color=INK_2, linewidth=0.6)
    ax1.axvline(qi, color=INFO, linestyle="--", linewidth=0.95, zorder=1)
    ax1.text(qi + 0.002, 0.25, f"$Q_I$\n+{qi:.3f}", ha="left", va="bottom", fontsize=6.7, color=INFO)
    for y, (label, key, col, mk, filled) in zip(yq, rows_q):
        vb = s32a[key]["vs_baseline_eq"]
        ax1.hlines(y, vb["ci_low"], vb["ci_high"], color=col, linewidth=1.4, zorder=2)
        ax1.plot(vb["mean"], y, marker=mk, markersize=4.9, color=col,
                 markerfacecolor=col if filled else "white", markeredgewidth=1.3, zorder=4)
        if vb["mean"] >= 0:
            ax1.text(vb["ci_high"] + 0.002, y, f"+{vb['mean']:.3f}", ha="left", va="center", fontsize=6.7, color=INK,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.3))
        else:
            ax1.text(vb["ci_low"] - 0.002, y, f"\u2212{abs(vb['mean']):.3f}", ha="right", va="center", fontsize=6.7, color=FIXED)
    ax1.set_yticks(yq)
    ax1.set_yticklabels([r[0] for r in rows_q], fontsize=7.1)
    ax1.set_xlim(-0.050, 0.098)
    ax1.set_xlabel("True ESG vs. Q_B (SD)")
    ax1.set_title("(a) Empirical persistence $Q$ (§32A)", loc="left", fontsize=8.4, color=INK)
    ax1.grid(axis="x", color=GRID, linewidth=0.5)
    ax1.set_axisbelow(True)

    # Panel (b): Non-ESG domain ai_washing (§32C-2), full n=30 + clean n=15 overlaid
    ai_i = s32c2["AI_I"]["vs_baseline_eq"]["mean"]
    rows_ai = [
        ("F100 + hyst. (choice)", "AI_F100_bayes_smooth", PROP, "D"),
        ("R16 assigned", "AI_R16_bayes", PROP, "o"),
        ("Information ($\\tau$=0.6)", "AI_I", INFO, "s"),
        ("F16 choice", "AI_F16_bayes", FIXED, "D"),
        ("$S_{\\mathrm{obs}}^{\\mathrm{sig}}$ (1×)", "AI_S_obs_sig", INFO, "s"),
        ("Stackelberg $S_{\\mathrm{obs}}$", "AI_S_obs", INFO, "o"),
        ("$S_{\\mathrm{obs}}^{\\mathrm{hid}}$ (83% hit)", "AI_S_obs_hid", INK_2, "o"),
        ("Value target (V)", "AI_V", FIXED, "o"),
    ]
    yai = list(range(len(rows_ai) - 1, -1, -1))
    ax2.axvline(0.0, color=INK_2, linewidth=0.6)
    ax2.axvline(ai_i, color=INFO, linestyle="--", linewidth=0.95, zorder=1)
    ax2.text(ai_i + 0.012, 0.2, f"AI_I\n+{ai_i:.3f}", ha="left", va="bottom", fontsize=6.7, color=INFO)
    for y, (label, key, col, mk) in zip(yai, rows_ai):
        vb = s32c2[key]["vs_baseline_eq"]
        vbc = s32c2[key]["vs_baseline_eq_clean"]
        w = 100.0 * s32c2[key]["wash_share"]
        # Full n=30 slightly above center, clean n=15 slightly below
        ax2.hlines(y + 0.11, vb["ci_low"], vb["ci_high"], color=col, linewidth=1.4, zorder=3)
        ax2.plot(vb["mean"], y + 0.11, marker=mk, markersize=4.9, color=col,
                 markerfacecolor=col, markeredgewidth=1.1, zorder=4)
        ax2.hlines(y - 0.13, vbc["ci_low"], vbc["ci_high"], color=col, linewidth=0.95, alpha=0.65, zorder=2)
        ax2.plot(vbc["mean"], y - 0.13, marker=mk, markersize=4.1, color=col,
                 markerfacecolor="white", markeredgewidth=1.05, zorder=4)
        x_txt = max(vb["ci_high"], vbc["ci_high"]) + 0.015
        ax2.text(x_txt, y, f"({w:.1f}%)", ha="left", va="center", fontsize=6.6, color=INK_2)

    ax2.plot([], [], marker="o", color=INK, linestyle="none", markersize=4.8, label="All $n=30$")
    ax2.plot([], [], marker="o", color=INK, markerfacecolor="white", linestyle="none", markersize=4.3, label="Clean $n=15$")
    ax2.set_yticks(yai)
    ax2.set_yticklabels([r[0] for r in rows_ai], fontsize=7.1)
    ax2.set_xlim(-0.07, 0.72)
    ax2.set_xlabel("True level vs. AI_B (SD); (wash %)")
    ax2.set_title("(b) Non-ESG domain: ai_washing (§32C-2)", loc="left", fontsize=8.4, color=INK)
    ax2.grid(axis="x", color=GRID, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none",
               framealpha=0.92, fontsize=6.7, handletextpad=0.25)

    fig.tight_layout(w_pad=1.4)
    save(fig, "fig_generalization",
         "Cross-regime and cross-domain generalization (30 paired seeds each, paired 95% intervals). (a) Under empirically "
         "estimated persistence Q(0.505, 0.423) (§32A, seeds 74001–74030, 0 core-val failures), pure-history S_obs beats "
         "transparency Q_I (+0.012 SD) and assurance hysteresis achieves >3× Q_I's gain; noisy score filtering S_obs_sig "
         "degrades below S_obs (−0.017 SD) under high exogenous noise (sigma_r = 0.351). (b) In the non-ESG ai_washing "
         "domain (§32C-2, seeds 77001–77030; baseline wash 65.9%), all 6 primary contrasts hold identically across full "
         "n=30 (filled) and clean n=15 without price-cap hits (hollow), with hysteresis cutting washing to 4.4%.")


def main() -> None:
    stage_f = load(STAGE_F)
    decay = load(DECAY)
    audits = load(AUDITS)
    ass = load(ASSURANCE)
    rob = load(ROBUST)
    joint = load(JOINT)
    h3 = load(STAGE_H3)
    stage_i = load(STAGE_I)
    mas = load(MAS_UPGRADE)
    fig_decay(decay)
    fig_frontier(stage_f)
    fig_threshold(stage_f)
    fig_size(stage_f)
    fig_targeted(audits)
    fig_assurance(ass)
    fig_regimes(stage_f, rob, joint, h3, stage_i)
    fig_stackelberg(mas)
    fig_remedies(mas)
    fig_bounded(mas)
    fig_generalization(mas)
    print(f"wrote figures to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
