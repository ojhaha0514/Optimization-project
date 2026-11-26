"""Make the figures of the report from the results files.  Usage: python3 figs.py [1] [2] [3] [A]
Colors: validated categorical palette (blue, orange, aqua for the estimators; green, violet, red for the
probe types), a blue-gray-red diverging map for the heatmaps, gray ink for references."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"      # ES estimate, shrinkage, recovery [3]
GREEN, VIOLET, RED = "#008a17", "#6a4ce2", "#ec4647"      # dense, rank-4, rank-1 probes
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5, "lines.markersize": 4.6,
    "legend.frameon": False, "pdf.fonttype": 42, "axes.titlepad": 5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 0, "ytick.minor.size": 0, "axes.axisbelow": True})
MK = dict(markeredgecolor="white", markeredgewidth=0.5)   # white ring around markers
which = sys.argv[1:] or ["1", "2", "3", "A"]


def title(ax, s):
    ax.set_title(s, loc="left", fontweight="bold")


if "1" in which:   # ------------------------------------------------------------- Figure 1: the idea
    z = np.load("results_illustration.npz"); zj = json.load(open("results_illustration.json"))
    G, Gh, S = (z[k].astype(float) for k in ("G", "Gh", "S"))
    e0, e1 = zj["rank1"]["es_error"], zj["rank1"]["shrunk_error"]; m, n = G.shape
    sv = z["sv"] / float(z["edge"])
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#efefef", RED])
    vmax = np.abs(G).max()
    fig = plt.figure(figsize=(6.5, 1.46))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1.22, 1, 0.045], wspace=0.16, left=0.012, right=0.955, top=0.87, bottom=0.115)
    for k, (M, t, sub) in enumerate([(G, "(a) true gradient $G$", "rank 2, $400\\times400$"),
                                     (Gh, "(b) ES estimate", "relative error %.1f" % e0), (None, "", ""),
                                     (S, "(d) after shrinkage", "relative error %.2f" % e1)]):
        if M is None:
            continue
        ax = fig.add_subplot(gs[k]); im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); title(ax, t); ax.set_xlabel(sub, labelpad=3, color=INK)
        for sp in ax.spines.values():
            sp.set_visible(True)
    cax = fig.add_subplot(gs[4]); cb = fig.colorbar(im, cax=cax, ticks=[-vmax, 0, vmax]); cb.ax.set_yticklabels(["$-$", "0", "$+$"])
    cb.outline.set_linewidth(0.4); cb.outline.set_edgecolor(AXIS); cb.ax.tick_params(length=0)
    ax = fig.add_subplot(gs[2]); pos = ax.get_position(); ax.set_position([pos.x0 + 0.043, pos.y0 + 0.085, pos.width - 0.058, pos.height - 0.085])
    idx = np.arange(1, len(sv) + 1); det = sv > 1
    ax.axhline(1, color=INK2, lw=0.8)
    ax.plot(idx[~det], sv[~det], "o", color="#9c9c9c", ms=2.6, markeredgewidth=0)
    ax.plot(idx[det], sv[det], "o", color=ORANGE, ms=5.2, **MK)
    ax.set_xscale("log"); ax.set_xlim(0.8, 500); ax.set_ylim(0, 1.12 * sv.max()); ax.set_xticks([1, 10, 100]); ax.set_xticklabels(["1", "10", "100"])
    ax.set_xlabel("index $i$", labelpad=1); title(ax, "(c) singular values"); ax.title.set_position((-0.22, 1))
    ax.text(480, 1.04, "noise edge", ha="right", va="bottom", color=INK2, fontsize=6.8)
    ax.text(2.6, sv[1] + 0.03, "the two directions of $G$", ha="left", va="bottom", fontsize=6.8)
    ax.text(3.0, 0.42, "noise", ha="left", va="center", color=INK2, fontsize=6.8)
    fig.savefig("../report/fig_idea.pdf")
    print("fig 1: ES error %.2f, after shrinkage %.3f, top singular values / edge" % (e0, e1), np.round(sv[:4], 2))

if "2" in which:   # ------------------------------------------------------------- Figure 2: error against N
    r = json.load(open("results_risk.json")); Ns = np.array(r["Ns"]); d = r["m"] * r["n"]
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 1.54))
    a = ax[0]
    a.loglog(Ns, (d + 1) / Ns, "-", color=BLUE, lw=1.1); a.loglog(Ns, r["dense"]["es"], "o", color=BLUE, **MK)
    a.loglog(Ns, r["pred"], "-", color=ORANGE, lw=1.1); a.loglog(Ns, r["dense"]["opt"], "o", color=ORANGE, **MK)
    for i, x in enumerate(r["Nstar"]):
        a.axvline(x, color=INK2, lw=0.7, ls=(0, (1, 2))); a.text(x * 1.08, 230, "$N^\\star_%d$" % (i + 1), fontsize=6.8, color=INK2, va="top")
    a.text(1500, 13, "ES estimate", fontsize=6.8); a.text(68, 0.1, "after shrinkage", fontsize=6.8)
    a.plot([], [], "o", color=INK2, label="simulation", **MK); a.plot([], [], "-", color=INK2, lw=1.1, label="theory")
    a.legend(loc="lower left", handlelength=1.3, borderaxespad=0.2, labelspacing=0.2, ncol=2, columnspacing=0.9)
    a.set_ylim(0.012, 300); a.set_ylabel("relative error"); title(a, "(a) dense probes")
    a = ax[1]
    for key, col, mk, lab in [("dense", GREEN, "o", "dense"), ("rank4", VIOLET, "^", "rank 4"), ("rank1", RED, "s", "rank 1")]:
        a.loglog(Ns, r[key]["flip"], mk + "-", color=col, lw=1.1, label=lab, **MK)
    a.legend(loc="lower left", title="probes", title_fontsize=7, handlelength=1.6, borderaxespad=0.2, labelspacing=0.2)
    a.set_ylim(0.03, 1.6); title(a, "(b) error after shrinkage")
    a = ax[2]
    a.axhline(1, color=INK2, lw=0.8)
    for key, col, mk, lab in [("dense", GREEN, "o", "dense"), ("rank4", VIOLET, "^", "rank 4"), ("rank1", RED, "s", "rank 1")]:
        a.semilogx(Ns, r[key]["edge"], mk + "-", color=col, lw=1.1, label=lab, **MK)
    a.text(330, 1.78, "rank 1", fontsize=6.8); a.text(75, 1.52, "rank 4", fontsize=6.8); a.text(Ns[-1], 0.64, "dense", ha="right", fontsize=6.8)
    a.set_ylim(0.55, 2.7); a.set_ylabel("top singular value / edge"); title(a, "(c) size of the noise")
    for a in ax:
        a.set_xlabel("number of probes $N$", labelpad=1)
    fig.tight_layout(pad=0.3, w_pad=0.9); fig.savefig("../report/fig_risk.pdf")
    print("fig 2 written")

if "3" in which:   # ------------------------------------------------------------- Figure 3: does it speed up descent?
    q = json.load(open("results_descent.json"))["isotropic"]; w = json.load(open("results_nn_wide.json")); t = json.load(open("results_nn_wide_train.json"))
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 1.54))
    a = ax[0]; st = np.arange(120)
    for key, col, ls, lw in [("gd", INK2, "-", 1.5), ("niht", AQUA, "-", 1.5), ("es", BLUE, "-", 1.5), ("shrink", ORANGE, "-", 1.5), ("two", ORANGE, (0, (3, 1.5)), 1.2)]:
        a.semilogy(st, np.maximum(np.array(q[key]["curve"]), 1e-9), color=col, ls=ls, lw=lw)
    a.set_ylim(1e-6, 3); a.set_xlim(-3, 120); a.set_xlabel("step", labelpad=1); a.set_ylabel("loss / initial loss")
    a.text(5, 2.2e-6, "exact gradient,\nrecovery [3]:\none step", fontsize=6.8, va="bottom", linespacing=1.0)
    a.text(54, 2.2e-2, "ES", fontsize=6.8); a.text(118, 1.1e-2, "shrinkage", fontsize=6.8, ha="right"); a.text(66, 9e-6, "two-level", fontsize=6.8, ha="left")
    title(a, "(a) quadratic, $H=I$")
    a = ax[1]; Ns = np.array(w["Ns"]); sN = w["states"]["gd100"]
    a.semilogx(Ns, np.sqrt(Ns / (Ns + sN["d"])), "-", color=BLUE, lw=1.0)
    a.semilogx(Ns, sN["cos"]["es"], "o", color=BLUE, **MK); a.semilogx(Ns, sN["cos"]["shrink"], "o-", color=ORANGE, lw=1.1, **MK)
    for L in sN["layers"]:
        a.axvline(L["Nstar"], color=INK2, lw=0.7, ls=(0, (1, 2)))
        a.text(30000, 0.2, "ES", fontsize=6.8); a.text(48000, 0.88, "shrinkage", fontsize=6.8, ha="right")
    a.set_ylim(0, 1); a.set_xlim(480, 90000); a.set_xlabel("number of probes $N$", labelpad=1); a.set_ylabel("cosine with true gradient")
    title(a, "(b) network: direction")
    a = ax[2]; st = np.arange(0, 60, 2)
    for key, col, ls, lw, lab in [("two|eta=1.2|w=0|N=16384", ORANGE, "-", 1.5, "shrinkage"), ("two|eta=1.2|w=1|N=16384", ORANGE, (0, (3, 1.5)), 1.2, "two-level"),
                                  ("gd|eta=1.2|w=0|N=16384", INK2, "-", 1.5, "exact gradient"), ("es|eta=1.2|w=0|N=16384", BLUE, "-", 1.5, "ES")]:
        a.semilogy(st, t[key], color=col, ls=ls, lw=lw, label=lab)
    a.legend(loc="upper right", handlelength=1.7, labelspacing=0.15, borderaxespad=0.1)
    a.set_ylim(2e-3, 3); a.set_xlabel("step", labelpad=1); a.set_ylabel("training loss")
    title(a, "(c) network: training")
    fig.tight_layout(pad=0.3, w_pad=0.9); fig.savefig("../report/fig_descent.pdf")
    print("fig 3 written")

if "A" in which:   # ------------------------------------------------------------- appendix: quadratic with decaying spectrum
    q = json.load(open("results_descent.json"))["decaying"]
    fig, a = plt.subplots(figsize=(3.0, 1.9)); st = np.arange(120)
    for key, col, ls, lw, lab in [("gd", INK2, "-", 1.5, "exact gradient"), ("es", BLUE, "-", 1.5, "ES"), ("shrink", ORANGE, "-", 1.5, "shrinkage"),
                                  ("two", ORANGE, (0, (3, 1.5)), 1.2, "two-level shrinkage")]:
        a.semilogy(st, q[key]["curve"], color=col, ls=ls, lw=lw, label=lab)
    a.set_ylim(0.2, 1.1); a.set_yticks([0.2, 0.3, 0.5, 1.0]); a.set_yticklabels(["0.2", "0.3", "0.5", "1"]); a.set_xlabel("step", labelpad=1); a.set_ylabel("loss / initial loss")
    a.legend(loc="upper right", handlelength=1.8, labelspacing=0.2)
    fig.tight_layout(pad=0.3); fig.savefig("../report/fig_decaying.pdf")
    print("appendix figure written")
