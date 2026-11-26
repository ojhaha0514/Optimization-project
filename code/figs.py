"""Make the figures of the report from the results_*.json files."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.4, "legend.fontsize": 7.5})
C = {"es": "#555555", "shrink": "#1f77b4", "two": "#d62728", "gd": "#2ca02c", "r1": "#ff7f0e", "r4": "#9467bd"}

r = json.load(open("results_risk.json"))
Ns = np.array(r["Ns"]); d = r["m"] * r["n"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.35))
a = ax[0]
a.loglog(Ns, r["dense"]["es"], "s", color=C["es"], ms=4, label="ES estimate (MC)")
a.loglog(Ns, (d + 1) / Ns, "-", color=C["es"], lw=0.8, label="$(d+1)/N$")
a.loglog(Ns, r["dense"]["opt"], "o", color=C["shrink"], ms=4, label="shrunk, dense probes (MC)")
a.loglog(Ns, r["pred"], "-", color=C["shrink"], lw=0.8, label="Theorem 3")
a.loglog(Ns, r["rank4"]["flip"], "^", color=C["r4"], ms=4, label="shrunk, rank-4 probes")
a.loglog(Ns, r["rank1"]["flip"], "v", color=C["r1"], ms=4, label="shrunk, rank-1 probes")
for x in r["Nstar"]:
    a.axvline(x, color="k", ls=":", lw=0.7)
a.set_xlabel("population size $N$ (directions)"); a.set_ylabel(r"$E\|\cdot-G\|^2/\|G\|^2$"); a.legend(loc="lower left")
a.set_title("(a) relative error", fontsize=9)
a = ax[1]
a.semilogx(Ns, r["dense"]["edge"], "o-", color=C["shrink"], ms=3, label="dense probes")
a.semilogx(Ns, r["rank4"]["edge"], "^-", color=C["r4"], ms=3, label="rank-4 probes")
a.semilogx(Ns, r["rank1"]["edge"], "v-", color=C["r1"], ms=3, label="rank-1 probes")
a.semilogx(Ns, r["rank1"]["null"], "v--", color=C["r1"], ms=3, mfc="none", lw=0.8, label="rank-1, calibration factor $\\hat c$")
a.axhline(1, color="k", lw=0.6)
a.set_xlabel("population size $N$"); a.set_ylabel(r"$\|\hat G-G\|_{op}\,/\,\nu(\sqrt{m}+\sqrt{n})$"); a.legend()
a.set_title("(b) noise edge", fontsize=9)
fig.tight_layout(); fig.savefig("../report/fig_risk.pdf")

r = json.load(open("results_descent.json"))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.1))
lab = {"gd": "exact gradient", "es": "ES (EGGROLL)", "shrink": "shrinkage only", "two": "two-level shrinkage"}
for a, k, t in zip(ax, ["isotropic", "decaying"], ["(a) $H=I$", "(b) decaying spectrum"]):
    for mth in ["gd", "es", "shrink", "two"]:
        c = np.array(r[k][mth]["curve"]); a.semilogy(np.maximum(c, 1e-12), color=C[mth], label=lab[mth])
    a.set_xlabel("step"); a.set_title(t + ", ceiling $1+\\rho=%.2f$" % (1 + r[k]["rho"]), fontsize=9)
    a.set_ylim(1e-6 if k == "isotropic" else 0.2, 1.5)
ax[0].set_ylabel("$f(W_t)/f(W_0)$"); ax[0].legend()
fig.tight_layout(); fig.savefig("../report/fig_descent.pdf")

r = json.load(open("results_nn_wide.json"))
Ns = np.array(r["Ns"])
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.1))
for a, k, t in zip(ax[:2], ["init", "gd100"], ["(a) at initialization", "(b) after 100 GD steps"]):
    s = r["states"][k]
    a.semilogx(Ns, s["cos"]["es"], "s-", color=C["es"], ms=4, label="ES")
    a.semilogx(Ns, s["cos"]["shrink"], "o-", color=C["shrink"], ms=4, label="shrinkage")
    a.semilogx(Ns, s["cos"]["two"], "^-", color=C["two"], ms=4, label="two-level")
    a.semilogx(Ns, np.sqrt(Ns / (Ns + s["d"])), ":", color=C["es"], label=r"$\sqrt{N/(N+d)}$")
    a.set_ylim(0, 1); a.set_xlabel("population size $N$"); a.set_title(t, fontsize=9)
ax[0].set_ylabel("cosine with true gradient"); ax[0].legend()
a = ax[2]
for l, col in zip(range(3), ["#1f77b4", "#d62728", "#2ca02c"]):
    L = r["states"]["gd100"]["layers"][l]
    a.semilogy(np.arange(1, len(L["sv2"]) + 1), L["sv2"], ".-", color=col, ms=3, label="layer %d (%d$\\times$%d)" % (l + 1, *L["shape"]))
    a.axhline(np.sqrt(L["shape"][0] * L["shape"][1]) / 16384, color=col, ls=":", lw=0.8)
a.set_xlabel("index $i$"); a.set_title("(c) $s_i^2/\\|g\\|^2$ and threshold at $N=16384$", fontsize=9); a.legend()
fig.tight_layout(); fig.savefig("../report/fig_nn.pdf")
print("figures written")
