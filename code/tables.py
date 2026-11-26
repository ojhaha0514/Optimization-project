"""Write the bodies of the appendix tables (report/tab_*.tex) from the results files."""
import json, numpy as np


def f(x):
    if not np.isfinite(x):
        return "diverged"
    for lo, fmt in [(99.5, "%.0f"), (9.95, "%.1f"), (0.0995, "%.2f"), (0.00995, "%.3f"), (0.000995, "%.4f")]:
        if x >= lo:
            return fmt % x
    e = int(np.floor(np.log10(x)))
    return "$%.1f\\cdot10^{%d}$" % (x / 10 ** e, e)


def write(name, rows):
    open("../report/tab_%s.tex" % name, "w").write("\n".join(" & ".join(r) + r"\\" if r else r"\midrule" for r in rows) + "\n")


r = json.load(open("results_risk.json")); Ns = r["Ns"]; d = r["m"] * r["n"]
idx = [Ns.index(N) for N in (60, 160, 400, 1000, 2500, 10000)]
write("flip", [["dense probes, no calibration"] + [f(r["dense"]["opt"][i]) for i in idx],
               ["dense probes, calibrated"] + [f(r["dense"]["flip"][i]) for i in idx],
               ["rank-4 probes, no calibration"] + [f(r["rank4"]["opt"][i]) for i in idx],
               ["rank-4 probes, calibrated"] + [f(r["rank4"]["flip"][i]) for i in idx],
               ["rank-1 probes, no calibration"] + [f(r["rank1"]["opt"][i]) for i in idx],
               ["rank-1 probes, calibrated"] + [f(r["rank1"]["flip"][i]) for i in idx], [],
               ["Theorem 3 (dense probes)"] + [f(r["pred"][i]) for i in idx],
               ["plain ES, $(d+1)/N$"] + [f((d + 1) / Ns[i]) for i in idx]])

r = json.load(open("results_recovery.json"))
write("recovery", [["$N$ / degrees of freedom"] + ["%.1f" % (N / r["dof"]) for N in r["Ns"]], [],
                   ["plain ES"] + [f(x) for x in r["rank1"]["es"]],
                   ["rank-1 probes: Algorithm 1"] + [f(x) for x in r["rank1"]["one"]],
                   ["rank-1 probes: recovery [3]"] + [f(x) for x in r["rank1"]["niht"]],
                   ["rank-8 probes: Algorithm 1"] + [f(x) for x in r["rank8"]["one"]],
                   ["rank-8 probes: recovery [3]"] + [f(x) for x in r["rank8"]["niht"]]])

t = json.load(open("results_descent.json"))["trace"]; sel = [0, 1, 2, 3, 4, 6, 8, 10]
write("stall", [["loss"] + ["%.2f" % t[i]["loss"] for i in sel], ["share of the top direction"] + ["%.2f" % t[i]["top_share"] for i in sel],
                ["directions kept"] + ["%d" % t[i]["kept"] for i in sel], ["relative error of the estimate"] + ["%.2f" % t[i]["error"] for i in sel]])

w = json.load(open("results_nn_wide.json")); s = w["states"]["gd100"]
write("nn", [["%d ($%d\\times%d$)" % (l + 1, *L["shape"]), "%.0f" % L["Nstar"]] +
             ["%s / %.2f / %.2f" % (f(L["es"][k]), L["shrink"][k], L["pred"][k]) for k in range(len(w["Ns"]))] for l, L in enumerate(s["layers"])])

c = json.load(open("results_nn_curvature.json")); w = json.load(open("results_nn_wide.json"))
write("curv", [[lab, "%.3f" % c[k]["loss"], "%.0f" % c[k]["trH"], "%.2f" % c[k]["lam_max"], "%.3f" % c[k]["lam_g"], "%.0f" % c[k]["T"],
                "%.3f" % ((1 + c[k]["T"]) / 16384), ", ".join("%.0f" % L["Nstar"] for L in w["states"][k]["layers"])]
               for k, lab in [("init", "initialization"), ("gd10", "after 10 steps"), ("gd100", "after 100 steps")]])

t = json.load(open("results_nn_wide_train.json"))


def cell(key):
    cur = np.array(t[key]); v = cur[25]
    if not np.isfinite(v) or v > cur[0]:
        return "diverged"
    return f(v) + ("$^\\dagger$" if cur[5:].max() > 0.5 else "")


rows = [["exact gradient"] + [cell("gd|eta=%g|w=0|N=16384" % e) for e in (0.3, 0.6, 1.2)], []]
rows += [["ES, $N=%d$" % N] + [cell("es|eta=%g|w=0|N=%d" % (e, N)) for e in (0.3, 0.6, 1.2)] for N in (16384, 4096, 1024, 256, 64)]
rows += [[], ["two-level shrinkage ($w=1$), $N=16384$"] + [cell("two|eta=%g|w=1|N=16384" % e) for e in (0.3, 0.6, 1.2)],
         ["shrinkage only ($w=0$), $N=16384$"] + [cell("two|eta=%g|w=0|N=16384" % e) for e in (0.3, 0.6, 1.2)]]
write("train", rows)
r = json.load(open("results_risk.json"))
for k in ["rank1", "rank4"]:
    dev = np.abs(np.array(r[k]["null"]) / 1.1 - np.array(r[k]["edge"])) / np.array(r[k]["edge"])
    print("sign-flip estimate (before the factor 1.1) against the true noise edge,", k, ": largest relative deviation %.3f" % dev.max())
for n in ["flip", "recovery", "stall", "nn", "curv", "train"]:
    print("---", n); print(open("../report/tab_%s.tex" % n).read())
