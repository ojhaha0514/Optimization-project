"""Experiment 4: gradient spectra and shrinkage risk in a wider network (64-512-512-10), using exact
directional derivatives y_j = sum_l a_l^T G_l b_l (the sigma -> 0 limit of an antithetic pair)."""
import numpy as np, json
import exp_nn
from shrink import *

exp_nn.h = 512
X, y = exp_nn.data()
rng = np.random.default_rng(3)
Ws = exp_nn.init(0)
Ns = [1024, 4096, 16384, 65536]
out = {"Ns": Ns, "states": {}}


def state(tag):
    G = exp_nn.grad(Ws, X, y); gall = sum((g ** 2).sum() for g in G); d = sum(g.size for g in G)
    res = {"loss": float(exp_nn.loss(Ws, X, y)), "d": d, "layers": []}
    for l in range(3):
        m, n = G[l].shape; sv = np.linalg.svd(G[l], compute_uv=False)
        res["layers"].append({"shape": [m, n], "share": float((sv ** 2).sum() / gall), "srank": float((sv ** 2).sum() / sv[0] ** 2),
                              "Nstar": float(np.sqrt(m * n) * gall / sv[0] ** 2), "sv2": (sv[:30] ** 2 / gall).tolist(),
                              "es": [], "shrink": [], "pred": [float(predicted_risk(sv, m, n, N, total=gall)) for N in Ns]})
    res["cos"] = {"es": [], "shrink": [], "two": []}
    for N in Ns:
        A = [rng.standard_normal((N, g.shape[0])) for g in G]; B = [rng.standard_normal((N, g.shape[1])) for g in G]
        yv = sum(((A[l] @ G[l]) * B[l]).sum(1) for l in range(3))
        nu = np.sqrt((yv ** 2).mean() / N); kap = 1 + d / N
        num = {k: 0.0 for k in res["cos"]}; den = {k: 0.0 for k in res["cos"]}
        for l in range(3):
            P = (A[l][:, :, None], B[l][:, :, None])
            U = es_estimate(yv, P); sc = flip_scale(yv, P, nu, rng)
            S = shrink(U, nu, scale=sc); T2 = shrink(U, nu, scale=sc, bulk=1 / kap)
            g2 = (G[l] ** 2).sum()
            res["layers"][l]["es"].append(float(((U - G[l]) ** 2).sum() / g2))
            res["layers"][l]["shrink"].append(float(((S - G[l]) ** 2).sum() / g2))
            for k, V in [("es", U), ("shrink", S), ("two", T2)]:
                num[k] += (V * G[l]).sum(); den[k] += (V ** 2).sum()
        for k in num:
            res["cos"][k].append(float(num[k] / np.sqrt(gall * den[k] + 1e-300)))
        print(tag, "N", N, "d/N %.1f" % (d / N), "cos", {k: round(v[-1], 3) for k, v in res["cos"].items()},
              "| layer risks ES", [round(L["es"][-1], 2) for L in res["layers"]], "shrunk", [round(L["shrink"][-1], 3) for L in res["layers"]],
              "pred", [round(L["pred"][len(res["cos"]["es"]) - 1], 3) for L in res["layers"]], flush=True)
    print(tag, "loss %.4f" % res["loss"], "N*", [round(L["Nstar"]) for L in res["layers"]], "srank", [round(L["srank"], 1) for L in res["layers"]], flush=True)
    out["states"][tag] = res


state("init")
for t in range(100):
    Ws = [W - 0.3 * g for W, g in zip(Ws, exp_nn.grad(Ws, X, y))]
    if t + 1 in (10, 100):
        state("gd%d" % (t + 1))
json.dump(out, open("results_nn_wide.json", "w"))
