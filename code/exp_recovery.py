"""Experiment: one-SVD shrinkage (rank unknown) against normalized IHT with known rank (the rank-aware
estimator of Allen et al. 2026) on the same probes and scalars. G is 200 x 200 with rank 3."""
import numpy as np, json
from shrink import *

m = n = 200; r = 3
rng = np.random.default_rng(0)
G = low_rank(rng, m, n, np.array([3.0, 2.0, 1.0])); g2 = (G ** 2).sum()
out = {"Ns": [500, 1000, 2000, 4000, 8000], "dof": r * (m + n - r), "Nstar": (np.sqrt(m * n) * g2 / np.array([9.0, 4.0, 1.0])).tolist()}
for pr in [1, 8]:
    res = {"es": [], "one": [], "niht": []}
    for N in out["Ns"]:
        acc = {k: [] for k in res}
        for rep in range(3):
            P = probes(rng, N, m, n, "rank1", pr); y = measure(G, P)
            U = es_estimate(y, P); nu = np.sqrt((y ** 2).mean() / N)
            acc["es"].append(((U - G) ** 2).sum() / g2)
            acc["one"].append(((shrink(U, nu, scale=flip_scale(y, P, nu, rng)) - G) ** 2).sum() / g2)
            acc["niht"].append(((niht(y, P, r) - G) ** 2).sum() / g2)
        for k in res:
            res[k].append(float(np.median(acc[k])))
        print("probe rank", pr, "N", N, "N/dof %.2f" % (N / out["dof"]), {k: "%.2e" % res[k][-1] for k in res}, flush=True)
    out["rank%d" % pr] = res
json.dump(out, open("results_recovery.json", "w"))
