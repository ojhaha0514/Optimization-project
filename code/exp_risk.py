"""Experiment 1: risk of ES and shrunk ES against the population size; noise covariance of low-rank probes."""
import numpy as np, json
from shrink import *

rng = np.random.default_rng(0)
m, n = 80, 120
d = m * n
sv = np.array([1.0, 0.6, 0.35]); sv /= np.linalg.norm(sv)
G = low_rank(rng, m, n, sv)
Ns = [60, 100, 160, 250, 400, 640, 1000, 1600, 2500, 4000, 6400, 10000]
out = {"m": m, "n": n, "sv": sv.tolist(), "Ns": Ns, "Nstar": (np.sqrt(d) / sv ** 2).tolist()}
reps = 20
for name, kind, pr in [("dense", "dense", 1), ("rank1", "rank1", 1), ("rank4", "rank1", 4)]:
    res = {k: [] for k in ["es", "opt", "flip", "edge", "null"]}
    for N in Ns:
        acc = {k: [] for k in res}
        for _ in range(reps):
            P = probes(rng, N, m, n, kind, pr)
            y = measure(G, P)
            Gh = es_estimate(y, P)
            nu = np.sqrt((y ** 2).mean() / N)
            null = flip_scale(y, P, nu, rng)
            acc["es"].append(((Gh - G) ** 2).sum())
            acc["opt"].append(((shrink(Gh, nu) - G) ** 2).sum())
            acc["flip"].append(((shrink(Gh, nu, scale=null) - G) ** 2).sum())
            acc["edge"].append(np.linalg.norm(Gh - G, 2) / (nu * (np.sqrt(m) + np.sqrt(n))))
            acc["null"].append(null)
        for k in res:
            res[k].append(float(np.mean(acc[k])))
        print(name, N, {k: round(res[k][-1], 3) for k in res}, flush=True)
    out[name] = res
out["pred"] = [float(predicted_risk(sv, m, n, N)) for N in Ns]

# noise covariance of rank-r probes: N E[Z Z^T] = (n+2/r)|G|^2 I + (2n/r+1+2/r) G G^T
cov = {}
mm, nn = 6, 9
Gs = low_rank(rng, mm, nn, np.array([0.9, np.sqrt(1 - 0.81)]))
for pr in [1, 2, 8]:
    K = 400000
    A, B = probes(rng, K, mm, nn, "rank1", pr)
    y = measure(Gs, (A, B))
    T = np.einsum("j,jmr,jnr->jmn", y, A, B, optimize=True) / np.sqrt(pr) - Gs
    emp = np.einsum("jmn,jkn->mk", T, T) / K
    th = (nn + 2 / pr) * np.eye(mm) + (2 * nn / pr + 1 + 2 / pr) * Gs @ Gs.T
    cov[pr] = float(np.abs(emp - th).max() / np.abs(th).max())
    print("probe rank", pr, "max rel. deviation of covariance", cov[pr])
out["cov_check"] = cov
json.dump(out, open("results_risk.json", "w"))
