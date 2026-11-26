"""Experiment 2: ES descent on quadratics f(W) = 1/2 <W-W*, L (W-W*) R>, with and without shrinkage.
All ES methods use the same N rank-one (EGGROLL) probes per step and exact directional derivatives."""
import numpy as np, json, sys
from shrink import *

m = n = 200
N, T = 4000, 120
kap = 1 + m * n / N


def run(L, R, Ws, method, eta, seed):
    rng = np.random.default_rng(seed)
    W = np.zeros((m, n))
    curve = []
    for t in range(T):
        D = W - Ws
        G = (L[:, None] * D) * R[None, :]
        f = 0.5 * (D * G).sum()
        curve.append(f)
        if not np.isfinite(f) or f > 1e6 * curve[0]:
            curve += [np.inf] * (T - len(curve)); break
        if method == "gd":
            U = G
        else:
            P = probes(rng, N, m, n, "rank1", 1)
            y = measure(G, P)
            U = es_estimate(y, P)
            if method != "es":
                nu = np.sqrt((y ** 2).mean() / N)
                bulk = {"shrink": 0.0, "two": 1 / kap, "auto": "auto"}[method]
                U = shrink(U, nu, scale=flip_scale(y, P, nu, rng), bulk=bulk)
                if method == "auto" and np.linalg.matrix_rank(U, tol=1e-12) == min(m, n):
                    U = U / kap * 1.0
        W = W - eta * U
    return np.array(curve)


if __name__ == "__main__":
    out = {"m": m, "n": n, "N": N}
    rng = np.random.default_rng(0)
    Ws = low_rank(rng, m, n, np.array([3.0, 2.0, 1.0]))
    specs = {"isotropic": (np.ones(m), np.ones(n)),
             "decaying": (1.0 / np.arange(1, m + 1), 1.0 / np.arange(1, n + 1))}
    for name, (L, R) in specs.items():
        D = -Ws; G = (L[:, None] * D) * R[None, :]
        tau, gHg, g2 = L.sum() * R.sum(), (G * ((L[:, None] * G) * R[None, :])).sum(), (G ** 2).sum()
        rho = (1 + tau * g2 / gHg) / N
        print(name, "tr H", tau, "ceiling 1+rho =", 1 + rho, flush=True)
        res = {"rho": rho}
        e0 = 1.0 if name == "isotropic" else 4.0
        grid = {"gd": [1.0, 1.9], "es": [e0 / kap * 2.0 ** k for k in range(-2, 5)],
                "shrink": [e0 * 2.0 ** k for k in range(-3, 3)], "two": [e0 * 2.0 ** k for k in range(-3, 3)],
                "auto": [e0 * 2.0 ** k for k in range(-3, 3)]}
        for method in grid:
            best = None
            for eta in grid[method]:
                c = np.mean([run(L, R, Ws, method, eta, s) for s in range(2)], 0)
                auc = np.log(max(c[-1] / c[0], 1e-300)) if np.isfinite(c[-1]) else np.inf
                print(" ", method, "%.3g" % eta, "final/initial %.3e" % (c[-1] / c[0]),
                      "steps to 1e-2,1e-4:", [int(np.argmax(c < t * c[0])) if (c < t * c[0]).any() else None for t in [1e-2, 1e-4]], flush=True)
                if best is None or auc < best[0]:
                    best = (auc, eta, c)
            res[method] = {"eta": best[1], "curve": (best[2] / best[2][0]).tolist()}
        out[name] = res
    json.dump(out, open("results_descent.json", "w"))
