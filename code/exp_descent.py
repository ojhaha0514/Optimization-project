"""Experiment: ES descent on quadratics f(W) = 1/2 <W-W*, L (W-W*) R> (H = L x R, diagonal), 200 x 200,
rank-3 target, N = 4000 rank-one (EGGROLL) probes per step, exact directional derivatives.
Methods: exact gradient, ES, shrinkage, two-level shrinkage, and (for H = I) recovery by normalized IHT."""
import numpy as np, json, os
from shrink import *

m = n = 200
N, T = 4000, 120
kap = 1 + m * n / N


def run(L, R, Ws, method, eta, seed, steps=T):
    rng = np.random.default_rng(seed)
    W = np.zeros((m, n))
    curve = []
    for t in range(steps):
        D = W - Ws
        G = (L[:, None] * D) * R[None, :]
        f = 0.5 * (D * G).sum()
        curve.append(f)
        if not np.isfinite(f) or f > 1e6 * curve[0]:
            curve += [np.inf] * (steps - len(curve)); break
        if f < 1e-14 * curve[0]:
            curve += [f] * (steps - len(curve)); break
        if method == "gd":
            U = G
        else:
            P = probes(rng, N, m, n, "rank1", 1)
            y = measure(G, P)
            if method == "niht":
                U = niht(y, P, 3, iters=60, tol=1e-12)
            else:
                U = es_estimate(y, P)
                if method != "es":
                    nu = np.sqrt((y ** 2).mean() / N)
                    U = shrink(U, nu, scale=flip_scale(y, P, nu, rng), bulk=1 / kap if method == "two" else 0.0)
        W = W - eta * U
    return np.array(curve)


def trace(Ws, steps=11, seed=0):
    """Shrinkage only, stepsize 1, H = I: how the residual W_t - W* spreads over more directions."""
    rng = np.random.default_rng(seed)
    W = np.zeros((m, n)); rows = []
    for t in range(steps):
        G = W - Ws; sv = np.linalg.svd(G, compute_uv=False)
        P = probes(rng, N, m, n, "rank1", 1); y = measure(G, P)
        nu = np.sqrt((y ** 2).mean() / N); c = flip_scale(y, P, nu, rng)
        S = shrink(es_estimate(y, P), nu, scale=c)
        rows.append({"step": t, "loss": float(0.5 * (G ** 2).sum()), "top_share": float(sv[0] ** 2 / (sv ** 2).sum()),
                     "kept": int(np.linalg.matrix_rank(S)), "error": float(((S - G) ** 2).sum() / (G ** 2).sum()), "c": float(c)})
        print(rows[-1], flush=True)
        W = W - S
    return rows


if __name__ == "__main__":
    out = json.load(open("results_descent.json")) if os.path.exists("results_descent.json") else {}
    out.update({"m": m, "n": n, "N": N})
    rng = np.random.default_rng(0)
    Ws = low_rank(rng, m, n, np.array([3.0, 2.0, 1.0]))
    specs = {"isotropic": (np.ones(m), np.ones(n)),
             "decaying": (1.0 / np.arange(1, m + 1), 1.0 / np.arange(1, n + 1))}
    for name, (L, R) in specs.items():
        D = -Ws; G = (L[:, None] * D) * R[None, :]
        tau, gHg, g2 = L.sum() * R.sum(), (G * ((L[:, None] * G) * R[None, :])).sum(), (G ** 2).sum()
        res = out.setdefault(name, {})
        res.pop("auto", None)
        res["T"] = float(tau * g2 / gHg); res["rho"] = float((1 + res["T"]) / N)
        res["Nstar"] = float(np.sqrt(m * n) * g2 / np.linalg.norm(G, 2) ** 2)
        print(name, "T = tr(H)|g|^2/<g,Hg> = %.0f" % res["T"], "gain of exact gradient 1+rho = %.2f" % (1 + res["rho"]), "N*_1 = %.0f" % res["Nstar"], flush=True)
        e0 = 1.0 if name == "isotropic" else 4.0
        grid = {"gd": [1.0, 1.9], "es": [e0 / kap * 2.0 ** k for k in range(-2, 5)],
                "shrink": [e0 * 2.0 ** k for k in range(-3, 3)], "two": [e0 * 2.0 ** k for k in range(-3, 3)]}
        if name == "isotropic":
            grid["niht"] = [1.0]
        for method in grid:
            if method in res:
                continue
            best = None
            for eta in grid[method]:
                seeds, steps = (range(1), 12) if method == "niht" else (range(2), T)
                c = np.mean([run(L, R, Ws, method, eta, s, steps) for s in seeds], 0)
                c = np.concatenate([c, np.full(T - len(c), c[-1])])
                auc = np.log(max(c[-1] / c[0], 1e-300)) if np.isfinite(c[-1]) else np.inf
                print(" ", method, "%.3g" % eta, "final/initial %.3e" % (c[-1] / c[0]), "first steps", np.array2string(c[:5] / c[0], precision=3),
                      "steps to 1e-2,1e-4:", [int(np.argmax(c < t * c[0])) if (c < t * c[0]).any() else None for t in [1e-2, 1e-4]], flush=True)
                if best is None or auc < best[0]:
                    best = (auc, eta, c)
            res[method] = {"eta": best[1], "curve": (best[2] / best[2][0]).tolist()}
            json.dump(out, open("results_descent.json", "w"))
    if "trace" not in out:
        out["trace"] = trace(Ws)
    json.dump(out, open("results_descent.json", "w"))
