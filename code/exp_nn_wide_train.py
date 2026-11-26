"""Experiment 5: training the wide network (64-512-512-10) with ES, exact directional derivatives,
N = 16384 rank-one probes per step. Plain steps (W -= eta*U) and normalized steps (W -= eta*U/|U|)."""
import numpy as np, json, sys
import exp_nn
from shrink import *

exp_nn.h = 512
N, T = 16384, 60
X, y = exp_nn.data()


def train(method, eta, w=0.0, seed=0):
    """method: gd | es | two.  two: optimal shrinker on detected directions + weight w on the rest."""
    Ws = exp_nn.init(seed); rng = np.random.default_rng(seed)
    curve = []
    for t in range(T):
        G = exp_nn.grad(Ws, X, y)
        if t % 2 == 0:
            curve.append(float(exp_nn.loss(Ws, X, y)))
            if not np.isfinite(curve[-1]) or curve[-1] > 20:
                curve += [float("inf")] * (T // 2 - len(curve)); break
        if method == "gd":
            U = G
        else:
            A = [rng.standard_normal((N, g.shape[0])) for g in G]; B = [rng.standard_normal((N, g.shape[1])) for g in G]
            yv = sum(((A[l] @ G[l]) * B[l]).sum(1) for l in range(3))
            nu = np.sqrt((yv ** 2).mean() / N)
            U = []
            for l in range(3):
                P = (A[l][:, :, None], B[l][:, :, None]); u = es_estimate(yv, P)
                if method == "two":
                    u = shrink(u, nu, scale=flip_scale(yv, P, nu, rng, k=1), bulk=w)
                U.append(u)
        Ws = [W - eta * u for W, u in zip(Ws, U)]
    return curve


if __name__ == "__main__":
    out = {"N": N, "T": T}
    runs = [("gd", 0.3, 0), ("gd", 1.0, 0), ("gd", 2.4, 0), ("es", 0.15, 0), ("es", 0.3, 0), ("es", 0.6, 0), ("es", 1.2, 0),
            ("es", 2.4, 0), ("two", 1.2, 0.0), ("two", 1.2, 1.0), ("two", 2.4, 0.5)]
    import os
    if os.path.exists("results_nn_wide_train.json"):
        out = json.load(open("results_nn_wide_train.json"))
    for method, eta, w in runs:
        if "%s_%g_%g" % (method, eta, w) in out:
            continue
        c = train(method, eta, w)
        out["%s_%g_%g" % (method, eta, w)] = c
        print(method, "eta %g" % eta, "bulk weight %g" % w, "loss at steps 0,10,..:", np.round(c[::5], 4), flush=True)
        json.dump(out, open("results_nn_wide_train.json", "w"))
