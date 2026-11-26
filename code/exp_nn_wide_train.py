"""Experiment: training the network 64-512-512-10 with ES (rank-one probes on all three matrices, one shared
scalar per probe, exact directional derivatives), plain updates W <- W - eta * U, 60 steps, seed 0.
Methods: gd (exact gradient), es, two (optimal shrinker on detected directions + weight w on the rest;
w = 0 is shrinkage only, w = 1 keeps the undetected part unchanged)."""
import numpy as np, json, os
import exp_nn
from shrink import *

exp_nn.h = 512
T = 60
X, y = exp_nn.data()


def train(method, eta, w=0.0, N=16384, seed=0):
    Ws = exp_nn.init(seed); rng = np.random.default_rng(seed)
    curve = []
    for t in range(T):
        G = exp_nn.grad(Ws, X, y)
        if t % 2 == 0:
            curve.append(float(exp_nn.loss(Ws, X, y)))
            if not np.isfinite(curve[-1]) or curve[-1] > 1e4:
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
    out = json.load(open("results_nn_wide_train.json")) if os.path.exists("results_nn_wide_train.json") else {"T": T}
    runs = [("gd", e, 0, 16384) for e in (0.3, 0.6, 1.0, 1.2, 2.4)]
    runs += [("es", e, 0, N) for N in (64, 256, 1024, 4096) for e in (0.3, 0.6, 1.2)]
    runs += [("es", e, 0, 16384) for e in (0.15, 0.3, 0.6, 1.2, 2.4)]
    runs += [("two", e, w, 16384) for e in (1.2, 0.6, 0.3) for w in (0.0, 1.0)]
    for method, eta, w, N in runs:
        key = "%s|eta=%g|w=%g|N=%d" % (method, eta, w, N)
        if key in out:
            continue
        out[key] = train(method, eta, w, N)
        print(key, "loss at steps 0,10,..,50:", np.array2string(np.array(out[key][::5]), precision=4), flush=True)
        json.dump(out, open("results_nn_wide_train.json", "w"))
