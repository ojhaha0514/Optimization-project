"""Experiment: curvature of the network 64-512-512-10 along training. Measures tr(H), <g,Hg> and the largest
eigenvalue of the Hessian H with Hessian-vector products (central differences of backpropagation gradients),
and the ratio T = tr(H)|g|^2/<g,Hg> of Proposition 5."""
import numpy as np, json
import exp_nn

exp_nn.h = 512
X, y = exp_nn.data()
rng = np.random.default_rng(5)


def dot(A, B):
    return sum((a * b).sum() for a, b in zip(A, B))


def hvp(Ws, V):
    eps = 1e-3 * np.sqrt(dot(Ws, Ws) / dot(V, V))
    gp = exp_nn.grad([W + eps * v for W, v in zip(Ws, V)], X, y)
    gm = exp_nn.grad([W - eps * v for W, v in zip(Ws, V)], X, y)
    return [(a - b) / (2 * eps) for a, b in zip(gp, gm)]


def state(Ws, tag, K=200):
    g = exp_nn.grad(Ws, X, y); g2 = dot(g, g); d = sum(w.size for w in g)
    gHg = dot(g, hvp(Ws, g))
    zs = []
    for _ in range(K):
        Z = [rng.standard_normal(w.shape) for w in Ws]
        zs.append(dot(Z, hvp(Ws, Z)))
    tr, se = float(np.mean(zs)), float(np.std(zs) / np.sqrt(K))
    V = [rng.standard_normal(w.shape) for w in Ws]
    for _ in range(40):
        HV = hvp(Ws, V); lam = dot(V, HV) / dot(V, V)
        nrm = np.sqrt(dot(HV, HV)); V = [h / nrm for h in HV]
    T = tr * g2 / gHg
    r = {"loss": float(exp_nn.loss(Ws, X, y)), "d": d, "g2": float(g2), "trH": tr, "trH_se": se, "lam_g": float(gHg / g2),
         "lam_max": float(lam), "T": float(T), "erank": float(tr / lam)}
    print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return r


out = {}
Ws = exp_nn.init(0)
out["init"] = state(Ws, "init")
for t in range(100):
    Ws = [W - 0.3 * g for W, g in zip(Ws, exp_nn.grad(Ws, X, y))]
    if t + 1 in (10, 100):
        out["gd%d" % (t + 1)] = state(Ws, "after %d GD steps" % (t + 1))
json.dump(out, open("results_nn_curvature.json", "w"))
