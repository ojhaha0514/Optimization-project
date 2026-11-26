"""Experiment 3: a small tanh network (64-96-96-10) on the `digits` data, EGGROLL rank-one probes on
all three weight matrices, one shared scalar per antithetic pair (finite differences, sigma = 1e-3)."""
import numpy as np, json, sys
from sklearn.datasets import load_digits
from shrink import *

h, Bsz, N, sigma = 96, 256, 1024, 1e-3


def data():
    X, y = load_digits(return_X_y=True)
    X = (X - X.mean(0)) / (X.std() + 1e-9)
    idx = np.random.default_rng(0).permutation(len(y))[:Bsz]
    return X[idx], y[idx]


def init(seed):
    rng = np.random.default_rng(100 + seed)
    return [rng.standard_normal((h, 64)) / 8, rng.standard_normal((h, h)) / np.sqrt(h), rng.standard_normal((10, h)) / np.sqrt(h)]


def ce(Z, y):
    Z = Z - Z.max(-1, keepdims=True)
    return (np.log(np.exp(Z).sum(-1)) - Z[..., np.arange(len(y)), y]).mean(-1)


def loss(Ws, X, y):
    return ce(np.tanh(np.tanh(X @ Ws[0].T) @ Ws[1].T) @ Ws[2].T, y)


def grad(Ws, X, y):
    H1 = np.tanh(X @ Ws[0].T); H2 = np.tanh(H1 @ Ws[1].T); Z = H2 @ Ws[2].T
    P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    P[np.arange(len(y)), y] -= 1; P /= len(y)
    D2 = (P @ Ws[2]) * (1 - H2 ** 2); D1 = (D2 @ Ws[1]) * (1 - H1 ** 2)
    return [D1.T @ X, D2.T @ H1, P.T @ H2]


def members(Ws, X, y, A, B, sig, chunk=128):
    out = []
    for c in range(0, len(A[0]), chunk):
        s = slice(c, c + chunk)
        H = X[None]
        for l in range(3):
            Z = H @ Ws[l].T + sig * (H @ B[l][s][:, :, None]) * A[l][s][:, None, :]
            H = np.tanh(Z) if l < 2 else Z
        out.append(ce(H, y))
    return np.concatenate(out)


def estimates(Ws, X, y, rng, n_pairs=N):
    A = [rng.standard_normal((n_pairs, W.shape[0])) for W in Ws]
    B = [rng.standard_normal((n_pairs, W.shape[1])) for W in Ws]
    yv = (members(Ws, X, y, A, B, sigma) - members(Ws, X, y, A, B, -sigma)) / (2 * sigma)
    P = [(A[l][:, :, None], B[l][:, :, None]) for l in range(3)]
    return yv, P


def updates(yv, P, rng, method, kap):
    nu = np.sqrt((yv ** 2).mean() / len(yv))
    U = [es_estimate(yv, p) for p in P]
    if method == "es":
        return U
    return [shrink(U[l], nu, scale=flip_scale(yv, P[l], nu, rng), bulk=1 / kap if method == "two" else 0.0) for l in range(3)]


def train(method, eta, seed, T):
    X, y = data(); Ws = init(seed); rng = np.random.default_rng(seed)
    d = sum(W.size for W in Ws); kap = 1 + d / N
    curve = []
    for t in range(T):
        if t % 5 == 0:
            curve.append(float(loss(Ws, X, y)))
            if not np.isfinite(curve[-1]) or curve[-1] > 50: 
                curve += [np.inf] * (T // 5 - len(curve)); break
        if method == "gd":
            U = grad(Ws, X, y)
        else:
            yv, P = estimates(Ws, X, y, rng)
            U = updates(yv, P, rng, method, kap)
        Ws = [W - eta * u for W, u in zip(Ws, U)]
    return curve


def diagnostics(Ws, X, y, rng, tag, reps=5):
    G = grad(Ws, X, y); gall = sum((g ** 2).sum() for g in G)
    rows = []
    for l in range(3):
        m, n = G[l].shape
        sv = np.linalg.svd(G[l], compute_uv=False)
        rows.append({"layer": l + 1, "shape": [m, n], "share": float((sv ** 2).sum() / gall),
                     "srank": float((sv ** 2).sum() / sv[0] ** 2), "Nstar": float(np.sqrt(m * n) * gall / sv[0] ** 2),
                     "above": int((sv ** 2 / gall > np.sqrt(m * n) / N).sum()),
                     "pred": float(predicted_risk(sv, m, n, N, total=gall)), "es": [], "shrink": [], "cos_es": [], "cos_shrink": []})
    tot = {"es": [], "shrink": [], "two": []}
    d = sum(g.size for g in G); kap = 1 + d / N
    for _ in range(reps):
        yv, P = estimates(Ws, X, y, rng)
        U = updates(yv, P, rng, "es", kap); S = updates(yv, P, rng, "shrink", kap); T2 = updates(yv, P, rng, "two", kap)
        for l in range(3):
            g2 = (G[l] ** 2).sum()
            rows[l]["es"].append(((U[l] - G[l]) ** 2).sum() / g2); rows[l]["shrink"].append(((S[l] - G[l]) ** 2).sum() / g2)
            rows[l]["cos_es"].append((U[l] * G[l]).sum() / np.sqrt(g2 * (U[l] ** 2).sum()))
            rows[l]["cos_shrink"].append((S[l] * G[l]).sum() / np.sqrt(g2 * (S[l] ** 2).sum() + 1e-300))
        for k, V in [("es", U), ("shrink", S), ("two", T2)]:
            tot[k].append(sum((v * g).sum() for v, g in zip(V, G)) / np.sqrt(gall * sum((v ** 2).sum() for v in V) + 1e-300))
    for r in rows:
        for k in ["es", "shrink", "cos_es", "cos_shrink"]:
            r[k] = float(np.mean(r[k]))
        print(tag, r, flush=True)
    cos = {k: float(np.mean(v)) for k, v in tot.items()}
    print(tag, "loss %.4f |g| %.4f d %d N %d whole-gradient cosine" % (loss(Ws, X, y), np.sqrt(gall), d, N), cos, flush=True)
    return {"rows": rows, "cos": cos, "loss": float(loss(Ws, X, y)), "gnorm": float(np.sqrt(gall))}


if __name__ == "__main__":
    X, y = data(); rng = np.random.default_rng(7)
    out = {"N": N, "diag": {}}
    Ws = init(0)
    out["diag"]["init"] = diagnostics(Ws, X, y, rng, "init")
    for t in range(300):
        Ws = [W - 0.5 * g for W, g in zip(Ws, grad(Ws, X, y))]
        if t + 1 in (30, 300):
            out["diag"]["gd%d" % (t + 1)] = diagnostics(Ws, X, y, rng, "after %d GD steps" % (t + 1))
    json.dump(out, open("results_nn.json", "w"))
    if len(sys.argv) > 1:
        T = int(sys.argv[1]); d = sum(W.size for W in Ws); kap = 1 + d / N
        out["train"] = {"T": T, "kappa": kap}
        for method, grid in [("gd", [0.5]), ("es", [0.5, 1.0, 2.0, 4.0]), ("two", [0.5, 1.0, 2.0, 4.0])]:
            for c in grid:
                eta = c if method == "gd" else (c / kap if method == "es" else c)
                cur = np.mean([train(method, eta, s, T) for s in range(2)], 0)
                out["train"]["%s_%g" % (method, c)] = cur.tolist()
                print(method, c, "loss at steps", np.round(cur[::8], 4), flush=True)
                json.dump(out, open("results_nn.json", "w"))
