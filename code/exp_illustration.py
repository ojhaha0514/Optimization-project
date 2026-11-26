"""Data for the illustration (Figure 1): a 400 x 400 rank-2 gradient with smooth singular vectors,
its ES estimate from N = 16000 probes (relative error about 10), and the estimate after shrinkage."""
import json
import numpy as np
from shrink import *

m = n = 400; N = 16000
x = np.linspace(0, 1, m)
U = np.linalg.qr(np.c_[np.sin(1.5 * np.pi * x + 0.3), np.cos(3.2 * np.pi * x)])[0]
V = np.linalg.qr(np.c_[np.cos(2.0 * np.pi * x + 0.5), np.sin(4.1 * np.pi * x)])[0]
sv = np.array([1.0, 0.65]); sv /= np.linalg.norm(sv)
G = (U * sv) @ V.T
out = {"N": N, "m": m, "n": n, "predicted_dense": float(predicted_risk(sv, m, n, N))}
for kind in ["rank1", "dense"]:
    rng = np.random.default_rng(1)
    if kind == "rank1":
        P = probes(rng, N, m, n, "rank1", 1); y = measure(G, P); Gh = es_estimate(y, P)
        nu = np.sqrt((y ** 2).mean() / N); c = flip_scale(y, P, nu, rng)
    else:                                   # dense probes do not fit in memory at once: accumulate in chunks
        Gh = np.zeros((m, n)); y2 = 0.0
        for _ in range(N // 250):
            E = rng.standard_normal((250, m, n)); yy = np.einsum("jmn,mn->j", E, G)
            Gh += np.einsum("j,jmn->mn", yy, E) / N; y2 += (yy ** 2).sum()
        nu = np.sqrt(y2 / N / N); c = 1.0
    S = shrink(Gh, nu, scale=c)
    out[kind] = {"es_error": float(((Gh - G) ** 2).sum()), "shrunk_error": float(((S - G) ** 2).sum()), "c": float(c)}
    print(kind, out[kind], flush=True)
    if kind == "rank1":                     # the figure shows the EGGROLL (rank-one) case
        np.savez_compressed("results_illustration.npz", G=G.astype(np.float16), Gh=Gh.astype(np.float16), S=S.astype(np.float16),
                            sv=np.linalg.svd(Gh, compute_uv=False), edge=c * nu * (np.sqrt(m) + np.sqrt(n)))
json.dump(out, open("results_illustration.json", "w"))
