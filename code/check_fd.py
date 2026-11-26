"""Check: antithetic finite differences (sigma = 1e-3) against exact directional derivatives, small network."""
import numpy as np, exp_nn
X, y = exp_nn.data(); Ws = exp_nn.init(0); rng = np.random.default_rng(0)
for sigma in [1e-3, 1e-2, 1e-1]:
    exp_nn.sigma = sigma
    yv, P = exp_nn.estimates(Ws, X, y, rng, n_pairs=256)
    G = exp_nn.grad(Ws, X, y)
    ex = sum(((P[l][0][:, :, 0] @ G[l]) * P[l][1][:, :, 0]).sum(1) for l in range(3))
    print("sigma", sigma, "relative l2 difference", np.linalg.norm(yv - ex) / np.linalg.norm(ex))
