"""Spectral shrinkage of ES / EGGROLL gradient estimates (core functions)."""
import numpy as np


def probes(rng, N, m, n, kind="rank1", r=1):
    """Return N perturbations as factors (A, B) with E_j = A_j B_j^T / sqrt(r), or a dense array."""
    if kind == "dense":
        return rng.standard_normal((N, m, n))
    return rng.standard_normal((N, m, r)), rng.standard_normal((N, n, r))


def measure(G, P):
    """Directional derivatives y_j = <E_j, G> (what an antithetic pair returns on a quadratic)."""
    if isinstance(P, tuple):
        A, B = P
        return sum(((A[:, :, k] @ G) * B[:, :, k]).sum(1) for k in range(A.shape[2])) / np.sqrt(A.shape[2])
    return np.einsum("jmn,mn->j", P, G)


def es_estimate(y, P):
    """Standard ES / EGGROLL estimate (1/N) sum_j y_j E_j."""
    if isinstance(P, tuple):
        A, B = P
        return sum((A[:, :, k] * y[:, None]).T @ B[:, :, k] for k in range(A.shape[2])) / (len(y) * np.sqrt(A.shape[2]))
    return np.einsum("j,jmn->mn", y, P) / len(y)


def shrinker(yv, beta):
    """Optimal Frobenius shrinker of Gavish & Donoho (2017) for unit noise, aspect ratio beta<=1."""
    out = np.zeros_like(yv)
    k = yv > 1 + np.sqrt(beta)
    out[k] = np.sqrt((yv[k] ** 2 - beta - 1) ** 2 - 4 * beta) / yv[k]
    return out


def shrink(Ghat, nu, scale=1.0, mode="opt", rank=None, bulk=0.0):
    """Singular-value shrinkage of Ghat when the noise entries have standard deviation nu.
    mode: 'opt' (optimal shrinker), 'hard' (keep values above the bulk edge), 'rank' (keep top `rank`).
    `scale` inflates the noise level (used for rank-one probes).
    `bulk`: weight given to the singular directions below the noise edge (0 = remove them;
    N/(N+d) = the best scalar shrinkage of the plain ES estimate, 'two-level shrinkage')."""
    m, n = Ghat.shape
    U, s, Vt = np.linalg.svd(Ghat, full_matrices=False)
    big, beta = max(m, n), min(m, n) / max(m, n)
    yv = s / (scale * nu * np.sqrt(big))
    if mode == "opt":
        s2 = shrinker(yv, beta) * scale * nu * np.sqrt(big)
        if bulk == "auto":   # plain ES step with the best scalar shrinkage when nothing is detected
            bulk = 0.0 if (s2 > 0).any() else 1.0
        s2 = np.where(s2 > 0, s2, bulk * s)
    elif mode == "hard":
        s2 = np.where(yv > 1 + np.sqrt(beta), s, 0.0)
    else:
        s2 = np.where(np.arange(len(s)) < rank, s, 0.0)
    return (U * s2) @ Vt


def predicted_risk(sv, m, n, N, total=None):
    """Asymptotic relative risk E||shrink(Ghat)-G||^2/||G||^2 for dense Gaussian probes (Theorem 3).
    sv: singular values of G; total: squared norm of the whole gradient that shares the scalars."""
    g2 = (sv ** 2).sum()
    total = g2 if total is None else total
    big, beta = max(m, n), min(m, n) / max(m, n)
    x2 = sv ** 2 * N / (total * big)
    risk = x2.copy()
    k = x2 > np.sqrt(beta)
    x4 = x2[k] ** 2
    c2 = (x4 - beta) / (x4 + beta * x2[k])
    ct2 = (x4 - beta) / (x4 + x2[k])
    risk[k] = x2[k] * (1 - c2 * ct2)
    return risk.sum() * total * big / (N * g2)


def low_rank(rng, m, n, sv):
    U, _ = np.linalg.qr(rng.standard_normal((m, len(sv))))
    V, _ = np.linalg.qr(rng.standard_normal((n, len(sv))))
    return (U * np.asarray(sv)) @ V.T


def flip_scale(y, P, nu, rng, k=3, safety=1.1):
    """Sign-flip calibration: flipping the signs of y_j removes the signal but keeps the noise law, so the
    top singular value of the flipped estimate measures the noise edge. Returns the inflation factor."""
    m, n = (P[0].shape[1], P[1].shape[1]) if isinstance(P, tuple) else P.shape[1:]
    top = max(np.linalg.norm(es_estimate(y * rng.choice([-1.0, 1.0], len(y)), P), 2) for _ in range(k))
    return max(1.0, safety * top / (nu * (np.sqrt(m) + np.sqrt(n))))


def niht(y, P, rank, iters=100, tol=1e-8):
    """Normalized iterative hard thresholding with known rank (stepsize rule of Tanner & Wei 2013),
    the rank-aware recovery of Allen et al. (2026), run on the SAME probes and scalars:
    G <- Proj_r(G + eta * D), D = (1/N) sum_j res_j E_j, eta = line search on the current column space."""
    N = len(y)
    D = es_estimate(y, P)
    U = np.linalg.svd(D, full_matrices=False)[0][:, :rank]
    G = np.zeros_like(D)
    y2 = (y ** 2).mean()
    for k in range(iters):
        res = y - measure(G, P)
        if (res ** 2).mean() < tol * y2:
            break
        D = es_estimate(res, P)
        DU = U @ (U.T @ D)
        eta = (DU ** 2).sum() / ((measure(DU, P) ** 2).mean() + 1e-300)
        Uf, s, Vt = np.linalg.svd(G + eta * D, full_matrices=False)
        U = Uf[:, :rank]
        G = (U * s[:rank]) @ Vt[:rank]
    return G
