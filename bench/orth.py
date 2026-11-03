import mlx.core as mx, time, sys, json, platform, os

def ridge_adjusted_normal_matrix(X, ridge):
    G = X.T @ X
    return G + mx.eye(G.shape[0], dtype=G.dtype) * ridge

def m_cpu_chol(X, G, **kw):
    L = mx.linalg.cholesky(G, stream=mx.cpu)
    return mx.linalg.solve_triangular(L, X.T, upper=False, stream=mx.cpu).T

def m_cpu_hybrid(X, G, **kw):
    L = mx.linalg.cholesky(G, stream=mx.cpu)
    Li = mx.linalg.tri_inv(L, upper=False, stream=mx.cpu)
    return X @ Li.T

def invsqrt_ns(A, iters):
    R = A.shape[0]
    d = mx.rsqrt(mx.diag(A))
    S = A * d[:, None] * d[None, :]
    c = mx.sqrt(mx.sum(S * S))
    Y = S / c
    I = mx.eye(R, dtype=A.dtype)
    Z = I
    I3 = I * 3.0
    for _ in range(iters):
        T = 0.5 * (I3 - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return d[:, None] * (Z * mx.rsqrt(c))

def m_ns(X, G, iters=18, **kw):
    return X @ invsqrt_ns(G, iters)

def rchol(A, b):
    R = A.shape[0]
    if R <= b:
        L = mx.linalg.cholesky(A, stream=mx.cpu)
        return L, mx.linalg.tri_inv(L, upper=False, stream=mx.cpu)
    h = R // 2
    L11, L11i = rchol(A[:h, :h], b)
    L21 = A[h:, :h] @ L11i.T
    L22, L22i = rchol(A[h:, h:] - L21 @ L21.T, b)
    Z = mx.zeros((h, R - h), dtype=A.dtype)
    L = mx.concatenate([mx.concatenate([L11, Z], axis=1),
                        mx.concatenate([L21, L22], axis=1)], axis=0)
    Li21 = -(L22i @ (L21 @ L11i))
    Li = mx.concatenate([mx.concatenate([L11i, Z], axis=1),
                         mx.concatenate([Li21, L22i], axis=1)], axis=0)
    return L, Li

def m_bchol(X, G, b=256, **kw):
    L, Li = rchol(G, b)
    return X @ Li.T

def rchol_ns(A, b, iters):
    R = A.shape[0]
    if R <= b:
        M = invsqrt_ns(A, iters)
        return None, M
    h = R // 2
    _, L11i = rchol_ns(A[:h, :h], b, iters)
    L21 = A[h:, :h] @ L11i.T
    _, L22i = rchol_ns(A[h:, h:] - L21 @ L21.T, b, iters)
    Z = mx.zeros((h, R - h), dtype=A.dtype)
    Li21 = -(L22i @ (L21 @ L11i))
    Li = mx.concatenate([mx.concatenate([L11i, Z], axis=1),
                         mx.concatenate([Li21, L22i], axis=1)], axis=0)
    return None, Li

def m_bchol_ns(X, G, b=256, iters=18, **kw):
    _, Li = rchol_ns(G, b, iters)
    return X @ Li.T

def m_cgs2(X, G, b=512, iters=8, ridge=0.0, **kw):
    Xt = X.T
    R, n = Xt.shape
    out = []
    Qp = None
    for k in range(0, R, b):
        W = Xt[k:k + b, :]
        if Qp is not None:
            W = W - (W @ Qp.T) @ Qp
            W = W - (W @ Qp.T) @ Qp
        g = W @ W.T + mx.eye(W.shape[0], dtype=W.dtype) * ridge
        Qb = invsqrt_ns(g, iters).T @ W
        out.append(Qb)
        Qp = Qb if Qp is None else mx.concatenate([Qp, Qb], axis=0)
    return Qp.T

METHODS = {
    "cpu_chol": m_cpu_chol,
    "cpu_hybrid": m_cpu_hybrid,
    "bchol": m_bchol,
    "bchol_ns": m_bchol_ns,
    "ns": m_ns,
    "cgs2": m_cgs2,
}

def orth_err(Q):
    R = Q.shape[1]
    E = Q.T @ Q - mx.eye(R, dtype=Q.dtype)
    return float(mx.sqrt(mx.sum(E * E)) / R)

def run(n, R, names, reps, ridge_rel, kw):
    mx.random.seed(0)
    X = mx.random.normal((n, R)).astype(mx.float32)
    mx.eval(X)
    G0 = X.T @ X
    mx.eval(G0)
    ridge = float(mx.mean(mx.diag(G0))) * ridge_rel
    calls = [("normal_matrix_only", lambda: ridge_adjusted_normal_matrix(X, ridge))]
    for nm in names:
        f = METHODS[nm]
        a = dict(kw); a["ridge"] = ridge
        calls.append((nm, (lambda f=f, a=a: f(X, ridge_adjusted_normal_matrix(X, ridge), **a))))
    errs = {}
    for nm, c in calls:
        Q = c()
        mx.eval(Q)
        errs[nm] = orth_err(Q) if nm != "normal_matrix_only" else float("nan")
        del Q
    best = {nm: float("inf") for nm, _ in calls}
    duty = float(os.environ.get("ORTH_DUTY", "0.0"))
    for _ in range(reps):
        for nm, c in calls:
            t = time.perf_counter()
            mx.eval(c())
            e = time.perf_counter() - t
            if e < best[nm]:
                best[nm] = e
            time.sleep(min(20.0, e * duty))
    return best["normal_matrix_only"], [(nm, best[nm], errs[nm]) for nm, _ in calls]

if __name__ == "__main__":
    n = int(sys.argv[1]); R = int(sys.argv[2])
    names = sys.argv[3].split(",")
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    kw = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    ridge_rel = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-3
    tg, rows = run(n, R, names, reps, ridge_rel, kw)
    host = platform.node()
    for nm, e, err in rows:
        fl = 2.0 * n * R * R
        print(json.dumps({"host": host, "n": n, "R": R, "m": nm, "ms": round(e * 1e3, 3),
                          "err": err, "normal_matrix_ms": round(tg * 1e3, 3),
                          "normal_matrix_gfs": round(fl / tg / 1e9, 1), "kw": kw}))
