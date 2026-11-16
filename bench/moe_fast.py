import mlx.core as mx, time, sys, os, socket

D   = int(os.environ.get("MOE_D", 2048))
FF  = int(os.environ.get("MOE_FF", 4096))
E   = int(os.environ.get("MOE_E", 32))
TOPK= int(os.environ.get("MOE_K", 2))
T   = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
IT  = int(os.environ.get("MOE_IT", 10))
REPS= int(os.environ.get("MOE_REPS", 5))

mx.random.seed(0)
W1 = (mx.random.normal((E, D, FF)) * 0.02).astype(mx.float32)
W2 = (mx.random.normal((E, FF, D)) * 0.02).astype(mx.float32)
X  = mx.random.normal((T, D)).astype(mx.float32)
Wr = mx.random.normal((D, E)).astype(mx.float32)
mx.eval(W1, W2, X, Wr)
FL = T * TOPK * 2 * 2 * D * FF

def route(X):
    return mx.argpartition(-(X @ Wr), TOPK, axis=-1)[:, :TOPK]

def moe_fast(X, idx=None):
    e = (route(X) if idx is None else idx).reshape(-1)
    order = mx.argsort(e)
    es = mx.take(e, order)
    tok = order // TOPK
    xs = mx.take(X, tok, axis=0).reshape(-1, 1, D)
    h = mx.maximum(mx.gather_mm(xs, W1, rhs_indices=es, sorted_indices=True), 0.0)
    y = mx.gather_mm(h, W2, rhs_indices=es, sorted_indices=True).reshape(-1, D)
    return mx.zeros((T, D), X.dtype).at[tok].add(y)

def moe_baseline(X, idx=None):
    idx = route(X) if idx is None else idx
    o = mx.zeros_like(X)
    for k in range(TOPK):
        e = idx[:, k]
        order = mx.argsort(e)
        es, xs = e[order], X[order]
        mx.eval(es)
        el = es.tolist()
        bounds, j = [0] * (E + 1), 0
        for ei in range(E):
            while j < len(el) and el[j] == ei: j += 1
            bounds[ei + 1] = j
        for ei in range(E):
            a, b = bounds[ei], bounds[ei + 1]
            if b <= a: continue
            o[order[a:b]] += mx.maximum(xs[a:b] @ W1[ei], 0.0) @ W2[ei]
    return o

def bound_balanced_bmm(X, idx=None):
    xb = mx.concatenate([X.reshape(E, T // E, D)] * TOPK, axis=1)
    return (mx.maximum(xb @ W1, 0.0) @ W2).reshape(-1, D)

def bench(f, idx):
    mx.eval(f(X, idx)); mx.eval(f(X, idx))
    reps = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for _ in range(IT): mx.eval(f(X, idx))
        reps.append((time.perf_counter() - t0) / IT)
    return min(reps), max(reps)

idx = route(X); mx.eval(idx)
ref = moe_baseline(X, idx); got = moe_fast(X, idx)
mx.eval(ref, got)
den = float(mx.linalg.norm(ref, stream=mx.cpu))
relerr = float(mx.linalg.norm(got - ref, stream=mx.cpu)) / den

print(f"{socket.gethostname()}  D={D} FF={FF} E={E} k={TOPK} T={T} "
      f"rows/expert~{T*TOPK//E}  useful={FL/1e9:.1f} GFLOP  relerr={relerr:.1e}")
for f in (moe_baseline, moe_fast, bound_balanced_bmm):
    lo, hi = bench(f, idx)
    print(f"  {f.__name__:20s} {lo*1e3:8.2f} ms (worst {hi*1e3:8.2f})  {FL/lo/1e9:7.0f} GF/s")
