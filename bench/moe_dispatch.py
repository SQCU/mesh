import mlx.core as mx, time, sys, socket

D, FF, E, TOPK = 2048, 4096, 32, 2
T = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
W1 = (mx.random.normal((E, D, FF)) * 0.02).astype(mx.float32)
W2 = (mx.random.normal((E, FF, D)) * 0.02).astype(mx.float32)
X = mx.random.normal((T, D)).astype(mx.float32)
Wr = mx.random.normal((D, E)).astype(mx.float32)
mx.eval(W1, W2, X, Wr)
FL = T * TOPK * 2 * 2 * D * FF
N = T * TOPK

def route(X):
    return mx.argpartition(-(X @ Wr), TOPK, axis=-1)[:, :TOPK]

def baseline_py_bounds(X):
    idx = route(X)
    outs = []
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
        o = mx.zeros_like(X)
        for ei in range(E):
            a, b = bounds[ei], bounds[ei + 1]
            if b <= a: continue
            o[order[a:b]] = mx.maximum(xs[a:b] @ W1[ei], 0.0) @ W2[ei]
        outs.append(o)
    return sum(outs)

def gather_mm_unsorted(X):
    e = route(X).reshape(-1)
    tok = mx.repeat(mx.arange(T), TOPK)
    h = mx.maximum(mx.gather_mm(X.reshape(T, 1, D), W1, lhs_indices=tok, rhs_indices=e), 0.0)
    y = mx.gather_mm(h, W2, rhs_indices=e).reshape(-1, D)
    return mx.zeros((T, D), X.dtype).at[tok].add(y)

def gather_mm_lhs_indices(X):
    e = route(X).reshape(-1)
    order = mx.argsort(e)
    es, tok = mx.take(e, order), order // TOPK
    h = mx.maximum(mx.gather_mm(X.reshape(T, 1, D), W1, lhs_indices=tok, rhs_indices=es, sorted_indices=True), 0.0)
    y = mx.gather_mm(h, W2, rhs_indices=es, sorted_indices=True).reshape(-1, D)
    return mx.zeros((T, D), X.dtype).at[tok].add(y)

def sort_take_gather_mm(X):
    e = route(X).reshape(-1)
    order = mx.argsort(e)
    es, tok = mx.take(e, order), order // TOPK
    xs = mx.take(X, tok, axis=0).reshape(-1, 1, D)
    h = mx.maximum(mx.gather_mm(xs, W1, rhs_indices=es, sorted_indices=True), 0.0)
    y = mx.gather_mm(h, W2, rhs_indices=es, sorted_indices=True).reshape(-1, D)
    return mx.zeros((T, D), X.dtype).at[tok].add(y)

def sort_take_gather_mm_invperm(X):
    e = route(X).reshape(-1)
    order = mx.argsort(e)
    es, tok = mx.take(e, order), order // TOPK
    xs = mx.take(X, tok, axis=0).reshape(-1, 1, D)
    h = mx.maximum(mx.gather_mm(xs, W1, rhs_indices=es, sorted_indices=True), 0.0)
    y = mx.gather_mm(h, W2, rhs_indices=es, sorted_indices=True).reshape(-1, D)
    inv = mx.zeros((N,), mx.int32).at[order].add(mx.arange(N, dtype=mx.int32))
    return mx.take(y, inv, axis=0).reshape(T, TOPK, D).sum(1)

def dense_all_experts(X):
    return (mx.maximum(X.reshape(1, T, D) @ W1, 0.0) @ W2).sum(axis=0)

def bound_balanced_bmm(X):
    xb = mx.concatenate([X.reshape(E, T // E, D)] * TOPK, axis=1)
    return (mx.maximum(xb @ W1, 0.0) @ W2).reshape(-1, D)

CANDS = [baseline_py_bounds, gather_mm_unsorted, gather_mm_lhs_indices, sort_take_gather_mm,
         sort_take_gather_mm_invperm, dense_all_experts, bound_balanced_bmm]

print(f"{socket.gethostname()} T={T} N={N} rows/expert~{N//E} useful={FL/1e9:.0f} GFLOP")
ref = None
for f in CANDS:
    n = f.__name__
    try:
        r = f(X); mx.eval(r)
    except Exception as ex:
        print(f"  {n:30s} FAILED {type(ex).__name__}: {str(ex)[:70]}"); continue
    if ref is None: ref = r
    err = float(mx.max(mx.abs(r - ref)) / (mx.max(mx.abs(ref)) + 1e-9)) if r.shape == ref.shape else float("nan")
    it = 5 if T > 8192 else 10
    reps = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(it): mx.eval(f(X))
        reps.append((time.perf_counter() - t0) / it)
    el = min(reps)
    print(f"  {n:30s} {el*1e3:8.2f} ms  {FL/el/1e9:7.0f} GF/s  relerr {err:.1e}")
