import mlx.core as mx, time, sys, socket

D, FF, E, TOPK = 2048, 4096, 32, 2
T = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
W1 = (mx.random.normal((E, D, FF)) * 0.02).astype(mx.float32)
W2 = (mx.random.normal((E, FF, D)) * 0.02).astype(mx.float32)
X = mx.random.normal((T, D)).astype(mx.float32)
Wr = mx.random.normal((D, E)).astype(mx.float32)
mx.eval(W1, W2, X, Wr)
FL = T * TOPK * 2 * 2 * D * FF

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

def sort_take_gather_mm(X):
    e = route(X).reshape(-1)
    order = mx.argsort(e)
    es, tok = mx.take(e, order), order // TOPK
    xs = mx.take(X, tok, axis=0).reshape(-1, 1, D)
    h = mx.maximum(mx.gather_mm(xs, W1, rhs_indices=es, sorted_indices=True), 0.0)
    y = mx.gather_mm(h, W2, rhs_indices=es, sorted_indices=True).reshape(-1, D)
    return mx.zeros((T, D), X.dtype).at[tok].add(y)

def route_and_permute_only(X):
    e = route(X).reshape(-1)
    order = mx.argsort(e)
    tok = order // TOPK
    return mx.take(X, tok, axis=0)

def bound_balanced_bmm(X):
    xb = mx.concatenate([X.reshape(E, T // E, D)] * TOPK, axis=1)
    return (mx.maximum(xb @ W1, 0.0) @ W2).reshape(-1, D)

print(f"{socket.gethostname()} T={T} N={T*TOPK} rows/expert~{T*TOPK//E} useful={FL/1e9:.0f} GFLOP")
for f in [baseline_py_bounds, sort_take_gather_mm, bound_balanced_bmm, route_and_permute_only]:
    mx.eval(f(X))
    it = 10
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(it): mx.eval(f(X))
        reps.append((time.perf_counter() - t0) / it)
    el, worst = min(reps), max(reps)
    print(f"  {f.__name__:24s} {el*1e3:8.2f} ms (worst {worst*1e3:7.2f})  {FL/el/1e9:7.0f} GF/s")
