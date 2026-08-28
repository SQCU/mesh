import mlx.core as mx, time, sys, socket

D, FF = 2048, 4096
CONFIGS = [(32, 2), (8, 2), (64, 2), (32, 4), (32, 8), (128, 4)]
T = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
X = mx.random.normal((T, D)).astype(mx.float32)
mx.eval(X)
print(f"{socket.gethostname()} T={T} D={D} FF={FF}")

def bench(f, X, FL, it):
    r = f(X); mx.eval(r)
    reps = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(it): mx.eval(f(X))
        reps.append((time.perf_counter() - t0) / it)
    el = min(reps)
    return el, FL / el / 1e9

for E, TOPK in CONFIGS:
    W1 = (mx.random.normal((E, D, FF)) * 0.02).astype(mx.float32)
    W2 = (mx.random.normal((E, FF, D)) * 0.02).astype(mx.float32)
    Wr = mx.random.normal((D, E)).astype(mx.float32)
    mx.eval(W1, W2, Wr)
    FL = T * TOPK * 2 * 2 * D * FF

    def winner(X, W1=W1, W2=W2, Wr=Wr, E=E, TOPK=TOPK):
        e = mx.argpartition(-(X @ Wr), TOPK, axis=-1)[:, :TOPK].reshape(-1)
        order = mx.argsort(e)
        es, tok = mx.take(e, order), order // TOPK
        xs = mx.take(X, tok, axis=0).reshape(-1, 1, D)
        h = mx.maximum(mx.gather_mm(xs, W1, rhs_indices=es, sorted_indices=True), 0.0)
        y = mx.gather_mm(h, W2, rhs_indices=es, sorted_indices=True).reshape(-1, D)
        return mx.zeros((T, D), X.dtype).at[tok].add(y)

    def bound(X, W1=W1, W2=W2, E=E, TOPK=TOPK):
        xb = mx.concatenate([X.reshape(E, T // E, D)] * TOPK, axis=1)
        return (mx.maximum(xb @ W1, 0.0) @ W2).reshape(-1, D)

    ew, gw = bench(winner, X, FL, 10)
    eb, gb = bench(bound, X, FL, 10)
    print(f"  E={E:3d} k={TOPK}  rows/expert={T*TOPK//E:5d}  sort_take_gather_mm {ew*1e3:7.2f} ms {gw:6.0f} GF/s   balanced_bmm {eb*1e3:7.2f} ms {gb:6.0f} GF/s   {gw/gb*100:.0f}% of bound")
    del W1, W2, Wr
    mx.clear_cache()
