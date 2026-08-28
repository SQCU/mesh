import mlx.core as mx, time, sys
D, FF, E, TOPK = 2048, 4096, 32, 2
T = int(sys.argv[1]) if len(sys.argv)>1 else 4096
print(f"  d_model={D} d_ff={FF} experts={E} top_k={TOPK} tokens={T}")
W1 = (mx.random.normal((E,D,FF))*0.02).astype(mx.float32)
W2 = (mx.random.normal((E,FF,D))*0.02).astype(mx.float32)
X  = mx.random.normal((T,D)).astype(mx.float32)
Wr = mx.random.normal((D,E)).astype(mx.float32)
mx.eval(W1,W2,X,Wr)
print(f"  expert weights {2*E*D*FF*4/1e9:.2f} GB   activations {T*D*4/1e6:.0f} MB")

def moe(X):
    logits = X @ Wr
    idx = mx.argpartition(-logits, TOPK, axis=-1)[:, :TOPK]
    outs = []
    for k in range(TOPK):
        e = idx[:,k]
        order = mx.argsort(e)                  # sort tokens by expert -> contiguous blocks
        es = e[order]
        xs = X[order]
        mx.eval(es)
        bounds = [0]*(E+1)
        el = es.tolist()
        j = 0
        for ei in range(E):
            while j < len(el) and el[j] == ei: j += 1
            bounds[ei+1] = j
        parts = []
        for ei in range(E):
            a,b = bounds[ei], bounds[ei+1]
            if b<=a: continue
            h = mx.maximum(xs[a:b] @ W1[ei], 0.0)   # gathered block -> dense GEMM
            parts.append((order[a:b], h @ W2[ei]))
        o = mx.zeros_like(X)
        for sel, val in parts: o[sel] = val         # scatter back
        outs.append(o)
    return outs[0] + outs[1] if TOPK==2 else outs[0]

mx.eval(moe(X))
it=3; t=time.perf_counter()
for _ in range(it): mx.eval(moe(X))
e=(time.perf_counter()-t)/it
fl = T*TOPK*2*2*D*FF
print(f"  time {e*1e3:.1f} ms   useful {fl/1e9:.1f} GFLOP   {fl/e/1e9:.0f} GF/s")
print(f"  wire/token {2*D*4} B -> {fl/T/(2*D*4):.0f} FLOP/byte")
