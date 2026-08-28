import mlx.core as mx, time, sys
N=int(sys.argv[1]) if len(sys.argv)>1 else 4096
best={}
for K in (2048,4096,8192):
    A=mx.random.normal((N,N)).astype(mx.float32); X=mx.random.normal((N,K)).astype(mx.float32)
    mx.eval(A,X)
    for _ in range(3): mx.eval(A@X)          # warmup
    ts=[]
    for _ in range(7):
        t=time.perf_counter()
        for _ in range(5): mx.eval(A@X)
        ts.append((time.perf_counter()-t)/5)
    e=min(ts)
    best[K]=2.0*N*N*K/e/1e9
    print(f"  N={N} K={K:<5} min-of-7 {e*1e3:7.2f} ms  {best[K]:8.0f} GF/s")
print(f"  peak observed: {max(best.values()):.0f} GF/s")
