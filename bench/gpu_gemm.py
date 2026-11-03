import mlx.core as mx, time, sys
N = int(sys.argv[1]) if len(sys.argv)>1 else 4096
print(f"  N={N}  A={N*N*4/1e6:.1f} MB   backend=mlx/metal")
print(f"  {'K':<6} {'GFLOP/it':<10} {'time ms':<12} {'GFLOP/s':<10} {'AI':<8}")
A = mx.random.normal((N,N)).astype(mx.float32); mx.eval(A)
for K in (1,8,64,256,512,1024):
    X = mx.random.normal((N,K)).astype(mx.float32); mx.eval(X)
    mx.eval(A @ X)
    it = 50 if K<=64 else 20 if K<=256 else 10
    t0=time.perf_counter()
    for _ in range(it):
        mx.eval(A @ X)
    e=(time.perf_counter()-t0)/it
    fl=2.0*N*N*K; ai=fl/(4.0*N*N+8.0*N*K)
    print(f"  {K:<6} {fl/1e9:<10.2f} {e*1e3:<12.3f} {fl/e/1e9:<10.1f} {ai:<8.1f}")
