import mlx.core as mx, time, sys
n,R = int(sys.argv[1]), int(sys.argv[2])
X = mx.random.normal((n,R)).astype(mx.float32); mx.eval(X)
mx.eval(X.T @ X)
it=10; t=time.perf_counter()
for _ in range(it): mx.eval(X.T @ X)
e=(time.perf_counter()-t)/it
fl=2.0*n*R*R
print(f"  n={n:<6} R={R:<6} {e*1e3:7.2f} ms {fl/1e9:8.2f} GFLOP {fl/e/1e9:7.0f} GF/s")
