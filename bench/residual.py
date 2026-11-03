import mlx.core as mx, time, sys
n = int(sys.argv[1]) if len(sys.argv)>1 else 4096
R = int(sys.argv[2]) if len(sys.argv)>2 else 512
X = mx.random.normal((n,R)).astype(mx.float32); mx.eval(X)
def step():
    G = X.T @ X
    G = G + mx.eye(R)*1e-3
    L = mx.linalg.cholesky(G, stream=mx.cpu)
    return L
mx.eval(step())
it=5; t=time.perf_counter()
for _ in range(it): mx.eval(step())
e=(time.perf_counter()-t)/it
fl = 2.0*n*R*R + R**3/3
print(f"  n={n:<6} R={R:<6} {e*1e3:7.2f} ms  {fl/1e9:8.2f} GFLOP  {fl/e/1e9:7.0f} GF/s")
