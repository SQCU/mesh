import mlx.core as mx, time, sys
N, K = 4096, 1024
print(f"  N={N} K={K}   dense ref, then CSR-style gather at varying density")
A=mx.random.normal((N,N)).astype(mx.float32); X=mx.random.normal((N,K)).astype(mx.float32)
mx.eval(A,X); mx.eval(A@X)
it=10; t=time.perf_counter()
for _ in range(it): mx.eval(A@X)
dense=(time.perf_counter()-t)/it
print(f"  {'density':<10} {'nnz/row':<9} {'time ms':<10} {'useful GF/s':<13} {'vs dense'}")
print(f"  {'1.00':<10} {N:<9} {dense*1e3:<10.2f} {2.0*N*N*K/dense/1e9:<13.1f} 1.00x")
for d in (0.30, 0.10, 0.03, 0.01):
    nnz=int(N*d)
    idx = mx.random.randint(0, N, (N, nnz))
    Av  = mx.random.normal((N, nnz)).astype(mx.float32)
    mx.eval(idx, Av)
    def spmm():
        G = X[idx]
        return mx.sum(G * Av[:,:,None], axis=1)
    mx.eval(spmm())
    it2 = 10 if d>=0.1 else 20
    t=time.perf_counter()
    for _ in range(it2): mx.eval(spmm())
    e=(time.perf_counter()-t)/it2
    print(f"  {d:<10.2f} {nnz:<9} {e*1e3:<10.2f} {2.0*N*nnz*K/e/1e9:<13.1f} {dense/e:.2f}x")
