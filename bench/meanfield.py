import mlx.core as mx, time, sys
N   = int(sys.argv[1]) if len(sys.argv)>1 else 4096
B   = int(sys.argv[2]) if len(sys.argv)>2 else 192
M, KPB, GAMMA, MU = 32, 8, 0.92, 0.35
K = B*KPB
print(f"  N={N} bots={B} K={K}  A={N*N*4/1e6:.0f}MB  V={N*K*4/1e6:.0f}MB")
A = mx.random.normal((N,N)).astype(mx.float32)
A = A*A; A = A / mx.sum(A,axis=1,keepdims=True)      # row-stochastic
R = mx.random.normal((N,K)).astype(mx.float32)
V = mx.zeros((N,K), dtype=mx.float32)
rho = mx.zeros((N,), dtype=mx.float32)
mx.eval(A,R,V,rho)

def solve(A,R,V,rho, exchange=None):
    exchanged = 0
    for m in range(M):
        Z = mx.logaddexp(R + V - MU*rho[:,None], mx.array(0.0))   # softplus
        V = GAMMA * (A @ Z)
        if m % 4 == 3:
            rho_local = mx.sum(Z, axis=1)
            mx.eval(rho_local)
            exchanged += 1
            rho = rho_local if exchange is None else exchange(rho_local)
    mx.eval(V)
    return V, exchanged

V2,ex = solve(A,R,V,rho); mx.eval(V2)
t0=time.perf_counter(); V2,ex = solve(A,R,V,rho); e=time.perf_counter()-t0
flop = M*K*(2*N*N + 8*N)
print(f"  solve: {e*1e3:.1f} ms   {flop/1e12:.3f} TFLOP   {flop/e/1e12:.2f} TFLOP/s")
print(f"  exchanges/solve: {ex}   rho = {N} floats = {N*4} B = {N*4//4096} pages")
print(f"  max solve rate: {1/e:.1f} Hz    at 10 Hz uses {10*e*100:.0f}% of the GPU")
print(f"  wire at 10 Hz: {ex*N*4*10/1e6:.2f} MB/s = {100*ex*N*4*10/8.7e9:.4f}% of link")
