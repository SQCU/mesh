import mlx.core as mx, time, sys
N=4096; K=1024
print(f"  {'dtype':<10} {'time ms':<10} {'GFLOP/s':<10}")
for dt,name in ((mx.float32,'float32'),(mx.float16,'float16'),(mx.bfloat16,'bfloat16')):
    A=mx.random.normal((N,N)).astype(dt); X=mx.random.normal((N,K)).astype(dt); mx.eval(A,X)
    mx.eval(A@X)
    it=20; t=time.perf_counter()
    for _ in range(it): mx.eval(A@X)
    e=(time.perf_counter()-t)/it
    print(f"  {name:<10} {e*1e3:<10.3f} {2.0*N*N*K/e/1e9:<10.1f}")
