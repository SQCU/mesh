import mlx.core as mx, time, sys
Bn = int(sys.argv[1]) if len(sys.argv)>1 else 4096
d  = int(sys.argv[2]) if len(sys.argv)>2 else 256
X = mx.random.normal((Bn,d)).astype(mx.float32); mx.eval(X)
def pairwise_distance():
    inner_products = X @ X.T
    n = mx.sum(X*X, axis=1)
    return n[:,None] + n[None,:] - 2*inner_products
mx.eval(pairwise_distance())
it=10; t=time.perf_counter()
for _ in range(it): mx.eval(pairwise_distance())
e=(time.perf_counter()-t)/it
fl=2.0*Bn*Bn*d
print(f"  bots={Bn} d={d}  {e*1e3:.2f} ms  {fl/1e9:.2f} GFLOP  {fl/e/1e9:.0f} GF/s")
print(f"  in {Bn*d*4/1e6:.1f} MB -> out {Bn*Bn*4/1e6:.0f} MB   {fl/(Bn*d*4):.0f} FLOP per input byte")
