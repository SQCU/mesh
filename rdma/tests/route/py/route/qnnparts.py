"""the learning step piece by piece (python -m route.qnnparts from tests/route/py): each alone at the trainer's sizes, evaluated a
call, the median less the eval's own sync (~0.24 ms)"""
import sys, time, numpy as np, mlx.core as mx
from route import qnn, qnnchecks as Q
B, K, R = 1024, 4, 262144
N = 6
model = qnn.Policy()
b = Q.to_mx(Q.batch(B, K, R, np.random.default_rng(1)))
seed = mx.array([5], mx.uint32)
wb, wq, wsc = model.weights(); wb = mx.broadcast_to(wb, (3, wb.shape[1]))
lo = qnn.lanes(wb, wq, wsc, b['lanes'], b['mask'], b['h0'], seed)
co = qnn.contexts(wb, wq, wsc, b['tokens'], b['pos'], b['seg'], lo[0], seed)
so = qnn.surrogate(co[0], b['w'], b['draws'], b['first'], b['scale0'], b['scale'])
dw = so[1]
mx.eval(lo, co, so)
dst = qnn._ctx_back_impl(dw, wq, wsc, b['pos'], b['seg'], lo[0][0], seed, *co[1:])[1]; mx.eval(dst)


def timed(name, f):
    mx.eval(f()); ts = []
    for _ in range(N):
        t0 = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t0)
    t = float(np.median(ts)) - 0.24e-3
    print(f'{name:28s} {t * 1e3:7.3f} ms', flush=True)


s1 = mx.zeros((1,), mx.uint32)
timed('weight codes', lambda: qnn.weight_codes(wb[0] + 0))
timed('lanes_fwd store', lambda: qnn._lanes_fwd(wq, wsc, b['lanes'], b['mask'], b['h0'] + 0, seed, True)[0])
timed('ctx_fwd store', lambda: qnn._ctx_fwd(wq, wsc, b['tokens'], b['pos'], lo[0][0] + 0, 0, 1, seed, True)[0])
timed('credit', lambda: qnn.surrogate(co[0] + 0, b['w'], b['draws'], b['first'], b['scale0'], b['scale'])[1])


def ctxb(wgrad):
    T, Rp = 3, qnn.pad64(R)
    table, _, (yq, yf) = qnn._grad_table(Rp, T)
    p = qnn._fp(B * K, 1, R, 0, Rp, seed)
    o = qnn._kernel('bwd.metal', 'ctx_bwd', **KW)({'dw': (dw + 0).reshape(-1), 'pos': b['pos'], 'st': lo[0][0], 'czc': co[3], 'cu': co[4], 'nw': qnn._c(qnn.NET), 'wq': wq, 'wsc': wsc, 'p': p,
                                            'gr': qnn._c(table), 'cn': co[5], 'cinv': co[6], 'H': qnn._c(qnn._H)},
                                           {'gq': ((yq,), mx.int8), 'gs': ((yf,), mx.float32), 'dhp': ((T * 2 * Rp * qnn.D,), mx.float32)}, (-(-R // 64), T), 256)
    if not wgrad:
        return o['gq']
    return [qnn._wgrad(o['gq'], o['gs'], co[1], co[2], table, i, Rp, T, -(-R // 32))[0] for i in range(8)]


KW = {}
timed('ctx_bwd', lambda: ctxb(False))
KW = {}
timed('ctx_bwd + wgrad', lambda: ctxb(True))
timed('lanes back (bwd + wgrad)', lambda: qnn._lanes_back_impl(dst + 0, wq, wsc, b['mask'], b['h0'], seed, *lo[2:]))
lr = qnn.Learner(model, 0.0)
timed('grads (whole)', lambda: lr.grads(b)[0])
