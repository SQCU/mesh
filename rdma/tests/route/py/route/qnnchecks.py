"""qnn.py's measurements and gates (python -m route qnn-roofline | qnn-deploy | qnn-check | qnn-i8mm):
  roofline   random numbers at the trainer's sizes through the compiled learning step and the rollouts' forward, each call
             evaluated (mx.eval of its result): ms, effective TOP/s (the network's multiply-adds, __main__'s count) and
             real TOP/s (every int8 product the matrix units execute, at its padded shape: tiles, tasks, wgrad blocks)
  deploy     forward.c (the deployed CPU forward, the shared headers) against qnn's forward on the same int8 codes
  check      the int8 network's three task gradients against the fp32 reference's (the same function from the modules,
             MLX's autodiff, the same credit): cosines; w's relative error
  i8mm       the generic int8 product exact against float on integer codes"""
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

from . import qnn
from .qnn import D, E, FS, IN, L

LANE_MACS = 64 * D + 2 * D * D + 2 * D * E + 2 * D * D
CTX_MACS = 64 * D + 2 * (2 * D * D + 2 * D * E) + FS * D
UNIT = 64 ** 3


def real_macs(B, K, R, learn=True):
    """the int8 products' multiply-adds as executed (tile.h's 64 x 64 x 64 chunks a 64-row tile; wgrad at its padded M,
    kf and rows)"""
    lt, ct = qnn.pad64(B * qnn._klp(K)) // 64, -(-R // 64)
    fwd = (lt * 11 + ct * 18) * UNIT
    if not learn:
        return fwd
    bwd = qnn.TASKS * (lt * 10 + ct * 17) * UNIT
    wg = 0
    for i, (_, rows, cols, _) in enumerate(qnn.MATS):
        m = qnn.pad64(qnn.TASKS * rows) * qnn.pad64(cols)
        wg += m * (ct * 64 + (lt * 64 if i in qnn.LANE_MATS else 0))
    return fwd + bwd + wg


def batch(B, K, R, rng, nd=None):
    """a learning batch of random numbers: B runs x K refreshes, R contexts spread over the refreshes, a draw a row (one
    destination, five lanes valid and allowed), as __main__'s roofline"""
    BK = B * K
    seg = np.repeat(np.arange(BK), R // BK); seg = np.concatenate([seg, np.full(R - len(seg), BK - 1)])
    bounds = np.searchsorted(seg, np.arange(BK + 1))
    nd = nd or R
    dr = np.zeros((nd, 40), np.uint32); f = dr.view(np.float32)
    f[:, :30] = rng.random((nd, 30)); dr[:, 30] = 31; dr[:, 31] = 31; dr[:, 32] = 1; dr[:, 33] = rng.integers(0, 5, nd); dr[:, 34] = 5; dr[:, 35] = 5
    rows = np.sort(rng.integers(0, R, nd)) if nd != R else np.arange(R)
    dr[:, 36] = rows; f[:, 37] = 1; f[:, 38] = 0.01; f[:, 39] = 0.01
    first = np.searchsorted(rows, np.arange(R + 1)).astype(np.uint32)
    return {'lanes': rng.standard_normal((B, K, L, IN)).astype(np.float32), 'mask': np.ones((B, K, L), np.float32),
            'h0': (0.1 * rng.standard_normal((B, 2, D))).astype(np.float32), 'tokens': rng.standard_normal((R, IN)).astype(np.float32),
            'pos': (seg * L + L - 1).astype(np.int32), 'seg': np.stack([bounds[:-1], bounds[1:]], 1).astype(np.uint32),
            'w': (0.1 * rng.standard_normal((R, FS))).astype(np.float32), 'draws': dr, 'first': first,
            'scale0': np.full(R, 1.0 / B, np.float32), 'scale': np.float32(1.0 / B)}


def to_mx(b):
    """the batch's arrays as MLX arrays without a copy (DLPack: the host buffers the data ring holds)"""
    return {k: mx.from_dlpack(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else mx.array(v) for k, v in b.items()}


def emit(rec):
    sys.stdout.write(json.dumps(rec) + '\n'); sys.stdout.flush()


def roofline(B=1024, K=4, R=262144, n=10):
    mx.random.seed(20261119)
    model = qnn.Policy(); lr = qnn.Learner(model, 0.0)
    b = to_mx(batch(B, K, R, np.random.default_rng(1)))
    stats, loss = lr.step(b); mx.eval(stats, loss, lr.state)
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); stats, loss = lr.step(b); mx.eval(stats, loss, lr.state); t.append(time.perf_counter() - t0)
    s = float(np.median(t))
    eff = 3 * 2 * (B * K * L * LANE_MACS + R * CTX_MACS)
    emit({'record': 'qnn-roofline', 'call': 'learning step', 'runs': B, 'refreshes': K, 'context_rows': R, 'ms': round(s * 1e3, 3), 'ms_min': round(min(t) * 1e3, 3),
          'effective_tops': round(eff / s / 1e12, 2), 'real_tops': round(2 * real_macs(B, K, R) / s / 1e12, 2), 'loss': [float(x) for x in np.array(loss)]})
    fb = to_mx(batch(B, 1, R, np.random.default_rng(2)))
    pos = mx.array(((np.arange(R) % B) * L + L - 1).astype(np.int32))
    codes = model.codes(); mx.eval(codes)   # the actor's, a publish's
    act = mx.compile(lambda: model.act(fb['lanes'], fb['mask'], fb['h0'], fb['tokens'], pos, codes))
    mx.eval(act())
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); mx.eval(act()); t.append(time.perf_counter() - t0)
    s = float(np.median(t))
    emit({'record': 'qnn-roofline', 'call': 'forward', 'runs': B, 'context_rows': R, 'ms': round(s * 1e3, 3), 'ms_min': round(min(t) * 1e3, 3),
          'effective_tops': round(2 * (B * L * LANE_MACS + R * CTX_MACS) / s / 1e12, 2), 'real_tops': round(2 * real_macs(B, 1, R, False) / s / 1e12, 2)})


def write_codes(model, path):
    """the weights as deployed (int8mm.h's format, forward.c reads it): each matrix's forward codes, scales, bias"""
    _, wq, wsc = model.weights(); wq, wsc = np.array(wq), np.array(wsc)
    out = [np.array([0x50384952, 2, len(qnn.MATS)], np.uint32).tobytes()]
    for i, (_, rows, cols, bias) in enumerate(qnn.MATS):
        wf, wfs, bo, *_ , kf = [int(x) for x in qnn.NET[12 * i:12 * i + 8]]
        out.append(np.array([rows, cols, kf, int(bias)], np.uint32).tobytes())
        out.append(wq[wf:wf + rows * kf].tobytes()); out.append(wsc[wfs:wfs + rows].tobytes())
        if bias:
            out.append(wsc[bo:bo + rows].tobytes())
    open(path, 'wb').write(b''.join(out))


def deploy(lib, ffi, runs=8, contexts=40):
    """forward.c against qnn's forward (act: one refresh) on the same codes: w and the next state, bit for bit"""
    mx.random.seed(7); model = qnn.Policy()
    rng = np.random.default_rng(3); R = runs * contexts
    b = batch(runs, 1, R, rng)
    nl = rng.integers(1, L + 1, runs); b['mask'][:] = (np.arange(L)[None, None, :] < nl[:, None, None])
    pos = np.repeat(np.arange(runs) * L + L - 1, contexts).astype(np.int32)
    w, nxt = model.act(mx.array(b['lanes']), mx.array(b['mask']), mx.array(b['h0']), mx.array(b['tokens']), mx.array(pos))
    w, nxt = np.array(w), np.array(nxt)
    path = f'/tmp/route-qnn-deploy-{os.getpid()}.i8'; write_codes(model, path)
    p = ffi.new('struct route_params *'); lib.route_params_read(p, path.encode()); os.unlink(path)
    scratch = ffi.new('float[]', lib.route_forward_scratch(contexts))
    dw, dh = 0.0, 0.0
    for r in range(runs):
        h = b['h0'][r].reshape(-1).copy(); wc = np.zeros((contexts, FS), np.float32)
        lanes = np.ascontiguousarray(b['lanes'][r, 0]); tok = np.ascontiguousarray(b['tokens'][r * contexts:(r + 1) * contexts])
        lib.route_forward(p, ffi.from_buffer('float[]', lanes), int(nl[r]), ffi.from_buffer('float[]', tok), contexts, ffi.from_buffer('float[]', h, require_writable=True),
                          ffi.from_buffer('float[]', wc, require_writable=True), scratch)
        dw = max(dw, float(np.abs(wc - w[r * contexts:(r + 1) * contexts]).max())); dh = max(dh, float(np.abs(h - nxt[r].reshape(-1)).max()))
    emit({'record': 'qnn-deploy', 'runs': runs, 'contexts': R, 'w_largest_difference': dw, 'w_largest': float(np.abs(w).max()), 'state_largest_difference': dh})


def check(B=4, K=4, R=1321):
    """the three task gradients of the int8 network (the compiled step's) against the fp32 reference's"""
    mx.random.seed(11); model = qnn.Policy(); lr = qnn.Learner(model, 0.0)
    rng = np.random.default_rng(5); bn = batch(B, K, R, rng, nd=4 * R); bn['h0'] *= 5
    b = to_mx(bn)
    J, loss = lr.grads(b); mx.eval(J, loss)
    w, _ = model(b['lanes'], b['mask'], b['h0'], b['tokens'], b['pos'], b['seg'], lr.rng['seed'])
    w = mx.broadcast_to(w, (3,) + w.shape[1:])
    dw3 = qnn.surrogate(w, b['w'], b['draws'], b['first'], b['scale0'], b['scale'])[1]; w = w[0]
    from mlx.utils import tree_flatten, tree_unflatten
    params = tree_flatten(model.trainable_parameters()); names, primals = [k for k, _ in params], [v for _, v in params]

    def ref(*p):
        model.update(tree_unflatten(list(zip(names, p))))
        return [model.reference(b['lanes'], b['mask'], b['h0'], b['tokens'], b['pos'])]
    wr = None; G = []
    for t in range(3):
        out, g = mx.vjp(ref, primals, [dw3[t]]); wr = out[0]
        G.append(mx.concatenate([x.reshape(-1) for x in g]))
    model.update(tree_unflatten(params))
    G = mx.stack(G); mx.eval(G, wr)
    J, G, w, wr = np.array(J, np.float64), np.array(G, np.float64), np.array(w), np.array(wr)
    cos = [float(J[t] @ G[t] / (np.linalg.norm(J[t]) * np.linalg.norm(G[t]))) for t in range(3)]
    emit({'record': 'qnn-check', 'runs': B, 'refreshes': K, 'rows': R, 'w_relative_error': float(np.linalg.norm(w - wr) / np.linalg.norm(wr)),
          'gradient_cosine_by_task': cos, 'loss': [float(x) for x in np.array(loss)]})


def i8mm_exact(M=1000, N=300, K=640):
    """the generic product on integer codes against float: exact"""
    rng = np.random.default_rng(9)
    a = rng.integers(-127, 128, (M, K)).astype(np.int8); b = rng.integers(-127, 128, (N, K)).astype(np.int8)
    c = np.array(qnn.i8(mx.array(a), mx.array(b)))
    want = a.astype(np.float64) @ b.astype(np.float64).T
    emit({'record': 'qnn-i8mm', 'shape': [M, N, K], 'exact': bool((c == want).all()), 'largest_difference': float(np.abs(c - want).max())})
