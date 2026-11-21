"""The routing policy in the PyTorch dialect (MLX on Apple GPUs): ops, modules, loss, step.

  ops      the tile kernels (fwd.metal, bwd.metal, credit.metal, opt.metal, mm.metal: tile.h's tiles on int8mm.h's
           numerics) as MLX ops, each with its VJP:
             codes      the weights' int8 codes (forward rows, dgrad columns) from W (straight through)
             lanes      the fused lane op: a run's lanes through inp, both layers' gates and the recurrence (scan.h's
                        affine scan), layer 0's feed-forward; the states each refresh leaves. VJP: lanes_bwd (the
                        reverse scan), the wgrad op
             contexts   the fused context-row chain op: a tile of contexts through every layer and the readout: w.
                        VJP: ctx_bwd, the states' gradients (states_grad), the wgrad op
             wgrad      dW = dY^T X over 32-row blocks of column codes (gemm_wg), exact int32 a block, fp32 across
             surrogate  the (unclipped, as credit.metal) importance-weighted score-function objective a task (three: the throughput,
                        latency and queueing advantages); its VJP the credit (credit_draws, credit_rows: split.h)
           Each backward takes T tasks at once: the parameters as T copies (a leading task axis: the step's three), the
           forward run once, each backward its kernel with T = 3 (mx.vmap re-traces a custom function's forward, which
           a Metal kernel cannot be). And quantize / i8mm, the generic int8 x int8 ->
           int32 product (matmul2d), its VJP int8 products too.
  modules  QLinear (W = g V/|V| a row, bias; called: the fp32 reference product), MinGRU (gates, the recurrence, the
           feed-forward: the fp32 reference), Policy (the int8 network on the fused ops; reference(): the same function
           in fp32 from the modules)
  loss     surrogate: per task, sum over draws of coefficient x stop(ratio) x log pi(token)
  step     the three tasks' gradients (task copies of the parameters, one value_and_grad), UPGrad's weights from their Gram matrix (upgrad.h), Adam, V's rows
           pinned; microbatches accumulated (gradient) and one exchange a step (apply: the sums, rows and microbatch
           norms, identical statistics at every rank); all compiled with mx.compile

Arrays are values the array library makes; modules hold parameters; nothing allocates by hand."""
import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from . import metal

D, E, IN, FS, L, TM, WB = 64, 192, 48, 40, 32, 64, 32
TASKS = 3


def pad64(n):
    return -(-n // 64) * 64


# ---- the layout: the matrices (tile.h's order), the flat weights vector the ops take, the kernels' tables ----
MATS = [('inp', D, IN, True), ('zh0', 2 * D, D, True), ('up0', E, D, True), ('down0', D, E, True),
        ('zh1', 2 * D, D, True), ('up1', E, D, True), ('down1', D, E, True), ('out', FS, D, False)]
LANE_MATS = (0, 1, 2, 3, 4)   # the lanes' (layer 1's lanes: their states only)


def _layout():
    net, items_r, items_c, at, q, f = [], [], [], 0, 0, 0
    woff = []
    for i, (_, rows, cols, bias) in enumerate(MATS):
        kb, kf = pad64(rows), pad64(cols)
        v = at; at += rows * cols; g = at; at += rows if bias else 0
        woff.append((v, g))
        net += [q, f, f + kb, q + kb * kf, f + 2 * kb, rows, cols, kf, kb, kf, v, g]
        q += 2 * kb * kf; f += 2 * kb + kf
        items_r += [(i, r) for r in range(kb)]; items_c += [(i, c) for c in range(kf)]
    return np.array(net, np.uint32), np.array(items_r + items_c, np.uint32).ravel(), len(items_r), at, q, f, woff


NET, ITEMS, NROWS, NW, NQ, NF, WOFF = _layout()
_H = np.array([[-1.0 if bin(i & j).count('1') % 2 else 1.0 for j in range(64)] for i in range(64)], np.float16)


def _grad_table(RT, T):
    """tile.h's Grad for one op's codes (X: its forward's, Y: its backward's, each its own arrays) over RT rows"""
    blocks, t, xq, xf, yq, yf = RT // WB, [], 0, 0, 0, 0
    for _, rows, cols, _ in MATS:
        M, kf = T * rows, pad64(cols)
        t += [yq, xq, yf, xf, yf + blocks * M, 0, 0, 0]
        xq += kf * RT; xf += blocks * kf; yq += pad64(M) * RT; yf += 2 * blocks * M
    return np.array(t + [RT, T, 0, 0], np.uint32), (xq, xf), (yq, yf)


def _klp(K):
    return L if K == 1 else pad64(K * L)


def _fp(B, K, R, RL, Rp, seed):
    """tile.h's FP: runs, refreshes, a run's lane rows, contexts, the lane rows' offset (0: each op its own codes), the
    seed (an array: it changes every step), the contexts' room, lane tiles a threadgroup"""
    k = _klp(K)
    R = mx.array([R], mx.uint32) if isinstance(R, int) else R.reshape(1).astype(mx.uint32)   # an array: a padded batch's own rows
    head = mx.concatenate([mx.array([B, K, k], mx.uint32), R, mx.array([RL], mx.uint32)]); tail = mx.array([Rp, 1 if K == 1 else k // 64], mx.uint32)
    return mx.concatenate([head, seed.reshape(1).astype(mx.uint32), tail])


_K = {}


def _kernel(file, name, **kw):
    key = (file, name) + tuple(sorted(kw.items()))
    if key not in _K:
        _K[key] = metal.Kernel(file, name, **kw)
    return _K[key]


def _c(a):
    return mx.array(a)


# ---- ops ----
def weight_codes(wb):
    """the weights' codes from W itself (opt.metal codes): wq (forward codes [pad64(rows)][kf] a matrix, dgrad codes
    [kf][pad64(rows)]), wsc (scales, biases)"""
    o = _kernel('opt.metal', 'codes')({'w': wb, 'nw': _c(NET), 'items': _c(ITEMS), 'n': mx.array([len(ITEMS) // 2, NROWS], mx.uint32)},
                                      {'wq': ((NQ,), mx.int8), 'wsc': ((NF,), mx.float32)}, (-(-(len(ITEMS) // 2) * 32 // 256),), 256)
    return o['wq'], o['wsc']


def _wgrad(gq, gs, gqx, gsx, table, i, RT, T, b1):
    """the wgrad op: matrix i's dW [T][rows][cols] = dY^T X over blocks [0, b1) of 32 rows (exact int32 a block,
    scaled, fp32 across: gemm_wg), and db [T][rows] its dY's column sums"""
    _, rows, cols, bias = MATS[i]
    yt, xt, sy, sx, dbp = (int(x) for x in table[8 * i:8 * i + 5])
    M, kf, blocks, per = T * rows, pad64(cols), RT // WB, 64
    nz = -(-b1 // per)
    part = _kernel('mm.metal', 'gemm_wg')({'a': gq[yt:yt + pad64(M) * RT], 'b': gqx[xt:xt + kf * RT], 'sa': gs[sy:sy + blocks * M], 'sb': gsx[sx:sx + blocks * kf],
                                                   'p': mx.array([M, kf, RT, per, 0, b1, 0, 0], mx.uint32)},
                                                  {'partial': ((nz * M * kf,), mx.float32)}, (kf // 64, -(-M // 64), nz), 128)['partial']
    dW = part.reshape(nz, T, rows, kf).sum(0)[:, :, :cols]
    db = gs[dbp:dbp + b1 * M].reshape(b1, T, rows).sum(0) if bias else None
    return dW, db


def _dwb(parts, T):
    """the flat weights' cotangent [T][NW] from each matrix's (dW, db) (None: a matrix the op does not use)"""
    out = []
    for i, (_, rows, cols, bias) in enumerate(MATS):
        dW, db = parts.get(i, (None, None))
        out.append(dW.reshape(T, rows * cols) if dW is not None else mx.zeros((T, rows * cols)))
        if bias:
            out.append(db if db is not None else mx.zeros((T, rows)))
    return mx.concatenate(out, 1)


# the fused lane op: lanes [B][K][L][IN], mask [B][K][L], h0 [B][2][D] -> st [B K][2][D] (each refresh's last
# states), next [B][2][D]; what its backward reads besides
def _lanes_fwd(wq, wsc, lanes, mask, h0, seed, store):
    B, K = lanes.shape[:2]; k = _klp(K); RL = pad64(B * k)
    table, (xq, xf), _ = _grad_table(RL, 1)
    n = lambda size, dt: ((size if store else 1,), dt)
    o = _kernel('fwd.metal', 'lanes_fwd', template=(('STORE', store), ('STAGE', 0)), tag='_s' if store else '')(
        {'lanes': lanes, 'mask': mask, 'h0': h0, 'nw': _c(NET), 'wq': wq, 'wsc': wsc, 'p': _fp(B, K, 0, RL, 0, seed), 'gr': _c(table), 'H': _c(_H)},
        {'st': ((B * K * 2 * D,), mx.float32), 'next': ((B * 2 * D,), mx.float32), 'gq': n(xq, mx.int8), 'gs': n(xf, mx.float32),
         'lzc': n(RL * 4 * D, mx.float16), 'lh': n(RL * 2 * D, mx.float32), 'lu': n(RL * 6, mx.uint32)},
        ((B + 1) // 2 if K == 1 else B,), 256)
    return o['st'].reshape(B * K, 2, D), o['next'].reshape(B, 2, D), o


def _lanes_back_impl(dst, wq, wsc, mask, h0, seed, gqx, gsx, lzc, lh, lu):
    T = dst.shape[0]; B, K = mask.shape[:2]; k = _klp(K); RL = pad64(B * k)
    table, _, (yq, yf) = _grad_table(RL, T)
    o = _kernel('bwd.metal', 'lanes_bwd')(
        {'mask': mask, 'h0': h0, 'lzc': lzc, 'lh': lh, 'lu': lu, 'nw': _c(NET), 'wq': wq, 'wsc': wsc, 'p': _fp(B, K, 0, RL, 0, seed), 'gr': _c(table),
         'dst': dst.reshape(-1), 'H': _c(_H)},
        {'gq': ((yq,), mx.int8), 'gs': ((yf,), mx.float32)}, ((B + 1) // 2 if K == 1 else B, T), 256)
    parts = {i: _wgrad(o['gq'], o['gs'], gqx, gsx, table, i, RL, T, RL // WB) for i in LANE_MATS}
    return _dwb(parts, T)


@mx.custom_function
def lanes(wb, wq, wsc, lanes_, mask, h0, seed):
    """wb [T][NW]: the weights a task (T copies, equal: the step's tasks; the forward is computed once, from wq and wsc,
    their codes) -> st [T][B K][2][D] (the same a task), next [B][2][D]; the VJP takes the T tasks' cotangents at once"""
    st, nxt, o = _lanes_fwd(wq, wsc, lanes_, mask, h0, seed, True)
    return mx.broadcast_to(st[None], (wb.shape[0],) + st.shape), nxt, o['gq'], o['gs'], o['lzc'], o['lh'], o['lu']


@lanes.vjp
def _(primals, cots, outs):
    wb, wq, wsc, lanes_, mask, h0, seed = primals
    dwb = _lanes_back_impl(cots[0], wq, wsc, mask, h0, seed, *outs[2:])
    return dwb, mx.zeros_like(wq), mx.zeros_like(wsc), mx.zeros_like(lanes_), mx.zeros_like(mask), mx.zeros_like(h0), mx.zeros_like(seed)


# the fused context-row chain op: tokens [R][IN], pos [R] (a context's state: its refresh's last lane row, pos / L the
# refresh), seg [B K][2] (each refresh's contexts' rows), st -> w [R][FS]
def _ctx_fwd(wq, wsc, tokens, pos, st, B, K, seed, store, rows=None):
    """rows: the contexts that are (an array; a padded batch's rows past it are the kernels' invalid rows), else all"""
    R = tokens.shape[0]; Rp = pad64(R)
    table, (xq, xf), _ = _grad_table(Rp, 1)
    n = lambda size, dt: ((size if store else 1,), dt)
    o = _kernel('fwd.metal', 'ctx_fwd', template=(('STORE', store), ('STAGE', 0)), tag='_s' if store else '')(
        {'tokens': tokens, 'pos': pos, 'st': st, 'nw': _c(NET), 'wq': wq, 'wsc': wsc, 'p': _fp(B, K, R if rows is None else rows, 0, Rp, seed), 'gr': _c(table), 'H': _c(_H)},
        {'w': ((R * FS,), mx.float32), 'gq': n(xq, mx.int8), 'gs': n(xf, mx.float32), 'czc': n(R * 4 * D, mx.float16), 'cu': n(R * 12, mx.uint32),
         'cn': n(R * D, mx.float32), 'cinv': n(R, mx.float32)}, (-(-R // TM),), 256)
    return o['w'].reshape(R, FS), o


def _ctx_back_impl(dw, wq, wsc, pos, seg, st, seed, rows, gqx, gsx, czc, cu, cn, cinv):
    T, R = dw.shape[:2]; Rp = pad64(R); BK = seg.shape[0]
    table, _, (yq, yf) = _grad_table(Rp, T)
    p = _fp(BK, 1, rows, 0, Rp, seed)   # B K refreshes as states_grad reads them (its B K)
    o = _kernel('bwd.metal', 'ctx_bwd')(
        {'dw': dw.reshape(-1), 'pos': pos, 'st': st, 'czc': czc, 'cu': cu, 'nw': _c(NET), 'wq': wq, 'wsc': wsc, 'p': p, 'gr': _c(table), 'cn': cn, 'cinv': cinv, 'H': _c(_H)},
        {'gq': ((yq,), mx.int8), 'gs': ((yf,), mx.float32), 'dhp': ((T * 2 * Rp * D,), mx.float32)}, (-(-R // TM), T), 256)
    dst = _kernel('bwd.metal', 'states_grad')({'dhp': o['dhp'], 'seg': seg.reshape(-1), 'p': p}, {'dst': ((T * BK * 2 * D,), mx.float32)}, (T * BK,), 2 * D)['dst']
    parts = {i: _wgrad(o['gq'], o['gs'], gqx, gsx, table, i, Rp, T, -(-R // WB)) for i in range(len(MATS))}
    return _dwb(parts, T), dst.reshape(T, BK, 2, D)


@mx.custom_function
def contexts(wb, wq, wsc, tokens, pos, seg, st, seed, rows):
    """wb [T][NW], st [T][B K][2][D] (a task each, equal: the forward once), rows (the contexts that are: a padded
    batch's) -> w [T][R][FS]; the VJP the T tasks'"""
    w, o = _ctx_fwd(wq, wsc, tokens, pos, st[0], 0, 1, seed, True, rows)
    return mx.broadcast_to(w[None], (wb.shape[0],) + w.shape), o['gq'], o['gs'], o['czc'], o['cu'], o['cn'], o['cinv']


@contexts.vjp
def _(primals, cots, outs):
    wb, wq, wsc, tokens, pos, seg, st, seed, rows = primals
    dwb, dst = _ctx_back_impl(cots[0], wq, wsc, pos, seg, st[0], seed, rows, *outs[1:])
    return (dwb, mx.zeros_like(wq), mx.zeros_like(wsc), mx.zeros_like(tokens), mx.zeros_like(pos), mx.zeros_like(seg), dst, mx.zeros_like(seed), mx.zeros_like(rows))


# ---- the generic int8 product (qnn's first op, kept: its own quantization, a 64-block Walsh-Hadamard as a product) ----
def _hadamard(n):
    h = mx.array([[1.0]])
    while h.shape[0] < n:
        h = mx.concatenate([mx.concatenate([h, h], 1), mx.concatenate([h, -h], 1)], 0)
    return h / math.sqrt(n)


_H64 = _hadamard(64)


def quantize(x):
    """rows (last axis, padded to 64-blocks) rotated by the 64-point Hadamard (a product), int8 codes, a scale a row"""
    k = x.shape[-1]; kp = -(-k // 64) * 64
    x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, kp - k)]) if kp != k else x
    y = (x.reshape(-1, kp // 64, 64) @ _H64).reshape(-1, kp)
    s = mx.maximum(mx.max(mx.abs(y), axis=-1, keepdims=True), 1e-30) / 127.0
    return mx.clip(mx.round(y / s), -127, 127).astype(mx.int8), s


_mm = mx.fast.metal_kernel(
    name="i8mm", input_names=["a", "b", "dims"], output_names=["c"],
    header="#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n",
    source="""
  const int M = dims[0], N = dims[1], K = dims[2];
  const uint2 tg = uint2(threadgroup_position_in_grid.x, threadgroup_position_in_grid.y);
  auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>((device int8_t *)a, dextents<int32_t, 2>(K, M));
  auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>((device int8_t *)b, dextents<int32_t, 2>(K, N));
  auto C = tensor<device int32_t, dextents<int32_t, 2>, tensor_inline>((device int32_t *)c, dextents<int32_t, 2>(N, M));
  constexpr auto d = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, true);
  matmul2d<d, execution_simdgroups<4>> op;
  auto tA = A.slice(0, tg.y * 64); auto tB = B.slice(0, tg.x * 64);
  auto r = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
  for (uint16_t i = 0; i < r.get_capacity(); ++i) if (r.is_valid_element(i)) r[i] = 0;
  op.run(tA, tB, r);
  r.store(C.slice(tg.x * 64, tg.y * 64));
""")


def i8(qa, qb):
    """int8 [M][K] x int8 [N][K]^T -> int32 [M][N] on the matrix units (64x64 tiles, M and N padded to whole tiles)"""
    M, K = qa.shape; N = qb.shape[0]; Mp, Np = -(-M // 64) * 64, -(-N // 64) * 64
    qa = mx.pad(qa, [(0, Mp - M), (0, 0)]) if Mp != M else qa
    qb = mx.pad(qb, [(0, Np - N), (0, 0)]) if Np != N else qb
    c, = _mm(inputs=[qa, qb, mx.array([Mp, Np, K], mx.int32)], output_shapes=[(Mp, Np)], output_dtypes=[mx.int32],
             grid=(Np // 64 * 128, Mp // 64, 1), threadgroup=(128, 1, 1))
    return c[:M, :N]


@mx.custom_function
def i8mm(x, w):
    """x [..., K] @ w [N, K]^T, both through int8 codes: an exact int32 sum, scaled back"""
    lead = x.shape[:-1]
    qx, sx = quantize(x.reshape(-1, x.shape[-1])); qw, sw = quantize(w)
    return (i8(qx, qw).astype(mx.float32) * sx * sw.T).reshape(*lead, w.shape[0])


@i8mm.vjp
def _(primals, cot, out):   # dgrad and wgrad, int8 products themselves (straight through the quantization)
    x, w = primals
    c2 = cot.reshape(-1, cot.shape[-1]); x2 = x.reshape(-1, x.shape[-1])
    return i8mm(c2, w.T).reshape(x.shape), i8mm(c2.T, x2.T)


# ---- modules ----
class QLinear(nn.Module):
    def __init__(self, rows, cols, bias=True):
        super().__init__()
        V = mx.random.normal((rows, cols))
        self.V = V / mx.linalg.norm(V, axis=1, keepdims=True); self.g = mx.ones((rows,))
        if bias:
            self.b = mx.zeros((rows,))

    def weight(self):   # (a leading task axis: the step's per-task copies)
        return self.g[..., None] * self.V / mx.linalg.norm(self.V, axis=-1, keepdims=True)

    def __call__(self, x):   # the fp32 reference product
        y = x @ self.weight().T
        return y + self.b if 'b' in self else y

    def pin(self):
        self.V = self.V / mx.linalg.norm(self.V, axis=1, keepdims=True)


class MinGRU(nn.Module):
    """a layer: gates z, h~ from x; the recurrence over positions; the feed-forward on x + h (fp32: the reference)"""
    def __init__(self):
        super().__init__()
        self.zh, self.up, self.down = QLinear(2 * D, D), QLinear(E, D), QLinear(D, E)

    def gates(self, x):
        zc = self.zh(x)
        return mx.sigmoid(zc[..., :D]), mx.tanh(zc[..., D:])

    def ffn(self, xm):
        return xm + self.down(nn.relu(self.up(xm)))

    def sequence(self, x, mask, h0):   # lanes [runs][positions][D]: every position's state
        z, c = self.gates(x); h, hs = h0, []
        for t in range(x.shape[1]):
            mz = mask[:, t, None] * z[:, t]; h = (1 - mz) * h + mz * c[:, t]; hs.append(h)
        return mx.stack(hs, 1)

    def step(self, x, h):              # context rows: one step from their refresh's state
        z, c = self.gates(x)
        return self.ffn(x + h + z * (c - h))


class Policy(nn.Module):
    """the int8 network (the fused ops) and, as reference(), the same function in fp32 from the modules"""
    def __init__(self):
        super().__init__()
        self.inp, self.layers, self.out = QLinear(D, IN), [MinGRU(), MinGRU()], QLinear(FS, D, bias=False)

    def mats(self):
        return [self.inp] + [m for l in self.layers for m in (l.zh, l.up, l.down)] + [self.out]

    def weights(self):
        """the flat weights vector the ops take [T][NW] (each matrix's W, then its bias; T: the parameters' leading task
        axis, or 1) and its codes"""
        lead = self.out.g.shape[:-1]
        wb = mx.concatenate([x.reshape(*lead, -1) for m in self.mats() for x in ([m.weight()] + ([m.b] if 'b' in m else []))], -1).reshape(-1, NW)
        wq, wsc = weight_codes(mx.stop_gradient(wb[0]))
        return wb, wq, wsc

    def __call__(self, lanes_, mask, h0, tokens, pos, seg, seed, rows=None):
        """a learning batch: B runs' K refreshes (lanes [B][K][L][IN], mask [B][K][L], h0 [B][2][D]), R contexts
        (tokens [R][IN], pos [R], seg [B K][2]) -> w [T][R][FS] (T: the parameters' task copies, or 1), next [B][2][D]"""
        wb, wq, wsc = self.weights()
        st, nxt = lanes(wb, wq, wsc, lanes_, mask, h0, seed)[:2]
        rows = mx.array([tokens.shape[0]], mx.uint32) if rows is None else rows.reshape(1).astype(mx.uint32)
        return contexts(wb, wq, wsc, tokens, pos, seg, st, seed, rows)[0], nxt

    def codes(self):
        """the weights' codes alone (an actor's, made once a publish)"""
        return self.weights()[1:]

    def act(self, lanes_, mask, h0, tokens, pos, codes=None):
        """the rollouts' forward (one refresh, nothing stored; codes: an actor's, else made from the weights now):
        w [R][FS], next [B][2][D]"""
        wq, wsc = codes or self.codes()
        z = mx.zeros((1,), mx.uint32)
        st, nxt, _ = _lanes_fwd(wq, wsc, lanes_, mask, h0, z, False)
        return _ctx_fwd(wq, wsc, tokens, pos, st, 0, 1, z, False)[0], nxt

    def reference(self, lanes_, mask, h0, tokens, pos):
        """the same function in fp32 (the modules' own products)"""
        B, K = lanes_.shape[:2]
        x = self.inp(lanes_.reshape(B, K * L, IN)); m = mask.reshape(B, K * L); xc = self.inp(tokens); ref = pos // L
        for l, layer in enumerate(self.layers):
            H = layer.sequence(x, m, h0[:, l])
            st = H[:, L - 1::L].reshape(B * K, D)
            if l == 0:
                x = layer.ffn(x + H)
            xc = layer.step(xc, st[ref])
        n = xc * mx.rsqrt(mx.mean(xc * xc, axis=-1, keepdims=True) + 1e-6)
        return self.out(n)

    def pin(self):
        for m in self.mats():
            m.pin()


# ---- loss: the surrogate, its VJP the credit ----
DRAW_WORDS = 40   # struct route_draw: 160 bytes


@mx.custom_function
def surrogate(w, wbeh, draws, first, scale0, scale, counts=None):
    """draws [ND][40] (struct route_draw's words; first [R + 1] a row's first) under w [3][R][FS] (task t's under
    w[t]: the step's task copies, equal; the credit computed once) and the rows' behaviour w (wbeh: the behaviour
    w they were drawn under): per task, sum over draws of c_t x stop(ratio) x log pi(token) (c: task 0 the row's
    scale0, tasks 1, 2 the draw's a1, a2 times scale) -> [3]; besides, its gradient dL/dw [3][R][FS] (the credit)"""
    R, ND = w.shape[1], draws.shape[0]
    counts = mx.array([R, ND], mx.uint32) if counts is None else counts   # (rows, draws) that are: a padded batch's
    o = _kernel('credit.metal', 'credit_draws', fast=True)({'draws': draws.reshape(-1), 'wb': wbeh.reshape(-1), 'wc': w[0].reshape(-1), 'n': counts[1:2]},
                                                         {'diff': ((ND * FS,), mx.float32), 'll': ((ND,), mx.float32), 'lp': ((ND,), mx.float32)},
                                                         (max(1, -(-ND * 8 // 256)),), 256)
    dw = _kernel('credit.metal', 'credit_rows', fast=True)({'diff': o['diff'], 'draws': draws.reshape(-1), 'first': first, 'scale0': scale0,
                                                         'p': mx.concatenate([counts[0:1], mx.array([0], mx.uint32), scale.reshape(1).view(mx.uint32), mx.array([0], mx.uint32)])},
                                                        {'dw': ((3 * R * FS,), mx.float32)}, (-(-3 * R * FS // 256),), 256)['dw']
    f = draws[:, 36:40].view(mx.float32); row = draws[:, 36]
    v = mx.where(mx.arange(ND) < counts[1], mx.exp(o['ll']) * o['lp'], 0.0)
    loss = mx.stack([(scale0[row] * v).sum(), scale * (f[:, 2] * v).sum(), scale * (f[:, 3] * v).sum()])
    return loss, dw.reshape(3, R, FS)


@surrogate.vjp
def _(primals, cots, outs):
    w, wbeh, draws, first, scale0, scale, counts = primals
    dw = cots[0][:, None, None] * outs[1]
    return dw, mx.zeros_like(wbeh), mx.zeros_like(draws), mx.zeros_like(first), mx.zeros_like(scale0), mx.zeros_like(scale), mx.zeros_like(counts)


# ---- step ----
_UP = mx.fast.metal_kernel(name='upgrad', input_names=['G'], output_names=['alpha'], header='#include <metal_stdlib>\n' + metal._inline('upgrad.h', set()) + '\n',
                           source='if (thread_position_in_grid.x == 0) { float g[9], a[3]; for (int i = 0; i < 9; i++) g[i] = G[i]; upgrad3(g, a); for (int i = 0; i < 3; i++) alpha[i] = a[i]; }')


def upgrad(G):
    """UPGrad's task weights from the Gram matrix of the three gradients (upgrad.h, one GPU thread)"""
    return _UP(inputs=[G.reshape(-1)], output_shapes=[(3,)], output_dtypes=[mx.float32], grid=(1, 1, 1), threadgroup=(1, 1, 1))[0]


BATCH = ('lanes', 'mask', 'h0', 'tokens', 'pos', 'seg', 'w', 'draws', 'first', 'scale0', 'scale')


class Learner:
    """the learning step on a Policy: gradient(batch, acc) adds a microbatch's three task gradients, times its rows,
    into acc (flat: [3][P] then rows, the microbatches' squared norms (3), their count); apply(acc, exchange) takes
    the step from the exchanged sum: the rows-weighted mean, UPGrad, Adam, V pinned; step(batch) both at once.
    exchange: the ranks' sum of a flat array (one rank: none)."""

    def __init__(self, model, rate, seed=1):
        self.model, self.opt = model, optim.Adam(learning_rate=rate, betas=[0.9, 0.999], eps=1e-8, bias_correction=True)
        self.rng = {'seed': mx.array([seed], mx.uint32)}   # the backward's stochastic rounding, a seed a step
        self.names = [k for k, _ in tree_flatten(model.trainable_parameters())]
        self.P = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
        self.state = [model.state, self.opt.state, self.rng]
        self.gradient = mx.compile(self._gradient, inputs=self.state, outputs=self.state)
        self._finish = mx.compile(self._apply, inputs=self.state, outputs=self.state)

    def grads(self, b):
        """the three tasks' gradients [3][P] and losses [3]: the parameters as three copies (a leading task axis), the
        surrogate's three losses summed, one value_and_grad: the forward once, each backward its kernel with T = 3"""
        model, params = self.model, tree_flatten(self.model.trainable_parameters())
        names = [k for k, _ in params]
        seed = self.rng['seed']

        def f(p3):
            model.update(tree_unflatten(list(zip(names, p3))))
            counts = b['counts'] if 'counts' in b else mx.array([b['tokens'].shape[0], b['draws'].shape[0]], mx.uint32)
            w, _ = model(b['lanes'], b['mask'], b['h0'], b['tokens'], b['pos'], b['seg'], seed, counts[0])
            loss = surrogate(w, b['w'], b['draws'], b['first'], b['scale0'], b['scale'], counts)[0]
            return loss.sum(), loss
        (_, loss), g = mx.value_and_grad(f)([mx.broadcast_to(v[None], (TASKS,) + v.shape) for _, v in params])
        model.update(tree_unflatten(params))
        return mx.concatenate([x.reshape(TASKS, -1) for x in g], 1), loss

    def _gradient(self, b, acc):
        J, loss = self.grads(b)
        rows = b['counts'][0].astype(mx.float32) if 'counts' in b else mx.array(b['tokens'].shape[0], mx.float32)   # a padded batch's own rows
        n = (J * J).sum(1)
        tail = mx.concatenate([rows[None], n, mx.ones((1,)), mx.zeros((3,))]) * (rows > 0)
        self.rng['seed'] = (self.rng['seed'] * 1664525 + 1013904223) | 1
        return acc + mx.concatenate([(rows * J).reshape(-1), tail]), loss

    def zero(self):
        return mx.zeros((TASKS * self.P + 8,))

    def _apply(self, acc):
        tail = acc[TASKS * self.P:]; rows = tail[0]; k = mx.where(rows > 0, 1.0 / rows, 0.0)
        g = acc[:TASKS * self.P].reshape(TASKS, self.P) * k
        alpha = upgrad(g @ g.T)
        d = alpha @ g
        out, at = [], 0
        for name, v in tree_flatten(self.model.trainable_parameters()):
            out.append((name, d[at:at + v.size].reshape(v.shape))); at += v.size
        self.opt.update(self.model, tree_unflatten(out))
        self.model.pin()
        cnt = tail[4]
        return mx.concatenate([rows[None], mx.where(cnt > 0, tail[1:4] / cnt, 0.0), cnt[None], mx.where(cnt > 0, rows / cnt, 0.0)[None], (g * g).sum(1), alpha])

    def apply(self, acc, exchange=None):
        """the step on the accumulated sum: exchanged (the sums, the rows and the microbatches' squared norms and count in
        one exchange), then the mean, UPGrad, Adam, pin. Returns the statistics, identical at every rank: [0] total rows,
        [1..3] the mean microbatch squared norm (a task each), [4] the microbatches, [5] their mean rows, [6..8] the
        rows-weighted mean's squared norms, [9..11] UPGrad's weights"""
        return self._finish(exchange(acc) if exchange else acc)

    def step(self, b, exchange=None):
        acc, loss = self.gradient(b, self.zero())
        return self.apply(acc, exchange), loss


class Exchange:
    """the ranks' sum of a flat float array through the mesh (libroute's route_exchange_sum: libnccl-mesh's declared
    all-reduce over a lease of its own); the step's one exchange carries the gradient sums, the rows and the
    microbatches' squared norms and count, so every rank derives the same statistics"""

    def __init__(self, lib, ffi, rank, world, mesh_id, n):
        self.lib, self.ffi, self.n = lib, ffi, n
        if lib.route_exchange_join(rank, world, mesh_id.encode(), n) != 0:
            raise RuntimeError(ffi.string(lib.route_exchange_failure()).decode())

    def __call__(self, acc):
        h = np.array(acc, np.float32)
        if self.lib.route_exchange_sum(self.ffi.from_buffer('float[]', h, require_writable=True), self.n) != 0:
            raise RuntimeError(self.ffi.string(self.lib.route_exchange_failure()).decode())
        return mx.array(h)


def theta(model):
    """the parameters as NetLayout's theta (model.swift: each matrix's V, g, then b), the fp32 master copy"""
    return mx.concatenate([x.reshape(-1) for m in model.mats() for x in ([m.V, m.g] + ([m.b] if 'b' in m else []))])


def load_theta(model, t):
    """theta (NetLayout's) into the model's parameters"""
    t, at = mx.array(t), 0
    for m in model.mats():
        for name in ('V', 'g', 'b'):
            if name in m:
                v = getattr(m, name); setattr(m, name, t[at:at + v.size].reshape(v.shape)); at += v.size
