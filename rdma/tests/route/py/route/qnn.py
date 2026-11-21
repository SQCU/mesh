"""The routing policy in the PyTorch dialect (MLX on Apple GPUs): ops, modules, loss, step.

  ops      quantize (int8mm.h: 64-block Walsh-Hadamard rotation as a product, absmax/127 codes) and i8mm, the int8 x
           int8 -> int32 product: a Metal matmul2d kernel registered as an MLX op, its VJP int8 products too
  modules  QLinear (W = g V/|V| a row, bias), MinGRU (gates, the recurrence, the feed-forward), Policy
  loss     surrogate: the clipped importance-weighted score-function objective whose gradient is the credit
  step     nn.value_and_grad + optim.Adam, compiled with mx.compile

Arrays are values the array library makes; modules hold parameters; nothing allocates by hand."""
import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

D, E, IN, FS = 64, 192, 48, 40

# ---- ops ----
_H64 = (lambda h: h)(None)


def _hadamard(n):
    h = mx.array([[1.0]])
    while h.shape[0] < n:
        h = mx.concatenate([mx.concatenate([h, h], 1), mx.concatenate([h, -h], 1)], 0)
    return h / math.sqrt(n)


_H = _hadamard(64)


def quantize(x):
    """rows (last axis, padded to 64-blocks) rotated by the 64-point Hadamard (a product), int8 codes, a scale a row"""
    k = x.shape[-1]; kp = -(-k // 64) * 64
    x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, kp - k)]) if kp != k else x
    y = (x.reshape(-1, kp // 64, 64) @ _H).reshape(-1, kp)
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


def _i8(qa, qb):
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
    return (_i8(qx, qw).astype(mx.float32) * sx * sw.T).reshape(*lead, w.shape[0])


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

    def __call__(self, x):
        W = self.g[:, None] * self.V / mx.linalg.norm(self.V, axis=1, keepdims=True)
        y = i8mm(x, W)
        return y + self.b if 'b' in self else y


class MinGRU(nn.Module):
    """a layer: gates z, h~ from x; the recurrence over positions; the feed-forward on x + h"""
    def __init__(self):
        super().__init__()
        self.zh, self.up, self.down = QLinear(2 * D, D), QLinear(E, D), QLinear(D, E)

    def gates(self, x):
        zc = self.zh(x)
        return mx.sigmoid(zc[..., :D]), mx.tanh(zc[..., D:])

    def ffn(self, xm):
        return xm + self.down(nn.relu(self.up(xm)))

    def sequence(self, x, mask, h0):   # lanes [runs][L][D]: every position's state, then the feed-forward
        z, c = self.gates(x); h, hs = h0, []
        for t in range(x.shape[1]):
            mz = mask[:, t, None] * z[:, t]; h = (1 - mz) * h + mz * c[:, t]; hs.append(h)
        H = mx.stack(hs, 1)
        return self.ffn(x + H), H

    def step(self, x, h):              # context rows: one step from their run's state
        z, c = self.gates(x)
        return self.ffn(x + h + z * (c - h))


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp, self.layers, self.out = QLinear(D, IN), [MinGRU(), MinGRU()], QLinear(FS, D, bias=False)

    def __call__(self, lanes, mask, h0, tokens, pos):
        x, xc, last = self.inp(lanes), self.inp(tokens), []
        for l, layer in enumerate(self.layers):
            x, H = layer.sequence(x, mask, h0[:, l])
            last.append(H[:, -1]); xc = layer.step(xc, H[pos, -1])
        n = xc * mx.rsqrt(mx.mean(xc * xc, axis=-1, keepdims=True) + 1e-6)
        return self.out(n), mx.stack(last, 1)


# ---- loss ----
def surrogate(w, phi, token, logp_b, adv, clip=10.0):
    """draws' splits phi [draws][splits][FS] against their rows' w: log pi(token); clipped ratio x advantage"""
    logits = 10.0 * mx.tanh(mx.einsum('dsf,df->ds', phi, w) / math.sqrt(FS))
    lp = mx.take_along_axis(logits - mx.logsumexp(logits, axis=-1, keepdims=True), token[:, None], axis=-1)[:, 0]
    ratio = mx.clip(mx.exp(mx.stop_gradient(lp) - logp_b), 1 / clip, clip)
    return -mx.mean(ratio * adv * lp)


# ---- step ----
def trainer(model, rate):
    opt = optim.Adam(learning_rate=rate)

    def loss(model, batch):
        w, _ = model(batch['lanes'], batch['mask'], batch['h0'], batch['tokens'], batch['pos'])
        return surrogate(w[batch['row']], batch['phi'], batch['token'], batch['logp_b'], batch['adv'])

    grad_fn = nn.value_and_grad(model, loss)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch):
        value, grads = grad_fn(model, batch)
        opt.update(model, grads)
        return value
    return step, state
