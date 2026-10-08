"""Streamed coded weights in the mesh library (streamed.metal): a matrix stored as tile codes, a torch tensor whose
operations are the decompression producer with their consumers as operands, encoded on torch's own stream.

An export (metal-microbench tools/model_code.py `code ...+export=`) holds, a matrix: 32 x 32 tiles, each a width
(bits a value, 0 to 12), a step and a word offset into `codes` (a lane a tile row, two values a word); its row order
and scale and column order and scale (A[row_order[r], col_order[j]] = value[r][j] * row_scale[r] * col_scale[j]); and
for some a rotation (int8, its scale a column: the matrix's columns are rotated coordinates, A_orig = A V^T).

  Streamed(path)[name]                a coded matrix, a torch.Tensor of the matrix's shape: an nn.Linear weight as it is
  F.linear(x, m), x @ m.T, x @ m      the library's kernels, forward and backward (torch's autograd derives the graph)
  m[a:b], m[:, c:d]                   its codes cut (multiples of 32); m.to('cpu') or m.dense() decodes it
  Matrix.ffn(x, gate, up, down)       the fused GELU FFN couple (torch_mesh.engines.ffn runs it where no gradient is due)

x @ m is one pipeline whichever way m is transposed: the input side's order and scale (a rotation where it lies there),
the product contracting the coded rows (streamed_product, the 16- and 128-row panels) or the coded columns
(streamed_panel_t: the same decoded panels, consumed the other way), the output side's order and scale (and rotation).
Operands are fp16 on the GPU (another dtype's rows scaled into its range by powers of two, results finished in the
caller's dtype); the codes are constants (no gradient reaches them, only through them). An operation a coded matrix lacks is an error naming the ones it has.

  python -m torch_mesh.streamed check EXPORT [NAME ...]   the decode against model_code's reference decode; products,
                                                          their gradients and the FFN against the decoded matrices"""
import argparse
import struct
import sys
from pathlib import Path

import torch

SOURCE = Path(__file__).with_name('streamed.metal').read_text()
_PIPELINES = {}
PRODUCT_SIMDGROUPS = 16
aten = torch.ops.aten


def _pipeline(name):
    from . import _stream
    if name not in _PIPELINES:
        _PIPELINES[name] = _stream.pipeline(SOURCE, name)
    return _PIPELINES[name]


def _dims(tiles=0, outputs=0, rows=0, per=0, stride=0, shares=0, split=0, finish=0, count=0, columns=0, inputs=0,
          cap=0.0, eps=0.0):
    """streamed_dims, field for field: its words as the constant block _stream.encode binds at buffer 15."""
    floats = [struct.unpack('<i', struct.pack('<f', v))[0] for v in (cap, eps)]
    return [tiles, outputs, rows, per, stride, shares, split, finish, count, columns, inputs, *floats]


def _encode(name, buffers, dims, gx, gy, threads):
    from . import _stream
    _stream.encode(_pipeline(name), list(buffers) + [None] * (15 - len(buffers)), dims, 15, gx, gy, threads)


def _half(n, width):
    return torch.empty(n, width, dtype=torch.float16, device='mps')


_TYPES = {torch.float16: '', torch.float32: '_float', torch.bfloat16: '_bfloat'}


class Matrix(torch.Tensor):
    """One coded matrix (module docstring) as a tensor: `rows` x `columns` the coded matrix's, its shape theirs or,
    `transposed`, swapped."""
    FIELDS = ('codes', 'widths', 'offsets', 'steps', 'row_scale', 'col_scale', 'row_order', 'col_order', 'rotation',
              'rotation_scale')
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, fields, name='', transposed=False, dtype=torch.float16):
        rows, columns = fields['widths'].shape[0] * 32, fields['widths'].shape[1] * 32
        out = torch.Tensor._make_wrapper_subclass(cls, (columns, rows) if transposed else (rows, columns), dtype=dtype,
                                                  device=fields['codes'].device)
        for k in cls.FIELDS:
            setattr(out, k, fields.get(k))
        out.label, out.transposed, out.rows, out.columns, out.tiles = name, transposed, rows, columns, columns // 32
        return out

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def load(cls, tensors, name):
        """The matrix `name` of an export's tensors."""
        get = lambda k: tensors[f'{name}.{k}'].to('mps')
        fields = {k: get(k) for k in cls.FIELDS[:8]}
        if f'{name}.rotation' in tensors:
            fields['rotation'] = tensors[f'{name}.rotation'].T.contiguous().to('mps')
            fields['rotation_scale'] = get('rotation_scale')
        return cls(fields, name)

    def like(self, transposed=None, dtype=None, **fields):
        return Matrix({**{k: getattr(self, k) for k in self.FIELDS}, **fields}, self.label,
                      self.transposed if transposed is None else transposed, dtype or self.dtype)

    def __repr__(self):
        return (f'Matrix({self.label} [{self.rows} x {self.columns}]{" rotated" if self.rotation is not None else ""}'
                f'{" transposed" if self.transposed else ""}, {self.dtype})')

    def __tensor_flatten__(self):
        return [k for k in self.FIELDS if getattr(self, k) is not None], (self.label, self.transposed, self.dtype)

    @staticmethod
    def __tensor_unflatten__(inner, context, outer_size, outer_stride):
        return Matrix(dict(inner), *context)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        operation = _OPERATIONS.get(func)
        if operation is None:
            raise NotImplementedError(f'a coded matrix has no {func}; it has {", ".join(sorted(map(str, _OPERATIONS)))}')
        return operation(*args, **(kwargs or {}))

    def _code(self):
        return [self.codes, self.widths, self.steps, self.offsets]

    def cut(self, rows=None, columns=None):
        """Coded rows [a, b) and columns [c, d) (multiples of 32, each order keeping its range: they are then the
        logical rows and columns [a, b) and [c, d)) as a matrix of their own, the tiles' codes repacked."""
        a, b = rows or (0, self.rows)
        c, d = columns or (0, self.columns)
        if (a, b, c, d) == (0, self.rows, 0, self.columns):
            return self.like()
        assert all(v % 32 == 0 for v in (a, b, c, d)), f'{self.label}: a cut in multiples of 32, not [{a}:{b}, {c}:{d}]'
        assert self.rotation is None or (c, d) == (0, self.columns), f'{self.label}: rotated columns are not cut'
        widths = self.widths[a // 32:b // 32, c // 32:d // 32].contiguous()
        old = (self.offsets[a // 32:b // 32, c // 32:d // 32].long() & 0xFFFFFFFF).reshape(-1)
        sizes = 32 * widths.reshape(-1).long()
        starts = torch.cumsum(sizes, 0) - sizes
        tile = torch.repeat_interleave(torch.arange(len(sizes), device='mps'), sizes)
        index = old[tile] + torch.arange(int(sizes.sum()), device='mps') - starts[tile]

        def order(values, lo, hi):
            shifted = values[lo:hi] - lo
            assert bool(((shifted >= 0) & (shifted < hi - lo)).all()), f'{self.label}: the order leaves [{lo}, {hi})'
            return shifted.contiguous()
        out = self.like(codes=torch.cat([self.codes[index], torch.zeros(2, dtype=self.codes.dtype, device='mps')]),
                        widths=widths, offsets=starts.to(torch.int32).reshape(widths.shape),
                        steps=self.steps[a // 32:b // 32, c // 32:d // 32].contiguous(),
                        row_scale=self.row_scale[a:b].contiguous(), col_scale=self.col_scale[c:d].contiguous(),
                        row_order=order(self.row_order, a, b), col_order=order(self.col_order, c, d))
        out.label = f'{self.label}[{a}:{b}, {c}:{d}]'
        return out

    def splits(self, rows):
        if rows == 1:
            return -(-self.tiles // PRODUCT_SIMDGROUPS)
        return -(-self.tiles // (8 * self.per(rows)))

    def per(self, rows):
        return 4 if 1 < rows <= 16 and self.tiles >= 64 else 1

    def dense(self):
        """The matrix decoded in its own coordinates, its shape and dtype (a rotated matrix decoded in its rotated
        coordinates, then times V^T)."""
        out = _half(self.rows, self.columns)
        identity = torch.arange(self.columns, dtype=torch.int32, device='mps')
        columns = identity if self.rotation is not None else self.col_order
        _encode('streamed_dense', self._code() + [out, self.row_order, self.row_scale, None, columns, self.col_scale],
                _dims(tiles=self.tiles), self.rows // 32, self.tiles, 32)
        if self.rotation is not None:
            rotated = torch.zeros_like(out, dtype=torch.float32)
            rotated[:, self.col_order.long()] = out.float()
            out = rotated @ self._v().T
        return (out.T if self.transposed else out).contiguous().to(self.dtype)

    def _v(self):
        return self.rotation.T.float() * self.rotation_scale[None, :]

    def coded_input(self, x, others=()):
        """x [n, columns] fp16 in this matrix's coded input coordinates (and those of up to two matrices sharing its
        column order and rotation): gathered and scaled, or rotated."""
        n = x.shape[0]
        matrices = [self, *others]
        outs = [_half(n, self.columns) for _ in matrices]
        if self.rotation is None:
            pad = outs + [outs[0]] * (3 - len(outs))
            scales = [m.col_scale for m in matrices] + [self.col_scale] * (3 - len(matrices))
            _encode('streamed_gather', [x, pad[0], self.col_order, scales[0], None, pad[1], pad[2], scales[1], scales[2]],
                    _dims(columns=self.columns, inputs=len(matrices)), -(-self.columns // 256), n, 256)
        elif n <= 16:
            second = outs[1] if len(outs) > 1 else outs[0]
            _encode('streamed_rotate', [x, outs[0], second, self.col_order, self.col_scale,
                                        (others[0] if others else self).col_scale, self.rotation, self.rotation_scale],
                    _dims(rows=n, columns=self.columns, inputs=len(matrices)), -(-self.columns // 16), 1, 128)
        else:
            rotated = (x.float() @ self._v())[:, self.col_order.long()]
            for m, out in zip(matrices, outs):
                out.copy_((rotated * m.col_scale[None, :]).half())
        return outs

    def product(self, xt):
        """The product of coded inputs xt [n, columns]: partials [shares, n, rows] in coded row order, unscaled (past 16
        rows one share, the 128-row panel's)."""
        n = xt.shape[0]
        shares = self.splits(n) if n <= 16 else 1
        partials = torch.empty(shares, n, self.rows, dtype=torch.float32, device='mps')
        if n > 16:
            _encode('streamed_panel_128', self._code() + [xt, partials, None, None, None, None, None, xt, self.row_order, self.row_scale],
                    _dims(tiles=self.tiles, outputs=self.rows, rows=n, per=-(-self.tiles // 8), split=n * self.rows),
                    self.rows // 32, -(-n // 128), 256)
        elif n == 1:
            _encode('streamed_product', self._code() + [xt, partials],
                    _dims(tiles=self.tiles, outputs=self.rows, split=n * self.rows), self.rows // 32, shares, PRODUCT_SIMDGROUPS * 32)
        else:
            _encode('streamed_panel_16', self._code() + [xt, partials, None, None, None, None, None, xt, self.row_order, self.row_scale],
                    _dims(tiles=self.tiles, outputs=self.rows, rows=n, per=self.per(n), split=n * self.rows),
                    self.rows // 32, shares, 256)
        return partials

    def product_t(self, xt):
        """The transposed product of coded rows xt (n rows of `rows`, gathered and scaled, band-major [rows / 32][n][32]):
        partials [shares, n, columns] in coded column order, unscaled (bands of the coded rows a share, as the forward's
        panels)."""
        n, bands = xt.numel() // self.rows, self.rows // 32
        m = 16 if n <= 16 else 64
        per = (4 if bands >= 64 else 1) if n <= 16 else bands
        shares = -(-bands // per)
        partials = torch.empty(shares, n, self.columns, dtype=torch.float32, device='mps')
        _encode(f'streamed_panel_t_{m}', self._code() + [xt, partials],
                _dims(tiles=self.tiles, outputs=self.rows, rows=n, per=per, split=n * self.columns),
                -(-self.tiles // 8), -(-n // m) * shares, 256)
        return partials

    def finish(self, partials, order=None, scale=None, factor=None, dtype=torch.float16):
        """Partials [shares, n, k] finished into y [n, k] of `dtype` at their logical positions (`order` and `scale`,
        default the row side's), each row over its `factor` where given (streamed_scale_rows's)."""
        shares, n, k = partials.shape
        y = torch.empty(n, k, dtype=dtype, device='mps')
        _encode('streamed_finish' + _TYPES[dtype], [partials, y, self.row_order if order is None else order,
                                                   self.row_scale if scale is None else scale, factor],
                _dims(outputs=k, stride=k, shares=shares, split=n * k, count=n * k, inputs=int(factor is not None)),
                -(-(n * k) // 256), 1, 256)
        return y

    def apply(self, xt):
        """The product of coded inputs xt [n, columns] finished: y [n, rows] fp16 at its logical rows."""
        n = xt.shape[0]
        if n <= 16:
            return self.finish(self.product(xt))
        y = _half(n, self.rows)
        _encode('streamed_panel_128', self._code() + [xt, None, None, None, None, None, None, y, self.row_order, self.row_scale],
                _dims(tiles=self.tiles, outputs=self.rows, rows=n, per=-(-self.tiles // 8), stride=self.rows, finish=1),
                self.rows // 32, -(-n // 128), 256)
        return y

    def times(self, x):
        """x [n, k] @ this matrix as its shape stands, in x's dtype: x A^T where transposed, else x A. Another dtype's
        rows are scaled into fp16's range by streamed_scale_rows and the products finished over the scale in that
        dtype (a backward's gradients lie far below fp16's normal range: cast as they are, a whole model's gradient a
        position 0.140 relative to the float32 model's, median; scaled 0.050)."""
        n, factor, dtype = x.shape[0], None, x.dtype
        x16 = x.contiguous()
        if dtype != torch.float16:
            x16, factor = _half(n, x.shape[1]), torch.empty(n, dtype=torch.float32, device='mps')
            _encode('streamed_scale_rows' + _TYPES[dtype], [x.contiguous(), x16, factor], _dims(columns=x.shape[1]), n, 1, 256)
        if self.transposed:
            xt = self.coded_input(x16)[0]
            return self.apply(xt) if factor is None else self.finish(self.product(xt), factor=factor, dtype=dtype)
        xt = _half(n, self.rows)
        _encode('streamed_gather', [x16, xt, self.row_order, self.row_scale],
                _dims(columns=self.rows, rows=n, stride=32, inputs=1), -(-self.rows // 256), n, 256)
        partials = self.product_t(xt)
        if self.rotation is None:
            return self.finish(partials, self.col_order, self.col_scale, factor, dtype)
        z = self.finish(partials, self.col_order, self.col_scale, factor, torch.float32)
        return (z @ self._v().T).to(dtype)

    @staticmethod
    def ffn(x, gate, up, down):
        """The fused GELU FFN couple down(GELU(x gate^T) * (x up^T)) of coded matrices sharing the hidden order (gate's
        and up's rows, down's columns: Streamed's FFNs), x [n, columns]: the input rotated or gathered once for gate and
        up, their products' partials consumed by streamed_gelu into down's coded input, down's product finished."""
        x16 = x.half().contiguous()
        n = x16.shape[0]
        gx, ux = gate.coded_input(x16, (up,))
        g, u = gate.product(gx), up.product(ux)
        hidden = _half(n, gate.rows)
        _encode('streamed_gelu', [g, u, gate.row_scale, up.row_scale, hidden],
                _dims(outputs=gate.rows, shares=g.shape[0], split=n * gate.rows, count=n * gate.rows), -(-(n * gate.rows) // 256), 1, 256)
        return down.apply(hidden).to(x.dtype)


def _no(m, what):
    raise NotImplementedError(f'{m.label}: a coded matrix has no {what}')


def _flip(m):
    return m.like(transposed=not m.transposed)


def _mm(a, b):
    if isinstance(b, Matrix):
        return b.times(a)
    return _flip(a).times(b.T.contiguous()).T.contiguous()


def _matmul(a, b):
    if isinstance(a, Matrix) or a.dim() == 2:
        return _mm(a, b)
    return _mm(a.reshape(-1, a.shape[-1]), b).reshape(*a.shape[:-1], b.shape[-1])


def _slice(m, dim=0, start=None, end=None, step=1):
    assert step == 1, 'a coded matrix is cut in unit steps'
    size = m.shape[dim]
    start = 0 if start is None else start + size if start < 0 else min(start, size)
    end = size if end is None else end + size if end < 0 else min(end, size)
    span = (start, end)
    return m.cut(rows=span) if (dim % 2) == int(m.transposed) else m.cut(columns=span)


def _split(m, sizes, dim=0):
    sizes = sizes if isinstance(sizes, (list, tuple)) else [sizes] * -(-m.shape[dim] // sizes)
    bounds = [0]
    for size in sizes:
        bounds.append(min(bounds[-1] + size, m.shape[dim]))
    return [_slice(m, dim, a, b) for a, b in zip(bounds, bounds[1:])]


def _to_copy(m, dtype=None, layout=None, device=None, pin_memory=None, non_blocking=False, memory_format=None):
    if device is not None and torch.device(device).type != m.device.type:
        return m.dense().to(device=device, dtype=dtype or m.dtype)
    return m.like(dtype=dtype)


def _linear(x, weight, bias=None):
    y = _flip(weight).times(x.reshape(-1, x.shape[-1])).reshape(*x.shape[:-1], weight.shape[0])
    return y if bias is None else y + bias


def _linear_backward(x, grad, weight, mask):
    assert not mask[1], f'{weight.label}: a coded matrix is a constant (no gradient for its codes)'
    flat = grad.reshape(-1, grad.shape[-1])
    return (weight.times(flat).reshape(x.shape) if mask[0] else None, None, flat.sum(0) if mask[2] else None)


_OPERATIONS = {
    aten.t.default: _flip,
    aten.numpy_T.default: _flip,
    aten.transpose.int: lambda m, a, b: m.like() if a % 2 == b % 2 else _flip(m),
    aten.permute.default: lambda m, dims: m.like() if [d % 2 for d in dims] == [0, 1] else _flip(m),
    aten.detach.default: lambda m: m.like(),
    aten.view.default: lambda m, shape: m.like() if list(shape) in (list(m.shape), [-1, m.shape[1]], [m.shape[0], -1]) else _no(m, f'a view as {shape}'),
    aten.alias.default: lambda m: m.like(),
    aten.clone.default: lambda m, memory_format=None: m.like(),
    aten._to_copy.default: _to_copy,
    aten.slice.Tensor: _slice,
    aten.split.Tensor: _split,
    aten.split_with_sizes.default: _split,
    aten.mm.default: _mm,
    aten.matmul.default: _matmul,
    aten.mm.out: lambda a, b, out: out.copy_(_mm(a, b)),
    aten.addmm.default: lambda bias, a, b, beta=1, alpha=1: beta * bias + alpha * _mm(a, b),
    aten.linear.default: _linear,
    aten.linear_backward.default: _linear_backward,
}


class Streamed:
    """Every coded matrix of an export, by name: an FFN's hidden in its code order (gate's and up's rows, down's
    columns: one permutation of the neurons, so the FFN is unchanged), so its neurons [a, b) are its codes' rows or
    columns [a, b)."""

    def __init__(self, path):
        from safetensors.torch import load_file
        tensors = load_file(str(path))
        names = sorted({k.rsplit('.', 1)[0] for k in tensors if k.endswith('.codes')})
        self.matrices = {name: Matrix.load(tensors, name) for name in names}
        self.tensors = tensors
        for name in names:
            if not name.endswith('.mlp.gate_proj'):
                continue
            base = name[:-len('gate_proj')]
            gate, up, down = (self.matrices[base + p] for p in ('gate_proj', 'up_proj', 'down_proj'))
            assert torch.equal(gate.row_order, up.row_order) and torch.equal(gate.row_order, down.col_order), base
            identity = torch.arange(gate.rows, dtype=torch.int32, device='mps')
            self.matrices[base + 'gate_proj'] = gate.like(row_order=identity)
            self.matrices[base + 'up_proj'] = up.like(row_order=identity)
            self.matrices[base + 'down_proj'] = down.like(col_order=identity)

    def __getitem__(self, name):
        return self.matrices[name]


def _relative(a, b):
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm().clamp_min(1e-30))


def _check(args):
    sys.path.insert(0, str(Path(args.tools).expanduser()))
    from model_code import decode_layout
    from . import engines
    F = torch.nn.functional
    streamed = Streamed(args.export)
    names = args.names or [n for n in streamed.matrices if '.layers.0.' in n or n == 'lm_head'][:8]
    torch.manual_seed(0)
    for name in names:
        m = Matrix.load(streamed.tensors, name)
        reference = decode_layout(streamed.tensors, name).half()
        dense = m.dense().float()
        line = f'{name} [{m.rows} x {m.columns}]{" rotated" if m.rotation is not None else ""}: decode {_relative(dense.cpu(), reference):.2e}'
        for n in (1, 8, 300):
            x = torch.randn(n, m.columns, device='mps').half().requires_grad_()
            w = torch.randn(n, m.rows, device='mps')
            y = F.linear(x, m)
            (y.float() * w).sum().backward()
            line += f', {n} rows {_relative(y, x.float() @ dense.T):.2e} gradient {_relative(x.grad, w @ dense):.2e}'
        print(line, flush=True)
    base = names[0].split('.mlp.')[0] + '.mlp.' if '.mlp.' in names[0] else 'model.language_model.layers.0.mlp.'
    gate, up, down = (streamed[base + p] for p in ('gate_proj', 'up_proj', 'down_proj'))
    gd, ud, dd = (t.dense().float().requires_grad_(False) for t in (gate, up, down))
    for n in (1, 8, 300):
        x = torch.randn(n, gate.columns, device='mps').half()
        with torch.no_grad():
            fused = engines.ffn(x, gate, up, down)
        x.requires_grad_()
        composed = engines.ffn(x, gate, up, down)
        w = torch.randn_like(composed, dtype=torch.float32)
        (composed.float() * w).sum().backward()
        xf = x.detach().float().requires_grad_()
        y = F.linear(F.gelu(F.linear(xf, gd), approximate='tanh') * F.linear(xf, ud), dd)
        (y * w).sum().backward()
        print(f'{base}ffn {n} rows: fused {_relative(fused, y):.2e}, composed {_relative(composed, y):.2e} '
              f'gradient {_relative(x.grad, xf.grad):.2e}', flush=True)


def main():
    parser = argparse.ArgumentParser(prog='python -m torch_mesh.streamed')
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check')
    check.add_argument('export')
    check.add_argument('names', nargs='*')
    check.add_argument('--tools', default='~/metal-microbench/tools')
    args = parser.parse_args()
    {'check': _check}[args.command](args)


if __name__ == '__main__':
    main()
