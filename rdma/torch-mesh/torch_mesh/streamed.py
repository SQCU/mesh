"""Streamed coded weights in the mesh library (streamed.metal): a matrix stored as tile codes, its products run by the
decompression producer with their consumers as operands, encoded on torch's own stream (_stream.encode).

An export (metal-microbench tools/model_code.py `code ...+export=`) holds, a matrix: 32 x 32 tiles, each a width
(bits a value, 0 to 12), a step and a word offset into `codes` (a lane a tile row, two values a word); its row order
and scale and column order and scale (A[row_order[r], col_order[j]] = value[r][j] * row_scale[r] * col_scale[j]); and
for some a rotation (int8, its scale a column: the matrix's columns are rotated coordinates, A_orig = A V^T).

  Streamed(path)                      every coded matrix of an export, by name
  Matrix.dense()                      the matrix decoded, in its own coordinates (streamed_dense, the rotation applied)
  Matrix.linear(x)                    x A^T: the input transform (gather, or rotation), the product, its finish
  ffn(x, gate, up, down)              the GELU FFN: gate and up products, GELU(gate) * up (streamed_gelu), down

A product of one row is streamed_product (a lane a tile row, a simdgroup a tile); of 2 to 16 rows the 16-row panel; of
more the 128-row panel, which finishes its own outputs. Products of up to 16 rows leave partials (a share a K split)
that their consumer sums (streamed_finish, streamed_gelu).

  python -m torch_mesh.streamed check EXPORT [NAME ...]   decode against model_code's reference decode, products and
                                                          the FFN against the dense ones"""
import argparse
import struct
import sys
from pathlib import Path

import torch

from . import _stream

SOURCE = Path(__file__).with_name('streamed.metal').read_text()
_PIPELINES = {}
SIMDGROUPS = 8


def _pipeline(name):
    if name not in _PIPELINES:
        _PIPELINES[name] = _stream.pipeline(SOURCE, name)
    return _PIPELINES[name]


def _dims(tiles=0, outputs=0, rows=0, per=0, stride=0, shares=0, split=0, finish=0, count=0, columns=0, inputs=0,
          cap=0.0, eps=0.0):
    """streamed_dims, field for field: its words as the constant block _stream.encode binds at buffer 15."""
    floats = [struct.unpack('<i', struct.pack('<f', v))[0] for v in (cap, eps)]
    return [tiles, outputs, rows, per, stride, shares, split, finish, count, columns, inputs, *floats]


def _encode(name, buffers, dims, gx, gy, threads):
    _stream.encode(_pipeline(name), list(buffers) + [None] * (15 - len(buffers)), dims, 15, gx, gy, threads)


class Matrix:
    """One coded matrix of an export (module docstring), its tensors on the GPU."""

    def __init__(self, tensors, name):
        get = lambda k: tensors[f'{name}.{k}'].to('mps')
        self.name = name
        self.codes, self.widths, self.offsets = get('codes'), get('widths'), get('offsets')
        self.steps, self.row_scale, self.col_scale = get('steps'), get('row_scale'), get('col_scale')
        self.row_order, self.col_order = get('row_order'), get('col_order')
        self.rows, self.columns = self.widths.shape[0] * 32, self.widths.shape[1] * 32
        self.tiles = self.columns // 32
        self.rotation = self.rotation_scale = None
        if f'{name}.rotation' in tensors:
            self.rotation = tensors[f'{name}.rotation'].T.contiguous().to('mps')
            self.rotation_scale = get('rotation_scale')

    def _code(self):
        return [self.codes, self.widths, self.steps, self.offsets]

    def splits(self, rows):
        per = self.per(rows)
        return -(-self.tiles // (SIMDGROUPS * per))

    def per(self, rows):
        return 4 if 1 < rows <= 16 and self.tiles >= 64 else 1

    def dense(self):
        """The matrix in its own coordinates, fp16: decoded with its row and column orders and scales; a rotated
        matrix decoded in its rotated coordinates, then times V^T."""
        out = torch.empty(self.rows, self.columns, dtype=torch.float16, device='mps')
        identity = torch.arange(self.columns, dtype=torch.int32, device='mps')
        columns = identity if self.rotation is not None else self.col_order
        _encode('streamed_dense', self._code() + [out, self.row_order, self.row_scale, None, columns, self.col_scale],
                _dims(tiles=self.tiles), self.rows // 32, self.tiles, 32)
        if self.rotation is None:
            return out
        rotated = torch.zeros_like(out, dtype=torch.float32)
        rotated[:, self.col_order.long()] = out.float()
        v = self.rotation.T.float() * self.rotation_scale[None, :]
        return (rotated @ v.T).half()

    def coded_input(self, x, others=()):
        """x [n, columns] fp16 in this matrix's coded input coordinates (and those of up to two matrices sharing its
        column order and rotation): gathered and scaled, or rotated."""
        n = x.shape[0]
        matrices = [self, *others]
        outs = [torch.empty(n, self.columns, dtype=torch.float16, device='mps') for _ in matrices]
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
            v = self.rotation.T.float() * self.rotation_scale[None, :]
            rotated = (x.float() @ v)[:, self.col_order.long()]
            for m, out in zip(matrices, outs):
                out.copy_((rotated * m.col_scale[None, :]).half())
        return outs

    def product(self, xt):
        """The product of coded inputs xt [n, columns]: partials [shares, n, rows] (n <= 16)."""
        n = xt.shape[0]
        shares = self.splits(n)
        partials = torch.empty(shares, n, self.rows, dtype=torch.float32, device='mps')
        if n == 1:
            _encode('streamed_product', self._code() + [xt, partials],
                    _dims(tiles=self.tiles, outputs=self.rows, split=n * self.rows), self.rows // 32, shares, SIMDGROUPS * 32)
        else:
            _encode('streamed_panel_16', self._code() + [xt, partials, None, None, None, None, None, xt, self.row_order, self.row_scale],
                    _dims(tiles=self.tiles, outputs=self.rows, rows=n, per=self.per(n), split=n * self.rows),
                    self.rows // 32, shares, 256)
        return partials

    def finish(self, partials, order=None, cap=0.0):
        """Partials [shares, n, rows] finished into y [n, rows] at their logical rows (`order`, default the row order)."""
        shares, n, rows = partials.shape
        y = torch.empty(n, rows, dtype=torch.float16, device='mps')
        _encode('streamed_finish', [partials, y, self.row_order if order is None else order, self.row_scale],
                _dims(outputs=rows, stride=rows, shares=shares, split=n * rows, count=n * rows, cap=cap),
                -(-(n * rows) // 256), 1, 256)
        return y

    def panel(self, xt, order=None, cap=0.0):
        """The product of more than 16 coded input rows, finished by the 128-row panel into y [n, rows]."""
        n = xt.shape[0]
        y = torch.empty(n, self.rows, dtype=torch.float16, device='mps')
        _encode('streamed_panel_128', self._code() + [xt, None, None, None, None, None, None, y,
                                                      self.row_order if order is None else order, self.row_scale],
                _dims(tiles=self.tiles, outputs=self.rows, rows=n, per=-(-self.tiles // 8), stride=self.rows, finish=1, cap=cap),
                self.rows // 32, -(-n // 128), 256)
        return y

    def linear(self, x, cap=0.0):
        """x [n, columns] fp16 (the matrix's own input coordinates) times A^T: y [n, rows] fp16."""
        xt, = self.coded_input(x.half().contiguous())
        return self.finish(self.product(xt), cap=cap) if x.shape[0] <= 16 else self.panel(xt, cap=cap)


def ffn(x, gate, up, down):
    """The GELU FFN down(GELU(x gate^T) * (x up^T)) of coded matrices sharing the hidden order (the export's: gate's and
    up's rows and down's columns in one order, down's column scale folded into up's rows)."""
    x = x.half().contiguous()
    n = x.shape[0]
    gx, ux = gate.coded_input(x, (up,))
    identity = torch.arange(gate.rows, dtype=torch.int32, device='mps')
    if n <= 16:
        g, u = gate.product(gx), up.product(ux)
        hidden = torch.empty(n, gate.rows, dtype=torch.float16, device='mps')
        _encode('streamed_gelu', [g, u, gate.row_scale, up.row_scale, hidden],
                _dims(outputs=gate.rows, shares=g.shape[0], split=n * gate.rows, count=n * gate.rows), -(-(n * gate.rows) // 256), 1, 256)
        return down.finish(down.product(hidden))
    g, u = gate.panel(gx, order=identity), up.panel(ux, order=identity)
    a = 0.7978845608 * (g.float() + 0.044715 * g.float() ** 3)
    hidden = (0.5 * g.float() * (1 + torch.tanh(a.clamp(-20, 20))) * u.float()).half()
    return down.panel(hidden)


class Streamed:
    """Every coded matrix of an export, by name."""

    def __init__(self, path):
        from safetensors.torch import load_file
        tensors = load_file(str(path))
        names = sorted({k.rsplit('.', 1)[0] for k in tensors if k.endswith('.codes')})
        self.matrices = {name: Matrix(tensors, name) for name in names}
        self.tensors = tensors

    def __getitem__(self, name):
        return self.matrices[name]


def _relative(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def _check(args):
    sys.path.insert(0, str(Path(args.tools).expanduser()))
    from model_code import decode_layout
    streamed = Streamed(args.export)
    names = args.names or [n for n in streamed.matrices if '.layers.0.' in n or n == 'lm_head'][:8]
    torch.manual_seed(0)
    for name in names:
        m = streamed[name]
        reference = decode_layout(streamed.tensors, name).half()
        dense = m.dense()
        line = f'{name} [{m.rows} x {m.columns}]{" rotated" if m.rotation is not None else ""}: decode {_relative(dense.cpu(), reference):.2e}'
        for n in (1, 8, 300):
            x = torch.randn(n, m.columns, device='mps').half()
            line += f', {n} rows {_relative(m.linear(x), x.float() @ dense.float().T):.2e}'
        print(line, flush=True)
    base = names[0].split('.mlp.')[0] + '.mlp.' if '.mlp.' in names[0] else 'model.language_model.layers.0.mlp.'
    gate, up, down = (streamed[base + p] for p in ('gate_proj', 'up_proj', 'down_proj'))
    gd, ud, dd = gate.dense().float(), up.dense().float(), down.dense().float()
    for n in (1, 8, 300):
        x = torch.randn(n, gate.columns, device='mps').half()
        g, u = x.float() @ gd.T, x.float() @ ud.T
        a = 0.7978845608 * (g + 0.044715 * g ** 3)
        y = (0.5 * g * (1 + torch.tanh(a.clamp(-20, 20))) * u) @ dd.T
        print(f'{base}ffn {n} rows: {_relative(ffn(x, gate, up, down), y):.2e}', flush=True)


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
