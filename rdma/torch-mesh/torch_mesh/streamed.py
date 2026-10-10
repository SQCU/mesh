"""Streamed coded weights in the mesh library (streamed.metal): a matrix stored as tile codes, a torch tensor whose
operations are the decompression producer with their consumers as operands, encoded on torch's own stream.

An export (metal-microbench tools/model_code.py `code ...+export=`, format fragment-1: metal-microbench
docs/kernels.md#fragment-tiles) holds, a matrix: its codes (bands of 32 rows, each 32 x 32 tile a set of word planes in
the lanes of matmul2d's right-input cooperative tensor), a tile word a tile (its offset from its band's base, width 0
to 12 and fp16 step) and a base a band; its row order and scale and column order and scale (A[row_order[r],
col_order[j]] = value[r][j] * row_scale[r] * col_scale[j]); and for some a basis of its input, the matrix's columns
rotated coordinates (A_orig = A V^T): a rotation (int8, its scale a column), or a randomized Hadamard transform, V^T =
T = (H_K (x) H_N) diag(signs) / sqrt(K N) from its signs (int8, one an input column) and H_K (`hadamard`, int8 K x K;
H_N Sylvester's), run as a butterfly (streamed_hadamard).

  Streamed(path)[name]                a coded matrix, a torch.Tensor of the matrix's shape: an nn.Linear weight as it is
  F.linear(x, m), x @ m.T, x @ m      the library's kernels, forward and backward (torch's autograd derives the graph)
  m[a:b], m[:, c:d]                   its codes cut (multiples of 32: a range of bands, or of each band's tile words)
  m.to('cpu') or m.dense()            decodes it
  Matrix.ffn(x, gate, up, down)       the fused GELU FFN couple (torch_mesh.engines.ffn runs it where no gradient is due)

x @ m is one pipeline whichever way m is transposed: the input side's order and scale (a rotation where it lies there),
the product contracting the coded rows (streamed_alu, streamed_coop, streamed_panel) or the coded columns
(streamed_panel_t: the same decoded tiles consumed the other way), the output side's order and scale (and rotation).
Each product is an instance with literal sizes, the matrix's widths and the node's choice for its rows
(streamed-nodes.json: the node's instance, simdgroups, bands and K shares a regime; the cooperative tensor's slots read
from the node by streamed_coopmap), compiled on first use. Operands are fp16 on the GPU (another dtype's rows scaled
into its range by powers of two, results finished in the caller's dtype); the codes are constants (no gradient reaches
them, only through them). An operation a coded matrix lacks is an error naming the ones it has.

  python -m torch_mesh.streamed check EXPORT [NAME ...]   the decode against model_code's reference decode; products,
                                                          their gradients and the FFN against the decoded matrices;
                                                          each regime's instance against float64
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import torch

SOURCE = Path(__file__).with_name('streamed.metal').read_text()
NODES = json.loads(Path(__file__).with_name('streamed-nodes.json').read_text())
_PIPELINES = {}
aten = torch.ops.aten


def instance(kernel, *arguments):
    """An instance's host name and its explicit instantiation (one line appended to the library source)."""
    args = ', '.join(str(int(a)) for a in arguments)
    name = '_'.join([kernel] + [str(int(a)) for a in arguments])
    return name, f'template [[host_name("{name}")]] [[kernel]] decltype({kernel}<{args}>) {kernel}<{args}>;'


def _pipeline(name, line=''):
    from . import _stream
    if name not in _PIPELINES:
        _PIPELINES[name] = _stream.pipeline(SOURCE + '\n' + line + '\n', name)
    return _PIPELINES[name]


def _dims(tiles=0, outputs=0, rows=0, per=0, stride=0, shares=0, split=0, finish=0, count=0, columns=0, inputs=0,
          cap=0.0, eps=0.0, flags=0, layer=0, pitch=0):
    """streamed_dims, field for field: its words as the constant block _stream.encode binds at buffer 15."""
    floats = [struct.unpack('<i', struct.pack('<f', v))[0] for v in (cap, eps)]
    return [tiles, outputs, rows, per, stride, shares, split, finish, count, columns, inputs, *floats, flags, layer, pitch]


def node():
    """This node's name (its chip) and GPU cores."""
    import re
    import subprocess
    if 'node' not in _PIPELINES:
        chip = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
        cores = re.search(r'"gpu-core-count" = ([0-9]+)', subprocess.run(['ioreg', '-l'], capture_output=True).stdout.decode('utf-8', 'replace'))
        _PIPELINES['node'] = (chip, int(cores.group(1)) if cores else 8)
    return _PIPELINES['node']


def regime(rows, op='product', bands_per_core=0.0, weights=0):
    """The node's choice for a product of `rows` input rows (op product, couple or head): the first of its regimes
    (streamed-nodes.json, the node's or the default's) covering the rows (and, where it states them, the bands a core
    and the matrix's weights), the op's own where it has one."""
    entries = NODES.get(node()[0], NODES['default'])
    for wanted in (op, 'product'):
        for entry in entries:
            if (entry['op'] == wanted and rows <= entry['rows'] and bands_per_core <= entry.get('bands_per_core', float('inf'))
                    and weights <= entry.get('max_weights', float('inf'))):
                return entry
    raise ValueError(f'no regime for {rows} rows')


def coop_slots():
    """The node's cooperative-tensor slots as streamed_coop's COOP (docs/kernels.md#fragment-tiles), or None where
    matmul2d<M, 32, 32>'s right-input map is not the canonical lanes' up to bit permutations (M 8 and 16 read)."""
    if 'coop' in _PIPELINES:
        return _PIPELINES['coop']
    from . import _stream
    found = None
    for m in (8, 16):
        out = torch.zeros(32 * 130, dtype=torch.int32, device='mps')
        xs = torch.zeros(32 * 32, dtype=torch.float16, device='mps')
        _stream.encode(_pipeline(f'streamed_coopmap_{m}'), [None, out, None, None, xs] + [None] * 10, _dims(), 15, 1, 1, 32)
        slots = coop_of(out.cpu().view(32, 130).tolist())
        if slots is None or (found is not None and slots != found):
            found = None
            break
        found = slots
    _PIPELINES['coop'] = found
    return found


def coop_of(maps):
    """COOP from a streamed_coopmap readout (32 lanes of 130 ints), or None."""
    kb = [((l >> 1) & 3) | (((l >> 4) & 1) << 2) for l in range(32)]
    c = [(l & 1) | (((l >> 3) & 1) << 1) for l in range(32)]
    if any(maps[l][0] != 32 for l in range(32)):
        return None
    right = [[(maps[l][1 + 2 * i], maps[l][2 + 2 * i]) for i in range(32)] for l in range(32)]
    sets = {}
    for l in range(32):
        sets.setdefault(c[l], sorted({n for n, _ in right[l]}))
        if sorted({n for n, _ in right[l]}) != sets[c[l]] or len(sets[c[l]]) != 8:
            return None
    sigma = {}
    for cc, ns in sets.items():
        for q, n in enumerate(ns):
            sigma[n] = 8 * cc + q
    if sorted(sigma) != list(range(32)):
        return None
    element = {}
    for l in range(32):
        for i, (n, k) in enumerate(right[l]):
            j, rest = divmod(k - kb[l], 8)
            q = sigma[n] - 8 * c[l]
            if rest or not 0 <= j < 4 or not 0 <= q < 8 or element.setdefault((q, j), i) != i:
                return None
    lows = [element[(2 * (P >> 2), P & 3)] for P in range(16)]
    if any(element[(2 * (P >> 2) + 1, P & 3)] != lows[P] + 1 for P in range(16)):
        return None
    position = lambda v: v.bit_length() - 1 if v and not v & (v - 1) else None
    a, b, cj, d = position(lows[4]), position(lows[8]), position(lows[1]), position(lows[2])
    if None in (a, b, cj, d) or any(lows[P] != ((P >> 2 & 1) << a | (P >> 3) << b | (P & 1) << cj | (P >> 1 & 1) << d) for P in range(16)):
        return None
    source = []
    for bit in range(5):
        hits = [s for s in range(5) if all(((sigma[n] >> bit) & 1) == ((n >> s) & 1) for n in range(32))]
        if not hits:
            return None
        source.append(hits[0])
    return a | b << 3 | cj << 6 | d << 9 | sum(s << (12 + 3 * i) for i, s in enumerate(source))


def _encode(name, buffers, dims, gx, gy, threads, line=''):
    from . import _stream
    _stream.encode(_pipeline(name, line), list(buffers) + [None] * (15 - len(buffers)), dims, 15, gx, gy, threads)


def _half(n, width):
    return torch.empty(n, width, dtype=torch.float16, device='mps')


_IDENTITIES = {}


def _identity(k):
    if k not in _IDENTITIES:
        _IDENTITIES[k] = torch.arange(k, dtype=torch.int32, device='mps')
    return _IDENTITIES[k]


_TYPES = {torch.float16: '', torch.float32: '_float', torch.bfloat16: '_bfloat'}


def _hadamard(signs, mix):
    """T = (H_K (x) H_N) diag(signs) / sqrt(K N) (float32, on the CPU): the rows its rotated coordinates."""
    k, K = signs.shape[0], mix.shape[0]
    H = torch.ones(1, 1)
    while H.shape[0] < k // K:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)])
    return torch.kron(mix.float().cpu(), H) * signs.float().cpu()[None, :] / k ** 0.5


class Matrix(torch.Tensor):
    """One coded matrix (module docstring) as a tensor: `rows` x `columns` the coded matrix's, its shape theirs or,
    `transposed`, swapped."""
    FIELDS = ('codes', 'tiles', 'bands', 'row_scale', 'col_scale', 'row_order', 'col_order', 'rotation',
              'rotation_scale', 'folded', 'signs', 'hadamard')
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, fields, name='', transposed=False, dtype=torch.float16, widths=None):
        rows, columns = fields['tiles'].shape[0] * 32, fields['tiles'].shape[1] * 32
        out = torch.Tensor._make_wrapper_subclass(cls, (columns, rows) if transposed else (rows, columns), dtype=dtype,
                                                  device=fields['codes'].device)
        for k in cls.FIELDS:
            setattr(out, k, fields.get(k))
        out.label, out.transposed, out.rows, out.columns, out.tiles_ = name, transposed, rows, columns, columns // 32
        out.widths = widths if widths is not None else sum(1 << b for b in ((fields['tiles'].cpu().long() >> 12) & 15).unique().tolist())
        return out

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def load(cls, tensors, name):
        """The matrix `name` of an export's tensors (fragment-1)."""
        assert f'{name}.tiles' in tensors, f'{name}: not a fragment-1 export (tools/model_code.py repack makes one)'
        words = lambda k: tensors[f'{name}.{k}'].contiguous().view(torch.int32).to('mps')
        get = lambda k: tensors[f'{name}.{k}'].to('mps')
        codes = tensors[f'{name}.codes'].contiguous()
        fields = {'codes': codes.view(torch.int32).to('mps'), 'tiles': words('tiles'), 'bands': words('bands'),
                  **{k: get(k) for k in cls.FIELDS[3:7]}}
        if f'{name}.rotation' in tensors:
            fields['rotation'] = tensors[f'{name}.rotation'].T.contiguous().to('mps')
            fields['rotation_scale'] = get('rotation_scale')
            order = fields['col_order'].long()
            fields['folded'] = (fields['rotation'][order].float() * fields['rotation_scale'][order, None]).half()
        if f'{name}.signs' in tensors:
            fields['signs'], fields['hadamard'] = get('signs'), get('hadamard')
            k, K = fields['signs'].shape[0], fields['hadamard'].shape[0]
            assert k // K in (32, 64, 128) and K <= 20 and K % (256 // (k // K)) == 0 and k <= 2560, f'{name}: a Hadamard basis streamed_hadamard runs'
            fields['folded'] = _hadamard(tensors[f'{name}.signs'], tensors[f'{name}.hadamard'])[tensors[f'{name}.col_order'].long()].half().to('mps')
        return cls(fields, name)

    def like(self, transposed=None, dtype=None, **fields):
        return Matrix({**{k: getattr(self, k) for k in self.FIELDS}, **fields}, self.label,
                      self.transposed if transposed is None else transposed, dtype or self.dtype, self.widths)

    def __repr__(self):
        return (f'Matrix({self.label} [{self.rows} x {self.columns}]{" rotated" if self.folded is not None else ""}'
                f'{" transposed" if self.transposed else ""}, {self.dtype})')

    def __tensor_flatten__(self):
        return [k for k in self.FIELDS if getattr(self, k) is not None], (self.label, self.transposed, self.dtype, self.widths)

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
        return [self.codes, self.tiles, self.bands]

    @property
    def pitch(self):
        return self.tiles.stride(0)

    def cut(self, rows=None, columns=None):
        """Coded rows [a, b) and columns [c, d) (multiples of 32, each order keeping its range: they are then the
        logical rows and columns [a, b) and [c, d)) as a matrix of their own: its bands' range and each band's tile
        words' range, the codes where they are."""
        a, b = rows or (0, self.rows)
        c, d = columns or (0, self.columns)
        if (a, b, c, d) == (0, self.rows, 0, self.columns):
            return self.like()
        assert all(v % 32 == 0 for v in (a, b, c, d)), f'{self.label}: a cut in multiples of 32, not [{a}:{b}, {c}:{d}]'
        assert self.folded is None or (c, d) == (0, self.columns), f'{self.label}: rotated columns are not cut'

        def order(values, lo, hi):
            shifted = values[lo:hi] - lo
            assert bool(((shifted >= 0) & (shifted < hi - lo)).all()), f'{self.label}: the order leaves [{lo}, {hi})'
            return shifted.contiguous()
        out = self.like(tiles=self.tiles[a // 32:b // 32, c // 32:d // 32], bands=self.bands[a // 32:b // 32],
                        row_scale=self.row_scale[a:b].contiguous(), col_scale=self.col_scale[c:d].contiguous(),
                        row_order=order(self.row_order, a, b), col_order=order(self.col_order, c, d))
        out.label = f'{self.label}[{a}:{b}, {c}:{d}]'
        return out

    def dispatch(self, rows, op='product'):
        """The instance of a product of `rows` coded input rows on this node (regime): its name and instantiation,
        threadgroups x and y, threads, K shares."""
        entry, T = regime(rows, op, weights=self.rows * self.columns), self.tiles_
        if entry['kernel'] == 'panel':
            name, line = instance('streamed_panel', entry['M'], T, self.widths)
            return name, line, self.rows // 32, -(-rows // entry['M']), 256, 1
        J = max(1, min(entry.get('shares') or -(-T // entry['share']), T))
        J = -(-T // -(-T // J))
        bands = entry['bands'] if (self.rows // 32) % entry['bands'] == 0 else 1
        sk, r = entry['SK'], entry.get('R')
        if entry['kernel'] == 'coop':
            if coop_slots() is not None:
                name, line = instance('streamed_coop', entry['M'], sk, bands, J, T, self.widths, coop_slots())
                return name, line, self.rows // 32 // bands, J, 32 * sk * bands, J
            r, sk, bands = (8 if rows <= 8 else 16), 8, 1
        name, line = instance('streamed_alu', r, sk, bands, J, T, self.widths, 0, entry.get('lut', 0) if op == 'head' else 0, entry.get('F', 1))
        return name, line, self.rows // 32 // bands, J, 32 * sk * bands, J

    def dense(self):
        """The matrix decoded in its own coordinates, its shape and dtype (a rotated matrix decoded in its coded
        columns, then times `folded`: the rotation V^T in coded column order, its scale applied)."""
        out = _half(self.rows, self.columns)
        columns = _identity(self.columns) if self.folded is not None else self.col_order
        _encode('streamed_dense', self._code() + [None, out, self.row_order, self.row_scale, None, columns, self.col_scale],
                _dims(tiles=self.tiles_, pitch=self.pitch), self.rows // 32, self.tiles_, 32)
        if self.folded is not None:
            out = torch.mm(out, self.folded)
        return (out.T if self.transposed else out).contiguous().to(self.dtype)

    def coded_input(self, x, others=(), tables=None):
        """x [n, columns] fp16 in this matrix's coded input coordinates (and those of up to two matrices sharing its
        column order and rotation): gathered and scaled, or rotated (with `tables` [n, tiles, 8, 16] fp32, the first's
        tables for 1-bit tiles too)."""
        n = x.shape[0]
        matrices = [self, *others]
        outs = [_half(n, self.columns) for _ in matrices]
        if self.folded is None:
            pad = outs + [outs[0]] * (3 - len(outs))
            scales = [m.col_scale for m in matrices] + [self.col_scale] * (3 - len(matrices))
            _encode('streamed_gather', [x, pad[0], self.col_order, scales[0], None, pad[1], pad[2], scales[1], scales[2]],
                    _dims(columns=self.columns, inputs=len(matrices)), -(-self.columns // 256), n, 256)
        elif self.signs is not None and len(outs) <= 2:
            second = outs[1] if len(outs) > 1 else outs[0]
            _encode('streamed_hadamard', [x, outs[0], second, self.col_order, self.col_scale,
                                          (others[0] if others else self).col_scale, self.signs, self.hadamard],
                    _dims(rows=n, columns=self.columns, per=self.columns // self.hadamard.shape[0], inputs=len(matrices)),
                    n, 1, 256)
        elif n <= 16:
            second = outs[1] if len(outs) > 1 else outs[0]
            _encode('streamed_rotate', [x, outs[0], second, self.col_order, self.col_scale,
                                        (others[0] if others else self).col_scale, self.rotation, self.rotation_scale, tables],
                    _dims(rows=n, columns=self.columns, inputs=len(matrices), flags=int(tables is not None)),
                    -(-self.columns // 32), 1, 256)
        else:
            rotated = torch.mm(x, self.folded.T)
            pad = outs + [outs[0]] * (3 - len(outs))
            scales = [m.col_scale for m in matrices] + [self.col_scale] * (3 - len(matrices))
            _encode('streamed_gather', [rotated, pad[0], _identity(self.columns), scales[0], None, pad[1], pad[2],
                                        scales[1], scales[2]],
                    _dims(columns=self.columns, inputs=len(matrices)), -(-self.columns // 256), n, 256)
        return outs

    def product(self, xt, op='product', tables=None):
        """The product of coded inputs xt [n, columns]: partials [shares, n, rows] in coded row order, unscaled (past 16
        rows one share, the panel's); `tables` (coded_input's) where the instance reads 1-bit tiles by tables."""
        n = xt.shape[0]
        name, line, gx, gy, threads, shares = self.dispatch(n, op)
        assert (tables is not None) == (name.startswith('streamed_alu') and name.split('_')[-2] == '1'), f'{name}: tables where it reads them'
        partials = torch.empty(shares, n, self.rows, dtype=torch.float32, device='mps')
        buffers = self._code() + [tables, xt, partials] + ([None] * 5 + [xt, self.row_order, self.row_scale] if n > 16 else [])
        _encode(name, buffers, _dims(tiles=self.tiles_, outputs=self.rows, rows=n, per=-(-self.tiles_ // 8), split=n * self.rows,
                                     pitch=self.pitch), gx, gy, threads, line)
        return partials

    def product_t(self, xt):
        """The transposed product of coded rows xt (n rows of `rows`, gathered and scaled, band-major [rows / 32][n][32]):
        partials [shares, n, columns] in coded column order, unscaled (bands of the coded rows a share)."""
        n, bands = xt.numel() // self.rows, self.rows // 32
        m = 16 if n <= 16 else 64
        per = (4 if bands >= 64 else 1) if n <= 16 else bands
        shares = -(-bands // per)
        partials = torch.empty(shares, n, self.columns, dtype=torch.float32, device='mps')
        name, line = instance('streamed_panel_t', m, self.tiles_, self.widths)
        _encode(name, self._code() + [None, xt, partials],
                _dims(tiles=self.tiles_, outputs=self.rows, rows=n, per=per, split=n * self.columns, pitch=self.pitch),
                -(-self.tiles_ // 8), -(-n // m) * shares, 256, line)
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
        name, line, gx, gy, threads, _ = self.dispatch(n)
        y = _half(n, self.rows)
        _encode(name, self._code() + [None, xt, None, None, None, None, None, None, y, self.row_order, self.row_scale],
                _dims(tiles=self.tiles_, outputs=self.rows, rows=n, per=-(-self.tiles_ // 8), stride=self.rows, finish=1,
                      pitch=self.pitch), gx, gy, threads, line)
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
        if self.folded is None:
            return self.finish(partials, self.col_order, self.col_scale, factor, dtype)
        y = torch.mm(self.finish(partials, _identity(self.columns), self.col_scale), self.folded)
        return y if factor is None else (y.float() / factor[:, None]).to(dtype)

    @staticmethod
    def ffn(x, gate, up, down):
        """The fused GELU FFN couple down(GELU(x gate^T) * (x up^T)) of coded matrices sharing the hidden order (gate's
        and up's rows, down's columns: Streamed's FFNs), x [n, columns]: the input rotated or gathered once for gate and
        up; at one row gate and up together in one kernel with the GELU in its reduction (streamed_couple) where the
        node's regime for its bands a core says so, else their products' partials consumed by streamed_gelu; down's coded input,
        down's product finished."""
        x16 = x.half().contiguous()
        n = x16.shape[0]
        gx, ux = gate.coded_input(x16, (up,))
        hidden = _half(n, gate.rows)
        entry = regime(n, 'couple', gate.rows / 32 / node()[1])
        if n == 1 and entry['kernel'] == 'couple':
            apart = entry.get('apart', 0)
            name, line = instance('streamed_couple', 1, entry['SK'], gate.tiles_, gate.widths | up.widths, entry.get('F', 1), apart)
            _encode(name, gate._code() + [None, gx, None] + up._code() + [ux, up.row_scale, hidden, None, gate.row_scale],
                    _dims(tiles=gate.tiles_, outputs=gate.rows, rows=n, pitch=gate.pitch), gate.rows // 32, 1,
                    (64 if apart else 32) * entry['SK'], line)
            return down.apply(hidden).to(x.dtype)
        g, u = gate.product(gx), up.product(ux)
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
        tensors = {k: v for k, v in load_file(str(path)).items() if not k.startswith('__pad_')}
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
            basis = [(getattr(gate, k), getattr(up, k)) for k in ('rotation', 'signs', 'hadamard')]
            shared = gate.folded is not None and torch.equal(gate.col_order, up.col_order) and all(
                (a is None) == (b is None) and (a is None or torch.equal(a, b)) for a, b in basis)
            self.matrices[base + 'up_proj'] = up.like(row_order=identity, **({k: getattr(gate, k) for k in ('rotation', 'signs', 'hadamard', 'folded')}
                                                                             if shared else {}))
            self.matrices[base + 'down_proj'] = down.like(col_order=identity)

    def __getitem__(self, name):
        return self.matrices[name]


def _relative(a, b):
    return float((a.detach().float() - b.detach().float()).norm() / b.detach().float().norm().clamp_min(1e-30))


def _check(args):
    import numpy as np
    sys.path.insert(0, str(Path(args.tools).expanduser()))
    from model_code import decode_layout
    from . import engines
    F = torch.nn.functional
    streamed = Streamed(args.export)
    names = args.names or [n for n in streamed.matrices if '.layers.0.' in n or n == 'lm_head'][:8]
    torch.manual_seed(0)
    print(f'{node()[0]}: cooperative slots {coop_slots()}', flush=True)
    for name in names:
        m = Matrix.load(streamed.tensors, name)
        reference = decode_layout({k: v.numpy() for k, v in streamed.tensors.items() if k.startswith(name + '.')}, name).double()
        dense = m.dense().float()
        line = f'{name} [{m.rows} x {m.columns}]{" rotated" if m.folded is not None else ""}: decode {_relative(dense.cpu(), reference):.2e}'
        for n in (1, 2, 4, 8, 16, 300):
            x = torch.randn(n, m.columns, device='mps').half().requires_grad_()
            w = torch.randn(n, m.rows, device='mps')
            y = F.linear(x, m)
            (y.float() * w).sum().backward()
            exact = x.detach().cpu().double() @ reference.T
            line += f', {n}: {_relative(y.cpu(), exact):.2e} {m.dispatch(n)[0].split("_")[1]} grad {_relative(x.grad.cpu(), w.cpu().double() @ reference):.2e}'
        if name == 'lm_head' and m.folded is not None:
            x = torch.randn(1, m.columns, device='mps').half()
            tables = torch.empty(1, m.tiles_, 8, 16, dtype=torch.float32, device='mps')
            xt = m.coded_input(x, tables=tables)[0]
            y = m.finish(m.product(xt, op='head', tables=tables))
            line += f', head with tables {_relative(y.cpu(), x.cpu().double() @ reference.T):.2e} ({m.dispatch(1, "head")[0]})'
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
