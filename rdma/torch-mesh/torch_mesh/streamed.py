"""Streamed coded weights in the mesh library (streamed.metal): a matrix stored as tile codes, a torch tensor whose
operations are the decompression producer with their consumers as operands, encoded on torch's own stream.

An export (metal-microbench tools/model_code.py `code ...+export=`, format fragment-1: metal-microbench
docs/kernels.md#fragment-tiles) holds, a matrix: its codes (bands of 32 rows, each 32 x 32 tile a set of word planes in
the lanes of matmul2d's right-input cooperative tensor), a tile word a tile (its offset from its band's base, width 0
to 12 and fp16 step) and a base a band; its row order and scale and column order and scale (A[row_order[r],
col_order[j]] = value[r][j] * row_scale[r] * col_scale[j]); and for some a basis of its input, the matrix's columns
rotated coordinates (A_orig = A V^T): a rotation (int8, in coded column order: row j the basis vector of coded column
j, its scale rotation_scale[j]; metadata basis_layout code-order), or a randomized Hadamard transform, V^T =
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
Each product is an instance with literal sizes, the matrix's widths and the node's choice for its rows (the instance
fit, below: the argmin of a model of the node's times over the instances' units; the cooperative tensor's slots read
from the node by streamed_coopmap), compiled on first use. Operands are fp16 on the GPU (another dtype's rows scaled
into its range by powers of two, results finished in the caller's dtype); the codes are constants (no gradient reaches
them, only through them). An operation a coded matrix lacks is an error naming the ones it has.

  python -m torch_mesh.streamed check EXPORT [NAME ...]   the decode against model_code's reference decode; products,
                                                          their gradients and the FFN against the decoded matrices;
                                                          each chosen instance against float64

The instance fit (metal-microbench docs/kernels.md#the-instance-fit): an instance's time on a slice is a model of its
units, t = c0 + smooth max(throughput, chain, bytes / bandwidth), the throughput the slice's tile visits, row work,
planes, simdgroups, partials and register pressure over the node's cores, the chain a simdgroup's dependent steps over
the waves the node holds; one set of coefficients a kernel family (streamed_alu with streamed_couple, streamed_coop,
streamed_panel) a node, fitted by least squares in log space with a ridge toward the prior from the node's
observations (each probe's and profile's measured instance: rows of the evidence file, MESH_INSTANCES or
~/.cache/mesh/instances.json, which holds the coefficients too; a node without evidence prices by the prior). A
choice is the argmin over the admissible instances of a slice at its rows; a measurement moves the coefficients and
never stands for its key.

  python -m torch_mesh.streamed fit OBSERVATIONS ... --gbps G   this node's evidence fitted (each file a probe's record:
                                                                rows of instance, slice units, rows_in, best_us; or
                                                                JSON lines: observe's, and the decode attention plans'
                                                                rows of plan and KV bytes read, kind attention)
  python -m torch_mesh.streamed resolve EXPORT [--program RECORD --rank R] [--classes 1,2,4,8,16,256] --out TABLE
                                                                each slice's chosen instance at each rows class (a slice's
                                                                key its own widths; a product whose input row folds into
                                                                it priced with that row's dispatch at other threads,
                                                                --folds), each argument named, and each attention class's
                                                                plan: the table an engine reads (LM_STREAMED_INSTANCES)
  python -m torch_mesh.streamed observe PROFILE EXPORT [--program RECORD --rank R] --out ROWS
                                                                a direct step's profile (LM_PROFILE_DECODE) as observations
                                                                of its instances, appended to ROWS (JSON lines)

A factor of the fit (a configuration's efficiency at a rows class) rests on two slice shapes at least; one seen on a
single shape takes its family's model. A decode attention plan (metal-microbench docs/kernels.md#decode-attention: the
vector form, a task a query head and row; the tile plan, a task 8 of one KV head's rows) is priced c0 + c1 MB of the KV
it reads, one line a plan and head dimension a node, from the attention probe's rows.
"""
import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import torch

SOURCE = Path(__file__).with_name('streamed.metal').read_text()
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


FIT_FEATURES = ('tiles', 'rowwork', 'planes', 'sg', 'partials', 'bytes', 'steps', 'chain_rows', 'passes', 'pressure')
FIT_TERMS = ('c0', 'tile', 'row', 'plane', 'sg', 'part', 'lat', 'latr', 'occ', 'press')
FIT_PRIOR = dict(c0=2.0, tile=0.02, row=0.01, plane=0.004, sg=0.01, part=0.002, lat=0.3, latr=0.15, occ=24.0, press=0.001)
FAMILIES = ('alu', 'coop', 'panel')
PERS = (8, 16, 24, 32, 48)


def evidence_path():
    return Path(os.environ.get('MESH_INSTANCES') or Path.home() / '.cache/mesh/instances.json')


def slice_units(m):
    """A slice's units: outputs, columns, tiles, nonzero tiles, planes (the sum of the widths), 1-bit tiles, bytes."""
    words = m.tiles.long() & 0xFFFFFFFF
    widths = (words >> 12) & 15
    return dict(outputs=m.rows, columns=m.columns, tiles=int(widths.numel()), nonzero=int((widths > 0).sum()),
                planes=int(widths.sum()), ones=int((widths == 1).sum()),
                bytes=int(widths.sum()) * 128 + 4 * words.numel() + 4 * m.bands.numel())


def instance_units(spec, s, rows):
    """The units of instance `spec` ({kernel: alu, coop, couple or panel; R or M, SK, BANDS, J, F, LUT, APART}) on slice
    units `s` at `rows` input rows (a couple's slice the sum of its two matrices')."""
    T, kernel = s['columns'] // 32, spec['kernel']
    nz, P, N = s['nonzero'], s['planes'], s['outputs']
    if kernel == 'panel':
        M = spec['M']
        panels = -(-rows // M)
        return dict(tiles=nz * panels, rowwork=nz * panels * M / 8, planes=P * panels, sg=(N // 32) * panels * 8,
                    partials=0, bytes=s['bytes'] * panels, steps=-(-T // 8), chain_rows=M / 8, passes=1,
                    pressure=nz * panels * (M / 64) ** 2)
    J, SK = spec['J'], spec['SK']
    pair = 2 if kernel == 'couple' else 1
    apart = spec.get('APART', 0)
    steps = -(-(-(-T // J)) // SK) * (1 if apart or pair == 1 else 2)
    sg = (N // 32) * J * SK * (2 if apart else 1)
    if kernel == 'coop':
        M = spec['M']
        return dict(tiles=nz, rowwork=nz * M / 8, planes=P, sg=sg, partials=J * rows * N / 32, bytes=s['bytes'] + 8 * J * rows * N,
                    steps=steps, chain_rows=M / 8, passes=1, pressure=nz * (M / 8) ** 2)
    R, ones = spec['R'], s.get('ones', 0) if spec.get('LUT') else 0
    live = R * (8 + 4 * spec['F']) * (2 if pair == 2 and not apart else 1)
    return dict(tiles=nz, rowwork=(nz - ones) * R + 0.5 * ones * R, planes=P - ones, sg=sg, partials=J * R * N / 32 * pair,
                bytes=s['bytes'] + (8 * J * min(rows, R) * N if pair == 1 else 0), steps=steps, chain_rows=R, passes=1,
                pressure=nz * live ** 2 / 144)


def _family(spec):
    return 'alu' if spec['kernel'] == 'couple' else spec['kernel']


class Fit:
    """A node's model of its instances' times (us): one coefficient vector a family, log-parametrized."""

    def __init__(self, cores, gbps, coefficients=None, factors=None):
        self.cores, self.gbps = cores, gbps
        self.theta = {f: torch.tensor([math.log((coefficients or {}).get(f, FIT_PRIOR)[k]) for k in FIT_TERMS], dtype=torch.float64)
                      for f in FAMILIES}
        self.factors = dict(factors or {})

    @classmethod
    def node(cls):
        """This node's fit from its evidence file, else the prior at its cores and the bandwidth the file names; its
        configurations the instance shapes the evidence observed (none: every admissible one)."""
        path = evidence_path()
        known = json.loads(path.read_text()) if path.exists() else {}
        made = cls(node()[1], known.get('gbps', 200.0), known.get('coefficients'), known.get('factors'))
        made.configurations = {tuple(c) for c in known.get('configurations', [])} or None
        return made

    configurations = None

    def _predict(self, family, X, theta=None, p=6.0):
        c0, tile, row, plane, sg, part, lat, latr, occ, press = torch.exp(self.theta[family] if theta is None else theta)
        f = dict(zip(FIT_FEATURES, X.T))
        thru = (tile * f['tiles'] + row * f['rowwork'] + plane * f['planes'] + sg * f['sg'] + part * f['partials']
                + press * f['pressure']) / self.cores
        waves = torch.clamp(f['sg'] / (self.cores * occ), min=1.0)
        chain = f['passes'] * lat * f['steps'] * (1 + latr * f['chain_rows']) * waves
        mem = f['bytes'] / self.gbps / 1e3
        return c0 * f['passes'] + (thru ** p + chain ** p + mem ** p) ** (1 / p)

    def time(self, spec, s, rows):
        u = instance_units(spec, s, rows)
        X = torch.tensor([[float(u[k]) for k in FIT_FEATURES]], dtype=torch.float64)
        return float(self._predict(_family(spec), X)[0]) * math.exp(self.factors.get(factor_key(spec), 0.0))

    def fit(self, observations, iters=900, ridge=0.02, spread=0.01):
        """Least squares in log space over observations (spec, slice units, rows, us, shape): the families' coefficients
        with a ridge toward the prior, and a factor a configuration and rows class (factor_key) with a ridge toward 1 (the
        shape of a kernel's own efficiency, pooled over every slice it ran on) where two shapes at least observed it; a
        configuration seen on one shape takes its family's model (its factor 0: one shape is no evidence of a shape)."""
        prior = torch.tensor([math.log(FIT_PRIOR[k]) for k in FIT_TERMS], dtype=torch.float64)
        shapes = {}
        for spec, s, _, _, shape in observations:
            shapes.setdefault(factor_key(spec), set()).add((s['outputs'], s['columns']))
        keys = sorted(k for k, seen in shapes.items() if len(seen) > 1)
        index = {k: i for i, k in enumerate(keys)}
        groups = {}
        for spec, s, rows, us, _ in observations:
            u = instance_units(spec, s, rows)
            g = groups.setdefault(_family(spec), ([], [], []))
            g[0].append([float(u[k]) for k in FIT_FEATURES])
            g[1].append(math.log(us))
            g[2].append(index.get(factor_key(spec), len(keys)))
        theta = {f: v.clone().requires_grad_() for f, v in self.theta.items()}
        factors = torch.zeros(len(keys), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam(list(theta.values()) + [factors], lr=0.05)
        data = {f: (torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64), torch.tensor(k))
                for f, (X, y, k) in groups.items()}
        count = sum(len(y) for _, y, _ in data.values())
        with torch.enable_grad():
            for _ in range(iters):
                opt.zero_grad()
                pooled = torch.cat([factors, factors.new_zeros(1)])
                loss = sum(((torch.log(self._predict(f, X, theta[f])) + pooled[k] - y) ** 2).sum() for f, (X, y, k) in data.items()) / count
                loss = loss + ridge * sum(((v - prior) ** 2).sum() for v in theta.values()) + (factors ** 2).sum() * spread / max(1, len(keys))
                loss.backward()
                opt.step()
        self.theta = {f: v.detach() for f, v in theta.items()}
        self.factors = {k: float(v) for k, v in zip(keys, factors.detach().tolist())}
        return float(loss.detach())

    def coefficients(self):
        return {f: dict(zip(FIT_TERMS, torch.exp(v).tolist())) for f, v in self.theta.items()}


def candidates(s, rows, op='product', tables=False):
    """The admissible instances of slice units `s` at `rows` input rows: rows <= 4 on the ALUs (R the rows' class)
    or the cooperative tensor (M 8); 8 or 16 rows the cooperative tensor (M 8 or 16); past 16 the panel (M 64, 128);
    shares of 8 to 48 tiles, simdgroups 2, 4 or 8 a band (the head's four bands of two too), one or two tiles in
    flight, a 1-bit tile by tables where the head's input writes them; `op` couple: the gate and up products together
    (its slice their units summed)."""
    T, out = s['columns'] // 32, []
    js = sorted({-(-T // -(-T // max(1, -(-T // per)))) for per in PERS})
    if op == 'couple' and rows > 4:
        return []
    if rows > 16:
        return [dict(kernel='panel', M=M) for M in (64, 128)]
    if op == 'couple':
        R = 1 if rows <= 1 else 2 if rows <= 2 else 4
        return [dict(kernel='couple', R=R, SK=sk, BANDS=1, J=1, F=F, LUT=0, APART=apart) for sk in (2, 4, 8) for F in (1, 2)
                for apart in (0, 1)]
    if rows <= 4:
        R = 1 if rows <= 1 else 2 if rows <= 2 else 4
        shapes = [(sk, 1) for sk in (2, 4, 8)] + ([(2, 4)] if op == 'head' else [])
        for sk, bands in shapes:
            if (s['outputs'] // 32) % bands:
                continue
            for J in js:
                for F in (1, 2):
                    for lut in ((0, 1) if tables else (0,)):
                        out.append(dict(kernel='alu', R=R, SK=sk, BANDS=bands, J=J, F=F, LUT=lut))
    if rows >= 2:
        M = 8 if rows <= 8 else 16
        for sk, bands in ((2, 2), (1, 4), (1, 1)):
            if (s['outputs'] // 32) % bands == 0:
                out += [dict(kernel='coop', M=M, SK=sk, BANDS=bands, J=J) for J in js]
    return out


def factor_key(spec):
    """A configuration and its rows class (an ALU's or couple's R, the tensor operation's M), as one string."""
    return '/'.join(str(v) for v in configuration(spec) + (spec.get('R', spec.get('M')),))


def configuration(spec):
    """An instance's shape apart from its K shares and rows: what an observation says of the instances near it."""
    k = spec['kernel']
    if k == 'panel':
        return (k, spec['M'])
    if k == 'coop':
        return (k, spec['SK'], spec['BANDS'])
    if k == 'couple':
        return (k, spec['SK'], spec['F'], spec['APART'])
    return (k, spec['SK'], spec['BANDS'], spec['F'], spec['LUT'])


def choose(fit, s, rows, op='product', tables=False, fold=None, level=0.0):
    """The fit's argmin over the candidates of the configurations its evidence observed (None where there is none),
    and its predicted us; where the product's input row folds into it at `fold` threads, a candidate at other threads
    is priced with the row's own dispatch (`level`)."""
    allowed = fit.configurations

    def price(c):
        threads = 256 if c['kernel'] == 'panel' else 32 * c['SK'] * c.get('BANDS', 1) * (2 if c.get('APART') else 1)
        return fit.time(c, s, rows) + (level if fold and threads != fold else 0.0)
    priced = [(price(c), i, c) for i, c in enumerate(candidates(s, rows, op, tables))
              if allowed is None or configuration(c) in allowed]
    if not priced:
        return None, None
    t, _, best = min(priced)
    return best, t


def spec_instance(spec, T, widths, finish=False):
    """An instance's host name and instantiation, threadgroups' x and y, threads and K shares (rows: the slice's bands)."""
    kernel = spec['kernel']
    if kernel == 'panel':
        return instance('streamed_panel', spec['M'], T, widths)
    if kernel == 'coop':
        return instance('streamed_coop', spec['M'], spec['SK'], spec['BANDS'], spec['J'], T, widths, coop_slots())
    if kernel == 'couple':
        return instance('streamed_couple', spec['R'], spec['SK'], T, widths, spec['F'], spec['APART'])
    return instance('streamed_alu', spec['R'], spec['SK'], spec['BANDS'], spec['J'], T, widths, int(finish), spec['LUT'], spec['F'])


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


def widths_of(tiles):
    """A matrix's (or a slice's) widths as its instances' WIDTHS mask: every width from 1 to the least of 4, 8 and 12 at
    least the widest its tile words hold (bits 1 ... c), so slices of one class share instances (the export's width caps
    are 8 and 12); the engine's key for the slice's instance is the same mask from the same words
    (streamed_weights.swift streamedWidths)."""
    widest = int((((tiles.cpu().long() & 0xFFFFFFFF) >> 12) & 15).max()) if tiles.numel() else 0
    return (1 << (next(c for c in (4, 8, 12, 15) if c >= widest) + 1)) - 2 if widest else 0


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
        out.widths = widths if widths is not None else widths_of(fields['tiles'])
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
            fields['rotation'], fields['rotation_scale'] = get('rotation'), get('rotation_scale')
            fields['folded'] = (fields['rotation'].float() * fields['rotation_scale'][:, None]).half()
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
        out.widths = widths_of(out.tiles)
        return out

    def dispatch(self, rows, op='product'):
        """The instance of a product of `rows` coded input rows on this node (the instance fit's choice): its name and
        instantiation, threadgroups x and y, threads, K shares."""
        spec, _ = choose(Fit.node(), slice_units(self), rows, op, tables=op == 'head' and self.folded is not None)
        if spec['kernel'] == 'coop' and coop_slots() is None:
            spec = dict(kernel='alu', R=min(rows, 4), SK=8, BANDS=1, J=spec['J'], F=1, LUT=0)
        name, line = spec_instance(spec, self.tiles_, self.widths)
        if spec['kernel'] == 'panel':
            return name, line, self.rows // 32, -(-rows // spec['M']), 256, 1
        return name, line, self.rows // 32 // spec['BANDS'], spec['J'], 32 * spec['SK'] * spec['BANDS'], spec['J']

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
            _encode('streamed_rotate', [x, outs[0], second, None, self.col_scale,
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
        up; gate and up together in one kernel with the GELU in its reduction (streamed_couple) where the fit prices it
        below their two products and streamed_gelu (couple), else those; down's coded input, down's product finished."""
        x16 = x.half().contiguous()
        n = x16.shape[0]
        gx, ux = gate.coded_input(x16, (up,))
        hidden = _half(n, gate.rows)
        spec = couple(Fit.node(), gate, up, n)
        if spec is not None:
            name, line = spec_instance(spec, gate.tiles_, gate.widths | up.widths)
            _encode(name, gate._code() + [None, gx, None] + up._code() + [ux, up.row_scale, hidden, None, gate.row_scale],
                    _dims(tiles=gate.tiles_, outputs=gate.rows, rows=n, pitch=gate.pitch), gate.rows // 32, 1,
                    (64 if spec['APART'] else 32) * spec['SK'], line)
            return down.apply(hidden).to(x.dtype)
        g, u = gate.product(gx), up.product(ux)
        _encode('streamed_gelu', [g, u, gate.row_scale, up.row_scale, hidden],
                _dims(outputs=gate.rows, shares=g.shape[0], inputs=u.shape[0], split=n * gate.rows, count=n * gate.rows),
                -(-(n * gate.rows) // 256), 1, 256)
        return down.apply(hidden).to(x.dtype)


def couple(fit, gate, up, rows):
    """The couple instance of gate and up at `rows` where the fit prices it below their own products and the GELU's
    dispatch (a dispatch's fixed cost), else None."""
    sg, su = slice_units(gate), slice_units(up)
    pair = {k: (sg[k] + su[k] if k not in ('outputs', 'columns') else sg[k]) for k in sg}
    spec, t = choose(fit, pair, rows, 'couple')
    if spec is None:
        return None
    apart = sum(choose(fit, u, rows)[1] for u in (sg, su)) + fit.coefficients()['alu']['c0']
    return spec if t < apart else None


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
        from safetensors import safe_open
        from safetensors.torch import load_file
        tensors = {k: v for k, v in load_file(str(path)).items() if not k.startswith('__pad_')}
        with safe_open(str(path), 'pt') as f:
            layout = (f.metadata() or {}).get('basis_layout')
        assert layout == 'code-order' or not any(k.endswith('.rotation') for k in tensors), \
            f'{path}: its rotations are not in coded column order (metal-microbench tools/model_code.py repack rewrites them)'
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
            for n in (1, 2, 4):
                x = torch.randn(n, m.columns, device='mps').half()
                chosen = m.dispatch(n, 'head')[0]
                tables = (torch.empty(n, m.tiles_, 8, 16, dtype=torch.float32, device='mps')
                          if chosen.startswith('streamed_alu') and chosen.split('_')[-2] == '1' else None)
                xt = m.coded_input(x, tables=tables)[0]
                y = m.finish(m.product(xt, op='head', tables=tables))
                line += f', head {n}: {_relative(y.cpu(), x.cpu().double() @ reference.T):.2e} ({chosen})'
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


ARGUMENTS = {'alu': ('R', 'SK', 'BANDS', 'J', 'TILES', 'WIDTHS', 'OUT', 'LUT', 'F'),
             'coop': ('M', 'SK', 'BANDS', 'J', 'TILES', 'WIDTHS', 'COOP'),
             'couple': ('R', 'SK', 'TILES', 'WIDTHS', 'F', 'APART'),
             'panel': ('M', 'TILES', 'WIDTHS')}


def parse_instance(name):
    """An instance's spec from its host name (instance): its template arguments by name (ARGUMENTS)."""
    parts = name.replace('streamed_', '').split('_')
    kernel = parts[0]
    if kernel not in ARGUMENTS:
        raise ValueError(name)
    spec = dict(zip(ARGUMENTS[kernel], (int(v) for v in parts[1:])), kernel=kernel)
    if kernel == 'couple':
        spec.update(BANDS=1, J=1, LUT=0)
    return spec


UNITS = ('outputs', 'columns', 'tiles', 'nonzero', 'planes', 'ones', 'bytes')


def _records(path):
    """A record's rows: a probe's JSON ({rows: [...]}) or JSON lines (observe's, the attention probe's)."""
    text = Path(path).read_text()
    if text.lstrip().startswith('{"rows"') or text.lstrip().startswith('{\n'):
        return json.loads(text)['rows']
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def observations(paths):
    """The admissible instances' observations of probe and profile records (rows of instance, units, rows_in, best_us:
    (spec, units, rows, us, shape)), and the attention plans' rows (kind attention)."""
    out, attention = [], []
    for path in paths:
        for row in _records(path):
            if row.get('kind') == 'attention':
                attention.append(row)
                continue
            spec = parse_instance(row['instance'])
            if spec['kernel'] == 'alu' and spec['R'] > 4:
                continue
            out.append((spec, {k: row[k] for k in UNITS}, row['rows_in'], row['best_us'], row.get('form', row['instance']).split(' R')[0]))
    return out, attention


PLANS = ('vector', 'tile')


def attention_reads(plan, dims, rows, heads, positions):
    """The KV bytes a decode attention plan reads (metal-microbench docs/kernels.md#decode-attention): the vector form's
    tasks each one query head of one row, the tile plan's each 8 of them, every task its K and V at `positions`."""
    tasks = rows * heads if plan == 'vector' else -(-rows * heads // 8)
    return tasks * positions * dims * 4


def attention_fit(rows):
    """Each plan's time on the node at each head dimension as c0 + c1 MB of KV read, by least squares on relative error,
    with its median error: keys 'plan dims'."""
    import numpy as np
    out = {}
    for plan in PLANS:
        for dims in sorted({r['dims'] for r in rows}):
            seen = [r for r in rows if r['plan'] == plan and r['dims'] == dims]
            if len(seen) < 2:
                continue
            X = np.array([[1.0, r['reads'] / 1e6] for r in seen])
            y = np.array([r['best_us'] for r in seen])
            c = np.linalg.lstsq(X / y[:, None], np.ones(len(y)), rcond=None)[0]
            out[f'{plan} {dims}'] = {'c0': float(c[0]), 'mb': float(c[1]), 'count': len(seen),
                                     'median_log_error': float(np.median(np.abs(np.log((X @ c) / y))))}
    return out


def attention_plan(model, dims, rows, heads, positions):
    """The plan the node's model prices least at (dims, rows, heads, positions), with each plan's price (us); the vector
    form where the model has no line for both plans at `dims`."""
    priced = {plan: model[f'{plan} {dims}']['c0'] + model[f'{plan} {dims}']['mb'] * attention_reads(plan, dims, rows, heads, positions) / 1e6
              for plan in PLANS if f'{plan} {dims}' in (model or {})}
    return (min(priced, key=priced.get) if len(priced) == len(PLANS) else 'vector'), priced


def _fit(args):
    obs, attention = observations(args.observations)
    fit = Fit(node()[1], args.gbps)
    loss = fit.fit(obs)
    errors = sorted(abs(math.log(fit.time(spec, s, n) / t)) for spec, s, n, t, _ in obs)
    held, regrets = [], []
    for shape in sorted({o[4] for o in obs}):
        other = Fit(node()[1], args.gbps)
        other.fit([o for o in obs if o[4] != shape], iters=600)
        for n in sorted({o[2] for o in obs if o[4] == shape}):
            seen = [(other.time(spec, s, n), t) for spec, s, nn, t, sh in obs if sh == shape and nn == n]
            held += [abs(math.log(p / t)) for p, t in seen]
            regrets.append(min(seen)[1] / min(t for _, t in seen) - 1)
    held.sort(); regrets.sort()
    record = {'node': node()[0], 'cores': node()[1], 'gbps': args.gbps, 'observations': [str(Path(p).resolve()) for p in args.observations],
              'configurations': sorted({configuration(o[0]) for o in obs}),
              'count': len(obs), 'loss': loss, 'median_log_error': errors[len(errors) // 2],
              'held_out': {'median_log_error': held[len(held) // 2], 'median_regret': regrets[len(regrets) // 2],
                           'worst_regret': regrets[-1]}, 'coefficients': fit.coefficients(), 'factors': fit.factors,
              'attention': attention_fit(attention)}
    path = evidence_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'coefficients'}))


def _observe(args):
    """A direct step's profile (forward_graph LM_PROFILE_DECODE: a line a pipeline label, its microseconds a step and
    dispatches a step) as observations of its instances: each label's slices the export's of its shape and widths
    (their units' mean), its time a dispatch less the profile's floor (the least time a dispatch of any label)."""
    import re
    streamed = Streamed(args.export)
    held = slices(streamed, args.program, args.rank)
    shapes = {}
    for _, m in held:
        shapes.setdefault((m.rows, m.columns, m.widths), []).append(slice_units(m))
    text = Path(args.profile).read_text()
    rows = int(re.search(r'decode step, (\d+) rows', text).group(1))
    lines = re.findall(r'^\s+(\S+)(?: (\d+)x(\d+))?\s+([0-9.]+) us a step\s+(\d+) dispatches a step', text, re.M)
    floor = min(float(us) / int(n) for _, _, _, us, n in lines)
    out = []
    for label, outputs, columns, us, n in lines:
        if not label.startswith(('streamed_alu', 'streamed_coop', 'streamed_panel')) or not outputs:
            continue
        spec = parse_instance(label)
        units = shapes.get((int(outputs), int(columns), spec['WIDTHS']))
        if not units:
            continue
        mean = {k: round(sum(u[k] for u in units) / len(units)) for k in UNITS}
        out.append({'instance': label, 'rows_in': rows, 'best_us': max(float(us) / int(n) - floor, 0.1), 'form': f'profile {outputs}x{columns} R{rows}',
                    'source': str(Path(args.profile).resolve()), **mean})
    with open(args.out, 'a') as f:
        for row in out:
            f.write(json.dumps(row) + '\n')
    print(f'{args.out}: {len(out)} observations from {args.profile} (floor {floor:.2f} us a dispatch)')


def slices(streamed, program=None, rank=None):
    """Each coded matrix's slice a rank holds (its record's slices: a tensor's output ranges are rows, its input range
    columns; the head's rows its vocabulary; and each layer's FFN neurons its GPU computes in a phase of their own, its
    engines' gpu range and its prefill columns), every matrix whole without a record: (name, matrix)."""
    part = json.loads(Path(program).read_text())['ranks'][rank] if program else None
    out = []
    for L, layer in enumerate((part or {}).get('layers') or []):
        for span in ((layer.get('engines') or {}).get('gpu'), (layer.get('prefill') or {}).get('columns')):
            base = f'model.language_model.layers.{L}.mlp.'
            if span and span[1] > span[0] and base + 'gate_proj' in streamed.matrices:
                for p in ('gate_proj', 'up_proj'):
                    out.append((base + p, streamed[base + p].cut(rows=tuple(span))))
                out.append((base + 'down_proj', streamed[base + 'down_proj'].cut(columns=tuple(span))))
    for name, m in streamed.matrices.items():
        if part is None:
            out.append((name, m))
            continue
        if name == 'lm_head':
            v = part.get('vocabulary')
            if v:
                out.append((name, m.cut(rows=tuple(v))))
            continue
        cut = (part.get('slices') or {}).get(name + '.weight')
        if cut is None:
            continue
        rows = tuple(cut['output'][0]) if cut.get('output') else None
        columns = tuple(cut['input']) if cut.get('input') else None
        if (rows and rows[0] == rows[1]) or (columns and columns[0] == columns[1]):
            continue
        out.append((name, m.cut(rows=rows, columns=columns)))
    return out


def _attention(export, part, ctx, classes, fit_model):
    """The decode attention plan a rank takes for each class of its layers (head dimension, window, its query heads) at
    each rows class to 16 (docs/kernels.md#decode-attention): the node's model's least at the class's positions (the
    window, else the call's context)."""
    from safetensors import safe_open
    with safe_open(str(export), 'pt') as f:
        model = Path((f.metadata() or {}).get('model', ''))
    config = json.loads((model / 'config.json').read_text()) if (model / 'config.json').exists() else {}
    text = config.get('text_config', config)
    kinds = text.get('layer_types') or []
    heads = text.get('num_attention_heads')
    if not kinds or not heads:
        return []
    out = {}
    for L, kind in enumerate(kinds):
        layer = ((part or {}).get('layers') or [None] * len(kinds))[L]
        held = (layer['heads'][1] - layer['heads'][0]) if layer and layer.get('heads') else (0 if part else heads)
        if held <= 0:
            continue
        sliding = kind == 'sliding_attention'
        dims = text['head_dim'] if sliding else text.get('global_head_dim') or text['head_dim']
        window = text.get('sliding_window', 0) if sliding else 0
        for rows in (c for c in classes if c <= 16):
            plan, priced = attention_plan(fit_model, dims, rows, held, window or ctx)
            out[(dims, window, held, rows)] = {'dims': dims, 'window': window, 'heads': held, 'rows': rows, 'plan': plan,
                                               'us': {k: round(v, 2) for k, v in priced.items()}}
    return [out[k] for k in sorted(out)]


def _resolve(args):
    streamed = Streamed(args.export)
    fit, classes = Fit.node(), [int(c) for c in args.classes.split(',')]
    folds = dict((k, int(v)) for k, v in (f.split('=') for f in args.folds.split(',') if f))
    held = slices(streamed, args.program, args.rank)
    for draft in args.draft:
        held += list(Streamed(draft).matrices.items())
    keyed, folded = {}, {}
    for name, m in held:
        op = 'head' if name == 'lm_head' else 'product'
        key = (op, m.rows, m.columns, m.widths, op == 'head' and m.folded is not None)
        keyed.setdefault(key, []).append(slice_units(m))
        for role_, lanes in folds.items():
            if name.endswith('.' + role_):
                folded[key] = lanes
        if name.endswith('.mlp.gate_proj'):
            up = next((u for n, u in held if n == name[:-len('gate_proj')] + 'up_proj' and u.rows == m.rows), None)
            if up is not None:
                keyed.setdefault(('couple', m.rows, m.columns, m.widths | up.widths, False), []).append((slice_units(m), slice_units(up)))
    entries = []
    mean = lambda units: {k: round(sum(u[k] for u in units) / len(units)) for k in UNITS}
    level = fit.coefficients()['alu']['c0']
    for (op, outputs, columns, mask, tables), units in sorted(keyed.items(), key=lambda kv: kv[0]):
        if op == 'couple':
            sg, su = mean([u[0] for u in units]), mean([u[1] for u in units])
            s = {k: (sg[k] + su[k] if k not in ('outputs', 'columns') else sg[k]) for k in UNITS}
        else:
            s = mean(units)
        lanes = folded.get((op, outputs, columns, mask, tables))
        for rows in classes:
            if op == 'couple':
                spec, t = choose(fit, s, rows, 'couple')
                if spec is None or t >= choose(fit, sg, rows)[1] + choose(fit, su, rows)[1] + level:
                    continue
            else:
                spec, t = choose(fit, s, rows, op, tables, fold=lanes, level=level)
            if spec['kernel'] == 'coop' and coop_slots() is None:
                continue
            name, _ = spec_instance(spec, columns // 32, mask)
            kernel = name.split('_')[1]
            args_ = [int(v) for v in name.split('_')[2:]]
            if spec['kernel'] == 'panel':
                grid, threads, shares = [outputs // 32, -(-rows // spec['M'])], 256, 1
            elif spec['kernel'] == 'couple':
                grid, threads, shares = [outputs // 32, 1], (64 if spec['APART'] else 32) * spec['SK'], 1
            else:
                grid, threads, shares = [outputs // 32 // spec['BANDS'], spec['J']], 32 * spec['SK'] * spec['BANDS'], spec['J']
            entries.append({'op': op, 'outputs': outputs, 'columns': columns, 'widths': mask, 'rows': rows, 'kernel': kernel,
                            'args': args_, 'names': list(ARGUMENTS[kernel]), 'grid': grid, 'threads': threads, 'shares': shares,
                            'lut': int(spec.get('LUT', 0)), 'us': round(t, 2)})
    record = json.loads(Path(args.program).read_text()) if args.program else {}
    ctx = args.ctx or record.get('ctx_mean') or 1024
    known = json.loads(evidence_path().read_text()) if evidence_path().exists() else {}
    part = record['ranks'][args.rank] if args.program else None
    attention = _attention(args.export, part, ctx, classes, known.get('attention'))
    table = {'node': node()[0], 'cores': node()[1], 'evidence': str(evidence_path()), 'export': str(args.export),
             'program': args.program, 'rank': args.rank, 'classes': classes, 'ctx': ctx, 'instances': entries, 'attention': attention}
    Path(args.out).write_text(json.dumps(table, indent=1) + '\n')
    print(f'{args.out}: {len(entries)} instances, {len(keyed)} slices, {len(attention)} attention classes, classes {classes}')


def main():
    parser = argparse.ArgumentParser(prog='python -m torch_mesh.streamed')
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check')
    check.add_argument('export')
    check.add_argument('names', nargs='*')
    check.add_argument('--tools', default='~/metal-microbench/tools')
    fit = sub.add_parser('fit')
    fit.add_argument('observations', nargs='+')
    fit.add_argument('--gbps', type=float, required=True, help="the node's DRAM read bandwidth, GB/s")
    resolve = sub.add_parser('resolve')
    resolve.add_argument('export')
    resolve.add_argument('--program', default=None)
    resolve.add_argument('--rank', type=int, default=None)
    resolve.add_argument('--classes', default='1,2,4,8,16,32,64,128,256,512')
    resolve.add_argument('--ctx', type=int, default=0, help="the positions a global layer's attention is priced at (default: the record's ctx_mean, else 1024)")
    resolve.add_argument('--folds', default='per_layer_projection=256',
                         help="NAME=LANES,...: a matrix whose input row the recorder folds into its product where the product's threads are LANES (the engine's streamed_ple)")
    resolve.add_argument('--draft', action='append', default=[],
                         help="a drafter's export whose matrices the member runs whole (metal-microbench tools/draft/dflash2.py export)")
    resolve.add_argument('--out', required=True)
    observe = sub.add_parser('observe')
    observe.add_argument('profile')
    observe.add_argument('export')
    observe.add_argument('--program', default=None)
    observe.add_argument('--rank', type=int, default=None)
    observe.add_argument('--out', required=True)
    args = parser.parse_args()
    {'check': _check, 'fit': _fit, 'resolve': _resolve, 'observe': _observe}[args.command](args)


if __name__ == '__main__':
    main()
