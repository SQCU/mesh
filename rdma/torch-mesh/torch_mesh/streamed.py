"""Streamed coded weights in the mesh library (streamed.metal): a matrix stored as tile codes. Torch describes
its operations; whole-function preparation binds the shared native producer and consumer commands.

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
  m[a:b], m[:, c:d:s]                 logical ranges over retained coded blocks, with masked boundary lanes
  m.to('cpu') or m.dense()            decodes it
  Matrix.ffn(x, gate, up, down)       the fused GELU FFN couple (torch_mesh.engines.ffn runs it where no gradient is due)

x @ m is one pipeline whichever way m is transposed: the input side's order and scale (a rotation where it lies there),
the product contracting the coded rows (streamed_alu, streamed_coop, streamed_panel) or the coded columns
(streamed_panel_t: the same decoded tiles consumed the other way), the output side's order and scale (and rotation).
Each product is an instance with literal sizes, the matrix's widths and the node's choice for its rows (the instance
fit, below: the argmin of a model of the node's times over the instances' units; the cooperative tensor's slots read
from the node by streamed_coopmap), selected and compiled during preparation. Operands are fp16 on the GPU (another dtype's rows scaled
into its range by powers of two, results finished in the caller's dtype); the codes are constants (no gradient reaches
them, only through them). An operation a coded matrix lacks is an error naming the ones it has.

A decomposition operand names the instance a product runs as in place of the fit's choice (metal-microbench
docs/kernels.md#decompositions): `spec_of('coop/M=64,SK=2,BANDS=1,J=1,LOADS=0,RING=32')` is an instance spec,
Matrix.product, apply and ffn take it as `decomposition`, and decomposition_of gives the engine's entry for it (its
kernel, args by ARGUMENTS' names, grid, threads, shares). The cooperative tensor's RING r >= 8 is the study's role split
by rows (streamed.metal streamed_coop; restored from a9d17c7^ tools/streamed_gemv.metal streamed_gemm): SK producer
simdgroups decode each tile once into a threadgroup ring and M / r consumers each multiply it by r rows of their own, at 32
to 256 rows (a verify step's, a prefill chunk's). No fit prices it: it runs where an operand names it.

The instance fit (metal-microbench docs/kernels.md#the-instance-fit): an instance's time on a slice is a model of its
units, t = c0 + smooth max(throughput, chain, bytes / bandwidth), the throughput the slice's tile visits, row work,
planes, simdgroups, partials and register pressure over the node's cores, the chain a simdgroup's dependent steps over
the waves the node holds; one set of coefficients a kernel family (streamed_alu with streamed_couple, streamed_coop,
streamed_panel) a node, fitted by least squares in log space with a ridge toward the prior from the node's
observations. The evidence file (MESH_INSTANCES or ~/.cache/mesh/instances.json) is one document a node prices by,
whole wherever it lies: the coefficients, the bandwidth, the configurations observed, the served readings and the
attention readings themselves, the attention model and the fused readings; made on a node of its chip and GPU cores
from observations read on that node (each row names its node: a bandwidth read's its own, a served reading's the
evidence it was priced by); a node without it, or with another chip's or core count's, is refused. A choice is
the argmin over the admissible instances of a slice at its rows: of the instances served readings read at that slice
and rows (observe's rows, each its readings' median) where there are any, a measurement at a shape outranking the
model there, else of the fit's prices (every served measurement moves the coefficients for every other shape). The
`probe` and `decompose` commands, which ran every candidate instance and the named ones on random inputs over cold
copies, each checked against float64, were deleted 2026-10-22 with the other test code (the fit reads served
readings).

  python examples/memory-read.py --mib M --rounds N
                                                                this node's DRAM read bandwidth (streamed_read), a row of
                                                                kind bandwidth reported through mesh_statistics
  python -m torch_mesh.streamed fit OBSERVATIONS ...            this node's evidence fitted from records read on it (JSON
                                                                lines: bandwidth's, observe's, the decode attention plans' rows
                                                                kind attention), the bandwidth its reads' (else, a refit,
                                                                its evidence's); a record of another node refused
  python -m torch_mesh.streamed resolve EXPORT [--program RECORD --rank R] [--rows 1,2,4,...] [--explore R=C[/FORM<v>],...]
                                [--plan R=PLAN[/PARTS][@sliding]] [--apart R] [--draft DRAFTER] --out CHOICES
                                                                the fit evaluated where the engine runs: each slice's chosen
                                                                instance at each row count the member runs (the record's,
                                                                call_rows: its step's, its chunks' and their head's; without a
                                                                record, --rows, the served engine's own batch buckets), a
                                                                slice's key its width class (WIDTH_CLASSES), each argument
                                                                named, and each layer's attention plan and part count at
                                                                each of those row counts: the choices an engine reads
                                                                (LM_STREAMED_INSTANCES), naming this node; no row count is
                                                                rounded to another, and none the member does not run is
                                                                evaluated; with --draft DRAFTER, the drafter's matrices
                                                                too, a drafting member's share of a product as views on
                                                                its `draft <kind>` lattice (draft_slices)
  python -m torch_mesh.streamed observe READING EXPORT [--loads L] [--program RECORD --rank R] --out ROWS
                                                                a served reading's profile (metal-microbench
                                                                tools/reading.py) as observations of its instances and
                                                                attention plans (at the part counts of the instance table
                                                                the reading names, whose node each row names), appended to
                                                                ROWS (JSON lines)
  python -m torch_mesh.streamed epilogue FUSED SEPARATE --rows NAME=LABEL ... --export EXPORT [--program RECORD --rank R]
                                                                two served readings of this node, its kinds run fused into a
                                                                product (fused_ratios: each epilogue against the product and
                                                                the row it replaces; the finish, a one-share product
                                                                finishing its rows, against its partials and their
                                                                streamed_finish) into the evidence file (fused), an
                                                                epilogue keyed by the stack the member runs it in, whose
                                                                rows classes each table entry names (epilogues, finish)
  python -m torch_mesh.streamed admit KL_RECORD H<h> BAR         a form whose numerics differ admitted on this node
                                                                (forms_at_bar) by a decode KL record of the engine served
                                                                with it, at or below the KL bar BAR the caller states

A factor of the fit (a configuration's efficiency at a rows class) rests on two slice shapes at least; one seen on a
single shape takes its family's model. An instance's forms (FORMS: LOADS, a tile's loads; H, the ALUs' fp16 pair sums,
listed only where the node admitted the form at the KL bar; RING, the cooperative tensor's producer-consumer ring) are
priced by their own factors, each the mean log ratio of its readings against its form 0's on the same shapes and rows; a
form never read in pairs is not taken. A decode attention plan (metal-microbench docs/kernels.md#decode-attention: the
vector form, a task a query head and row; the tile plan, a task 8 of one KV head's rows) is priced c0 + c1 MB of the KV
it reads (c1 at least 0), one line a plan and head dimension a node at the engine's part count, from the attention rows;
a plan and part count read at a layer's shape and rows is priced at its reading; a layer whose plans neither its
readings nor the model price is refused, never given a plan unpriced. Every evidence file and table is written whole
(a file renamed over it: _write).
"""
import argparse
import json
import math
import os
import struct
from pathlib import Path

import torch

import mesh_compute
COMPUTE = mesh_compute.compilation()
POLICY = json.loads((Path(mesh_compute.__file__).resolve().parents[1] / 'etc/compute/streamed.json').read_text())
SOURCE = Path(__file__).with_name('streamed.metal').read_text()
_PIPELINES = {}
aten = torch.ops.aten


def instance(kernel, *arguments):
    """An instance's host name and its explicit instantiation (one line appended to the library source)."""
    words = [a if isinstance(a, str) else str(int(a)) for a in arguments]
    args = ', '.join(words)
    name = '_'.join([kernel] + words)
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
    """This node's name (its chip) and GPU cores, the one count its registry names (a node whose registry names none, or
    several, is refused: the fit prices a node by its cores)."""
    from mesh_compute import hardware
    facts = hardware()
    return facts['chip'], facts['gpu']['cores']


FIT_FEATURES = ('tiles', 'rowwork', 'planes', 'sg', 'partials', 'bytes', 'steps', 'chain_rows', 'passes', 'pressure')
FIT_TERMS = ('c0', 'tile', 'row', 'plane', 'sg', 'part', 'lat', 'latr', 'occ', 'press')
FIT_PRIOR = POLICY['fit_prior']
FAMILIES = ('alu', 'coop', 'panel')
# An instance's forms, each read in pairs against its form 0 on the same shape, rows and configuration (a factor a
# configuration and rows class, never the coefficients'): LOADS a tile's loads (streamed.metal fragment_load), H the sums
# (fragment_walk: fp32, or fp16 pair sums added into them every H tiles, at the ALUs' widths to 11: listed where the node
# admitted it at a KL bar, or a resolve names it), RING the cooperative tensor's producer-consumer ring
# (streamed_coop).
FORMS = ('LOADS', 'H', 'RING')
PERS = POLICY['tiles_per_share']
WIDTH_CLASSES = (4, 8, 12, 15)
WIDTH_CLASS = [0] + [next(c for c in WIDTH_CLASSES if c >= w) for w in range(1, 16)]


def evidence_path():
    return Path(os.environ.get('MESH_INSTANCES') or Path.home() / '.cache/mesh/instances.json')


_EVIDENCE_PATH = evidence_path()
_EVIDENCE = json.loads(_EVIDENCE_PATH.read_text()) if _EVIDENCE_PATH.exists() else None


def slice_units(m):
    """A slice's units: outputs, columns, tiles, nonzero tiles, planes (the sum of the widths), 1-bit tiles, bytes."""
    if getattr(m, 'units', None) is not None:
        return dict(m.units)
    widths = [(word >> 12) & 15 for word in m.metadata['tiles']]
    planes = sum(widths)
    return dict(outputs=m.rows, columns=m.columns, tiles=len(widths), nonzero=sum(w > 0 for w in widths),
                planes=planes, ones=widths.count(1),
                bytes=planes * 128 + 4 * len(widths) + 4 * len(m.metadata['bands']))


def instance_units(spec, s, rows):
    """The units of instance `spec` ({kernel: alu, coop, couple or panel; R or M, SK, BANDS, J, F, LUT, APART}) on slice
    units `s` at `rows` input rows (a couple's slice the sum of its two matrices')."""
    T, kernel = s['columns'] // 32, spec['kernel']
    nz, P, N = s['nonzero'], s['planes'], s['outputs']
    if kernel == 'panel':
        M = spec['M']
        panels = -(-rows // M)
        return dict(tiles=nz * panels, rowwork=nz * panels * M / 8, planes=P * panels, sg=(N // 32) * panels * COMPUTE['STREAMED_PANEL_TILES'],
                    partials=0, bytes=s['bytes'] * panels, steps=-(-T // COMPUTE['STREAMED_PANEL_TILES']), chain_rows=M / 8, passes=1,
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
        self.explore, self.forms = {}, {}

    @classmethod
    def node(cls):
        """This node's fit from its evidence file, made (fit) on a node of its chip and GPU cores from observations read on
        that node, at the bandwidth the file names: any such node's (one of four like nodes fitted serves the others, its
        file copied whole: the file holds its served readings and attention readings themselves, `read` and
        `attention_read`, its attention model `attention` and its fused readings `fused`); its configurations the
        instance shapes the evidence observed (none: every admissible one). A node without evidence, whose file another
        chip or core count made, or one fitted before the file held its readings, is refused: no prior or other node's
        coefficients price its instances (metal-microbench docs/kernels.md#the-instance-fit, Evidence)."""
        path, (chip, cores) = _EVIDENCE_PATH, node()
        known = _EVIDENCE
        if known is None:
            raise RuntimeError(f'{path}: no instance evidence on this {chip} ({cores} GPU cores): `python examples/memory-read.py`, '
                               f'`observe` and `fit` make it, or another {chip} of {cores} GPU cores gives its file (metal-microbench '
                               f'docs/kernels.md#the-instance-fit, Evidence)')
        if (known.get('node'), known.get('cores')) != (chip, cores) or not known.get('gbps') or not known.get('coefficients'):
            raise RuntimeError(f'{path}: instance evidence of {known.get("node") or "no named node"} ({known.get("cores")} GPU cores, '
                               f'{known.get("gbps")} GB/s, {"fitted" if known.get("coefficients") else "not fitted"}), '
                               f'not a fit made on this {chip} ({cores} GPU cores)')
        if 'readings' not in known or 'attention_readings' not in known:
            raise RuntimeError(f'{path}: fitted before the file held its readings (`python -m torch_mesh.streamed fit` again over its '
                               f'observations, on this node: metal-microbench docs/kernels.md#the-instance-fit, Evidence)')
        made = cls(cores, known['gbps'], known['coefficients'], known.get('factors'))
        made.configurations = {tuple(c) for c in known.get('configurations', [])} or None
        made.admitted = set(known.get('forms_at_bar', {}))
        made.read = {}
        for r in known['readings']:
            made.read.setdefault(tuple(r['slice']), {})[tuple(r['instance'])] = r['us']
        made.attention_read = {tuple(r[k] for k in ATTENTION_READ): r['us'] for r in known['attention_readings']}
        made.attention, made.fused = known.get('attention') or {}, known.get('fused') or {}
        return made

    configurations = None
    admitted = frozenset()
    read = attention_read = attention = fused = {}

    def _predict(self, family, X, theta=None, p=POLICY['smooth_power']):
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
        forms = sum(self.factors.get(form_key(spec, form), 0.0) for form in FORMS if spec.get(form))
        return float(self._predict(_family(spec), X)[0]) * math.exp(self.factors.get(factor_key(spec), 0.0) + forms)

    def fit(self, observations, iters=POLICY['fit_iterations'], ridge=POLICY['fit_ridge'], spread=POLICY['fit_spread']):
        """Least squares in log space over observations (spec, slice units, rows, us, shape): the families' coefficients
        with a ridge toward the prior, and a factor a configuration and rows class (factor_key) with a ridge toward 1 (the
        shape of a kernel's own efficiency, pooled over every slice it ran on) where two shapes at least observed it; a
        configuration seen on one shape takes its family's model (its factor 0: one shape is no evidence of a shape). A
        form's factor (form_key: FORMS) is the mean log ratio of the instances read in that form and in its form 0 on the
        same shape and rows, whose readings measure the form and not the coefficients."""
        prior = torch.tensor([math.log(FIT_PRIOR[k]) for k in FIT_TERMS], dtype=torch.float64)
        ratios, paired = {}, set()
        for form in FORMS:
            read = {}
            for i, (spec, s, rows, us, shape) in enumerate(observations):
                same = (tuple(sorted((k, v) for k, v in spec.items() if k != form)), shape, rows)
                read.setdefault(same, {}).setdefault(spec.get(form, 0), []).append((us, i))
            for (same, _, _), seen in read.items():
                if 0 not in seen or len(seen) < 2:
                    continue
                for value, taken in seen.items():
                    paired.update(i for _, i in taken)
                    if value:
                        ratios.setdefault(form_key(dict(same, **{form: value}), form), []).append(math.log(min(taken)[0] / min(seen[0])[0]))
        forms = {k: sum(v) / len(v) for k, v in ratios.items()}
        observations = [o for i, o in enumerate(observations) if i not in paired]
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
        opt = torch.optim.Adam(list(theta.values()) + [factors], lr=POLICY['fit_learning_rate'])
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
        self.factors = {**{k: float(v) for k, v in zip(keys, factors.detach().tolist())}, **forms}
        return float(loss.detach())

    def coefficients(self):
        return {f: dict(zip(FIT_TERMS, torch.exp(v).tolist())) for f, v in self.theta.items()}


def rows_class(rows):
    """The ALUs' rows class R of `rows` input rows (to 4): 1, 2 or 4."""
    return next(r for r in POLICY['alu_rows'] if r >= min(rows, max(POLICY['alu_rows'])))


def candidates(s, rows, op='product', tables=False, widths=15, admitted=frozenset()):
    """The admissible instances of slice units `s` at `rows` input rows: rows <= 4 on the ALUs (R the rows' class); 8 or
    16 rows the cooperative tensor (M 8 or 16), the ceiling round's instance family (the coop's M 8 work, read at eight rows,
    at or above the ALUs' four-row time on every served slice of the Mini and all but one of the M5's, and the ALUs at eight
    rows on the small slices 1.4 to 2.6 times the coop's: metal-microbench docs/kernels.md#stacked-products); past 16 the
    panel (M 64, 128);
    shares of 8 to 48 tiles, simdgroups 2, 4 or 8 a band (the head's four bands of two too), one or two tiles in
    flight, a 1-bit tile by tables where the head's input writes them, either loads (LOADS: fragment_load's), on the ALUs
    either sums (H: fp32, or the fp16 pair sums every H tiles for each H<h> the node `admitted` at the KL bar where the
    slice admits it, admits), on the cooperative tensor either ring (RING); `op` couple: the gate and up products together
    (its slice their units summed). A form is taken where its factor prices it below its form 0 (FORMS), each candidate
    listed after its form 0."""
    T, out = s['columns'] // 32, []
    halves = sorted({int(f[1:]) for f in admitted if f[:1] == 'H' and f[1:].isdigit()})
    js = sorted({-(-T // -(-T // max(1, -(-T // per)))) for per in PERS})
    if op == 'couple' and rows > max(POLICY['alu_rows']):
        return []
    if rows > max(POLICY['cooperative_rows']):
        return [dict(kernel='panel', M=M, LOADS=loads) for M in POLICY['panel_rows'] for loads in (0, 1)]
    if op == 'couple':
        out = [dict(kernel='couple', R=rows_class(rows), SK=sk, BANDS=1, J=1, F=F, LUT=0, APART=apart, LOADS=loads) for sk in POLICY['alu_splits']
               for F in POLICY['tiles_in_flight'] for apart in (0, 1) for loads in (0, 1)]
    elif rows <= max(POLICY['alu_rows']):
        R = rows_class(rows)
        shapes = [(sk, 1) for sk in POLICY['alu_splits']] + ([POLICY['head_shape']] if op == 'head' else [])
        for sk, bands in shapes:
            if (s['outputs'] // 32) % bands:
                continue
            for J in js:
                for F in POLICY['tiles_in_flight']:
                    for lut in ((0, 1) if tables else (0,)):
                        out += [dict(kernel='alu', R=R, SK=sk, BANDS=bands, J=J, F=F, LUT=lut, LOADS=loads) for loads in (0, 1)]
    else:
        M = next(r for r in POLICY['cooperative_rows'] if r >= rows)
        for sk, bands in POLICY['cooperative_shapes']:
            if (s['outputs'] // 32) % bands == 0:
                out += [dict(kernel='coop', M=M, SK=sk, BANDS=bands, J=J, LOADS=loads, RING=ring) for J in js for loads in (0, 1) for ring in (0, 1)]
    return [dict(c, **f) for c in out for f in [{}] + [{'H': h} for h in halves if admits(c, 'H', h, widths)]]


def admits(spec, form, value, widths):
    """Whether instance `spec` on a slice of width class `widths` has form `form` at `value` (FORMS): H on the ALUs (alu,
    couple) at widths exact in half2 (width class 8 at most) in whole steps of its F (fragment_walk's static assertion),
    RING on the cooperative tensor, LOADS on every kernel."""
    if form == 'H':
        return spec['kernel'] in ('alu', 'couple') and widths <= POLICY['half_width_bits'] and value % spec['F'] == 0
    return form == 'LOADS' or form == 'RING' and spec['kernel'] == 'coop'


def form_key(spec, form):
    """A form's factor's key (FORMS): the form and its value, a configuration and its rows class (LOADS 1 against 0:
    'loads/'; H h against 0: 'H<h>/'; RING 1 against 0: 'RING1/')."""
    return ('loads/' if form == 'LOADS' else f'{form}{spec[form]}/') + factor_key(spec)


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


def choose(fit, s, rows, op='product', tables=False, read=None, widths=15):
    """The argmin over the candidates of the configurations its evidence observed (None where there is none), or at a
    rows class the fit explores, of the configurations named for it (measured once, in the next profile), and its price
    (us): where candidates were `read` at the slice and rows (read_key: their readings' median), the least read (a
    measurement at a shape outranks the model there), else the fit's argmin. Forms named for the class (fit.forms: a
    resolve's --explore, FORM<v>) change only those arguments, where the instance admits them (admits): of the slice's
    least read instance where it was read and no configurations are named (its pair, the same instance in the form),
    else of each candidate in its form 0 (a form any H whose steps F divides)."""
    allowed = fit.explore.get(rows, fit.configurations)
    forced = fit.forms.get(rows, {})
    listed = candidates(s, rows, op, tables, widths, fit.admitted)
    if forced:
        formed = lambda c: dict(c, **{k: v for k, v in forced.items() if admits(c, k, v, widths)})
        held = min(read, key=read.get) if read and rows not in fit.explore else None
        base = [c for c in listed if read_key(c) == held]
        listed = [formed(base[0])] if base else list({read_key(formed(c)): formed(c) for c in listed if not any(c.get(k, 0) for k in forced)}.values())
    priced = [(fit.time(c, s, rows), i, c) for i, c in enumerate(listed) if allowed is None or configuration(c) in allowed]
    read = [(read[read_key(c)], i, c) for _, i, c in priced if read_key(c) in (read or {})] if rows not in fit.explore else []
    if not priced:
        return None, None
    t, _, best = min(read or priced)
    return best, t


def read_key(spec):
    """An instance's identity among the readings: its kernel and template arguments but the slice's (TILES, WIDTHS) and
    the engine's (OUT, COOP)."""
    return (spec['kernel'],) + tuple(spec.get(k, 0) for k in ARGUMENTS[spec['kernel']] if k not in ('TILES', 'WIDTHS', 'OUT', 'COOP'))


def readings(obs):
    """Each slice's (outputs, columns, width class, rows) instances read in served profiles (observe's rows: the step's
    own conditions) by read_key, and their readings' median us."""
    out = {}
    for spec, s, rows, us, shape in obs:
        if shape.startswith('profile'):
            out.setdefault((s['outputs'], s['columns'], width_class(spec['WIDTHS']), rows), {}).setdefault(read_key(spec), []).append(us)
    return {k: {c: sorted(v)[len(v) // 2] for c, v in read.items()} for k, read in out.items()}


def spec_arguments(spec, T, widths, finish=False):
    """An instance's kernel and template arguments in ARGUMENTS' order on a slice of T tile columns and width class
    `widths`: its WIDTHS the class's widths 1 ... c as a mask (width_mask)."""
    kernel = spec['kernel']
    values = dict({form: 0 for form in FORMS}, **spec, TILES=T, WIDTHS=width_mask(widths), OUT=int(finish))
    if kernel == 'coop':
        slots = coop_slots()
        values['COOP'] = slots
    return kernel, tuple(values[name] for name in ARGUMENTS['panel' if kernel == 'panel_t' else kernel])


def width_mask(widths):
    """Width class c's WIDTHS mask, the widths 1 ... c (0 for none); width_class its inverse."""
    return (1 << (widths + 1)) - 2 if widths else 0


def width_class(mask):
    return max(0, mask.bit_length() - 1)


def spec_instance(spec, T, widths, finish=False, masked=False):
    """An instance's host name and instantiation (spec_arguments)."""
    kernel, arguments = spec_arguments(spec, T, widths, finish)
    if masked:
        arguments += (('streamed_plain', 'half', 1) if kernel == 'alu'
                      else ('half', 1) if kernel == 'panel' else (1,))
    return instance('streamed_' + kernel, *arguments)


def launch(spec, outputs, rows):
    """An instance's threadgroups (x, y), threads and K shares on a slice of `outputs` coded rows at `rows` input rows (a
    ring's band its producers and its consumers: RING 1 SK of each, the role split's (RING r >= 8) SK producers and M / r
    consumers)."""
    if spec['kernel'] == 'panel':
        return (outputs // 32, -(-rows // spec['M'])), COMPUTE['STREAMED_PANEL_TILES'] * COMPUTE['SIMD_WIDTH'], 1
    if spec['kernel'] == 'couple':
        return (outputs // 32, 1), COMPUTE['SIMD_WIDTH'] * (2 if spec['APART'] else 1) * spec['SK'], 1
    ring = spec.get('RING', 0)
    group = spec['SK'] + spec['M'] // ring if ring >= COMPUTE['COOPERATIVE_MIN_ROWS'] else spec['SK'] * (2 if ring else 1)
    return (outputs // 32 // spec['BANDS'], spec['J']), COMPUTE['SIMD_WIDTH'] * group * spec['BANDS'], spec['J']


def spec_of(text):
    """An instance spec from a decomposition operand's text, KIND/NAME=V,... (its template arguments by ARGUMENTS' names but
    the slice's TILES and WIDTHS and the node's COOP, which spec_arguments supplies; a name not given 0): the restored role
    split at 64 rows, for one, coop/M=64,SK=2,BANDS=1,J=1,LOADS=0,RING=32 (streamed.metal streamed_coop)."""
    kernel, _, named = text.partition('/')
    if kernel not in ARGUMENTS:
        raise ValueError(f'{text}: KIND/NAME=V,..., KIND one of {sorted(ARGUMENTS)}')
    given = dict(v.split('=', 1) for v in named.split(',') if v)
    unknown = set(given) - set(ARGUMENTS[kernel]) - {'TILES', 'WIDTHS', 'COOP'}
    if unknown or {'TILES', 'WIDTHS', 'COOP'} & set(given):
        raise ValueError(f'{text}: {kernel}\'s arguments are {ARGUMENTS[kernel]}, TILES, WIDTHS and COOP the slice\'s and the node\'s')
    spec = dict({k: int(given.get(k, 0)) for k in ARGUMENTS[kernel] if k not in ('TILES', 'WIDTHS', 'COOP', 'OUT')}, kernel=kernel)
    if kernel == 'couple':
        spec.update(BANDS=1, J=1, LUT=0)
    return spec


def admitted(spec, outputs, rows):
    """Instance `spec` on a slice of `outputs` coded rows at `rows` input rows, or an error where the slice does not admit
    it (its bands not a multiple of BANDS; a cooperative instance's rows past its M)."""
    if spec['kernel'] != 'panel' and (outputs // 32) % spec.get('BANDS', 1):
        raise ValueError(f'{spec}: {outputs // 32} bands, not a multiple of its BANDS')
    if spec['kernel'] == 'coop' and rows > spec['M']:
        raise ValueError(f'{spec}: {rows} rows past its M (the cooperative product holds every row in one threadgroup)')
    return spec


def decomposition_of(spec, outputs, columns, widths, rows):
    """A decomposition operand's entry for instance `spec` on a slice of `outputs` x `columns` (width class `widths`) at
    `rows` input rows (metal-microbench docs/kernels.md#decompositions: {"kernel", "args", "names", "grid", "threads",
    "shares"}, the args in ARGUMENTS' order; admitted)."""
    kernel, arguments = spec_arguments(admitted(spec, outputs, rows), columns // 32, widths)
    grid, threads, shares = launch(spec, outputs, rows)
    return {'kernel': kernel, 'args': list(arguments), 'names': list(ARGUMENTS[kernel]), 'grid': list(grid), 'threads': threads, 'shares': shares}


def coop_slots():
    """The declared layout; absence never launches a probe or changes kernels."""
    if _EVIDENCE is None or type(_EVIDENCE.get('coop')) is not int:
        raise RuntimeError(f'{_EVIDENCE_PATH}: missing cooperative tensor layout (coop); realize the backend capabilities before compilation')
    return _EVIDENCE['coop']


def read_cooperative_layout():
    """The node's cooperative-tensor slots as streamed_coop's COOP (docs/kernels.md#fragment-tiles), or None where
    matmul2d<M, 32, 32>'s right-input map is not the canonical lanes' up to bit permutations (M 8 and 16 read)."""
    if 'coop' in _PIPELINES:
        return _PIPELINES['coop']
    from . import _stream
    found = None
    for m in POLICY['cooperative_rows']:
        out = torch.zeros(32 * 130, dtype=torch.int32, device='mps')
        xs = torch.zeros(32 * 32, dtype=torch.float16, device='mps')
        _stream.encode(_pipeline(f'streamed_coopmap_{m}'), [None, out, None, None, xs] + [None] * 10, _dims(), 15, 1, 1, 32)
        slots = coop_of(out.cpu().view(32, 130).tolist())
        if slots is None or (found is not None and slots != found):
            found = None
            break
        found = slots
    if found is None:
        raise RuntimeError('the measured cooperative tensor layout is not represented by this backend; supply its layout before compilation')
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


def _encode(name, buffers, dims, gx, gy, threads, line='', *, outputs, indexed=(), pitched=(), indices=None):
    from .kernel import declare
    from .layout import strides
    from .operand import descriptor
    requirements = {i: descriptor(value).strides if i in pitched else strides(value.shape)
                    for i, value in enumerate(buffers) if value is not None}
    return declare('streamed.metal', name, buffers, outputs, dims, 15, (gx, gy, 1), (threads, 1, 1), line,
                   indexed=tuple(outputs) + tuple(indexed), requirements=requirements, indices=indices)


def _half(n, width):
    return torch.empty(n, width, dtype=torch.float16, device='mps')


def _identity(k):
    return torch.arange(k, dtype=torch.int32, device='mps')


_TYPES = {torch.float16: '', torch.float32: '_float', torch.bfloat16: '_bfloat'}


def widths_of(tiles):
    """A matrix's (or a slice's) width class: the least of WIDTH_CLASSES at least the widest width its tile words hold (0
    where none), so slices of one class share instances (the export's width caps are 8 and 12): WIDTH_CLASS[widest], the
    table's width_class, which the engine reads for its slices (streamed_weights.swift streamedWidths)."""
    return WIDTH_CLASS[int((((tiles.cpu().long() & 0xFFFFFFFF) >> 12) & 15).max()) if tiles.numel() else 0]


def _hadamard(signs, mix):
    """T = (H_K (x) H_N) diag(signs) / sqrt(K N) (computed in float64, float32, on the CPU): the rows its rotated
    coordinates (metal-microbench tools/model_code.py decode_layout's too)."""
    k, K = signs.shape[0], mix.shape[0]
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < k // K:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)])
    return (torch.kron(mix.double().cpu(), H) * signs.double().cpu()[None, :] / k ** 0.5).float()


class Matrix(torch.Tensor):
    """A logical matrix over encoded blocks and an optional input basis.

    ``rows``/``columns`` are the retained encoded domain. ``logical_shape`` is
    the tensor's domain before transpose. Immutable index/scale metadata stays
    available to preparation, including while FX carries fake tensor operands.
    """
    FIELDS = ('codes', 'tiles', 'bands', 'row_scale', 'col_scale', 'row_order', 'col_order', 'rotation',
              'rotation_scale', 'folded', 'signs', 'hadamard', 'input_order')
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, fields, name='', transposed=False, dtype=torch.float16, widths=None, units=None,
                logical_shape=None, metadata=None, input_domain=None):
        rows, columns = fields['tiles'].shape[0] * 32, fields['tiles'].shape[1] * 32
        logical_shape = tuple(logical_shape) if logical_shape is not None else (rows, columns)
        out = torch.Tensor._make_wrapper_subclass(cls, logical_shape[::-1] if transposed else logical_shape, dtype=dtype,
                                                  device=fields['codes'].device)
        for k in cls.FIELDS:
            setattr(out, k, fields.get(k))
        out.label, out.transposed, out.rows, out.columns, out.tiles_ = name, transposed, rows, columns, columns // 32
        out.logical_shape = logical_shape
        out.metadata = metadata if metadata is not None else {
            key: tuple(fields[key].detach().cpu().reshape(-1).tolist()) for key in cls.FIELDS[1:7]}
        out.input_domain = input_domain
        out.widths = widths if widths is not None else WIDTH_CLASS[max(((word >> 12) & 15 for word in out.metadata['tiles']), default=0)]
        out.units = units if units is not None else slice_units(out)
        return out

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def load(cls, tensors, name, device='mps'):
        """The matrix `name` of an export's tensors (fragment-1), on `device` (the CPU where only its tile words are read:
        a resolve's, an observe's, an epilogue's slices)."""
        assert f'{name}.tiles' in tensors, f'{name}: not a fragment-1 export (tools/model_code.py repack makes one)'
        words = lambda k: tensors[f'{name}.{k}'].contiguous().view(torch.int32).to(device)
        get = lambda k: tensors[f'{name}.{k}'].to(device)
        codes = tensors[f'{name}.codes'].contiguous()
        fields = {'codes': codes.view(torch.int32).to(device), 'tiles': words('tiles'), 'bands': words('bands'),
                  **{k: get(k) for k in cls.FIELDS[3:7]}}
        if f'{name}.rotation' in tensors:
            fields['rotation'], fields['rotation_scale'] = get('rotation'), get('rotation_scale')
            fields['folded'] = (fields['rotation'].float() * fields['rotation_scale'][:, None]).half()
        if f'{name}.signs' in tensors:
            fields['signs'], fields['hadamard'] = get('signs'), get('hadamard')
            k, K = fields['signs'].shape[0], fields['hadamard'].shape[0]
            assert k // K in (32, 64, 128) and K <= 20 and K % (256 // (k // K)) == 0 and k <= 2560, f'{name}: a Hadamard basis streamed_hadamard runs'
            fields['folded'] = _hadamard(tensors[f'{name}.signs'], tensors[f'{name}.hadamard'])[tensors[f'{name}.col_order'].long()].half().to(device)
        return cls(fields, name)

    def like(self, transposed=None, dtype=None, *, logical_shape=None, metadata=None, input_domain=None, **fields):
        if metadata is None:
            metadata = {**self.metadata, **{key: tuple(fields[key].detach().cpu().reshape(-1).tolist())
                        for key in self.metadata if key in fields}}
        return Matrix({**{k: getattr(self, k) for k in self.FIELDS}, **fields}, self.label,
                      self.transposed if transposed is None else transposed, dtype or self.dtype, self.widths,
                      self.units if 'tiles' not in fields else None,
                      self.logical_shape if logical_shape is None else logical_shape,
                      metadata,
                      self.input_domain if input_domain is None else input_domain)

    def __repr__(self):
        return (f'Matrix({self.label} [{self.rows} x {self.columns}]{" rotated" if self.folded is not None else ""}'
                f'{" transposed" if self.transposed else ""}, {self.dtype})')

    def __tensor_flatten__(self):
        return [k for k in self.FIELDS if getattr(self, k) is not None], (
            self.label, self.transposed, self.dtype, self.widths, self.units, self.logical_shape, self.metadata, self.input_domain)

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
        """Select logical axes and the encoded blocks/basis values they need.

        Cuts compose through the carried index maps. Only small immutable
        metadata is gathered during preparation; codes are never repacked.
        """
        from . import _stream
        row_slice = slice(*(rows or (None, None))).indices(self.logical_shape[0])
        col_slice = slice(*(columns or (None, None))).indices(self.logical_shape[1])
        if row_slice == (0, self.logical_shape[0], 1) and col_slice == (0, self.logical_shape[1], 1):
            return self.like()
        group = COMPUTE['WEIGHT_ROW_GROUP']
        rb, row_order = _stream.view_block_slice(self.metadata['row_order'], *row_slice, group)
        cb, col_order = (list(range(self.tiles_)), self.metadata['col_order']) if self.folded is not None else (
            _stream.view_block_slice(self.metadata['col_order'], *col_slice, group))
        ri = [b * group + j for b in rb for j in range(group)]
        ci = [b * group + j for b in cb for j in range(group)]
        metadata = dict(tiles=tuple(self.metadata['tiles'][r * self.tiles_ + c] for r in rb for c in cb),
                        bands=tuple(self.metadata['bands'][r] for r in rb),
                        row_scale=tuple(self.metadata['row_scale'][r] for r in ri), row_order=tuple(row_order),
                        col_scale=tuple(self.metadata['col_scale'][c] for c in ci), col_order=tuple(col_order))
        fields = {key: torch.tensor(values, dtype=getattr(self, key).dtype) for key, values in metadata.items()}
        fields['tiles'] = fields['tiles'].reshape(len(rb), len(cb))
        input_domain = self.input_domain
        if self.folded is not None:
            selection = slice(*col_slice)
            fields['folded'] = self.folded[:, selection]
            if self.rotation is not None:
                fields['rotation'] = self.rotation[:, selection]
            if self.signs is not None:
                original = input_domain if input_domain is not None else tuple(range(self.columns))
                _, input_domain = _stream.view_block_slice(original, *col_slice, len(original))
                input_domain = tuple(input_domain) or (-1,) * self.columns
                fields['input_order'] = torch.tensor(input_domain, dtype=torch.int32)
        out = self.like(logical_shape=(len(range(*row_slice)), len(range(*col_slice))), metadata=metadata,
                        input_domain=input_domain, **fields)
        out.label = f'{self.label}[{row_slice}, {col_slice}]'
        out.widths = WIDTH_CLASS[max((((word & 0xffffffff) >> 12) & 15 for word in metadata['tiles']), default=0)]
        return out

    @property
    def row_masked(self):
        return -1 in self.metadata['row_order']

    @property
    def column_masked(self):
        return -1 in self.metadata['col_order']

    def tabled(self, op):
        """Whether this matrix's `op` products may read 1-bit tiles by tables: the head's, its input pass (streamed_rotate,
        a rotation basis) writing them."""
        return op == 'head' and self.rotation is not None

    def dispatch(self, rows, op='product', decomposition=None):
        """The instance of a product of `rows` coded input rows on this node: the `decomposition` operand's (an instance
        spec: spec_of) where given, else the instance fit's choice; its spec, name and instantiation, threadgroups x and y,
        threads, K shares."""
        if decomposition is not None:
            spec = dict(admitted(decomposition, self.rows, rows))
        else:
            spec, _ = choose(Fit.node(), slice_units(self), rows, op, tables=self.tabled(op), widths=self.widths)
        (gx, gy), threads, shares = launch(spec, self.rows, rows)
        return (spec, *spec_instance(spec, self.tiles_, self.widths, masked=self.row_masked and spec['kernel'] == 'panel'), gx, gy, threads, shares)

    def dense(self):
        """The matrix decoded in its own coordinates, its shape and dtype (a rotated matrix decoded in its coded
        columns, then times `folded`: the rotation V^T in coded column order, its scale applied)."""
        width = self.columns if self.folded is not None else self.logical_shape[1]
        out = _half(self.logical_shape[0], width)
        if not all(self.logical_shape):
            return _half(*self.shape).to(self.dtype)
        columns = _identity(self.columns) if self.folded is not None else self.col_order
        name, line = instance('streamed_dense', 1) if self.row_masked or self.column_masked else ('streamed_dense', '')
        out, = _encode(name, self._code() + [None, out, self.row_order, self.row_scale, None, columns, self.col_scale],
                _dims(tiles=self.tiles_, pitch=self.pitch, stride=width), self.rows // 32, self.tiles_, 32, line,
                outputs=(4,), indexed=(5, 6, 8, 9), pitched=(1,))
        if self.folded is not None:
            out = torch.mm(out, self.folded)
        return (out.T if self.transposed else out).to(self.dtype)

    def coded_input(self, x, others=(), tables=None):
        """x [n, columns] fp16 in this matrix's coded input coordinates (and those of up to two matrices sharing its
        column order and rotation): gathered and scaled, or rotated (with `tables` [n, tiles, 8, 16] fp32, the first's
        tables for 1-bit tiles too)."""
        if any(self.columns != m.columns or self.logical_shape[1] != m.logical_shape[1]
               or self.metadata['col_order'] != m.metadata['col_order'] or self.input_domain != m.input_domain
               or any(getattr(self, key) is not getattr(m, key) for key in ('rotation', 'signs', 'hadamard', 'folded')) for m in others):
            first, tables = self.coded_input(x, tables=tables)
            return first + [m.coded_input(x)[0][0] for m in others], tables
        n = x.shape[0]
        matrices = [self, *others]
        outs = [_half(n, self.columns) for _ in matrices]
        if self.folded is None:
            pad = outs + [outs[0]] * (3 - len(outs))
            scales = [m.col_scale for m in matrices] + [self.col_scale] * (3 - len(matrices))
            name, line = instance('streamed_row_gather', 0, 0, 0, int(self.column_masked), self.logical_shape[1])
            outs = _encode(name, [x, pad[0], None, scales[0], None, pad[1], pad[2], scales[1], scales[2]],
                    _dims(columns=self.columns, inputs=len(matrices)), n, 1, 256, line, outputs=(1, 5, 6), indexed=(0, 3, 7, 8),
                    indices={2: self.metadata['col_order']})
        elif self.signs is not None and len(outs) <= 2:
            second = outs[1] if len(outs) > 1 else outs[0]
            name, line = (instance('streamed_hadamard', self.logical_shape[1], 1)
                          if self.input_order is not None else ('streamed_hadamard', ''))
            outs = _encode(name, [x, outs[0], second, self.col_order, self.col_scale,
                                          (others[0] if others else self).col_scale, self.signs, self.hadamard,
                                          None, None, None, None, self.input_order],
                    _dims(rows=n, columns=self.columns, per=self.columns // self.hadamard.shape[0], inputs=len(matrices)),
                    n, 1, 256, line, outputs=(1, 2), indexed=(0, 3, 4, 5, 6, 7, 12))
        elif n <= 16:
            second = outs[1] if len(outs) > 1 else outs[0]
            name, line = (instance('streamed_rotate', self.logical_shape[1])
                          if self.rotation.shape[1] != self.columns or self.rotation.stride() != (self.columns, 1)
                          else ('streamed_rotate', ''))
            made = _encode(name, [x, outs[0], second, None, self.col_scale,
                                        (others[0] if others else self).col_scale, self.rotation, self.rotation_scale, tables],
                    _dims(rows=n, columns=self.columns, inputs=len(matrices), flags=int(tables is not None)),
                    -(-self.columns // 32), 1, 256, line, outputs=(1, 2) + ((8,) if tables is not None else ()), indexed=(0, 4, 5, 6, 7))
            outs = made[:len(matrices)]
            if tables is not None: tables = made[-1]
        else:
            rotated = torch.mm(x, self.folded.T)
            pad = outs + [outs[0]] * (3 - len(outs))
            scales = [m.col_scale for m in matrices] + [self.col_scale] * (3 - len(matrices))
            name, line = instance('streamed_row_gather', 0, 0, 0)
            outs = _encode(name, [rotated, pad[0], None, scales[0], None, pad[1], pad[2],
                                        scales[1], scales[2]],
                    _dims(columns=self.columns, inputs=len(matrices)), n, 1, 256, line, outputs=(1, 5, 6), indexed=(0, 3, 7, 8),
                    indices={2: tuple(range(self.columns))})
        return outs, tables

    def product(self, xt, op='product', tables=None, decomposition=None):
        """The product of coded inputs xt [n, columns]: partials [shares, n, rows] in coded row order, unscaled (past 16
        rows one share, the panel's, or the `decomposition`'s: dispatch); `tables` (coded_input's) where the instance reads
        1-bit tiles by tables."""
        n = xt.shape[0]
        if not n or not self.rows or not self.columns:
            return torch.zeros(1, n, self.rows, dtype=torch.float32, device=xt.device)
        spec, name, line, gx, gy, threads, shares = self.dispatch(n, op, decomposition)
        assert (tables is not None) == bool(spec.get('LUT')), f'{name}: tables where it reads them'
        partials = torch.empty(shares, n, self.rows, dtype=torch.float32, device='mps')
        buffers = self._code() + [tables, xt, partials] + ([None] * 5 + [xt, None, self.row_scale] if spec['kernel'] == 'panel' else [])
        partials, = _encode(name, buffers, _dims(tiles=self.tiles_, outputs=self.rows, rows=n, per=-(-self.tiles_ // COMPUTE['STREAMED_PANEL_TILES']), split=n * self.rows,
                                     pitch=self.pitch), gx, gy, threads, line, outputs=(5,), pitched=(1,),
                indices={12: self.metadata['row_order']})
        return partials

    def product_t(self, xt, rows):
        """The transposed product of coded rows xt (n rows of `rows`, gathered and scaled, band-major [rows / 32][n][32]):
        partials [shares, n, columns] in coded column order, unscaled (bands of the coded rows a share)."""
        n, bands = rows, self.rows // 32
        if not n or not bands or not self.columns:
            return torch.zeros(1, n, self.columns, dtype=torch.float32, device=xt.device)
        m = 16 if n <= 16 else 64
        per = (4 if bands >= 64 else 1) if n <= 16 else bands
        shares = -(-bands // per)
        partials = torch.empty(shares, n, self.columns, dtype=torch.float32, device='mps')
        name, line = spec_instance(dict(kernel='panel_t', M=m), self.tiles_, self.widths)
        partials, = _encode(name, self._code() + [None, xt, partials],
                _dims(tiles=self.tiles_, outputs=self.rows, rows=n, per=per, split=n * self.columns, pitch=self.pitch),
                -(-self.tiles_ // COMPUTE['STREAMED_PANEL_TILES']), -(-n // m) * shares, COMPUTE['STREAMED_PANEL_TILES'] * COMPUTE['SIMD_WIDTH'], line, outputs=(5,), pitched=(1,))
        return partials

    def finish(self, partials, *, positions=None, scale=None, factor=None, dtype=torch.float16, columns=None):
        """Finish logical values through prepared views of partials and scales.

        The encoded-to-logical positions are matrix metadata. Padded lanes do
        not enter the reduction's domain; no order buffer is read at execution.
        """
        from .operand import TensorView
        shares, n, _ = partials.shape
        columns = self.logical_shape[0] if columns is None else columns
        positions = self.metadata['row_order'] if positions is None else positions
        inverse = {value: at for at, value in enumerate(positions) if value >= 0}
        selected = [inverse[value] for value in range(columns)]
        values = TensorView.of(partials).select(2, selected)
        scales = TensorView.of(self.row_scale if scale is None else scale).select(0, selected)
        y = torch.empty(n, columns, dtype=dtype, device='mps')
        name, line = (instance('streamed_finish', 'bfloat') if dtype == torch.bfloat16
                      else ('streamed_finish' + _TYPES[dtype], ''))
        y, = _encode(name, [values, y, None, scales, factor],
                _dims(outputs=columns, shares=shares, split=n * columns, count=n * columns, inputs=int(factor is not None)),
                -(-(n * columns) // 256), 1, 256, line, outputs=(1,), indexed=(0, 3, 4))
        return y

    def apply(self, xt, decomposition=None):
        """The product of coded inputs xt [n, columns] finished: y [n, rows] fp16 at its logical rows (the
        `decomposition`'s instance where given: dispatch; its partials finished by streamed_finish unless it is a panel)."""
        n = xt.shape[0]
        if not n or not all(self.logical_shape):
            return torch.zeros(n, self.logical_shape[0], dtype=xt.dtype, device=xt.device)
        if (n <= max(POLICY['cooperative_rows']) if decomposition is None else decomposition['kernel'] != 'panel'):
            return self.finish(self.product(xt, decomposition=decomposition))
        _, name, line, gx, gy, threads, _ = self.dispatch(n, decomposition=decomposition)
        y = _half(n, self.logical_shape[0])
        y, = _encode(name, self._code() + [None, xt, None, None, None, None, None, None, y, None, self.row_scale],
                _dims(tiles=self.tiles_, outputs=self.rows, rows=n, per=-(-self.tiles_ // COMPUTE['STREAMED_PANEL_TILES']), stride=self.logical_shape[0], finish=1,
                      pitch=self.pitch), gx, gy, threads, line, outputs=(11,), indexed=(13,), pitched=(1,),
                indices={12: self.metadata['row_order']})
        return y

    def times(self, x):
        """x [n, k] @ this matrix as its shape stands, in x's dtype: x A^T where transposed, else x A. Another dtype's
        rows are scaled into fp16's range by streamed_scale_rows and the products finished over the scale in that
        dtype (a backward's gradients lie far below fp16's normal range: cast as they are, a whole model's gradient a
        position 0.140 relative to the float32 model's, median; scaled 0.050)."""
        n, factor, dtype = x.shape[0], None, x.dtype
        if not n or not all(self.logical_shape):
            return torch.zeros(n, self.shape[1], dtype=dtype, device=x.device)
        x16 = x
        if dtype != torch.float16:
            x16, factor = _half(n, x.shape[1]), torch.empty(n, dtype=torch.float32, device='mps')
            x16, factor = _encode('streamed_scale_rows' + _TYPES[dtype], [x, x16, factor], _dims(columns=x.shape[1]), n, 1, COMPUTE['STREAMED_SCALE_THREADS'], outputs=(1, 2), indexed=(0,))
        if self.transposed:
            xt = self.coded_input(x16)[0][0]
            return self.apply(xt) if factor is None else self.finish(self.product(xt), factor=factor, dtype=dtype)
        xt = _half(n, self.rows)
        name, line = instance('streamed_row_gather', 0, 0, 32, int(self.row_masked), self.logical_shape[0])
        xt, = _encode(name, [x16, xt, None, self.row_scale],
                _dims(columns=self.rows, rows=n, inputs=1), n, 1, 256, line, outputs=(1,), indexed=(0, 3),
                indices={2: self.metadata['row_order']})
        partials = self.product_t(xt, n)
        if self.folded is None:
            return self.finish(partials, positions=self.metadata['col_order'], scale=self.col_scale, factor=factor, dtype=dtype,
                               columns=self.logical_shape[1])
        y = torch.mm(self.finish(partials, positions=range(self.columns), scale=self.col_scale, columns=self.columns), self.folded)
        return y if factor is None else (y.float() / factor[:, None]).to(dtype)

    @staticmethod
    def ffn(x, gate, up, down, decomposition=None):
        """Compose the projections through their logical neuron domain.

        Each encoded order remains part of its operand. The activation writes
        the down product's input directly where no numerical basis transform
        is required. Products, reductions and stores use shared tensor views.
        """
        from .operand import TensorView
        x16 = x.half()
        n = x16.shape[0]
        if not n or not all(gate.logical_shape) or not down.logical_shape[0]:
            return torch.zeros(n, down.logical_shape[0], dtype=x.dtype, device=x.device)
        (gx, ux), _ = gate.coded_input(x16, (up,))
        rotated = down.folded is not None
        order = tuple(range(down.logical_shape[1])) if rotated else down.metadata['col_order']
        hidden = _half(n, len(order))
        target = TensorView.of(hidden)
        scaled = not rotated and any(v != 1 for v in down.metadata['col_scale'])
        paired = (not gate.row_masked and gate.metadata['row_order'] == up.metadata['row_order']
                  and gate.columns == up.columns and gate.pitch == up.pitch and -1 not in order)
        spec = None if decomposition is not None or not paired else couple(
            Fit.node(), slice_units(gate), slice_units(up), n, widths=max(gate.widths, up.widths))[0]
        if spec is not None:
            positions = {value: at for at, value in enumerate(order)}
            stores = [positions[value] for value in gate.metadata['row_order']]
            kernel, args = spec_arguments(spec, gate.tiles_, max(gate.widths, up.widths))
            name, line = instance('streamed_' + kernel, *args, 0, int(scaled))
            (groups, _), threads, _ = launch(spec, gate.rows, n)
            scale = TensorView.of(down.col_scale).select(0, stores) if scaled else None
            hidden, = _encode(name, gate._code() + [None, gx, None] + up._code()
                + [ux, up.row_scale, target.select(1, stores), gate.row_order, gate.row_scale, scale],
                _dims(tiles=gate.tiles_, outputs=gate.rows, rows=n, pitch=gate.pitch), groups, 1, threads, line,
                outputs=(11,), indexed=(10, 12, 13, 14), pitched=(1, 7))
        else:
            g, u = gate.product(gx, decomposition=decomposition), up.product(ux, decomposition=decomposition)
            def rows(matrix):
                positions = {value: at for at, value in enumerate(matrix.metadata['row_order']) if value >= 0}
                return [positions[value] if value >= 0 else 0 for value in order]
            gi, ui = rows(gate), rows(up)
            name, line = instance('streamed_gelu', 'float', int(-1 in order), int(scaled))
            threads = COMPUTE['STREAMED_SCALE_THREADS']
            hidden, = _encode(name, [TensorView.of(g).select(2, gi), TensorView.of(u).select(2, ui),
                                    TensorView.of(gate.row_scale).select(0, gi), TensorView.of(up.row_scale).select(0, ui),
                                    target, down.col_order if -1 in order else None, down.col_scale if scaled else None],
                _dims(outputs=len(order), shares=g.shape[0], inputs=u.shape[0], split=n * len(order), per=n * len(order)),
                -(-len(order) // (4 * threads)), n, threads, line, outputs=(4,), indexed=(0, 1, 2, 3, 5, 6))
        coded = down.coded_input(hidden)[0][0] if rotated else hidden
        return down.apply(coded, decomposition).to(x.dtype)


def pair_units(sg, su):
    """A couple's slice units: gate's and up's summed, its outputs and columns gate's."""
    return {k: (sg[k] + su[k] if k not in ('outputs', 'columns') else sg[k]) for k in sg}


def mean_units(units):
    """The mean of slices' units (each rounded)."""
    return {k: round(sum(u[k] for u in units) / len(units)) for k in UNITS}


def couple(fit, sg, su, rows, read=None, widths=15):
    """The couple instance of gate and up (slice units sg and su, width class `widths`) at `rows` and its price (choose:
    `read` its readings) where it was read at the slice and rows, or the fit prices its argmin below their own products
    and the GELU's dispatch (a dispatch's fixed cost); else (None, None)."""
    pair = pair_units(sg, su)
    _, t = choose(fit, pair, rows, 'couple', widths=widths)
    if not read and (t is None or t >= sum(choose(fit, u, rows, widths=widths)[1] for u in (sg, su)) + fit.coefficients()['alu']['c0']):
        return None, None
    return choose(fit, pair, rows, 'couple', read=read, widths=widths)


def _no(m, what):
    raise NotImplementedError(f'{m.label}: a coded matrix has no {what}')


def _flip(m):
    return m.like(transposed=not m.transposed)


def _mm(a, b):
    if isinstance(b, Matrix):
        return b.times(a)
    return _flip(a).times(b.T).T


def _matmul(a, b):
    if isinstance(a, Matrix) or a.dim() == 2:
        return _mm(a, b)
    return _mm(a.reshape(-1, a.shape[-1]), b).reshape(*a.shape[:-1], b.shape[-1])


def _slice(m, dim=0, start=None, end=None, step=1):
    span = (start, end, step)
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
    from .algebra import linear_backward
    assert not mask[1], f'{weight.label}: a coded matrix is a constant (no gradient for its codes)'
    return linear_backward(x, grad, weight, mask)


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
    """Every coded matrix of an export, with its logical axes and encoded orders intact."""

    def __init__(self, path, device='mps'):
        from safetensors import safe_open
        from safetensors.torch import load_file
        tensors = {k: v for k, v in load_file(str(path)).items() if not k.startswith('__pad_')}
        with safe_open(str(path), 'pt') as f:
            layout = (f.metadata() or {}).get('basis_layout')
        assert layout == 'code-order' or not any(k.endswith('.rotation') for k in tensors), \
            f'{path}: its rotations are not in coded column order (metal-microbench tools/model_code.py repack rewrites them)'
        names = sorted({k.rsplit('.', 1)[0] for k in tensors if k.endswith('.codes')})
        self.matrices = {name: Matrix.load(tensors, name, device) for name in names}
        self.tensors = tensors
        for name in names:
            if not name.endswith('.mlp.gate_proj'):
                continue
            base = name[:-len('gate_proj')]
            gate, up = (self.matrices[base + p] for p in ('gate_proj', 'up_proj'))
            basis = [(getattr(gate, k), getattr(up, k)) for k in ('rotation', 'signs', 'hadamard')]
            shared = gate.folded is not None and torch.equal(gate.col_order, up.col_order) and all(
                (a is None) == (b is None) and (a is None or torch.equal(a, b)) for a, b in basis)
            self.matrices[base + 'up_proj'] = up.like(**({k: getattr(gate, k) for k in ('rotation', 'signs', 'hadamard', 'folded')}
                                                                             if shared else {}))

    def __getitem__(self, name):
        return self.matrices[name]


ARGUMENTS = {'alu': ('R', 'SK', 'BANDS', 'J', 'TILES', 'WIDTHS', 'OUT', 'LUT', 'F', 'LOADS', 'H'),
             'coop': ('M', 'SK', 'BANDS', 'J', 'TILES', 'WIDTHS', 'COOP', 'LOADS', 'RING'),
             'couple': ('R', 'SK', 'TILES', 'WIDTHS', 'F', 'APART', 'LOADS', 'H'),
             'panel': ('M', 'TILES', 'WIDTHS', 'LOADS')}


def parse_instance(name, loads=0):
    """An instance's spec from its host name (instance): its template arguments by name (ARGUMENTS); a name without its
    LOADS (named before the loads were an argument) takes `loads`, the form its library had (0 before mesh 290449d), one
    without its H the fp32 sums (H 0: every name before the fp16 pair sums), and a coop without its RING no ring."""
    parts = name.replace('streamed_', '').split('_')
    kernel = parts[0]
    if kernel not in ARGUMENTS:
        raise ValueError(name)
    values = [int(v) for v in parts[1:]]
    spec = dict(zip(ARGUMENTS[kernel], values + [loads if k == 'LOADS' else 0 for k in ARGUMENTS[kernel][len(values):]]), kernel=kernel)
    if kernel == 'couple':
        spec.update(BANDS=1, J=1, LUT=0)
    return spec


UNITS = ('outputs', 'columns', 'tiles', 'nonzero', 'planes', 'ones', 'bytes')
# Products sharing an input that an engine stacks into one (metal-microbench docs/kernels.md#stacked-products): a slice of
# their summed units, its outputs theirs.
STACKS = (('self_attn.q_proj', ('self_attn.k_proj', 'self_attn.v_proj')), ('mlp.gate_proj', ('mlp.up_proj',)))
# The product each epilogue runs in (metal-microbench docs/kernels.md#stacked-products, the epilogues: the engine's, its
# attention's q|k|v and its per-layer gate, streamed_weights.swift register): its first slice and those stacked with it,
# so a layer holding no k and v (a layer reading another's KV) runs the heads in its q alone.
EPILOGUES = {'mmb_heads': STACKS[0], 'mmb_ple': ('per_layer_input_gate', ())}


def stacked(held, name, first, rest, m):
    """The slices a stack of `m` (held as `name`, ending `first`) takes, each (its name's last part, slice): m, then each
    of `rest` of its prefix of m's columns; of a tensor held more than once (a layer's FFN in its part and again in its
    engines' gpu range or its prefill columns: slices) the one of m's rows, so a range's gate stacks with that range's up."""
    if not name.endswith(first):
        return []
    out = [(first.rsplit('.', 1)[-1], m)]
    cut = lambda u: u.label[u.label.index('['):] if '[' in u.label else ''
    for r in rest:
        found = [u for n, u in held if n == name[:-len(first)] + r and u.columns == m.columns]
        same = [u for u in found if u.rows == m.rows]
        ranged = [u for u in same if cut(u) == cut(m)]
        if ranged or len(same) == 1 or len(found) == 1:
            out.append((r.rsplit('.', 1)[-1], (ranged or same or found)[0]))
    return out


def products(held):
    """The products a member runs of the slices it holds (`held`: slices), {(op, outputs, columns, widths, tables): [units,
    ...]}: each slice's own (the head's op 'head'), each stack's (STACKS: the stacked slices' units summed, their outputs
    summed, their widest class), each FFN range's couple (op 'couple', each (gate's units, up's) where up's range is held
    beside gate's); and the epilogues' products among them, {(outputs, columns, widths): {kind: {stack, ...}}}, a stack
    named by its slices ('q_proj|k_proj|v_proj', or 'q_proj' in a layer holding no k and v: EPILOGUES)."""
    keyed, runs = {}, {}
    for name, m in held:
        op = 'head' if name == 'lm_head' else 'product'
        keyed.setdefault((op, m.rows, m.columns, m.widths, m.tabled(op)), []).append(slice_units(m))
        if name.endswith('.mlp.gate_proj'):
            ups = [u for n, u in held if n == name[:-len('gate_proj')] + 'up_proj' and u.rows == m.rows]
            up = next((u for u in ups if u.label[u.label.find('['):] == m.label[m.label.find('['):]), ups[0] if ups else None)
            if up is not None:
                keyed.setdefault(('couple', m.rows, m.columns, max(m.widths, up.widths), False), []).append((slice_units(m), slice_units(up)))
        for first, rest in STACKS:
            parts = stacked(held, name, first, rest, m)
            if len(parts) > 1:
                keyed.setdefault(stack_key(parts), []).append(stack_units(parts))
        for kind, (first, rest) in EPILOGUES.items():
            parts = stacked(held, name, first, rest, m)
            if parts:
                runs.setdefault(stack_key(parts)[1:4], {}).setdefault(kind, set()).add('|'.join(n for n, _ in parts))
    return keyed, runs


def stack_key(parts):
    """A stack's product key (products): its outputs summed, its columns, its widest width class."""
    return ('product', sum(u.rows for _, u in parts), parts[0][1].columns, max(u.widths for _, u in parts), False)


def stack_units(parts):
    """A stack's units: its slices' summed, its columns theirs."""
    return {k: sum(slice_units(u)[k] for _, u in parts) if k != 'columns' else parts[0][1].columns for k in UNITS}


def _write(path, document):
    """`document` written whole at `path`: a file of its own renamed over it, so a reader (a rank's resolve, a copy to a
    like node) meets the old document or the new one, never part of one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = path.with_name(f'{path.name}.{os.getpid()}.new')
    new.write_text(json.dumps(document, indent=1) + '\n')
    os.replace(new, path)


def reading_evidence(reading):
    """The node a served reading was read on: the instance evidence the engine priced it by (it names the file:
    engine.generation.evidence; the engine evaluates its instances and attention plans from it, metal-microbench
    docs/kernels.md#the-instance-fit), whose node and cores are the node's, [chip, GPU cores]; None where the reading,
    its evidence or the evidence's node is missing."""
    try:
        evidence = json.loads(Path(json.loads(Path(reading).read_text())['engine']['generation']['evidence']).read_text())
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return [evidence['node'], evidence['cores']] if evidence.get('node') and evidence.get('cores') else None


_SERVED = {}


def served_node(reading):
    """The node a served reading was read on (reading_evidence), [chip, GPU cores], or None."""
    if reading not in _SERVED:
        _SERVED[reading] = reading_evidence(reading)
    return _SERVED[reading]


def _records(path):
    """A record's rows, each with the node it was read on (`node`, [chip, GPU cores]): JSON lines (bandwidth's,
    observe's: a row's own node, else a served reading's row's reading's, served_node); None where neither names one."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [dict(r, node=r.get('node') or (served_node(r['source']) if r.get('source') else None)) for r in rows]


def observations(paths, here):
    """The observations of bandwidth and profile records made on node `here` ((chip, GPU cores): node): the
    admissible instances' (rows of instance, units, rows_in, best_us: (spec, units, rows, us, shape)), the decode
    attention plans' rows (kind attention) and the bandwidth reads (kind bandwidth). A record holding a row of another
    node, or of none, is refused, naming the nodes its rows name: a node's evidence is read on it (metal-microbench
    docs/kernels.md#the-instance-fit, Evidence)."""
    out, attention, bandwidth, foreign = [], [], [], {}
    for path in paths:
        for row in _records(path):
            if row['node'] != list(here):
                foreign.setdefault(str(path), set()).add(str(row['node']))
            elif row.get('kind') == 'attention':
                attention.append(row)
            elif row.get('kind') == 'bandwidth':
                bandwidth.append(row)
            else:
                spec = parse_instance(row['instance'])
                if spec['kernel'] == 'alu' and spec['R'] > 4:
                    continue
                out.append((spec, {k: row[k] for k in UNITS}, row['rows_in'], row['best_us'], row.get('form', row['instance']).split(' R')[0]))
    if foreign:
        raise SystemExit(f'observations not read on this {here[0]} ({here[1]} GPU cores): '
                         + '; '.join(f'{p} holds rows of {", ".join(sorted(n))}' for p, n in foreign.items()))
    return out, attention, bandwidth


PLANS = ('vector', 'tile')


def attention_reads(plan, dims, rows, heads, positions):
    """The KV bytes a decode attention plan reads (metal-microbench docs/kernels.md#decode-attention): the vector form's
    tasks each one query head of one row, the tile plan's each 8 of them, every task its K and V at `positions`."""
    tasks = rows * heads if plan == 'vector' else -(-rows * heads // 8)
    return tasks * positions * dims * 4


def attention_fit(rows):
    """Each plan's time on the node at each head dimension at the engine's part count (parts 0) as c0 + c1 MB of KV read,
    by least squares on relative error (c1 at least 0: a plan's time does not fall with the bytes it reads; c0 alone where
    the free line's c1 is negative), with its median error: keys 'plan dims'."""
    import numpy as np
    out = {}
    for plan in PLANS:
        for dims in sorted({r['dims'] for r in rows}):
            seen = [r for r in rows if r['plan'] == plan and r['dims'] == dims and not r.get('parts', 0)]
            if len(seen) < 2:
                continue
            X = np.array([[1.0, r['reads'] / 1e6] for r in seen])
            y = np.array([r['best_us'] for r in seen])
            c = np.linalg.lstsq(X / y[:, None], np.ones(len(y)), rcond=None)[0]
            if c[1] < 0:
                c = np.array([np.sum(1 / y) / np.sum(1 / y ** 2), 0.0])
            out[f'{plan} {dims}'] = {'c0': float(c[0]), 'mb': float(c[1]), 'count': len(seen),
                                     'median_log_error': float(np.median(np.abs(np.log((X @ c) / y))))}
    return out


ATTENTION_READ = ('plan', 'dims', 'rows', 'heads', 'positions', 'parts', 'merged')


def attention_readings(rows):
    """Each decode attention plan read at a shape (ATTENTION_READ: plan, head dimension, rows, query heads, positions,
    part count, whether the reading paid the merge stage) and its readings' median us."""
    read = {}
    for r in rows:
        read.setdefault((r['plan'], r['dims'], r['rows'], r['heads'], r['positions'], r.get('parts', 0), r.get('merged', r['plan'] == 'tile')),
                        []).append(r['best_us'])
    return {k: sorted(v)[len(v) // 2] for k, v in read.items()}


def attention_plan(model, dims, rows, heads, positions, read=None):
    """The plan and part count priced least at (dims, rows, heads, positions), with each one's price (us): each plan and
    part count read at the shape (`read`: attention_readings; a form read with its merge stage and without it, as an
    engine's own count moved, two candidates), else a plan at the engine's count (parts 0) by the node's model. A plan
    neither prices is refused (ValueError naming it): no plan is taken unpriced. Keys of the prices: PLAN, or PLAN/PARTS
    for a named count, ' merged' where the reading paid the merge stage."""
    priced = {}
    for (plan, d, r, h, p, parts, merged), us in (read or {}).items():
        if (d, r, h, p) == (dims, rows, heads, positions):
            priced[(plan, parts, merged)] = us
    for plan in PLANS:
        if not any(k[:2] == (plan, 0) for k in priced) and f'{plan} {dims}' in (model or {}):
            line = model[f'{plan} {dims}']
            priced[(plan, 0, None)] = line['c0'] + line['mb'] * attention_reads(plan, dims, rows, heads, positions) / 1e6
    named = {f'{plan}{f"/{parts}" if parts else ""}{" merged" if merged else ""}': us for (plan, parts, merged), us in priced.items()}
    unpriced = [plan for plan in PLANS if plan not in {k[0] for k in priced}]
    if unpriced:
        raise ValueError(f'the {" and ".join(unpriced)} plan at {dims} channels: not read at {rows} rows of {heads} heads at '
                         f'{positions} positions, and no line of the node\'s attention model')
    return min(priced, key=priced.get)[:2], named


def _fit(args):
    """This node's evidence (Fit.node) from observations read on it (observations): the coefficients and factors
    (Fit.fit), the configurations observed, the attention model (attention_fit), and the served readings (readings)
    and attention readings (attention_readings) themselves, so the file is the whole of what prices the node's
    instances wherever it lies (a like node's copy among them; the records stay where they were read); the bandwidth the
    median of its bandwidth reads (bandwidth), else, a refit, the one this node's evidence was fitted at. The epilogues'
    and forms' entries (fused, forms_at_bar) are kept."""
    here = node()
    obs, attention, bandwidth = observations(args.observations, here)
    path = evidence_path()
    kept = json.loads(path.read_text()) if path.exists() else {}
    if bandwidth:
        gbps, source = sorted(r['useful_gigabytes_per_second'] for r in bandwidth)[len(bandwidth) // 2], 'read'
    elif [kept.get('node'), kept.get('cores')] == list(here) and kept.get('gbps'):
        gbps, source = kept['gbps'], f'kept ({path})'
    else:
        raise SystemExit(f'no bandwidth read of this {here[0]} ({here[1]} GPU cores) among the observations, and no fit of it at {path} '
                         f'(`python examples/memory-read.py --mib M --rounds N` reads it, a row of the observations)')
    fit = Fit(here[1], gbps)
    loss = fit.fit(obs)
    errors = sorted(abs(math.log(fit.time(spec, s, n) / t)) for spec, s, n, t, _ in obs)
    held, regrets = [], []
    for shape in sorted({o[4] for o in obs}):
        other = Fit(here[1], gbps)
        other.fit([o for o in obs if o[4] != shape], iters=POLICY['fit_holdout_iterations'])
        for n in sorted({o[2] for o in obs if o[4] == shape}):
            seen = [(other.time(spec, s, n), t) for spec, s, nn, t, sh in obs if sh == shape and nn == n]
            held += [abs(math.log(p / t)) for p, t in seen]
            regrets.append(min(seen)[1] / min(t for _, t in seen) - 1)
    held.sort(); regrets.sort()
    record = {'node': here[0], 'cores': here[1], 'gbps': gbps, 'bandwidth': source, 'observations': [str(Path(p).resolve()) for p in args.observations],
              'configurations': sorted({configuration(o[0]) for o in obs}),
              'count': len(obs), 'loss': loss, 'median_log_error': errors[len(errors) // 2],
              'held_out': {'median_log_error': held[len(held) // 2], 'median_regret': regrets[len(regrets) // 2],
                           'worst_regret': regrets[-1]}, 'coefficients': fit.coefficients(), 'factors': fit.factors,
              'attention': attention_fit(attention), 'coop': read_cooperative_layout(),
              'readings': [{'slice': list(k), 'instance': list(c), 'us': us} for k, read in sorted(readings(obs).items()) for c, us in sorted(read.items())],
              'attention_readings': [dict(zip(ATTENTION_READ, k), us=us) for k, us in sorted(attention_readings(attention).items())],
              'fused': kept.get('fused', {}), 'forms_at_bar': kept.get('forms_at_bar', {})}
    _write(path, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('coefficients', 'readings', 'attention_readings')}))


def _observe(args):
    """A served reading's profile (metal-microbench tools/reading.py read --profile: per class, per command label its least
    us a dispatch) as observations of its instances, each row naming the node it was read on (the instance evidence the
    engine priced the reading by: reading_evidence): each label's slices the export's products of its shape and widths (products: a slice's own,
    a stack's, a couple's gate and up summed; their units' mean), its time less the class's floor (its least command), a
    label without its LOADS taking --loads (the form the reading's library had); and each decode class's attention plans
    (a dimension's vector form, or its tile with its merge stage where the class paid one: merged) as attention rows, at
    the window or --ctx positions, each at the engine's own part count (0: the engine evaluates its plans from its
    evidence and takes none from outside)."""
    import re
    made = reading_evidence(args.reading)
    if made is None:
        raise SystemExit(f'{args.reading}: names no instance evidence that names its node (engine.generation.evidence: a fit writes node and cores)')
    reading = json.loads(Path(args.reading).read_text())
    streamed = Streamed(args.export, 'cpu')
    record = _record(args.program) if args.program else None
    keyed, _ = products(slices(streamed, record, args.rank))
    shapes = {}
    for (op, outputs, columns, widths, _), units in keyed.items():
        shapes.setdefault((op == 'couple', outputs, columns, widths), []).extend(pair_units(*u) if op == 'couple' else u for u in units)
    out = []
    for cls, commands in reading['engine']['profile']['commands'].items():
        rows, floor = int(cls.split()[1]), min(c['least_us'] for c in commands.values())
        for label, c in commands.items():
            found = re.fullmatch(r'streamed_(alu|coop|couple|panel)_([0-9_]+) (\d+)x(\d+)', label)
            if not found:
                continue
            spec = parse_instance(label.split()[0], args.loads)
            outputs, columns = int(found.group(3)), int(found.group(4))
            units, n = shapes.get((spec['kernel'] == 'couple', outputs, columns, width_class(spec['WIDTHS']))), min(rows, spec.get('R', rows))
            if units:
                out.append({'instance': '_'.join(['streamed_' + spec['kernel']] + [str(spec[k]) for k in ARGUMENTS[spec['kernel']]]),
                            'rows_in': n, 'best_us': max(c['least_us'] - floor, 0.1), 'form': f'profile {outputs}x{columns} R{n}',
                            'source': str(Path(args.reading).resolve()), 'node': made, **mean_units(units)})
    text = _text_config(args.export)
    windows = {text['head_dim']: text.get('sliding_window', 0), text.get('global_head_dim') or text['head_dim']: 0}
    kinds, held = text.get('layer_types') or [], {}
    part = record['ranks'][args.rank] if record else None
    for L, kind in enumerate(kinds):
        layer = ((part or {}).get('layers') or [None] * len(kinds))[L]
        heads = (layer['heads'][1] - layer['heads'][0]) if layer and layer.get('heads') else (0 if part else text['num_attention_heads'])
        if heads:
            held.setdefault(text['head_dim'] if kind == 'sliding_attention' else text.get('global_head_dim') or text['head_dim'], set()).add(heads)
    for dims, counts in held.items():
        if len(counts) != 1:
            print(f'{args.reading}: the member\'s layers of {dims} channels hold {sorted(counts)} query heads, one a reading; their attention rows not kept')
    for cls, commands in reading['engine']['profile']['commands'].items():
        for dims, window in windows.items() if cls.startswith('decode') else ():
            if len(held.get(dims, ())) != 1:
                continue
            heads = next(iter(held[dims]))
            for plan, label in (('vector', rf'decode_attention_{dims}(?:_[12])?(?:_mesh_[a-z]+_[0-9]+_)?'), ('tile', rf'decode_attention_tile_{dims}(?:_mesh_[a-z]+_[0-9]+_)?')):
                seen = [c['least_us'] for l, c in commands.items() if re.fullmatch(label, l)]
                if seen:
                    merge = commands.get(f'decode_attention_merge_{dims}', {}).get('least_us', 0.0) if plan == 'tile' else 0.0
                    rows, positions = int(cls.split()[1]), window or args.ctx
                    out.append({'kind': 'attention', 'plan': plan, 'dims': dims, 'rows': rows, 'heads': heads, 'positions': positions, 'window': window,
                                'reads': attention_reads(plan, dims, rows, heads, positions), 'best_us': min(seen) + merge,
                                'parts': 0, 'merged': merge > 0, 'source': str(Path(args.reading).resolve()), 'node': made})
    with open(args.out, 'a') as f:
        for row in out:
            f.write(json.dumps(row) + '\n')
    print(f'{args.out}: {len(out)} observations from {args.reading}, read on {made[0]} ({made[1]} GPU cores)')


def fused_ratios(fused, separate, rows_of):
    """Each kind run fused into its product, for each product (its label's OUTPUTSxCOLUMNS and the instance's width class),
    at each decode class two served readings ticked three times at least in both (a reading's other classes are its
    first, cold ticks): [log least ratio, log median ratio] of the fused form (the first reading) against the product
    apart (the second) and the row the kind replaces, us a dispatch (metal-microbench docs/kernels.md#stacked-products, the
    epilogues; #the-instance-fit, the finish). An epilogue (`rows_of`: its name to its row's label) runs in the instance
    whose name ends in it, apart the same instance without it; the finish ('finish') is a one-share ALU instance finishing
    its rows (OUT 1), apart the same instance's partials (OUT 0) and their streamed_finish of the same shape."""
    import re
    profiles = [json.loads(Path(p).read_text())['engine']['profile'] for p in (fused, separate)]
    one, two = (p['commands'] for p in profiles)
    out = {}
    for cls in sorted(set(one) & set(two)):
        if not cls.startswith('decode') or min(p['ticks'].get(cls, {}).get('ticks', 0) for p in profiles) < 3:
            continue
        a, b = one[cls], two[cls]
        for label in a:
            found = re.fullmatch(r'(streamed_(?:alu|coop)_[0-9_]+)(?:_([a-z_]+))? (\d+x\d+)', label)
            if not found:
                continue
            name, kind, shape = found.groups()
            spec = parse_instance(name)
            if kind is None and spec['kernel'] == 'alu' and spec['OUT'] == 1:
                kind, plain = 'finish', '_'.join(['streamed_alu'] + [str(0 if k == 'OUT' else spec[k]) for k in ARGUMENTS['alu']]) + ' ' + shape
                row = f'streamed_finish {shape}'
            elif kind in rows_of:
                plain, row = f'{name} {shape}', rows_of[kind]
            else:
                continue
            if plain in b and row in b:
                ratio = lambda stat: math.log(a[label][stat] / (b[plain][stat] + b[row][stat]))
                out.setdefault(kind, {}).setdefault(f'{shape} {width_class(spec["WIDTHS"])}', {})[cls.split()[1]] = [ratio('least_us'), ratio('median_us')]
    return out


def fused_classes(ratios, classes):
    """Each kind's rows classes for each product where it runs fused. A kind has a side it keeps unread (the finish fused,
    a dispatch fewer; an epilogue apart): a read class takes the other side where its least and median ratios both say so
    (both above 0 for the finish, both below for an epilogue), an unread class where its nearest read classes below and
    above both did; a product never read keeps the kind's side (the caller's default)."""
    out = {}
    for kind, shapes in ratios.items():
        fused = kind == 'finish'
        other = lambda r: all(v > 0 for v in r) if fused else all(v < 0 for v in r)
        for shape, read in shapes.items():
            read = {int(c): r for c, r in read.items()}
            below = lambda c: [k for k in read if k < c]
            above = lambda c: [k for k in read if k > c]
            out.setdefault(kind, {})[shape] = [c for c in classes if fused != (other(read[c]) if c in read else bool(below(c) and above(c))
                                                                                  and other(read[max(below(c))]) and other(read[min(above(c))]))]
    return out


def _epilogue(args):
    """Two served readings of this node's (served_node), the kinds fused and apart (fused_ratios), into its evidence
    (fused): the finish keyed by its product's shape and width class, an epilogue by the product it runs in too, the
    stack the member that served the readings holds at that shape (products: 'OUTPUTSxCOLUMNS WIDTHS q_proj|k_proj|v_proj',
    a layer's q|k|v; '... q_proj' where a layer holds no k and v), so a ratio read on one product is taken by no other of
    its shape; one no product of the member runs, or several of different stacks, is not kept (each printed)."""
    here = list(node())
    for reading in (args.fused, args.separate):
        if served_node(reading) != here:
            raise SystemExit(f'{reading}: read on {served_node(reading) or "no named node"} (its instance evidence\'s), not this {here[0]} ({here[1]} GPU cores)')
    _, runs = products(slices(Streamed(args.export, 'cpu'), _record(args.program) if args.program else None, args.rank))
    path = evidence_path()
    known = json.loads(path.read_text()) if path.exists() else {}
    for kind, shapes in fused_ratios(args.fused, args.separate, dict(r.split('=', 1) for r in args.rows)).items():
        for shape, classes in shapes.items():
            if kind != 'finish':
                size, widths = shape.split()
                stacks = runs.get((*map(int, size.split('x')), int(widths)), {}).get(kind, set())
                if len(stacks) != 1:
                    print(f'{kind} at {shape}: {"no product" if not stacks else "products of stacks " + ", ".join(sorted(stacks))} of the member '
                          f'run{"s" if not stacks else ""} it; not kept')
                    continue
                shape = f'{shape} {next(iter(stacks))}'
            known.setdefault('fused', {}).setdefault(kind, {}).setdefault(shape, {}).update(classes)
    _write(path, known)
    print(json.dumps(known.get('fused', {})))


def _record(source):
    """A call's record: `source` a path, or "-" for the record on stdin (a resident caller's stream, never a file)."""
    import sys
    return json.loads(sys.stdin.read() if source == '-' else Path(source).read_text())


def _admit(args):
    """A form admitted on this node (forms_at_bar in its evidence: candidates lists it) by a decode KL record of the engine
    served with it (metal-microbench tools/retained.py's JSON line: kl, positions), at or below the bar the caller
    states (`bar`: the operator's, metal-microbench docs/measurement.md); a record above it is refused."""
    import re
    if not re.fullmatch(r'H[0-9]+', args.form):
        raise SystemExit(f'{args.form}: a form admitted at the bar is H<h> (the fp16 pair sums: FORMS), the forms whose numerics differ')
    lines = [json.loads(line) for line in Path(args.record).read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise SystemExit(f'{args.record}: one KL record (a JSON line of kl and positions), not {len(lines)}')
    record = lines[0]
    if record['kl'] > args.bar:
        raise SystemExit(f'{args.record}: KL {record["kl"]:.4f} above the bar {args.bar}; {args.form} not admitted')
    path = evidence_path()
    known = json.loads(path.read_text()) if path.exists() else {}
    known.setdefault('forms_at_bar', {})[args.form] = {'kl': record['kl'], 'bar': args.bar, 'positions': record.get('positions'),
                                                       'record': str(Path(args.record).resolve())}
    _write(path, known)
    print(json.dumps(known['forms_at_bar']))


def prepare_read(data, output, count, repetitions):
    """Bind a complete streamed memory-read function before submission."""
    from . import _stream
    kernel = _pipeline('streamed_read')
    schedule = _stream.Schedule()
    for _ in range(repetitions):
        schedule.compute(kernel, [data, output], _dims(count=count), 15, 1024, 1, 256, 1 | (2 << COMPUTE['ACCESS_WRITE_SHIFT']))
    schedule.finish()
    return schedule


def lattice(record, axis, key=None):
    """The (extent, grain) of each of the call record's share decisions on `axis` (metal-microbench tools/mesh/programs.py
    decisions: 'column', 'head', 'row'), those whose keys name `key` where given ('<layer>.mlp', '<layer>.attention'),
    each once."""
    return sorted({(sum(d['parts']), d['grain']) for d in ((record or {}).get('decisions') or {}).values()
                   if d.get('axis') == axis and (key is None or key in (d.get('keys') or ()))})


def spans(whole, extent, grain, held):
    """Every range of `whole` units a ragged decision of `extent` units in grains of `grain` (a layer's scaled from the
    decision's, an MoE or a wider layer's carried) gives the member holding `held` ([a, b) of the whole): each width
    on the lattice, from the member's side of the extent (its end where it holds the last unit and not the first,
    else its start, a middle member's from its first unit as far as the extent allows)."""
    step = grain * whole // extent
    if not step or step * extent != grain * whole:
        return []
    a, b = held
    out = []
    for n in range(step, whole + 1, step):
        first = whole - n if b == whole and a > 0 else min(a, whole - n)
        out.append((first, first + n))
    return out


def slices(streamed, record=None, rank=None):
    """Each coded matrix's slice a rank holds (its record's slices: a tensor's output ranges are rows, its input range
    columns; the head's rows its vocabulary; and each layer's FFN neurons its GPU computes in a phase of their own, its
    engines' gpu range and its prefill columns), every matrix whole without a record: (name, matrix).  Where the record's
    `ragged` names an axis (metal-microbench docs/placement.md#the-records-contract-for-every-ragged-axis: each call's
    ranges of it are its data, of the whole the member holds), every width on each of that axis's decisions' lattices
    (spans), each a view at the whole matrix's code widths: `column`, each layer's FFN neurons (gate's and up's rows,
    down's columns) for the decode and the prefill column decisions, the GPU's range of the prefill columns among them
    (the prefill columns less whole blocks at one end: a width on the prefill lattice); `head`, each layer's q rows and o
    columns of each head count (k and v whole, every KV head); `row`, the head's vocabulary rows.  The resolver takes
    every slice at every row count of the call (call_rows), the decode step's and the prefill chunk's."""
    part = record['ranks'][rank] if record else None
    out, seen = [], set()
    ragged = set((record or {}).get('ragged') or ())

    def add(name, m, whole=None):
        if whole is not None:
            m.widths = whole.widths
        if (name, m.label) not in seen:
            seen.add((name, m.label))
            out.append((name, m))

    for L, layer in enumerate((part or {}).get('layers') or []):
        base = f'model.language_model.layers.{L}.mlp.'
        if 'column' in ragged and base + 'gate_proj' in streamed.matrices:
            gate, up, down = (streamed[base + p] for p in ('gate_proj', 'up_proj', 'down_proj'))
            held = (layer.get('prefill') or {}).get('columns') or layer.get('columns')
            for extent, grain in lattice(record, 'column', f'{L}.mlp') if held else ():
                for span in spans(gate.rows, extent, grain, held):
                    for name, m, cut in ((base + 'gate_proj', gate, dict(rows=span)), (base + 'up_proj', up, dict(rows=span)),
                                         (base + 'down_proj', down, dict(columns=span))):
                        add(name, m.cut(**cut), m)
        else:
            for span in ((layer.get('engines') or {}).get('gpu'), (layer.get('prefill') or {}).get('columns')):
                if span and span[1] > span[0] and base + 'gate_proj' in streamed.matrices:
                    for p in ('gate_proj', 'up_proj'):
                        add(base + p, streamed[base + p].cut(rows=tuple(span)))
                    add(base + 'down_proj', streamed[base + 'down_proj'].cut(columns=tuple(span)))
        base = f'model.language_model.layers.{L}.self_attn.'
        if 'head' in ragged and layer.get('heads') and base + 'q_proj' in streamed.matrices:
            q, o = streamed[base + 'q_proj'], streamed[base + 'o_proj']
            for extent, grain in lattice(record, 'head', f'{L}.attention'):
                dim = q.rows // extent
                for first, last in spans(extent, extent, grain, layer['heads']):
                    add(base + 'q_proj', q.cut(rows=(first * dim, last * dim)), q)
                    add(base + 'o_proj', o.cut(columns=(first * dim, last * dim)), o)
    for name, m in streamed.matrices.items():
        if part is None:
            out.append((name, m))
            continue
        if name == 'lm_head':
            v = part.get('vocabulary')
            if v and 'row' in ragged:
                for extent, grain in lattice(record, 'row'):
                    for span in spans(m.rows, extent, grain, v):
                        add(name, m.cut(rows=span), m)
            elif v:
                add(name, m.cut(rows=tuple(v)))
            continue
        cut = (part.get('slices') or {}).get(name + '.weight')
        if cut is None:
            continue
        rows = tuple(cut['output'][0]) if cut.get('output') else None
        columns = tuple(cut['input']) if cut.get('input') else None
        if (rows and rows[0] == rows[1]) or (columns and columns[0] == columns[1]):
            continue
        add(name, m.cut(rows=rows, columns=columns))
    return out


# metal-microbench tools/mesh/programs.py DRAFT_PRODUCTS: the kind of the `draft <kind>` decision that splits each drafter
# product, by its export's name (a layer's in every layer); a convolution's tap generator is no decision's
DRAFT_KINDS = {'fc': 'fc', 'self_attn.qk': 'qk', 'self_attn.o_proj': 'o', 'mlp.gate_up': 'gate_up', 'mlp.down_proj': 'down',
               'head_gate': 'head'}


def draft_kind(name):
    """The kind of the `draft <kind>` decision that splits the drafter's matrix `name` (DRAFT_KINDS), else None."""
    return DRAFT_KINDS.get(name.split('.', 2)[2] if name.startswith('layers.') else name)


def _band_view(m, a, b):
    """Coded rows [a, b) of `m` (whole bands) as the engine's StreamedMatrix.view takes them: their bands, tile words, row
    scales and row order (the places among the whole's outputs they write: the drafter's export orders a product's rows by
    energy, so a run of coded rows is natural rows scattered over the product), at the whole's width class.  A slice the
    resolver keys and prices (products: outputs, columns, width class, units); no product of it runs here."""
    assert a % 32 == 0 and b % 32 == 0 and 0 <= a < b <= m.rows, f'{m.label}: a view of whole bands, not [{a}:{b}]'
    out = m.like(tiles=m.tiles[a // 32:b // 32], bands=m.bands[a // 32:b // 32], row_scale=m.row_scale[a:b].contiguous(),
                 row_order=m.row_order[a:b].contiguous(), metadata={**m.metadata,
                     'tiles': m.metadata['tiles'][a // 32 * m.tiles_:b // 32 * m.tiles_],
                     'bands': m.metadata['bands'][a // 32:b // 32],
                     'row_scale': m.metadata['row_scale'][a:b], 'row_order': m.metadata['row_order'][a:b]})
    out.label = f'{m.label}[{a}:{b}] view'
    return out


def draft_slices(drafter, record=None, rank=None):
    """The drafter's slices a rank holds (metal-microbench docs/kernels.md#the-drafter-as-a-share), (name, matrix): each
    matrix whole, but a product the record's speculate gives the rank a run of (`draft`, one {kind: [start, end)} a
    drafting member in `drafting` order: programs.py draft_shares; the engine's Drafter.part runs it as a view): where
    the record's `ragged` names `draft`, every width on that product's `draft <kind>` decision's lattice (spans: whole
    bands of 32 coded rows, from the member's side), as slices takes the head's rows on the `row` lattice, else its run;
    each a view of the whole's bands (_band_view) at the whole matrix's code widths.  Without a record, or on a rank
    that drafts no share, every matrix whole (one drafting member: HEAD's form)."""
    speculate = (record or {}).get('speculate') or {}
    drafting = list(speculate.get('drafting') or ([speculate['on']] if 'on' in speculate else []))
    runs = speculate.get('draft') or []
    own = runs[drafting.index(rank)] if rank in drafting and len(runs) == len(drafting) else {}
    ragged = 'draft' in ((record or {}).get('ragged') or ())
    out, seen = [], set()
    for name, m in drafter.matrices.items():
        kind = draft_kind(name)
        held = own.get(kind) if kind else None
        if not held:
            out.append((name, m))
            continue
        widths = [s for extent, grain in lattice(record, 'draft', f'draft {kind}') for s in spans(m.rows, extent, grain, held)] if ragged else []
        for a, b in widths or [tuple(held)]:
            if (name, a, b) not in seen:
                seen.add((name, a, b))
                out.append((name, _band_view(m, a, b)))
    return out


def _text_config(export):
    """The text model's configuration an export names (its metadata's model directory)."""
    from safetensors import safe_open
    with safe_open(str(export), 'pt') as f:
        model = Path((f.metadata() or {}).get('model', ''))
    config = json.loads((model / 'config.json').read_text()) if (model / 'config.json').exists() else {}
    return config.get('text_config', config)


def _attention(export, part, ctx, counts, fit_model, read, explore=None):
    """The decode attention plan a rank takes for each of its layers (its head dimension, window and query heads) at each
    row count it runs (`counts`; docs/kernels.md#decode-attention) and its part count (0: the engine's own count): the
    least priced (attention_plan: its readings `read`, else the node's model) at the layer's positions (the window, else
    the call's context); at a row count `explore` names ({rows: (plan, parts, sliding)}: every layer's, or a sliding
    window's alone), that plan at that count, unpriced (read once, in the next profile). A layer a named plan does not
    cover needs the model's line of both plans at its head dimension; where one lacks, the resolve is refused, naming
    them (a served reading of each plan, a resolve --plan naming it, and observe and fit make them)."""
    text = _text_config(export)
    kinds = text.get('layer_types') or []
    heads = text.get('num_attention_heads')
    if not kinds or not heads:
        return []
    out, missing = [], set()
    for L, kind in enumerate(kinds):
        layer = ((part or {}).get('layers') or [None] * len(kinds))[L]
        held = (layer['heads'][1] - layer['heads'][0]) if layer and layer.get('heads') else (0 if part else heads)
        if held <= 0:
            continue
        sliding = kind == 'sliding_attention'
        dims = text['head_dim'] if sliding else text.get('global_head_dim') or text['head_dim']
        window = text.get('sliding_window', 0) if sliding else 0
        for rows in counts:
            named = (explore or {}).get(rows)
            if named and (window or not named[2]):
                out.append({'layer': L, 'rows': rows, 'plan': named[0], 'parts': named[1], 'us': {}})
                continue
            lacking = {f'{plan} {dims}' for plan in PLANS if f'{plan} {dims}' not in (fit_model or {})}
            if lacking:
                missing |= lacking
                continue
            (plan, parts), priced = attention_plan(fit_model, dims, rows, held, window or ctx, read)
            out.append({'layer': L, 'rows': rows, 'plan': plan, 'parts': parts, 'us': {k: round(v, 2) for k, v in priced.items()}})
    if missing:
        raise SystemExit(f'{evidence_path()}: no attention line for {", ".join(sorted(missing))} (plan, head dimension) on this {node()[0]}: '
                         f'a served reading of each plan (resolve --plan R=PLAN), observe and fit make them (metal-microbench '
                         f'docs/kernels.md#the-instance-fit, Evidence)')
    return out


def call_rows(record):
    """The row counts a member of the call `record` runs (metal-microbench metal_runtime.swift Placement and
    engine_recording.swift): its decode step's (bs: a speculative step's rows are its batch class), each prefill chunk's
    (its streams times the chunk's rows) and its last chunk's head (its streams)."""
    bs = int(record['bs'])
    prefill = record.get('prefill') or {}
    streams = int(prefill.get('streams') or bs)
    return sorted({bs} | ({streams} | {streams * int(c['rows']) for c in prefill.get('chunks') or []} if prefill else set()))


def _resolve(args):
    import re
    streamed = Streamed(args.export, 'cpu')
    record = _record(args.program) if args.program else {}
    if args.rows:
        classes = sorted({int(c) for c in args.rows.split(',')})
    elif record:
        classes = call_rows(record)
    else:
        raise SystemExit('resolve: --rows (the row counts the engine runs: its batch buckets) or --program (a call\'s record, whose rows call_rows reads)')
    fit = Fit.node()
    for e in args.explore:
        rows, named = e.split('=', 1)
        parts = [c.split('/') for c in named.split(',')]
        forms = {}
        for c in parts:
            for v in c:
                found = re.fullmatch(f'({"|".join(FORMS)})([0-9]+)', v)
                if found:
                    forms.setdefault(found.group(1), set()).add(int(found.group(2)))
        if any(len(v) > 1 for v in forms.values()):
            raise SystemExit(f'--explore {e}: one value a form a rows class, not {forms}')
        configurations = {tuple(int(v) if v.isdigit() else v for v in c if not re.fullmatch(f'({"|".join(FORMS)})[0-9]+', v)) for c in parts}
        if configurations - {()}:
            fit.explore[int(rows)] = configurations - {()}
        fit.forms[int(rows)] = {form: values.pop() for form, values in forms.items()}
    plans = {}
    for e in args.plan:
        found = re.fullmatch(r'([0-9]+)=([a-z]+)(?:/([0-9]+))?(@sliding)?', e)
        if not found or found.group(2) not in PLANS or found.group(3) and (found.group(2) != 'tile' or int(found.group(3)) < 1):
            raise SystemExit(f'--plan {e}: ROWS=PLAN[/PARTS][@sliding], PLAN one of {PLANS}, PARTS a positive count of the tile plan')
        plans[int(found.group(1))] = (found.group(2), int(found.group(3) or 0), bool(found.group(4)))
    held = slices(streamed, record or None, args.rank)
    for draft in args.draft:
        held += draft_slices(Streamed(draft, 'cpu'), record or None, args.rank)
    keyed, runs = products(held)
    fused = fused_classes(fit.fused, [c for c in classes if c <= 16])
    entries = []
    for (op, outputs, columns, widths, tables), units in sorted(keyed.items(), key=lambda kv: kv[0]):
        pair = op == 'couple' and (mean_units([u[0] for u in units]), mean_units([u[1] for u in units]))
        for rows in classes:
            spec, t = (couple(fit, *pair, rows, fit.read.get((outputs, columns, widths, rows)), widths) if pair
                       else choose(fit, mean_units(units), rows, op, tables, fit.read.get((outputs, columns, widths, rows)), widths))
            if spec is None or spec['kernel'] == 'coop' and coop_slots() is None:
                continue
            kernel, arguments = spec_arguments(spec, columns // 32, widths)
            grid, threads, shares = launch(spec, outputs, rows)
            shape = f'{outputs}x{columns} {widths}'
            stacks = runs.get((outputs, columns, widths), {}) if op == 'product' else {}
            entries.append({'op': op, 'outputs': outputs, 'columns': columns, 'widths': widths, 'rows': rows, 'kernel': kernel,
                            'args': list(arguments), 'names': list(ARGUMENTS[kernel]), 'grid': list(grid), 'threads': threads,
                            'shares': shares, 'lut': int(spec.get('LUT', 0)), 'us': round(t, 2),
                            'finish': kernel == 'alu' and shares == 1 and rows <= POLICY['finish_rows'] and rows not in args.apart and rows in fused.get('finish', {}).get(shape, [rows]),
                            'epilogues': sorted(k for k, held_ in stacks.items() if all(rows in fused.get(k, {}).get(f'{shape} {s}', []) for s in held_))})
    ctx = args.ctx or record.get('ctx_mean') or 1024
    part = record['ranks'][args.rank] if args.program else None
    attention = _attention(args.export, part, ctx, classes, fit.attention, fit.attention_read, plans)
    for rows, forms in sorted(fit.forms.items()):
        for form, value in forms.items():
            print(f'--explore {rows}: {form}{value} in {sum(e["rows"] == rows and e["args"][e["names"].index(form)] == value for e in entries if form in e["names"])} '
                  f'of the class\'s {sum(e["rows"] == rows for e in entries)} instances')
    table = {'node': node()[0], 'cores': node()[1], 'evidence': str(evidence_path()), 'export': str(args.export),
             'program': args.program, 'rank': args.rank, 'rows': classes, 'width_class': WIDTH_CLASS, 'ctx': ctx,
             'instances': entries, 'attention': attention}
    _write(args.out, table)
    print(f'{args.out}: {len(entries)} instances, {len(keyed)} slices, {len(attention)} attention plans, rows {classes}')


def main():
    parser = argparse.ArgumentParser(prog='python -m torch_mesh.streamed')
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('fit')
    fit.add_argument('observations', nargs='+')
    resolve = sub.add_parser('resolve')
    resolve.add_argument('export')
    resolve.add_argument('--program', default=None)
    resolve.add_argument('--rank', type=int, default=None)
    resolve.add_argument('--rows', default=None, help="the row counts the engine runs (the served engine's batch buckets); without it, the record's (call_rows)")
    resolve.add_argument('--ctx', type=int, default=0, help="the positions a global layer's attention is priced at (default: the record's ctx_mean, else 1024)")
    resolve.add_argument('--draft', action='append', default=[],
                         help="a drafter's export whose matrices the member runs (metal-microbench tools/draft/dflash2.py export): whole, "
                              "or a product the record's speculate gives the --rank a run of, as views on its 'draft <kind>' lattice (draft_slices)")
    resolve.add_argument('--explore', action='append', default=[],
                         help="a rows class whose slices take the fit's argmin over the named configurations alone, ROWS=KIND/ARGS[/FORM<v>],..., "
                              "or ROWS=FORM<v> alone: a form named (FORMS: H<h>, RING1, LOADS1; one value a form) taken by every slice of the class "
                              "that admits it, in place of the slice's least read instance where no configurations are named")
    resolve.add_argument('--apart', action='append', type=int, default=[],
                         help="a rows class whose products finish apart (their partials and streamed_finish), read once against the finish "
                              "in the product (epilogue FUSED SEPARATE)")
    resolve.add_argument('--plan', action='append', default=[],
                         help="a rows class whose layers take the named decode attention plan, ROWS=PLAN[/PARTS][@sliding] (vector or tile; "
                              "PARTS the tile's part count, else the engine's; @sliding the sliding-window layers alone), read once")
    resolve.add_argument('--out', required=True)
    observe = sub.add_parser('observe')
    observe.add_argument('reading')
    observe.add_argument('--loads', type=int, default=1, help="the LOADS of labels that do not name it (0 before mesh 290449d)")
    observe.add_argument('export')
    observe.add_argument('--program', default=None)
    observe.add_argument('--rank', type=int, default=None)
    observe.add_argument('--ctx', type=int, default=1024, help="the positions a global layer's attention read")
    observe.add_argument('--out', required=True)
    epilogue = sub.add_parser('epilogue')
    epilogue.add_argument('fused')
    epilogue.add_argument('separate')
    epilogue.add_argument('--rows', action='append', default=[], help="an epilogue's name and the row kernel's label it replaces, NAME=LABEL")
    epilogue.add_argument('--export', required=True, help='the export the readings were served from')
    epilogue.add_argument('--program', default=None, help="the record of the call whose member served them")
    epilogue.add_argument('--rank', type=int, default=None, help="that member's rank")
    admit = sub.add_parser('admit')
    admit.add_argument('record', help="a decode KL record of the engine served with the form (metal-microbench tools/retained.py's JSON line)")
    admit.add_argument('form', help='the form, H<h>')
    admit.add_argument('bar', type=float, help='the KL bar the record must be at or below (the operator\'s)')
    args = parser.parse_args()
    {'fit': _fit, 'resolve': _resolve,
     'observe': _observe, 'epilogue': _epilogue, 'admit': _admit}[args.command](args)


if __name__ == '__main__':
    main()
