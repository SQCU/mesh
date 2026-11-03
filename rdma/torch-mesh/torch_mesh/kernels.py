"""Kernel text for the compiler's commands (design/page-maps-2026-11-02.md; rdma/metal.h's second thing compiles it).

Every operand is a strided view: its shape, strides and element offset are literal constants of its kernel, so the
Metal compiler folds each index (a power-of-two extent's digit is a shift and a mask). A SEND source's stores and a
landing's loads are system-coherent (rdma/mesh.metal); every other access is an ordinary one. One kernel a command:
a pointwise map (a fused expression over its operands), a matrix product with its bias (one simdgroup an output, as
tools/mesh/lstm_metal.m's floor kernel), an indexed read or write through a device index, a sum over axes, and the
run's store of a word (a link's or a release's: mesh_publish, a system fence and a maximum).
"""
import functools
import pathlib

import torch

MSL = {torch.float16: 'half', torch.float32: 'float', torch.bfloat16: 'bfloat', torch.int64: 'long', torch.int32: 'int',
       torch.int16: 'short', torch.int8: 'char', torch.uint8: 'uchar', torch.bool: 'bool'}


@functools.cache
def prelude():
    """rdma/mesh.metal's text, beside the mesh module the binding loads (rdma/mesh.py)"""
    import mesh
    return (pathlib.Path(mesh.__file__).resolve().parent / 'mesh.metal').read_text()


class View:
    """an operand of a kernel: its argument index, element type, shape, strides and element offset (torch's), and
    whether its bytes are a SEND's source (stored coherently) or a landing's (loaded coherently)"""
    def __init__(self, index, dtype, shape, stride, offset, coherent=False):
        self.index, self.dtype, self.shape, self.stride, self.offset = index, dtype, tuple(shape), tuple(stride), offset
        self.coherent = coherent

    def parameter(self, writes):
        qualifier = 'volatile coherent(system) device' if self.coherent else 'device'
        const = '' if writes else ' const'
        return f'{qualifier}{const} {MSL[self.dtype]} *a{self.index} [[buffer({self.index})]]'

    def at(self, coordinates):
        """the element index of this view at `coordinates` (one expression an axis, broadcast axes left out)"""
        terms = [f'{self.offset}l'] + [f'({c}) * {s}l' for c, s, n in zip(coordinates, self.stride, self.shape)
                                        if s and n != 1]
        return ' + '.join(terms)

    def load(self, coordinates):
        place = f'a{self.index} + ({self.at(coordinates)})'
        return f'mesh_payload({place})' if self.coherent else f'a{self.index}[{self.at(coordinates)}]'

    def store(self, coordinates, value):
        place = f'a{self.index} + ({self.at(coordinates)})'
        if self.coherent:
            return f'mesh_payload_store({place}, {MSL[self.dtype]}({value}));'
        return f'a{self.index}[{self.at(coordinates)}] = {MSL[self.dtype]}({value});'


def _digits(shape, linear='i'):
    """the coordinates of linear index `linear` over `shape`, row-major: one expression an axis, constants folded"""
    out, stride = [], 1
    for extent in reversed(shape):
        out.append(f'(({linear} / {stride}ul) % {extent}ul)' if extent != 1 else '0ul')
        stride *= extent
    return list(reversed(out))


def _source(name, parameters, body):
    return f'{prelude()}\nusing namespace metal;\nkernel void {name}({", ".join(parameters)}, uint3 at [[thread_position_in_grid]]) {{\n{body}\n}}\n'


def pointwise(name, out, operands, expression, extra=()):
    """out[c] = expression over the operands loaded at c (each a View broadcast over out's shape: `x{k}` names operand
    k's value as float); `extra` the parameters of an indexed operand (index views). One thread an element."""
    count = 1
    for extent in out.shape:
        count *= extent
    c = _digits(out.shape)
    loads = '\n'.join(f'    const float x{k} = float({view.load(c)});' for k, view in enumerate(operands))
    body = (f'    const ulong i = at.x;\n    if (i >= {count}ul) return;\n{loads}\n'
            f'    {out.store(c, expression)}')
    parameters = [out.parameter(True)] + [view.parameter(False) for view in operands] + list(extra)
    return _source(name, parameters, body), (count, 1, 1), (min(256, max(1, count)), 1, 1)


def indexed(name, out, source, index, axis, writes_through_index):
    """an indexed copy along `axis`: reading (index_select) out[c] = source[c with c[axis] = index[c[axis]]], or writing
    (index_copy) out[c with c[axis] = index[c[axis]]] = source[c], over the source's or out's coordinates"""
    over = source if writes_through_index else out
    count = 1
    for extent in over.shape:
        count *= extent
    c = _digits(over.shape)
    chosen = list(c)
    chosen[axis] = f'ulong(a{index.index}[{index.at([c[axis]])}])'
    if writes_through_index:
        body = f'    {out.store(chosen, source.load(c))}'
    else:
        body = f'    {out.store(c, source.load(chosen))}'
    body = f'    const ulong i = at.x;\n    if (i >= {count}ul) return;\n{body}'
    parameters = [out.parameter(True), source.parameter(False), index.parameter(False)]
    return _source(name, parameters, body), (count, 1, 1), (min(256, max(1, count)), 1, 1)


def product(name, out, a, b, bias=None, alpha=1.0, beta=1.0):
    """out[m, n] = beta bias[n] + alpha sum_k a[m, k] b[k, n], each a View of rank 2 (bias of rank 1 or broadcast); `a` a
    View or a list of Views, the parts of a concatenation along k (a cat the product reads in place, compiler.py), part p
    over b's rows from the sum of the parts before it; one simdgroup an output, its lanes over k, summed with simd_sum
    (tools/mesh/lstm_metal.m's shape)"""
    M, N = out.shape
    parts = a if isinstance(a, list) else [a]
    parameters = [out.parameter(True)] + [p.parameter(False) for p in parts] + [b.parameter(False)] + \
        ([bias.parameter(False)] if bias else [])
    bias_term = f' + {beta}f * float({bias.load(["n"] if len(bias.shape) == 1 else ["m", "n"])})' if bias else ''
    loops, first = [], 0
    for p in parts:
        K = p.shape[1]
        loops.append(f'    for (ulong k = lane; k < {K}ul; k += 32ul) total += float({p.load(["m", "k"])}) * '
                     f'float({b.load([f"(k + {first}ul)", "n"])});')
        first += K
    body = f'''    const ulong n = at.x / 32ul, m = at.y, lane = at.x % 32ul;
    if (n >= {N}ul || m >= {M}ul) return;
    float total = 0.0f;
{chr(10).join(loops)}
    total = simd_sum(total);
    if (lane == 0ul) {{ {out.store(["m", "n"], f"{alpha}f * total{bias_term}")} }}'''
    return _source(name, parameters, body), (N * 32, M, 1), (32, 1, 1)


def summed(name, out, source, axes):
    """out[c] = the sum of source over `axes` at the other coordinates c (source's axes not in `axes` are out's, in
    order); one thread an output element"""
    kept = [d for d in range(len(source.shape)) if d not in axes]
    count = 1
    for extent in out.shape:
        count *= extent
    c = _digits(out.shape)
    loops, coordinates = [], [None] * len(source.shape)
    for k, d in enumerate(kept):
        coordinates[d] = c[k]
    for d in axes:
        coordinates[d] = f'r{d}'
        loops.append(f'for (ulong r{d} = 0; r{d} < {source.shape[d]}ul; r{d}++)')
    body = (f'    const ulong i = at.x;\n    if (i >= {count}ul) return;\n    float total = 0.0f;\n'
            f'    {" ".join(loops)} total += float({source.load(coordinates)});\n    {out.store(c, "total")}')
    parameters = [out.parameter(True), source.parameter(False)]
    return _source(name, parameters, body), (count, 1, 1), (min(256, max(1, count)), 1, 1)


STORE = 'mesh_store'


@functools.cache
def store():
    """a word's store: argument 0 the word, argument 1 its value (a link's table offset or a release's count), stored
    after a system fence as a maximum (mesh_publish), so a count never moves backwards"""
    return (f'{prelude()}\nusing namespace metal;\nkernel void {STORE}(volatile coherent(system) device ulong *word '
            f'[[buffer(0)]], device const ulong *value [[buffer(1)]]) {{ mesh_publish(word, *value); }}\n')


OBSERVE = 'mesh_observe'
# an observation's bound: at most this many fenced reads (well under a second on the members), and none once the run's
# defect word is set, so a word that never comes ends the run and names itself, never a GPU held
OBSERVE_POLLS = 1 << 22


@functools.cache
def observe():
    """a landing's or a SEND completion's presence on the GPU: argument 0 the channel's `landed` or `sent` word, argument
    1 the count it must reach (in the run's value slots), argument 2 the run's defect word and the word read at a
    give-up beside it. Read through volatile coherent(system) loads, each after a system-scope fence (mesh 1f794b8f^
    mesh_wait_bounded's form: a part already running sees a CPU's store in 1-4 us, metal-microbench docs/facts.md,
    where a CPU-signalled start costs an M5 Ultra 100-180 us); present where the word reaches the count, lost where it
    reads ~0 (a lost port's words, as the bridge stores them). A give-up at OBSERVE_POLLS, or a lost word, sets the
    defect word to the observation's slot x 4 + 1 (a give-up) or + 2 (a lost word) if none is set, and an observation
    of a run with a defect set ends at once"""
    return (f'{prelude()}\nusing namespace metal;\nkernel void {OBSERVE}(volatile coherent(system) device const ulong *word '
            f'[[buffer(0)]], device const ulong *value [[buffer(1)]], volatile coherent(system) device ulong *defect '
            f'[[buffer(2)]], uint at [[thread_position_in_grid]]) {{\n'
            f'    if (defect[0]) return;\n'
            f'    const ulong want = value[0], slot = value[1];\n    ulong seen;\n    uint n = 0u;\n'
            f'    do {{\n        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, '
            f'static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE));\n        seen = *word;\n'
            f'    }} while (seen < want && ++n < {OBSERVE_POLLS}u && ((n & 4095u) || !defect[0]));\n'
            f'    if (seen < want || seen == ~0ul) {{ if (!defect[0]) {{ defect[1] = seen; defect[0] = slot * 4ul + (seen == ~0ul ? 2ul : 1ul); }} }}\n}}\n')
