"""Kernel text for the compiler's commands (design/page-maps-2026-11-02.md; rdma/metal.h's second thing compiles it).

Every operand is a strided view: its shape, strides and element offset are literal constants of its kernel, so the
Metal compiler folds each index (a power-of-two extent's digit is a shift and a mask). A SEND source's stores and a
landing's loads are system-coherent (rdma/mesh.metal); every other access is an ordinary one. One kernel a command:
a pointwise map (a fused expression over its operands), a matrix product with its bias (product's two forms), an
indexed read or write through a device index, a sum over axes, and the run's observation of a word (a landing's or a
SEND's completion). A link's or release's word is written by a copy, not a kernel (compiler.py stored).
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


# the tensor operations' element types (MetalPerformancePrimitives MPPTensorOpsMatMul2d.h: every pair of these
# with a float destination), and their strides' bound (docs/facts.md: matmul2d reads wrong rows past the first where a
# tensor's stride is 2^16 elements or more, M5 Max 2026-10-07)
TENSOR_TYPES = {torch.float16: 'half', torch.float32: 'float', torch.bfloat16: 'bfloat'}
TENSOR_STRIDE_BOUND = 1 << 16
# a simdgroup's output block (docs/kernels.md: a simdgroup accumulates a 32 x 32 block, the dense kernel 0.82 of MPS
# on the M5 Max): 32 columns, and 32, 16 or 8 rows, whichever the product's rows fill
TENSOR_COLUMNS = 32
# the simdgroups a product keeps in flight (k split among a threadgroup's simdgroups where its blocks are fewer): the
# FFN's 64 x 2048 x 1127 as 72 one-simdgroup blocks ran 172 us a layer on an M5 Ultra, 3.4 TFLOP/s, its 9.2 MB of
# weights a layer at 54 GB/s (docs/facts.md), each simdgroup's loads one at a time along all of k
TENSOR_SIMDGROUPS = 1024
TENSOR_SPLIT_MOST = 8          # a threadgroup's simdgroups over k
TENSOR_ROWS_MOST = 16          # a block's rows at most (32, 16 or 8): 16 the FFN sweep's best (docs/facts.md, ffn run-23)
TENSOR_SPLIT_LEAST_K = 128     # the least k a simdgroup takes
THREADGROUP_BYTES = 32768      # a threadgroup's memory (Apple GPUs' maxThreadgroupMemoryLength)


def _tensor_operand(view):
    """whether a product reads `view` through a tensor: an ordinary (not system-coherent) load of a tensor element
    type, every stride non-negative and under the bound"""
    return (not view.coherent and view.dtype in TENSOR_TYPES
            and all(0 <= s < TENSOR_STRIDE_BOUND for s, n in zip(view.stride, view.shape) if n != 1))


def _tensor(view, pointer, rows, columns, offset='0', row_axis=0, column_axis=1):
    """a tensor over `view`'s elements from element `offset` (an expression) on: `columns` its x extent (stride along
    column_axis), `rows` its y (MPP's order: the fastest index first)"""
    stride = lambda axis: view.stride[axis] if view.shape[axis] != 1 else 0
    return (f'tensor(const_cast<device {TENSOR_TYPES[view.dtype]} *>({pointer}) + {view.offset}l + long({offset}), '
            f'dextents<int, 2>({columns}, {rows}), array<int, 2>{{{stride(column_axis)}, {stride(row_axis)}}})')


def product(name, out, a, b, bias=None, alpha=1.0, beta=1.0):
    """out[m, n] = beta bias[n] + alpha sum_k a[m, k] b[k, n], each a View of rank 2 (bias of rank 1 or broadcast); `a` a
    View or a list of Views, the parts of a concatenation along k (a cat the product reads in place, compiler.py), part p
    over b's rows from the sum of the parts before it.

    Two forms, chosen by the operands at compile time. Where every operand is a tensor's (_tensor_operand) and the rows
    are a multiple of 8 (Metal 4 tensor operations need it on the M4 Pro: docs/kernels.md), the M5's neural
    accelerators: MetalPerformancePrimitives matmul2d, a simdgroup's 32 x 32 (or 16, or 8 rows) block with float
    accumulation, the form mesh 1f794b8f^ matrix.metal had, which that commit replaced with the other. Where the blocks
    are fewer than TENSOR_SIMDGROUPS, k is split among a threadgroup's simdgroups, each over its range (across the parts
    it meets), their partials summed through threadgroup memory into the first, whose registers hold the epilogue
    (alpha, bias, the store). Otherwise (a landing's system-coherent loads, which a tensor does not express; fewer
    than 8 rows): one simdgroup an output, its lanes over k, summed with simd_sum (tools/mesh/lstm_metal.m's shape)."""
    M, N = out.shape
    parts = a if isinstance(a, list) else [a]
    parameters = [out.parameter(True)] + [p.parameter(False) for p in parts] + [b.parameter(False)] + \
        ([bias.parameter(False)] if bias else [])
    bias_term = f' + {beta}f * float({bias.load(["n"] if len(bias.shape) == 1 else ["m", "n"])})' if bias else ''
    if M % 8 == 0 and all(_tensor_operand(v) for v in parts + [b]):
        rows = next(r for r in (32, 16, 8) if r <= TENSOR_ROWS_MOST and M % r == 0)
        K = sum(p.shape[1] for p in parts)
        blocks = -(-N // TENSOR_COLUMNS) * (M // rows)
        block = rows * TENSOR_COLUMNS
        split = 1
        while (split * 2 <= TENSOR_SPLIT_MOST and blocks * split < TENSOR_SIMDGROUPS
               and K // (split * 2) >= TENSOR_SPLIT_LEAST_K and (split * 2 - 1) * block * 4 <= THREADGROUP_BYTES):
            split *= 2
        span = -(-K // (split * 32)) * 32
        # a product whose blocks all lie inside it reads them unchecked (static slice extents: MPPTensorOpsMatMul2d.h, the edge
        # check is the dynamic slice's); otherwise every block is checked against the operands' extents
        full = M % rows == 0 and N % TENSOR_COLUMNS == 0
        left_slice = f'slice<dynamic_extent, {rows}>(0, row)' if full else 'slice(0, row)'
        right_slice = f'slice<{TENSOR_COLUMNS}, dynamic_extent>(col, 0)' if full else 'slice(col, 0)'
        runs, first = [], 0
        for k, p in enumerate(parts):
            lo, hi = f'max(k0, {first})', f'min(k1, {first + p.shape[1]})'
            runs.append(f'    if ({lo} < {hi}) {{\n        const int lo = {lo}, hi = {hi};\n'
                        f'        auto left = {_tensor(p, f"a{p.index}", M, "hi - lo", f"(lo - {first}) * {p.stride[1]}")}.{left_slice};\n'
                        f'        auto right = {_tensor(b, f"a{b.index}", "hi - lo", N, f"lo * {b.stride[0]}")}.{right_slice};\n'
                        f'        op.run(left, right, total);\n    }}')
            first += p.shape[1]
        p0 = parts[0]
        reduce = '' if split == 1 else f'''
    threadgroup float partials[{(split - 1) * block}];
    if (s) {{
        #pragma unroll
        for (uint i = 0; i < total.get_capacity(); i++) if (total.is_valid_element(i)) {{
            const auto e = total.get_multidimensional_index(i);
            partials[(s - 1) * {block} + e[1] * {TENSOR_COLUMNS} + e[0]] = total[i];
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s) return;
    #pragma unroll
    for (uint i = 0; i < total.get_capacity(); i++) if (total.is_valid_element(i)) {{
        const auto e = total.get_multidimensional_index(i);
        for (uint t = 0; t < {split - 1}u; t++) total[i] += partials[t * {block} + e[1] * {TENSOR_COLUMNS} + e[0]];
    }}'''
        body = f'''    const uint s = (at.x / 32u) % {split}u;
    const int col = int(at.x / {32 * split}u) * {TENSOR_COLUMNS}, row = int(at.y) * {rows};
    const int k0 = int(s) * {span}, k1 = min(k0 + {span}, {K});
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor({rows}, {TENSOR_COLUMNS}, static_cast<int>(dynamic_extent),
        false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> op;
    using Left = decltype({_tensor(p0, f"a{p0.index}", M, p0.shape[1])}.{left_slice});
    using Right = decltype({_tensor(b, f"a{b.index}", K, N)}.{right_slice});
    auto total = op.get_destination_cooperative_tensor<Left, Right, float>();
    #pragma unroll
    for (uint i = 0; i < total.get_capacity(); i++) total[i] = 0.0f;
{chr(10).join(runs)}{reduce}
    #pragma unroll
    for (uint i = 0; i < total.get_capacity(); i++) if (total.is_valid_element(i)) {{
        const auto e = total.get_multidimensional_index(i);
        const ulong m = ulong(row + e[1]), n = ulong(col + e[0]);
        if (m < {M}ul && n < {N}ul) {{ {out.store(["m", "n"], f"{alpha}f * total[i]{bias_term}")} }}
    }}'''
        source = ('#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n'
                  + _source(name, parameters, body))
        return source, (-(-N // TENSOR_COLUMNS) * 32 * split, M // rows, 1), (32 * split, 1, 1)
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


def summed(name, out, source, axes, scale=1.0):
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
            f'    {" ".join(loops)} total += float({source.load(coordinates)});\n    {out.store(c, f"total * {scale}f" if scale != 1.0 else "total")}')
    parameters = [out.parameter(True), source.parameter(False)]
    return _source(name, parameters, body), (count, 1, 1), (min(256, max(1, count)), 1, 1)


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
