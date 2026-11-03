"""Kernel text for the compiler's commands (design/page-maps-2026-11-02.md; rdma/metal.h's second thing compiles it).

Every operand is a strided view: its shape, strides and element offset are literal constants of its kernel, so the
Metal compiler folds each index (a power-of-two extent's digit is a shift and a mask). A SEND source's stores and a
landing's loads are system-coherent (rdma/mesh.metal); every other access is an ordinary one. One kernel a command:
a pointwise map (a fused expression over its operands), a matrix product with its bias (product's two forms), an
indexed read or write through a device index, a sum over axes, and the run's observation of a word (a landing's or a
SEND's completion), and a link's or release's word store (compiler.py stored).
"""
import functools
import itertools
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
    def __init__(self, index, dtype, shape, stride, offset, coherent=False, tg=None):
        self.index, self.dtype, self.shape, self.stride, self.offset = index, dtype, tuple(shape), tuple(stride), offset
        # tg: the view is of no buffer but of products' outputs its kernel computes (fused): `at` names the element
        self.coherent, self.tg = coherent, tg

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
        if self.tg:
            return f'{self.tg}[{self.at(coordinates)}]'
        place = f'a{self.index} + ({self.at(coordinates)})'
        return f'mesh_payload({place})' if self.coherent else f'a{self.index}[{self.at(coordinates)}]'

    def store(self, coordinates, value):
        if self.tg:
            return f'{self.tg}[{self.at(coordinates)}] = {MSL[self.dtype]}({value});'
        place = f'a{self.index} + ({self.at(coordinates)})'
        if self.coherent:
            return f'mesh_payload_store({place}, {MSL[self.dtype]}({value}));'
        return f'a{self.index}[{self.at(coordinates)}] = {MSL[self.dtype]}({value});'


def _digits(shape, linear='i'):
    """the coordinates of linear index `linear` over `shape`, row-major: one expression an axis, constants folded. In
    32 bits where the shape's elements fit (each a division by a constant then a remainder, which the GPU compiler
    turns into a multiply and shift in 32 bits): the 64-bit forms cost the M4 Pro 5.9 us of the LSTM's one-dispatch
    tick (11.95 against 6.06) and an M5 Ultra 0.56 (metal-microbench tools/mesh/tick.m grouped64, docs/facts.md)"""
    out, stride, total = [], 1, 1
    for extent in shape:
        total *= extent
    wide = total >= 1 << 32
    for extent in reversed(shape):
        if extent == 1:
            out.append('0ul')
        elif wide:
            out.append(f'(({linear} / {stride}ul) % {extent}ul)')
        else:
            out.append(f'((uint({linear}) / {stride}u) % {extent}u)')
        stride *= extent
    return list(reversed(out))


def _source(name, parameters, body):
    return f'{prelude()}\nusing namespace metal;\nkernel void {name}({", ".join(parameters)}, uint3 at [[thread_position_in_grid]]) {{\n{body}\n}}\n'


def pointwise(name, out, operands, expression, extra=(), shared=()):
    """out[c] = expression over the operands loaded at c (each a View broadcast over out's shape: `x{k}` names operand
    k's value as float); `extra` the parameters of an indexed operand (index views). `shared`: (view, expression, name)
    each, a value computed once (as float, rounded through its view's type as its store is), stored to its view at c and
    named in the expressions after it: a write the next kernel would read back (compiler.py _emit). One thread an
    element."""
    count = 1
    for extent in out.shape:
        count *= extent
    c = _digits(out.shape)
    loads = '\n'.join(f'    const float x{k} = float({view.load(c)});' for k, view in enumerate(operands))
    pre = ''.join(f'\n    const float {named} = float({MSL[view.dtype]}({text}));\n    {view.store(c, named)}'
                  for view, text, named in shared)
    body = (f'    const ulong i = at.x;\n    if (i >= {count}ul) return;\n{loads}{pre}\n'
            f'    {out.store(c, expression)}')
    parameters = [out.parameter(True)] + [view.parameter(True) for view, _, _ in shared] + \
        [view.parameter(False) for view in operands] + list(extra)
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
    type, every stride non-negative and under the bound, its fastest axis (the last) of unit stride: the training
    LSTM's products of transposed views (linear's W, linear_backward's grad^T) came out wrong through matmul2d by 0.012
    (metal-microbench lstm-train-2026-11-03 sweep-1-3)"""
    return (not view.coherent and view.dtype in TENSOR_TYPES
            and all(0 <= s < TENSOR_STRIDE_BOUND for s, n in zip(view.stride, view.shape) if n != 1)
            and (view.shape[-1] == 1 or view.stride[-1] == 1))


def _tensor(view, pointer, rows, columns, offset='0', row_axis=0, column_axis=1):
    """a tensor over `view`'s elements from element `offset` (an expression) on: `columns` its x extent (stride along
    column_axis), `rows` its y (MPP's order: the fastest index first)"""
    stride = lambda axis: view.stride[axis] if view.shape[axis] != 1 else 0
    return (f'tensor(const_cast<device {TENSOR_TYPES[view.dtype]} *>({pointer}) + {view.offset}l + long({offset}), '
            f'dextents<int, 2>({columns}, {rows}), array<int, 2>{{{stride(column_axis)}, {stride(row_axis)}}})')


STAGED_COLUMNS = 4  # the simdgroup form's columns a threadgroup, over one staged row of A


ACCUMULATE_THREADGROUPS = 176  # an accumulating product's threadgroups at least, k split over them (gemm.m's best: 4 ways)
ACCUMULATE_LEAST_K = 256        # the least k a split takes


def accumulates(M, parts, b):
    """whether a product of these operands (Views) takes the accumulating form (product's `accumulate`): 64-row tiles of
    tensor operands"""
    transposed = (not b.coherent and b.dtype in TENSOR_TYPES and b.shape[0] != 1 and b.stride[0] == 1 and b.shape[1] != 1
                  and 0 < b.stride[1] < TENSOR_STRIDE_BOUND)
    return M % 64 == 0 and all(_tensor_operand(v) for v in parts) and (transposed or _tensor_operand(b))


def product(name, out, a, b, bias=None, alpha=1.0, beta=1.0, accumulate=False):
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
    than 8 rows): a threadgroup a row's STAGED_COLUMNS columns over the row of A staged in threadgroup memory, a
    simdgroup a column, its lanes over k, summed with simd_sum (tools/mesh/lstm_metal.m's shape)."""
    M, N = out.shape
    parts = a if isinstance(a, list) else [a]
    parameters = [out.parameter(True)] + [p.parameter(False) for p in parts] + [b.parameter(False)] + \
        ([bias.parameter(False)] if bias else [])
    bias_term = f' + {beta}f * float({bias.load(["n"] if len(bias.shape) == 1 else ["m", "n"])})' if bias else ''
    # B read as the transpose of its own storage where its K axis is the fast one (linear's W: [K, N] with strides (1, K)):
    # matmul2d's transpose_right, its tensor W itself ([N, K] row-major, height N and width K: MPPTensorOpsMatMul2dImpl.h),
    # where a strided tensor came out wrong (_tensor_operand)
    transposed = (not b.coherent and b.dtype in TENSOR_TYPES and b.shape[0] != 1 and b.stride[0] == 1 and b.shape[1] != 1
                  and 0 < b.stride[1] < TENSOR_STRIDE_BOUND)
    if accumulate:
        # a threadgroup of 4 simdgroups a 64 x 32 tile (matmul2d at threadgroup scope, execution_simdgroups<4>), k split
        # over threadgroups (grid z), each split's partial added to out (float, zeroed before: compiler.py _product)
        # atomically. The FFN's products on an M5 Ultra: 11.1 and 10.6 us (521 and 542 GB/s of weights) against 28.7
        # and 20.7 for the whole k a tile, and 39-48 for kernels.product's 16-row form with k split in its threadgroup
        # (metal-microbench tools/mesh/gemm.m tg64x32k4)
        rows, K = 64, sum(p.shape[1] for p in parts)
        tiles = -(-N // TENSOR_COLUMNS) * (M // rows)
        split = 1
        while split < 8 and tiles * split < ACCUMULATE_THREADGROUPS and K // (split * 2) >= ACCUMULATE_LEAST_K:
            split *= 2
        span = -(-K // split)
        full = N % TENSOR_COLUMNS == 0
        left_slice = f'slice<dynamic_extent, {rows}>(0, row)'
        if transposed:
            right_slice = f'slice<dynamic_extent, {TENSOR_COLUMNS}>(0, col)' if full else 'slice(0, col)'
            right_of = lambda k_extent, offset='0': _tensor(b, f"a{b.index}", N, k_extent, offset, row_axis=1, column_axis=0)
        else:
            right_slice = f'slice<{TENSOR_COLUMNS}, dynamic_extent>(col, 0)' if full else 'slice(col, 0)'
            right_of = lambda k_extent, offset='0': _tensor(b, f"a{b.index}", k_extent, N, offset)
        runs, first = [], 0
        for p in parts:
            lo, hi = f'max(k0, {first})', f'min(k1, {first + p.shape[1]})'
            runs.append(f'    if ({lo} < {hi}) {{\n        const int lo = {lo}, hi = {hi};\n'
                        f'        auto left = {_tensor(p, f"a{p.index}", M, "hi - lo", f"(lo - {first}) * {p.stride[1]}")}.{left_slice};\n'
                        f'        auto right = {right_of("hi - lo", f"lo * {b.stride[0]}")}.{right_slice};\n'
                        f'        op.run(left, right, total);\n    }}')
            first += p.shape[1]
        p0 = parts[0]
        bias_add = f' + (s == 0u ? {beta}f * float({bias.load(["n"] if len(bias.shape) == 1 else ["m", "n"])}) : 0.0f)' if bias else ''
        body = f'''    const uint s = at.z;
    const int col = int(at.x / 128u) * {TENSOR_COLUMNS}, row = int(at.y) * {rows};
    const int k0 = int(s) * {span}, k1 = min(k0 + {span}, {K});
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor({rows}, {TENSOR_COLUMNS}, static_cast<int>(dynamic_extent),
        false, {'true' if transposed else 'false'}, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroups<4>> op;
    using Left = decltype({_tensor(p0, f"a{p0.index}", M, p0.shape[1])}.{left_slice});
    using Right = decltype({right_of(K)}.{right_slice});
    auto total = op.get_destination_cooperative_tensor<Left, Right, float>();
    #pragma unroll
    for (uint i = 0; i < total.get_capacity(); i++) total[i] = 0.0f;
{chr(10).join(runs)}
    #pragma unroll
    for (uint i = 0; i < total.get_capacity(); i++) if (total.is_valid_element(i)) {{
        const auto e = total.get_multidimensional_index(i);
        const ulong m = ulong(row + e[1]), n = ulong(col + e[0]);
        if (m < {M}ul && n < {N}ul)
            atomic_fetch_add_explicit((device atomic_float *)(a{out.index} + {out.at(["m", "n"])}), {alpha}f * total[i]{bias_add},
                                      memory_order_relaxed);
    }}'''
        source = ('#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n'
                  + _source(name, parameters, body))
        return source, (-(-N // TENSOR_COLUMNS) * 128, M // rows, split), (128, 1, 1)
    if M % 8 == 0 and all(_tensor_operand(v) for v in parts) and (transposed or _tensor_operand(b)):
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
        if transposed:
            right_slice = f'slice<dynamic_extent, {TENSOR_COLUMNS}>(0, col)' if full else 'slice(0, col)'
            right_of = lambda k_extent, offset='0': _tensor(b, f"a{b.index}", N, k_extent, offset, row_axis=1, column_axis=0)
        else:
            right_slice = f'slice<{TENSOR_COLUMNS}, dynamic_extent>(col, 0)' if full else 'slice(col, 0)'
            right_of = lambda k_extent, offset='0': _tensor(b, f"a{b.index}", k_extent, N, offset)
        runs, first = [], 0
        for k, p in enumerate(parts):
            lo, hi = f'max(k0, {first})', f'min(k1, {first + p.shape[1]})'
            runs.append(f'    if ({lo} < {hi}) {{\n        const int lo = {lo}, hi = {hi};\n'
                        f'        auto left = {_tensor(p, f"a{p.index}", M, "hi - lo", f"(lo - {first}) * {p.stride[1]}")}.{left_slice};\n'
                        f'        auto right = {right_of("hi - lo", f"lo * {b.stride[0]}")}.{right_slice};\n'
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
        false, {'true' if transposed else 'false'}, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> op;
    using Left = decltype({_tensor(p0, f"a{p0.index}", M, p0.shape[1])}.{left_slice});
    using Right = decltype({right_of(K)}.{right_slice});
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
    # a threadgroup a row and STAGED_COLUMNS columns: its threads load the row's A (every part, a landing's
    # system-coherently) into threadgroup memory once, then a simdgroup a column over it. A simdgroup an output loading
    # its own row coherently cost the LSTM's product 7.3 us a dispatch against 2.3 with ordinary loads on an M5 Ultra
    # (13.0 against 7.2 on the Mini; metal-microbench tools/mesh/chain.m dot_coherent, dot_plain)
    S = min(STAGED_COLUMNS, N)
    staged, first = [], 0
    for p in parts:
        K = p.shape[1]
        staged.append(f'    for (ulong k = me; k < {K}ul; k += {32 * S}ul) row[{first}ul + k] = float({p.load(["m", "k"])});')
        first += K
    body = f'''    const ulong m = at.y, me = at.x % {32 * S}ul, n = at.x / {32 * S}ul * {S}ul + me / 32ul, lane = me % 32ul;
    threadgroup float row[{first}];
{chr(10).join(staged)}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (n >= {N}ul) return;
    float total = 0.0f;
    for (ulong k = lane; k < {first}ul; k += 32ul) total += row[k] * float({b.load(["k", "n"])});
    total = simd_sum(total);
    if (lane == 0ul) {{ {out.store(["m", "n"], f"{alpha}f * total{bias_term}")} }}'''
    return _source(name, parameters, body), (-(-N // S) * 32 * S, M, 1), (32 * S, 1, 1)


def fused(name, products, out, operands, expression, shared):
    """products whose outputs one pointwise consumer alone reads, each output once, and that consumer, in one kernel
    (compiler.py _fuse): one threadgroup a consumer element, one simdgroup each product output its loads of the
    products' root name (a View with tg set: the root's layout), as product's simdgroup form does (its lanes over k,
    simd_sum), then its first thread the pointwise (out, operands, expression and shared as pointwise's). products: (out,
    parts, b, bias, alpha, beta) each, out a View of rows [M, N] at its offset in the root (strides (N, 1)). The LSTM's
    tick lowered its products and cell to three dependent dispatches, 2-4 us each at a light load's clock on every
    member (metal-microbench docs/facts.md, chain.m); fused, one, every core still over the products' outputs. One
    threadgroup for all of them was right but 80-101 us a tick against 25.5 (lstm-2026-11-03 fused-3, fused-4)."""
    views = {}
    for v in [out] + [view for view, _, _ in shared]:
        views[v.index] = v.parameter(True)
    for v in operands:
        if not v.tg:
            views.setdefault(v.index, v.parameter(False))
    for _, parts, b, bias, _, _ in products:
        for v in parts + [b] + ([bias] if bias else []):
            views.setdefault(v.index, v.parameter(False))
    count = 1
    for extent in out.shape:
        count *= extent
    c = _digits(out.shape)
    # a consumer element a threadgroup, one simdgroup each product output it reads (its gates at once: four in a row
    # in one simdgroup were no faster than the parts, fused-6), the outputs met in threadgroup memory, then the
    # pointwise on the threadgroup's first thread
    computed = [k for k, view in enumerate(operands) if view.tg]
    G = max(1, len(computed))
    # its places in 32 bits (a fused root's outputs at most compiler.py FUSED_OUTPUTS): 64-bit divisions and remainders
    # made the M4 Pro's tick 26.7 us from its last landing to its release against 9.1 unfused (lstm-2026-11-03
    # grouped-1, mesh ef451e2a; tools/mesh/tick.m grouped64)
    body = [f'    const uint i = at.x / {32 * G}u, local = at.x % {32 * G}u, g = local / 32u, lane = local % 32u;',
            f'    threadgroup float outputs[{G}];']
    # Changed from 233c4355's form (removed by 395d7f0d, restored): where every consumer element's computed outputs lie
    # in one row of one product (the LSTM's cell: an element's four gates in its row of the gates' product), proved
    # here over every element, the threadgroup stages that row's A parts once (every thread, as product's staged
    # simdgroup form, 97eee528) and each simdgroup's dot reads it there: 233c4355's simdgroups each loaded the row
    # coherently themselves, four times the landing's coherent loads (34.5 us a tick against the parts' 27.4,
    # fused-7, before the parts staged theirs). tools/mesh/tick.m grouped, this form: 6.40 us a tick with its release
    # against 7.43-7.54 for the staged parts on an M5 Ultra, 6.04 against 7.5 on the M4 Pro (metal-microbench
    # docs/facts.md). What must stay: the staged row only where the proof holds; elsewhere each simdgroup's own loads.
    def element(view, coordinates):
        return view.offset + sum(x * st for x, st, n in zip(coordinates, view.stride, view.shape) if st and n != 1)

    def owner(e):
        for q, (o, *_rest) in enumerate(products):
            if o.offset <= e < o.offset + o.shape[0] * o.shape[1]:
                return q, (e - o.offset) // o.shape[1]
        return None
    staged = bool(computed) and all(
        len({owner(element(operands[k], at)) for k in computed}) == 1 and owner(element(operands[computed[0]], at)) is not None
        for at in itertools.product(*(range(n) for n in out.shape)))
    if staged:
        widest = max(sum(part.shape[1] for part in parts) for _, parts, *_rest in products)
        body += [f'    threadgroup float row[{widest}];', f'    const uint e0 = uint({operands[computed[0]].at(c)});']
        for q, (o, parts, *_rest) in enumerate(products):
            M, N = o.shape
            test = f'e0 < {o.offset + M * N}u' if q + 1 < len(products) else 'true'
            body += [f'    {"if" if q == 0 else "else if"} ({test}) {{', f'        const uint m = (e0 - {o.offset}u) / {N}u;']
            first = 0
            for part in parts:
                K = part.shape[1]
                body.append(f'        for (uint k = local; k < {K}u; k += {32 * G}u) row[k + {first}u] = float({part.load(["m", "k"])});')
                first += K
            body.append('    }')
        body.append('    threadgroup_barrier(mem_flags::mem_threadgroup);')
    for slot, k in enumerate(computed):
        # the product output this load names: its product by the root's offset, its row and column within it
        body += [f'    if (g == {slot}u) {{', f'        const uint e = uint({operands[k].at(c)});', '        float total = 0.0f;']
        for p, (o, parts, b, bias, alpha, beta) in enumerate(products):
            M, N = o.shape
            test = f'e < {o.offset + M * N}u' if p + 1 < len(products) else 'true'
            bias_term = f' + {beta}f * float({bias.load(["n"] if len(bias.shape) == 1 else ["m", "n"])})' if bias else ''
            body += [f'        {"if" if p == 0 else "else if"} ({test}) {{',
                     f'            const uint m = (e - {o.offset}u) / {N}u, n = (e - {o.offset}u) % {N}u;']
            first = 0
            for part in parts:
                K = part.shape[1]
                a = f'row[k + {first}u]' if staged else f'float({part.load(["m", "k"])})'
                body.append(f'            for (uint k = lane; k < {K}u; k += 32u) total += {a} * '
                            f'float({b.load([f"(k + {first}u)", "n"])});')
                first += K
            body += ['            total = simd_sum(total);',
                     f'            if (lane == 0u) outputs[{slot}] = {alpha}f * total{bias_term};', '        }']
        body.append('    }')
    body += ['    threadgroup_barrier(mem_flags::mem_threadgroup);', '    if (local != 0u) return;']
    for k, view in enumerate(operands):
        body.append(f'    const float x{k} = outputs[{computed.index(k)}];' if view.tg else f'    const float x{k} = float({view.load(c)});')
    for view, text, named in shared:
        body += [f'    const float {named} = float({MSL[view.dtype]}({text}));', f'    {view.store(c, named)}']
    body.append(f'    {out.store(c, expression)}')
    parameters = ', '.join(views[k] for k in sorted(views))
    return _source(name, [parameters], '\n'.join(body)), (count * 32 * G, 1, 1), (32 * G, 1, 1)


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
OBSERVE_POLLS = int(__import__("os").environ.get("MESH_OBSERVE_POLLS", 1 << 22))  # the run's configuration may set it


STORE = 'mesh_store'


@functools.cache
def store():
    """a link's or release's word: argument 0 the word, argument 1 its value (in the run's value slots), stored after a
    system-scope fence as a plain volatile coherent(system) store. Changed by the one-part session from a copy (and,
    before it, from mesh_publish's atomic maximum): with no dispatch reading the word's 64-byte sector, a running
    buffer's copy is never seen by another process within 1 s on the Mini, nor on an M5 Ultra without a barrier after
    it; mesh_publish's atomic never on the Mini, 4-169 ms on an M5 Ultra; this store 5.5-6.2 us from the buffer's start
    on the Mini and 4.6-7.0 on an M5 Ultra, its coherent payload (a SEND's source, stored coherently) complete behind
    it (metal-microbench docs/facts.md, visible.m "shaped"). A word's stores are ordered by the compiler's edges
    (compiler.py stored), not by a maximum. What must stay: a plain coherent(system) store after the fence, never an
    atomic and never a copy, for any word another process reads while the run's buffer runs."""
    return (f'{prelude()}\nusing namespace metal;\nkernel void {STORE}(volatile coherent(system) device ulong *word '
            f'[[buffer(0)]], device const ulong *value [[buffer(1)]], uint at [[thread_position_in_grid]]) {{\n'
            f'    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, '
            f'static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE));\n    *word = value[0];\n}}\n')


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
