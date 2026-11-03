"""Kernel text for the compiler's commands (design/page-maps-2026-11-02.md; rdma/metal.h's second thing compiles it).

Every operand is a strided view: its shape, strides and element offset are literal constants of its kernel, so the
Metal compiler folds each index (a power-of-two extent's digit is a shift and a mask). A SEND source's stores and a
landing's loads are system-coherent (rdma/mesh.metal); every other access is an ordinary one. One kernel a command:
a pointwise map (a fused expression over its operands), a matrix product with its bias (product's two forms), an
indexed read or write through a device index, a sum over axes, and the run's observation of a word (a landing's or a
SEND's completion), a run of link and release word stores (compiler.py stored), and a run of REDUCE folds, each a
landing added into its bytes (compiler.py folded).
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
        """the element index of this view at `coordinates` (one expression an axis, broadcast axes left out): in 32 bits
        where the view's last element's index fits (its offset and every axis's last coordinate by its stride), as
        _digits' coordinates: the 64-bit products of an address cost the M4 Pro's one-dispatch LSTM tick about 2 us
        over its two-dispatch form (lstm-2026-11-03 narrow-on, mesh f22c0e90)"""
        last = self.offset + sum((n - 1) * s for s, n in zip(self.stride, self.shape) if s and n != 1)
        if 0 <= self.offset and all(s >= 0 for s in self.stride) and last < 1 << 32:
            terms = [f'{self.offset}u'] + [f'uint({c}) * {s}u' for c, s, n in zip(coordinates, self.stride, self.shape)
                                            if s and n != 1]
        else:
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
    # a threadgroup STAGED_COLUMNS columns of every row: its threads stage every row's A (every part, a landing's
    # system-coherently) a chunk of k at a time in threadgroup memory, then a simdgroup a column over the chunk, its
    # lanes over k, each lane's weight load used for every row. A simdgroup an output loading its own row coherently
    # cost the LSTM's product 7.3 us a dispatch against 2.3 with ordinary loads on an M5 Ultra (13.0 against 7.2 on
    # the Mini; metal-microbench tools/mesh/chain.m dot_coherent, dot_plain).
    # Changed from a threadgroup a row over a whole staged row (97eee528), which read every column's weights once a
    # row and staged k floats (8,192 at an LSTM of width 4096, whose tick took 28 ms against about 0.2 ms of weight
    # reads, metal-microbench docs/facts.md, balance-2026-11-03 sweep-4). What must stay: A's coherent loads staged
    # once, the weights read once
    S = min(STAGED_COLUMNS, N)
    K = sum(p.shape[1] for p in parts)
    KB = min(K, max(32, (4096 // M) // 32 * 32))  # a chunk: M x KB floats, 16 KiB of threadgroup memory at most
    loads, first = [], 0
    for q, p in enumerate(parts):
        Kp = p.shape[1]
        test = f'k < {first + Kp}ul' if q + 1 < len(parts) else 'true'
        loads.append(f'{"if" if q == 0 else "else if"} ({test}) {{ const ulong kp = k - {first}ul; v = float({p.load(["m", "kp"])}); }}')
        first += Kp
    body = f'''    const ulong me = at.x % {32 * S}ul, n = at.x / {32 * S}ul * {S}ul + me / 32ul, lane = me % 32ul;
    threadgroup float rows[{M * KB}];
    float total[{M}];
    for (uint m = 0; m < {M}u; m++) total[m] = 0.0f;
    for (ulong kc = 0; kc < {K}ul; kc += {KB}ul) {{
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ulong e = me; e < {M * KB}ul; e += {32 * S}ul) {{
            const ulong m = e / {KB}ul, k = kc + e % {KB}ul;
            float v = 0.0f;
            if (k < {K}ul) {{ {' '.join(loads)} }}
            rows[e] = v;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (n < {N}ul)
            for (ulong kk = lane; kk < {KB}ul && kc + kk < {K}ul; kk += 32ul) {{
                const ulong k = kc + kk;
                const float w = float({b.load(["k", "n"])});
                for (uint m = 0; m < {M}u; m++) total[m] += rows[m * {KB}u + kk] * w;
            }}
    }}
    for (uint mm = 0; mm < {M}u; mm++) {{
        const float t = simd_sum(total[mm]);
        const ulong m = mm;
        if (lane == 0ul && n < {N}ul) {{ {out.store(["m", "n"], f"{alpha}f * t{bias_term}")} }}
    }}'''
    return _source(name, parameters, body), (-(-N // S) * 32 * S, 1, 1), (32 * S, 1, 1)


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
# an observation's polls before it gives up and names itself (a fence and a coherent load each, a few to tens of ns):
# a stall named before the run's 8 s watchdog, not a slow program failed. At 2^22 (about 20-40 ms) an LSTM of width 4096
# (ticks of 20-28 ms behind its 8-row products) gave up on landings 3 short of their counts in its third round
# (metal-microbench balance-2026-11-03 sweep-3); the run's configuration may set it
OBSERVE_POLLS = int(__import__("os").environ.get("MESH_OBSERVE_POLLS", 1 << 28))


STORE = 'mesh_store'
# the run's argument table's buffers (rdma/metal.m, maxBufferBindCount): a dispatch binds at most this many
ARGUMENT_BUFFERS = 31
# a store's words at most: the argument table's buffers, one of them the words' values
STORE_WORDS = ARGUMENT_BUFFERS - 1


@functools.cache
def store(words):
    """a run of link and release words: argument 0 their values (consecutive in the run's value slots), argument k + 1
    word k, each stored its value, in order, after one system-scope fence, as a plain volatile coherent(system) store.
    Changed by the one-part session from a copy (and, before it, from mesh_publish's atomic maximum): with no dispatch
    reading the word's 64-byte sector, a running buffer's copy is never seen by another process within 1 s on the Mini,
    nor on an M5 Ultra without a barrier after it; mesh_publish's atomic never on the Mini, 4-169 ms on an M5 Ultra;
    this store 5.5-6.2 us from the buffer's start on the Mini and 4.6-7.0 on an M5 Ultra, its coherent payload (a SEND's
    source, stored coherently) complete behind it (metal-microbench docs/facts.md, visible.m "shaped"). Changed from a
    word a dispatch to a run's words a dispatch (compiler.py stored): a run's releases went one dependent dispatch
    after another, 2.6-4 us apart on an M5 Ultra and 7-8 us on the Mini (docs/facts.md, "The planner's all-reduce stalls
    about 1 ms"). A word's stores are ordered by the compiler's edges, not by a maximum. What must stay: a plain
    coherent(system) store after the fence, never an atomic and never a copy, for any word another process reads while
    the run's buffer runs."""
    parameters = ''.join(f', volatile coherent(system) device ulong *word{k} [[buffer({k + 1})]]' for k in range(words))
    body = ''.join(f'    *word{k} = value[{k}];\n' for k in range(words))
    return (f'{prelude()}\nusing namespace metal;\nkernel void {STORE}(device const ulong *value [[buffer(0)]]'
            f'{parameters}, uint at [[thread_position_in_grid]]) {{\n'
            f'    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, '
            f'static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE));\n{body}}}\n')


FOLD = 'mesh_fold'
FOLD_GROUP = 256  # a fold's threads: whole threadgroups of this many, so no threadgroup spans two folds


@functools.cache
def fold(types, folds):
    """a run of REDUCE folds (compiler.py folded), one dispatch: `types` each bound buffer's element type (argument k,
    bound from its storage's first byte), `folds` (element type, elements, writes, bytes, landing) each, a place a
    (buffer, element offset) pair and `writes` one or two places (_halves' x and y). Fold f's threads are a contiguous
    range of whole FOLD_GROUP-thread threadgroups after fold f - 1's; its thread j adds landing[j] into bytes[j] (in float,
    rounded once to the element type) and stores the sum to each place of `writes`, every load and store system-coherent
    (the landing the NIC wrote, the bytes a later SEND reads). Every offset is a literal constant of the text.
    Changed from a kernel a fold (compiler.py's pointwise 'x0 + x1', a dispatch each): a rank's 41-45 folds of the
    planner's all-reduce went one dependent dispatch after another behind their observations, 2.6-4 us apart on an M5
    Ultra and 7-8 us on the Mini, and its reduce-scatter half took about 1 ms for about 90 us of bytes (metal-microbench
    docs/facts.md, "The all-reduce's all-gather lands in pages of its own"). What must stay: each fold's sum rounded once
    and stored to each of its writes, its loads and stores system-coherent; folds over the same bytes in one dispatch
    never (the compiler ends a run there)."""
    parameters = [View(b, t, (1,), (1,), 0, coherent=True).parameter(True) for b, t in enumerate(types)]
    branches, start = [], 0
    for dtype, n, writes, bytes_, landing in folds:
        end = start + -(-n // FOLD_GROUP) * FOLD_GROUP
        place = lambda at, dtype=dtype, n=n: View(at[0], dtype, (n,), (1,), at[1], coherent=True)
        stores = ''.join(f'\n            {place(w).store(["j"], "s")}' for w in writes)
        branches.append(f'    if (i < {end}u) {{\n        const uint j = i - {start}u;\n        if (j < {n}u) {{\n'
                        f'            const {MSL[dtype]} s = {MSL[dtype]}(float({place(bytes_).load(["j"])}) + '
                        f'float({place(landing).load(["j"])}));{stores}\n        }}\n        return;\n    }}')
        start = end
    body = '    const uint i = at.x;\n' + '\n'.join(branches)
    return _source(FOLD, parameters, body), (start, 1, 1), (FOLD_GROUP, 1, 1)


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
