import hashlib
import re
import ctypes as c
import numpy as np

from .tensor import Dimension, broadcast_shape

class View(c.Structure):
    _fields_ = [('offset', c.c_uint64), ('size', c.c_uint64), ('shape', c.c_uint64 * 8),
        ('stride', c.c_uint64 * 8), ('matrix_columns', c.c_uint64), ('block_rows', c.c_uint64), ('block_columns', c.c_uint64), ('grid_columns', c.c_uint64)] + [(name, c.c_uint32) for name in ('dtype', 'rank', 'first')]


class Block(c.Structure):
    _fields_ = [('row_stride', c.c_uint64), ('column_stride', c.c_uint64)]


TYPES = {'float32': 'float', 'int32': 'int', 'uint32': 'uint', 'int64': 'long', 'uint64': 'ulong', 'bool': 'uchar'}
PREFIX = r'''
#include <metal_stdlib>
using namespace metal;
struct View {
    ulong offset, size, shape[8], stride[8], matrix_columns, block_rows, block_columns, grid_columns;
    uint dtype, rank, first;
};
struct Block { ulong row_stride, column_stride; };
ulong coordinate(ulong index, device const View& view, uint axis) {
    return index / view.stride[axis] % view.shape[axis];
}
ulong broadcast_index(ulong index, device const View& source, device const View& destination) {
    ulong result = 0;
    for (uint axis = 0; axis < source.rank; ++axis)
        result += (source.shape[axis] == 1 ? 0 : coordinate(index, destination, destination.rank - source.rank + axis)) * source.stride[axis];
    return result;
}
// ../../../design/algorithm-sources.md#xonotic-block-indexed-lowering
 device uchar* page_address(device const ulong* regions, device const Block* blocks, device const View& view, ulong byte) {
    ulong index=byte/view.dtype,row=index/view.matrix_columns,column=index%view.matrix_columns;
    ulong block=view.first+row/view.block_rows*view.grid_columns+column/view.block_columns;
    ulong offset=(row%view.block_rows)*blocks[block].row_stride+(column%view.block_columns)*blocks[block].column_stride;
    return (device uchar*)regions[block]+offset*view.dtype+byte%view.dtype;
}
// ../../../design/algorithm-sources.md#xonotic-block-indexed-lowering
 template<typename T> T read_value(device const ulong* regions, device const Block* blocks, device const View& view, ulong index) {
    return *(device const T*)page_address(regions, blocks, view, view.offset + index * sizeof(T));
}
// ../../../design/algorithm-sources.md#xonotic-block-indexed-lowering
 template<typename T> void write_value(device const ulong* regions, device const Block* blocks, device const View& view, ulong index, T value) {
    *(device T*)page_address(regions, blocks, view, view.offset + index * sizeof(T)) = value;
}
float tensor_log1p(float x) {
    float u = 1.0f + x;
    return u == 1.0f ? x : log(u) * (x / (u - 1.0f));
}
float tensor_expm1(float x) {
    return abs(x) < 0.01f ? x * (1.0f + x * (0.5f + x * (1.0f/6.0f + x * (1.0f/24.0f + x/120.0f)))) : exp(x) - 1.0f;
}
float stable_logaddexp(float x, float y) {
    float high = max(x, y), low = min(x, y);
    return isinf(high) ? high : high + tensor_log1p(exp(low - high));
}
float tensor_asinh(float x) {
    float a = abs(x);
    return a < 0.5f ? asinh(x) : copysign(a > 1.0e19f ? log(a) + 0.6931471805599453f : log(a + sqrt(a*a + 1.0f)), x);
}
uint4 philox(uint4 counter, uint2 key) {
    for (uint round = 0; round < 10; ++round) {
        ulong a = ulong(counter.x) * 0xD2511F53u, c = ulong(counter.z) * 0xCD9E8D57u;
        counter = uint4(uint(c >> 32) ^ counter.y ^ key.x, uint(c), uint(a >> 32) ^ counter.w ^ key.y, uint(a));
        key += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return counter;
}
'''
ARGUMENTS = '''device const ulong* regions [[buffer(0)]], device const View* v [[buffer(1)]],
    constant ulong* dimensions [[buffer(2)]], device const Block* blocks [[buffer(3)]],
    uint3 position [[thread_position_in_grid]], uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]], constant uint* arguments [[buffer(4)]]'''


def expr(value):
    if isinstance(value, Dimension):
        if value.op == 'axis': return f'dimensions[{value.args[0]}]'
        if value.op == 'poly':
            return '(' + '+'.join(str(coefficient) + ''.join(f'*dimensions[{axis}]' for axis in monomial) for coefficient, monomial in value.args) + ')'
        return '(' + expr(value.args[0]) + {'//': '/'}.get(value.op, value.op) + expr(value.args[1]) + ')'
    return str(value)


def read(value, index='t'):
    return f'read_value<{TYPES[value.dtype]}>(regions,blocks,v[{value.index}],{index})'


def write(value, result, index='t'):
    return f'write_value<{TYPES[value.dtype]}>(regions,blocks,v[{value.index}],{index},{TYPES[value.dtype]}({result}));'


def atomic(value, result, index):
    return f'atomic_fetch_add_explicit((device atomic_float*)page_address(regions,blocks,v[{value.index}],v[{value.index}].offset+({index})*sizeof(float)),float({result}),memory_order_relaxed);'


def coordinate_code(shape, flat='t', prefix='c'):
    result = [f'ulong {prefix}_remaining={flat};']
    for axis in reversed(range(len(shape))):
        result.append(f'ulong {prefix}{axis}={prefix}_remaining%({expr(shape[axis])}); {prefix}_remaining/=({expr(shape[axis])});')
    return '\n'.join(result)


def element(op, values):
    a = 'a0'
    c = 'a1'
    binary = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/', 'equal': '==', 'not_equal': '!=',
              'less': '<', 'less_equal': '<=', 'greater': '>', 'greater_equal': '>=', 'logical_and': '&&',
              'logical_or': '||', 'bitwise_and': '&', 'bitwise_or': '|'}
    if op in binary: return f'({a}{binary[op]}{c})'
    unary = {'negative': '-', 'logical_not': '!', 'bitwise_invert': '~'}
    if op in unary: return unary[op] + a
    functions = {'arcsinh': 'tensor_asinh', 'abs': 'abs', 'minimum': 'min', 'maximum': 'max', 'power': 'pow', 'logaddexp': 'stable_logaddexp', 'expm1': 'tensor_expm1', 'log1p': 'tensor_log1p'}
    if op == 'where': return 'a0?a1:a2'
    if op == 'sigmoid': return '(a0>=0?1.0f/(1.0f+exp(-a0)):exp(a0)/(1.0f+exp(a0)))'
    if op == 'floor_divide': return '(a0/a1)'
    if op in ('minimum', 'maximum', 'power'):
        dtype = 'float' if op == 'power' or any(value.dtype == 'float32' for value in values) else 'long' if any(value.dtype in ('int64', 'uint64') for value in values) else 'int'
        return functions[op] + '(' + ','.join(f'{dtype}(a{i})' for i in range(len(values))) + ')'
    return functions.get(op, op) + '(' + ','.join('a' + str(i) for i in range(len(values))) + ')'


def kernel(node):
    output, op, values, attrs, owner = node
    index = output.index
    name = f'mesh_tensor_{index}'
    body = [f'ulong t=position.x; if(t>=v[{index}].size) return;']
    mode, clear = 'linear', False
    if op in ('input', 'constant', 'dimension', 'reshape', 'stop_gradient'):
        if op != 'dimension': return None
        body.append(write(output, expr(attrs['expression'])))
    elif op in ('reshape', 'stop_gradient', 'cast', 'assign'):
        body.append(write(output, read(values[0])))
    elif op.startswith('reduce_'):
        source = values[0]
        axes = attrs['axes']
        reduced = math_product(tuple(source.shape[i] for i in axes))
        dtype = TYPES[source.dtype] if source.dtype != 'bool' else 'int'
        low, high = {'float32': ('-INFINITY', 'INFINITY'), 'int32': ('(-2147483647-1)', '2147483647'),
                     'uint32': ('0u', '0xffffffffu'), 'int64': ('(-9223372036854775807L-1L)', '9223372036854775807L'),
                     'uint64': ('0ul', '0xfffffffffffffffful'), 'bool': ('0', '1')}[source.dtype]
        identity = low if op == 'reduce_max' else high if op == 'reduce_min' else '1' if op == 'reduce_all' else '0'
        body = [f'ulong t=group.x; if(t>=v[{index}].size) return;', f'{dtype} result={identity};']
        base, target_axis = [], 0
        for i in range(source.ndim):
            if i not in axes:
                base.append(f'coordinate(t,v[{index}],{i if attrs["keepdims"] else target_axis})*v[{source.index}].stride[{i}]')
                target_axis += 1
        body.append('ulong base=' + ('+'.join(base) or '0') + ';')
        body.append(f'for(ulong r=tid;r<{expr(reduced)};r+=256) {{ ulong rest=r,address=base;')
        for i in reversed(axes):
            body.append(f'address+=(rest%v[{source.index}].shape[{i}])*v[{source.index}].stride[{i}]; rest/=v[{source.index}].shape[{i}];')
        combine = 'max(result,value)' if op == 'reduce_max' else 'min(result,value)' if op == 'reduce_min' else 'result||value' if op == 'reduce_any' else 'result&&value' if op == 'reduce_all' else 'result+value'
        operation = 'simd_max' if op in ('reduce_max', 'reduce_any') else 'simd_min' if op in ('reduce_min', 'reduce_all') else 'simd_sum'
        body += [f'{dtype} value={dtype}({read(source,"address")}); result={combine}; }}',
                 f'threadgroup {dtype} partial[8]; result={operation}(result); if(!lane) partial[simd]=result;',
                 'threadgroup_barrier(mem_flags::mem_threadgroup);',
                 f'if(!simd) {{ result=lane<8?partial[lane]:{dtype}({identity}); result={operation}(result); if(!lane) {{',
                 write(output, f'result/float({expr(reduced)})' if op == 'reduce_mean' else 'result'), '}}']
        mode = 'reduce'
    elif op == 'arange':
        body.append(write(output, f'{expr(attrs["start"])}+t*({expr(attrs["step"])})'))
    elif op == 'argsort':
        source, axis = values[0], attrs['axis']
        body.append(f'ulong at=coordinate(t,v[{source.index}],{axis}), stride=v[{source.index}].stride[{axis}]; ulong base=t-at*stride;')
        body.append(f'auto value={read(source)}; ulong rank=0; for(ulong i=0;i<v[{source.index}].shape[{axis}];++i) {{ auto other={read(source,"base+i*stride")}; rank+=(other<value || (other==value && i<at)); }}')
        body.append(write(output, 'at', 'base+rank*stride'))
    elif op == 'random_normal':
        key = values[0]
        body.append(f'uint2 key=uint2({read(key,"0")},{read(key,"1")}); uint4 bits=philox(uint4(uint(t/2),uint((t/2)>>32),0,0),key);')
        body.append('float radius=sqrt(-2.0f*log((float(bits.x>>9)+0.5f)*0x1p-23f)); float angle=6.283185307179586f*(float(bits.y>>9)*0x1p-23f);')
        body.append(write(output, '(t&1)?radius*sin(angle):radius*cos(angle)'))
    elif op == 'matmul':
        body, mode = matmul_body(output, values, attrs), 'matmul'
    elif op == 'expert_route':
        selected = values[0]
        body = [f'ulong t=position.x; if(t>=v[{selected.index}].size) return;',
                f'ulong expert=ulong({read(selected)}),stride=v[{output.index}].shape[1];',
                f'uint slot=atomic_fetch_add_explicit((device atomic_uint*)page_address(regions,blocks,v[{output.index}],v[{output.index}].offset+expert*stride*sizeof(uint)),1u,memory_order_relaxed);',
                write(output, 't', 'expert*stride+1+slot')]
        mode, clear = ('gradient', selected.index), True
    elif op in ('expert_matmul', 'expert_input_vjp', 'expert_weight_vjp'):
        body, mode, clear = expert_body(output, values, op)
    elif op in ('neighborhood', 'neighborhood_vjp'):
        body, mode, clear = neighborhood_body(output, values, attrs, op)
    else:
        for i, value in enumerate(values):
            address = f'broadcast_index(t,v[{value.index}],v[{index}])'
            body.append(f'auto a{i}={read(value,address)};')
        body.append(write(output, 'a0' if op == 'broadcast' else element(op, values)))
    return {'name': name, 'source': f'kernel void {name}({ARGUMENTS}) {{\n' + '\n'.join(body) + '\n}\n',
            'node': index, 'mode': mode, 'clear': clear, 'owner': owner,
            'arguments': list(dict.fromkeys(value.index for value in (*values, output)))}


def math_product(shape):
    result = 1
    for value in shape: result *= value
    return result


def matmul_body(output, values, attrs):
    left, right = values
    l, r, o = left.index, right.index, output.index
    transpose_left, transpose_right = attrs['transpose_left'], attrs['transpose_right']
    rows, columns = f'v[{o}].shape[{output.ndim-2}]', f'v[{o}].shape[{output.ndim-1}]'
    inner = f'v[{l}].shape[{left.ndim-(2 if transpose_left else 1)}]'
    batch_left = batch_address(left, output)
    batch_right = batch_address(right, output)
    a = f'base_a+({"k*"+rows+"+row" if transpose_left else "row*"+inner+"+k"})'
    c = f'base_b+({"column*"+inner+"+k" if transpose_right else "k*"+columns+"+column"})'
    return [f'ulong batch=group.z, base_a={batch_left}, base_b={batch_right};'] + tiled_product(
        output, rows, columns, inner, read(left, a), read(right, c), f'(batch*{rows}+row)*{columns}+column')


def tiled_product(output, rows, columns, inner, left, right, address):
    return [            'threadgroup float tile_a[64*32], tile_b[32*32], tile_c[64*32];',
            'simdgroup_float8x8 accum[4]; for(uint i=0;i<4;++i) accum[i]=simdgroup_float8x8(0);',
            f'for(ulong first=0;first<{inner};first+=32) {{',
            f'for(uint i=tid;i<64*32;i+=256) {{ ulong row=group.y*64+i/32,k=first+i%32; tile_a[i]=(row<{rows}&&k<{inner})?float({left}):0; }}',
            f'for(uint i=tid;i<32*32;i+=256) {{ ulong k=first+i/32,column=group.x*32+i%32; tile_b[i]=(column<{columns}&&k<{inner})?float({right}):0; }}',
            'threadgroup_barrier(mem_flags::mem_threadgroup);',
            'for(uint part=0;part<4;++part) { uint tile=simd+part*8,row=tile/4*8,column=tile%4*8;',
            'for(uint k=0;k<32;k+=8) { simdgroup_float8x8 a,b; simdgroup_load(a,tile_a+row*32+k,32); simdgroup_load(b,tile_b+k*32+column,32); simdgroup_multiply_accumulate(accum[part],a,b,accum[part]); }}',
            'threadgroup_barrier(mem_flags::mem_threadgroup); }',
            'for(uint part=0;part<4;++part) { uint tile=simd+part*8; simdgroup_store(accum[part],tile_c+(tile/4*8)*32+tile%4*8,32); }',
            'threadgroup_barrier(mem_flags::mem_threadgroup);',
            f'for(uint i=tid;i<64*32;i+=256) {{ ulong row=group.y*64+i/32,column=group.x*32+i%32; if(row<{rows}&&column<{columns}) {{ {write(output,"tile_c[i]",address)} }} }}']


def batch_address(source, output):
    pieces = []
    for i in range(source.ndim - 2):
        if source.shape[i] == 1: continue
        axis = output.ndim - source.ndim + i
        trailing = math_product(output.shape[axis+1:-2])
        pieces.append(f'((batch/({expr(trailing)}))%v[{source.index}].shape[{i}])*v[{source.index}].stride[{i}]')
    return '+'.join(pieces) or '0'


def expert_body(output, values, op):
    inputs, weights, selected, routing = values[:4]
    n, d, h = (expr(value) for value in (inputs.shape[0], weights.shape[1], weights.shape[2]))
    count = read(routing, f'expert*({n}+1)')
    def routed(index): return 'ulong(' + read(routing, f'expert*({n}+1)+1+({index})') + ')'
    prefix = [f'ulong expert=group.z,count=ulong({count});']
    if op == 'expert_weight_vjp':
        gradient = values[4]
        rows, columns, inner = d, h, 'count'
        left = read(inputs, f'{routed("k")}*{d}+row')
        right = read(gradient, f'{routed("k")}*{h}+column')
        address = f'(expert*{d}+row)*{h}+column'
        mode = 'matmul'
    else:
        backward = op == 'expert_input_vjp'
        rows, columns, inner = 'count', d if backward else h, h if backward else d
        source = values[4] if backward else inputs
        left = read(source, f'{routed("row")}*({inner})+k')
        right = read(weights, f'(expert*{d}+column)*{h}+k' if backward else f'(expert*{d}+k)*{h}+column')
        address = f'{routed("row")}*({columns})+column'
        prefix += ['if(group.y*64>=count) return;']
        mode = ('expert', weights.index)
    return prefix + tiled_product(output, rows, columns, inner, left, right, address), mode, False


def neighborhood_body(output, values, attrs, op):
    query, keys, vectors, indices, weights = values[:5]
    width, neighbors = query.shape[1], weights.shape[1]
    target, gram = attrs.get('target'), attrs['gram']
    backward = op.endswith('_vjp')
    if not backward:
        body = [f'ulong observer=group.x; if(observer>=v[{query.index}].shape[0]) return;',
                f'float result[{(width+31)//32}]={{}};',
                f'for(ulong j=0;j<{expr(neighbors)};++j) {{ ulong at=observer*({expr(neighbors)})+j,source=ulong({read(indices,"at")}); float affinity=1;']
        if gram:
            body += ['float dot=0;', f'for(uint d=lane;d<{width};d+=32) dot+=float({read(query,f"observer*{width}+d")})*float({read(keys,f"source*{width}+d")});', f'affinity=simd_sum(dot)*rsqrt(float({width}));']
        body += [f'float weight=float({read(weights,"at")})*affinity;',
                 f'for(uint d=lane;d<{width};d+=32) result[d/32]+=weight*float({read(vectors,f"source*{width}+d")}); }}',
                 f'for(uint d=lane;d<{width};d+=32) {{ {write(output,"result[d/32]",f"observer*{width}+d")} }}']
        return body, ('neighborhood', query.index), False
    gradient = values[5]
    body = [f'ulong at=group.x; if(at>=v[{weights.index}].size) return;',
            f'ulong observer=at/({expr(neighbors)}),source=ulong({read(indices,"at")}); float dot=0,dot_v=0;']
    if gram:
        body += [f'for(uint d=lane;d<{width};d+=32) dot+=float({read(query,f"observer*{width}+d")})*float({read(keys,f"source*{width}+d")});', f'dot=simd_sum(dot)*rsqrt(float({width}));']
    else: body += ['dot=1;']
    body += [f'for(uint d=lane;d<{width};d+=32) dot_v+=float({read(gradient,f"observer*{width}+d")})*float({read(vectors,f"source*{width}+d")});',
             f'dot_v=simd_sum(dot_v); float weight=float({read(weights,"at")});']
    if target == 4:
        body += [f'if(!lane) {{ {write(output,"dot_v*dot","at")} }}']
    elif target == 2:
        body += [f'for(uint d=lane;d<{width};d+=32) {{ {atomic(output,f"weight*dot*float({read(gradient,f' observer*{width}+d')})",f"source*{width}+d")} }}']
    elif gram:
        address = f' observer*{width}+d' if target == 0 else f'source*{width}+d'
        other = read(keys,f'source*{width}+d') if target == 0 else read(query,f'observer*{width}+d')
        body += [f'for(uint d=lane;d<{width};d+=32) {{ {atomic(output,f"weight*dot_v*rsqrt(float({width}))*float({other})",address)} }}']
    return body, ('edges', weights.index), target != 4


# ../../../design/algorithm-sources.md#indexed-expression-lowering
def source(nodes):
    kernels = [value for node in nodes if (value := kernel(node)) is not None]
    sources = {}
    for item in kernels:
        nodes = item['arguments']
        text = re.sub(r'v\[(\d+)\]', lambda match: f'v[arguments[{nodes.index(int(match[1]))}]]', item['source'])
        text = text.replace(item['name'], 'FUNCTION')
        name = 'mesh_tensor_' + hashlib.sha256(text.encode()).hexdigest()[:16]
        sources[name] = '// ../../../design/algorithm-sources.md#literal-row-functions\n' + text.replace('FUNCTION', name)
        item.update(name=name, arguments=nodes)
    sources['mesh_tensor_zero'] = f'// ../../../design/algorithm-sources.md#literal-row-functions\nkernel void mesh_tensor_zero({ARGUMENTS}) {{ ulong t=position.x; if(t<v[arguments[0]].size*v[arguments[0]].dtype) *page_address(regions,blocks,v[arguments[0]],v[arguments[0]].offset+t)=0; }}'
    return PREFIX + '\n'.join(sources.values()), kernels


# ../../../design/algorithm-sources.md#xonotic-partitioned-reshape
def matrix_view(tensor, shape):
    import math
    from mesh import Tensor, Ref
    from mesh._native import View as NativeView
    shape = tuple(shape)
    if len(shape) != 2 or min(shape) <= 0 or math.prod(shape) != math.prod(tensor.shape):
        raise ValueError('Matrix view must preserve element count with two positive dimensions')
    if tensor.shape == shape:
        return tensor
    if tensor.shape[::-1] == shape and 1 in shape:
        return tensor.T
    if tensor.shape[1] == 1 and tensor.shape[0] == shape[0] * shape[1] and tensor.block_shape[0] % shape[1] == 0:
        block_shape = (tensor.block_shape[0] // shape[1], shape[1])
        grid = (tensor.grid[0], 1)
        regions = tuple((coordinate, ref, (ref.shape[0] // shape[1], shape[1])) for coordinate, ref in tensor.blocks.items())
    elif tensor.grid == (1, 1) and (1 in tensor.shape or
            tensor[0, 0].view.row_stride == tensor.shape[1] * tensor[0, 0].view.column_stride):
        block_shape, grid = shape, (1, 1)
        regions = (((0, 0), tensor[0, 0], shape),)
    else:
        width = math.gcd(shape[1], tensor.shape[1], tensor.block_shape[1])
        block_shape, grid = (1, width), (shape[0], shape[1] // width)
        regions = []
        for i in range(grid[0]):
            for j in range(grid[1]):
                row, column = divmod(i * shape[1] + j * width, tensor.shape[1])
                regions.append(((i, j), tensor.region(row, column, 1, width), (1, width)))
    result = object.__new__(Tensor)
    result.program, result.dtype, result.handle = tensor.program, tensor.dtype, tensor.handle
    result.shape, result.block_shape, result.grid = shape, block_shape, grid
    result.blocks = {}
    for coordinate, ref, extent_shape in regions:
        array = ref.array.view()
        array.shape = extent_shape
        view = NativeView.from_buffer_copy(ref.view)
        view.rows, view.columns = extent_shape
        view.row_stride, view.column_stride = (stride // tensor.dtype.itemsize for stride in array.strides)
        result.blocks[coordinate] = Ref(tensor.program, view, tensor.dtype)
    return result


# ../../../design/algorithm-sources.md#xonotic-logical-indexing
def logical_coordinates(kernels, shape, block):
    import math
    row, column = kernels.indices()
    row = kernels.program_id(0) * block[0] + row
    column = kernels.program_id(1) * block[1] + column
    return tuple((row // math.prod(shape[axis + 1:-1])) % size
                 for axis, size in enumerate(shape[:-1])) + ((column,) if shape else ())


# ../../../design/algorithm-sources.md#xonotic-gather-transpose
def gather_coordinates(kernels, arguments, values, shapes, attributes, coordinates, capacity):
    mapping = tuple(tuple(part.resolve(capacity) if isinstance(part, Dimension) else part for part in item)
                    for item in attributes['mapping'])
    advanced, adjacent = attributes['advanced'], attributes['adjacent']
    axis = 0 if adjacent else len(advanced)
    advanced_axis = None if adjacent else 0
    source_axis, source_coordinates = 0, []
    for item in mapping:
        if item[0] == 'new':
            axis += 1
            continue
        if item[0] == 'index':
            if advanced_axis is None:
                advanced_axis = axis
                axis += len(advanced)
            selected = values[1 + item[1]]
            selected_shape = shapes[selected.index]
            at = tuple(0 if size == 1 else coordinates[advanced_axis + len(advanced) - len(selected_shape) + i]
                       for i, size in enumerate(selected_shape))
            selected_value = arguments[item[1]].at(*at)
            coordinate = kernels.select(selected_value < 0, selected_value + shapes[values[0].index][source_axis], selected_value) if selected.dtype.startswith('int') else selected_value
        elif item[0] == 'fixed':
            coordinate = item[1]
        else:
            coordinate = item[1] + coordinates[axis] * item[2]
            axis += 1
        source_coordinates.append(coordinate)
        source_axis += 1
    return tuple(source_coordinates)


# ../../../design/algorithm-sources.md#xonotic-take-transpose
def take_coordinates(kernels, source_shape, index_shape, index_dtype, index_value, axis, coordinates):
    index_at = tuple(0 if size == 1 else coordinate for size, coordinate in zip(index_shape, coordinates))
    selected = index_value.at(*index_at)
    if index_dtype.startswith('int'):
        selected = kernels.select(selected < 0, selected + source_shape[axis], selected)
    return tuple(selected if i == axis else 0 if size == 1 else coordinates[i]
                 for i, size in enumerate(source_shape))


# ../../../design/algorithm-sources.md#xonotic-output-liveness
def row_gather_gradient(value, operation, values, attributes, shapes):
    return operation == 'gather_vjp' and len(values) == 3 and len(shapes[values[1].index]) == 1 and (
        (len(shapes[value.index]) == 1 and tuple(attributes['mapping']) == (('index', 0),)) or
        (len(shapes[value.index]) == 2 and tuple(attributes['mapping']) == (('index', 0), ('slice', 0, 1)) and
         shapes[values[-1].index] == (shapes[values[1].index][0], shapes[value.index][1])))


# ../../../design/algorithm-sources.md#application-metal-kernels
def kernel_calls(program, graph, capacity, inputs, *, outputs, root_peer=None,
                 tile_rows=64, tile_k=128, tile_columns=128):
    from mesh import BlockSpec, ShapeDtypeStruct, kernels, nn
    from mesh.kernels import Metal, MetalDispatch
    import math

    shapes = {value.index: tuple(size.resolve(capacity) if isinstance(size, Dimension) else size for size in value.shape)
              for value, _, _, _, _ in graph.nodes}
    outputs = tuple(outputs)
    if any(value.graph is not graph for value in outputs):
        raise ValueError('Requested outputs must belong to the compiled graph')
    live = {value.index for value in outputs}
    nodes = []
    for node in reversed(graph.nodes):
        value, operation, values, attributes, owner = node
        if value.index not in live or value.index in inputs:
            continue
        nodes.append(node)
        dependencies = values[1:] if operation in ('gather_vjp', 'take_along_axis_vjp') else values
        live.update(operand.index for operand in dependencies)
    nodes.reverse()
    constants = {value.index: data for value, data in graph.constants.values()}
    tensors = dict(inputs)
    peers = {0: program.node if root_peer is None else root_peer}
    peers.update({region['owner']: region['peer'] for region in graph.regions.values()})
    owners = {value.index: peers[owner] for value, _, _, _, owner in graph.nodes}
    replicas = {}
    for value, operation, values, attributes, owner in nodes:
        peer = peers[owner]
        if value.index in tensors:
            continue
        if operation in ('reshape', 'stop_gradient'):
            tensors[value.index] = tensors[values[0].index]
            owners[value.index] = owners[values[0].index]
            continue
        if operation == 'constant':
            shape = shapes[value.index]
            storage_shape = (max(1, math.prod(shape[:-1])), max(1, shape[-1])) if shape else (1, 1)
            tensor = program.tensor(storage_shape, dtype=value.dtype)
            data = constants[value.index]
            if program.node == peer:
                program.constant(tensor[0, 0], data.reshape(storage_shape))
            tensors[value.index] = tensor
            continue
        row_gradient = row_gather_gradient(value, operation, values, attributes, shapes)
        local = {}
        for operand in values:
            if operation in ('gather_vjp', 'take_along_axis_vjp') and operand.index == values[0].index:
                continue
            tensor = tensors[operand.index]
            sender = owners[operand.index]
            if sender != peer:
                key = (operand.index, peer)
                if key not in replicas:
                    transposed = tensor[0, 0].view.row_stride == 1 and tensor[0, 0].view.column_stride != 1
                    backing = tensor.T if transposed else tensor
                    replica = program.tensor(backing.shape, block_shape=backing.block_shape, dtype=backing.dtype)
                    replica = replica.T if transposed else replica
                    program.copy(tensor.on(sender), replica.on(peer))
                    replicas[key] = replica
                tensor = replicas[key]
            local[operand.index] = tensor
        shape = shapes[value.index]
        if row_gradient:
            vector = len(shape) == 1
            output_shape = (shape[0], 1) if vector else shape
            destinations = matrix_view(local[values[1].index], (shapes[values[1].index][0], 1))
            gradient_shape = (shapes[values[-1].index][0], 1) if vector else shapes[values[-1].index]
            updates = matrix_view(local[values[-1].index], gradient_shape)
            block = (min(tile_rows, output_shape[0]), math.gcd(min(tile_columns, output_shape[1]),
                updates.block_shape[1] if updates.grid[1] > 1 else 0))
            base = program.tensor(output_shape, block_shape=block, dtype=value.dtype)
            if program.node == peer:
                for ref in base.blocks.values():
                    program.constant(ref, np.zeros(ref.shape, dtype=value.dtype))
            base_arg, index_arg, update_arg = kernels.arguments(3)
            result = program.kernel_call(kernels.expression(kernels.indexed_add(base_arg, index_arg, update_arg)),
                grid=base.grid, in_specs=(BlockSpec(None),) * 3,
                out_specs=BlockSpec(block, lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct(output_shape, value.dtype), peer=peer)(base, destinations, updates)
            tensors[value.index] = result.T if vector else result
            continue
        # ../../../design/algorithm-sources.md#xonotic-gather-transpose
        if operation in ('gather_vjp', 'take_along_axis_vjp'):
            indices = tuple(local[v.index] for v in values[1:-1])
            cotangent = local[values[-1].index]
            cotangent_shape = shapes[values[-1].index]
            updates = math.prod(cotangent_shape)
            key_block = (min(tile_rows, updates), 1)
            row, _ = kernels.indices()
            ordinal = kernels.program_id(0) * key_block[0] + row
            coordinates = tuple((ordinal // math.prod(cotangent_shape[i + 1:])) % size
                                for i, size in enumerate(cotangent_shape))
            index_values = tuple(argument.reshape(shapes[value.index])
                                 for argument, value in zip(kernels.arguments(len(indices)), values[1:-1]))
            source_at = (gather_coordinates(kernels, index_values, values, shapes, attributes, coordinates, capacity)
                if operation == 'gather_vjp' else take_coordinates(kernels, shape, shapes[values[1].index],
                    values[1].dtype, index_values[0], attributes['axis'], coordinates))
            destination = sum(coordinate * math.prod(shape[i + 1:]) for i, coordinate in enumerate(source_at))
            valid = True
            for coordinate, size in zip(source_at, shape):
                valid = kernels.select(valid, (coordinate >= 0) & (coordinate < size), False)
            destination = kernels.select(valid, destination, 0xffffffff)
            destinations = program.kernel_call(kernels.expression(destination),
                grid=((updates + key_block[0] - 1) // key_block[0], 1),
                in_specs=(BlockSpec(None),) * len(indices), out_specs=BlockSpec(key_block, lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct((updates, 1), 'int64'), peer=peer)(*indices)
            storage_shape = (math.prod(shape[:-1]), shape[-1]) if shape else (1, 1)
            block = (min(tile_rows, storage_shape[0]) * storage_shape[1], 1)
            base = program.tensor((math.prod(shape), 1), block_shape=block, dtype=value.dtype)
            if program.node == peer:
                for ref in base.blocks.values():
                    program.constant(ref, np.zeros(ref.shape, dtype=value.dtype))
            base_value, index_value, cotangent_value = kernels.arguments(3)
            row, _ = kernels.indices()
            result = program.kernel_call(kernels.expression(kernels.indexed_add(base_value, index_value,
                cotangent_value.reshape((updates,)).at(row))),
                grid=base.grid, in_specs=(BlockSpec(None),) * 3,
                out_specs=BlockSpec(block, lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct(base.shape, value.dtype), peer=peer)(base, destinations, cotangent)
            tensors[value.index] = matrix_view(result, storage_shape)
            continue
        # ../../../design/algorithm-sources.md#xonotic-row-scatter
        if operation == 'scatter_add':
            if np.dtype(values[1].dtype).kind not in 'iu':
                raise TypeError('Scatter indices must have integer dtype')
            index_shape, update_shape = (shapes[v.index] for v in values[1:])
            selected_shape = index_shape + shape[1:]
            if broadcast_shape(update_shape, selected_shape) != selected_shape:
                raise ValueError('Scatter updates must broadcast to the selected rows')
            storage_shape = (shape[0], 1) if len(shape) == 1 else (math.prod(shape[:-1]), shape[-1])
            base = matrix_view(local[values[0].index], storage_shape)
            indices, updates = (local[v.index] for v in values[1:])
            row_expansion = math.prod(shape[1:-1]) if len(shape) > 1 else 1
            update_rows = math.prod(index_shape) * row_expansion
            if row_expansion == 1 and len(index_shape) <= 1 and indices.shape in ((update_rows, 1), (1, update_rows)):
                destinations = matrix_view(indices, (update_rows, 1))
            else:
                key_block = (min(tile_rows, update_rows), 1)
                row, _ = kernels.indices()
                ordinal = kernels.program_id(0) * key_block[0] + row
                selected_ordinal = ordinal // row_expansion
                index_at = tuple((selected_ordinal // math.prod(index_shape[i + 1:])) % size
                                 for i, size in enumerate(index_shape))
                index_value, = kernels.arguments(1)
                selected = index_value.reshape(index_shape).at(*index_at)
                if values[1].dtype.startswith('int'):
                    selected = kernels.select(selected < 0, selected + shape[0], selected)
                destination = kernels.select((selected >= 0) & (selected < shape[0]),
                    selected * row_expansion + ordinal % row_expansion, 0xffffffff)
                destinations = program.kernel_call(kernels.expression(destination),
                    grid=((update_rows + key_block[0] - 1) // key_block[0], 1),
                    in_specs=(BlockSpec(None),), out_specs=BlockSpec(key_block, lambda i, j: (i, j)),
                    out_shape=ShapeDtypeStruct((update_rows, 1), 'int64'), peer=peer)(indices)
            base_value, index_value, update_value = kernels.arguments(3)
            update_matrix = (math.prod(update_shape), 1) if len(shape) == 1 else update_shape if len(update_shape) == 2 else (1, math.prod(update_shape))
            direct = len(index_shape) <= 1 and len(shape) <= 2 and len(update_shape) <= 2 and (
                updates.shape == update_matrix or (updates.shape[::-1] == update_matrix and 1 in update_matrix))
            if direct:
                updates = matrix_view(updates, update_matrix).broadcast_to((update_rows, storage_shape[1]))
            else:
                row, column = kernels.indices()
                selected_ordinal, trailing_ordinal = row // row_expansion, row % row_expansion
                coordinates = tuple((selected_ordinal // math.prod(index_shape[i + 1:])) % size
                                    for i, size in enumerate(index_shape))
                coordinates += tuple((trailing_ordinal // math.prod(shape[i + 2:-1])) % size
                                     for i, size in enumerate(shape[1:-1]))
                if len(shape) > 1:
                    coordinates += (column,)
                update_at = tuple(0 if size == 1 else coordinate for size, coordinate in
                                  zip(update_shape, coordinates[len(selected_shape) - len(update_shape):]))
                update_value = update_value.reshape(update_shape).at(*update_at)
            block = (math.gcd(min(tile_rows, base.shape[0]), base.block_shape[0] if base.grid[0] > 1 or len(shape) == 1 else 0),
                     math.gcd(min(tile_columns, base.shape[1]), base.block_shape[1] if base.grid[1] > 1 else 0))
            if direct and updates.grid[1] > 1:
                block = (block[0], math.gcd(block[1], updates.block_shape[1]))
            result = program.kernel_call(kernels.expression(kernels.indexed_add(base_value, index_value, update_value)),
                grid=tuple((size + tile - 1) // tile for size, tile in zip(base.shape, block)),
                in_specs=(BlockSpec(None),) * 3, out_specs=BlockSpec(block, lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct(base.shape, value.dtype), peer=peer)(base, destinations, updates)
            tensors[value.index] = result.T if len(shape) == 1 else result
            continue
        # ../../../design/algorithm-sources.md#streamed-row-reductions-in-the-shared-region-owner
        if operation in ('reduce_sum', 'reduce_mean') and 1 <= len(shapes[values[0].index]) <= 2 and attributes['axes']:
            operand_shape = shapes[values[0].index]
            operand = matrix_view(local[values[0].index], operand_shape if len(operand_shape) == 2 else (1, math.prod(operand_shape)))
            axes = tuple(sorted(set(attributes['axes']))) if len(operand_shape) == 2 else (1,)
            reduced_shape = tuple(1 if axis in axes else size for axis, size in enumerate(operand.shape))
            block = tuple(math.gcd(min(tile, size), operand.block_shape[axis] if operand.grid[axis] > 1 else 0)
                          for axis, (tile, size) in enumerate(zip((tile_rows, tile_rows), reduced_shape)))
            argument, = kernels.arguments(1)
            term = argument & 0xffffffffffffffff if operand.dtype.kind in 'iu' else argument
            result = term.sum(axis=axes)
            divisor = math.prod(operand.shape[axis] for axis in axes)
            if operation == 'reduce_mean' and operand.dtype.kind == 'f':
                result = result / divisor
            reduced = program.kernel_call(kernels.expression(result),
                grid=tuple((size + tile - 1) // tile for size, tile in zip(reduced_shape, block)),
                in_specs=(BlockSpec(None),),
                out_specs=BlockSpec(block, lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct(reduced_shape, value.dtype), peer=peer)(operand)
            tensors[value.index] = nn._pointwise(program, kernels.expression(argument / divisor),
                (reduced,), block[0], peer=peer, output_dtype=value.dtype) if operation == 'reduce_mean' and operand.dtype.kind != 'f' else reduced
            continue
        # ../../../design/algorithm-sources.md#xonotic-logical-indexing
        if operation in ('gather', 'take_along_axis', 'concatenate', 'transpose'):
            matrix_shape = (math.prod(shape[:-1]), shape[-1]) if shape else (1, 1)
            block = (min(tile_rows, matrix_shape[0]), min(tile_columns, matrix_shape[1]))
            coordinates = logical_coordinates(kernels, shape, block)
            arguments = tuple(argument.reshape(shapes[value.index])
                              for argument, value in zip(kernels.arguments(len(values)), values))
            if operation == 'gather':
                source_at = gather_coordinates(kernels, arguments[1:], values, shapes, attributes, coordinates, capacity)
                result = arguments[0].at(*source_at)
            elif operation == 'take_along_axis':
                source_at = take_coordinates(kernels, shapes[values[0].index], shapes[values[1].index],
                    values[1].dtype, arguments[1], attributes['axis'], coordinates)
                result = arguments[0].at(*source_at)
            elif operation == 'transpose':
                source_at = [0] * len(shape)
                for coordinate, axis in zip(coordinates, attributes['axes']):
                    source_at[axis] = coordinate
                result = arguments[0].at(*source_at)
            else:
                axis, offset, terms = attributes['axis'], 0, []
                for argument, operand in zip(arguments, values):
                    at = list(coordinates)
                    at[axis] = at[axis] - offset
                    offset += shapes[operand.index][axis]
                    terms.append((coordinates[axis] < offset, argument.at(*at)))
                result = terms[-1][1]
                for condition, value_expression in reversed(terms[:-1]):
                    result = kernels.select(condition, value_expression, result)
            operands = tuple(local[v.index] for v in values)
            tensors[value.index] = program.kernel_call(kernels.expression(result),
                grid=tuple((size + extent - 1) // extent for size, extent in zip(matrix_shape, block)),
                in_specs=(BlockSpec(None),) * len(operands),
                out_specs=BlockSpec(block, lambda i, j: (i, j)),
                out_shape=ShapeDtypeStruct(matrix_shape, value.dtype), peer=peer)(*operands)
            continue
        numerical = value.dtype in ('float16', 'float32')
        matrix_shapes = {v.index: shapes[v.index] if len(shapes[v.index]) == 2 else (1, math.prod(shapes[v.index])) for v in values}
        shaped = all(local[v.index].shape == matrix_shapes[v.index] or
            (1 in matrix_shapes[v.index] and local[v.index].shape[::-1] == matrix_shapes[v.index]) for v in values)
        if numerical and operation == 'matmul' and len(shape) == 2 and all(len(shapes[v.index]) == 2 and local[v.index].shape == shapes[v.index] for v in values):
            left, right = (local[v.index] for v in values)
            left = left.T if attributes['transpose_left'] else left
            right = right.T if attributes['transpose_right'] else right
            tensors[value.index] = nn.linear(program, left, right, tile_rows=tile_rows,
                tile_k=tile_k, tile_columns=tile_columns, peer=peer)
            continue
        if operation in ('add', 'subtract', 'multiply', 'divide', 'negative', 'exp', 'tanh', 'rsqrt', 'sigmoid', 'maximum', 'minimum', 'cast', 'assign', 'where', 'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal', 'bitwise_and', 'bitwise_or'):
            args = kernels.arguments(len(values))
            direct = shaped and len(shape) <= 2
            if not direct:
                matrix_shape = (math.prod(shape[:-1]), shape[-1]) if shape else (1, 1)
                block = (min(tile_rows, matrix_shape[0]), min(tile_columns, matrix_shape[1]))
                coordinates = logical_coordinates(kernels, shape, block)
                args = tuple(argument.reshape(shapes[operand.index]).at(*(
                    0 if size == 1 else coordinate for size, coordinate in
                    zip(shapes[operand.index], coordinates[len(shape)-len(shapes[operand.index]):])))
                    for argument, operand in zip(args, values))
            if operation in ('add', 'subtract', 'multiply', 'divide'):
                left, right = args
                if value.dtype in ('int32', 'uint32', 'int64', 'uint64') and operation != 'divide':
                    left, right = left & 0xffffffffffffffff, right & 0xffffffffffffffff
                result = {'add': lambda: left + right, 'subtract': lambda: left - right,
                          'multiply': lambda: left * right, 'divide': lambda: left / right}[operation]()
            elif operation in ('cast', 'assign'):
                result = args[0].astype(value.dtype) if operation == 'cast' else args[0]
            elif operation in ('maximum', 'minimum'):
                left, right = args
                result = kernels.select(left > right if operation == 'maximum' else left < right, left, right)
            elif operation == 'where':
                result = kernels.select(*args)
            elif operation in ('equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal', 'bitwise_and', 'bitwise_or'):
                left, right = args
                result = {'equal': lambda: left.equal(right), 'not_equal': lambda: kernels.select(left.equal(right), False, True),
                          'less': lambda: left < right, 'less_equal': lambda: left <= right,
                          'greater': lambda: left > right, 'greater_equal': lambda: left >= right,
                          'bitwise_and': lambda: left & right, 'bitwise_or': lambda: left | right}[operation]()
            elif operation == 'negative':
                result = (0 - (args[0] & 0xffffffffffffffff)) if value.dtype in ('int32', 'uint32', 'int64', 'uint64') else -1 * args[0]
            elif operation == 'sigmoid':
                result = 1 / (1 + (-1 * args[0]).exp())
            else:
                result = getattr(args[0], operation)()
            if direct:
                matrix_shape = shape if len(shape) == 2 else (1, math.prod(shape))
                operands = tuple(matrix_view(local[v.index], matrix_shapes[v.index]).broadcast_to(matrix_shape) for v in values)
                tensors[value.index] = nn._pointwise(program, kernels.expression(result),
                    operands, tile_rows, peer=peer, output_dtype=value.dtype)
            else:
                tensors[value.index] = program.kernel_call(kernels.expression(result),
                    grid=tuple((size + extent - 1) // extent for size, extent in zip(matrix_shape, block)),
                    in_specs=(BlockSpec(None),) * len(values),
                    out_specs=BlockSpec(block, lambda i, j: (i, j)),
                    out_shape=ShapeDtypeStruct(matrix_shape, value.dtype), peer=peer)(*(local[v.index] for v in values))
            continue
        text, operations = source(((value, operation, values, attributes, owner),))
        item = operations[0]
        shape = shapes[value.index]
        size = math.prod(shape)
        storage_shape = (max(1, math.prod(shape[:-1])), max(1, shape[-1])) if shape else (1, 1)
        bound = tuple(dict.fromkeys((*values, value)))
        positions = {operand.index: i for i, operand in enumerate(bound)}
        views, blocks, operands, specs = [], [], [], []
        for operand in bound:
            operand_shape = shapes[operand.index]
            tensor = None if operand.index == value.index else local[operand.index]
            matrix_shape = storage_shape if tensor is None else tensor.shape
            block_shape = storage_shape if tensor is None else tensor.block_shape
            grid_columns = 1 if tensor is None else tensor.grid[1]
            view = View(size=math.prod(operand_shape), dtype=np.dtype(operand.dtype).itemsize,
                rank=len(operand_shape), first=len(blocks), matrix_columns=matrix_shape[1],
                block_rows=block_shape[0], block_columns=block_shape[1], grid_columns=grid_columns)
            stride = 1
            for axis in reversed(range(len(operand_shape))):
                view.shape[axis], view.stride[axis] = operand_shape[axis], stride
                stride *= operand_shape[axis]
            if tensor is None:
                blocks.append(Block(storage_shape[1], 1))
            else:
                for coordinate, ref in sorted(tensor.blocks.items()):
                    blocks.append(Block(ref.view.row_stride, ref.view.column_stride))
                    operands.append(tensor)
                    specs.append(BlockSpec(tensor.block_shape, lambda i, coordinate=coordinate: coordinate))
            views.append(view)
        mode, threads = item['mode'], 256
        if mode == 'matmul':
            grid = ((shape[-1] + 31) // 32, (shape[-2] + 63) // 64, math.prod(shape[:-2]))
        elif isinstance(mode, tuple) and mode[0] == 'expert':
            grid = ((shape[-1] + 31) // 32, (shape[-2] + 63) // 64, shapes[mode[1]][0])
        elif mode == 'reduce':
            grid = (size, 1, 1)
        elif isinstance(mode, tuple) and mode[0] in ('neighborhood', 'edges'):
            threads = 32
            grid = (shapes[mode[1]][0] if mode[0] == 'neighborhood' else math.prod(shapes[mode[1]]), 1, 1)
        else:
            grid = (((math.prod(shapes[mode[1]]) if isinstance(mode, tuple) else size) + 255) // 256, 1, 1)
        arguments = [positions[index] for index in item['arguments']] + [positions[value.index]]
        dispatches = []
        if item['clear']:
            dispatches.append(MetalDispatch('mesh_tensor_zero', (max(1, (size * np.dtype(value.dtype).itemsize + 255) // 256), 1, 1),
                                           argument_buffer=4, argument_offset=len(item['arguments']) * 4))
        dispatches.append(MetalDispatch(item['name'], tuple(max(1, v) for v in grid), (threads, 1, 1), argument_buffer=4))
        kernel = Metal(text, tuple(dispatches), (bytes((View * len(views))(*views)),
            bytes((c.c_uint64 * max(1, len(capacity)))(*capacity)), bytes((Block * len(blocks))(*blocks)), bytes((c.c_uint32 * len(arguments))(*arguments))))
        tensors[value.index] = program.kernel_call(kernel, grid=(1,), peer=peer,
            in_specs=tuple(specs),
            out_specs=BlockSpec(storage_shape, lambda i: (0, 0)),
            out_shape=ShapeDtypeStruct(storage_shape, value.dtype))(*operands)
    return tensors
