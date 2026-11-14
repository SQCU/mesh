import ctypes as C, json, re, sys, torch
from mesh_compiler import metal, operations
from mesh_compiler.driver import compile_module

OUT = sys.argv[1]
lib = metal._compiler()
count = lib.metal_ir_operations


class OperationInfo(C.Structure):
    _fields_ = [('identity', C.c_uint64), ('place', C.c_uint32), ('erased', C.c_uint32), ('kind', C.c_uint32),
                ('padding', C.c_uint32)]


info_of = lib.metal_ir_operation_info
info_of.argtypes, info_of.restype = [C.c_void_p, C.c_uint32, C.POINTER(OperationInfo)], None
KINDS = {getattr(operations, n): n for n in ('BOOL', 'I8', 'U8', 'I16', 'U16', 'I32', 'U32', 'I64', 'U64', 'F16', 'BF16',
                                              'F32', 'F64') if isinstance(getattr(operations, n, None), int)}
PLACES = {0: 'GPU', 1: 'CPU', 2: 'ANE', 3: 'MPS'}
records, clones, result = {}, {}, {'construction_succeeded': False}


def address(p):
    return p if isinstance(p, int) else getattr(p, 'value', None)


def tensor(t):
    return {'dtype': str(getattr(t, 'dtype', None)), 'shape': list(getattr(t, 'shape', ()))}


def node_of(n):
    if n is None or not hasattr(n, 'op'):
        return None
    val = n.meta.get('val') if hasattr(n, 'meta') else None
    return {'name': n.name, 'op': n.op, 'target': str(n.target),
            'val': tensor(val) if isinstance(val, torch.Tensor) else repr(val)[:200]}


def bitcasts(g, path='g'):
    found = []
    for attr in ('body', 'combine', 'finalize'):
        region = getattr(g, attr, None)
        if region is None or not hasattr(region, 'code'):
            continue
        for i, record in enumerate(region.code):
            if record[0] != 'bitcast':
                continue
            src = record[2]
            source = region.code[src]
            operands = {j: repr(region.code[j]) for j in source[2:] if isinstance(j, int) and j < i and
                        source[0] in ('binary', 'unary', 'convert', 'bitcast')}
            found.append({'region': path + '.' + attr, 'index': i, 'record': repr(record),
                          'destination': KINDS.get(record[1], record[1]), 'source_index': src,
                          'source_kind': KINDS.get(region.kinds[src], region.kinds[src]), 'source_code': repr(source),
                          'source_operand_codes': operands})
    for k, inner in enumerate(getattr(g, 'records', ()) or ()):
        found += bitcasts(inner, f'{path}.records[{k}]')
    return found


def entry(e):
    g, before, after, returned, caller, owner = e
    return {'interval': [before, after], 'returned': returned, 'class': type(g).__name__,
            'place': PLACES.get(getattr(g, 'place', None), getattr(g, 'place', None)), 'node': getattr(g, 'node', None),
            'caller': caller, 'owner': owner,
            'inputs': [tensor(t) for t, _ in getattr(g, 'inputs', ())],
            'outputs': [tensor(t) for t, _ in getattr(g, 'outputs', ())], 'bitcasts': bitcasts(g)}


def owning(program, op):
    for e in records.get(program, ()):
        g, before, after, returned = e[:4]
        if before <= op < after or (before == after and returned == op):
            return e
    return None


original_generic = metal._Builder.generic


def generic(self, g):
    p = address(self.p)
    before = count(p)
    returned = original_generic(self, g)
    frame = sys._getframe(1)
    owner = frame.f_locals.get('owner')
    copy = frame.f_locals.get('copy')
    described = {'task': repr(owner[0])[:120], 'node': node_of(owner[3])} if isinstance(owner, tuple) and len(owner) == 4 \
        else ({'copy_leaf': node_of(getattr(copy, 'leaf', None))} if copy is not None else None)
    records.setdefault(p, []).append((g, before, count(p), returned, frame.f_code.co_name, described))
    return returned


metal._Builder.generic = generic
original_clone, original_construct, original_lower = lib.metal_ir_clone, lib.metal_ir_construct, lib.metal_ir_lower


def clone(p, error, room):
    made = original_clone(p, error, room)
    clones[address(made)] = address(p)
    return made


class Stop(Exception):
    pass


def construct(p, gpu, error, room):
    status = original_construct(p, gpu, error, room)
    if status and 'failed_operation' not in result:
        p = address(p)
        source = p
        while source in clones:
            source = clones[source]
        text = error.value.decode()
        found = re.search(r'operation (\d+)', text)
        n = count(p)
        infos = []
        for o in range(n):
            x = OperationInfo()
            info_of(p, o, C.byref(x))
            infos.append(x)
        failed = int(found.group(1)) if found else None
        result.update({'status': status, 'error': text, 'failed_operation': failed, 'construct_program': p,
                       'source_program': source, 'operations': n,
                       'failed_info': None if failed is None or failed >= n else
                       {'identity': infos[failed].identity, 'place': PLACES.get(infos[failed].place),
                        'erased': infos[failed].erased, 'kind': infos[failed].kind},
                       'erased_operations': [o for o in range(n) if infos[o].erased]})
        e = owning(source, failed) if failed is not None else None
        result['owner'] = entry(e) if e else None
        seen, erased = set(), []
        for o in result['erased_operations']:
            e = owning(source, o)
            if e is None or id(e[0]) in seen:
                continue
            seen.add(id(e[0]))
            summary = entry(e)
            if summary['bitcasts']:
                erased.append({'erased_operations_in_interval': [x for x in result['erased_operations']
                                                                 if e[1] <= x < e[2]], **summary})
        result['erased_records_with_bitcasts'] = erased
        result['erased_note'] = ('erased after the failed construct; which erased operations were fused into the failed '
                                 'operation is not exposed by a public getter')
    return status


def lower(p, error, room):
    result['construction_succeeded'] = True
    raise Stop('construction succeeded; stopped before lowering')


lib.metal_ir_clone, lib.metal_ir_construct, lib.metal_ir_lower = clone, construct, lower


class Hermitian(torch.nn.Module):
    def forward(self, a):
        return torch.linalg.eigh(a)


a = torch.tensor([[2., 1.], [1., 4.]], dtype=torch.float32)
try:
    compile_module(Hermitian(), (a,), {'p': (0, 'ane')}, lambda node: 'p')
    result['compiled'] = True
except Exception as error:
    result['exception'] = f'{type(error).__name__}: {error}'
with open(OUT, 'w') as f:
    json.dump(result, f, indent=1, default=repr)
print('snapshot', 'failed_operation' in result, result.get('construction_succeeded'))
