# design/compilation-composition-change-request.md items 1, 3, 6 and 7; design/aot-compilation-change-request.md A8 and
# A9: the live native owner. The C ABI's types and library bindings (metal-microbench metal_ir.h, metal_recording.h;
# rdma/mesh-metal.h), the Objective-C/Metal resource primitives, borrowed buffers, archive byte ownership, and the
# typed C encoder (_Builder) of the direct native projections over program values: values and views, a produced
# kernel's operand (metal_ir_upstream), a typed LAPACK call, an INFO check and the canonical logical copy
# (metal_ir_copy); a produced kernel's composed callable collection (native129: metal_ir_upstream_callables,
# metal_ir_callables_emit) and its numerical body's binary64 numeric collection (Binary64)
import collections
import ctypes as C
import ctypes.util
import errno
import os
import pathlib
import sys
import weakref

import torch

from .operations import CPU, GPU

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh
import mesh_graph
from mesh_index import INDEX_LT, INDEX_LE, INDEX_GT, INDEX_GE, INDEX_EQ, INDEX_NE

# metal-microbench metal_ir.h METAL_IR_INPUT .. METAL_IR_STATE (a value's role), its region kinds, UINT32_MAX (no
# value), and METAL_IR_DATA, METAL_IR_TRANSPORT, METAL_IR_STEP (metal_ir_value_info `word` of the step value)
INPUT, CONSTANT, TEMPORARY, OUTPUT, STATE = range(5)
BODY, COMBINE, FINALIZE = range(3)
NONE = 0xFFFFFFFF
METAL_IR_DATA, METAL_IR_TRANSPORT, METAL_IR_STEP, METAL_IR_RANGES, METAL_IR_LITERAL = range(5)
# metal_ir.h METAL_IR_GESDD .. METAL_IR_GELSY (struct metal_ir_lapack_call `routine`; metal-microbench 9404b7b2 GELS 5
# and GELSY 6) and METAL_IR_CHECK_LU .. METAL_IR_CHECK_NATIVE (metal_ir_check `code`)
LAPACK_GESDD, LAPACK_HEEVD, LAPACK_GEEV, LAPACK_GELSD, LAPACK_GELSS, LAPACK_GELS, LAPACK_GELSY = range(7)
CHECK_LU, CHECK_CHOLESKY, CHECK_CONVERGENCE, CHECK_ASSERT, CHECK_NATIVE = range(1, 6)
# metal_recording.h METAL_EXTERNAL_GPU: a callable operand the program uses on the GPU
METAL_EXTERNAL_GPU = 1
# metal_ir.h METAL_IR_SOURCE_VALUE .. METAL_IR_SOURCE_EXTERNAL: a GIVEN cause's operand kind (struct
# metal_ir_source_operand)
SOURCE_VALUE, SOURCE_LITERAL, SOURCE_SENT, SOURCE_RECEIVED, SOURCE_EXTERNAL = range(5)
# metal_recording.h METAL_UNIT_GPU, METAL_UNIT_HOST: an observation unit's kind
METAL_UNIT_GPU, METAL_UNIT_HOST = 0, 1
# metal_ir.h METAL_IR_WORK_ITERATION .. METAL_IR_WORK_TUNED: a retained Work term's kind (struct metal_ir_work_term)
WORK_ITERATION, WORK_POINT, WORK_COMBINE, WORK_MERGE, WORK_FRAGMENT, WORK_TUNED = range(6)
# metal_ir.h METAL_IR_NEURAL: a dispatch's kind (struct metal_ir_dispatch `kind`), a Core ML program
METAL_IR_NEURAL = 8


# objc/message.h objc_msgSend; MTLResource.h -device; NSObject -retain and -release
OBJC = C.CDLL('/usr/lib/libobjc.A.dylib')
OBJC.sel_registerName.restype, OBJC.sel_registerName.argtypes = C.c_void_p, [C.c_char_p]
SEND = C.cast(OBJC.objc_msgSend, C.c_void_p).value
GET = C.CFUNCTYPE(C.c_void_p, C.c_void_p, C.c_void_p)(SEND)
PUT = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)(SEND)
SELECTOR = {name: OBJC.sel_registerName(name) for name in (b'device', b'retain', b'release')}
# MTLResource.h MTLResourceStorageModeShared
SHARED = 0
# MTLResource.h `device`; torch v2.14.0 include/ATen/mps/MPSAllocator.h, MPSHeapAllocatorImpl `m_device`;
# include/ATen/native/mps/OperationUtils.h getMTLBufferStorage; torch/csrc/StorageMethods.cpp THPStorage_dataPtr
DEVICE = []


def _metal_device():
    if not DEVICE:
        probe = torch.empty(1, device='mps')
        device = GET(probe.untyped_storage().data_ptr(), SELECTOR[b'device'])
        if not device:
            raise OSError(errno.ENODEV, 'torch\'s MPS buffer names no MTLDevice (MTLResource.h `device`)')
        DEVICE.append(device)
    return DEVICE[0]


# objc/message.h; metal-microbench metal_recording.h
def _message(restype, *argtypes):
    return C.CFUNCTYPE(restype, C.c_void_p, C.c_void_p, *argtypes)(SEND)


_SEL = OBJC.sel_registerName
OBJECT = _message(C.c_void_p)
VALUE = _message(C.c_uint64)
# MTLDevice.h newBufferWithLength:options:
_NEW_BUFFER = _message(C.c_void_p, C.c_ulong, C.c_ulong)


# metal_ir.h metal_ir_value_data `release`, metal_ir_archive_load `release`, MetalBufferBorrow `release`: a C owner's
# release of the Python object it adopted (_lent)
RELEASE = C.CFUNCTYPE(None, C.c_void_p)


def _array(kind, values):
    values = list(values)
    return (kind * max(len(values), 1))(*values)


# design/aot-compilation-change-request.md A1 and A8; metal_recording.h MetalProgramSave: the C-owned bytes a saver
# writes, copied out and freed
_FREE = C.CDLL(None).free
_FREE.restype, _FREE.argtypes = None, [C.c_void_p]


def _saved_bytes(function, *arguments, diagnostic=True):
    data, size, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(4096)
    status = function(*arguments, C.byref(data), C.byref(size), *([error, len(error)] if diagnostic else []))
    try:
        if status:
            raise OSError(status, error.value.decode() if diagnostic else function.__name__)
        return C.string_at(data, size.value)
    finally:
        _FREE(data)


# CPython C API Py_IncRef, Py_DecRef (Reference Counting); ctypes, Callback functions: one strong reference to `owner`,
# a Python object whose bytes a C owner borrows, lent to `load(release, owner)`, which adopts it where
# `adopted(result)` (on success only) and drops it through _RELEASED_OBJECT (a ctypes callback, which holds the GIL
# while it runs) once its last holder is freed; otherwise the reference is dropped here
_INCREF = C.PYFUNCTYPE(None, C.c_void_p)(('Py_IncRef', C.pythonapi))
_DECREF = C.PYFUNCTYPE(None, C.c_void_p)(('Py_DecRef', C.pythonapi))
_RELEASED_OBJECT = RELEASE(_DECREF)


def _lent(owner, load, adopted):
    _INCREF(id(owner))
    try:
        result = load(_RELEASED_OBJECT, id(owner))
    except BaseException:
        _DECREF(id(owner))
        raise
    if not adopted(result):
        _DECREF(id(owner))
    return result


# metal_ir.h metal_ir_archive_load, metal_ir_archive_free: the archive of `data`, an immutable bytes object whose own
# buffer it borrows (no copy; CPython 3.12 bytes keep it 8-byte aligned at the object's 32nd byte) and, loaded, owns
# (_lent) until the archive and every relation it lent are freed, every loaded program's relations and upstream access
# relations made in `context` (the loading compilation's one rdma/mesh_graph.py Context, which the archive retains).
# The caller's own reference to the archive is released with metal_ir_archive_free
def _archive_loaded(compiler, data, context, device, error):
    import numpy as np

    view = np.frombuffer(data, dtype=np.uint8)
    archive = _lent(view, lambda release, owner: compiler.metal_ir_archive_load(
        context.pointer, view.ctypes.data, view.nbytes, device, release, owner, error, len(error)), bool)
    if not archive:
        raise OSError(errno.EINVAL, 'the saved program: %s' % error.value.decode())
    return archive


# design/aot-compilation-change-request.md, appendix item 7; design/plan-to-bridge-change-request.md R2;
# metal_recording.h MetalBufferBorrow: a shared buffer (+1) borrowing `view`'s bytes. With `owner` (the Python object
# those bytes belong to) one strong reference is lent to the buffer (_lent), whose deallocator drops it. Without one,
# the bytes outlive the buffer by the caller's means (a mapped plan)
def _placed_buffer(view, owner=None):
    lib, device = _recorder()[0], _metal_device()
    buffer = lib.MetalBufferBorrow(device, view.address, view.bytes, RELEASE(), None) if owner is None else _lent(
        owner, lambda release, held: lib.MetalBufferBorrow(device, view.address, view.bytes, release, held), bool)
    if not buffer:
        raise OSError(errno.ENOMEM, 'Metal made no buffer over the program\'s mapping, %d bytes (MTLDevice.h '
                                    'newBufferWithBytesNoCopy)' % view.bytes)
    return buffer


# metal-microbench metal_ir.h struct metal_ir_term, struct metal_ir_expr, struct metal_ir_lapack and struct
# metal_ir_matrix (mesh-graph.h struct mesh_graph_term)
class Term(C.Structure):
    _fields_ = [('dim', mesh.U), ('padding', mesh.U), ('coefficient', C.c_int64), ('divisor', mesh.Q),
                ('modulus', mesh.Q)]


class Expr(C.Structure):
    _fields_ = [('first', mesh.U), ('terms', mesh.U), ('constant', C.c_int64)]


class Materialization(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('operation', 'input', 'value', 'axes', 'sources')] + [
        ('axis', C.POINTER(mesh.U)), ('source', C.POINTER(mesh.U))]


class RewriteResult(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('choices', 'operations', 'values', 'groups')] + [
        ('survivor', C.POINTER(mesh.U)), ('group_first', C.POINTER(C.c_size_t)),
        ('group_member', C.POINTER(mesh.U)), ('materializations', C.c_size_t),
        ('materialization', C.POINTER(Materialization)), ('bytes', C.POINTER(mesh.Q)), ('accesses', C.c_void_p)]


INLINE_MAPS, FUSE_EPILOGUE, FUSE_ROWS, ALLOW_RECOMPUTE, ALLOW_COSTLIER_GROUP = 1, 2, 4, 8, 16
ALL_FUSIONS = INLINE_MAPS | FUSE_EPILOGUE | FUSE_ROWS


class CallIR:
    def __init__(self, template, create=None):
        compiler = _compiler()
        create = compiler.metal_ir_specialize if create is None else create
        error = C.create_string_buffer(4096)
        self.program = create(template, error, len(error))
        if not self.program:
            raise ValueError(error.value.decode())
        self.close = weakref.finalize(self, compiler.metal_ir_program_free, self.program)

    def lower(self):
        compiler = _compiler()
        error = C.create_string_buffer(4096)
        if compiler.metal_ir_lower(self.program, error, len(error)):
            raise ValueError(error.value.decode())
        self.relation = compiler.metal_ir_relation(self.program)
        dispatch, storage = C.POINTER(Dispatch)(), C.POINTER(Storage)()
        dispatches = compiler.metal_ir_dispatches(self.program, C.byref(dispatch))
        storages = compiler.metal_ir_storages(self.program, C.byref(storage))
        count = C.c_size_t()
        allocations = compiler.metal_ir_allocations(self.program, C.byref(count))
        for name, pointer, type_, size in (('dispatches', dispatch, Dispatch, dispatches),
                                            ('storages', storage, Storage, storages),
                                            ('allocations', allocations, mesh.U, count.value)):
            table = C.cast(pointer, C.POINTER(type_ * size)).contents if size else ()
            if size:
                table.owner = self
            setattr(self, name, table)


class Rewrite:
    def __init__(self, source, device, choices):
        lib = _compiler()
        error = C.create_string_buffer(4096)
        self.program = lib.metal_ir_rewrite(source, device, choices, error, len(error))
        if not self.program:
            raise ValueError(error.value.decode())
        self.close = weakref.finalize(self, lib.metal_ir_program_free, self.program)
        self.result = lib.metal_ir_rewritten(self.program).contents


class Lapack(C.Structure):
    _fields_ = [('routine', mesh.U), ('batch', mesh.U), ('job', C.c_char * 8), ('rcond', C.c_double)]


class Matrix(C.Structure):
    _fields_ = [('row', mesh.U), ('column', mesh.U), ('part', mesh.U)]


# metal_recording.h struct MetalReplay; struct MetalCall: the row's plan call `call` - 1 (mesh-collective.h
# planned->call), whose `ordinal` of its function's `every` calls in a block runs plan execution b every + ordinal at
# block execution b; `native` and `residues` its compiled CPU entries
class Replay(C.Structure):
    _fields_ = [('commands', C.c_void_p), ('residency', C.c_void_p), ('owner', C.c_void_p), ('count', mesh.U),
                ('buffers', C.c_uint16), ('memory', C.c_uint16)]


class Call(C.Structure):
    _fields_ = [('replay', C.POINTER(Replay)), ('call', mesh.Q), ('native', C.c_void_p), ('residues', mesh.Q)]


# metal_ir.h struct metal_ir_span
class Span(C.Structure):
    _fields_ = [('data', C.c_void_p), ('bytes', mesh.Q)]


# rdma/mesh-metal.h struct mesh_metal_transport, struct mesh_metal_input (metal_recording.h struct MetalRemote), struct
# mesh_metal_sent (metal_recording.h struct MetalSent), mesh-graph.h struct mesh_graph_generation, struct
# mesh_metal_range, struct mesh_metal_release and struct mesh_metal_sending
class Transport(C.Structure):
    _fields_ = [('state', C.c_void_p), ('offset', mesh.Q), ('bytes', mesh.Q)]


class Remote(C.Structure):
    _fields_ = [('phases', mesh.Q), ('group', mesh.Q), ('count', mesh.Q), ('pages', mesh.U), ('copies', mesh.U),
                ('time', C.c_double)]


METAL_IR_COUNTER_ABSOLUTE, METAL_IR_COUNTER_OFFSET, METAL_IR_COUNTER_OPERAND = range(3)


class MetalSent(C.Structure):
    _fields_ = [('counter', C.c_void_p), ('sends', mesh.Q), ('copies', mesh.Q)]


class Generated(C.Structure):
    _fields_ = [('every', mesh.U), ('producer', mesh.U), ('ordinal', mesh.U)]


class Range(C.Structure):
    _fields_ = [('input', Remote), ('generation', Generated), ('partitions', mesh.U), ('page', mesh.Q),
                ('field', C.POINTER(mesh.U)), ('at', mesh.Q), ('transfer', mesh.U), ('owner', mesh.U),
                ('extent', C.POINTER(mesh_graph.Access)), ('extents', C.c_size_t)]


# mesh_metal_release: one release record of a value, declared {transfer, owner} (its group's transfer and owner call),
# prepared in place (supplied, status); 32 bytes
class Release(C.Structure):
    _fields_ = [('supplied', mesh.Q), ('status', mesh.Q), ('total', mesh.Q), ('transfer', mesh.U), ('owner', mesh.U)]


class Origin(C.Structure):
    _fields_ = [('word', mesh.Q), ('copies', mesh.U), ('cuts', mesh.U)]


class Sending(C.Structure):
    _fields_ = [('sent', MetalSent), ('extent', C.POINTER(mesh_graph.Access)), ('extents', C.c_size_t),
                ('transfer', mesh.U), ('padding', mesh.U)]


# metal_recording.h struct MetalCheckFailure
class CheckFailure(C.Structure):
    _fields_ = [('identity', mesh.Q), ('batch', mesh.Q), ('info', C.c_int64), ('code', mesh.U), ('padding', mesh.U)]


# metal_ir.h struct metal_ir_value_info (`received`, `sent` and `sendings` record counts: metal_ir_value_ranges,
# metal_ir_value_releases, metal_ir_value_sendings; `page_bytes` at 104, `owned` its storage root's Value::owned at
# 120, 128 bytes) and struct metal_ir_storage
class ValueInfo(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('role', 'scalar', 'storage', 'function', 'index', 'word', 'of', 'range',
                                          'received', 'sent', 'zeroed', 'sendings')] + [
        (name, mesh.Q) for name in ('identity', 'bytes', 'offset', 'at', 'data_bytes')] + [
        ('data', C.c_void_p), ('buffer', C.c_void_p), ('page_bytes', mesh.Q), ('pages', mesh.U), ('source', mesh.U),
        ('owned', mesh.U), ('padding', mesh.U)]


class Storage(C.Structure):
    _fields_ = [('bytes', mesh.Q), ('begin', mesh.U), ('end', mesh.U), ('alignment', mesh.Q)]


# metal_recording.h struct MetalActors (MetalRecorderActors)
class _Actors(C.Structure):
    _fields_ = [('parts', mesh.U), ('host', C.POINTER(C.c_double)), ('work', C.POINTER(C.c_double)),
                ('edges', C.c_size_t), ('edge', C.POINTER(mesh_graph.Dependence)), ('relation', C.c_void_p)]


# metal_ir.h struct metal_ir_source_operand: a GIVEN cause's typed operand (`kind` a SOURCE_* code, `value` its ValueID)
class SourceOperand(C.Structure):
    _fields_ = [('kind', mesh.U), ('value', mesh.U)]


# metal_recording.h struct MetalUnit, MetalSlice and MetalObserved (the observation entry): a compiled unit (its kind,
# call, native site, residue count `copies`, its residues' records from `slice`), residue e's member dispatches and
# source identities, and one invocation's observation of a unit
class Unit(C.Structure):
    _fields_ = [('kind', mesh.U), ('call', mesh.U), ('site', mesh.U), ('copies', mesh.U), ('slice', mesh.Q)]


class UnitSlice(C.Structure):
    _fields_ = [('member', mesh.Q), ('count', mesh.Q), ('identity', mesh.Q), ('identities', mesh.Q)]


class Observed(C.Structure):
    _fields_ = [('seconds', C.c_double), ('executions', mesh.Q), ('residue', mesh.Q)]


# metal_recording.h struct MetalArgument (an invocation operand) and struct MetalExternalUse (an operand's uses)
class Argument(C.Structure):
    _fields_ = [('buffer', C.c_void_p), ('offset', mesh.Q), ('base', C.c_void_p), ('handle', C.c_void_p)]


class ExternalUse(C.Structure):
    _fields_ = [('argument', mesh.U), ('row', mesh.U), ('value', mesh.U), ('uses', mesh.U)]


# metal_ir.h struct metal_ir_work_axis, struct metal_ir_work_extent and struct metal_ir_work_term (metal_ir_work_terms,
# metal_ir_operation_axis): a source association (source task, original full extent, typed value and dimension); a term
# domain's extent (its compiled bound, its loop, its source associations axes[axis .. axis + axes)); one retained
# arithmetic Work term of numeric kernel member (kernel, member, phase), `coefficient` times the product of
# factors[first .. first + factors), its source operations origins[origin .. origin + origins); 24, 24 and
# sizeof(mesh_graph_work) + 48 bytes
class WorkAxis(C.Structure):
    _fields_ = [('task', mesh.Q), ('full', mesh.Q), ('value', mesh.U), ('dimension', mesh.U)]


class WorkExtent(C.Structure):
    _fields_ = [('extent', mesh.Q), ('loop', mesh.U), ('axis', mesh.U), ('axes', mesh.U), ('padding', mesh.U)]


class WorkTerm(C.Structure):
    _fields_ = [('coefficient', mesh_graph.Work), ('kernel', mesh.U), ('member', mesh.U), ('phase', mesh.U),
                ('operation', mesh.U), ('kind', mesh.U), ('domain', mesh.U), ('extents', mesh.U), ('first', mesh.U),
                ('factors', mesh.U), ('origin', mesh.U), ('origins', mesh.U), ('padding', mesh.U)]


# metal_ir.h struct metal_ir_work_association (metal_ir_work_associations): one source association of a surviving
# operation (task, full; containing operation, original operation and operand ordinal, typed value and dimension, its
# Unit loop, NONE for none); 40 bytes
class WorkAssociation(C.Structure):
    _fields_ = [('task', mesh.Q), ('full', mesh.Q)] + [(name, mesh.U) for name in (
        'operation', 'source', 'operand', 'value', 'dimension', 'loop')]


# metal_ir.h struct metal_ir_numeric (metal_ir_dispatch_numerics): a dispatch's numeric kernel, member range [first,
# last) and later phase (kernel NONE: no numerical term); 16 bytes
class Numeric(C.Structure):
    _fields_ = [('kernel', mesh.U), ('first', mesh.U), ('last', mesh.U), ('later', mesh.U)]


# metal_ir.h struct metal_ir_dispatch (metal_ir_dispatches): one dispatch of a program, its kind and kernel; 80 bytes
class Dispatch(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('kind', 'kernel', 'binding', 'bindings', 'effect', 'effects', 'source',
                                            'sources', 'allocation', 'allocations', 'arguments', 'later', 'flags',
                                            'fill')] + [('groups', mesh.U * 3), ('threads', mesh.U * 3)]


# metal_ir.h struct metal_ir_repeat_region (metal_ir_repeats): one numerical repeat region, its operation interval
# [first, end), condition, enclosing region `parent` of kind `parent_kind` and its branch `parent_branch` (UINT32_MAX
# under a repetition or none), carried roots metal_ir_repeat_carried()[carried .. + carried_count), and as produced by
# metal_ir_lower its test dispatch and inclusive metal_ir_order span; 44 bytes
class RepeatRegion(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('first', 'end', 'condition', 'parent', 'carried', 'carried_count', 'test',
                                            'order_first', 'order_last', 'parent_kind', 'parent_branch')]


# metal_ir.h struct metal_ir_switch_region (metal_ir_switches): one indexed switch as registered, its `index` ValueID,
# its branches' {first, end} intervals, yields and common results as offsets into metal_ir_switch_values, its enclosing
# region, its branch and its kind, and as produced by metal_ir_lower its selector dispatch and inclusive metal_ir_order
# span; 48 bytes
class SwitchRegion(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('index', 'branches', 'branch', 'outputs', 'yielded', 'results', 'parent',
                                            'parent_branch', 'select', 'order_first', 'order_last', 'parent_kind')]


# metal_ir.h METAL_IR_REGION_REPEAT, METAL_IR_REGION_SWITCH and struct metal_ir_region_child, the direct-child input of
# metal_ir_repeat and metal_ir_switch: an already registered region of that kind (its metal_ir_repeats or
# metal_ir_switches index) and the enclosing arm it is in (a switch's branch; UINT32_MAX in a repetition); 12 bytes
REGION_REPEAT, REGION_SWITCH = 1, 2


class RegionChild(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('kind', 'region', 'arm')]


# metal_ir.h metal_ir_operations: one direct projection's actual emission by _Builder: `record` the producer's own
# record it implements (lower_inductor.py Kernel, ExternCall, a control body's Inlined, a Neural Engine body's
# Produced; a machine body node for an interface copy), `operation` its returned operation ID, [first, end) the
# operations it created
Emitted = collections.namedtuple("Emitted", "record operation first end")
# metal_ir.h metal_ir_repeat and metal_ir_switch: one registered region of _Builder, `record` the producer's own
# control record (lower_inductor.py ExternCall: an ir.WhileLoop's, an ir.Switch's), `kind` REGION_REPEAT or
# REGION_SWITCH, `region` its metal_ir_repeats or metal_ir_switches index, `first`/`end` its pass's or each branch's
# operation interval, `selector` a switch's selector operation (NONE a repeat's)
Region = collections.namedtuple("Region", "record kind region first end selector")


# metal-microbench metal_ir.h metal_ir_upstream and its stages: METAL_IR_UPSTREAM_BUFFER .. _ERROR (an argument's
# kind), struct metal_ir_upstream_argument and struct metal_ir_upstream (a produced kernel's operand before
# metal_ir_construct), what metal_ir_upstream_composition returns (struct metal_ir_upstream_word, _received, _sent,
# _page, _readiness, _table, _composed), the composed callable collection's printed source (struct
# metal_ir_callable_source), a pass's terminal emission (struct
# metal_ir_upstream_emission) and the first pass's configured launch (struct metal_ir_upstream_configuration)
UPSTREAM_BUFFER, UPSTREAM_SCALAR, UPSTREAM_ERROR = range(3)


class UpstreamArgument(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('kind', 'value', 'access', 'address_access')] + [('alignment', mesh.Q)]


# struct metal_ir_upstream_access: one access as its producer made it, `argument` its launch argument, `write` 0 a
# read, 1 a write, 2 a commuting update; `index` its element index from that argument binding's first element (a
# mesh-graph.h struct mesh_graph_index over the kernel's domain, NULL not quasi-affine: then [lo, hi] its value range),
# `guard` its predicate (NULL unguarded), `guard_data` 1 where that predicate reads data, `guard_source` its identity
class UpstreamAccess(C.Structure):
    _fields_ = [('argument', mesh.U), ('write', mesh.U), ('index', C.c_void_p), ('guard', C.c_void_p),
                ('guard_source', mesh.Q), ('guard_data', mesh.U), ('padding', mesh.U), ('lo', C.c_int64),
                ('hi', C.c_int64)]


# struct metal_ir_upstream (metal-microbench metal_ir.h as of 3330ed9b and its working binding_group; the next matched
# build's, not native137's): `domain` the kernel's index domain (mesh-graph.h struct mesh_graph_domain, in the
# compilation's one analysis context; native retains it), `access` its accesses, `work` its semantic work (NULL
# unknown); `unit` its producer's compilation unit (0 none); `module`, `entry`, `binding` and `binding_group` a typed
# body's original parsed module (a WGPUShaderModule), its entry and each argument's @binding and @group, `target` its
# original target options (tint::msl::writer::Options), `image` the LC_UUID of the loaded image its code is relative
# to: each NULL for an Inductor kernel, whose domain, accesses and terminal emission are its own
class Upstream(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('place', 'arguments', 'accesses', 'sources')] + [
        ('argument', C.POINTER(UpstreamArgument)), ('domain', C.c_void_p), ('access', C.POINTER(UpstreamAccess)),
        ('source', C.POINTER(mesh.Q)), ('work', C.POINTER(mesh_graph.Work)), ('identity', mesh.Q),
        ('grid', mesh.U * 3), ('group', mesh.U * 3), ('grouped', mesh.U), ('unit', mesh.U),
        ('module', C.c_void_p), ('entry', C.c_char_p), ('binding', C.POINTER(mesh.U)), ('target', C.c_void_p),
        ('image', C.POINTER(C.c_uint8)), ('binding_group', C.POINTER(mesh.U))]


# struct metal_ir_upstream_feature, struct metal_ir_upstream_state and struct metal_ir_upstream_neural
# (metal-microbench 51988113: 16, 16 and 64 bytes): a Neural Engine body's original Core ML program, `package` the
# directory its MLModel's own save wrote, `function` its function name, `compute_units` the conversion's selected units
# by name (ct.ComputeUnit .name), each model input and output feature by name bound to its metal_ir_upstream argument,
# each state (MLState) by name to the argument its read (read_state) views and the argument its update
# (coreml_update_state) writes, the same argument where the update is in place
class UpstreamFeature(C.Structure):
    _fields_ = [('name', C.c_char_p), ('argument', mesh.U), ('padding', mesh.U)]


class UpstreamState(C.Structure):
    _fields_ = [('name', C.c_char_p), ('read', mesh.U), ('written', mesh.U)]


class UpstreamNeural(C.Structure):
    _fields_ = [('package', C.c_char_p), ('function', C.c_char_p), ('compute_units', C.c_char_p)] + [
        (name, mesh.U) for name in ('inputs', 'outputs', 'states', 'padding')] + [
        (name, C.POINTER(UpstreamFeature)) for name in ('input', 'output')] + [('state', C.POINTER(UpstreamState))]


# rdma/mesh-graph.h MESH_GRAPH_DOMAIN_ROOT .. _LOOP and struct mesh_graph_domain_dim: one dimension of a kernel's index
# domain, an original symbol over [lo, hi] (ROOT: `axis` its active_range_trees() position, `reduction` its
# is_reduction; INDIRECT an indirect symbol; LOOP a CPU loop variable), and MESH_GRAPH_INDEX_LT .. _NE
DOMAIN_ROOT, DOMAIN_INDIRECT, DOMAIN_LOOP = range(3)


class DomainDim(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('kind', 'axis', 'reduction', 'padding')] + [
        ('lo', C.c_int64), ('hi', C.c_int64)]


class UpstreamWord(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('word', 'slot', 'kind', 'write')]


class UpstreamReceived(C.Structure):
    _fields_ = [('value', mesh.U), ('range', mesh.U), ('record', C.POINTER(Range))]


class UpstreamSent(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('value', 'counter', 'slot', 'releases')] + [
        ('release', C.POINTER(Release))]


class UpstreamPage(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('argument', 'slot', 'frames', 'pages')] + [
        ('page_bytes', mesh.Q), ('origin', mesh.Q)]


class UpstreamReadiness(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('classes', 'members', 'class_slot', 'member_slot', 'later_class_slot',
                                            'later_member_slot', 'ranges', 'padding')]


class UpstreamTable(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('value', 'slot', 'later_slot', 'words', 'ranges', 'releases',
                                            'origin_words', 'padding')]


class UpstreamComposed(C.Structure):
    _fields_ = [('word', C.POINTER(UpstreamWord)), ('later_word', C.POINTER(UpstreamWord)),
                ('received', C.POINTER(UpstreamReceived)), ('sent', C.POINTER(UpstreamSent)),
                ('page', C.POINTER(UpstreamPage)), ('readiness', C.POINTER(UpstreamReadiness)),
                ('table', C.POINTER(UpstreamTable))] + [
        (name, mesh.U) for name in ('words', 'later_words', 'receives', 'sends', 'pages', 'padding')]


# struct metal_ir_callable_source (metal_ir_callables_emit)
class CallableSource(C.Structure):
    _fields_ = [('source', C.c_void_p), ('notice', C.c_void_p), ('bytes', mesh.Q), ('notice_bytes', mesh.Q)]


# metal_ir.h METAL_IR_BINARY64_BINARY .. _UNARY, struct metal_ir_binary64_primitive (a request: `code` a
# metal_ir_binary, metal_ir_unary or metal_ir_builtin_named code, a CONVERT's scalars `from` and `to`) and struct
# metal_ir_binary64_function (the library function serving it: its name, parameter kinds, result kind, GPU work)
BINARY64_BINARY, BINARY64_BUILTIN, BINARY64_CONVERT, BINARY64_UNARY = range(4)


class Binary64Primitive(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('kind', 'code', 'from_', 'to')]


class Binary64Function(C.Structure):
    _fields_ = [('name', C.c_char_p), ('parameter', C.POINTER(mesh.U)), ('parameters', mesh.U), ('result', mesh.U),
                ('work', mesh_graph.Work)]


# struct metal_ir_upstream_parameter: one parameter of a CPU pass's final entry, its C type as its emitter printed it,
# the kernel binding it receives (an argument index or a composition slot), 1 for the bound pointer, 0 its pointee
class UpstreamParameter(C.Structure):
    _fields_ = [('type', C.c_char_p), ('slot', mesh.U), ('pointer', mesh.U)]


# metal_ir.h metal_ir_build_action, native140: the original compiler action's ordered operands and context
class BuildAction(C.Structure):
    _fields_ = [(name, C.POINTER(C.c_char_p)) for name in ('command', 'directories', 'environment')] + [
        (name, C.POINTER(mesh.U)) for name in ('sources', 'target', 'ldflags', 'libraries', 'library_directories')
    ] + [(name, C.c_char_p) for name in ('product', 'target_file', 'precompiled_header', 'working_directory',
                                        'executable')] + [(name, mesh.U) for name in (
        'commands', 'source_count', 'target_count', 'ldflag_count', 'library_count', 'library_directory_count',
        'directory_count', 'environments', 'relative', 'padding')]


# struct metal_ir_upstream_emission (metal-microbench 6a48588d metal_ir.h, native140's build): `language` a CPU
# source's language; `environment` its compile command's whole environment ("NAME=value" each), `directory` its
# working directory, `compile_executable` the executable its compile runs (compile[0] as its producer's subprocess
# resolves it), `compile_source` and `compile_output` the positions in `compile` of its own source file and output
# object; `link_action` its original complete linking action. NULL and 0 where a pass has no action of its own
class UpstreamEmission(C.Structure):
    _fields_ = [('source', C.c_char_p), ('entry', C.c_char_p), ('parameter', C.POINTER(UpstreamParameter)),
                ('compile', C.POINTER(C.c_char_p)), ('link', C.POINTER(C.c_char_p)), ('bytes', mesh.Q),
                ('parameters', mesh.U), ('compiles', mesh.U), ('links', mesh.U), ('language', C.c_char_p),
                ('environment', C.POINTER(C.c_char_p)), ('directory', C.c_char_p), ('compile_executable', C.c_char_p),
                ('link_action', C.POINTER(BuildAction)),
                ('environments', mesh.U), ('compile_source', mesh.U), ('compile_output', mesh.U), ('padding', mesh.U)]


class UpstreamConfiguration(C.Structure):
    _fields_ = [(name, mesh.U * 3) for name in ('grid', 'group', 'groups')] + [('padding', mesh.U)]


def _view(kind, pointer, count):
    return C.cast(pointer, C.POINTER(kind * count)).contents if count else (kind * 0)()


# metal_ir.h metal_ir_upstream_composition `composed`, borrowed from `program` (native.py Target.instanced's Owned scope)
class Composed:
    def __init__(self, program, op, composed, callables, later):
        self.program, self.op, self.composed, self.callables, self.later = program, op, composed, callables, later

    words = property(lambda self: _view(UpstreamWord, self.composed.word, self.composed.words))
    later_words = property(lambda self: _view(UpstreamWord, self.composed.later_word, self.composed.later_words))
    received = property(lambda self: _view(UpstreamReceived, self.composed.received, self.composed.receives))
    sent = property(lambda self: _view(UpstreamSent, self.composed.sent, self.composed.sends))
    pages = property(lambda self: _view(UpstreamPage, self.composed.page, self.composed.pages))
    readiness = property(lambda self: self.composed.readiness.contents if self.composed.readiness else None)
    table = property(lambda self: self.composed.table.contents if self.composed.table else None)


# metal_ir.h METAL_IR_CALLABLE_ACQUIRE .. _OTHER, METAL_IR_CALLABLE_OPERAND_VIEW .. _FRAMES
CALLABLE_ACQUIRE, CALLABLE_LATER, CALLABLE_RELEASE, CALLABLE_LOAD, CALLABLE_STORE, CALLABLE_OTHER = range(6)
(OPERAND_VIEW, OPERAND_CLASSES, OPERAND_MEMBERS, OPERAND_RECORDS, OPERAND_VOTES, OPERAND_LID, OPERAND_GID, OPERAND_LANE,
 OPERAND_THREADS, OPERAND_SUBSCRIPT, OPERAND_FRAMES) = range(11)


# metal_ir.h metal_ir_upstream_callables `handle` and metal_ir_callables_emit `emitted`, borrowed as its composition
class Callables:
    def __init__(self, program, handle, emitted):
        self.program, self.handle, self.emitted = program, handle, emitted

    output = property(lambda self: _upstream().metal_ir_callables_output(self.handle))
    printed = property(lambda self: (C.c_char * self.emitted.bytes).from_address(self.emitted.source))
    notices = property(lambda self: (C.c_char * self.emitted.notice_bytes).from_address(self.emitted.notice)
                       if self.emitted.notice else (C.c_char * 0)())
    source = property(lambda self: (bytes(self.notices) + bytes(self.printed)).decode())
    notice = property(lambda self: bytes(self.notices).decode())

    module = property(lambda self: _upstream().metal_ir_callables_module(self.handle))

    def function(self, *args):
        return _upstream().metal_ir_callables_function(self.handle, *args)

    def name(self, role):
        function = self.function(role)
        if function:
            lib = _upstream()
            size = C.c_size_t()
            data = lib.metal_ir_callables_function_name(self.handle, function, C.byref(size))
            return C.string_at(data, size.value).decode()

    def role(self, *args):
        return _upstream().metal_ir_callables_role(self.handle, *args)

    def address(self, *args):
        return _upstream().metal_ir_callables_address(self.handle, *args)

    def operand(self, *args):
        return _upstream().metal_ir_callables_operand(self.handle, *args)


# dlfcn.h dladdr; metal-microbench metal_recording.h: the recorder (libmetal_recording) and the checkout it was built
# in, whose .build holds libmetal_ir and libmesh_metal
RECORDER = []


class _LibraryInfo(C.Structure):
    _fields_ = [('path', C.c_char_p), ('base', C.c_void_p), ('symbol', C.c_char_p), ('address', C.c_void_p)]


def _repository():
    info = _LibraryInfo()
    locate = C.CDLL(None).dladdr
    locate.argtypes, locate.restype = [C.c_void_p, C.POINTER(_LibraryInfo)], C.c_int
    if not locate(_recorder()[0].MetalRecorderOpen, C.byref(info)):
        raise OSError(errno.ENOENT, 'the loaded recorder has no library path')
    return os.path.dirname(os.path.dirname(os.path.realpath(os.fsdecode(info.path))))


def _recorder():
    if not RECORDER:
        path = next((p for p in os.environ.get('DYLD_INSERT_LIBRARIES', '').split(':')
                     if os.path.basename(p) == 'libmetal_recording.dylib'), None) \
            or ctypes.util.find_library('metal_recording') or 'libmetal_recording.dylib'
        lib, metal = C.CDLL(path), C.CDLL('/System/Library/Frameworks/Metal.framework/Metal')
        P, U, Q, text, room = C.c_void_p, mesh.U, mesh.Q, C.c_char_p, C.c_size_t
        machine = C.POINTER(mesh_graph.Machine)
        for name, result, arguments in (
                ('MetalRecorderOpen', P, [P, text, room]),
                ('MetalBufferBorrow', P, [P, P, Q, RELEASE, P]),
                ('MetalRecorderPrepare', P, [P, P, text, room]),
                ('MetalRecorderBindings', P, [P]),
                ('MetalRecorderReplay', C.c_int, [P, P, C.POINTER(Replay), text, room]),
                ('MetalCaptureCounters', C.POINTER(Q), [P, C.c_size_t]),
                ('MetalRecorderState', C.c_int, [P, P, P, Q, text, room]),
                ('MetalReplayDestroy', None, [C.POINTER(Replay)]),
                ('MetalRecorderActors', C.c_int, [P, machine, machine, C.POINTER(_Actors)]),
                ('MetalRecorderActorsFree', None, [C.POINTER(_Actors)]),
                ('MetalRecorderProgram', P, [P, C.POINTER(Call), U, C.POINTER(mesh.Planned), C.POINTER(mesh.PlanBinding), text, room]),
                ('MetalProgramFunctions', P, []),
                ('MetalRecorderProgramWithFunctions', P, [P, C.POINTER(Call), U, C.POINTER(mesh.Planned), C.POINTER(mesh.PlanBinding), P, P, text, room]),
                ('MetalRecorderProgramScheduled', P, [P, C.POINTER(Call), U, C.POINTER(mesh.Planned), C.POINTER(mesh.PlanBinding), P, text, room]),
                ('MetalProgramSave', C.c_int, [P, C.POINTER(Span), U, C.POINTER(P), C.POINTER(C.c_size_t), text,
                                               room]),
                ('MetalProgramLoad', P, [P, C.POINTER(Call), U, C.POINTER(mesh.Planned), C.POINTER(mesh.PlanBinding), P, text, room]),
                ('MetalProgramPhases', None, [P, C.POINTER(Q), C.POINTER(Q)]),
                ('MetalProgramIssued', Q, [P]),
                ('MetalProgramSignatureNode', C.c_int, [P, U, C.c_size_t, C.POINTER(Q),
                                                        C.POINTER(mesh_graph.Signature), text, room]),
                ('MetalProgramSignatureBind', C.c_int, [P, U, C.POINTER(mesh_graph.Signature), text, room]),
                ('MetalProgramExternals', C.POINTER(ExternalUse), [P, C.POINTER(U)]),
                ('MetalProgramRunArguments', C.c_int, [P, P, Q, Q, P, P, P, C.POINTER(Argument), U, text, room]),
                ('MetalProgramRunObserved', C.c_int, [P, P, Q, Q, P, P, P, C.POINTER(Argument), U, text, room]),
                ('MetalProgramUnits', C.POINTER(Unit), [P, C.POINTER(U), C.POINTER(C.POINTER(UnitSlice)),
                                                        C.POINTER(C.POINTER(U)), C.POINTER(C.POINTER(Q))]),
                ('MetalProgramObserved', C.c_int, [P, C.POINTER(Observed), U, text, room]),
                ('MetalProgramEnd', C.c_int, [P, P, text, room]),
                ('MetalProgramCheckFailure', C.c_int, [P, C.POINTER(CheckFailure)])):
            function = getattr(lib, name)
            function.restype, function.argtypes = result, arguments
        metal.MTLCreateSystemDefaultDevice.restype = C.c_void_p
        RECORDER.append((lib, metal.MTLCreateSystemDefaultDevice))
    return RECORDER[0]


# rdma/mesh-metal.h: the transport's records in two stages, declared from the program before mesh_plan
# (mesh_metal_range_declare) and prepared from its results (mesh_metal_range_prepare, mesh_metal_release_prepare,
# mesh_metal_sending_prepare, mesh_metal_status_prepare), and this node's machines (mesh_metal_machines)
TRANSPORT = []


def _transport():
    if not TRANSPORT:
        lib = C.CDLL(os.path.join(_repository(), '.build', 'libmesh_metal.dylib'))
        binding, transport = C.POINTER(mesh.PlanBinding), C.POINTER(Transport)
        for name, result, arguments in (
                ('mesh_metal_transport_create', C.c_int, [mesh.CONTEXT, binding, C.c_void_p, transport]),
                ('mesh_metal_transport_destroy', None, [transport]),
                ('mesh_metal_input_words', None, [C.c_void_p, C.c_void_p, mesh.U, C.c_int64, C.POINTER(Remote)]),
                ('mesh_metal_release_words', None, [C.c_void_p, C.c_void_p, C.c_int64, C.POINTER(Release),
                                                    C.POINTER(Origin), C.POINTER(mesh.Q)]),
                ('mesh_metal_sending_rows', C.c_int, [C.c_void_p, C.c_void_p, C.POINTER(mesh.Planned), mesh.U,
                                                       mesh.U, mesh.Q, C.POINTER(Sending)]),
                ('mesh_metal_range_declare', C.c_int, [mesh.CONTEXT, C.POINTER(mesh.U), mesh.U, mesh.U, mesh.U, mesh.U,
                                                       mesh.Q, mesh.Q, C.POINTER(Range)]),
                ('mesh_metal_range_prepare', C.c_int, [mesh.CONTEXT, transport, binding, mesh.Section,
                                                       C.POINTER(mesh.U), mesh.U, C.POINTER(Range)]),
                ('mesh_metal_range_fields', C.c_int, [C.POINTER(mesh.Planned), C.POINTER(Range),
                                                    C.POINTER(mesh.U), mesh.U]),
                ('mesh_metal_release_prepare', C.c_int, [mesh.CONTEXT, transport, binding, mesh.U,
                                                         C.POINTER(Release)]),
                ('mesh_metal_origin_prepare', C.c_int, [mesh.CONTEXT, transport, binding, mesh.U,
                                                        C.POINTER(Release), C.POINTER(mesh.Q)]),
                ('mesh_metal_origin_slots_prepare', C.c_int, [mesh.CONTEXT, transport, binding, mesh.U,
                                                              C.POINTER(Release), C.POINTER(Origin)]),
                ('mesh_metal_sending_prepare', C.c_int, [mesh.CONTEXT, transport, binding, mesh.U, mesh.U,
                                                         C.POINTER(Sending)]),
                ('mesh_metal_status_prepare', C.c_int, [mesh.CONTEXT, transport, mesh.Q, C.POINTER(mesh.Q)]),
                ('mesh_metal_machines', C.c_int, [C.c_char_p, C.POINTER(mesh_graph.Machine),
                                                  C.POINTER(mesh_graph.Machine)])):
            function = getattr(lib, name)
            function.restype, function.argtypes = result, arguments
        TRANSPORT.append(lib)
    return TRANSPORT[0]


# Roofline (S. Williams, A. Waterman and D. Patterson, CACM 52(4), 2009, §3) operands of this node's resources, the
# configuration's record (rdma/mesh-metal.h mesh_metal_machines, no path: the etc/mesh-capacity.json of the mesh
# checkout whose libmesh the process loaded), read once; ENOENT: no record of this node's model, so no estimate
MACHINES = []


def _machines():
    if not MACHINES:
        gpu, cpu = mesh_graph.Machine(), mesh_graph.Machine()
        status = _transport().mesh_metal_machines(None, C.byref(gpu), C.byref(cpu))
        if status not in (0, errno.ENOENT):
            mesh.check(status)
        MACHINES.append({GPU: gpu, CPU: cpu} if status == 0 else {})
    return MACHINES[0]


# metal-microbench metal_ir.h, its C ABI over .build/libmetal_ir.dylib: the calls of construction (_Builder), lowering,
# specialization, placement, finalization and archive loading
COMPILER = []


def _compiler():
    if not COMPILER:
        lib = C.CDLL(os.path.join(_repository(), '.build', 'libmetal_ir.dylib'))
        P, U, Q = C.c_void_p, mesh.U, mesh.Q
        for name, result, arguments in (
                ('metal_ir_program_new', P, []), ('metal_ir_program_free', None, [P]),
                ('metal_ir_append', C.c_int, [P, P, C.POINTER(U), C.c_size_t, C.c_char_p, C.c_size_t]),
                ('metal_ir_values', U, [P]), ('metal_ir_operations', U, [P]),
                ('metal_ir_switch', U, [P, U, U, C.POINTER(U), C.POINTER(U), U, C.POINTER(U), C.POINTER(U), U, U,
                                        C.POINTER(RegionChild)]),
                ('metal_ir_switch_selector', U, [P, U, U, U, Q]),
                ('metal_ir_switches', C.POINTER(SwitchRegion), [P, C.POINTER(C.c_size_t)]),
                ('metal_ir_switch_values', C.POINTER(U), [P, C.POINTER(C.c_size_t)]),
                ('metal_ir_repeat', U, [P, U, U, U, U, C.POINTER(U), U, C.POINTER(RegionChild)]),
                ('metal_ir_repeats', C.POINTER(RepeatRegion), [P, C.POINTER(C.c_size_t)]),
                ('metal_ir_repeat_carried', C.POINTER(U), [P, C.POINTER(C.c_size_t)]),
                ('metal_ir_dispatch_repeats', C.POINTER(U), [P, C.POINTER(C.c_size_t)]),
                ('metal_ir_archive_load', P, [P, P, C.c_size_t, P, RELEASE, P, C.c_char_p, C.c_size_t]),
                ('metal_ir_graph_context', P, [P]),
                ('metal_ir_graph_context_set', C.c_int, [P, P]),
                ('metal_ir_archive_program', P, [P, U]), ('metal_ir_archive_free', None, [P]),
                ('metal_ir_status_word', Q, [P]),
                ('metal_ir_transport_addressing', None, [P, U]), ('metal_ir_transport_resolved', U, [P]),
                ('metal_ir_loaded', C.c_int, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_value_info', None, [P, U, C.POINTER(ValueInfo)]),
                ('metal_ir_value_layout', U, [P, U, C.POINTER(Term), U, C.POINTER(C.c_int64)]),
                ('metal_ir_value_shape', U, [P, U, C.POINTER(Q), U]),
                ('metal_ir_value_external', C.c_int, [P, U, U]),
                ('metal_ir_storages', C.c_size_t, [P, C.POINTER(C.POINTER(Storage))]),
                ('metal_ir_value_placed', None, [P, U, P, C.POINTER(Q), Q, Q, U]),
                ('metal_ir_step_placed', None, [P, P, C.POINTER(Q), Q, U]),
                ('metal_ir_place', C.c_int, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_value', U, [P, U, U, U, C.POINTER(Q), U, C.POINTER(Term), C.c_int64, Q]),
                ('metal_ir_value_view', None, [P, U, U]),
                ('metal_ir_value_data', None, [P, U, P, C.c_size_t, P, Q, P, RELEASE]),
                ('metal_ir_source_relation', C.c_int, [P, P, C.POINTER(C.c_size_t), C.POINTER(U), C.POINTER(U),
                                                       C.c_size_t, C.POINTER(SourceOperand), RELEASE, P, C.c_char_p,
                                                       C.c_size_t]),
                ('metal_ir_step_value', U, [P]), ('metal_ir_step_state', U, [P]),
                ('metal_ir_value_planned', None, [P, U, U, U, Q]), ('metal_ir_value_paged', None, [P, U, Q, U]),
                ('metal_ir_value_receives', None, [P, U, C.POINTER(Range)]),
                ('metal_ir_value_sends', None, [P, U, C.POINTER(Release)]),
                ('metal_ir_value_received', None, [P, U, C.POINTER(Range)]),
                ('metal_ir_value_sent', None, [P, U, C.POINTER(Release)]),
                ('metal_ir_value_origin_sent', None, [P, U, C.POINTER(Release), Q]),
                ('metal_ir_value_origin_slots', None, [P, U, C.POINTER(Release), C.POINTER(Origin)]),
                ('metal_ir_value_releases', C.POINTER(Release), [P, U, C.POINTER(U)]),
                ('metal_ir_value_sending', None, [P, U, C.POINTER(Sending)]),
                ('metal_ir_transport', None, [P, Q]),
                ('metal_ir_generic', U, [P, U, C.POINTER(Q), C.POINTER(U), U, C.POINTER(U), U, C.POINTER(U),
                                         C.POINTER(U), C.POINTER(U), C.POINTER(U), C.POINTER(Expr), U, C.POINTER(Term),
                                         U, U, Q]),
                ('metal_ir_copy', U, [P, U, U, U, C.POINTER(Q), U, Q]),
                ('metal_ir_states', None, [P, U, U, C.POINTER(U), C.POINTER(Q)]),
                ('metal_ir_symbol', U, [P, U, U, Q]), ('metal_ir_loop_extent', None, [P, U, U, U]),
                ('metal_ir_check', U, [P, U, U, Q]),
                ('metal_ir_check_message', None, [P, U, C.c_char_p]),
                ('metal_ir_check_dep_token', None, [P, U, U]),
                ('metal_ir_lapack', U, [P, C.POINTER(Lapack), U, C.POINTER(Q), U, C.POINTER(U), U, C.POINTER(U),
                                        C.POINTER(Matrix), C.POINTER(Expr), U, C.POINTER(Term), Q]),
                ('metal_ir_sort', U, [P, U, U, Q, U, U, U, U, U, Q]),
                ('metal_ir_region', U, [P, U, U]), ('metal_ir_parameter', U, [P, U, U]),
                ('metal_ir_literal', U, [P, U, U, Q]), ('metal_ir_binary', U, [P, U, U, U, U]),
                ('metal_ir_unary', U, [P, U, U, U]), ('metal_ir_convert', U, [P, U, U, U]),
                ('metal_ir_bitcast', U, [P, U, U, U]), ('metal_ir_builtin', U, [P, U, U, U, C.POINTER(U)]),
                ('metal_ir_extract', U, [P, U, U, U, C.POINTER(U)]),
                ('metal_ir_extract_if', U, [P, U, U, U, C.POINTER(U), U, U]),
                ('metal_ir_builtin_named', U, [C.c_char_p]),
                ('metal_ir_scatter', None, [P, U, U, U, C.POINTER(U), U, U]),
                ('metal_ir_yield', None, [P, U, U, C.POINTER(U)]),
                ('metal_ir_if_begin', None, [P, U, U]), ('metal_ir_if_else', None, [P, U, U, C.POINTER(U)]),
                ('metal_ir_if_end', U, [P, U, U, C.POINTER(U)]),
                ('metal_ir_operation_erase', None, [P, U]),
                ('metal_ir_recorder_target', C.c_int, [P]),
                ('metal_ir_construct', C.c_int, [P, P, C.c_char_p, C.c_size_t]),
                ('metal_ir_rewrite', P, [P, P, U, C.c_char_p, C.c_size_t]),
                ('metal_ir_rewritten', C.POINTER(RewriteResult), [P]),
                ('metal_ir_work_terms', C.c_int, [P, C.POINTER(C.c_size_t), C.POINTER(C.POINTER(WorkTerm)),
                                                  C.POINTER(C.POINTER(Expr)), C.POINTER(C.POINTER(Term)),
                                                  C.POINTER(C.POINTER(WorkExtent)), C.POINTER(C.POINTER(WorkAxis)),
                                                  C.POINTER(C.POINTER(U))]),
                ('metal_ir_operation_axis', C.c_int, [P, U, U, U, Q, Q]),
                ('metal_ir_work_associations', C.POINTER(WorkAssociation), [P, C.POINTER(C.c_size_t)]),
                ('metal_ir_dispatches', U, [P, C.POINTER(C.POINTER(Dispatch))]),
                ('metal_ir_allocations', C.POINTER(U), [P, C.POINTER(C.c_size_t)]),
                ('metal_ir_sources', C.POINTER(U), [P, C.POINTER(C.c_size_t)]),
                ('metal_ir_relation', P, [P]),
                ('metal_ir_dispatch_numerics', C.POINTER(Numeric), [P, C.POINTER(C.c_size_t)]),
                ('metal_ir_kernel_devices', None, [P, U, U * 3]),
                ('metal_ir_lower', C.c_int, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_finalize', C.c_int, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_specialize', P, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_select_occurrence', P, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_emit_selected', C.c_int, [P, P, C.c_char_p, C.c_size_t]),
                ('metal_ir_program_buffers', P, [P]),
                ('metal_ir_bind_value', C.c_int, [P, P, U, P, C.POINTER(Q), Q, Q, U, C.c_char_p, C.c_size_t]),
                ('metal_ir_bind_step', None, [P, P, C.POINTER(Q), Q, U]),
                ('metal_ir_clone', P, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_machine', None, [P, C.POINTER(mesh_graph.Machine), C.POINTER(mesh_graph.Machine)])):
            function = getattr(lib, name)
            function.restype, function.argtypes = result, arguments
        COMPILER.append(lib)
    return COMPILER[0]


# metal-microbench metal_ir.h: the produced-kernel stages' calls and the binary64 numeric collection's (native129),
# bound on their first use over the loaded compiler (_compiler), apart from its construction bindings
UPSTREAM = []


def _upstream():
    if not UPSTREAM:
        lib, P, U = _compiler(), C.c_void_p, mesh.U
        for name, result, arguments in (
                ('metal_ir_upstream', U, [P, C.POINTER(Upstream)]),
                ('metal_ir_upstream_composition', C.c_int, [P, U, C.POINTER(UpstreamComposed)]),
                ('metal_ir_upstream_callables', P, [P, U, P, C.c_char_p, C.c_size_t]),
                ('metal_ir_callables_emit', C.c_int, [P, C.POINTER(CallableSource), C.c_char_p, C.c_size_t]),
                ('metal_ir_callables_output', P, [P]),
                ('metal_ir_output_view', C.c_void_p, [P, P, C.POINTER(C.c_size_t)]),
                ('metal_ir_output_functions', C.c_size_t, [P]),
                ('metal_ir_output_function', P, [P, C.c_size_t]),
                ('metal_ir_output_function_name', P, [P]), ('metal_ir_output_function_result', P, [P]),
                ('metal_ir_output_function_declaration', P, [P]),
                ('metal_ir_output_function_parameters', C.c_size_t, [P]),
                ('metal_ir_output_function_parameter', P, [P, C.c_size_t]),
                ('metal_ir_output_parameter_type', P, [P]), ('metal_ir_output_parameter_name', P, [P]),
                ('metal_ir_output_parameter_declaration', P, [P]),
                ('metal_ir_callables_role', U, [P, P]), ('metal_ir_callables_address', P, [P, P]),
                ('metal_ir_callable_address_page', C.POINTER(UpstreamPage), [P]),
                ('metal_ir_callable_address_value', U, [P]),
                ('metal_ir_callables_operand', P, [P, P, P]),
                ('metal_ir_callable_operand_kind', U, [P]), ('metal_ir_callable_operand_word', U, [P]),
                ('metal_ir_callable_operand_binding', C.POINTER(UpstreamWord), [P, U]),
                ('metal_ir_callable_operand_readiness', C.POINTER(UpstreamReadiness), [P]),
                ('metal_ir_callable_operand_table', C.POINTER(UpstreamTable), [P]),
                ('metal_ir_callable_operand_page', C.POINTER(UpstreamPage), [P]),
                ('metal_ir_callables_module', P, [P]), ('metal_ir_callables_function', P, [P, U]),
                ('metal_ir_callables_function_name', P, [P, P, C.POINTER(C.c_size_t)]),
                ('metal_ir_callable_address_function', P, [P, U]), ('metal_ir_output_function_ir', P, [P]),
                ('metal_ir_output_parameter_ir', P, [P]), ('metal_ir_callable_operand_param', P, [P]),
                ('metal_ir_callable_operand_view', P, [P, U]),
                ('metal_ir_binary64_open', P, [C.c_char_p, C.c_size_t]),
                ('metal_ir_binary64_request', C.c_int, [P, C.POINTER(Binary64Primitive), C.POINTER(Binary64Function),
                                                        C.c_char_p, C.c_size_t]),
                ('metal_ir_binary64_free', None, [P]),
                ('metal_ir_upstream_emitted', C.c_int, [P, U, U, C.POINTER(UpstreamEmission), C.c_char_p,
                                                         C.c_size_t]),
                ('metal_ir_upstream_configured', C.POINTER(UpstreamConfiguration), [P, U]),
                ('metal_ir_upstream_error_workspace', U, [P]),
                ('metal_ir_upstream_neural', C.c_int, [P, U, C.POINTER(UpstreamNeural), C.c_char_p, C.c_size_t]),
                ('metal_ir_value_zeroed', None, [P, U])):
            function = getattr(lib, name)
            function.restype, function.argtypes = result, arguments
        UPSTREAM.append(lib)
    return UPSTREAM[0]


# rdma/mesh-graph.h: a kernel's index domain and its analysis values (mesh_graph_domain_create, mesh_graph_index_*), in
# the library holding the compilation's analysis context (rdma/mesh_graph.py LIB, Context), bound on first use
GRAPH = []


def _graph():
    if not GRAPH:
        lib, P = mesh_graph.LIB, C.c_void_p
        for name, result, arguments in (
                ('mesh_graph_domain_create', C.c_int, [P, mesh.U, C.POINTER(DomainDim), C.POINTER(P)]),
                ('mesh_graph_domain_free', None, [P]),
                ('mesh_graph_index_dim', P, [P, mesh.U]), ('mesh_graph_index_constant', P, [P, C.c_int64]),
                ('mesh_graph_index_truth', P, [P, C.c_int]), ('mesh_graph_index_add', P, [P, P]),
                ('mesh_graph_index_mul', P, [P, P]), ('mesh_graph_index_floordiv', P, [P, P]),
                ('mesh_graph_index_modular_indexing', P, [P, P, P]), ('mesh_graph_index_python_mod', P, [P, P]),
                ('mesh_graph_index_tdiv_q', P, [P, P]), ('mesh_graph_index_tdiv_r', P, [P, P]),
                ('mesh_graph_index_minimum', P, [P, P]), ('mesh_graph_index_maximum', P, [P, P]),
                ('mesh_graph_index_where', P, [P, P, P]), ('mesh_graph_index_compare', P, [mesh.U, P, P]),
                ('mesh_graph_index_and', P, [P, P]), ('mesh_graph_index_or', P, [P, P]),
                ('mesh_graph_index_not', P, [P]), ('mesh_graph_index_wrap', P, [P, mesh.U, C.c_int]),
                ('mesh_graph_index_free', None, [P])):
            function = getattr(lib, name)
            function.restype, function.argtypes = result, arguments
        GRAPH.append(lib)
    return GRAPH[0]


# metal_ir_upstream_neural, after metal_ir_construct: Neural Engine operation `op`'s original Core ML program, its
# saved package directory `package`, function `function` and selected `units` (a ct.ComputeUnit's name); `inputs`
# and `outputs` each (feature name, argument index), `states` each (state name, read argument, written argument)
def _neural(program, op, package, function, units, inputs, outputs, states):
    error = C.create_string_buffer(4096)
    features = [_array(UpstreamFeature, [UpstreamFeature(name.encode(), argument, 0) for name, argument in made])
                for made in (inputs, outputs)]
    state = _array(UpstreamState, [UpstreamState(name.encode(), read, written) for name, read, written in states])
    made = UpstreamNeural(str(package).encode(), function.encode(), units.encode(), len(inputs), len(outputs),
                          len(states), 0, *features, state)
    status = _upstream().metal_ir_upstream_neural(program, op, C.byref(made), error, len(error))
    if status:
        raise OSError(status, 'operation %d: its Core ML program (metal_ir_upstream_neural): %s'
                      % (op, error.value.decode()))


def _copied(record):
    return type(record).from_buffer_copy(record)


# metal_ir.h metal_ir_binary64_open, metal_ir_binary64_request and metal_ir_binary64_free: one upstream numerical body's
# binary64 numeric collection, its IR only, which its backend kernel's own emitter fills as it meets each typed
# operation (`request`: a primitive's kind BINARY64_*, its metal_ir_binary / metal_ir_unary / metal_ir_builtin_named
# code, a CONVERT's scalars) and every composition of that kernel clones (metal_ir_upstream_callables `numeric`), never
# re-requested; `functions` each request's function as the library returned it (name, parameter kinds, result kind,
# its GPU work), freed once with its owner
class Binary64:
    def __init__(self):
        lib, error = _upstream(), C.create_string_buffer(4096)
        self.handle, self.functions = lib.metal_ir_binary64_open(error, len(error)), {}
        if not self.handle:
            raise OSError(errno.ENOMEM, 'a binary64 numeric collection (metal_ir_binary64_open): %s'
                          % error.value.decode())
        self.close = weakref.finalize(self, lib.metal_ir_binary64_free, self.handle)

    def request(self, kind, code, from_=0, to=0):
        key = (kind, code, from_, to)
        if key not in self.functions:
            made, error = Binary64Function(), C.create_string_buffer(4096)
            status = _upstream().metal_ir_binary64_request(self.handle, C.byref(Binary64Primitive(*key)),
                                                           C.byref(made), error, len(error))
            if status:
                raise OSError(status, 'binary64 primitive %r (metal_ir_binary64_request): %s'
                              % (key, error.value.decode()))
            self.functions[key] = (made.name.decode(), tuple(made.parameter[k] for k in range(made.parameters)),
                                   made.result, _copied(made.work))
        return self.functions[key]


# metal_ir_upstream_composition of produced kernel `op` of `program` (after its ports are declared) for pass `later`
# (Composed); its callable collection made and printed once (_callables, cloning the kernel's binary64 collection
# `numeric`, a Binary64 or None), the later pass given its first pass's (`callables`)
def _composition(program, op, numeric=None, later=0, callables=None):
    made = UpstreamComposed()
    status = _upstream().metal_ir_upstream_composition(program, op, C.byref(made))
    if status:
        raise OSError(status, 'operation %d: no constructed upstream kernel (metal_ir_upstream_composition)' % op)
    return Composed(program, op, made, _callables(program, op, numeric) if callables is None else callables, later)


# metal_ir_upstream_callables then metal_ir_callables_emit: produced kernel `op`'s one callable collection in `program`
# (its transport protocol, its paged arguments' addresses and its numerical body's binary64 primitives, `numeric`
# cloned) printed once (Callables)
def _callables(program, op, numeric):
    lib, emitted = _upstream(), CallableSource()
    error = C.create_string_buffer(4096)
    collection = lib.metal_ir_upstream_callables(program, op, None if numeric is None else numeric.handle,
                                                 error, len(error))
    if not collection:
        raise OSError(errno.ENODATA, 'operation %d: its callable collection (metal_ir_upstream_callables): %s'
                      % (op, error.value.decode()))
    status = lib.metal_ir_callables_emit(collection, C.byref(emitted), error, len(error))
    if status:
        raise OSError(status, 'operation %d: its callable collection (metal_ir_callables_emit): %s'
                      % (op, error.value.decode()))
    return Callables(program, collection, emitted)


# metal_ir.h metal_ir_upstream_emitted. The selected upstream writer supplies its complete entry.
def _emitted(program, op, number, source, entry, parameters, language):
    lib, error = _upstream(), C.create_string_buffer(4096)
    source = source.encode()
    arguments = _array(UpstreamParameter, [UpstreamParameter(p.type.encode(), p.slot, p.pointer) for p in parameters])
    made = UpstreamEmission(source=source, entry=entry.encode(), bytes=len(source), parameter=arguments,
                            parameters=len(parameters), language=language.encode() if language else None)
    status = lib.metal_ir_upstream_emitted(program, op, number, C.byref(made), error, len(error))
    if status:
        raise OSError(status, "operation %d pass %d: %s" % (op, number, error.value.decode()))


# metal_ir_upstream_configured: produced kernel `op`'s first pass's configured launch, its fields by name
def _configured(program, op):
    made = _upstream().metal_ir_upstream_configured(program, op)
    if not made:
        raise OSError(errno.ENODATA, 'operation %d: no configured first pass (metal_ir_upstream_emitted)' % op)
    return {name: tuple(getattr(made[0], name)) for name in ('grid', 'group', 'groups')}


# mesh-call.h mesh_transfers_start and mesh-dataflow.h mesh_host_started, the program's `start` and `started`
START = C.cast(mesh.LIB.mesh_transfers_start, C.c_void_p)
STARTED = C.cast(mesh.LIB.mesh_host_started, C.c_void_p)


# metal_recording.h MetalProgramCheckFailure: a run's failed check as the error its operation raises in torch
def _execution_error(replay, status, message):
    failure = CheckFailure()
    if status == errno.EDOM and not replay['lib'].MetalProgramCheckFailure(replay['program'], C.byref(failure)):
        kind = torch.linalg.LinAlgError if failure.code in (1, 2, 3) else RuntimeError
        error = OSError(failure.info, message) if failure.code == 5 else kind(message)
        error.operation_identity, error.batch, error.info, error.code = \
            failure.identity, failure.batch, failure.info, failure.code
        return error
    return OSError(status, message)


# metal-microbench metal_ir.h metal_ir_value, metal_ir_upstream, metal_ir_copy, metal_ir_lapack, metal_ir_check,
# metal_ir_repeat and metal_ir_switch: the direct native projections over program values, encoded into `program` (an
# existing program, metal_ir_append's destination), else a new one; `emitted` each projection's actual emission
# (Emitted: its record, its returned operation and the operations it created), in emission order, `regions` each
# registered region (Region), in postorder; `base` the caller's identity, `identity` the current projection's (base |
# its source position)
class _Builder:
    def __init__(self, lib, identity, program=None):
        self.lib = lib
        self.p = lib.metal_ir_program_new() if program is None else program
        self.emitted, self.regions, self.values = [], [], {}
        self.base = self.identity = identity

    # metal_ir.h metal_ir_value: one storage root of role `role`, `elements` of metal_ir scalar `scalar`, its caller's
    # source identity (0 none)
    def stored(self, role, scalar, elements, identity=0):
        return self.lib.metal_ir_value(self.p, role, scalar, 1, _array(mesh.Q, [elements]), 1,
                                       _array(Term, [Term(0, 0, 1, 1, 0)]), 0, identity)

    # metal_ir.h metal_ir_value and metal_ir_value_view: the view of storage root `root` (of role `role`) as metal_ir
    # scalar `scalar` at `sizes` and `strides` from element `offset` (in that scalar's elements from the root's first
    # byte), one ValueID per distinct view of a root
    def viewed(self, root, role, scalar, sizes, strides, offset):
        signature = (root, scalar, tuple(sizes), tuple(strides), offset)
        if signature not in self.values:
            v = self.lib.metal_ir_value(self.p, role, scalar, len(sizes), _array(mesh.Q, sizes), len(strides),
                                        _array(Term, [Term(d, 0, s, 1, 0) for d, s in enumerate(strides)]), offset,
                                        self.identity)
            self.lib.metal_ir_value_view(self.p, v, root)
            self.values[signature] = v
        return self.values[signature]

    # metal_ir.h metal_ir_value_data: a CONSTANT or STATE root's initial contents, `base`'s storage zero-copy (an MPS
    # storage's MTLBuffer and its contents, or the host bytes); the program adopts one reference to `base` (_lent; the
    # call has no failure result, so it always adopts) and releases it once at metal_ir_program_free
    def attach(self, v, base):
        storage = base.untyped_storage()
        buffer = storage.data_ptr() if base.device.type == 'mps' else None
        data = OBJECT(buffer, _SEL(b'contents')) if buffer else storage.data_ptr()
        _lent(base, lambda release, owner: self.lib.metal_ir_value_data(self.p, v, data, storage.nbytes(), buffer, 0,
                                                                        owner, release), lambda _: True)

    # metal_ir_upstream: one produced kernel (`record` the producer's own record: lower_inductor.py Kernel, a Neural
    # Engine body's Produced) as one operation at `place`, its identity the builder's base with source position `node`:
    # `arguments` (kind UPSTREAM_*, ValueID, access) in its launch's order, `domain` its index domain (a
    # mesh_graph_domain pointer, None for none), `accesses` its UpstreamAccess records, `work` its semantic work (a
    # mesh_graph.Work, None unknown), `sources` the source positions it implements, `grid` its total threads ((1, 1, 1)
    # where its launch has none: a CPU kernel's one typed call, a Core ML program) and `group` its threadgroup, None its
    # pipeline's (METAL_IR_DEFAULT_GROUP); kept as an emission (Emitted) and returned
    def upstream(self, record, place, node, arguments, domain, accesses, work, sources, grid, group):
        lib = _upstream()
        self.identity = self.base | node
        body = Upstream(place, len(arguments), len(accesses), len(sources),
                        _array(UpstreamArgument, [UpstreamArgument(kind, value, access, access, 0)
                                                  for kind, value, access in arguments]),
                        domain, _array(UpstreamAccess, accesses), _array(mesh.Q, [self.base | s for s in sources]),
                        None if work is None else C.pointer(work), self.identity, (mesh.U * 3)(*grid),
                        (mesh.U * 3)(*(group or (0, 0, 0))), int(group is not None), 0, None, None, None, None,
                        None, None)
        first = lib.metal_ir_operations(self.p)
        operation = lib.metal_ir_upstream(self.p, C.byref(body))
        if operation == NONE:
            raise OSError(errno.EINVAL, 'a produced kernel\'s operand (metal_ir_upstream): an argument names no '
                                        'program value')
        self.emitted.append(Emitted(record, operation, first, lib.metal_ir_operations(self.p)))
        return operation

    # metal_ir.h metal_ir_copy: the canonical logical copy into = convert(given) over its original transfer domain
    # (`bounds` its loop extents, each view's dimension d at loop d), at `place` and source position `node` (0: newly
    # produced physical-copy work), `into` and `given` (ValueID, ...) their views; `record` the producer object it
    # implements (a CopyRecord, a landing), kept with its returned operation (Emitted)
    def copy(self, record, place, node, bounds, into, given):
        self.identity = self.base | node
        first = self.lib.metal_ir_operations(self.p)
        operation = self.lib.metal_ir_copy(self.p, into[0], given[0], len(bounds), _array(mesh.Q, bounds), place,
                                           self.identity)
        self.emitted.append(Emitted(record, operation, first, self.lib.metal_ir_operations(self.p)))
        return operation

    # metal_ir.h metal_ir_lapack: one typed LAPACK call (struct metal_ir_lapack_call: `routine` LAPACK_GESDD ..
    # LAPACK_GELSY, `job` its JOB characters, `rcond`) at source position `node` over `batch` (its batch extents): each
    # of `inputs` then `outputs` (ValueID, its typed sizes, `rank` per batch member: a matrix 2, a vector 1, a scalar 0,
    # `paired` its last typed dimension a complex value's real pair), its leading dimensions the batch's
    # (right-aligned, an extent 1 broadcast), its own dimensions loops of their own: a matrix's row and column (and
    # pair) the call's matrix (struct metal_ir_lapack_matrix), every map one loop or constant 0
    def lapack(self, record, node, routine, job, rcond, batch, inputs, outputs):
        self.identity = self.base | node
        sizes, picks, matrices, operands = list(batch), [], [], []
        for value, shape, rank, paired in [*inputs, *outputs]:
            own = rank + paired
            lead = len(shape) - own
            first = len(batch) - lead
            picks.append([None if n == 1 and batch[first + d] != 1 else first + d for d, n in enumerate(shape[:lead])]
                         + list(range(len(sizes), len(sizes) + own)))
            loops = list(range(len(sizes), len(sizes) + rank)) + [NONE] * (2 - rank)
            matrices.append(Matrix(loops[0], loops[1], len(sizes) + rank if paired else NONE))
            sizes += shape[lead:]
            operands.append(value)
        exprs, terms = [], []
        for pick in picks:
            for loop in pick:
                exprs.append(Expr(len(terms), int(loop is not None), 0))
                terms += [Term(loop, 0, 1, 1, 0)] if loop is not None else []
        call, first, n = Lapack(routine, len(batch), job, rcond), self.lib.metal_ir_operations(self.p), len(inputs)
        operation = self.lib.metal_ir_lapack(self.p, C.byref(call), len(sizes), _array(mesh.Q, sizes), n,
                                             _array(mesh.U, operands[:n]), len(outputs), _array(mesh.U, operands[n:]),
                                             _array(Matrix, matrices), _array(Expr, exprs), len(terms),
                                             _array(Term, terms), self.identity)
        self.emitted.append(Emitted(record, operation, first, self.lib.metal_ir_operations(self.p)))
        return operation

    # metal_ir.h metal_ir_check and metal_ir_check_message: the INFO check of `info` (a ValueID, one INFO per batch
    # member) at source position `node`, its kind `code` (CHECK_*), its api `message` (torch's _linalg_check_errors)
    def check(self, record, node, info, code, message):
        self.identity = self.base | node
        first = self.lib.metal_ir_operations(self.p)
        operation = self.lib.metal_ir_check(self.p, info, code, self.identity)
        self.lib.metal_ir_check_message(self.p, operation, message.encode())
        self.emitted.append(Emitted(record, operation, first, self.lib.metal_ir_operations(self.p)))
        return operation

    # metal_ir.h metal_ir_repeat: one numerical repeat region, registered in postorder once its pass's operations
    # [first, end) are emitted: `condition` the one-element I32 ValueID tested before every pass (the first included),
    # `carried` the storage roots a pass reads that the initialization or an earlier pass wrote, `children` its direct
    # children (RegionChild); `record` the producer's own control record (lower_inductor.py ExternCall), kept with the
    # region (Region) beside the emissions; returns the region (a metal_ir_repeats index, never an operation)
    def repeat(self, record, first, end, condition, carried, children):
        region = self.lib.metal_ir_repeat(self.p, first, end, condition, len(carried), _array(mesh.U, carried),
                                          len(children), _array(RegionChild, children))
        self.regions.append(Region(record, REGION_REPEAT, region, (first,), (end,), NONE))
        return region

    # metal_ir.h metal_ir_switch_selector: the switch's selector at its source control point, before its branches'
    # operations, of its index's ValueID at `place` and source position `node`, kept as its record's emission
    def selector(self, record, place, node, index, branches):
        self.identity = self.base | node
        first = self.lib.metal_ir_operations(self.p)
        operation = self.lib.metal_ir_switch_selector(self.p, index, branches, place, self.identity)
        self.emitted.append(Emitted(record, operation, first, self.lib.metal_ir_operations(self.p)))
        return operation

    # metal_ir.h metal_ir_switch (registration only): an indexed switch in postorder once its branches are emitted,
    # branch k operations [first[k], end[k]), `yielded` each branch's yields of the common `results` branch-major (the
    # caller's own copies inside each branch store them), its `selector` and direct `children`; kept (Region)
    def switch(self, record, index, first, end, yielded, results, selector, children):
        region = self.lib.metal_ir_switch(self.p, index, len(first), _array(mesh.U, first), _array(mesh.U, end),
                                          len(results), _array(mesh.U, yielded), _array(mesh.U, results), selector,
                                          len(children), _array(RegionChild, children))
        self.regions.append(Region(record, REGION_SWITCH, region, tuple(first), tuple(end), selector))
        return region
