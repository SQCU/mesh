# design/compilation-composition-change-request.md items 1, 3, 6 and 7; design/aot-compilation-change-request.md A8 and
# A9: the live native owner. The C ABI's types and library bindings (metal-microbench metal_ir.h, metal_recording.h;
# rdma/mesh-metal.h), the Objective-C/Metal resource primitives, borrowed buffers, archive byte ownership, the typed C
# encoder (_Builder) of the direct native projections (values and views, a produced kernel, a typed LAPACK call, an
# INFO check, an elementwise copy or fill over program values) and of the transitional records native.py still gives
# it (compiler/mesh_compiler/operations.py _Generic, _Repeat, _Switch, _Sort, _Check)
import collections
import ctypes as C
import ctypes.util
import errno
import os
import pathlib
import sys

import torch

from .operations import (CLASS, CPU, GPU, KINDS, SCALAR, _Check, _Repeat, _Sort, _Switch, _Words, _bits, _copy, _key,
                         _placed, _stored)

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh
import mesh_graph

# metal-microbench metal_ir.h METAL_IR_INPUT .. METAL_IR_STATE (a value's role), its region kinds, UINT32_MAX (no
# value), and METAL_IR_DATA, METAL_IR_TRANSPORT, METAL_IR_STEP (metal_ir_value_info `word` of the step value)
INPUT, CONSTANT, TEMPORARY, OUTPUT, STATE = range(5)
BODY, COMBINE, FINALIZE = range(3)
NONE = 0xFFFFFFFF
METAL_IR_STEP = 2
# metal_ir.h METAL_IR_PARALLEL (linalg iterator_types), METAL_IR_ASSIGN (an output's combination where the operation
# has none) and METAL_IR_IDENTITY (its starting value): a direct elementwise operation's (_Builder.elementwise)
PARALLEL, ASSIGN, IDENTITY = 0, 0, 0
# metal_ir.h METAL_IR_GESDD .. METAL_IR_GELSS (struct metal_ir_lapack_call `routine`) and METAL_IR_CHECK_LU ..
# METAL_IR_CHECK_NATIVE (metal_ir_check `code`)
LAPACK_GESDD, LAPACK_HEEVD, LAPACK_GEEV, LAPACK_GELSD, LAPACK_GELSS = range(5)
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
# (_lent) until the archive and every relation it lent are freed. The caller's own reference to the archive is
# released with metal_ir_archive_free
def _archive_loaded(compiler, data, device, error):
    archive = _lent(data, lambda release, owner: compiler.metal_ir_archive_load(data, len(data), device, release, owner,
                                                                              error, len(error)), bool)
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


class MetalSent(C.Structure):
    _fields_ = [('counter', C.c_void_p), ('sends', mesh.Q), ('copies', mesh.Q)]


class Generated(C.Structure):
    _fields_ = [('every', mesh.U), ('producer', mesh.U), ('ordinal', mesh.U)]


class Range(C.Structure):
    _fields_ = [('input', Remote), ('generation', Generated), ('partitions', mesh.U), ('page', mesh.Q),
                ('field', C.POINTER(mesh.U)), ('at', mesh.Q), ('transfer', mesh.U), ('owner', mesh.U)]


# mesh_metal_release: one release record of a value, declared {transfer, owner} (its group's transfer and owner call),
# prepared in place (supplied, status); 32 bytes
class Release(C.Structure):
    _fields_ = [('supplied', mesh.Q), ('status', mesh.Q), ('total', mesh.Q), ('transfer', mesh.U), ('owner', mesh.U)]


class Sending(C.Structure):
    _fields_ = [('sent', MetalSent), ('extent', C.POINTER(mesh_graph.Access)), ('extents', C.c_size_t),
                ('transfer', mesh.U), ('padding', mesh.U)]


# metal_recording.h struct MetalCheckFailure
class CheckFailure(C.Structure):
    _fields_ = [('identity', mesh.Q), ('batch', mesh.Q), ('info', C.c_int64), ('code', mesh.U), ('padding', mesh.U)]


# metal_ir.h struct metal_ir_value_info (`received`, `sent` and `sendings` record counts: metal_ir_value_ranges,
# metal_ir_value_releases, metal_ir_value_sendings; `page_bytes` at 104, 120 bytes) and struct metal_ir_storage
class ValueInfo(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('role', 'scalar', 'storage', 'function', 'index', 'word', 'of', 'range',
                                          'received', 'sent', 'zeroed', 'sendings')] + [
        (name, mesh.Q) for name in ('identity', 'bytes', 'offset', 'at', 'data_bytes')] + [
        ('data', C.c_void_p), ('buffer', C.c_void_p), ('page_bytes', mesh.Q), ('pages', mesh.U), ('source', mesh.U)]


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
    _fields_ = [('buffer', C.c_void_p), ('offset', mesh.Q), ('base', C.c_void_p)]


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


# metal_ir.h metal_ir_operations: one structured record's actual emission by _Builder.generic: `record` the record
# itself (compiler/mesh_compiler/operations.py, the source owner), `operation` its returned operation ID, [first, end)
# the operations it created (metal_ir_sort several)
Emitted = collections.namedtuple("Emitted", "record operation first end")
# metal_ir.h metal_ir_repeat: one registered repetition: `record` the operations.py _Repeat, `region` its
# metal_ir_repeats index (never an operation), [first, end) its pass's operation interval, nested ones included
Repeated = collections.namedtuple("Repeated", "record region first end")


# metal-microbench metal_ir.h (transitional) metal_ir_upstream and its stages: METAL_IR_UPSTREAM_BUFFER .. _ERROR (an
# argument's kind), struct metal_ir_upstream_argument and struct metal_ir_upstream (a produced kernel's operand before
# metal_ir_construct; its views rdma/mesh-graph.h struct mesh_graph_view, mesh_graph.Viewed), what
# metal_ir_upstream_composition returns (struct metal_ir_upstream_word, _received, _sent, _page, _readiness, _table,
# _composed), the transport helper metal_ir_upstream_helper prints (struct metal_ir_upstream_helper), a pass's terminal
# emission (struct metal_ir_upstream_emission), the first pass's configured launch (struct
# metal_ir_upstream_configuration) and a threadgroup's footprint (mesh-graph.h struct mesh_graph_footprint)
UPSTREAM_BUFFER, UPSTREAM_SCALAR, UPSTREAM_ERROR = range(3)


class UpstreamArgument(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('kind', 'value', 'access', 'padding')]


class Upstream(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('place', 'arguments', 'views', 'sources')] + [
        ('argument', C.POINTER(UpstreamArgument)), ('view', C.POINTER(mesh_graph.Viewed)),
        ('source', C.POINTER(mesh.Q)), ('identity', mesh.Q), ('grid', mesh.U * 3), ('group', mesh.U * 3),
        ('grouped', mesh.U), ('padding', mesh.U)]


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
    _fields_ = [(name, mesh.U) for name in ('classes', 'spans', 'class_slot', 'span_slot', 'later_class_slot',
                                            'later_span_slot', 'ranges', 'padding')]


class UpstreamTable(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('value', 'slot', 'later_slot', 'words', 'ranges', 'releases')] + [
        ('padding', mesh.U * 2)]


class UpstreamComposed(C.Structure):
    _fields_ = [('word', C.POINTER(UpstreamWord)), ('later_word', C.POINTER(UpstreamWord)),
                ('received', C.POINTER(UpstreamReceived)), ('sent', C.POINTER(UpstreamSent)),
                ('page', C.POINTER(UpstreamPage)), ('readiness', C.POINTER(UpstreamReadiness)),
                ('table', C.POINTER(UpstreamTable))] + [
        (name, mesh.U) for name in ('words', 'later_words', 'receives', 'sends', 'pages', 'padding')]


class UpstreamHelper(C.Structure):
    _fields_ = [(name, C.c_char_p) for name in ('source', 'acquire', 'later', 'release')] + [('bytes', mesh.Q)]


# struct metal_ir_upstream_parameter: one parameter of a CPU pass's final entry, its C type as its emitter printed it,
# the kernel binding it receives (an argument index or a composition slot), 1 for the bound pointer, 0 its pointee
class UpstreamParameter(C.Structure):
    _fields_ = [('type', C.c_char_p), ('slot', mesh.U), ('pointer', mesh.U)]


class UpstreamEmission(C.Structure):
    _fields_ = [('source', C.c_char_p), ('entry', C.c_char_p), ('parameter', C.POINTER(UpstreamParameter)),
                ('compile', C.POINTER(C.c_char_p)), ('link', C.POINTER(C.c_char_p)), ('bytes', mesh.Q),
                ('parameters', mesh.U), ('compiles', mesh.U), ('links', mesh.U)]


class UpstreamConfiguration(C.Structure):
    _fields_ = [(name, mesh.U * 3) for name in ('grid', 'group', 'groups')] + [('padding', mesh.U)]


class Footprint(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('dispatch', 'binding', 'exact', 'first', 'terms', 'padding')] + [
        ('start', C.c_int64), ('extent', mesh.Q)]


# one produced kernel's composition as the native program made it for pass `later` (0 its first, 1 its later), each
# record copied out of the program's borrowed arrays: its first and later passes' transport words
# (UpstreamWord), received ranges (value, range, its mesh_metal_range), sent values (value, counter, slot, its
# mesh_metal_releases), paged arguments (Paged), readiness table (UpstreamReadiness or None), prepared record table
# (UpstreamTable or None) and transport helper (Helper, None where it has no transport: ENODATA); the operand the
# producer's terminal emission composes with (compiler/mesh_compiler/lower_inductor.py emitted)
Composed = collections.namedtuple("Composed", "words later_words received sent pages readiness table helper later")
# a paged argument as the native composition placed it (struct metal_ir_upstream_page): its argument, its arena's
# slot, its frame table's slot, its pages of `page_bytes`, `origin` its value's byte origin in its root (the dense
# binding's offset), which its body adds before the page division
Paged = collections.namedtuple("Paged", "argument slot frames pages page_bytes origin")
# struct metal_ir_upstream_helper: its printed source (Tint MSL library mode) and the declarations of the functions its
# final emitter calls (`acquire` mesh_acquire_first, `later` mesh_acquire_later where it receives, `release`
# mesh_release where it sends; None where absent)
Helper = collections.namedtuple("Helper", "source acquire later release")


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
                ('MetalRecorderReplay', C.c_int, [P, P, C.POINTER(Replay), text, room]),
                ('MetalRecorderPlan', C.c_int, [P, P, text, room]),
                ('MetalRecorderState', C.c_int, [P, P, P, Q, text, room]),
                ('MetalReplayDestroy', None, [C.POINTER(Replay)]),
                ('MetalRecorderActors', C.c_int, [P, machine, machine, C.POINTER(_Actors)]),
                ('MetalRecorderActorsFree', None, [C.POINTER(_Actors)]),
                ('MetalRecorderProgram', P, [P, C.POINTER(Call), U, C.POINTER(mesh.Planned), text, room]),
                ('MetalProgramSave', C.c_int, [P, C.POINTER(Span), U, C.POINTER(P), C.POINTER(C.c_size_t), text,
                                               room]),
                ('MetalProgramLoad', P, [P, C.POINTER(Call), U, C.POINTER(mesh.Planned), P, text, room]),
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
        planned, transport = C.POINTER(mesh.Planned), C.POINTER(Transport)
        for name, result, arguments in (
                ('mesh_metal_transport_create', C.c_int, [mesh.CONTEXT, planned, C.c_void_p, transport]),
                ('mesh_metal_transport_destroy', None, [transport]),
                ('mesh_metal_range_declare', C.c_int, [mesh.CONTEXT, C.POINTER(mesh.U), mesh.U, mesh.U, mesh.U, mesh.U,
                                                       mesh.Q, mesh.Q, C.POINTER(Range)]),
                ('mesh_metal_range_prepare', C.c_int, [mesh.CONTEXT, transport, planned, mesh.Section,
                                                       C.POINTER(mesh.U), mesh.U, C.POINTER(Range)]),
                ('mesh_metal_release_prepare', C.c_int, [mesh.CONTEXT, transport, planned, mesh.U,
                                                         C.POINTER(Release)]),
                ('mesh_metal_sending_prepare', C.c_int, [mesh.CONTEXT, transport, planned, mesh.U, mesh.U,
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
                ('metal_ir_archive_load', P, [P, C.c_size_t, P, RELEASE, P, C.c_char_p, C.c_size_t]),
                ('metal_ir_archive_program', P, [P, U]), ('metal_ir_archive_free', None, [P]),
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
                ('metal_ir_value_releases', C.POINTER(Release), [P, U, C.POINTER(U)]),
                ('metal_ir_value_sending', None, [P, U, C.POINTER(Sending)]),
                ('metal_ir_transport', None, [P, Q]),
                ('metal_ir_generic', U, [P, U, C.POINTER(Q), C.POINTER(U), U, C.POINTER(U), U, C.POINTER(U),
                                         C.POINTER(U), C.POINTER(U), C.POINTER(U), C.POINTER(Expr), U, C.POINTER(Term),
                                         U, U, Q]),
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
                ('metal_ir_construct', C.c_int, [P, P, C.c_char_p, C.c_size_t]),
                ('metal_ir_work_terms', C.c_int, [P, C.POINTER(C.c_size_t), C.POINTER(C.POINTER(WorkTerm)),
                                                  C.POINTER(C.POINTER(Expr)), C.POINTER(C.POINTER(Term)),
                                                  C.POINTER(C.POINTER(WorkExtent)), C.POINTER(C.POINTER(WorkAxis)),
                                                  C.POINTER(C.POINTER(U))]),
                ('metal_ir_operation_axis', C.c_int, [P, U, U, U, Q, Q]),
                ('metal_ir_work_associations', C.POINTER(WorkAssociation), [P, C.POINTER(C.c_size_t)]),
                ('metal_ir_dispatches', U, [P, C.POINTER(C.POINTER(Dispatch))]),
                ('metal_ir_dispatch_numerics', C.POINTER(Numeric), [P, C.POINTER(C.c_size_t)]),
                ('metal_ir_kernel_devices', None, [P, U, U * 3]),
                ('metal_ir_lower', C.c_int, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_finalize', C.c_int, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_specialize', P, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_clone', P, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_machine', None, [P, C.POINTER(mesh_graph.Machine), C.POINTER(mesh_graph.Machine)])):
            function = getattr(lib, name)
            function.restype, function.argtypes = result, arguments
        COMPILER.append(lib)
    return COMPILER[0]


# metal-microbench metal_ir.h (transitional, the executor's native release): the produced-kernel stages' calls, bound
# on their first use over the loaded compiler (_compiler), apart from its construction bindings
UPSTREAM = []


def _upstream():
    if not UPSTREAM:
        lib, P, U = _compiler(), C.c_void_p, mesh.U
        for name, result, arguments in (
                ('metal_ir_upstream', U, [P, C.POINTER(Upstream)]),
                ('metal_ir_upstream_composition', C.c_int, [P, U, C.POINTER(UpstreamComposed)]),
                ('metal_ir_upstream_helper', C.c_int, [P, U, C.POINTER(UpstreamHelper), C.c_char_p, C.c_size_t]),
                ('metal_ir_upstream_emitted', C.c_int, [P, U, U, C.POINTER(UpstreamEmission), C.c_char_p,
                                                         C.c_size_t]),
                ('metal_ir_upstream_configured', C.POINTER(UpstreamConfiguration), [P, U]),
                ('metal_ir_upstream_footprints', None, [P, U, U, C.POINTER(Footprint), U, C.POINTER(Term)]),
                ('metal_ir_upstream_error_workspace', U, [P]),
                ('metal_ir_value_zeroed', None, [P, U])):
            function = getattr(lib, name)
            function.restype, function.argtypes = result, arguments
        UPSTREAM.append(lib)
    return UPSTREAM[0]


def _copied(record):
    return type(record).from_buffer_copy(record)


def _text(value):
    return None if value is None else value.decode()


# metal_ir_upstream_composition and metal_ir_upstream_helper of produced kernel `op` of `program` (after its ports are
# declared) for pass `later`, copied out (Composed), each paged argument's origin the native composition's own
def _composition(program, op, later=0):
    lib, made, helper, error = _upstream(), UpstreamComposed(), UpstreamHelper(), C.create_string_buffer(4096)
    status = lib.metal_ir_upstream_composition(program, op, C.byref(made))
    if status:
        raise OSError(status, 'operation %d: no constructed upstream kernel (metal_ir_upstream_composition)' % op)
    pages = [made.page[k] for k in range(made.pages)]
    composed = Composed(
        tuple(_copied(made.word[k]) for k in range(made.words)),
        tuple(_copied(made.later_word[k]) for k in range(made.later_words)),
        tuple((r.value, r.range, _copied(r.record[0])) for r in (made.received[k] for k in range(made.receives))),
        tuple((s.value, s.counter, s.slot, tuple(_copied(s.release[k]) for k in range(s.releases)))
              for s in (made.sent[k] for k in range(made.sends))),
        tuple(Paged(p.argument, p.slot, p.frames, p.pages, p.page_bytes, p.origin) for p in pages),
        _copied(made.readiness[0]) if made.readiness else None, _copied(made.table[0]) if made.table else None,
        None, later)
    status = lib.metal_ir_upstream_helper(program, op, C.byref(helper), error, len(error))
    if status == errno.ENODATA:
        return composed
    if status:
        raise OSError(status, 'operation %d: its transport helper (metal_ir_upstream_helper): %s'
                      % (op, error.value.decode()))
    return composed._replace(helper=Helper(C.string_at(helper.source, helper.bytes).decode(), _text(helper.acquire),
                                           _text(helper.later), _text(helper.release)))


# metal_ir_upstream_emitted: pass `number` of produced kernel `op`, its terminal emission (the producer's Emission:
# source, entry, types, compile, link); a CPU pass's final parameters (struct metal_ir_upstream_parameter) its
# Emission.types, each its C type as the emitter printed it, paired in order with `bindings`, each (the kernel binding
# it receives, 1 the bound pointer or 0 its pointee) as the consumer bound it (lower_metal.py emitted)
def _emitted(program, op, number, emission, bindings=()):
    lib, error = _upstream(), C.create_string_buffer(4096)
    source = emission.source.encode() if isinstance(emission.source, str) else bytes(emission.source)
    texts = [_array(C.c_char_p, [t.encode() for t in texts]) for texts in (emission.compile, emission.link)]
    parameters = [UpstreamParameter(t.encode(), slot, pointer)
                  for t, (slot, pointer) in zip(emission.types, bindings, strict=True)]
    made = UpstreamEmission(source, emission.entry.encode(), _array(UpstreamParameter, parameters) if parameters
                            else None, texts[0], texts[1], len(source), len(parameters), len(emission.compile),
                            len(emission.link))
    status = lib.metal_ir_upstream_emitted(program, op, number, C.byref(made), error, len(error))
    if status:
        raise OSError(status, 'operation %d pass %d: %s' % (op, number, error.value.decode()))


# metal_ir_upstream_configured: produced kernel `op`'s first pass's configured launch, its fields by name
def _configured(program, op):
    made = _upstream().metal_ir_upstream_configured(program, op)
    if not made:
        raise OSError(errno.ENODATA, 'operation %d: no configured first pass (metal_ir_upstream_emitted)' % op)
    return {name: tuple(getattr(made[0], name)) for name in ('grid', 'group', 'groups')}


# metal_ir_upstream_footprints: produced kernel `op`'s threadgroup footprints as its producer gives them (binding,
# exact, start, extent, ((dim, coefficient, divisor, modulus), ...)), each its slice of the terms
def _footprints(program, op, footprints):
    made, terms = [], []
    for binding, exact, start, extent, over in footprints:
        made.append(Footprint(0, binding, exact, len(terms), len(over), 0, start, extent))
        terms += [Term(d, 0, c, q, m) for d, c, q, m in over]
    _upstream().metal_ir_upstream_footprints(program, op, len(made), _array(Footprint, made), len(terms),
                                             _array(Term, terms))


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


# metal-microbench metal_ir.h metal_ir_value, metal_ir_generic, metal_ir_states, metal_ir_check, metal_ir_lapack,
# metal_ir_sort, metal_ir_repeat and the region builder: construction's typed records (compiler/mesh_compiler/
# operations.py) encoded as one C program, each storage root one value with its role, each view a value over it;
# `program` an existing program to encode into (metal_ir_append's destination), else a new one; `emitted` each leaf
# record's actual emission (Emitted) and `repeated` each registered repetition (Repeated), in emission order
class _Builder:
    def __init__(self, lib, identity, constants=None, program=None):
        self.lib = lib
        self.p = lib.metal_ir_program_new() if program is None else program
        self.emitted, self.repeated, self.regions = [], [], []
        self.values, self.roots, self.roles, self.builtins = {}, {}, {}, {}
        self.constants = {} if constants is None else constants
        self.base = self.identity = identity

    def root(self, t):
        key = _key(t)
        if key in self.roots:
            return self.roots[key]
        base = _stored(t)
        role = CONSTANT if key in self.constants or base.device.type != 'meta' else TEMPORARY
        # metal-microbench metal_ir.h metal_ir_value `identity`, metal_ir.mm metal_ir_append
        v = self.whole(base, role, 0 if role == CONSTANT else self.identity)
        if role == CONSTANT:
            self.attach(v, self.constants.get(key, base))
        self.roots[key], self.roles[key] = v, role
        return v

    # metal_ir.h METAL_IR_INPUT .. METAL_IR_STATE: a storage root whose role its caller states (an exported program's
    # InputKind, a result, a received channel), with the initial contents `data` where the role is CONSTANT, and its
    # caller's source identity (metal_ir_value `identity`; 0 none, so no definition is shared by identity alone)
    def declare(self, t, role, data=None, identity=0):
        key = _key(t)
        if key not in self.roots:
            v = self.whole(_stored(t), role, identity)
            if data is not None:
                self.attach(v, _stored(data))
            self.roots[key], self.roles[key] = v, role
        return self.roots[key]

    def whole(self, base, role, identity):
        return self.stored(role, SCALAR[base.dtype], base.untyped_storage().nbytes() // base.element_size(), identity)

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

    def value(self, t):
        base, words = (t.tensor, 2) if isinstance(t, _Words) else (t, 1)
        root = self.root(t)
        return self.viewed(root, self.roles[_key(t)], SCALAR[t.dtype], t.shape, [s * words for s in base.stride()],
                           base.storage_offset() * words)

    # metal_ir_upstream: one produced kernel (`record` the producer's Kernel, its retained owner) as one operation at
    # `place`, its identity the builder's base with source position `node`: `arguments` (kind UPSTREAM_*, ValueID,
    # access) in its launch's order, `views` (argument, byte offset, element bytes, ((size, byte stride), ...), write)
    # in that argument's storage root's bytes, `sources` the source positions it implements, `grid` its total threads
    # ((1, 1, 1) where its launch has none: a CPU kernel's one typed call) and `group` its threadgroup, None its
    # pipeline's (METAL_IR_DEFAULT_GROUP); kept as an emission (Emitted) and returned
    def upstream(self, record, place, node, arguments, views, sources, grid, group):
        lib = _upstream()
        self.identity = self.base | node
        pairs = [_array(mesh.Q, [n for pair in made for n in pair]) for _, _, _, made, _ in views]
        viewed = [mesh_graph.Viewed(argument, offset, element, len(made), 0,
                                    C.cast(held, C.POINTER(C.c_uint64)), 0, int(write))
                  for (argument, offset, element, made, write), held in zip(views, pairs, strict=True)]
        body = Upstream(place, len(arguments), len(views), len(sources),
                        _array(UpstreamArgument, [UpstreamArgument(kind, value, access, 0)
                                                  for kind, value, access in arguments]),
                        _array(mesh_graph.Viewed, viewed), _array(mesh.Q, [self.base | s for s in sources]),
                        self.identity, (mesh.U * 3)(*grid), (mesh.U * 3)(*(group or (0, 0, 0))),
                        int(group is not None), 0)
        first = lib.metal_ir_operations(self.p)
        operation = lib.metal_ir_upstream(self.p, C.byref(body))
        if operation == NONE:
            raise OSError(errno.EINVAL, 'a produced kernel\'s operand (metal_ir_upstream): an argument names no '
                                        'program value')
        self.emitted.append(Emitted(record, operation, first, lib.metal_ir_operations(self.p)))
        return operation

    # metal_ir.h metal_ir_generic over program values: one elementwise operation at `place` and source position
    # `node`, its loops `sizes`, each operand's dimension d at loop d (its value's own layout the view), `inputs` their
    # ValueIDs, one assigned output `output` of metal_ir scalar `scalar`; its BODY `body(region, parameters)` the
    # element it yields from its inputs' elements (metal_ir_parameter k: input k's). The direct projection of a copy, a
    # fill and a landing; `record` the producer object it implements, kept as its emission (Emitted)
    def elementwise(self, record, place, node, sizes, inputs, output, scalar, body):
        lib, p, rank = self.lib, self.p, len(sizes)
        self.identity = self.base | node
        terms = [Term(d, 0, 1, 1, 0) for d in range(rank)]
        exprs = [Expr(d, 1, 0) for _ in range(len(inputs) + 1) for d in range(rank)]
        first = lib.metal_ir_operations(p)
        operation = lib.metal_ir_generic(p, rank, _array(mesh.Q, sizes), _array(mesh.U, [PARALLEL] * rank),
                                         len(inputs), _array(mesh.U, inputs), 1, _array(mesh.U, [output]),
                                         _array(mesh.U, [ASSIGN]), _array(mesh.U, [IDENTITY]),
                                         _array(mesh.U, [CLASS[scalar]]), _array(Expr, exprs), len(terms),
                                         _array(Term, terms), 0, place, self.identity)
        r = lib.metal_ir_region(p, operation, BODY)
        made = body(r, [lib.metal_ir_parameter(p, r, k) for k in range(len(inputs))])
        lib.metal_ir_yield(p, r, 1, _array(mesh.U, [made]))
        self.emitted.append(Emitted(record, operation, first, lib.metal_ir_operations(p)))
        return operation

    # an elementwise copy of `given` into `into`, both (ValueID, metal_ir scalar) at loops `sizes`: its element as loaded
    # (one metal_ir_convert where the stored kinds compute differently: operations.py CLASS)
    def copy(self, record, place, node, sizes, into, given):
        def body(r, parameters):
            if CLASS[given[1]] == CLASS[into[1]]:
                return parameters[0]
            return self.lib.metal_ir_convert(self.p, r, CLASS[into[1]], parameters[0])
        return self.elementwise(record, place, node, sizes, [given[0]], into[0], into[1], body)

    # `into` (ValueID, metal_ir scalar) at loops `sizes` filled with the literal `bits` (metal_ir_literal's)
    def fill(self, record, place, node, sizes, into, bits):
        return self.elementwise(record, place, node, sizes, [], into[0], into[1],
                                lambda r, parameters: self.lib.metal_ir_literal(self.p, r, CLASS[into[1]], bits))

    # metal_ir.h metal_ir_lapack: one typed LAPACK call (struct metal_ir_lapack_call: `routine` LAPACK_GESDD ..
    # LAPACK_GELSS, `job` its JOB characters, `rcond`) at source position `node` over `batch` (its batch extents): each
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

    def builtin(self, name):
        if name not in self.builtins:
            self.builtins[name] = self.lib.metal_ir_builtin_named(name.encode())
        return self.builtins[name]

    def replay(self, r, region, inputs):
        lib, p, h = self.lib, self.p, []
        for record in region.code:
            call = record[0]
            if call == 'parameter':
                h.append(lib.metal_ir_parameter(p, r, record[1]))
            elif call == 'loop':
                h.append(lib.metal_ir_parameter(p, r, inputs + record[1]))
            elif call == 'literal':
                h.append(lib.metal_ir_literal(p, r, record[1], record[2]))
            elif call == 'binary':
                h.append(lib.metal_ir_binary(p, r, record[1], h[record[2]], h[record[3]]))
            elif call == 'unary':
                h.append(lib.metal_ir_unary(p, r, record[1], h[record[2]]))
            elif call == 'convert':
                h.append(lib.metal_ir_convert(p, r, record[1], h[record[2]]))
            elif call == 'bitcast':
                h.append(lib.metal_ir_bitcast(p, r, record[1], h[record[2]]))
            elif call == 'builtin':
                given = _array(mesh.U, [h[x] for x in record[2]])
                h.append(lib.metal_ir_builtin(p, r, self.builtin(record[1]), len(record[2]), given))
            elif call == 'extract':
                given = _array(mesh.U, [h[x] for x in record[2]])
                h.append(lib.metal_ir_extract(p, r, record[1], len(record[2]), given))
            elif call == 'extract_if':
                given = _array(mesh.U, [h[x] for x in record[2]])
                h.append(lib.metal_ir_extract_if(p, r, record[1], len(record[2]), given, h[record[3]], h[record[4]]))
            elif call == 'scatter':
                # metal_ir.h metal_ir_scatter: a predicate of None the store always (UINT32_MAX)
                given = _array(mesh.U, [h[x] for x in record[2]])
                lib.metal_ir_scatter(p, r, record[1], len(record[2]), given, h[record[3]],
                                     NONE if record[4] is None else h[record[4]])
                h.append(None)
            elif call == 'yield':
                given = _array(mesh.U, [h[x] for x in record[1]])
                lib.metal_ir_yield(p, r, len(record[1]), given)
                h.append(None)
            # metal_ir.h metal_ir_if_begin .. metal_ir_if_end (compiler/mesh_compiler/lower_inductor.py _Loop.masked): the
            # records between if_begin and if_else build its true block, those to if_end its false block; if_end's
            # record the first merged result, ('if_result', that record, k) result k
            elif call == 'if_begin':
                lib.metal_ir_if_begin(p, r, h[record[1]])
                h.append(None)
            elif call == 'if_else':
                given = _array(mesh.U, [h[x] for x in record[1]])
                lib.metal_ir_if_else(p, r, len(record[1]), given)
                h.append(None)
            elif call == 'if_end':
                given = _array(mesh.U, [h[x] for x in record[1]])
                h.append(lib.metal_ir_if_end(p, r, len(record[1]), given))
            elif call == 'if_result':
                h.append(h[record[1]] + record[2])
            else:
                # Rejection filters are not part of even temporary integration validation: an operation operations.py
                # _Region records that this encoder has no IR call for is unsupported here, not another one
                raise NotImplementedError('a region\'s record %r (%s): no metal_ir.h instruction encodes it'
                                          % (record[:2], KINDS.get(region.kinds[len(h)])))
            if h[-1] == NONE:
                raise NotImplementedError('a region\'s %s of %s: not an instruction of the IR (metal_ir.h, UINT32_MAX)'
                                          % (' '.join(str(x) for x in record[:2]), KINDS.get(region.kinds[len(h) - 1])))

    # one record: a repetition (operations.py _Repeat) through `repeat`, None (a repetition is no operation); a leaf
    # record encoded (`encoded`), its actual emission kept (Emitted: the record, its returned operation and the
    # interval metal_ir_operations gives around it), its operation returned
    def generic(self, g):
        if isinstance(g, _Repeat):
            return self.repeat(g)
        if isinstance(g, _Switch):
            return self.switch(g)
        first = self.lib.metal_ir_operations(self.p)
        operation = self.encoded(g)
        self.emitted.append(Emitted(g, operation, first, self.lib.metal_ir_operations(self.p)))
        return operation

    # `regions` the open regions while their records are emitted, innermost last, each [its direct children
    # (RegionChild), its open arm: a switch's branch, UINT32_MAX a repetition's]: a region registered there (postorder)
    # the innermost open one's child in that arm
    def enclosed(self, kind, region):
        if self.regions:
            children, arm = self.regions[-1]
            children.append(RegionChild(kind, region, arm))

    # metal_ir.h metal_ir_repeat: one pass's records emitted in order through `generic` (each region in it registered
    # first, its direct child), then the pass's whole operation interval registered once with its condition's ValueID
    # (a one-element int32 view), its carried tensors' storage roots and its direct children; the region (Repeated)
    # kept beside the leaves' emissions, never as an operation
    def repeat(self, g):
        first, children = self.lib.metal_ir_operations(self.p), []
        self.regions.append([children, NONE])
        for record in g.records:
            self.generic(record)
        self.regions.pop()
        end, carried = self.lib.metal_ir_operations(self.p), [self.root(t) for t in g.carried]
        region = self.lib.metal_ir_repeat(self.p, first, end, self.value(g.condition), len(carried),
                                          _array(mesh.U, carried), len(children), _array(RegionChild, children))
        self.repeated.append(Repeated(g, region, first, end))
        self.enclosed(REGION_REPEAT, region)

    # metal_ir.h metal_ir_switch_selector and metal_ir_switch (registration only: it constructs no copy), in
    # postorder: at the switch's source control point, before its branches' operations, its selector, one operation of
    # its index's ValueID (a one-element integer view, operations._stored as each operand) at the switch's place and
    # identity (`identity`, as `encoded`), emitted as its record's (Emitted); then each branch k's records emitted once
    # in order through `generic` (each region in it registered first, its direct child in arm k), then
    # operations._copy of each of its yields into its common output, at the switch's own place and node, inside that
    # branch, its interval [first, end) taken around both; then the switch registered once with its index, the yields'
    # own ValueIDs branch-major and the common outputs', its selector and its direct children (`regions`); itself so
    # the innermost open region's child. The switch region is no operation
    def switch(self, g):
        first, end, yielded, children = [], [], [], []
        self.identity = self.base | g.node
        index, before = self.value(_stored(g.index)), self.lib.metal_ir_operations(self.p)
        selector = self.lib.metal_ir_switch_selector(self.p, index, len(g.branches), g.place, self.identity)
        self.emitted.append(Emitted(g, selector, before, self.lib.metal_ir_operations(self.p)))
        opened = [children, 0]
        self.regions.append(opened)
        for k, (records, yields) in enumerate(g.branches):
            opened[1] = k
            first.append(self.lib.metal_ir_operations(self.p))
            for record in records:
                self.generic(record)
            for out, value in zip(g.outputs, yields, strict=True):
                written = _copy(out, value)
                _placed(written, g.place, g.node)
                for record in written:
                    self.generic(record)
            end.append(self.lib.metal_ir_operations(self.p))
            yielded += [self.value(_stored(value)) for value in yields]
        self.regions.pop()
        results = [self.value(_stored(out)) for out in g.outputs]
        region = self.lib.metal_ir_switch(
            self.p, index, len(g.branches), _array(mesh.U, first), _array(mesh.U, end), len(g.outputs),
            _array(mesh.U, yielded), _array(mesh.U, results), selector, len(children), _array(RegionChild, children))
        self.enclosed(REGION_SWITCH, region)

    # metal_ir.h `identity`: the producer's identity of an operation and of the values it first names, (function,
    # call ordinal, FX node position) of the call's graph, kept through every transformation (metal_ir_operation_info)
    def encoded(self, g):
        if isinstance(g, _Check):
            self.identity = self.base | g.node
            operation = self.lib.metal_ir_check(self.p, self.value(g.info), g.code, self.identity)
            if g.message:
                self.lib.metal_ir_check_message(self.p, operation, g.message.encode())
            if g.dep_token is not None:
                self.lib.metal_ir_check_dep_token(self.p, operation, self.value(_stored(g.dep_token)))
            return operation
        if isinstance(g, _Sort):
            self.identity = self.base | g.node
            return self.lib.metal_ir_sort(self.p, self.value(g.x), g.axis, g.k, int(g.descending),
                                          self.value(g.values), self.value(g.indices),
                                          0xFFFFFFFF if g.extent is None else self.value(g.extent), g.place,
                                          self.identity)
        self.identity = self.base | g.node
        lib, p = self.lib, self.p
        inputs, outputs = [self.value(t) for t, _ in g.inputs], [self.value(t) for t, _ in g.outputs]
        exprs, terms, loops = [], [], len(g.sizes)
        dims = loops + len(g.symbols)
        for t, rows in g.inputs + g.outputs:
            for d in range(len(t.shape)):
                row = rows[d] if rows is not None else (0,) * (dims + 1)
                first = len(terms)
                terms += [Term(l, 0, c, 1, 0) for l, c in enumerate(row[:dims]) if c]
                exprs.append(Expr(first, len(terms) - first, row[dims]))
        op = lib.metal_ir_generic(p, loops, _array(mesh.Q, g.sizes), _array(mesh.U, g.iterators), len(inputs),
                                  _array(mesh.U, inputs), len(outputs), _array(mesh.U, outputs),
                                  _array(mesh.U, g.combiner), _array(mesh.U, g.start), _array(mesh.U, g.accumulator),
                                  _array(Expr, exprs), len(terms), _array(Term, terms), g.precision, g.place,
                                  self.identity)
        if g.states:
            lib.metal_ir_states(p, op, len(g.states), _array(mesh.U, [kind for kind, _ in g.states]),
                                _array(mesh.Q, [_bits(identity, kind) for kind, identity in g.states]))
        # metal_ir_symbol in the record's order (its dimension loops + k), each scalar its own typed value, then each
        # loop's metal_ir_loop_extent where a symbol clips it
        symbol = [lib.metal_ir_symbol(p, op, self.value(t), bound) for t, bound in g.symbols]
        for loop, k in enumerate(g.extents):
            if k is not None:
                lib.metal_ir_loop_extent(p, op, loop, symbol[k])
        self.replay(lib.metal_ir_region(p, op, BODY), g.body, len(inputs))
        if g.states:
            self.replay(lib.metal_ir_region(p, op, COMBINE), g.combine, 0)
            self.replay(lib.metal_ir_region(p, op, FINALIZE), g.finalize, 0)
        return op
