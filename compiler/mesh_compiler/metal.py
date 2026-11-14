# design/compilation-composition-change-request.md items 1, 3, 6 and 7; design/aot-compilation-change-request.md A8 and
# A9: the live native owner. The C ABI's types and library bindings (metal-microbench metal_ir.h, metal_recording.h;
# rdma/mesh-metal.h), the Objective-C/Metal resource primitives, borrowed buffers, archive byte ownership, the run's
# waits, and the typed C encoder of construction's records (_Builder), whose numerical records and tables are
# compiler/mesh_compiler/operations.py's
import ctypes as C
import ctypes.util
import errno
import os
import pathlib
import sys
import time

import torch

from .operations import CPU, GPU, KINDS, SCALAR, _Check, _Lapack, _Sort, _Words, _bits, _key, _rows, _stored

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh
import mesh_graph

# metal-microbench metal_ir.h METAL_IR_INPUT .. METAL_IR_STATE (a value's role), its region kinds, UINT32_MAX (no
# value), and METAL_IR_DATA, METAL_IR_TRANSPORT, METAL_IR_STEP (metal_ir_value_info `word` of the step value)
INPUT, CONSTANT, TEMPORARY, OUTPUT, STATE = range(5)
BODY, COMBINE, FINALIZE = range(3)
NONE = 0xFFFFFFFF
METAL_IR_STEP = 2
# metal_recording.h METAL_EXTERNAL_GPU: a callable operand the program uses on the GPU
METAL_EXTERNAL_GPU = 1
# metal_ir.h METAL_IR_SOURCE_VALUE .. METAL_IR_SOURCE_EXTERNAL: a GIVEN cause's operand kind (struct
# metal_ir_source_operand)
SOURCE_VALUE, SOURCE_LITERAL, SOURCE_SENT, SOURCE_RECEIVED, SOURCE_EXTERNAL = range(5)
# metal_recording.h METAL_UNIT_GPU, METAL_UNIT_HOST: an observation unit's kind
METAL_UNIT_GPU, METAL_UNIT_HOST = 0, 1


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
# metal_recording.h state:buffer:status:; MTLDevice.h newBufferWithLength:options:
STATE_BUFFER = _message(None, C.c_void_p, C.c_void_p, C.c_ulong)
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


class Release(C.Structure):
    _fields_ = [('supplied', mesh.Q), ('status', mesh.Q), ('total', mesh.Q), ('transfer', mesh.U), ('padding', mesh.U)]


class Sending(C.Structure):
    _fields_ = [('sent', MetalSent), ('extent', C.POINTER(mesh_graph.Access)), ('extents', C.c_size_t),
                ('transfer', mesh.U), ('padding', mesh.U)]


# metal_recording.h struct MetalCheckFailure
class CheckFailure(C.Structure):
    _fields_ = [('identity', mesh.Q), ('batch', mesh.Q), ('info', C.c_int64), ('code', mesh.U), ('padding', mesh.U)]


# metal_ir.h struct metal_ir_value_info and struct metal_ir_storage
class ValueInfo(C.Structure):
    _fields_ = [(name, mesh.U) for name in ('role', 'scalar', 'storage', 'function', 'index', 'word', 'of', 'range',
                                          'received', 'sent', 'zeroed', 'sendings')] + [
        (name, mesh.Q) for name in ('identity', 'bytes', 'offset', 'at', 'data_bytes')] + [
        ('data', C.c_void_p), ('buffer', C.c_void_p), ('release', Release), ('page_bytes', mesh.Q),
        ('pages', mesh.U), ('source', mesh.U)]


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
                ('MetalReplayDestroy', None, [C.POINTER(Replay)]),
                ('MetalRecorderActors', C.c_int, [P, machine, machine, C.POINTER(_Actors)]),
                ('MetalRecorderActorsFree', None, [C.POINTER(_Actors)]),
                ('MetalRecorderProgram', P, [P, C.POINTER(Call), U, C.POINTER(mesh.Planned), text, room]),
                ('MetalProgramSave', C.c_int, [P, C.POINTER(Span), U, C.POINTER(P), C.POINTER(C.c_size_t), text,
                                               room]),
                ('MetalProgramLoad', P, [P, C.POINTER(Call), U, C.POINTER(mesh.Planned), P, text, room]),
                ('MetalProgramPhases', None, [P, C.POINTER(Q), C.POINTER(Q)]),
                ('MetalProgramSignatureNode', C.c_int, [P, U, C.c_size_t, C.POINTER(Q),
                                                        C.POINTER(mesh_graph.Signature), text, room]),
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
                ('metal_ir_archive_load', P, [P, C.c_size_t, P, RELEASE, P, C.c_char_p, C.c_size_t]),
                ('metal_ir_archive_program', P, [P, U]), ('metal_ir_archive_free', None, [P]),
                ('metal_ir_loaded', C.c_int, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_value_info', None, [P, U, C.POINTER(ValueInfo)]),
                ('metal_ir_value_layout', U, [P, U, C.POINTER(Term), C.c_size_t, C.POINTER(C.c_int64)]),
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
                ('metal_ir_value_sending', None, [P, U, C.POINTER(Sending)]),
                ('metal_ir_transport', None, [P, Q]),
                ('metal_ir_generic', U, [P, U, C.POINTER(Q), C.POINTER(U), U, C.POINTER(U), U, C.POINTER(U),
                                         C.POINTER(U), C.POINTER(U), C.POINTER(U), C.POINTER(Expr), U, C.POINTER(Term),
                                         U, U, Q]),
                ('metal_ir_states', None, [P, U, U, C.POINTER(U), C.POINTER(Q)]),
                ('metal_ir_check', U, [P, U, U, Q]),
                ('metal_ir_check_message', None, [P, U, C.c_char_p]),
                ('metal_ir_lapack', U, [P, C.POINTER(Lapack), U, C.POINTER(Q), U, C.POINTER(U), U, C.POINTER(U),
                                        C.POINTER(Matrix), C.POINTER(Expr), U, C.POINTER(Term), Q]),
                ('metal_ir_sort', U, [P, U, U, Q, U, U, U, U, Q]),
                ('metal_ir_region', U, [P, U, U]), ('metal_ir_parameter', U, [P, U, U]),
                ('metal_ir_literal', U, [P, U, U, Q]), ('metal_ir_binary', U, [P, U, U, U, U]),
                ('metal_ir_unary', U, [P, U, U, U]), ('metal_ir_convert', U, [P, U, U, U]),
                ('metal_ir_bitcast', U, [P, U, U, U]), ('metal_ir_builtin', U, [P, U, U, U, C.POINTER(U)]),
                ('metal_ir_extract', U, [P, U, U, U, C.POINTER(U)]),
                ('metal_ir_extract_if', U, [P, U, U, U, C.POINTER(U), U, U]),
                ('metal_ir_builtin_named', U, [C.c_char_p]),
                ('metal_ir_scatter', None, [P, U, U, U, C.POINTER(U), U, U]),
                ('metal_ir_yield', None, [P, U, U, C.POINTER(U)]),
                ('metal_ir_lower', C.c_int, [P, P, C.c_char_p, C.c_size_t]),
                ('metal_ir_finalize', C.c_int, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_specialize', P, [P, C.c_char_p, C.c_size_t]),
                ('metal_ir_machine', None, [P, C.POINTER(mesh_graph.Machine), C.POINTER(mesh_graph.Machine)])):
            function = getattr(lib, name)
            function.restype, function.argtypes = result, arguments
        COMPILER.append(lib)
    return COMPILER[0]


# mesh-call.h mesh_transfers_start and mesh-dataflow.h mesh_host_started, the program's `start` and `started`
START = C.cast(mesh.LIB.mesh_transfers_start, C.c_void_p)
STARTED = C.cast(mesh.LIB.mesh_host_started, C.c_void_p)


# mesh-collective.h mesh_host_arrived; mesh.h MESH_PLATFORM_NS: wait for `ready`, the bound restarting at each change of
# `progress`
def _wait(ready, progress, what):
    seen, since = progress(), time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    while not ready():
        now, value = time.clock_gettime_ns(time.CLOCK_MONOTONIC), progress()
        if value != seen:
            seen, since = value, now
        elif now - since > mesh.PLATFORM_NS:
            raise OSError(errno.ETIMEDOUT, what)


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
# metal_ir_sort and the region builder: construction's typed records (compiler/mesh_compiler/operations.py) encoded as
# one C program, each storage root one value with its role, each view a value over it
class _Builder:
    def __init__(self, lib, owners, role, row, identity, constants=None):
        self.lib, self.p, self.owners, self.role, self.row = lib, lib.metal_ir_program_new(), owners, role, row
        self.values, self.roots, self.roles, self.planned, self.builtins = {}, {}, {}, {}, {}
        self.constants = {} if constants is None else constants
        self.base = self.identity = identity

    def root(self, t):
        key = _key(t)
        if key in self.roots:
            return self.roots[key]
        base = _stored(t)
        owner = self.owners.get(key)
        role = CONSTANT if key in self.constants else self.role(owner) if owner is not None else \
            CONSTANT if base.device.type != 'meta' else TEMPORARY
        # metal-microbench metal_ir.h metal_ir_value `identity`, metal_ir.mm metal_ir_append
        v = self.whole(base, role, 0 if role == CONSTANT else self.identity)
        if owner is not None:
            k, index, _, pages, row = owner
            self.lib.metal_ir_value_planned(self.p, v, k, index, 0)
            if pages:
                self.lib.metal_ir_value_paged(self.p, v, row, pages)
            self.planned[v] = owner
        if key in self.constants or owner is None and base.device.type != 'meta':
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
        numel = base.untyped_storage().nbytes() // base.element_size()
        return self.lib.metal_ir_value(self.p, role, SCALAR[base.dtype], 1, _array(mesh.Q, [numel]), 1,
                                       _array(Term, [Term(0, 0, 1, 1, 0)]), 0, identity)

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
        strides = tuple(s * words for s in base.stride())
        signature = (_key(t), SCALAR[t.dtype], tuple(t.shape), strides, base.storage_offset() * words)
        if signature not in self.values:
            root = self.root(t)
            v = self.lib.metal_ir_value(self.p, self.roles[_key(t)], SCALAR[t.dtype], len(t.shape),
                                        _array(mesh.Q, t.shape), len(strides),
                                        _array(Term, [Term(d, 0, s, 1, 0) for d, s in enumerate(strides)]),
                                        signature[4], self.identity)
            self.lib.metal_ir_value_view(self.p, v, root)
            self.values[signature] = v
        return self.values[signature]

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
                given = _array(mesh.U, [h[x] for x in record[2]])
                lib.metal_ir_scatter(p, r, record[1], len(record[2]), given, h[record[3]], h[record[4]])
                h.append(None)
            else:
                given = _array(mesh.U, [h[x] for x in record[1]])
                lib.metal_ir_yield(p, r, len(record[1]), given)
                h.append(None)
            if h[-1] == NONE:
                raise NotImplementedError('a region\'s %s of %s: not an instruction of the IR (metal_ir.h, UINT32_MAX)'
                                          % (' '.join(str(x) for x in record[:2]), KINDS.get(region.kinds[len(h) - 1])))

    # metal_ir.h `identity`: the producer's identity of an operation and of the values it first names, (function,
    # call ordinal, FX node position) of the call's graph, kept through every transformation (metal_ir_operation_info)
    def generic(self, g):
        if isinstance(g, _Check):
            self.identity = self.base | g.node
            operation = self.lib.metal_ir_check(self.p, self.value(g.info), g.code, self.identity)
            if g.message:
                self.lib.metal_ir_check_message(self.p, operation, g.message.encode())
            return operation
        if isinstance(g, _Lapack):
            if g.place != CPU:
                raise NotImplementedError('a LAPACK driver\'s node on a processor other than the CPU: its lowering is '
                                          'unfinished (design/torch-source-lowering-change-request.md item 6)')
            self.identity = self.base | g.node
            sizes, picks, matrices, operands = list(g.batch), [], [], []
            for t, rank in g.inputs + g.outputs:
                real = torch.view_as_real(t) if t.is_complex() else t
                own = rank + t.is_complex()
                lead, first = real.dim() - own, len(g.batch) - (real.dim() - own)
                picks.append([None if n == 1 and g.batch[first + d] != 1 else first + d
                              for d, n in enumerate(real.shape[:lead])] + list(range(len(sizes), len(sizes) + own)))
                loops = list(range(len(sizes), len(sizes) + rank)) + [NONE] * (2 - rank)
                matrices.append(Matrix(loops[0], loops[1], len(sizes) + rank if t.is_complex() else NONE))
                sizes += real.shape[lead:]
                operands.append(self.value(real))
            exprs, terms = [], []
            for pick in picks:
                for row in _rows(len(sizes), pick):
                    first = len(terms)
                    terms += [Term(l, 0, c, 1, 0) for l, c in enumerate(row[:len(sizes)]) if c]
                    exprs.append(Expr(first, len(terms) - first, row[len(sizes)]))
            call = Lapack(g.routine, len(g.batch), g.job, g.rcond)
            return self.lib.metal_ir_lapack(self.p, C.byref(call), len(sizes), _array(mesh.Q, sizes), len(g.inputs),
                                            _array(mesh.U, operands[:len(g.inputs)]), len(g.outputs),
                                            _array(mesh.U, operands[len(g.inputs):]), _array(Matrix, matrices),
                                            _array(Expr, exprs), len(terms), _array(Term, terms), self.identity)
        if isinstance(g, _Sort):
            self.identity = self.base | g.node
            return self.lib.metal_ir_sort(self.p, self.value(g.x), g.axis, g.k, int(g.descending),
                                          self.value(g.values), self.value(g.indices), g.place, self.identity)
        self.identity = self.base | g.node
        lib, p = self.lib, self.p
        inputs, outputs = [self.value(t) for t, _ in g.inputs], [self.value(t) for t, _ in g.outputs]
        exprs, terms, loops = [], [], len(g.sizes)
        for t, rows in g.inputs + g.outputs:
            for d in range(len(t.shape)):
                row = rows[d] if rows is not None else (0,) * (loops + 1)
                first = len(terms)
                terms += [Term(l, 0, c, 1, 0) for l, c in enumerate(row[:loops]) if c]
                exprs.append(Expr(first, len(terms) - first, row[loops]))
        op = lib.metal_ir_generic(p, loops, _array(mesh.Q, g.sizes), _array(mesh.U, g.iterators), len(inputs),
                                  _array(mesh.U, inputs), len(outputs), _array(mesh.U, outputs),
                                  _array(mesh.U, g.combiner), _array(mesh.U, g.start), _array(mesh.U, g.accumulator),
                                  _array(Expr, exprs), len(terms), _array(Term, terms), g.precision, g.place,
                                  self.identity)
        if g.states:
            lib.metal_ir_states(p, op, len(g.states), _array(mesh.U, [kind for kind, _ in g.states]),
                                _array(mesh.Q, [_bits(identity, kind) for kind, identity in g.states]))
        self.replay(lib.metal_ir_region(p, op, BODY), g.body, len(inputs))
        if g.states:
            self.replay(lib.metal_ir_region(p, op, COMBINE), g.combine, 0)
            self.replay(lib.metal_ir_region(p, op, FINALIZE), g.finalize, 0)
        return op
