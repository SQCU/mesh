import ctypes as C
import os
import time

import numpy as np

U, Q, Z = C.c_uint32, C.c_uint64, C.c_size_t


class Context(C.Structure):
    _fields_ = [('M', C.c_void_p), ('len', Z), *((k, Q) for k in ('client', 'send_off', 'send_bytes')),
                *((k, U) for k in ('rows', 'arena', 'wire', 'shared_pages', 'row_cursor', 'wire_cursor', 'bulk_cursor')),
                ('fd', C.c_int)]


class Header(C.Structure):
    _fields_ = [(k, U) for k in ('magic', 'version', 'pgsz', 'block', 'rows', 'node', 'qps', 'links')]


class Section(C.Structure):
    _fields_ = [('first', U), ('pages', U), ('bytes', Z), ('count', U), ('stride', U)]


class Operand(C.Structure):
    _fields_ = [('type', U), ('element_bytes', U), ('elements', Q)]


class Step(C.Structure):
    _fields_ = [('op', U), ('peer', U), ('round', U), ('padding', U), ('first', Q), ('piece', Operand)]


class LinkMap(C.Structure):
    _fields_ = [('kind', U), ('nodes', U), ('links', U), ('link', U * 2 * (32 * 31 // 2))]


SEND, REDUCE, COPY = range(3)
DIRECT, CANCELLED = 0, 2 ** 64 - 1
LIB = C.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libmesh.dylib'))
CONTEXT = C.POINTER(Context)
for name, result, arguments in (
        ('mesh_attach', C.c_int, [CONTEXT, C.c_char_p]),
        ('mesh_detach', C.c_int, [CONTEXT]),
        ('mesh_link_map_read', C.c_int, [C.c_char_p, C.POINTER(LinkMap)]),
        ('mesh_allreduce_plan', U, [C.POINTER(LinkMap), U, Operand, C.POINTER(Step)]),
        ('mesh_section_create', C.c_int, [CONTEXT, Z, U, C.c_int, C.POINTER(Section)]),
        ('mesh_section_slice', C.c_int, [CONTEXT, Section, Z, Z, U, U, C.POINTER(Section)]),
        ('mesh_section_address', C.c_void_p, [CONTEXT, Section, U]),
        ('mesh_allreduce_bind', C.c_int, [CONTEXT, C.POINTER(Step), U, U, Section, C.POINTER(Section), U, U,
                                          C.POINTER(Section)]),
        ('mesh_transfers_prepare', C.c_int, [CONTEXT, U, U, U]),
        ('mesh_host_inputs', C.c_int, [CONTEXT]),
        ('mesh_transfers_start', C.c_int, [CONTEXT]),
        ('mesh_host_publish', None, [CONTEXT, Section, U]),
        ('mesh_host_arrived', Q, [CONTEXT, Section, U])):
    function = getattr(LIB, name)
    function.restype, function.argtypes = result, arguments


def check(status):
    if status:
        raise OSError(status, os.strerror(status))


class AllReduce:
    """One rank's all-reduce of a typed array over an explicit link map (mesh-collective.h), prepared
    once for `invocations` calls.  The rank is the bridge's node; storage rings over `depth` slots."""

    def __init__(self, shape, dtype, links, invocations, depth=4, identity=1, region=None):
        self.context = Context()
        context = C.byref(self.context)
        check(LIB.mesh_attach(context, os.fsencode(region) if region else None))
        header = Header.from_address(self.context.M)
        link_map = LinkMap()
        check(LIB.mesh_link_map_read(os.fsencode(links), C.byref(link_map)))
        self.rank, self.shape, self.dtype, self.depth = header.node, tuple(shape), np.dtype(dtype), depth
        self.direct = link_map.kind == DIRECT
        steps = (Step * (4 * link_map.nodes))()
        count = LIB.mesh_allreduce_plan(C.byref(link_map), self.rank,
            Operand(ord(self.dtype.char), self.dtype.itemsize, int(np.prod(shape))), steps)
        self.steps = steps[:count]

        def ring(nbytes):
            pages = -(-nbytes // (header.pgsz * header.block)) * header.block
            section = Section()
            check(LIB.mesh_section_create(context, depth * pages * header.pgsz, 1, 1, C.byref(section)))
            memory = (C.c_char * (depth * pages * header.pgsz)).from_address(LIB.mesh_section_address(context, section, 0))
            return section, pages, np.frombuffer(memory, np.uint8).reshape(depth, -1)

        self.bytes = int(np.prod(shape)) * self.dtype.itemsize
        self.operand, pages, self.slots = ring(self.bytes)
        received, self.received = [], []
        for step in self.steps:
            if step.op == REDUCE:
                section, step_pages, slots = ring(step.piece.elements * step.piece.element_bytes)
                received.append(Section())
                check(LIB.mesh_section_slice(context, section, 0, step.piece.elements * step.piece.element_bytes,
                                             depth, step_pages, C.byref(received[-1])))
                self.received.append(slots)
        self.pieces = (Section * count)()
        check(LIB.mesh_allreduce_bind(context, steps, count, identity, self.operand,
                                      (Section * max(1, len(received)))(*received), depth, pages, self.pieces))
        check(LIB.mesh_transfers_prepare(context, 1, invocations, depth))
        check(LIB.mesh_host_inputs(context))
        check(LIB.mesh_transfers_start(context))

    def slot(self, invocation):
        """Where this rank writes its contribution to `invocation`, in registered pages."""
        return self.slots[invocation % self.depth, :self.bytes].view(self.dtype).reshape(self.shape)

    def __call__(self, invocation, deadline=float('inf')):
        """Runs this rank's steps in plan order and returns the reduced array.  A direct exchange
        sums into a copy, because its SEND may still be reading the slot; a ring or tree sums in place."""
        own = self.slot(invocation)
        total = own.copy() if self.direct else own
        flat, context, received = total.reshape(-1), C.byref(self.context), iter(self.received)
        for step, piece in zip(self.steps, self.pieces):
            if step.op == SEND:
                LIB.mesh_host_publish(context, piece, invocation)
                continue
            while not (arrived := LIB.mesh_host_arrived(context, piece, invocation)):
                if time.monotonic() > deadline:
                    raise TimeoutError(f'invocation {invocation} step {step.op} from {step.peer}')
            if arrived == CANCELLED:
                raise ConnectionAbortedError(f'invocation {invocation} cancelled on the link to {step.peer}')
            if step.op == REDUCE:
                count = step.piece.elements
                flat[step.first:step.first + count] += next(received)[invocation % self.depth, :count * self.dtype.itemsize].view(self.dtype)
        return total

    def close(self, linger=0.5):
        """Holds the pair open while the peer's last receive lands, then retires this client."""
        time.sleep(linger)
        return LIB.mesh_detach(C.byref(self.context))
