import ctypes as c
import errno
import os

import numpy as np

_lib = c.CDLL(os.path.join(os.path.dirname(__file__), "libmesh.dylib"), use_errno=True)
ABSENT = (1 << 32) - 1
WRITING = 1 << 63


class Row(c.Structure):
    _fields_ = [("page", c.c_uint32), ("uses", c.c_uint32), ("stamp", c.c_uint64)]


class RowRange(c.Structure):
    _fields_ = [("first", c.c_uint32), ("count", c.c_uint32)]


class RowMap(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("first", "count", "stride", "physical", "physical_stride", "immutable")] + [("uses", c.POINTER(c.c_uint32)), ("ranges", c.POINTER(RowRange))]


class RowFunction(c.Structure):
    _fields_ = [("input", c.POINTER(RowMap)), ("output", c.POINTER(RowMap)), ("inputs", c.c_uint32), ("outputs", c.c_uint32), ("rows", c.c_uint32), ("indices", RowMap)]


class RowBinding(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("first", "count", "remote", "headers")] + [("uses", c.POINTER(c.c_uint32)), ("peer", c.c_uint16), ("receive", c.c_uint16), ("remote_table", c.c_uint64)]


class Region(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("magic", "version", "pgsz", "pool", "arena", "node")] + [("headers_off", c.c_uint64), ("data_off", c.c_uint64)]


class Rows(c.Structure):
    pass


class PageHeader(c.Structure):
    _fields_ = [(name, c.c_uint16) for name in ("source_node", "destination_node", "hops", "padding")] + [
        ("table", c.c_uint64), ("stamp", c.c_uint64), ("source", c.c_uint32), ("target", c.c_uint32),
        ("when", c.c_uint64), ("code", c.c_int64), ("domain", c.c_uint32), ("function", c.c_uint32),
        ("index", c.c_uint32), ("peer", c.c_uint32)]


class Context(c.Structure):
    _fields_ = [("M", c.POINTER(Region)), ("arena", c.c_void_p), ("len", c.c_size_t), ("tables", c.POINTER(c.POINTER(Rows))), ("table_count", c.c_size_t), ("allocation", c.c_size_t), ("mapping_pinned", c.c_int)]


Rows._fields_ = [("memory", c.POINTER(Region)), ("table", c.POINTER(Row)), ("count", c.c_size_t), ("offset", c.c_uint32), ("bytes", c.c_uint32), ("context", c.POINTER(Context)), ("identity", c.c_uint64), ("bindings", c.POINTER(RowBinding)), ("binding_count", c.c_size_t), ("functions", c.POINTER(RowFunction)), ("function_count", c.c_size_t), ("returns", c.POINTER(RowMap)), ("return_count", c.c_size_t)]


class Metadata(c.Structure):
    _fields_ = [("stamp", c.c_uint64), ("when", c.c_uint64), ("function", c.c_uint32), ("index", c.c_uint32), ("peer", c.c_uint32), ("code", c.c_int64), ("domain", c.c_uint32), ("reserved", c.c_uint32)]


for name, result, arguments in (
    ("mesh_context", c.POINTER(Context), []),
    ("mesh_region", c.POINTER(Region), [c.POINTER(Context)]),
    ("mesh_peers", c.c_size_t, [c.POINTER(Context), c.POINTER(c.c_uint16), c.c_size_t]),
    ("mesh_link_metadata", Metadata, [c.POINTER(Context), c.c_size_t]),
    ("mesh_attach", c.c_int, [c.POINTER(Context), c.c_char_p]),
    ("mesh_detach", c.c_int, [c.POINTER(Context)]),
    ("mesh_rows_create", c.POINTER(Rows), [c.POINTER(Context), c.c_size_t, c.c_uint64]),
    ("mesh_rows_allocate", c.c_uint32, [c.POINTER(Rows), c.c_size_t, c.c_size_t]),
    ("mesh_storage_pages", c.c_size_t, [c.c_size_t, c.c_uint32]),
    ("mesh_region_table_pages", c.c_size_t, [c.c_size_t, c.c_uint32]),
    ("mesh_rows_configuration_pages", c.c_size_t, [c.c_uint32, c.c_size_t, c.POINTER(RowFunction), c.c_size_t, c.POINTER(RowBinding), c.c_size_t, c.POINTER(RowMap), c.c_size_t]),
    ("mesh_rows_realize", c.c_int, [c.POINTER(Rows), c.POINTER(RowFunction), c.c_size_t, c.POINTER(RowBinding), c.c_size_t, c.POINTER(RowMap), c.c_size_t]),
    ("mesh_rows_map", None, [c.POINTER(Rows), c.c_uint32, c.c_uint32, c.c_uint32, c.c_uint32, c.c_uint64]),
    ("mesh_rows_poll", c.c_size_t, [c.POINTER(Context)]),
    ("mesh_rows_issue", c.c_uint32, [c.POINTER(Rows), c.POINTER(RowFunction), c.c_uint64]),
    ("mesh_rows_complete", None, [c.POINTER(Rows), c.POINTER(RowFunction), c.c_uint64, c.c_uint32]),
    ("mesh_rows_publish", None, [c.POINTER(Rows), c.POINTER(RowFunction), c.c_uint32, c.c_uint64]),
    ("mesh_rows_report", None, [c.POINTER(Rows), RowMap, c.c_uint32, Metadata]),
    ("mesh_row_release", None, [c.POINTER(Rows), c.c_uint32]),
    ("mesh_row_data", c.c_void_p, [c.POINTER(Rows), c.c_uint32]),
    ("mesh_rows_present", c.c_int, [c.POINTER(Rows), RowMap, c.c_uint32, c.c_uint64]),
    ("mesh_context_metadata", c.POINTER(PageHeader), [c.POINTER(Context), c.c_uint32]),
    ("mesh_context_consume", c.c_int, [c.POINTER(Context), c.c_uint32]),
    ("mesh_rows_invalidate", c.c_int, [c.POINTER(c.POINTER(Rows)), c.POINTER(RowMap), c.c_size_t]),
):
    function = getattr(_lib, name)
    function.restype, function.argtypes = result, arguments


class Mesh:
    # design/algorithm-sources.md#complete-page-ownership
    def __init__(self, nbytes=16 << 20, *, peers=None, remote_table=0):
        self.region = os.environ.get("MESH_REGION", "/mesh0")
        self.context = _lib.mesh_context()
        status = _lib.mesh_attach(self.context, self.region.encode())
        if status:
            raise OSError(status, os.strerror(status))
        self.memory = _lib.mesh_region(self.context)
        self.stride = self.usable = self.memory.contents.pgsz
        self.slots = (nbytes + self.usable - 1) // self.usable
        configured = (c.c_uint16 * _lib.mesh_peers(self.context, None, 0))()
        _lib.mesh_peers(self.context, configured, len(configured))
        self.peers = tuple(dict.fromkeys(configured if peers is None else peers))
        self.native = None
        self.pages = _lib.mesh_rows_create(self.context, 2 * len(self.peers) * self.slots, self.context.contents.table_count)
        if not self.pages:
            raise OSError(c.get_errno(), "mesh page table allocation")
        self.first = _lib.mesh_rows_allocate(self.pages, len(self.peers) * self.slots, self.stride)
        if self.first == ABSENT:
            raise OSError(c.get_errno(), "mesh operand allocation")
        self.base = c.addressof(self.memory.contents) + self.memory.contents.data_off + self.first * self.stride
        self.outputs = (RowMap * len(self.peers))()
        self.functions = (RowFunction * len(self.peers))()
        self.bindings = (RowBinding * (2 * (max(self.peers + (self.memory.contents.node,)) + 1)))()
        self.returns = (RowMap * len(self.peers))()
        self.source = {}
        for index, peer in enumerate(self.peers):
            first = index * self.slots
            received = (len(self.peers) + index) * self.slots
            self.source[peer] = first
            self.outputs[index] = RowMap(first, 1, 1, self.first + first, 1, 0, None)
            self.functions[index] = RowFunction(None, c.pointer(self.outputs[index]), 0, 1, self.slots)
            self.bindings[2 * peer] = RowBinding(first, self.slots, 2 * self.memory.contents.node + 1, 0, None, peer, 0, remote_table)
            self.bindings[2 * peer + 1] = RowBinding(received, self.slots, 0, 0, None, peer, 1, remote_table)
            self.returns[index] = RowMap(received, self.slots, 0, 0, 0, 0, None)
        status = _lib.mesh_rows_realize(self.pages, self.functions, len(self.functions), self.bindings, len(self.bindings), self.returns, len(self.returns))
        if status:
            raise OSError(status, os.strerror(status))

    # design/algorithm-sources.md#complete-page-ownership
    def block(self, first, count, node=None):
        peer = self.peers[0] if node is None else node
        begin = self.source[peer] + first
        storage = (c.c_uint8 * (count * self.stride)).from_address(self.base + begin * self.stride)
        return np.ctypeslib.as_array(storage).reshape(count, self.stride)

    # design/algorithm-sources.md#complete-page-ownership
    def slot(self, index, dtype=np.float32, count=None, node=None):
        value = self.block(index, 1, node).reshape(-1).view(dtype)
        return value if count is None else value[:count]

    # design/algorithm-sources.md#complete-page-ownership
    def reserve(self, node, count):
        begin = self.source[node]
        table = self.pages.contents.table
        for first in range(self.slots - count + 1):
            if all(table[begin + first + offset].page == ABSENT for offset in range(count)):
                for index in range(begin + first, begin + first + count):
                    _lib.mesh_rows_map(self.pages, index, self.first + index, 1, 1, WRITING | (table[index].stamp + 1))
                return self.block(first, count, node)
        return None

    # design/algorithm-sources.md#complete-page-ownership
    def send(self, frames, node):
        first = (frames.ctypes.data - self.base) // self.stride
        begin = self.source[node]
        if frames.dtype != np.uint8 or frames.ndim != 2 or frames.shape[1] != self.stride or frames.strides != (self.stride, 1) or frames.ctypes.data != self.base + first * self.stride or first < begin or first + len(frames) > begin + self.slots:
            raise ValueError("transport input must be the configured node's literal mesh pages")
        for index in range(first, first + len(frames)):
            row = self.pages.contents.table[index]
            if row.page != self.first + index or not row.stamp & WRITING:
                raise ValueError("publish requires a reserved input page")
            _lib.mesh_rows_map(self.pages, index, self.first + index, 1, 1, row.stamp & ~WRITING)
        _lib.mesh_rows_poll(self.context)
        return len(frames)

    # design/algorithm-sources.md#complete-page-ownership
    def write(self, first, count, node):
        return self.send(self.block(first, count, node), node)

    # design/algorithm-sources.md#complete-page-ownership
    def pump(self):
        return _lib.mesh_rows_poll(self.context)

    # design/algorithm-sources.md#complete-page-ownership
    def inflight(self):
        table = self.pages.contents.table
        return sum(table[row].uses for row in range(len(self.peers) * self.slots))

    # design/algorithm-sources.md#complete-page-ownership
    def queued(self):
        return self.inflight()

    # design/algorithm-sources.md#complete-page-ownership
    def read(self, dtype=np.float32, max_batches=None):
        self.pump()
        for peer_index, peer in enumerate(self.peers):
            first = (len(self.peers) + peer_index) * self.slots
            for index in range(first, first + self.slots):
                row = self.pages.contents.table[index]
                if row.page == ABSENT or not row.stamp:
                    continue
                address = _lib.mesh_row_data(self.pages, index)
                value = np.ctypeslib.as_array((c.c_uint8 * self.stride).from_address(address)).view(dtype)
                try:
                    yield value, peer
                finally:
                    _lib.mesh_row_release(self.pages, index)

    # design/algorithm-sources.md#complete-page-ownership
    def close(self):
        if self.pages:
            status = _lib.mesh_rows_invalidate(c.byref(self.pages), self.returns, len(self.returns))
            if status: return status
        return _lib.mesh_detach(self.context)
