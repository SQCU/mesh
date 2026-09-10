import ctypes as c
import os

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / ".build"))
from mesh_abi import ABSENT, WRITING, Row, RowRange, RowMap, RowFunction, RowBinding, Region, Rows, Context, Metadata, MemorySpan, _lib


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
            self.bindings[2 * peer] = RowBinding(first, self.slots, 2 * self.memory.contents.node + 1, 0, peer, 0, remote_table)
            self.bindings[2 * peer + 1] = RowBinding(received, self.slots, 0, 0, peer, 1, remote_table)
            self.returns[index] = RowMap(received, self.slots, 0, 0, 0, 0, None)
        status = _lib.mesh_rows_realize(self.pages, self.functions, len(self.functions), self.bindings, len(self.bindings), self.returns, len(self.returns))
        if status:
            raise OSError(status, os.strerror(status))

    # design/algorithm-sources.md#complete-page-ownership
    def block(self, first, count, node=None):
        peer = self.peers[0] if node is None else node
        begin = self.source[peer] + first
        storage = (c.c_uint8 * (count * self.stride)).from_address(self.base + begin * self.stride)
        storage.owner = self.pages
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
                    _lib.mesh_rows_map(self.pages, index, self.first + index, 1, 1, WRITING | table[index].stamp)
                return self.block(first, count, node)
        return None

    # design/algorithm-sources.md#complete-page-ownership
    def send(self, frames, node):
        first = (frames.ctypes.data - self.base) // self.stride
        for index in range(first, first + len(frames)):
            row = self.pages.contents.table[index]
            _lib.mesh_rows_map(self.pages, index, self.first + index, 1, 1, (row.stamp & ~WRITING) + 1)
        return len(frames)

    # design/algorithm-sources.md#complete-page-ownership
    def write(self, first, count, node):
        return self.send(self.block(first, count, node), node)

    # design/algorithm-sources.md#complete-page-ownership
    def read(self, dtype=np.float32, max_batches=None):
        for peer_index, peer in enumerate(self.peers):
            first = (len(self.peers) + peer_index) * self.slots
            for index in range(first, first + self.slots):
                row = self.pages.contents.table[index]
                if row.page == ABSENT or not row.stamp:
                    continue
                address = _lib.mesh_row_data(self.pages, index)
                storage = (c.c_uint8 * self.stride).from_address(address)
                storage.owner = self.pages
                value = np.ctypeslib.as_array(storage).view(dtype)
                try:
                    yield value, peer
                finally:
                    _lib.mesh_row_release(self.pages, index)

    # design/algorithm-sources.md#complete-page-ownership
    def close(self):
        self.pages = None
        return _lib.mesh_detach(self.context)
