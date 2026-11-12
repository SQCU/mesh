# design/the-compiler.md; design/aot-compilation-change-request.md A4, A5, A7 and A9;
# design/plan-to-bridge-change-request.md R2 and R8
import ctypes as C
import errno
import itertools
import os
import pathlib
import sys
import time

import torch

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh

SYSTEM = C.CDLL("/usr/lib/libSystem.B.dylib")
SYSTEM.os_sync_wait_on_address.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t, C.c_uint32]
SYSTEM.os_sync_wake_by_address_all.argtypes = [C.c_void_p, C.c_size_t, C.c_uint32]
SHARED = 1
CONTEXT, PUBLISHED = mesh.Context(), itertools.count(1)


# rdma/mesh-collective.h mesh_plan_load; rdma/mesh-dataflow.h mesh_retire
class Published:
    def __init__(self, plan, group, here):
        self.function, self.held = mesh.function(plan.declared, plan.weights, plan.values, plan.dependences,
                                                 plan.outstanding)
        calls, made = (mesh.U * 1)(0), mesh.Planned()
        mesh.LIB.mesh_retire(CONTEXT.M, CONTEXT.client)
        mesh.check(mesh.LIB.mesh_plan_load(
            C.byref(CONTEXT), group, plan.program, self.function, 1, calls, 1, None, None, 0, None,
            time.clock_gettime_ns(time.CLOCK_MONOTONIC) + mesh.PLATFORM_NS, plan.record, len(plan.record),
            C.byref(made)))
        mesh.LIB.mesh_planned_free(C.byref(made))
        self.plan, self.execution = self.function[0].plan, 0
        self.state = mesh.State.from_address(CONTEXT.M + self.plan)
        self.sends = [transfer for transfer, (source, *_) in enumerate(plan.declared) if source == here]

    # rdma/mesh-dataflow.h mesh_host_started, mesh_transfers_start
    def start(self):
        mesh.LIB.mesh_host_started(C.byref(CONTEXT), self.plan, 0, self.execution)
        mesh.check(mesh.LIB.mesh_transfers_start(C.byref(CONTEXT), self.plan, self.execution + 1))

    # rdma/mesh-collective.h mesh_host_sent; design/disjoint-pages-change-request.md C3.3
    def end(self):
        for transfer in self.sends:
            while not (sent := mesh.LIB.mesh_host_sent(C.byref(CONTEXT), self.plan, transfer, self.execution)):
                pass
            if sent == mesh.FAILED:
                raise OSError(errno.ECANCELED, f"the bridge did not carry execution {self.execution} of a channel")
        self.execution += 1


class Carried:
    def __init__(self, published, transfer, section, dtype, shape, stride):
        self.plan, self.transfer, self.execution = published.plan, transfer, 0
        entry = mesh.Publication.from_address(CONTEXT.M + published.state.transfers + transfer * mesh.PUBLICATION_BYTES)
        copies = range(published.state.copies)
        self.counts = [C.c_uint64.from_address(CONTEXT.M + entry.count + 8 * copy) for copy in copies]
        self.flat = [torch.frombuffer((C.c_byte * section.bytes).from_address(
            mesh.LIB.mesh_section_address(C.byref(CONTEXT), section, copy * section.stride)), dtype=dtype)
            for copy in copies]
        self.pages = [flat.as_strided(shape, stride) for flat in self.flat]

    # rdma/mesh-collective.h mesh_host_publish
    def put(self, value):
        flat = self.flat[self.execution % len(self.flat)]
        flat.copy_(value.as_strided(flat.shape, (1,)))
        mesh.LIB.mesh_host_publish(C.byref(CONTEXT), self.plan, self.transfer, self.execution)
        self.execution += 1

    # rdma/mesh-collective.h mesh_host_arrived
    def get(self):
        count = self.counts[self.execution % len(self.counts)]
        while True:
            seen = count.value
            arrived = mesh.LIB.mesh_host_arrived(C.byref(CONTEXT), self.plan, self.transfer, self.execution)
            if arrived:
                break
            SYSTEM.os_sync_wait_on_address(C.addressof(count), seen, 8, SHARED)
        if arrived == mesh.FAILED:
            raise OSError(errno.ECANCELED, f"the bridge cancelled execution {self.execution} of a channel")
        self.execution += 1
        return self.pages[(self.execution - 1) % len(self.pages)]


class Empty:
    def __init__(self, dtype, shape):
        self.pages = torch.empty(shape, dtype=dtype)

    def put(self, value):
        pass

    def get(self):
        return self.pages


def channels(scheduled, queues):
    mesh.check(mesh.LIB.mesh_attach(C.byref(CONTEXT), None))
    found, plan, executions = mesh.Nodes(), scheduled.plan, []
    mesh.check(mesh.LIB.mesh_nodes_read(C.byref(CONTEXT), C.byref(found)))
    layouts = {channel: layout for channel, _, _, *layout in scheduled.carried}
    for channel, (dtype, shape, _) in layouts.items():
        if not all(shape):
            queues[channel] = Empty(dtype, shape)
    if plan is not None:
        published = Published(plan, (int(os.environ["MESH_GROUP"]) << 32) + next(PUBLISHED), found.self)
        executions.append(published)
        for transfer, channel in enumerate(plan.channels):
            value = plan.declared[transfer][4]
            if value != mesh.ABSENT:
                queues[channel] = Carried(published, transfer, published.function[0].value[value].section,
                                          *layouts[channel])
    return found.node[found.self], executions
