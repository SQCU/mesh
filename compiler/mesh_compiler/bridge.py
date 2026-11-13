# design/the-compiler.md; design/aot-compilation-change-request.md A4, A5, A7 and A9;
# design/plan-to-bridge-change-request.md R2 and R8
import ctypes as C
import itertools
import mmap
import os
import pathlib
import sys
import time

import torch

from .ir import Planning

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh

CONTEXT, PUBLISHED = mesh.Context(), itertools.count(1)


# rdma/mesh-dataflow.h mesh_view_whole, mesh_nodes_read; design/aot-compilation-change-request.md A7;
# compiler/mesh_compiler/ir.py Planning
def planning():
    live = mesh.Context()
    mesh.check(mesh.LIB.mesh_attach(C.byref(live), None))
    header = mesh.Header.from_address(live.M)
    private = mmap.mmap(-1, header.length)
    address = C.addressof(C.c_char.from_buffer(private))
    while True:
        before = header.view_map
        C.memmove(address, live.M, header.data_off)
        if mesh.LIB.mesh_view_whole(live.M, before):
            break
    C.memset(address + header.algo_off, 0, C.sizeof(mesh.Algo))
    made = mesh.Context(M=address, len=header.length, client=live.client, fd=-1, start=live.start)
    mesh.check(mesh.LIB.mesh_detach(C.byref(live)))
    nodes = mesh.Nodes()
    mesh.check(mesh.LIB.mesh_nodes_read(C.byref(made), C.byref(nodes)))
    return Planning(private, made, tuple(nodes.node[:nodes.count]), nodes.self, int(os.environ["MESH_GROUP"]),
                    sum(1 << int(node) for node in os.environ["MESH_PROGRAM"].split(",")))


# rdma/mesh-collective.h mesh_plan_load; rdma/mesh-dataflow.h mesh_retire
class Published:
    def __init__(self, plan, group, here):
        self.function, self.held = mesh.function(plan.declared, plan.weights, plan.values, plan.dependences,
                                                 plan.outstanding, plan.tasks, plan.before, plan.senders,
                                                 plan.receivers, plan.resources)
        calls, made = (mesh.U * 1)(0), mesh.Planned()
        mesh.LIB.mesh_retire(CONTEXT.M, CONTEXT.client)
        mesh.check(mesh.LIB.mesh_plan_load(
            C.byref(CONTEXT), group, plan.program, self.function, 1, calls, 1, None, None, 0, None,
            time.clock_gettime_ns(time.CLOCK_MONOTONIC) + mesh.PLATFORM_NS, plan.record, len(plan.record),
            C.byref(made)))
        mesh.LIB.mesh_planned_free(C.byref(made))
        self.plan, self.execution = self.function[0].plan, 0
        self.transfers = range(len(plan.declared))
        sections = [value.section for value in self.function[0].value[:len(plan.values)]]
        first, offset, view = ((mesh.Q * (len(sections) + 1))(), (mesh.Q * sum(s.count for s in sections))(),
                               mesh.View())
        mesh.LIB.mesh_view_values(C.byref(CONTEXT), self.function, 1, first, offset, C.byref(view))
        self.copies = [[(view.address + offset[at], section.bytes) for at in range(first[value], first[value + 1])]
                       for value, section in enumerate(sections)]

    # rdma/mesh-dataflow.h mesh_host_started, mesh_transfers_start
    def start(self):
        mesh.LIB.mesh_host_started(C.byref(CONTEXT), self.plan, 0, self.execution)
        mesh.check(mesh.LIB.mesh_transfers_start(C.byref(CONTEXT), self.plan, self.execution + 1))

    # rdma/mesh-collective.h mesh_host_wait; design/problems-of-14b9a757-change-request.md 7.2 step 2 and 7.3 step 3
    def end(self):
        for transfer in self.transfers:
            mesh.check(mesh.LIB.mesh_host_wait(C.byref(CONTEXT), self.plan, transfer, self.execution, mesh.SEND))
        self.execution += 1

    # design/disjoint-pages-change-request.md C3.3
    def address(self, value):
        copies = self.copies[value]
        return copies[self.execution % len(copies)][0]


class Carried:
    def __init__(self, published, transfer, copies, dtype, shape, view):
        self.plan, self.transfer, self.execution = published.plan, transfer, 0
        offset, element, pairs = view
        flat = [torch.frombuffer((C.c_byte * size).from_address(address), dtype=dtype) for address, size in copies]
        strides = [stride // element for _, stride in pairs]
        self.pages = [pages.as_strided(shape, strides, offset // element) for pages in flat]
        self.distinct = [pages.as_strided([size for size, _ in pairs], strides, offset // element) for pages in flat]

    # design/plan-to-bridge-change-request.md R1
    def out(self):
        return self.pages[self.execution % len(self.pages)]

    # rdma/mesh-collective.h mesh_host_publish
    def publish(self):
        mesh.LIB.mesh_host_publish(C.byref(CONTEXT), self.plan, self.transfer, self.execution)
        self.execution += 1

    # design/problems-of-14b9a757-change-request.md item 3, "Only a result torch allocates"
    def put(self, value):
        distinct = self.distinct[self.execution % len(self.distinct)]
        distinct.copy_(value.as_strided(distinct.shape, value.stride()))
        self.publish()

    # rdma/mesh-collective.h mesh_host_wait
    def get(self):
        mesh.check(mesh.LIB.mesh_host_wait(C.byref(CONTEXT), self.plan, self.transfer, self.execution, mesh.RECEIVE))
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
                source, *_, sent, landed = plan.declared[transfer]
                queues[channel] = Carried(published, transfer, published.copies[value], *layouts[channel][:2],
                                          sent if source == found.self else landed)
    return found.node[found.self], executions
