# design/the-compiler.md; design/aot-compilation-change-request.md A4, A5, A7 and A9;
# design/plan-to-bridge-change-request.md R2 and R8
import ctypes as C
import itertools
import os
import pathlib
import sys
import time

import torch

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh

CONTEXT, PUBLISHED = mesh.Context(), itertools.count(1)


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
    def __init__(self, published, transfer, copies, dtype, shape, stride):
        self.plan, self.transfer, self.execution = published.plan, transfer, 0
        self.flat = [torch.frombuffer((C.c_byte * size).from_address(address), dtype=dtype)
                     for address, size in copies]
        self.pages = [flat.as_strided(shape, stride) for flat in self.flat]

    # design/plan-to-bridge-change-request.md R1
    def out(self):
        return self.pages[self.execution % len(self.pages)]

    # rdma/mesh-collective.h mesh_host_publish
    def publish(self):
        mesh.LIB.mesh_host_publish(C.byref(CONTEXT), self.plan, self.transfer, self.execution)
        self.execution += 1

    # design/problems-of-14b9a757-change-request.md item 3, "Only a result torch allocates"
    def put(self, value):
        flat = self.flat[self.execution % len(self.flat)]
        flat.copy_(value.as_strided(flat.shape, (1,)))
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
                queues[channel] = Carried(published, transfer, published.copies[value], *layouts[channel])
    return found.node[found.self], executions
