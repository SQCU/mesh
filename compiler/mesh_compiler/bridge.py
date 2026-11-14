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


# rdma/mesh-collective.h mesh_plan_load; rdma/mesh-dataflow.h mesh_retire, mesh_view_values;
# design/disjoint-pages-change-request.md C3.3; design/plan-to-bridge-change-request.md R1;
# design/problems-of-14b9a757-change-request.md item 3; compiler/mesh_compiler/schedule.py carried, planned declared
class Published:
    def __init__(self, plan, carried, group, here):
        self.source, self.context = plan, C.addressof(CONTEXT)
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
        sections = [value.section for value in self.function[0].value[:len(plan.values)]]
        first, offset, view = ((mesh.Q * (len(sections) + 1))(), (mesh.Q * sum(s.count for s in sections))(),
                               mesh.View())
        mesh.LIB.mesh_view_values(C.byref(CONTEXT), self.function, 1, first, offset, C.byref(view))
        self.copies = [[(view.address + offset[at], section.bytes) for at in range(first[value], first[value + 1])]
                       for value, section in enumerate(sections)]
        layouts = {channel: (dtype, shape) for channel, _, _, dtype, shape, _ in carried}
        self.views, self.distinct = {}, {}
        for transfer, (channel, (source, *_, value, sent, landed)) in enumerate(zip(plan.channels, plan.declared)):
            if value == mesh.ABSENT:
                continue
            dtype, shape = layouts[channel]
            start, element, pairs = sent if source == here else landed
            strides = [stride // element for _, stride in pairs]
            flat = [torch.frombuffer((C.c_byte * size).from_address(address), dtype=dtype)
                    for address, size in self.copies[value]]
            self.views[transfer] = tuple(pages.as_strided(shape, strides, start // element) for pages in flat)
            self.distinct[transfer] = tuple(pages.as_strided([size for size, _ in pairs], strides, start // element)
                                            for pages in flat)


# design/compilation-composition-change-request.md item 7; rdma/mesh-dataflow.h mesh_attach, mesh_nodes_read;
# output_data/compiler-workgroup-20261113/executor/native.status.md Model.prepare
def prepare(scheduled, entry=None):
    if scheduled.plan is None:
        key = scheduled.output_processor if entry is None else entry
        return key, scheduled.functions[key]
    mesh.check(mesh.LIB.mesh_attach(C.byref(CONTEXT), None))
    found = mesh.Nodes()
    mesh.check(mesh.LIB.mesh_nodes_read(C.byref(CONTEXT), C.byref(found)))
    (key,) = [group for group in scheduled.functions if scheduled.machines[group] == found.node[found.self]] if (
        entry is None) else (entry,)
    published = Published(scheduled.plan, scheduled.carried, (int(os.environ["MESH_GROUP"]) << 32) + next(PUBLISHED),
                          found.self)
    return key, scheduled.functions[key].bind(published)
