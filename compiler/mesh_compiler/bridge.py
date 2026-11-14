# design/the-compiler.md; design/aot-compilation-change-request.md A4, A5, A7 and A9;
# design/plan-to-bridge-change-request.md R2 and R8
import ctypes as C
import itertools
import mmap
import os
import pathlib
import sys
import time
import weakref

import torch
from torch.export.graph_signature import InputKind

from .ir import Planning

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh

PUBLISHED = itertools.count(1)


# rdma/mesh-dataflow.h mesh_view_whole, mesh_nodes_read, mesh_publications_clear (the private copy holds none of the
# region's publications: their records are not copied); design/aot-compilation-change-request.md A7;
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
    mesh.LIB.mesh_publications_clear(address)
    made = mesh.Context(M=address, len=header.length, client=live.client, fd=-1, start=live.start)
    mesh.check(mesh.LIB.mesh_detach(C.byref(live)))
    nodes = mesh.Nodes()
    mesh.check(mesh.LIB.mesh_nodes_read(C.byref(made), C.byref(nodes)))
    return Planning(private, made, tuple(nodes.node[:nodes.count]), nodes.self, int(os.environ["MESH_GROUP"]),
                    sum(1 << int(node) for node in os.environ["MESH_PROGRAM"].split(",")))


# design/plan-to-bridge-change-request.md R8; design/aot-compilation-change-request.md, appendix item 6: one prepared
# program's attached region mapping (its own client) and, adopted only once its load succeeded, its loaded result.
# Every holder of its mapped bytes keeps it: the Published, its arena's backing (torch.frombuffer keeps that buffer
# object, so every view of the arena does) and a native buffer lent it; it refers to none of them. `release`, its one
# finalizer (_retired), runs when the last holder goes, or at once where its preparation fails
class _Mapping:
    __slots__ = ('context', 'loaded', 'release', '__weakref__')

    def __init__(self):
        self.context, self.loaded = mesh.Context(), []
        mesh.check(mesh.LIB.mesh_attach(C.byref(self.context), None))
        self.release = weakref.finalize(self, _retired, self.context, self.loaded)


# rdma/mesh-collective.h mesh_program_load: one prepared network program, its saved source `source`
# (compiler/mesh_compiler/schedule.py SavedNetwork, or the solved network result: its saved() bytes, its program)
# loaded once into its own mapping (`mapping`, _Mapping: its own client), its rows and pages reserved beside every live
# publication's, relocated and published, the loaded Local adopted by `mapping` (lent its bytes:
# compiler/mesh_compiler/metal.py _lent) until _retired; `group` its preparation's. `solution` the loaded result
# (schedule.published_program: Function[], roots, prefixes, offsets) whose views hold `mapping` and the byte offsets;
# `function`, `planned` and `context` its; `offset` every copy's byte offset in `arena`, one mapping of every run 1
# copy (mesh_view_program; UINT64_MAX a run 0 copy), at `base`; design/plan-to-bridge-change-request.md R8 and
# design/aot-compilation-change-request.md, appendix item 6: each Published its own mapping and client, retired and
# released with it alone; torch 2.14 frombuffer
class Published:
    def __init__(self, source, group, mapping):
        from .metal import _lent
        from .schedule import published_program
        record, program = source.saved(), source.program
        self.source, self.mapping, self.context = source, mapping, C.addressof(mapping.context)
        made = mesh.Local()
        mesh.check(_lent(record, lambda release, owner: mesh.LIB.mesh_program_load(
            C.byref(mapping.context), group, program, time.clock_gettime_ns(time.CLOCK_MONOTONIC) + mesh.PLATFORM_NS,
            record, len(record), release, owner, C.byref(made)), lambda status: status == 0))
        mapping.loaded.append(made)
        self.offset, view = (mesh.Q * max(made.offsets, 1))(), mesh.View()
        mesh.LIB.mesh_view_program(C.byref(mapping.context), C.byref(made), self.offset, C.byref(view))
        self.base = C.addressof(self.offset)
        self.solution = published_program(made, (mapping, self.offset), self.offset, source)
        self.function, self.planned = made.function, made.planned
        backing = (C.c_byte * view.bytes).from_address(view.address) if view.bytes else None
        if backing is not None:
            backing.mapping = mapping
        self.arena = torch.frombuffer(backing, dtype=torch.uint8) if backing is not None else torch.empty(
            0, dtype=torch.uint8)

    def bound(self, owned, M):
        from .native import planned_buffer
        return planned_buffer(self, owned, M)

    def span(self, first, index):
        from .native import viewed
        return viewed(first, self.base, index)


# a _Mapping's end: its loaded network result's publication closed (the bridge releases its rows and pages once no
# port holds it) and that result freed (mesh_local_free), then its mapping detached
def _retired(context, loaded):
    for local in loaded:
        mesh.LIB.mesh_retire(context.M, C.byref(local.planned))
        mesh.LIB.mesh_local_free(C.byref(local))
    mesh.check(mesh.LIB.mesh_detach(C.byref(context)))


# design/compilation-composition-change-request.md item 7; rdma/mesh-dataflow.h mesh_attach, mesh_nodes_read;
# compiler/mesh_compiler/native.py Target.prepare: the selected machine entry's completed native target relocated onto
# its run-time owner (a local program's own arena, or the published plan), its constant roots the exported constants
def prepare(scheduled, entry=None):
    constants = {name: tensor.detach() for name, (_, tensor) in scheduled.exported.state.items()
                 if scheduled.exported.state_kinds[name] == InputKind.CONSTANT_TENSOR}
    if scheduled.plan is None:
        key = scheduled.output_processor if entry is None else entry
        return key, scheduled.functions[key].prepare(None, constants)
    mapping = _Mapping()
    try:
        found = mesh.Nodes()
        mesh.check(mesh.LIB.mesh_nodes_read(C.byref(mapping.context), C.byref(found)))
        (key,) = [group for group in scheduled.functions if scheduled.machines[group] == found.node[found.self]] if (
            entry is None) else (entry,)
        published = Published(scheduled.plan, (int(os.environ["MESH_GROUP"]) << 32) + next(PUBLISHED), mapping)
        return key, scheduled.functions[key].prepare(published, constants)
    except BaseException:
        mapping.release()
        raise
