# design/the-compiler.md
import ctypes as C
import errno
import itertools
import os
import pathlib
import struct
import sys
from dataclasses import dataclass
from queue import SimpleQueue

import torch

from .ir import RECV, SEND
from .schedule import channel_ends

sys.path.append(str(pathlib.Path(__file__).parents[2] / "rdma"))
import mesh

SYSTEM = C.CDLL("/usr/lib/libSystem.B.dylib")
SYSTEM.OSAtomicAdd64Barrier.restype = C.c_int64
SYSTEM.OSAtomicAdd64Barrier.argtypes = [C.c_int64, C.c_void_p]
SYSTEM.os_sync_wait_on_address.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t, C.c_uint32]
SYSTEM.os_sync_wake_by_address_all.argtypes = [C.c_void_p, C.c_size_t, C.c_uint32]
SHARED = 1
ROOM = 2
CONTEXT, PUBLISHED = mesh.Context(), itertools.count(1)


@dataclass
class Plan:
    channel: int
    receiver: str
    value: torch.Tensor
    ends: list
    direction: int
    row: int
    page: int
    blocks: int
    size: int
    base: int = 0
    image: bytearray = None


def carried(scheduled):
    ends = channel_ends(scheduled.modules)
    found = []
    for channel in sorted(ends):
        (sender, _), (receiver, arrival) = ends[channel][SEND], ends[channel][RECV]
        if scheduled.machines[sender] != scheduled.machines[receiver]:
            found.append((channel, sender, receiver, arrival.meta["val"]))
    return found


def extent(value):
    reach = 1 + sum((n - 1) * s for n, s in zip(value.shape, value.stride()))
    return value.dtype.itemsize * reach if value.numel() else 0


def align(at):
    return -(-at // 64) * 64


def entries(scheduled, nodes, header, group):
    index = {nodes.node[i]: i for i in range(nodes.count)}
    quantum = header.block * header.pgsz
    arena = (header.length - header.data_off) // quantum * header.block
    window = min(header.wire_pages, arena)
    plans, row, wire = [], 0, 0
    for channel, sender, receiver, value in carried(scheduled):
        ends = [index[scheduled.machines[processor]] for processor in (sender, receiver)]
        size = extent(value)
        blocks = -(-size // quantum)
        direction = ends.index(nodes.self) if nodes.self in ends else None
        held = blocks * (direction is not None)
        plans.append(Plan(channel, receiver, value, ends, direction, row, wire, blocks, size))
        row, wire = row + held, wire + held * header.block
    local = wire if window == arena else window
    for place, plan in enumerate(plans):
        ends, blocks, held = plan.ends, plan.blocks, plan.blocks * (plan.direction is not None)
        phases = mesh.STATE_BYTES
        counts = align(phases + (blocks + 63) // 64 * 16)
        completions = align(counts + 8)
        supplied = align(completions + 16)
        transfers = align(supplied + 8)
        deliveries = align(transfers + mesh.PUBLICATION_BYTES)
        submits = align(deliveries + ROOM * mesh.DELIVERY_BYTES)
        rows = align(submits + ROOM * mesh.SUBMISSION_BYTES)
        hops = align(rows + 8)
        plan.image = image = bytearray(hops + held * C.sizeof(mesh.Hop))
        plan.base = base = header.data_off + local * header.pgsz
        local += -(-len(image) // quantum) * header.block
        state = mesh.State.from_buffer(image)
        state.copies, state.calls, state.room, state.submit_room = 1, 1, ROOM, ROOM
        state.phases, state.counts, state.completions = base + phases, base + counts, base + completions
        state.supplied, state.transfers, state.hops = base + supplied, base + transfers, base + hops
        state.deliveries, state.submits, state.syncs, state.rows = (base + deliveries, base + submits,
                                                                     base + rows, base + rows)
        state.group, state.epoch, state.entries = group, place + 1, held
        state.serial, state.sequencer, state.boot = nodes.serial, nodes.sequencer, nodes.boot
        state.nodes, state.sender = 1 << ends[0] | 1 << ends[1], ends[0]
        publication = mesh.Publication.from_buffer(image, transfers)
        publication.state = base
        if plan.direction == mesh.SEND:
            publication.sends, publication.completion = blocks, base + completions
            publication.targets[0] = mesh.Target(base + supplied, 1, 0)
        if plan.direction == mesh.RECEIVE:
            publication.pages, publication.phases, publication.count = blocks, base + phases, base + counts
        sends = plan.direction == mesh.SEND
        for i, hop in enumerate((mesh.Hop * held).from_buffer(image, hops)):
            hop.page, hop.node, hop.direction = plan.page + i * header.block, ends[sends], plan.direction
            hop.field = mesh.ABSENT if sends else 2 * (i % 32)
            hop.word = base + (supplied if sends else phases + i // 32 * 8)
            hop.count = base + (completions if sends else counts)
            hop.pages = 0 if sends else blocks
            hop.bytes, hop.serial = min(quantum, plan.size - i * quantum), nodes.serial
    toward = [[place for place, plan in enumerate(plans)
               if plan.direction is not None and plan.blocks and plan.ends[1 - plan.direction] == i]
              for i in range(nodes.count)]
    adjacent = [*itertools.accumulate(map(len, toward), initial=0), *itertools.chain(*toward)]
    table = struct.pack(f"{len(plans)}Q{len(adjacent)}I", *(plan.base for plan in plans), *adjacent)
    return plans, header.data_off + local * header.pgsz, table


def publish(header, plans, at, table, group, count):
    algo = mesh.Algo.from_address(CONTEXT.M + header.algo_off)
    mesh.LIB.mesh_retire(CONTEXT.M, CONTEXT.client)
    while True:
        released = algo.released
        if not algo.published or released == algo.published:
            break
        SYSTEM.os_sync_wait_on_address(C.addressof(algo) + mesh.Algo.released.offset, released, 8, SHARED)
    for plan in plans:
        C.memmove(CONTEXT.M + plan.base, bytes(plan.image), len(plan.image))
        if plan.direction is not None and plan.blocks:
            mesh.LIB.mesh_backing_bind(C.byref(CONTEXT), plan.row, plan.blocks * header.block, plan.page, 0)
    C.memmove(CONTEXT.M + at, table, len(table))
    algo.plans, algo.nodes, algo.group = len(plans), count, group
    algo.table, algo.adjacent = at, at + 8 * len(plans)
    header.client, header.client_start = CONTEXT.client, CONTEXT.start
    algo.published = SYSTEM.OSAtomicAdd64Barrier(1, C.addressof(header) + mesh.Header.serial.offset)
    SYSTEM.OSAtomicAdd64Barrier(1, C.addressof(header) + mesh.Header.plans.offset)
    SYSTEM.os_sync_wake_by_address_all(C.addressof(header) + mesh.Header.plans.offset, 8, SHARED)


class Carried:
    def __init__(self, plan, header):
        self.plan, self.state = plan.base, mesh.State.from_address(CONTEXT.M + plan.base)
        self.count = C.c_uint64.from_address(CONTEXT.M + self.state.counts)
        section = mesh.Section(plan.row, plan.blocks * header.block, plan.size, 1, plan.blocks)
        address = mesh.LIB.mesh_section_address(C.byref(CONTEXT), section, 0)
        self.flat = torch.empty(0, dtype=plan.value.dtype)
        if plan.size:
            self.flat = torch.frombuffer((C.c_byte * plan.size).from_address(address), dtype=plan.value.dtype)
        self.pages = self.flat.as_strided(tuple(plan.value.shape), plan.value.stride())

    def put(self, value):
        execution = self.state.submissions
        self.flat.copy_(value.as_strided(self.flat.shape, (1,)))
        mesh.LIB.mesh_host_publish(C.byref(CONTEXT), self.plan, 0, execution)
        mesh.LIB.mesh_host_started(C.byref(CONTEXT), self.plan, 0, execution)
        mesh.check(mesh.LIB.mesh_transfers_start(C.byref(CONTEXT), self.plan, execution + 1))
        while not (sent := mesh.LIB.mesh_host_sent(C.byref(CONTEXT), self.plan, 0, execution)):
            pass
        if sent == mesh.FAILED:
            raise OSError(errno.ECANCELED, f"the bridge did not carry execution {execution} of a channel")

    def start(self):
        mesh.LIB.mesh_host_started(C.byref(CONTEXT), self.plan, 0, self.state.executions)

    def get(self):
        execution = self.state.executions - 1
        while True:
            count = self.count.value
            arrived = mesh.LIB.mesh_host_arrived(C.byref(CONTEXT), self.plan, 0, execution)
            if arrived:
                break
            SYSTEM.os_sync_wait_on_address(C.addressof(self.count), count, 8, SHARED)
        if arrived == mesh.FAILED:
            raise OSError(errno.ECANCELED, f"the bridge cancelled execution {execution} of a channel")
        return self.pages


def channels(scheduled):
    queues, received = [SimpleQueue() for _ in range(scheduled.channels)], {}
    machines = set(scheduled.machines.values())
    if len(machines) == 1:
        return queues, received, machines.pop()
    mesh.check(mesh.LIB.mesh_attach(C.byref(CONTEXT), None))
    nodes, header = mesh.Nodes(), mesh.Header.from_address(CONTEXT.M)
    mesh.check(mesh.LIB.mesh_nodes_read(C.byref(CONTEXT), C.byref(nodes)))
    group = (int(os.environ["MESH_GROUP"]) << 32) + next(PUBLISHED)
    plans, at, table = entries(scheduled, nodes, header, group)
    if plans:
        publish(header, plans, at, table, group, nodes.count)
    for plan in plans:
        if plan.direction is not None:
            queues[plan.channel] = Carried(plan, header)
        if plan.direction == mesh.RECEIVE:
            received.setdefault(plan.receiver, []).append(queues[plan.channel])
    return queues, received, nodes.node[nodes.self]
