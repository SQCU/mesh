#!/usr/bin/env python3
"""Two programs, each isolating one ordering defect of prepared WRITEs, judged by the bytes they move.

Every word a rank sends encodes (rank, step), so a reader names the step it got with no reference computation: a read
of step k holds the peer's step k, whole, or it is a counterexample (torn: two values; ahead or stale: another step;
never_landed or unproduced: a word no step encodes; wrong_rank).

Style A, pipelined one-sided writes (review critical 1). One Execution of STEPS realized WRITE steps from rank 0 to
rank 1 (a ViewWrites binding, ncclMeshWriteExecute, as partition._follow_spans uses one): rank 0's step a producer
kernel and the WRITE, rank 1's the WRITE and a consumer kernel (PASSES reads: the slow peer), all inside each rank's one
prepared function. Each kernel takes its step from a device counter it advances, so torch's queue takes no part and
critical 2 can neither cause nor mask a counterexample. Every step reuses the one source and the one destination, so
step k+1's producer must wait for step k's source to be read (a credit), and step k+1 must not land before step k's
consumer has read it (the landing gate). The write is one-way: a rank that also waited for its peer's landing each step
would hold the two in lockstep and hide the source's reuse.

Style B, overlapped consumer and update (review critical 2). UPDATES prepared WRITE steps, each with its own source and
destination, so no WRITE generation is reused and critical 1 cannot reach them. Torch produces step k's source, the
step is enqueued, and torch reads its destination into a record while the next step lands, with extra GPU work on
rank 0. Unless torch's queue and the prepared queue are ordered, the step sends a source torch has not yet written, or
torch reads a destination before it lands.

Each style prints its reads by kind, and style A its function's seconds (a record the hourly aggregates). The program
exits 1 on any counterexample and 0 when every read held its step.
Memory is region leases (torch_mesh.empty), made before their style runs and read on the host through their shared
storage. A wait that sees no progress for PROGRESS seconds ends the program through the library's expiry
(_stream.expired: its GPU work and posted receives stopped first), naming the wait; faulthandler ends a process stuck
anywhere else, with every thread's traceback. Ranks come from the launcher (RANK, WORLD_SIZE,
MASTER_ADDR, MASTER_PORT); the hourly places it on two nodes (metal-microbench tools/mesh/hourly/workloads.json).
"""
import collections
import faulthandler
import json
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import torch_mesh
from torch_mesh import _stream
from torch_mesh.backend import WRITE, Call
from torch_mesh.declarations import execution_sequence
from torch_mesh.engines import host

STEPS = int(os.environ.get('DEMO_STEPS', 2048))               # style A: WRITE steps in its one prepared function
WORDS = int(os.environ.get('DEMO_WORDS', 65536))              # style A: 32-bit words each step writes
UPDATES = int(os.environ.get('DEMO_UPDATES', 128))            # style B: prepared steps, each its own operands
UPDATE_WORDS = int(os.environ.get('DEMO_UPDATE_WORDS', 16384))
PASSES = int(os.environ.get('DEMO_PASSES', 4))                # style A: the consumer's passes over each read
EXTRA = int(os.environ.get('DEMO_EXTRA', 1024))               # style B: rank 0's extra work, an EXTRA-square half matmul
PROGRESS = float(os.environ.get('DEMO_PROGRESS_SECONDS', 15))
PREPARATION = float(os.environ.get('DEMO_PREPARATION_SECONDS', 120))
SENTINEL, UNWRITTEN = 0x7E000000, 0x7D000000                  # never landed, never produced: no (rank, step) encodes them
KERNELS = '''
#include <metal_stdlib>
using namespace metal;
// w: the words, the rank whose (rank, step) they encode, the consumer's passes over its read; each kernel one
// threadgroup that advances its rank's step counter once every thread has read it
kernel void produce(device uint *source [[buffer(0)]], device uint *counter [[buffer(1)]], constant uint *w [[buffer(2)]],
                    uint i [[thread_index_in_threadgroup]], uint n [[threads_per_threadgroup]]) {
  const uint step = counter[0] + 1, word = (w[1] << 24) | (step & 0xFFFFFF);
  for (uint j = i; j < w[0]; j += n) source[j] = word;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i == 0) counter[0] = step;
}
// the step's read: its words differing from the step's, their least and greatest
kernel void consume(device const uint *landing [[buffer(0)]], device uint *counter [[buffer(1)]], device uint *record [[buffer(2)]],
                    constant uint *w [[buffer(3)]], uint i [[thread_index_in_threadgroup]], uint n [[threads_per_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint g [[simdgroup_index_in_threadgroup]],
                    uint groups [[simdgroups_per_threadgroup]]) {
  threadgroup uint part[3][32];
  const uint step = counter[0] + 1, expected = (w[1] << 24) | (step & 0xFFFFFF);
  uint bad = 0, lo = ~0u, hi = 0;
  for (uint p = 0; p < w[2]; p++)
    for (uint j = i; j < w[0]; j += n) { const uint v = landing[j]; bad += v != expected; lo = min(lo, v); hi = max(hi, v); }
  bad = simd_sum(bad); lo = simd_min(lo); hi = simd_max(hi);
  if (lane == 0) { part[0][g] = bad; part[1][g] = lo; part[2][g] = hi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i == 0) {
    for (uint k = 1; k < groups; k++) { bad += part[0][k]; lo = min(lo, part[1][k]); hi = max(hi, part[2][k]); }
    record[3 * step] = bad; record[3 * step + 1] = lo; record[3 * step + 2] = hi;
    counter[0] = step;
  }
}
'''


def encode(rank, step):
    return rank << 24 | step & 0xFFFFFF


def kind(bad, lo, hi, step, peer):
    """A read of `step` from `peer`: its words differing from encode(peer, step), their least and greatest."""
    if not bad:
        return 'held'
    if lo != hi:
        return 'torn'
    if lo in (SENTINEL, UNWRITTEN):
        return 'never_landed' if lo == SENTINEL else 'unproduced'
    if lo >> 24 != peer:
        return 'wrong_rank'
    return 'stale' if lo & 0xFFFFFF < step else 'ahead'


def watch(seconds):
    """The backstop: no progress for `seconds` ends the process (faulthandler dumps every thread, then _exit(1);
    it needs no GIL)."""
    faulthandler.dump_traceback_later(seconds, exit=True)


def settle(done, progress, wait):
    """Poll `done` while `progress` moves; PROGRESS seconds without a move ends the program through the library's expiry,
    naming `wait`."""
    seen, since = progress(), time.monotonic()
    while not done():
        if (now := progress()) != seen:
            seen, since = now, time.monotonic()
            watch(2 * PROGRESS)
        elif time.monotonic() - since > PROGRESS:
            _stream.expired(f'demo error: {wait}, no progress at {seen}', int(PROGRESS * 1e9))
        time.sleep(0.001)


def meet(store, name, rank):
    """Both ranks prepared before either enqueues, so no crossing waits out a peer's preparation."""
    store.set(f'demo/{name}/{rank}', '1')
    store.wait([f'demo/{name}/{1 - rank}'], timedelta(seconds=PREPARATION))


def writes(group, rank, sources, landings, words, issued, roots=(0, 1)):
    """A step's WRITE from each of `roots` (its source to the other rank's landing), declared for `issued` steps."""
    group.plan([([Call(WRITE, 4 * words, torch.uint8, root=a, destination=1 - a) for a in roots], issued)])
    return [group.prepare_writes([(s if rank == a else d, a, 1 - a) for a in roots]) for s, d in zip(sources, landings)]


def style_a(group, rank, store):
    shift = _stream.compilation['ACCESS_WRITE_SHIFT']
    operand, counter = torch_mesh.empty(WORDS, dtype=torch.int32), torch_mesh.empty(1, dtype=torch.int32)
    record = torch_mesh.empty(STEPS + 1, 3, dtype=torch.int32)
    for t, value in zip((operand, counter, record), (SENTINEL if rank else UNWRITTEN, 0, 0)):
        host(t).fill_(value)
    kernel = _stream.pipeline(KERNELS, 'consume' if rank else 'produce')
    width = min(1024, _stream.pipeline_facts(kernel)[0])
    [write] = writes(group, rank, [operand], [operand], WORDS, STEPS, roots=(0,))

    class Step:
        def append(self, schedule):
            if rank:
                write.append(schedule)
                schedule.compute(kernel, [operand, counter, record], [WORDS, 0, PASSES], 3, 1, 1, width, 0b11 | 0b110 << shift)
            else:
                schedule.compute(kernel, [operand, counter], [WORDS, 0], 2, 1, 1, width, 0b10 | 0b11 << shift)
                write.append(schedule)

    program = execution_sequence([Step()] * STEPS)
    program.realize()
    meet(store, 'A', rank)
    started = time.monotonic()
    program.forward()
    done, count = program.completion(), host(counter)
    settle(lambda: _stream.completed(done), lambda: int(count[0]), "style A's prepared function (its step counter)")
    print(json.dumps({'seconds': time.monotonic() - started, 'useful_output_bytes': 4 * WORDS * STEPS}), flush=True)
    rows = host(record)[1:].tolist() if rank else []
    return collections.Counter(kind(*row, step, 0) for step, row in enumerate(rows, 1)), STEPS


def style_b(group, rank, store):
    peer = 1 - rank
    sources, landings, reads = (torch_mesh.empty(UPDATES, UPDATE_WORDS, dtype=torch.int32) for _ in range(3))
    extra = [torch_mesh.empty(EXTRA, EXTRA, dtype=torch.float16) for _ in range(3 * (rank == 0))]
    for t, value in zip((sources, landings, reads, *extra), (UNWRITTEN, SENTINEL, 0, 1 / EXTRA, 1 / EXTRA, 0)):
        host(t).fill_(value)
    steps = writes(group, rank, sources, landings, UPDATE_WORDS, UPDATES)
    for step in steps:
        step.realize()
    meet(store, 'B', rank)
    for k, step in enumerate(steps):
        if extra:
            torch.mm(*extra[:2], out=extra[2])
        sources[k].fill_(encode(rank, k + 1))
        step.enqueue()
        reads[k].copy_(landings[k])
    done, torch_done = [step.completion() for step in steps], torch.mps.Event()
    torch_done.record()
    settle(lambda: torch_done.query() and all(map(_stream.completed, done)),
           lambda: (torch_done.query(), sum(map(_stream.completed, done))), "style B's prepared steps and torch's reads")
    rows = host(reads)
    expected = torch.tensor([encode(peer, k + 1) for k in range(UPDATES)], dtype=torch.int32)
    stats = zip((rows != expected[:, None]).sum(1).tolist(), rows.amin(1).tolist(), rows.amax(1).tolist())
    return collections.Counter(kind(*row, k + 1, peer) for k, row in enumerate(stats)), UPDATES


def main():
    watch(PREPARATION)
    dist.init_process_group(backend=os.environ.get('BACKEND', 'mesh'), device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    if size != 2:
        raise SystemExit(f'demo runs on 2 ranks, not {size}')
    group = dist.distributed_c10d._get_default_group()._get_backend(torch.device('mps'))
    store = dist.distributed_c10d._get_default_store()
    for name, style in (('A (critical 1: pipelined one-sided writes)', style_a),
                        ('B (critical 2: overlapped consumer and update)', style_b)):
        watch(PREPARATION)
        reads, steps = style(group, rank, store)
        hit = sum(reads.values()) - reads['held']
        store.add('demo/counterexamples', hit)
        print(f'rank{rank} style {name}: steps={steps} reads={sum(reads.values())} '
              + ' '.join(f'{k}={reads[k]}' for k in ('held', 'torn', 'ahead', 'stale', 'never_landed', 'unproduced', 'wrong_rank'))
              + f" -> {'COUNTEREXAMPLES' if hit else 'clean'}", flush=True)
    meet(store, 'end', rank)
    hits = store.add('demo/counterexamples', 0)
    if hits:
        print(f'demo error: the ranks read {hits} counterexamples', file=sys.stderr, flush=True)
    watch(2 * PROGRESS)
    dist.destroy_process_group()
    return 1 if hits else 0


if __name__ == '__main__':
    sys.exit(main())
