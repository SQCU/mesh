#!/usr/bin/env python3
"""Two efficient programming styles whose prepared collectives miscarry on current main.

Both styles use the declared/prepared collective API as a real program does
(torch_mesh.backend.prepare_fragments, ProcessGroupMesh.plan, one realized
schedule that replays a single WRITE generation, exactly as examples/bandwidth.py
and the torch runs bake it). Each writes a region source whose every word encodes
(rank, step), so a reader names exactly which step's bytes it got with no
reference computation.

Style A -- pipelined producer: the producer writes step k+1's input into the one
source as soon as step k is issued, never waiting for the send to retire it, as
an efficient training or serving loop does (review Critical 1: the schedule
replays one WRITE generation over one source/relay/destination set with the
retirement and landing gates skipped; _stream.mm Collectives::realize,
nccl-mesh.c).

Style B -- overlapped consumer and update: the program reads step k's
destination and folds a torch-side update into a parameter, with extra GPU work
on one rank, while the next prepared step is still landing (review Critical 2:
nothing orders torch's own queue against the prepared queue; _stream.mm).

A segment written whole with one step value must land whole. A landed segment
that holds two step values (torn), or a rank other than its sender, or that never
lands, is a counterexample the gates would have prevented. Each style prints the
steps it ran and the counterexamples it saw, and the program exits nonzero if any
style saw one; so once the gates are restored it exits zero. Ranks and the node
set come from the environment and the workload configuration (the launcher sets
RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT).

Nothing here runs unbounded. A wall-clock watchdog ends the process with a
message naming the phase it waited in; every collective wait is a bounded poll;
every loop has an iteration cap and a deadline.
"""
import json
import os
import sys
import threading
import time

import torch
import torch.distributed as dist

import torch_mesh
from torch_mesh import _stream
from torch_mesh.backend import ALLGATHER, Call, prepare_fragments

PHASE = 'startup'
SENTINEL = 0x7E000000  # a word no (rank, step) encodes: 'this segment never landed'


def encode(rank, step):
    return (rank << 24) | (step & 0xFFFFFF)


def decode(word):
    return (word >> 24) & 0xFF, word & 0xFFFFFF


def watchdog(seconds):
    """End the process in finite time wherever it is stuck (operator, 2026-10-28)."""
    def wait():
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            time.sleep(0.5)
        sys.stderr.write(f'demo watchdog: {seconds:.0f} s elapsed waiting in phase {PHASE!r}; ending\n')
        sys.stderr.flush()
        os._exit(124)
    threading.Thread(target=wait, name='demo-watchdog', daemon=True).start()


def settle(completion, seconds):
    """Poll a prepared schedule to completion within a bound; never block forever."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if _stream.completed(completion):
            return True
        time.sleep(0.002)
    return False


def crossing(group, rank, size, seg, replays):
    """Prepare one all-gather of a per-rank region segment, replayed `replays` times
    over the one source and output, as the declared/prepared path bakes it."""
    source = torch_mesh.empty(seg, dtype=torch.int32)
    output = torch_mesh.empty(size * seg, dtype=torch.int32)
    source.fill_(encode(rank, 0))
    output.fill_(SENTINEL)
    torch.mps.synchronize()
    group.reset_fused_configuration()
    declaration = Call(ALLGATHER, size * seg, torch.int32, counts=[seg] * size)
    binding, declarations = prepare_fragments([(declaration, source, output)], group)
    group.plan([(declarations, replays)])
    binding.realize(replays)
    return binding, source, output


def inspect(segment, peer, bad, steps):
    """One read of the peer's landed segment, judged by its (rank, step) encoding."""
    lo, hi = int(segment.min()), int(segment.max())
    if lo == SENTINEL and hi == SENTINEL:
        bad['never_landed'] += 1
    elif lo != hi:
        bad['torn'] += 1  # two step values in a segment the peer wrote whole
    elif decode(lo)[0] != peer:
        bad['wrong_rank'] += 1
    else:
        steps.add(decode(lo)[1])


def run_style(style, group, rank, size, seg, replays, bound, update):
    """Enqueue the replayed crossing, then overlap a producer (and, for B, a consumer
    update) with it in one thread, reading the peer's landing against its encoding."""
    global PHASE
    PHASE = f'style-{style} prepare'
    peer = 1 - rank
    binding, source, output = crossing(group, rank, size, seg, replays)
    peer_mps = output.narrow(0, peer * seg, seg)
    parameter = torch_mesh.empty(seg, dtype=torch.int32).zero_() if update else None
    filler = torch_mesh.empty(256, 256, dtype=torch.float16).fill_(1.0 / 256) if update and rank == 0 else None
    bad, steps = {'torn': 0, 'wrong_rank': 0, 'never_landed': 0}, set()
    PHASE = f'style-{style} run'
    binding.enqueue()
    completion = binding.completion()
    step, reads, end = 0, 0, time.monotonic() + bound
    while time.monotonic() < end and reads < 200000:
        step += 1
        source.fill_(encode(rank, step))   # the producer's next input, issued without waiting
        if update:
            parameter.add_(peer_mps)        # fold the destination into a parameter, in flight or not
            if filler is not None:
                filler = filler @ filler    # extra GPU work on torch's queue, unordered against the prepared one
        inspect(peer_mps.to('cpu'), peer, bad, steps)
        reads += 1
        if _stream.completed(completion):
            break
    PHASE = f'style-{style} settle'
    settle(completion, max(1.0, bound))
    return {'steps_produced': step, 'reads': reads, 'distinct_steps_seen': len(steps), **bad}


def config():
    raw = json.load(sys.stdin) if len(sys.argv) > 1 and sys.argv[1] == '-' else {}
    raw = raw or {}
    pick = lambda key, env, default: raw.get(key, os.environ.get(env, default))
    return {'segment': int(pick('segment', 'SEGMENT', 262144)),
            'replays': int(pick('replays', 'REPLAYS', 2048)),
            'style_bound_seconds': float(pick('style_bound_seconds', 'STYLE_BOUND', 20)),
            'wall_bound_seconds': float(pick('wall_bound_seconds', 'WALL_BOUND', 120))}


def main():
    global PHASE
    cfg = config()
    watchdog(cfg['wall_bound_seconds'])
    PHASE = 'init_process_group'
    dist.init_process_group(backend=os.environ.get('BACKEND', 'mesh'), device_id=torch.device('mps', 0))
    rank, size = dist.get_rank(), dist.get_world_size()
    if size != 2:
        raise SystemExit(f'demo runs on 2 ranks (the workload names the two nodes), not {size}')
    group = dist.distributed_c10d._get_default_group()._get_backend(torch.device('mps'))
    seg, replays, bound = cfg['segment'], cfg['replays'], cfg['style_bound_seconds']
    styles = [('A pipelined-producer', False), ('B overlapped-consumer-update', True)]
    PHASE = 'report'
    hits = 0
    for name, update in styles:
        r = run_style(name[0], group, rank, size, seg, replays, bound, update)
        hit = r['torn'] + r['wrong_rank'] + r['never_landed']
        hits += hit
        print(f"rank{rank} style {name}: steps={r['steps_produced']} reads={r['reads']} "
              f"distinct_steps={r['distinct_steps_seen']} torn={r['torn']} wrong_rank={r['wrong_rank']} "
              f"never_landed={r['never_landed']} -> {'COUNTEREXAMPLES' if hit else 'clean'}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if hits else 0)  # end in finite time without a shutdown wait


if __name__ == '__main__':
    main()
