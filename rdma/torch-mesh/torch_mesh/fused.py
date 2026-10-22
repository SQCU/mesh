"""Fused crossings on torch's MPS stream (rdma/nccl.h ncclMeshFusableRooted, ncclMeshFused; ../DESIGN-fused-stream.md):
the one-shot exchange among n ranks, whose steps the caller's own kernels run, zero copy.  The kernel that makes a
rank's part writes it into the rank's send tensor and publishes it to every peer, and the kernel after the call reads
each peer's part from its landing tensor: it adds them to its own (an all-reduce, the chain) or reads them in place
(an all-gather, the LSTM).  What it does with them is the caller's: the library cannot see when a kernel reads a
landing buffer of its own, so it never combines a fused crossing.  The call is ncclSend of
the send tensor to each other rank and ncclRecv from each into its own landing tensor, in one group; it dispatches no
step.  The steps are the library's: a kernel
source is compiled after steps() and calls them on the bindings exchange returns, which encode passes it.

  steps()                                    the Metal text a kernel source is compiled after, from the library the
                                             process loaded (ncclMeshMetalPrelude): rdma/mesh.metal as it was built
                                             with, then its steps, struct mesh_step_binding among them
  exchange(send, landings, polls, group=None)
                                             the fused exchange of `send` (this rank's part) and `landings` (where each
                                             other rank's part lands, in rank order, n - 1 of them, each its own size
                                             and dtype: a ragged all-gather lands a peer's share): contiguous MPS
                                             tensors, each in a region tensor and starting on a
                                             page (torch_mesh.empty: a lease, which the bridge sends from and lands
                                             in); a Crossing, its publications' bindings and its landings', in rank
                                             order; one the library does not fuse raises (no slower form runs in its
                                             place), any size and dtype fusing
  encode(handle, buffers, constants, at, gx, gy, threads, peers, land=None, publish=None)
                                             _stream.encode's dispatch with `land`'s landing bindings and `publish`'s
                                             publication bindings after the constants, `peers` of each, declaring the
                                             buffers those bindings name (each landing's buffer, record and counts read,
                                             each publication's record read and its operand and counts written) and no
                                             other; a dispatch that publishes is a publication under the stream's one
                                             commit rule

The kernel's constant block at `at` is its own words padded to an even count, then the landings' mesh_step_bindings,
then the publications' (all zero where encode is given none), each address two words, its low word first:
struct { uint words[2k]; mesh_step_binding land[PEERS], publish[PEERS]; }.  Each all_reduce call is one crossing, encoded before its
publishing kernel: a step kernel that lands crossing t - 1 and publishes crossing t is encoded after crossing t's call,
with crossing t - 1's Crossing as `land` and crossing t's as `publish`.

The publishing kernel: mesh_step_enter(*b.p, b.v, b.progress, first) in one thread for each publication b, then its
threadgroups write the part into the send tensor through mesh_step_write (system-coherent: the NIC reads a plain store
only once its command buffer has ended), then the last threadgroup to arrive (the kernel's own count of
arrivals, after every other threadgroup's stores) calls mesh_step_release(*b.p, b.a, tid) for each: one system fence and
one store of that holding's `ready`.  The landing kernel: mesh_step_enter, then mesh_step_await(*l.p, l.a, l.v, tid) for
each landing l in every threadgroup that reads (each its own wait for the landed bytes alone, bounded by polls), then
element k of that peer's part is mesh_step_read<T>(l.d + l.p->dst, k), loaded system-coherent (the NIC wrote it).

The transport writes landings as the plan presents them: the receive into a landing tensor is posted when the call
presents it, so a peer's part may land there as soon as that peer publishes, before any earlier kernel of this rank
that reads or writes those pages has run (the operator, 2026-10-22: "if source says to write continuously into ab
uffer, the nic must write continuously into that buffer even if this means that dat aprocessing code can race or lose
parts of a stream").  A program that names a landing tensor again guards it itself: rows for crossing t + 1 point at
other pages until crossing t's are read (design/pages-and-functions.md, Freeing 7).  Likewise the NIC reads a send
tensor's pages until its publication's `sent` passes them, and no landing wait covers that: a kernel that writes a send
tensor again calls mesh_step_sent(*b.p, b.a, b.v, tid) first on the bindings of the publication that last sent from it
(`sent` is a count, so a later publication's on the same channel covers it too).  polls bounds each wait in loads (0:
NCCL_MESH_POLLS; _stream.polls is that bound, which holds a GPU at most the platform's 5 seconds: rdma/NCCL.md "Bounded
waits").

A publishing kernel is a publication like a group that publishes, under the stream's one commit rule
(_stream.mm MeshStream::published, whose work is throughput: the GPU fed with the fewest command buffers).  The
kernels are the data function's, and the steps they call are the library's under NCCL's contract: every rank issues the
same collectives in the same order and leaves in-flight buffers alone, and the library delivers the bytes."""
from collections import namedtuple

from . import _stream

# the checkout this package is installed from (its editable install: README "Install"), whose library it links
STEPS = _stream.metal_prelude() + '\n'
Crossing = namedtuple('Crossing', 'publish land')
_NONE = (0,) * 6
# the bindings a dispatch declares (_stream.encode reads, writes), by nccl.h ncclMeshFused's order (0 d, 2 the record,
# 3 a): a landing's buffer, record and counts read; a publication's record read, its operand and counts written
_DECLARED = {'land': ((0, 2, 3), ()), 'publish': ((2,), (0, 3))}


def steps():
    """The Metal text a kernel that runs a fused crossing's steps is compiled after (module docstring)."""
    return STEPS


def exchange(send, landings, polls, group=None):
    """The fused exchange over a mesh group of `send` (this rank's part, which its publishing kernel writes to every
    peer) and `landings` (where each other rank's part lands, in rank order, each its own size, which its landing kernel
    reads): a Crossing, each publication's and each landing's six addresses (nccl.h ncclMeshFused).  One the library
    does not fuse raises RuntimeError with the library's reason (ncclInvalidUsage: a tensor that is no region tensor, or
    does not start on a page)."""
    import torch.distributed as dist
    group = group or dist.distributed_c10d._get_default_group()
    publish, land = _stream.fused_exchange(send, list(landings), group.handle, int(polls))
    return Crossing(tuple(map(tuple, publish)), tuple(map(tuple, land)))


def _words(addresses):
    return [w for a in addresses for w in (a & 0xFFFFFFFF, a >> 32)]


def encode(handle, buffers, constants, at, gx, gy, threads, peers, land=None, publish=None):
    """_stream.encode of kernel `handle`, its constants followed by `land`'s `peers` landing bindings and `publish`'s
    `peers` publication bindings (module docstring); one that publishes is committed by the stream's rule for a
    publication."""
    words, reads, writes = list(constants) + [0] * (len(constants) % 2), [], []
    for crossing, bindings in ((land, 'land'), (publish, 'publish')):
        records = getattr(crossing, bindings) if crossing else ()
        words += [w for k in range(peers) for w in _words(records[k] if k < len(records) else _NONE)]
        read, written = _DECLARED[bindings]
        reads += [record[i] for record in records[:peers] for i in read]
        writes += [record[i] for record in records[:peers] for i in written]
    _stream.encode(handle, buffers, words, at, gx, gy, threads, reads, writes)
    if publish:
        _stream.fused_published()
