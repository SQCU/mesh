"""Fused crossings on torch's MPS stream (rdma/nccl.h ncclMeshFusableRooted, ncclMeshFused; ../DESIGN-fused-stream.md):
a two-rank exchange whose two steps the caller's own kernels run, zero copy.  The kernel that makes a rank's part writes
it into the rank's send tensor and publishes it, and the kernel after the call reads the peer's part from the rank's
landing tensor and adds it to its own (the sum is the caller's to make: the library cannot see when a kernel reads a
landing buffer of its own, so it never combines a fused crossing).  The call is ncclSend of the send tensor and
ncclRecv into the landing tensor in one group; it dispatches neither step.  The steps are the library's: a kernel
source is compiled after steps() and calls them on the bindings all_reduce returns, which encode passes it.

  steps()                                    the Metal text a kernel source is compiled after, from the library the
                                             process loaded (ncclMeshMetalPrelude): rdma/mesh.metal as it was built
                                             with, then its steps, struct mesh_step_binding among them
  all_reduce(send, landing, polls, group=None)
                                             the fused exchange of `send` (this rank's part) and `landing` (where the
                                             peer's part lands): contiguous MPS tensors of one dtype and size, each a
                                             region tensor from its first byte (torch_mesh.empty: a lease, which the
                                             bridge sends from and lands in, each starting on a page); a Crossing, its
                                             publication's bindings and its landing's; one the library does not fuse
                                             raises (no slower form runs in its place), any size and dtype fusing
  encode(handle, buffers, constants, at, gx, gy, threads, land=None, publish=None)
                                             _stream.encode's dispatch with `land`'s landing bindings and `publish`'s
                                             publication bindings after the constants; a dispatch that publishes is a
                                             publication under the stream's one commit rule

The kernel's constant block at `at` is its own words padded to an even count, then the landing's mesh_step_binding,
then the publication's (all zero where encode is given none), each address two words, its low word first:
struct { uint words[2k]; mesh_step_binding land, publish; }.  Each all_reduce call is one crossing, encoded before its
publishing kernel: a step kernel that lands crossing t - 1 and publishes crossing t is encoded after crossing t's call,
with crossing t - 1's Crossing as `land` and crossing t's as `publish`.

The publishing kernel: mesh_step_enter(*b.p, b.v, b.progress, first) in one thread, then its threadgroups write the
part into the send tensor (b.d + b.p->dst is its first byte), then the last threadgroup to arrive (the kernel's own
count of arrivals, after every other threadgroup's stores) calls mesh_step_release(*b.p, b.a, tid): one system fence
and one store of the holding's `ready`.  The landing kernel: mesh_step_enter, then mesh_step_await(*l.p, l.a, l.v, tid)
in every threadgroup that reads (each its own wait, bounded by polls: the landed bytes, and this rank's send done, so
the send tensor may be written again once it returns), then element k of the peer's part is
mesh_step_read<T>(l.d + l.p->dst, k), loaded system-coherent (the NIC wrote it).  The landing tensor may be the same
every crossing: its receive is posted only once this rank's publication has stored `ready`, after the kernel that read
it before.  A kernel that writes the send tensor again with no landing kernel between calls
mesh_step_sent(*b.p, b.a, b.v, tid) on the publication's bindings first.  polls bounds each wait in loads (0:
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


def steps():
    """The Metal text a kernel that runs a fused crossing's steps is compiled after (module docstring)."""
    return STEPS


def all_reduce(send, landing, polls, group=None):
    """The fused exchange over a mesh group of `send` (this rank's part, which its publishing kernel writes) and
    `landing` (where the peer's part lands, which its landing kernel reads and adds to its own), the peer rank 1 - rank
    (past two ranks, ranks 0 and 1 exchange, and a rank past 1 names no peer and raises): a Crossing, its publication's
    and its landing's six addresses each (nccl.h ncclMeshFused).  One the library does not fuse raises RuntimeError with
    the library's reason (ncclInvalidUsage: a tensor that is no region tensor, or does not start on a page)."""
    import torch.distributed as dist
    group = group or dist.distributed_c10d._get_default_group()
    publish, land = _stream.fused_all_reduce(send, landing, group.handle, int(polls))
    return Crossing(tuple(publish), tuple(land))


def _words(addresses):
    return [w for a in addresses for w in (a & 0xFFFFFFFF, a >> 32)]


def encode(handle, buffers, constants, at, gx, gy, threads, land=None, publish=None):
    """_stream.encode of kernel `handle`, its constants followed by `land`'s landing bindings and `publish`'s
    publication bindings (module docstring); one that publishes is committed by the stream's rule for a publication."""
    words = list(constants) + [0] * (len(constants) % 2)
    words += _words(land.land if land else _NONE) + _words(publish.publish if publish else _NONE)
    _stream.encode(handle, buffers, words, at, gx, gy, threads)
    if publish:
        _stream.fused_published()
