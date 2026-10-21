"""Fused crossings on torch's MPS stream (rdma/nccl.h ncclMeshFusableRooted, ncclMeshFused; ../DESIGN-fused-stream.md): a
two-rank in-place float16 all-reduce whose two steps the caller's own kernels run, the kernel that makes a rank's part
publishing it into the crossing's slot and the kernel after the call landing the peer's part and adding it to its own;
the call dispatches neither step.  The steps are the library's: a kernel source is compiled after steps() and
calls them (mesh_step_enter, mesh_step_store and mesh_step_release to publish; mesh_step_enter, mesh_step_await and
mesh_step_read to land) on the bindings all_reduce returns, which encode passes it.

  steps()                                    the Metal text a kernel source is compiled after: rdma/mesh.metal, then the
                                             library's steps (ncclMeshMetalSteps), struct mesh_step_binding among them
  all_reduce(tensor, polls, group=None)      the fused all-reduce of tensor (a float16 region tensor: empty()) in
                                             place: a Crossing, its publication's bindings and its landing's; one that
                                             is not fusable raises, and nothing runs a slower form in its place
  encode(handle, buffers, constants, at, gx, gy, threads, land=None, publish=None)
                                             _stream.encode's dispatch with `land`'s landing bindings and `publish`'s
                                             publication bindings after the constants

The kernel's constant block at `at` is its own words padded to an even count, then the landing's mesh_step_binding,
then the publication's (all zero where encode is given none), each address two words, its low word first:
struct { uint words[2k]; mesh_step_binding land, publish; }.  Each all_reduce call is one crossing, encoded before its
publishing kernel: a step kernel that lands crossing t - 1 and publishes crossing t is encoded after crossing t's
call, with crossing t - 1's Crossing as `land` and crossing t's as `publish`.  The operand never holds the sum: the
landing kernel reads its own part (the landing's d + dst) and the peer's (mesh_step_read at its s) and keeps the sum
where it needs it.  polls bounds each landing's wait in loads (0: NCCL_MESH_POLLS), the caller's from its plan
(rdma/NCCL.md "Bounded waits").

Nothing here commits torch's command buffer: the stream's commit rule (keeping the GPU fed with the fewest command
buffers) and the host waits commit it, as they commit every kernel.  A host that blocks while its publication sits in
an open buffer stalls its peer; the program's order is its own to keep, and the peer's bounded wait ends such a stall
as a value.  The operator, 2026-10-21: "we are here to REMOVE fences, REMOVE guards, REMOVE waits. it is the
responsibility of DATA PROCESSING FUNCTIONS to not overwrite each other's buffers"."""
from collections import namedtuple
from pathlib import Path

from . import _stream

# the checkout this package is installed from (its editable install: README "Install"), whose library it links
STEPS = (Path(__file__).resolve().parents[2] / 'mesh.metal').read_text() + '\n' + _stream.fused_steps() + '\n'
Crossing = namedtuple('Crossing', 'publish land')
_NONE = (0,) * 6


def steps():
    """The Metal text a kernel that runs a fused crossing's steps is compiled after (module docstring)."""
    return STEPS


def all_reduce(tensor, polls, group=None):
    """The fused in-place sum of `tensor` (a contiguous float16 MPS region tensor, torch_mesh.empty) over a two-rank
    mesh group: a Crossing, its publication's and its landing's six addresses each (nccl.h ncclMeshFused).  Not fusable
    raises RuntimeError naming why (libnccl-mesh's ncclInvalidUsage)."""
    import torch.distributed as dist
    group = group or dist.distributed_c10d._get_default_group()
    publish, land = _stream.fused_all_reduce(tensor, group.handle, int(polls))
    return Crossing(tuple(publish), tuple(land))


def _words(addresses):
    return [w for a in addresses for w in (a & 0xFFFFFFFF, a >> 32)]


def encode(handle, buffers, constants, at, gx, gy, threads, land=None, publish=None):
    """_stream.encode of kernel `handle`, its constants followed by `land`'s landing bindings and `publish`'s
    publication bindings (module docstring)."""
    words = list(constants) + [0] * (len(constants) % 2)
    words += _words(land.land if land else _NONE) + _words(publish.publish if publish else _NONE)
    _stream.encode(handle, buffers, words, at, gx, gy, threads)
