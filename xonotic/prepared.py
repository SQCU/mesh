"""Page operands and membership for Xonotic's ordinary compiled functions.

This module owns no numerical executor. Programs declare a Function and use
its prepared submission and completion directly.
"""
import numpy as np
import torch
import torch.distributed as dist
import torch_mesh
from torch_mesh.engines import host


def pages(value, *, device='mps'):
    """Populate a declared input once, through its shared host page view."""
    source = np.asarray(value)
    if not source.flags.c_contiguous:
        source = np.ascontiguousarray(source)
    dtype = torch.from_numpy(source).dtype
    result = torch_mesh.empty(source.shape, dtype=dtype, device=device)
    host(result).numpy()[...] = source
    return result


class World:
    """Communicator configuration; numerical transfers belong to Function source."""
    def __init__(self, device):
        from torch_mesh import backend
        import mesh
        dist.init_process_group(backend='mesh')
        self.rank, self.size = dist.get_rank(), dist.get_world_size()
        named = dist.distributed_c10d._get_default_group()
        self.group = backend._mesh(named.group_name, 'Xonotic function')
        self.native = self.group.comm
        self.device = device
        self.members = mesh.configuration_members(self.native)

    def configuration(self, value):
        import mesh
        return mesh.broadcast_configuration(self.native, value)

    def free(self):
        dist.destroy_process_group()
