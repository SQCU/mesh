import time

from .cast_header import scale_operator
from .matmul import matrix_multiply_transpose_left


class RemoteCross:
    """57628982^:scale_rpc.RemoteCross, the row-Gram cross contraction Rᵀ V placed on the mesh, onto the library
    (products.World): the contraction's rows are divided by the world's `cross` parts, each rank contracting its rows
    of both operands, the shares summed (Megatron's copy and reduce, products.py), so autograd derives both operand
    VJPs from the same program on every rank.  Parts [0, 1] put the whole contraction on rank 1, the 09-06 placement
    (measurements/policy-feature-integration-20260906 rdma-cross.json)."""
    tensor_function = staticmethod(matrix_multiply_transpose_left)
    product = 'cross'

    def __init__(self, world, owner=None):
        self.world, self.owner = world, owner
        self.last = {'completed': False, 'backend': 'libnccl-mesh', 'product': self.product,
                     'parts': world.parts(self.product)}

    def __call__(self, lhs, rhs):
        if self.world.parts(self.product) is None:
            return matrix_multiply_transpose_left(lhs, rhs)
        started = time.perf_counter()
        lo, hi = self.world.span(self.product, lhs.shape[0])
        lhs, rhs = self.world.copy(lhs), self.world.copy(rhs)
        product = self.world.reduce(matrix_multiply_transpose_left(lhs[lo:hi], rhs[lo:hi]))
        self.last = {'completed': True, 'backend': 'libnccl-mesh', 'product': self.product,
                     'parts': self.world.parts(self.product), 'rank': self.world.rank, 'rows': [lo, hi],
                     'row_mass': int(lhs.shape[0]), 'issue_s': time.perf_counter() - started}
        return product


class RemoteScale(RemoteCross):
    """57628982^:scale_rpc.RemoteScale, the routed scale experts placed on the mesh, onto the library: the experts'
    hidden width is divided by the world's `scale` parts (each rank's model keeps its columns, cast_header.Wally
    scale_shard), the shares summed (cast_header.scale_operator with its world)."""
    tensor_function = staticmethod(scale_operator)
    product = 'scale'

    def __call__(self, *tensors, topk):
        if self.world.parts(self.product) is None:
            return scale_operator(*tensors, topk=topk)
        started = time.perf_counter()
        result = scale_operator(*tensors, topk=topk, world=self.world)
        self.last = {'completed': True, 'backend': 'libnccl-mesh', 'product': self.product,
                     'parts': self.world.parts(self.product), 'rank': self.world.rank,
                     'hidden_columns': int(tensors[3].shape[-1]),
                     'issue_s': time.perf_counter() - started}
        return result
