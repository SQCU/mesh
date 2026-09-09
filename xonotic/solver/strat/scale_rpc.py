from .cast_header import scale_operator
from .matmul import matrix_multiply_transpose_left


class RemoteCross:
    tensor_function = staticmethod(matrix_multiply_transpose_left)

    def __init__(self, mesh, node, backlog, stopping=None, owner=None):
        self.mesh, self.node, self.backlog = mesh, node, backlog
        self.stopping = stopping if stopping is not None else {'signal': None}
        self.owner = owner
        self.last = {'completed': False, 'backend': 'persistent_mesh_metal'}


class RemoteScale(RemoteCross):
    tensor_function = staticmethod(scale_operator)
