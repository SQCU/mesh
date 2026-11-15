# design/compilation-composition-change-request.md items 1, 3 and 5; metal-microbench docs/provider.md
import math

import torch
import torch.nn.functional as F


_LIBRARY = torch.library.Library('mesh', 'FRAGMENT')
_LIBRARY.define('fragment(Tensor codes, Tensor tiles, Tensor bands, Tensor row_scale, Tensor col_scale, '
                'Tensor rows, Tensor columns, ScalarType dtype) -> Tensor')
fragment = torch.ops.mesh.fragment.default


# compile-time callback (decorator registration torch.library.register_fake(fragment)): callback inversion of control is outside the working
# specification, and runtime callback overhead is inadmissible in performance-critical code.
@mesh_callbacks_are_outside_spec(torch.library.register_fake(fragment))
def _shape(codes, tiles, bands, row_scale, col_scale, rows, columns, dtype):
    # compile-time callback (registered by torch.library.register_fake(fragment)): callback inversion of control is outside the working
    # specification, and runtime callback overhead is inadmissible in performance-critical code.
    mesh_callbacks_are_outside_spec
    return torch.empty((rows.numel(), columns.numel()), device=codes.device, dtype=dtype)


class Matrix(torch.nn.Module):
    def __init__(self, fields, rows=None, columns=None):
        super().__init__()
        for name, value in fields.items():
            self.register_buffer(name, value)
        self.fields = tuple(fields)
        nr, nc = fields['tiles'].shape[0] * 32, fields['tiles'].shape[1] * 32
        rows = torch.arange(nr) if rows is None else rows.to(device='cpu', dtype=torch.int64)
        columns = torch.arange(nc) if columns is None else columns.to(device='cpu', dtype=torch.int64)
        row_order, col_order = (fields[k].to(device='cpu', dtype=torch.int64) for k in ('row_order', 'col_order'))
        selected = torch.argsort(col_order)[columns]
        permutation = torch.argsort(selected)
        self.register_buffer('rows', torch.argsort(row_order)[rows])
        self.register_buffer('input_ids', columns)
        self.register_buffer('input_order', permutation)
        self.register_buffer('columns', torch.arange(nc) if 'rotation' in fields or 'signs' in fields else selected[permutation])
        self.shape = (rows.numel(), columns.numel())
        self._mesh_constants = tuple(self._buffers)

    def select(self, rows=None, columns=None):
        return Matrix({name: getattr(self, name) for name in self.fields}, rows, columns)

    def transform(self, x):
        if hasattr(self, 'rotation'):
            basis = self.rotation.index_select(1, self.input_ids).float() * self.rotation_scale[:, None]
            return F.linear(x, basis.to(x.dtype))
        if hasattr(self, 'signs'):
            total = self.signs.numel()
            full = x.new_zeros((*x.shape[:-1], total)).index_copy(-1, self.input_ids, x)
            mixed = (full * self.signs.to(x.dtype)).reshape(*x.shape[:-1], self.hadamard.shape[0], -1)
            width, step = mixed.shape[-1], 1
            while step < width:
                pair = mixed.reshape(*mixed.shape[:-1], -1, 2 * step)
                left, right = pair[..., :step], pair[..., step:]
                mixed = torch.cat((left + right, left - right), -1).reshape(mixed.shape)
                step *= 2
            mixed = torch.einsum('ab,...bn->...an', self.hadamard.to(x.dtype), mixed).flatten(-2)
            return (mixed / math.sqrt(total)).index_select(-1, self.col_order.long())
        return x.index_select(-1, self.input_order)

    def product(self, transformed):
        weight = fragment(self.codes, self.tiles, self.bands, self.row_scale, self.col_scale,
                          self.rows, self.columns, transformed.dtype)
        return F.linear(transformed, weight)

    def forward(self, x):
        return self.product(self.transform(x))


class FFN(torch.nn.Module):
    def __init__(self, matrices, first, count):
        super().__init__()
        gate, up, down = matrices
        hidden = gate.row_order[first:first + count]
        self.gate, self.up, self.down = gate.select(rows=hidden), up.select(rows=hidden), down.select(columns=hidden)
        fields = ('col_order', 'rotation', 'rotation_scale', 'signs', 'hadamard')
        self.shared = all(hasattr(gate, name) == hasattr(up, name) and
                          (not hasattr(gate, name) or torch.equal(getattr(gate, name), getattr(up, name))) for name in fields)

    def forward(self, x):
        transformed = self.gate.transform(x)
        gate = self.gate.product(transformed)
        up = self.up.product(transformed if self.shared else self.up.transform(x))
        return self.down(F.gelu(gate, approximate='tanh') * up)


def load(path):
    from safetensors import safe_open
    from safetensors.torch import load_file
    tensors = load_file(str(path), device='cpu')
    with safe_open(str(path), framework='pt') as source:
        metadata = source.metadata() or {}
    matrices = {}
    for name in sorted(k[:-6] for k in tensors if k.endswith('.codes')):
        fields = {k[len(name) + 1:]: v for k, v in tensors.items() if k.startswith(name + '.')}
        fields['codes'] = fields['codes'].view(torch.uint32)
        fields['tiles'], fields['bands'] = fields['tiles'].view(torch.uint32), fields['bands'].view(torch.uint32)
        if 'rotation' in fields and metadata.get('basis_layout') != 'code-order':
            order = fields['col_order'].long()
            fields['rotation'] = fields['rotation'].index_select(1, order).mT.contiguous()
            fields['rotation_scale'] = fields['rotation_scale'].index_select(0, order)
        matrices[name] = Matrix(fields)
    return matrices
