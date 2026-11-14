# Hermitian pinv client

Same public compile_module→Program path as the eigensolver case, on one MPS processor. This case is real, rank-deficient and indefinite, so the result exercises eigenvalue thresholding and a negative signed reciprocal. It does not establish general spectral, complex, ANE, or performance coverage.

Working directory `/Users/mdot/dox/mesh`; interpreter `/Users/mdot/.venv-mesh-uv/bin/python -u`. Environment:

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0
DYLD_INSERT_LIBRARIES=/Users/mdot/metal-microbench/.build/libmetal_recording.dylib
DYLD_LIBRARY_PATH=/Users/mdot/metal-microbench/.build:/Users/mdot/dox/mesh/rdma
PYTHONPATH=/Users/mdot/dox/mesh/compiler:/Users/mdot/dox/mesh/rdma
```

Exact stdin:

```python
import json, torch
from mesh_compiler.driver import compile_module
from mesh_compiler.runtime import Program
class Pinv(torch.nn.Module):
    def forward(self, a):
        return torch.linalg.pinv(a, hermitian=True)
a = torch.tensor([[1., 2., 0.], [2., 4., 0.], [0., 0., -2.]])
print(json.dumps({'stage': 'compile', 'shape': list(a.shape), 'dtype': str(a.dtype)}), flush=True)
scheduled = compile_module(Pinv(), (a,), {'p': (0, torch.device('mps'))}, lambda node: 'p')
print(json.dumps({'stage': 'prepare'}), flush=True)
program = Program(scheduled)
print(json.dumps({'stage': 'invoke'}), flush=True)
x = program(a).cpu().double()
aa = a.double()
reference = torch.linalg.pinv(aa, hermitian=True)
ax = aa @ x
print(json.dumps({'stage': 'result',
    'relative_error': float(torch.linalg.norm(x - reference) / torch.linalg.norm(reference)),
    'axa_residual': float(torch.linalg.norm(ax @ aa - aa) / torch.linalg.norm(aa)),
    'xax_residual': float(torch.linalg.norm(x @ aa @ x - x) / torch.linalg.norm(x)),
    'symmetry_residual': float(torch.linalg.norm(ax - ax.mT)),
    'finite': bool(torch.isfinite(x).all()),
    'result': x.tolist(), 'reference': reference.tolist()}), flush=True)
```
