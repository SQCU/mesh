# Exact current Hermitian client

Working directory `/Users/mdot/dox/mesh`. Environment:

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0
DYLD_INSERT_LIBRARIES=/Users/mdot/metal-microbench/.build/libmetal_recording.dylib
DYLD_LIBRARY_PATH=/Users/mdot/metal-microbench/.build:/Users/mdot/dox/mesh/rdma
PYTHONPATH=/Users/mdot/dox/mesh/compiler:/Users/mdot/dox/mesh/rdma
```

Interpreter `/Users/mdot/.venv-mesh-uv/bin/python -u`, this stdin:

```python
import json, torch
from mesh_compiler.driver import compile_module
from mesh_compiler.runtime import Program

class Hermitian(torch.nn.Module):
    def forward(self, a):
        return torch.linalg.eigh(a)

a = torch.tensor([[4., 1., 0., 0.], [1., 3., 1., 0.],
                  [0., 1., 2., 1.], [0., 0., 1., 1.]])
print(json.dumps({'stage': 'compile', 'shape': list(a.shape), 'dtype': str(a.dtype)}), flush=True)
scheduled = compile_module(Hermitian(), (a,), {'p': (0, torch.device('mps'))}, lambda node: 'p')
print(json.dumps({'stage': 'prepare'}), flush=True)
program = Program(scheduled)
print(json.dumps({'stage': 'invoke'}), flush=True)
values, vectors = program(a)
values, vectors = values.cpu(), vectors.cpu()
reference = torch.linalg.eigvalsh(a.double())
norm = torch.linalg.norm(a.double())
print(json.dumps({
    'stage': 'result',
    'eigenvalue_error': float(torch.max(torch.abs(values.double() - reference)) / norm),
    'residual': float(torch.linalg.norm(a.double() @ vectors.double() - vectors.double() * values.double()) / norm),
    'orthogonality': float(torch.linalg.norm(vectors.double().mT @ vectors.double() - torch.eye(4, dtype=torch.float64))),
    'values': values.tolist(), 'reference_values': reference.tolist(),
    'finite': bool(torch.isfinite(values).all() and torch.isfinite(vectors).all())}), flush=True)
```

Attempt2 (`eigh-native-2.stdout/.stderr`) stops inside `compile_module` at the Step replay binding value652/slot4. For the assigned actual-operand diagnostic, execute only through that same `compile_module` call and stop there; do not prepare/invoke, change the model, alter placement or create another evaluator. Supervisor owns the numerical invocation after the concrete replay defect is corrected.
