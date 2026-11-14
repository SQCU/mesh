# Singleton, empty and invalid-input native caller

One existing public compile_module→Program caller returns singleton eig and empty eigvals. The same prepared Program is called with a valid scalar, a NaN input, then another valid scalar. The observation records the actual error and recovery; it does not force nonconvergence or change the algorithm.

```python
import hashlib, json, pathlib
root = pathlib.Path('/Users/mdot/dox/mesh')
out = root / 'output_data/compiler-workgroup-20261114/compiler'
paths = [root / ('compiler/mesh_compiler/' + name + '.py') for name in
         ('operations', 'decompose', 'driver', 'schedule', 'native', 'metal')]
paths += [pathlib.Path('/Users/mdot/metal-microbench/.build') / name for name in
          ('libmetal_recording.dylib', 'libmetal_ir.dylib')]
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(out / 'real-eigen-domain-source.json').write_text(json.dumps(before, indent=2) + '\n')
import torch
from mesh_compiler.driver import compile_module
from mesh_compiler.runtime import Program
print(json.dumps({'import_complete': True, 'source_unchanged_during_import':
                  before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}), flush=True)
class Edge(torch.nn.Module):
    def forward(self, a, empty):
        w, v = torch.linalg.eig(a)
        return w, v, torch.linalg.eigvals(empty)
a, empty = torch.tensor([[3.]], dtype=torch.float32), torch.empty((0, 0), dtype=torch.float32)
print('compile', flush=True)
scheduled = compile_module(Edge(), (a, empty), {'p': (0, torch.device('mps'))}, lambda node: 'p')
print('prepare', flush=True)
program = Program(scheduled)
print('invoke valid', flush=True)
w, v, z = program(a, empty)
first = {'value_error': float(torch.max(torch.abs(w.cpu() - 3))),
         'vector_error': float(torch.max(torch.abs(v.cpu() - 1))),
         'empty_shape': list(z.shape), 'empty_dtype': str(z.dtype)}
print('invoke invalid', flush=True)
error = None
try:
    program(torch.full_like(a, float('nan')), empty)
except RuntimeError as failure:
    error = str(failure)
print('invoke recovery', flush=True)
w, v, z = program(-2 * torch.ones_like(a), empty)
result = {'first': first, 'invalid_error': error,
          'recovered_value_error': float(torch.max(torch.abs(w.cpu() + 2))),
          'recovered_vector_error': float(torch.max(torch.abs(v.cpu() - 1))),
          'recovered_finite': bool(torch.isfinite(w).all() and torch.isfinite(v).all()),
          'empty_shape': list(z.shape),
          'changed_paths': [str(p) for p in paths if before[str(p)] != hashlib.sha256(p.read_bytes()).hexdigest()]}
(out / 'real-eigen-domain-result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
```
