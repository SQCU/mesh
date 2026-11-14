# Batched real values-only native client

The existing public compile_module→Program caller, with the matched native libraries. This one case combines a two-member transposed float64 batch, values-only eig, and a downstream scale. The second member has spectrum8.5,2.5±4i before the output scale. The CPU reference is an output metric only. No diagnostic hook or alternate numerical implementation is installed.

```python
import hashlib, itertools, json, pathlib
root = pathlib.Path('/Users/mdot/dox/mesh')
out = root / 'output_data/compiler-workgroup-20261114/compiler'
paths = [root / ('compiler/mesh_compiler/' + name + '.py') for name in
         ('operations', 'decompose', 'driver', 'schedule', 'native', 'metal')]
paths += [pathlib.Path('/Users/mdot/metal-microbench/.build') / name for name in
          ('libmetal_recording.dylib', 'libmetal_ir.dylib')]
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(out / 'real-eigen-batched-source.json').write_text(json.dumps(before, indent=2) + '\n')
import torch
from mesh_compiler.driver import compile_module
from mesh_compiler.runtime import Program
imported = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
print(json.dumps({'import_complete': True, 'source_unchanged_during_import': imported == before}), flush=True)
class Values(torch.nn.Module):
    def forward(self, a):
        return torch.linalg.eigvals(a) * 0.5
base = torch.tensor([[1., -2., 2.], [-.5, 1.5, 2.5], [-2.5, .5, 3.5]], dtype=torch.float64)
a = torch.stack((base, 2 * base + .5 * torch.eye(3, dtype=torch.float64))).mT
print(json.dumps({'input_shape': list(a.shape), 'input_stride': list(a.stride()), 'input_dtype': str(a.dtype)}), flush=True)
print('compile', flush=True)
scheduled = compile_module(Values(), (a,), {'p': (0, torch.device('mps'))}, lambda node: 'p')
print('prepare', flush=True)
program = Program(scheduled)
print('invoke', flush=True)
actual = program(a)
values = actual.cpu()
reference = torch.linalg.eigvals(a) * 0.5
errors = [min(float(torch.max(torch.abs(got - ref[list(order)])) / torch.linalg.norm(matrix))
              for order in itertools.permutations(range(3)))
          for got, ref, matrix in zip(values, reference, a)]
after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
result = {'input_shape': list(a.shape), 'input_stride': list(a.stride()), 'input_dtype': str(a.dtype),
          'output_shape': list(values.shape), 'output_dtype': str(values.dtype), 'eigenvalue_errors': errors,
          'finite': bool(torch.isfinite(values).all()), 'real': values.real.tolist(), 'imag': values.imag.tolist(),
          'changed_paths': [str(p) for p in paths if before[str(p)] != after[str(p)]]}
(out / 'real-eigen-batched-result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
```
