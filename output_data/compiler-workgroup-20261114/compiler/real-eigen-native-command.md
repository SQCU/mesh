# Real nonsymmetric native client

The existing public compile_module→Program client, using the same environment as nonsymmetric-native-command.md. One dense real3×3 input has eigenvalues4 and1±2i. It is a real similarity transform of that block diagonal form, with exact binary entries. No diagnostic hook, extra output, altered lifetime or alternate implementation is involved.

```python
import itertools, json, torch
from mesh_compiler.driver import compile_module
from mesh_compiler.runtime import Program
class Eigen(torch.nn.Module):
    def forward(self, a):
        return torch.linalg.eig(a)
a = torch.tensor([[1., -2., 2.], [-.5, 1.5, 2.5], [-2.5, .5, 3.5]], dtype=torch.float32)
print('compile', flush=True)
scheduled = compile_module(Eigen(), (a,), {'p': (0, torch.device('mps'))}, lambda node: 'p')
print('prepare', flush=True)
program = Program(scheduled)
print('invoke', flush=True)
w, v = program(a)
w, v, aa = w.cpu().cdouble(), v.cpu().cdouble(), a.cdouble()
reference = torch.linalg.eigvals(aa)
norm = torch.linalg.norm(aa)
matching = min(float(torch.max(torch.abs(w-reference[list(order)]))/norm) for order in itertools.permutations(range(3)))
lengths = torch.linalg.vector_norm(v, dim=0)
print(json.dumps({'eigenvalue_error': matching, 'residual': float(torch.linalg.norm(aa@v-v*w)/(norm*torch.linalg.norm(v))), 'normalization': float(torch.max(torch.abs(lengths-1))), 'finite': bool(torch.isfinite(w).all() and torch.isfinite(v).all()), 'values_real': w.real.tolist(), 'values_imag': w.imag.tolist(), 'vectors_real': v.real.tolist(), 'vectors_imag': v.imag.tolist()}), flush=True)
```

The reference is only an output metric. Raw stdout/stderr, terminal status and source/binary hashes accompany the actual observation when it runs. This document records the next input, not an already completed run.
