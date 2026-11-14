# Complex nonsymmetric eig native client

One direct public compile_module → Program MPS case after the connected source is reviewed. No diagnostic hooks, extra outputs, altered lifetimes, alternate evaluator or fallback. Values are compared without assuming LAPACK ordering; the primary contract is the eigenpair residual and normalized columns.

```python
import itertools, json, torch
from mesh_compiler.driver import compile_module
from mesh_compiler.runtime import Program
class Eigen(torch.nn.Module):
    def forward(self, a):
        return torch.linalg.eig(a)
a = torch.tensor([[1+.5j, 2-1j, -.5+.25j], [.25+.75j, -2+1j, 1.5-.5j], [1-.25j, .75+1.25j, 3-.75j]], dtype=torch.complex64)
scheduled = compile_module(Eigen(), (a,), {'p': (0, torch.device('mps'))}, lambda node: 'p')
program = Program(scheduled)
w, v = program(a)
w, v, reference_a = w.cpu().cdouble(), v.cpu().cdouble(), a.cdouble()
reference = torch.linalg.eigvals(reference_a)
norm = torch.linalg.norm(reference_a)
matching = min(float(torch.max(torch.abs(w-reference[list(order)]))/norm) for order in itertools.permutations(range(3)))
lengths = torch.linalg.vector_norm(v, dim=0)
phase = v[v.abs().argmax(dim=0), torch.arange(3)]
print(json.dumps({'eigenvalue_error': matching, 'residual': float(torch.linalg.norm(reference_a@v-v*w)/(norm*torch.linalg.norm(v))), 'normalization': float(torch.max(torch.abs(lengths-1))), 'phase_imaginary': float(phase.imag.abs().max()), 'finite': bool(torch.isfinite(w).all() and torch.isfinite(v).all()), 'values_real': w.real.tolist(), 'values_imag': w.imag.tolist(), 'vectors_real': v.real.tolist(), 'vectors_imag': v.imag.tolist()}))
```

Actual stdin additionally labels compile, prepare, invoke, and result boundaries. The matching reference is only an output metric, not an implementation backend. Source hashes, process exit, stdout/stderr and archive restoration statements are retained with the result.
