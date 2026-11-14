# Complex Hermitian native client

One direct public compile_module→Program MPS case, using the environment in `pinv-native-command.md`. No diagnostic hooks, added outputs, or changed lifetimes. The lower triangle is authoritative; the upper triangle is deliberately inconsistent and the imaginary diagonal must be ignored.

```python
import json, torch
from mesh_compiler.driver import compile_module
from mesh_compiler.runtime import Program
class Hermitian(torch.nn.Module):
    def forward(self, a):
        return torch.linalg.eigh(a, UPLO='L')
a = torch.tensor([[2+7j, 42+11j], [1+2j, 5-9j]], dtype=torch.complex64)
scheduled = compile_module(Hermitian(), (a,), {'p': (0, torch.device('mps'))}, lambda node: 'p')
program = Program(scheduled)
w, q = program(a)
w, q = w.cpu().double(), q.cpu().cdouble()
aa = a.cdouble()
lower = torch.tril(aa, -1)
h = lower + lower.mH + torch.diag(aa.diagonal().real).to(torch.complex128)
reference = torch.linalg.eigvalsh(aa, UPLO='L')
norm = torch.linalg.norm(h)
print(json.dumps({
    'eigenvalue_error': float(torch.max(torch.abs(w-reference))/norm),
    'residual': float(torch.linalg.norm(h@q-q*w)/norm),
    'orthogonality': float(torch.linalg.norm(q.mH@q-torch.eye(2,dtype=torch.complex128))),
    'finite': bool(torch.isfinite(w).all() and torch.isfinite(q).all()),
    'values': w.tolist(), 'reference_values': reference.tolist(),
    'vectors_real': q.real.tolist(), 'vectors_imag': q.imag.tolist()}))
```

The actual stdin additionally labels compile, prepare, invoke and result boundaries. Raw output hashes and process exit are retained in the result record.
