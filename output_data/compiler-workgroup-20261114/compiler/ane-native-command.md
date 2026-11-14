# Requested-ANE Hermitian native client

One direct public compile_module → Program case after the ANE runtime-domain/reduction/storage source is reviewed and the shared library matches. The actual MLComputePlan summary is read from the prepared native program. A successful requested placement does not establish that Core ML selected the Neural Engine for every operation, or establish work scaling/performance.

```python
import json, torch
from mesh_compiler.driver import compile_module
from mesh_compiler.runtime import Program
class Hermitian(torch.nn.Module):
    def forward(self, a):
        return torch.linalg.eigh(a)
a = torch.tensor([[2.,1.],[1.,4.]], dtype=torch.float32)
scheduled = compile_module(Hermitian(), (a,), {'p':(0,'ane')}, lambda node:'p')
program = Program(scheduled)
w, q = program(a)
w, q, aa = w.cpu().double(), q.cpu().double(), a.double()
norm = torch.linalg.norm(aa)
print(json.dumps({'eigenvalue_error': float(torch.max(torch.abs(w-torch.linalg.eigvalsh(aa)))/norm), 'residual': float(torch.linalg.norm(aa@q-q*w)/norm), 'orthogonality': float(torch.linalg.norm(q.T@q-torch.eye(2,dtype=torch.float64))), 'finite': bool(torch.isfinite(w).all() and torch.isfinite(q).all()), 'values': w.tolist(), 'vectors': q.tolist(), 'placements':program.function.placements()}))
```

No source fallback or numerical placement rewrite is used. Actual stdin labels compilation, preparation, invocation and result; raw failure output is retained if any stage fails.
