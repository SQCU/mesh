# Full rectangular rank-deficient SVD client

Working directory `/Users/mdot/dox/mesh`, interpreter `/Users/mdot/.venv-mesh-uv/bin/python -u`, same environment as pinv-native-command.md. One direct public compile_module→Program MPS case; no diagnostic hooks.

```python
import json, torch
from mesh_compiler.driver import compile_module
from mesh_compiler.runtime import Program
class Singular(torch.nn.Module):
    def forward(self, a):
        return torch.linalg.svd(a, full_matrices=True)
a = torch.tensor([[1., 2.], [2., 4.], [3., 6.]])
print(json.dumps({'stage':'compile', 'shape':list(a.shape), 'dtype':str(a.dtype), 'full_matrices':True}), flush=True)
scheduled = compile_module(Singular(), (a,), {'p':(0, torch.device('mps'))}, lambda node:'p')
print(json.dumps({'stage':'prepare'}), flush=True)
program = Program(scheduled)
print(json.dumps({'stage':'invoke'}), flush=True)
u, s, vh = (x.cpu().double() for x in program(a))
aa = a.double()
reference = torch.linalg.svdvals(aa)
norm = torch.linalg.norm(aa)
print(json.dumps({'stage':'result',
    'singular_value_error':float(torch.max(torch.abs(s-reference))/norm),
    'residual':float(torch.linalg.norm((u[:,:2]*s)@vh-aa)/norm),
    'left_orthogonality':float(torch.linalg.norm(u.mT@u-torch.eye(3,dtype=torch.float64))),
    'right_orthogonality':float(torch.linalg.norm(vh@vh.mT-torch.eye(2,dtype=torch.float64))),
    'finite':bool(torch.isfinite(u).all() and torch.isfinite(s).all() and torch.isfinite(vh).all()),
    'shapes':[list(x.shape) for x in (u,s,vh)], 'values':s.tolist(),
    'reference_values':reference.tolist(), 'u':u.tolist(), 'vh':vh.tolist()}), flush=True)
```

Attempt 1 reached joint binding and stopped on contradictory source storage roots before numerical invocation. The same case is the retry after the producer's alias correspondence is corrected.
