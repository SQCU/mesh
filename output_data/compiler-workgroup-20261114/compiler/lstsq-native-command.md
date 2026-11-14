# Wide fixed-output SVD least-squares client

Working directory `/Users/mdot/dox/mesh`, interpreter `/Users/mdot/.venv-mesh-uv/bin/python -u`, environment from pinv-native-command.md. One public compile_module→Program MPS case, with no diagnostic hook.

```python
import json, torch
from mesh_compiler.driver import compile_module
from mesh_compiler.runtime import Program
class Least(torch.nn.Module):
    def forward(self, a, b):
        return torch.linalg.lstsq(a, b, driver='gelsd')
a = torch.tensor([[1., 2., 0.], [0., 1., 3.]])
b = torch.tensor([[1., 2.], [3., 4.]])
scheduled = compile_module(Least(), (a, b), {'p': (0, torch.device('mps'))}, lambda node: 'p')
program = Program(scheduled)
x, residual, rank, s = program(a, b)
x, residual, s = x.cpu().double(), residual.cpu().double(), s.cpu().double()
rank = rank.cpu()
aa, bb = a.double(), b.double()
reference = torch.linalg.lstsq(aa, bb, driver='gelsd')
print(json.dumps({
    'solution_error': float(torch.linalg.norm(x-reference.solution)/torch.linalg.norm(reference.solution)),
    'equation_residual': float(torch.linalg.norm(aa@x-bb)/torch.linalg.norm(bb)),
    'normal_residual': float(torch.linalg.norm(aa.mT@(aa@x-bb))/(torch.linalg.norm(aa)*torch.linalg.norm(bb))),
    'singular_value_error': float(torch.linalg.norm(s-reference.singular_values)/torch.linalg.norm(reference.singular_values)),
    'rank': rank.tolist(), 'reference_rank': reference.rank.tolist(),
    'shapes': [list(z.shape) for z in (x, residual, rank, s)],
    'finite': bool(torch.isfinite(x).all() and torch.isfinite(s).all()),
    'solution': x.tolist(), 'reference_solution': reference.solution.tolist(),
    'singular_values': s.tolist(), 'reference_singular_values': reference.singular_values.tolist()}))
```

The actual stdout additionally labels compile, prepare, invoke and result boundaries. Complete stdout/stderr hashes are recorded in lstsq-native-result.json.
