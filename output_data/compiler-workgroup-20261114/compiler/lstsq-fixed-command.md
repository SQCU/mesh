# Tall batched fixed-result least-squares caller

Use the same public compile_module→Program caller as lstsq-native-command.md, after the paired source patches and matching native ABI are released. One F32 two-member tall batch contains a full-rank and a rank-deficient matrix. Its source selects solution/rank/singular values, reads only fixed residual dtype/device facts, retains a source assertion, and composes the solution with a scale. This is the originally blocked fixed-result domain; no custom operator is invoked by the source client.

```python
import hashlib, json, pathlib, torch
from mesh_compiler.driver import compile_module
from mesh_compiler.runtime import Program

class Least(torch.nn.Module):
    def forward(self, a, b):
        result = torch.linalg.lstsq(input=a, b=b, rcond=1e-4, driver='gelsd')
        torch._assert_async((result.rank >= 0).all(), 'source rank assertion')
        scale = 2.0 if result.residuals.dtype == a.dtype and result.residuals.device == a.device else 3.0
        return result.solution * scale, result.rank, result.singular_values

a = torch.tensor([[[1., 0.], [0., 2.], [1., 1.]],
                  [[1., 2.], [2., 4.], [-1., -2.]]])
b = torch.tensor([[[1.], [2.], [.5]], [[1.], [3.], [-2.]]])
print('compile', flush=True)
scheduled = compile_module(Least(), (a, b), {'p': (0, torch.device('mps'))}, lambda node: 'p')
print('prepare', flush=True)
program = Program(scheduled)
print('invoke', flush=True)
x, rank, singular = program(a, b)
x, rank, singular = x.cpu().double(), rank.cpu(), singular.cpu().double()
reference = torch.linalg.lstsq(a.double(), b.double(), rcond=1e-4, driver='gelsd')
expected = reference.solution * 2
result = {
    'solution_errors': [float(torch.linalg.norm(got-ref)/torch.linalg.norm(ref)) for got, ref in zip(x, expected)],
    'singular_value_errors': [float(torch.linalg.norm(got-ref)/torch.linalg.norm(ref))
                              for got, ref in zip(singular, reference.singular_values)],
    'rank': rank.tolist(), 'reference_rank': reference.rank.tolist(),
    'shapes': [list(value.shape) for value in (x, rank, singular)],
    'finite': bool(torch.isfinite(x).all() and torch.isfinite(singular).all())}
print(json.dumps(result), flush=True)
```

Record before/after compiler, installed-patch and native binary hashes with the actual stdout/stderr. No invocation has been made from this command yet. Frontend may first use this same source with its existing capture observation to verify the paired strict/alias result boundary; actual numerical execution remains compiler-supervised after executor's mapper build release.
