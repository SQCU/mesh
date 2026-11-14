# Matrix exponential count-controlled native case

One public compile_module→Program MPS call, environment in `pinv-native-command.md`. Python source hashes immediately before and after imports were equal; those imported owners were then retained in memory. Native v37 binaries were unchanged. No diagnostic hook or added output.

```python
import json, torch
from mesh_compiler.driver import compile_module
from mesh_compiler.runtime import Program
class Exponential(torch.nn.Module):
    def forward(self, a):
        return torch.linalg.matrix_exp(a)
a = torch.tensor([[[0., .25], [-.125, .5]], [[0., 16.], [-12., 4.]]])
scheduled = compile_module(Exponential(), (a,), {'p': (0, torch.device('mps'))}, lambda node: 'p')
program = Program(scheduled)
y = program(a).cpu().double()
reference = torch.linalg.matrix_exp(a.double())
print(json.dumps({
    'relative_errors': (torch.linalg.norm(y-reference,dim=(-2,-1))/torch.linalg.norm(reference,dim=(-2,-1))).tolist(),
    'finite': bool(torch.isfinite(y).all()),
    'source_counts': ((torch.linalg.matrix_norm(a.double(),ord=1)/3.925724783138660).log2().floor().clamp(min=0)).tolist(),
    'result': y.tolist(), 'reference': reference.tolist()}))
```

Actual stdout also labels import/compile/prepare/invoke/result. Source counts are computed from the cited formula, not a measurement of native iteration counts.
