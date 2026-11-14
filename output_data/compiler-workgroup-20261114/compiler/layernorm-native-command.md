# Noncontiguous affine LayerNorm public caller

One ordinary `compile_module` → `Program` observation on requested MPS, using the existing public numerical observation path. The weight shape is [3,4], stride [1,3]; no contiguous conversion or placement fallback. CPU double is the numerical reference.

```python
import json
import torch
from mesh_compiler.driver import compile_module
from mesh_compiler.runtime import Program

class Normalize(torch.nn.Module):
    def forward(self, x, weight, bias):
        return torch.nn.functional.layer_norm(x, (3, 4), weight, bias, 1e-5)

x = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) - 7) / 5
weight = (torch.arange(12, dtype=torch.float32).reshape(4, 3) + 1).T / 7
bias = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 11
print(json.dumps({'input_shapes': [list(v.shape) for v in (x, weight, bias)],
                  'input_strides': [list(v.stride()) for v in (x, weight, bias)],
                  'weight_contiguous': weight.is_contiguous()}), flush=True)
print('compile', flush=True)
scheduled = compile_module(Normalize(), (x, weight, bias), {'p': (0, torch.device('mps'))}, lambda node: 'p')
print('prepare', flush=True)
program = Program(scheduled)
print('invoke', flush=True)
got = program(x, weight, bias).cpu().double()
expected = torch.nn.functional.layer_norm(x.double(), (3, 4), weight.double(), bias.double(), 1e-5)
print(json.dumps({'relative_error': float(torch.linalg.norm(got-expected)/torch.linalg.norm(expected)),
                  'max_absolute_error': float((got-expected).abs().max()),
                  'finite': bool(torch.isfinite(got).all()), 'shape': list(got.shape),
                  'values': got.tolist()}), flush=True)
```

The source and library hashes plus canonical environment are in `layernorm-native-1-source.json`; stdout records native archive restoration without a relation solve.
