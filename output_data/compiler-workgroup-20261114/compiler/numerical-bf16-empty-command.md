# Existing public numerical caller observations

Environment and public invocation are those in pinv-native-command.md. Interpreter: /Users/mdot/.venv-mesh-uv/bin/python -u; working directory /Users/mdot/dox/mesh. Before imports and after results, SHA256 was recorded for operations/decompose/driver/schedule/native/metal/export/ir/runtime/archive and the two native libraries; exact maps are matrix-exp-bf16-source.json and eigh-empty-source.json.

BF16 uses matrix-exp-repeat-command.md unchanged except `a = torch.tensor([[[0., .25], [-.125, .5]], [[0., 16.], [-12., 4.]]], dtype=torch.bfloat16)`. The returned tensor's dtype is recorded before converting it to double for comparison. stdout/stderr/result: matrix-exp-bf16.* and matrix-exp-bf16-result.json. Actual temporary command was /tmp/mesh-matrix-exp-bf16-observation.py.

Empty source and inputs, through that same public compile/prepare/invoke path:
```python
class Empty(torch.nn.Module):
    def forward(self, zero, batch):
        a = torch.linalg.eigh(zero)
        b = torch.linalg.eigh(batch)
        return a.eigenvalues, a.eigenvectors, b.eigenvalues, b.eigenvectors
zero = torch.empty((0, 0), dtype=torch.float32)
batch = torch.empty((0, 2, 2), dtype=torch.float32)
scheduled = compile_module(Empty(), (zero, batch), {'p': (0, torch.device('mps'))}, lambda node: 'p')
program = Program(scheduled)
returned = program(zero, batch)
expected = (*torch.linalg.eigh(zero), *torch.linalg.eigh(batch))
```
The result records each shape, dtype and numel and each reference shape; stdout/stderr/result: eigh-empty.* and eigh-empty-result.json. Actual temporary command was /tmp/mesh-eigh-empty-observation.py. No retained evaluator was added.
