# Retained upstream numerical body

This is producer evidence for the already-recorded `Normalize` caller in `layernorm-native-command.md`. Its actual inputs remain CPU tensors; the observed `OuterLoopFusedSchedulerNode` is the CPU upstream scheduling result. The original Mesh caller requested MPS, whose selected-target schedule is not measured here. No Mesh module/native ABI was imported, no target kernel code was emitted, and no numerical workload or correctness comparison ran.

The final process exited 0 with empty stderr. `numerical61-upstream-body-3.json` retains four `ComputedBuffer` records under one `OuterLoopFusedSchedulerNode`, their mixed reduction/pointwise domains, all origins and the noncontiguous weight access `i1 + 3*i2`. The final pointwise buffer has eight source origins; mapping it solely by `get_origin_node()` would discard correspondence. `GraphLowering.operations`, `.buffers`, `.graph_inputs_original`, `.graph_input_storage_offsets` and `.scheduler.nodes` are available before target source emission; Python wrapper lines are not required to obtain these facts.

The first attempt omitted Inductor's selected decomposition table and failed at `aten.native_layer_norm`; the second omitted its normal `DebugContext` and failed in scheduler debug reporting. Their stderr files are retained as observation setup failures. The corrected command uses `select_decomp_table` as `compile_fx.py` does and `DebugContext`; no compiler source workaround was applied.

A separate source review of the actual output body finds `index_expr(12, torch.float32)`. `DtypePropagationOpsHandler.index_expr` (247–258), `MetalOverrides.index_expr` (228) and `CppOverrides.index_expr` (1036) preserve this requested non-integer dtype. The current draft handler's unconditional integer result is not accepted.

The scalar If native constructor separately passed strict syntax with unchanged source hashes (`numerical60-native-check.json`); this does not demonstrate a connected public compiler result or displace `_instruction`.

## Command

`TORCH_DEVICE_BACKEND_AUTOLOAD=0 /Users/mdot/.venv-mesh-uv/bin/python /tmp/mesh-compiler-read/upstream_layernorm.py`

```python
import json
import torch
from torch._guards import detect_fake_mode
from torch._inductor.graph import GraphLowering
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.debug import DebugContext
from torch._inductor.virtualized import V

class Normalize(torch.nn.Module):
    def forward(self, x, weight, bias):
        return torch.nn.functional.layer_norm(x, (3, 4), weight, bias, 1e-5)

x = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) - 7) / 5
weight = (torch.arange(12, dtype=torch.float32).reshape(4, 3) + 1).T / 7
bias = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 11
exported = torch.export.export(Normalize(), (x, weight, bias), strict=True).run_decompositions(select_decomp_table())
module = exported.graph_module
inputs = [n.meta['val'] for n in module.graph.nodes if n.op == 'placeholder']
fake = detect_fake_mode(inputs)
with DebugContext(), V.set_fake_mode(fake):
    graph = GraphLowering(module, example_inputs=inputs, shape_env=fake.shape_env, is_inference=True)
    with V.set_graph_handler(graph), V.set_extern_kernel_nodes([]):
        graph.run(*inputs)
        graph.init_wrapper_code()
        graph._update_scheduler()
        record = {
            'caller': 'Normalize from compiler/layernorm-native-command.md',
            'scope': 'upstream CPU-device metadata only; Mesh requested MPS placement is not lowered or executed',
            'input_shapes': [list(t.shape) for t in (x, weight, bias)],
            'input_strides': [list(t.stride()) for t in (x, weight, bias)],
            'input_offsets': {k: str(v) for k, v in graph.graph_input_storage_offsets.items()},
            'inputs': list(graph.graph_input_names),
            'buffers': [{'name': b.get_name(), 'type': type(b).__name__, 'layout': str(b.get_layout()),
                         'origins': [n.name for n in b.get_origins()]} for b in graph.buffers],
            'operations': [{'name': o.get_name(), 'type': type(o).__name__,
                            'origins': [n.name for n in o.get_origins()]} for o in graph.operations],
            'scheduler': [{'type': type(n).__name__, 'name': n.get_name(),
                           'members': [{'name': m.get_name(), 'type': type(m).__name__,
                                        'group': str(getattr(m, 'group', None)),
                                        'read_writes': str(m.read_writes),
                                        'origins': [o.name for o in m.node.get_origins()]}
                                       for m in n.get_nodes()]} for n in graph.scheduler.nodes],
            'mutation_real_name': dict(graph.scheduler.mutation_real_name),
            'outputs': [str(o) for o in graph.graph_outputs],
            'torch': torch.__version__,
        }
print(json.dumps(record, indent=2))
```

## Installed source hashes

- `_inductor/graph.py`: `efffd6b54ed0e67ad1f41c840957780fae092708adf460ecc16914dcaaba27fe`
- `_inductor/scheduler.py`: `8e3ca0c3717a4f8973c7bfe41cab52cd0dd7154a5401fd8d60e83b935893178c`
- `_inductor/decomposition.py`: `d0a64356b96ea1c9fa232a23b7bc2720c4b07895ba6977c0297df00415dcd9f6`
- `_inductor/dtype_propagation.py`: `850c45101fcf1b59814e7204547b509ec0237629e9b4639fdd8f206d36662477`
- `_inductor/codegen/mps.py`: `d9734c7abceef021a73ab5ba02862d83907b142fac0f0aca3bb902cabfa4f774`
- `_inductor/codegen/cpp.py`: `885b73279e8686181d3706c6d6a590d07102c44ea739e8c11bb5be5ef758292f`
