# Matrix exponential switch producer

SAME64 numerical51–52 retains the cited scalar Padé and overflow control as canonical Torch switch graphs. Batched inputs reuse the same polynomial body and shared matrix powers under JAX's batched-index selection rule. The fixed squaring scan, result cast and result-to-Check dependency remain intact. Missing theta tables raise at the required lookup; no separate dtype admission rule remains.

The typed record is `_Switch(index, branches, outputs)`, with ordered `(records, yields)` branches and explicit common results. Indices are I32; logical-not-less-than preserves equality and NaN degree selection. The source modules and original node identities survive expansion, outlining, device deepcopy and machine assembly (a046e92b). Branches remain compilation operands, not outer runtime calls.

Independent source observation used the existing scalar error caller input `[[0, 1e6], [0, 0]]` with `make_fx(..., tracing_mode='fake')`; it performed no numerical invocation. Exit 0 and unchanged source hashes:

- Padé switch: I32 index, three branch graphs with 8/13/18 call-function nodes; both F32 results have shape (2,2), stride (2,1).
- Overflow switch: I32 index, two branch graphs with 6/2 call-function nodes; both yield F32 shape (2,2), stride (1,2).
- AST passes for operations/decompose/devices. Pyflakes reports only the existing unused `real` and `two` locals. Diff whitespace checks pass.

Native registration and encoding are being paired by executor. Device-selected command ranges, selector ordering, inactive compaction counts, common-result writes, completion and the CPU/MIL consumers are required before a native result can be claimed. No speedup or complete scalar-control claim is made by this source checkpoint.

Primary sources: [JAX expm and Padé](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py), [JAX conditional batching](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/conditionals.py), installed Torch `torch/_higher_order_ops/switch.py` and `torch/fx/graph_module.py`.
