# Prepared wire storage identity

2026-11-13. Composition request requirement 7: preserve typed view, alias and storage facts through their actual
compiler consumers. Review and correction are confined to the Plan wire-value handoff, not a new alias analysis.

## Finding and observation

At caller checkpoint `2b2e0625`, `bridge.Published` called `torch.frombuffer` separately for each transfer.
`schedule.planned` already deduplicates transfers naming the same source writer into the same Plan value, and each
value has solved copies. Separate wrappers around one copy's address did not retain that existing identity.

A direct Torch 2.14.0 API observation over one ctypes buffer found:

| Construction | Equal data pointers | Same StorageImpl / `_is_alias_of` | Alias after FakeTensorMode conversion |
|---|---|---|---|
| Two independent `frombuffer` wrappers | yes | no | no |
| Logical and distinct views from one root | yes | yes | yes |

Both fake views retained storage offset 2. A uint8 root reinterpreted as float32 and int32 retained the same storage
and 64-byte extent. This was a compact existing API observation, not a new evaluator or live bridge run.

## Paired handoff

The same Torch Claude session `9740398d-f513-4c78-8578-f80b79fa2225` (reported `claude-opus-5-5`, harness 2.1.321)
implemented the Published correction. Raw turns 8 and 9 and `prepared-alias-4.md` remain under metal-microbench
`output_data/compiler-workgroup-20261113/frontends/torch`.

- Published owns `roots[(value, copy)]`, one byte tensor per declared local wire value and solved copy. All transfer
  typed views, logical shapes and distinct write shapes derive from those roots with their original byte offsets
  and strides. No address comparison is used to discover aliases.
- The cache input is exactly the unique non-ABSENT value IDs in Plan.declared. Review corrected the initial eager
  cache of every Plan value: mesh_view_values returns UINT64_MAX offsets for run=false numerical values, which are
  absent storage rather than additional buffers to wrap.
- The executor's paired driver projection now carries transfer, Plan value ID, copy count, byte extent and original
  sent/landed views. Its AOT example construction uses one byte backing per Plan value and derives the corresponding
  typed views. Independent per-site empty allocations no longer define the example alias relation.
- `Target.bind` directly consumes Published.views/distinct; its Bound callable retains Published. Model.prepare and
  native Binding retain at::Tensor handles and choose the solved copy; they do not reconstruct or clone storage.
  Root/value identities are compile/preparation data, not a per-invocation alias solve.

Supervisor source inspection followed bridge.py, driver.py, lower_inductor.py, mesh_aoti.py and mesh-aoti.cpp.
`py_compile` and `git diff --check` pass for the corrected bridge source. The AOT consumer is an executor-owned
working source checkpoint at this review; actual effect compilation/execution evidence is recorded by that owner.
No live bridge or end-to-end throughput result is claimed here.

## Distinct remaining obligation

Different Plan values may share physical pages because their lifetimes permit reuse. Section.first is a virtual
row assigned before placement; it is not a complete physical alias witness. This patch deliberately does not
infer cross-value aliases from pointer equality. Their existing solved storage/dependency relation must survive
native optimization separately. Current lower_inductor consumes typed data and effect operands but does not
directly consume module.meta['dependences'] or the storage-reuse causes; preserving them through those typed
operands remains to establish. Same-value root caching alone does not close that obligation.
