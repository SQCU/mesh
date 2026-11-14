# Bounded sort and actual emitted-operation membership

The existing `metal_ir_sort` expansion accepts an optional runtime extent. Padding, bitonic compare groups and final writes consume the existing symbol/loop-extent operands; the static callers pass `UINT32_MAX`. The existing Torch structured `_Sort` and two LiteRT callers use that same producer. The sort returns its actual first operation, including extent preparation.

`lower_metal.construct` retains the entire operation-ID interval created by each structured record in its existing task/source/write/store/generator correspondence. A sort's preparation and compare operations therefore reach the same source-relation and cost consumers as its final copy. This changes membership, not graph edge policy.

Validation: the coordinated v35 native build and provider link passed with the new extent ABI (`joint-v35-native-retry.log`, `joint-v35-abi.log`). The subsequent first-operation return and full-membership correction passed private strict current-source syntax (`sort-source-syntax.log`) and Python AST/pyflakes (SAME64 `numerical-13.jsonl`). Staged Python sources were parsed independently. Existing CPU/archive observations are recorded by executor436b9e2d; they exercise the shared current compiler, not bounded-sort numerical accuracy.

No eigensolver completion claim: source-outlined QDWH-eigh still lacks Algorithm 3's genuine subspace-iteration continuation and stagnation remedies. Those partial formulas and outline changes are excluded from this checkpoint. No native eigensolver or bounded-sort output comparison was run here. Original malformed reference paths remain untouched.
