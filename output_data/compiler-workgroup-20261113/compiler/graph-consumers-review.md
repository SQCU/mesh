# Retained graph consumers: supervisor review

Source checkpoint: mesh `11e39c11`. The joint checkpoint connects three existing callers to the canonical graph owner; it does not add a second evaluator or numerical runtime.

| Existing caller | Retained input and result | Review boundary |
| --- | --- | --- |
| Public partition `_complete` | Original node order, atom vocabulary and projection operands feed `mesh_graph_sharding`; the IR keeps the native state and its original node/atom correspondence before `_stripe` inserts nodes. | The same two propagation levels, forward/reverse node order, operand order, compatibility and distinct-axis rules remain. Frontend owns the source review and actual partition/lower observation. |
| Public `lower` and `_place` lifetime consumers | One explicit use-event table feeds `mesh_graph_lifetimes`; the IR retains the native first/last arrays and original logical/backing keys. | Root reviewed earliest logical use, latest backing-owner use and existing wait events. This replaces the mismatched old `ends` callers without a new dependence solve. |
| `_analyse` | One follower relation and DFS supply borrowed chain slices; the existing predecessor closure supplies the native predicate to the existing `mesh_graph_chains` algorithm. `Calls.owner` retains the search, both closures, per-function selected members, counts and native chain results. | Source review and private compilation only. The public partition/lower observation does not invoke `_analyse`. |

The `_analyse` follower relation has at most one predecessor and successor per call and follows use-def order. Each root's DFS subtree is therefore the same disjoint chain returned by the former per-root BFS; isolated calls remain excluded. The width predicate reads the same closure bit at `(selected[b], selected[a])`, with the same unit counts and selected-call order. The empty-root result is also an owned view, so it retains the search. These are source arguments, not a machine-checked proof or runtime comparison.

`Calls` is an intermediate compiler result: reducing it to outstanding counts would discard the solved objects too early. The new owner is retained through that handoff. The existing saved final-program recipe still omits the analysis owner after compilation and restores without a new solve.

## Matching build

Canonical `make libmesh.dylib` exited 0. This batch adds functions and leaves the already-matched native consumer layouts unchanged. Fresh imports through `/Users/mdot/.venv-mesh-uv/bin/python`, with `TORCH_DEVICE_BACKEND_AUTOLOAD=0` and `PYTHONPATH=rdma`, checked actual sizes: Sharding 80, Projection 24, Searched 56, RelationData 160, Function 168 and Local 144. The built library exports `mesh_graph_sharding`, `mesh_graph_lifetimes` and `mesh_graph_closure_chains`.

Evidence:

- `compiler/graph-consumers-build.log`: canonical build output.
- `compiler/graph-consumers-abi.log`: fresh loaded-library size/symbol check, `PASS`.
- [Call-analysis work order](call-analysis-views.md), [retention correction](call-analysis-retain.md), and [worker source review](call-analysis-views.status.md).
- Root's `integration-use-lifetimes.status.md` covers the lifetime unit. Frontend's `output_data/compiler-workgroup-20261113/frontends/torch/finite-sharding.status.md` in the metal-microbench repository covers sharding.

Built `rdma/libmesh.dylib` SHA-256: `04bc9f61bd4ad28ffad69a03a513240cb3a0e92cf009fe2012bcd6dffdf6dc37`.

The previous CPU/MPS client and outer filesystem archive observations remain recorded in executor's `function-binding-review.md` (`c7a2f6ba`). This additive graph batch does not claim another target execution, live transport validation or throughput measurement.

## Caller observation

The frontend's existing `examples.engines.Engines` export → explicit `Placement`/`Nodes` partition → lower command exited 0. The actual unequal `Shard(0, (1, 3))` retained counts 8/24, with 23 source nodes and 44 dimension slots. Lowering produced 28 partition items, four declared transfers and 13 planned values. All 27 logical-value and 13 backing-owner lifetime bounds were valid, and the native solved results remained retained.

Exact evidence is `frontends/partition-lower-check.json` and `frontends/partition-lower-check.stderr` under this workgroup. The latter contains only the existing `treespec.children_specs` deprecation warning. This observation starts no services and executes no target program. It does not call `_analyse`; that separate local contract has the source/private-compile evidence above. No standalone evaluator was added.
