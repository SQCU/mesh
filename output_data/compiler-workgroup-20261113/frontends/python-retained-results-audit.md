# Python compiler: retained-result audit

2026-11-13. Read-only source review of the current export → placement → devices → schedule → target → archive
chain, at `01163a03` plus the explicitly identified fanout WIP. This inventories concrete lost operands and their
consumers. It does not classify every metadata deletion as a defect or claim target execution coverage.

## Active boundaries

| Producer result | Discard / misconstrual | Actual downstream behavior | Required connection and displaced mechanism |
|---|---|---|---|
| `torch.export` input specifications distinguish parameters, buffers and literal constant tensors | `export.read` stores all three as `Exported.state[name]=(target,tensor)` without retaining their kinds or an immutable-definition contract | `driver.lowered` treats every state tensor absent `BUFFER_MUTATION` as a constant; `lower_inductor.body` replaces those placeholders with model buffers/get_attr and removes them from the actual argument list | Retain immutable authority separately from mutable invocation state. Embed only explicitly immutable definitions; keep parameter/buffer operands live unless that authority establishes otherwise. Absence of a captured internal write does not freeze external state. Delete blanket no-BUFFER_MUTATION→constant classification. |
| Canonical source conflict relation **and causes** from `mesh_graph.dependences_views` | `schedule.dependence` formerly kept only result[0] edges | Fanout lowering attempted to guess versions from all numerical predecessors and alias chains | Active same-worker correction retains RAW writer→read correspondence from the same call and remaps actual Nodes in `assemble`; the copy transform consumes it. Delete guessed-version inference. |
| Native demand's actual supplied/completion words and count/copy ownership | Previously only declaration/value/offset data crossed into Python target lowering | Each SEND independently emitted SENT→copy→PUBLISH, exposing duplicate rewrites and losing independent overlapping readers | Producer checkpoint `01163a03` retains only applicable actual action representatives. Active body correction consumes them with source versions and new-copy hazards. Original logical SEND endpoints map to actual emitted sites for the signature. |

The state-input issue is an active numerical binding loss: `Program` can prepare state arguments, but a placeholder
removed by `body` cannot consume their later values. Parameter gradients do not constitute `BUFFER_MUTATION`, so
the current classifier also embeds training weights. Whether a particular package happens to share some storage
does not supply the missing source-level mutability contract. Runtime update evidence should use the existing
stateful/training caller after the connected correction, not another evaluator.

That connection is now implemented in compiler-supervised checkpoint `e107c5e6`: original `InputKind` survives as
`Exported.state_kinds`, literal constants alone are embedded, live state binds at invocation, source buffers receive
mutation outputs, and the archive preserves the same roles. The same held placement worker implemented it;
compiler review reported exact CPU and restored comparisons for parameter updates, persistent/nonpersistent buffer
updates and visible buffer mutation. The callable old CoreML path's separate classifier remains below.

## Remaining callable target paths

These paths are callable but are currently outside `driver.TARGETS`, which selects only the completed CPU target.
That distinction matters: they are retained alternate mechanisms to displace, not proof that the active CPU path
still invokes them.

| Producer result | Discard / reconstruction | Consumer and replacement obligation |
|---|---|---|
| `lower_metal` View/Fused/Product source operand structure | `Lowering.reads` calls expression/read emission with a dummy Bound/index to rediscover access operands; `fit` and `settle` consume the resulting probe | Retain exact typed access operands when constructing the numerical form. Resource fitting/materialization and terminal emission consume that result; delete emitter probing. |
| `schedule` supplies copied Node dependencies, task/resource correspondence, solved Plan values and offsets | `lower_metal.Lowering` accepts only module/laid/stored; `ordered` reconstructs conflicts from emitted step accesses, and `function` creates separate storage/binding arrangements | Carry original-node→lowered-step correspondence and original solved relations/placement into the target. Analyze only genuinely introduced target hazards, combine through the canonical owner, and bind actual planned storage. Delete unconditional original-relation reconstruction and independent placement where the plan already owns it. |
| Placement/device transformation retains exact transfer endpoints and layouts | `lower_metal.targeted` scans all SEND/RECV nodes again to recreate received/laid maps | The target receives the retained transfer/layout operands directly; remove discovery traversal. |
| Explicit target assignment should produce a completed target artifact | `lower_metal.targeted` and `lower_coreml.lower` return residual FX modules for nonselected processors; CoreML `function` reconstructs a per-segment slots/entries/sends/receives runtime recipe | Finish each selected target from the common typed body/effects/placement contract, then use the same completed-result caller/archive path. Delete partial-return/alternate recipe mechanisms when their actual consumer is replaced. No new eager fallback or smaller operator domain. |
| Completed body, source task/resource correspondence and native timing observations | `examples/tp.py::taken` instead executes grouped FX through `Timed`, an interpreter; external SEND/RECV reaches `ir.LOCAL.channels`, whose queue owner was removed | Connect the existing calibration caller to completed target timing with retained cost-unit provenance, and update its actual cost consumer if fusion changes cost units. Remove dense queue kernels only with that connection; do not delete measurement coverage. |

These findings identify ownership, not a license to renovate the old target bodies one function at a time. A
target replacement must retain the numerical body and its solved operands through the actual final consumer and
remove the corresponding alternate entry points as one connected unit.

For calibration, the exact current result is `times[original_processor][shares.kind(node)][shares.units(node)[0]]`.
`shares.computation` consumes those per-node size samples as `n / speed(samples, x)` and sums them by stage. Replacing
the entire table with one group elapsed time would mislabel the measurement. A paired replacement must either
measure those compiled numerical cost units through the selected target or retain actual fused-unit membership
and revise stage costing to count each measured unit once. `Timed`'s interpreter and orphan queue kernels cannot
remain an alternative execution mechanism. `export.Captured` is a different, intentional source-capture owner.

## Eliminations inspected that are not current loss findings

- `placement` removes `blocks`/`collectives` annotations **after** transferring their geometry/collective identity
  to the actual SEND/RECV endpoints. The retained endpoint map is the scheduling consumer's operand.
- `devices.on_devices` explicitly remaps those endpoints through the deepcopy memo. Recomputed views cover the
  newly inserted device-copy operands; that is a changed representation, not a discarded old schedule.
- `schedule.assemble` contracts internal same-target channels and retains copied dependency/task/source maps.
  A logical channel need not remain an executable queue after it becomes an ordinary value edge.
- `lower_inductor` uses upstream `make_fx` decomposition and forward compilation on the composed typed body.
  That is an actual lowering stage. The issue is whether required effects/aliases survive into it, not that a
  fresh target IR exists. The installed mixed-width mutation correction and fanout review check those handoffs.
- `archive.load` restores completed packages, state/result recipes and native solved records without original FX
  capture or a scheduling solve. `Exported.graph_module=None` in that final executable archive is intentional
  after its compiler consumers are finished; it is not by itself a defective IR handoff.

`export.read` also rejects successfully captured programs through generic higher-order/get_attr/input/output-kind
guards. The guard's own text does not establish the requirement boundary, but the actual canon does: Torch source
lowering item 2 and compilation composition's “Withdrawn, 2026-11-09” exclude arbitrary data-dependent control
between linear algebra, including dynamic `cond`/`while`/`map`/`scan`. A previous paragraph incorrectly inferred
that fixed tensor shapes admitted those regions; that claim is withdrawn. No dynamic-region implementation is
assigned from this audit.

The blanket HigherOrderOperator check still needs inspection for statically composable wrappers inside the stated
domain. Constant tensor attributes and captured `USER_INPUT_MUTATION` also retain exact upstream value/target
facts that current guards can discard rather than hand to the compiler or caller writeback. Removing guards and
failing later would not close an admitted case: its static body, constant, mutation and structured-result
consumers must be connected. Admission comes from the cited scope, not generic torch.export success.

## Existing typed numerical owner for the target replacement

`rdma/mesh_compile.py::_lowering(local, page)` already constructs `_Generic`, `_Sort`, `_Check` and `_Lapack`
operations from explicit arguments and output views. `_Builder.generic` emits those operations into the common
`metal_ir_program`; its `value` retains storage identity, shape, stride and storage offset. This is the existing
numerical owner to reuse, not a reason to implement a second Torch-to-Metal mathematical translation.

The narrow missing input boundary is an assigned scheduled FX node plus its actual logical input/output views,
original metadata, target, source operation identity and explicit input/state/constant roles. `Local.annotation`
carries attention origin; determinant and pseudoinverse lowering also consume original input metadata for
contiguity and source-device precision. Those facts cannot be replaced with the device of a meta stand-in.
`_Builder.root` currently treats an unowned real tensor as constant, so the new boundary must supply retained
source roles rather than revive the freezing defect fixed in `e107c5e6`.

The result needed by the native consumer is the retained Program, input/output/state root ValueIDs, and original
node-to-operation correspondence. Existing `metal_ir_value_receives`/`value_sends` declarations and subsequent
range/release/sending preparation bind actual Plan values before finalize. The completed chain is
`metal_ir_finalize`/archive → `MetalRecorderPrepare`/Replay → `MetalRecorderProgram`/Save/Load → Run. Neither
`_Export` recapture nor `_programs`/`_bound` placement orchestration belongs in the new assigned-body consumer.
Numerical source roots and the fanout correction's versioned wire roots remain distinct operands.

Installed Torch also supplies `torch/_inductor/codegen/cpp_wrapper_mps.py` and `mps.py`; the wrapper calls the
actual MPS AOT APIs and shares CPU call construction where appropriate. That is a real target implementation to
inspect, but choosing it alone would not discharge the existing ANE or mixed native-effect composition
obligations. The next replacement must connect the selected target to the completed-result contract and delete
the old probing/reconstruction/residual-return bodies, not merely expose an unused construction helper.
# Graph ownership correction, 2026-11-13

The current thin callers mostly invoke `mesh_graph`, but `rdma/mesh_graph.py:11–211` independently implements
directed walk/reach, chain, closure/reduction, precedence, propagation/forward, postorder/longest and matching/width.
This is only partial centralization. The existing native owner is `rdma/mesh-graph.h` and `.c`; the active SAME Torch
session 9740398d is replacing those Python algorithms with one retained native graph/result and object/edge
correspondence. The same source graph must serve ordering, closure and contraction without repeated serialization
or freeing a reduced copied answer. Native arrays and annotations remain alive through their consumers.

The C `mesh_graph_reach` is undirected, whereas Python reach follows the supplied directed adjacency. A name match
does not establish equivalence. The shared native DFS/order owner must expose the required directed result; no
second DFS is acceptable. Existing live callers include schedule's causal order/closure/transfer precedence,
placement's reach/propagation, ir.views' alias propagation, fanout's constructed effect DAG, and shares' longest
stage path. Domain transfer callbacks and typed correspondence remain at their producers.

The source-cause connection is separately open: retained native conflict witnesses now survive in their owner,
but typed construction must consume exact view/cause correspondence before the operation-edge/fusion boundary.
Root-wide Cartesian operation barriers and an `unresolved` tag with no consumer were rejected in review. This is
the existing composition CR requirement at lines 312–314, not an optional new annotation.
