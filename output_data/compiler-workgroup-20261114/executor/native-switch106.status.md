# Native conditional source — prepared span handoff retained

Source checkpoint: metal `622c3bc6`; selector encoder: mesh `d3a3ac6f`.
SAME e48, Claude Opus 5.5, native113–116. Conditional execution remains incomplete; no build or workload run.

Properties P: fixed AOT shapes/call topology, supplied placement, retained region/order/copy geometry, runtime numerical indices.
Composition 3/4/6/7 supplies the retained-result and partial-evaluation contract. Existing Metal APIs consume the prepared spans.

- `metal_ir_order_cuts` borrows the existing `PrepareOrder` result. Replay projects those cuts through actual emitted dispatch identities to command positions once.
- `ReplayOrder` permutes only inside those cuts. They cover part, host and region boundaries, so ordinary GPU and repeat-run consumers select whole plan steps.
- `MetalReplaySteps` exposes steps, existing `NativeICBSpan` rows, per-piece first commands and the command-to-dispatch map.
- Native emission borrows those rows; the old span derivation, identical span copy and dispatch-map reconstruction are removed.
- `p->held` retains both the plan arrays and original `replaySegmentsKey` owner of the ICB objects.
- Repeat child identity is the first encountered child's test/member; the redundant child scan and ancestor walk are removed.
- `RegionForest` uses `mesh_graph_rows/search/climb`; nearest/outermost repeat results are retained, and construction errors reach the existing error owner before queries.
- Mixed source children/postorder, selector/yield/common-result correspondence and archive remapping remain retained. Ranged observation units remain present.

Checks: worker strict `-Werror` syntax on all eight IR sources, capture/program and two existing tools passed; final owner change passed the affected recorder checks. Supervisor source/diff review passed. The intentionally malformed `metal_recording.m` is unchanged.

Remaining: selector literal stores/range allocation from this plan, joined completion, CPU indexed control and the MIL tree. No full conditional execution claim.
Next bounded lifetime correction: `commands == segments[0]` and its retained segment association owns that same object; inspect all borrowers/destruction before closing this existing cycle.
