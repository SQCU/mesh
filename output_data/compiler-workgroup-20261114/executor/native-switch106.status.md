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
The segment ownership cycle is corrected by native117 below.

## native117: replay segment ownership cycle closed
- **The cycle:** `result.commands = [segments[0] retain]`, and `@[segments, size, repetitions]` is associated on it with
  RETAIN, so `commands` ⇄ the array. `MetalReplayDestroy` released `commands`, but the array kept it alive, and with it
  every association.
- **Borrowers accounted for:**
  - The program borrows each replay by `MetalCall.replay` pointer, with no struct copies. Python `Owned` releases the
    program before the replays' `MetalReplayDestroy`, in reverse registration order (native.py ≈514).
  - NativePrepared holds the plan and the segment array in `p->held`.
  - The other callers are the replay's own error path and the tools' destroys.
- **Correction (`metal_capture.m` `MetalReplayDestroy`):** only the `replaySegmentsKey` association is cleared, then
  `commands` is released. The array lives while a borrower holds it, and `commands` deallocates at its last owner's
  release, freeing its other associations then. No metadata is dropped early, and nothing is copied.
- **Deleted duplicate handoff:** the `dispatchOf` parameter of `MetalNativeCommands`, `MetalNativeGPU`'s fetch of the
  same map, and `MetalRepeatEmit.dispatchOf`. `MetalNativeCommands` uses the plan's map. Diff: +10/−20.
- **Checks:** strict rc=0 on capture, program and both tools; `git diff --check` clean.
