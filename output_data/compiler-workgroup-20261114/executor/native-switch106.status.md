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

## Native118 review and active continuation

Native118 supplied only a `Source` skip; `Wrap`, `TargetKey` and `Pipeline` still consumed the module. Its claim that
pipeline emission was deferred was false. The draft was not accepted or checkpointed as complete; private syntax
checks did not establish the missing phase contract.

Native119–121 continue the actual phase split: lowered capture and placement -> retained replay plan -> final emission
-> replay consuming that same plan. SAME047 pairs the canonical recorder binding and `Target.placed` order. Typed
selector stores remain in their original kernel; original span rows supply their literals. Plan-specific compiled
output ownership, full range workspace geometry, plan teardown and saved geometry reuse remain source-review items.
No conditional workload has run. Joined completion, later compaction control, CPU indexed control and MIL remain open.

## native119–122: plan before emission, retained plan operand (source; strict checks only)
- **Chronology (every entry caller).** `MetalRecorderPrepare` (lowered or loaded program) → placements,
  `MetalRecorderState`, `metal_ir_place` (computes the replay order) → `int MetalRecorderPlan(MetalRecorder *, id
  prepared, char *error, size_t room)` (ctypes `[c_void_p, c_void_p, c_char_p, c_size_t] -> c_int`, 0 or EINVAL) →
  `metal_ir_finalize`, or for a loaded program `metal_ir_loaded` → `MetalRecorderReplay`.
  - `metal_ir_loaded` only relocates later-pass readiness and transport status and checks transfer records and
    CONSTANT data, so it takes finalize's slot.
  - `tools/native_bench.m` `Realize` and `tools/e2b_replay.m` phase setup now finalize (or call loaded) after Plan and
    before Replay. The earlier finalize/loaded calls are removed.
- **One plan operand.** A fresh `-plan:` alone runs `MetalReplayPartition`, which owns handovers, levels (barriers and
  branch edges), blits and the order/later cuts. It gives `metal_ir_replay_plan(p, {span, copies, steps, count,
  records})` to the program.
  - A loaded or planned program (`metal_ir_replay_planned`) reuses the saved span, copies and logical rows. It does not
    query order cuts or compute handovers, levels or cuts.
  - Both paths project the retained replay order once to get the later passes' cut and order (`replayLatersKey`).
  - `MetalReplaySegments` relocates the rows onto live ICBs.
  - Displaced state is removed: `replayHandoverKey` (native's end-of-call producer barrier now reads the plan's ALIAS
    step at `end % count`), `replayLevelsKey` (the barrier count reads PASS/ALIAS steps) and the unconsumed
    `replayCutsKey`.
  - Archive v40 carries span, copies, records and steps after each dispatch's switch and branch, outside the identity.
    The v40 comment no longer says "no stream changed".
- **Stores.** `SwitchBodies` runs before `Emit`, once into each call specialization's empty clone. Per run j whose head
  the switch holds, u32 word 2j+1 of the residue's row gets `select(0, end − first, selection == arm)`.
  - With no plan, finalize returns EINVAL.
  - `Invariant` is false with selections (no adoption or publication), and `CloneHandles` remaps the selection's
    handles.
- **Count proof.** Segments hold whole copies, so a switched run is one piece of end − first commands in every residue.
  The initial table contents are each residue's actual span row.
- **Workspace (partial).** The `MetalRangeTable` descriptor and contents live 8-aligned after the argument records in
  the replay's ordinary argument storage, resident and held. `rangeWord` rows sit at offset + e·stride, and
  `NativeICB.ranges` points at the run's record, selecting `NativeICBRangeEncode`.
  - Not complete: the selector's lowering-time effect view is still the 4-byte selector value, not the table row.
- **Ownership.** `capture->planned` is released only through `MetalReplayDestroy` (plan failure, replacement, capture
  dealloc). `-replay:` moves it, so there is no capture cycle.
- **Checks.** Strict `-Werror` rc=0 on the eight IR sources, capture, program and both tools; `git diff --check` clean.
  `metal_recording.m` unchanged. Nothing built, imported or run.
- **Open.**
  - The table's lowering-time effect and bytes are the next required connection. Indirect-range fetch ordering under
    the branch-edge barrier is unverified.
  - BLIT, later INDIRECT and host steps inside a branch are not gated.
  - Nested inactive safety beyond ICB runs, joined completion, CPU and MIL.

## native123: range workspace accesses through the existing projection (source; strict checks only)
- **Stage.** Prepare → placements, `MetalRecorderState` → `MetalRecorderPlan` → `metal_ir_finalize`/`metal_ir_loaded`
  → `MetalRecorderReplay` → `metal_ir_place` → native program. Both tools follow this order; SAME047 pairs
  `Target.placed`.
  - `int metal_ir_replay_planned(p, struct metal_ir_replay_plan *, char *error, size_t room)` makes the replay order
    once and returns the retained plan. `ReplayOrder` left `metal_ir_place` and finalize.
  - `-replay:` calls the new `void metal_ir_ranges_placed(p, buffer, offsets, copies)`: argument storage, borrowed, row
    e at `table->offset + e·stride`.
- **Retained correspondence.** `metal_ir_replay_plan` (fresh plans only) produces `ReplayPlan::stores` {switch, step,
  arm} once with `SwitchArm`'s climb. It is archived with the plan. `SwitchBodies` literals and the physical writes
  both consume it; `SwitchRecords` is removed.
- **Projection.** In `metal_ir_place`, each store's selector dispatch (`switches[s].select`) writes 4 bytes at row +
  8·range + 4, and each ranged step's head dispatch reads 8 bytes at row + 8·range, in that dispatch's part. Both are
  keyed to the real buffer, like every placed access. Original EffectViews and the source relation are unchanged.
  Residency (held argument storage) and the plan's branch-edge barriers are the existing mechanisms.
- **Deleted.**
  - The capture's effect copy (`capture->effects`, `MetalRecordedDispatch.effect`/`effects`, slot bookkeeping, the
    copy check), after proving nothing reads it.
  - `Ranges()`, `Program::ranges`, its specialize copy and its v40 archive slot. No version bump.
- **Checks.** Strict `-Werror` rc=0 on the eight IR sources, capture, program and both tools; `git diff --check` clean.
  `metal_recording.m` unchanged. Nothing built, imported or run.
- **Open.**
  - BLIT, later INDIRECT (zeros and compaction) and host steps inside a branch are not gated.
  - Nested inactive safety beyond ICB runs.
  - A switch with no ranged ICB run has no table placement for its selector binding.
  - Joined completion, CPU and MIL. Full conditional execution is not claimed.

## native124: regions' search retained through archive load (source; strict checks only)
- **Graph side.** New `int mesh_graph_search_save(const struct mesh_graph_search *, void **bytes, size_t *length)` and
  `int mesh_graph_search_load(const void *bytes, size_t length, struct mesh_graph_search **made)` in
  `rdma/mesh-graph.h`/`.c`.
  - Bytes: {vertices, count, depth-first, 0}, then preorder and postorder [count], parent, place and end [vertices]
    (u32 each), entry [vertices] (u64).
  - Load copies into its own search (`mesh_graph_search_free`). EINVAL on a wrong length, count or out-of-range index;
    ENOMEM on allocation failure.
- **Archive.** `Records` writes `regionSearch` once per search through a stream table, as the source owners are named,
  so rows that share it still share it after load. Then each program's `repeatEnclosing`/`repeatOutermost`.
  Everything is outside the identity.
  - Load no longer calls `RegionForest`; construct is its only caller.
  - v40 comment updated; no version bump.
- **Checks.** Strict `-Werror` rc=0 on `mesh-graph.c` (rdma Makefile flags), the eight IR sources, capture, program
  and both tools; `git diff --check` clean in both repositories. Nothing built, imported or run.
