# Native sending generation — status

Session e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, explicit claude-opus-5-5, 2026-11-13. Unit: native-generation-assignment.md.
Source checkpoint: metal-microbench 732f58c9 (supervisor commit). No shared build or live transport validation yet.

## Changed paths (for coordinated supervisor commit)

metal-microbench: `metal_program.h`, `metal_lower.mm`, `metal_recording.m`, and the corresponding SENT contract in `metal_recording.h`. No archive change (see "Archive path").

## Fields and producers

`struct NativeTransfer { void *const *words; uint64_t copies, plan, every, ordinal, ring, delay; uint32_t transfer; }`
(metal_program.h). `copies` is still the pointer table's residue count; the new fields are separate from it.

- `ring`: the transport word ring's actual copies. For SENT it is `mesh_metal_sending.sent.copies`; for ARRIVED it is the received
  range's `mesh_metal_range.input.copies`. PUBLISH has no ring (0): its object is the single supplied word that
  `mesh_host_publish` raises to e + 1.
- `delay`: the prior execution distance. For SENT it is `sent.copies`; for ARRIVED and PUBLISH it is 0.
- Single producer: MetalNativeHost (metal_recording.m) reads both from the canonical records MetalHostEntries already
  indexes: `metal_ir_value_sendings` / `metal_ir_value_ranges`, at the dispatch record's operand and ordinal.
  `sent.copies` is the plan's `mesh_state.copies` (rdma/mesh-metal.m `mesh_metal_sending_prepare`). That is the same copies value
  `mesh_host_sent` takes its slot (`execution % copies`) and threshold (`(execution / copies + 1) * sends`) from.

## Fixed native wait and startup (one retained NativeCondition, after the supervisor source review)

- The owner execution E = `MetalCallExecution(owner, t)` = `t * every + ordinal`, with the owner's every and ordinal. The
  ARRIVED and PUBLISH mappings are unchanged.
- **Startup has one owner.** MetalNativeHost builds one existing `struct NativeCondition` per SENT step whose
  delay > ordinal: `{warm = T, period = 1, phases = T + 1, mask = bit T}` with T = ceil((delay − ordinal) / every). It is
  zero (no condition) when ordinal >= delay. The same condition index goes on both sites of the step:
  - the `NATIVE_SUBMIT` that precedes the wait (the early queue cut);
  - the `NATIVE_DISPATCH` wait itself.
  It is appended to the preparation's existing `conditions`, its mask held in `p->held`. `MetalNativeSiteWhen` appends a
  site with a given condition; `MetalNativeSite` is now that call with `UINT32_MAX`.
- **Runtime and signature read the same condition.** LowerProgram emits `mesh_condition_N(t)` from it and wraps both sites.
  MetalSitesEmit skips both sites by the same phase test before the switch. There is no wrapper with a hidden wait or
  submit, and no second control algorithm: the earlier inline `E >= delay` in metal_lower.mm and `owner < delay` in
  MetalSitesEmit are removed.
- **The wait.** metal_lower.mm SENT emits `transfer_wait(run->sent, …, words[e % copies], E - delay, …)` inside that
  condition, i.e. `mesh_host_sent(plan, transfer, E - copies)` from the first block with E >= copies. Before then there is
  neither submit nor wait. The progress word `words[e % copies]` is the `{count, error}` cell of copy `e % sent.copies`;
  since sent.copies divides the residue count and e = E for a SENT step (owner = its own row), that is the slot of
  E − copies.
- **ARRIVED and PUBLISH unchanged.** ARRIVED and PUBLISH (delay 0) get no condition. ARRIVED's submission stays
  unconditional, and both emitted lines are the previous ones exactly.
- **Threshold check.** A brute-force check of the formula (every 1–8, every ordinal < every, delay 0–39, t 0–59; the
  condition's phase test against `t·every + ordinal >= delay`) found 86,400 cases and 0 mismatches.
- No new counter, scheduler, runtime instrumentation or delay inference.

## Signature (metal_recording.m)

- `MetalTransportObject` is keyed by `(plan, transfer, role, slot)`: publication (slot 0), arrival count word of copy `slot`,
  and completion counter of copy `slot`, as separate objects. The old `(plan, transfer)` key merged publication, arrival
  and completion into one object.
- `MetalSitesEmit` NATIVE_DISPATCH uses the same operands:
  - Startup omission is by the site's shared condition, as above. The skipped submit means no SUBMIT operation either.
  - Source execution is E − delay.
  - The value on a ring word is its generation `source / ring + 1` (the per-slot form of `(round + 1) × pages` or
    `(round + 1) × sends`); the publication value stays `source + 1`.
- The operation's own execution and vertex are unchanged. No bridge raise is added, and no `MESH_GRAPH_PLAN_COUNT` or
  `MESH_GRAPH_PLAN_SUBMISSION` objects are used. rdma/mesh-collective.c `plan_signature` emits only plan count waits and raises and
  BRIDGE_START, so these transport waits remain unmatched waits: nothing claims a bridge event that was never emitted.
  NATIVE_SENT_WAIT (`METAL_SENT_OBJECT + source`) is unchanged.
- **Remaining correspondence: the key is not fanout shared-word identity.** `(plan, transfer, role, copy slot)` separates
  roles and slots, but no canonical same-word representative is retained in NativeTransfer or in the records this unit
  reads:
  - arrival `input.count` and the `sent.counter` pointers are this process's map placement;
  - publication's `targets[0].word` is not retained.
  Two transfers, or fanout targets, that the bridge maps onto one physical word therefore get distinct objects. No
  identity is inferred from transfer ordinals. This needs a canonical shared-word representative from its producer
  (plan/fanout lowering).
- `MetalProgramSignatureSpans` now calls `mesh_graph_signature_span(planned->signature, executions, &expanded)`, where
  executions is the sum of the span counts, then `mesh_graph_compose(&expanded, &made, …)`, then frees `expanded`. compose
  allocates everything it keeps (rdma/mesh-graph.c). The program's spans are cumulative first/count as before. An old kept
  record with operations and no rows returns ENODATA with the message "the plan's signature over N executions"; there is no
  fallback to the finite record. The API was checked against rdma/mesh-graph.h at mesh HEAD b5155090 (clean).

## Archive path (metal_archive.mm, unchanged)

`MetalProgramSave` → `metal_ir_archive_save` stores:
- the library build identity;
- the typed programs: Values, with received ranges as `RangeFields` (generation, partitions, page, at, transfer, owner), so
  `input` words not stored;
- sendings: not stored;
- the solved key, wait lists, dependences and signals;
- the final native binary and symbol.

Loading (`MetalProgramMake` with `saved`):
- It checks the key, then rebuilds ranges and sendings from the loaded plan's state words. The preparation table, including
  NativeTransfer, comes from MetalProgramProject → MetalNativeHost.
- `metal_ir_native_bind` binds the saved binary.
- `delay` is in the saved binary as an immediate. `ring` and `delay` in the rebuilt table are re-derived from the loaded
  plan's `mesh_state.copies`.
- No persisted operand is genuinely missing, so no serializer field was added.

**Known independent correspondence gap:** `MetalProgramKey` hashes the residue count, horizon and planned call fields, but
not `mesh_state.copies` (derived in rdma/mesh-collective.c from horizon × calls, given lags and more). A reloaded plan whose
copies differ while residues match would pair the saved binary's baked `delay` and `mesh_condition_N` threshold with different
re-derived operands. This is not closed here. Separately, this unit does not solve independent reconstruction of the saved binding recipe; that is
its own obligation.

## Checks run (private)

Existing Makefile flags with `-fsyntax-only` and `-Wall -Wextra -Werror`, all rc=0:
- metal_lower.mm and metal_archive.mm (`clang++ -x objective-c++ $(TINT_CFLAGS) -fobjc-arc …`)
- metal_recording.m
- tools/native_bench.m
- tools/e2b_replay.m (`$(PROVIDER_CFLAGS)`; includes metal_recording.h → metal_program.h and tools/mesh_signature.h)

metal_program.h, as NativeBuild compiles generated code (`xcrun clang++ -std=c++20 -O2 -fPIC -x objective-c++ -isystem
<dir>`), with a hand expansion of one SENT step (every 2, ordinal 1, delay 6 → T 3, mask {8}, residues 12): the
`mesh_condition_0` predicate, then the NATIVE_SUBMIT and NATIVE_DISPATCH sites each wrapped in it. rc=0. This is a syntax
probe of the template only, not generated-source evidence. (A probe compiled with `-fobjc-arc` fails on metal_program.h's
existing `release`. Generated code is not compiled with ARC, so that is the wrong probe, not a defect.)

Re-run after the condition change: all five sources above rc=0.

**READY** for a named-file source commit: metal-microbench `metal_program.h`, `metal_lower.mm`, `metal_recording.m`.

## Remaining evidence scope

- **Generated entry source:** not inspected. NativeBuild unlinks `kernel.cpp` after a successful compile, and LowerProgram
  needs a lowered Program from a live recorder, so there is no feasible existing emission check without a new evaluator.
  What remains, in actual generated source after the root's shared build: the conditioned submit and wait (E < copies:
  neither), later E − copies, and the every > 1 or ordinal ≠ 0 thresholds.
- **Not done:** link, dylib build, generated-code compile, native context execution, live bridge waits. These are
  root-coordinated.
- Supervisor updated metal_recording.h to the same prior-copy/startup semantics. The pointer remains the cell of copy e mod copies.

## Supervisor review

The final source uses one prepared NativeCondition for both the queue submission and the completion wait. The fixed native emission and signature consume that same condition; startup has neither primitive. Independent strict recorder syntax and diff checks passed. The actual bit-mask threshold was checked by the worker; no runtime marker or new counter was introduced. The archive recipe/physical fanout representative gaps above remain assigned to the paired native target/archive connection.
