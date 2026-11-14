# Compiler completion workgroup, 14 November 2026

This is the current implementation assignment under
[compiler-workgroup-2026-11-13.md](compiler-workgroup-2026-11-13.md).
The change requests and their cited algorithms remain the requirements.

## Operator instruction

> then lets handle the concrete uncovered work next. if there are 'legacy' paths delete arbitrary substrings from their function headers by line so that they are unrunnable (producing python or c or whatever compile/interpreter errors) so there is no confusion about what their role in the codebase is: dead code. you do understand that we are not here to 'support code' within mesh or metal microbench, but to 'implement the basic algorithm specified by the change request documents and not something else'? there is no room for multiple methods here, inlined code, complexity, excess, or distractions.

> the requirement to make the old code uncompileable while leaving its structure behind for reference is literal and didactic; it is easily done and strictly communicates something important.

Mesh `b25ec14b` removes `def ` from the headers of `rdma/mesh_compile.py` `_replays`, `_bound`, `program`,
`load`, `_runner`; `compiler/mesh_compiler/lower_coreml.py` `lower`; and `examples/tp.py` `taken`.
Their bodies remain intact. Actual parsing fails in all three files. This is intentional.
Do not repair these headers, execute the old paths, introduce compatibility dispatch, or count their
feature coverage as a reason to preserve their execution. Dead bodies may be moved, with malformed headers
unchanged, to explicitly identified reference source outside production imports. This is relocation of the
operator-requested dead material, not a source backup or an alternative implementation.

## Connected implementation units

| Owner | Producer → result → actual consumer | Applicable clauses and displaced work |
|---|---|---|
| Frontends / Torch | Captured known functions, call graph, shape/alias/state/output roles → explicit compilation operands → the current machine compiler, callable and archive | AOT A1/A10; composition 1–7; schedule-emit's stage composition. Connect call-graph compilation to the current typed machine path, including user-input gradients and inter-call mutations. Retire the dead closure orchestration, not its required semantics. |
| Compiler / placement | Function bodies and call graph → retained call dependence/width and actual actor/storage operands → one joint planner and final machine program | AOT A5–A10; disjoint-pages; Pathways/Bernstein/Dilworth as cited. Pair the exact result with frontends/executor; no second caller-specific solve or one-function restriction. |
| Compiler / schedule | Program/function identity and joint allocation → coexisting published schedules → existing island sequencer and bridge lookup/reclamation | AOT A7, appendix 6; plan-to-bridge R8. Replace singleton publication/retire-on-next-prepare behavior. Use the cited total-order sequencer and deferred reclamation; no new protocol family or resource policy. |
| Compiler / numerical | Fixed-shape spectral/least-squares operations with explicit processor placement → cited matrix/QR formulations → existing typed numerical lowerer on GPU/ANE | Source-lowering 5/6. Complete those conversions; no CPU fallback, target rejection or second numerical backend. |
| Executor / native | Completed typed neural operation plus invocation operand addresses → prepared neural invocation → generated native call | Composition 1/6/7; source-lowering. Connect changing external inputs/results/state through the existing actual native argument boundary; no unconditional staging copies or alternate execution path. |
| Executor / archive | Joint compiled functions/calls, retained plan and native entry → one saved artifact → relocation and invocation | AOT A8/A9/A10; composition 6/7. Pair with frontend/compiler; loaded output consumes the saved result, not old `load` or reconstruction. |
| Frontends / measurement | Actual compiled units and their source correspondence → execution-cost observations → existing cost/share consumer | Unequal-shards 6/7/14, AOT A3. Displace dead `taken`/FX timing. Supplied costs remain compiler inputs. Use existing measurements/client machinery; no new evaluator or benchmark grid. |
| Root / integration | Actual compiled phase/node correspondence → signature → existing consumers; dead reference relocation | Synchronization-signature; composition 6/7. Use retained phase results and correct map identity, not legacy `_period`. Keep deliberate malformed reference bodies unparseable and outside production dependencies. |

Same Opus 5.5 sessions, Codex supervisors, no new worker tree. Supervisors settle the explicit producer/result/
consumer contracts together before implementation crosses a shared file. Each unit must actually connect its
consumer and remove its displaced live body. No source-template authority is conferred on mesh's inherited code.
Use the literal algorithms/data structures and vocabulary in each cited clause and the runnable schedule-emit
composition. No new formal type system, tests, guards, fallback, invented restriction or independent runtime solve.

Commits are working checkpoints, not completion gates. Preserve raw sessions locally and compact reviewed
evidence under `output_data/compiler-workgroup-20261114/`. Coordinate shared indices and native build products;
do not touch services or unrelated processes. Current assignment is implementation and its existing local
caller observations; live fleet measurement is a separate evidence phase after the connected code exists.
