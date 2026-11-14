# Fused local escaping its owner (repeat-live-local.md) — status

Session e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, explicit claude-opus-5-5. Correction committed as metal430a9fe1; matching build and ordinary CPU/archive checks passed.
- The live query stopped before Program/invocation. The unchanged 4×4 MPS retry is still pending.
- Original malformed sources are untouched.

## Live facts (before Owned cleanup)

**How they were read.**
- The existing `Target.placed` exception hook was copied as `repeat-live-local-hook.py`, with the same
  `compile_module` call.
- It reads programs[r] of the original row object and runs a private read-only field reader:
  `repeat-live-local-facts.mm`, built in /tmp, header-backed, no mutation, no shared dylib.
- Output: `repeat-live-local-facts.json`.

**Root 652.**
- TEMPORARY F32, shape [16]: not empty.
- **`internal = 7`**, `realized = 0`.
- Its aliases 653, 656, 660, 662, 666, 668 and 672 are all internal 7.

**Kernel 7** (GPU, ops 317–377) holds 652 as a local of extent 16. Its writers and in-kernel readers:

| Writer op | Writes | Elements | Readers inside kernel 7 |
|---|---|---|---|
| 368 | 653 | [0, 4) | 369 |
| 370 | 660 | [4, 8) | 371 |
| 372 | 666 | [8, 12) | 373, 377 |
| 374 | 672 | [12, 16) | 375 |

**Kernel 8** (GPU, ops 378–422, outside every repeat region):
- op 379 reads 660 and op 381 reads 653;
- its binding 4 (slot 4, access 1, bytes 0) names root 652;
- no local of kernel 8 names it.
- That is the escaped binding behind "Command 8 binds value 652 at 4, which the plan did not place".

**Repeat edits are not the cause.**
- All ops involved have `repeat = NONE`.
- The repeat-keyed fusion conditions are equal-NONE here.
- ExternallyVisible's region additions only add visibility.

## Producer mismatch (metal_transform.mm `Proven`, `Internal`)

**Locality is proven per writer.** `Proven` checks each written operand only against the ops that conflict with that
operand's own write views:
- the writers of [8, 12) and [12, 16) prove locality, because readers 373, 375 and 377 are inside the group;
- the writers of [0, 4) and [4, 8) fail, because 381 and 379 in kernel 8 read those parts.

**The root is made local whole.** Yet `Local::value` is the storage root, and `Internal` marks every value of that
root internal. So `Admitted` took 372's candidate, and the whole root became kernel 7's threadgroup local:
- Realize then gave it 0 bytes;
- kernel 8's binding still named it.

## Correction (repeat-local-proof.md, iteration68): whole-root proof inside the existing `Proven`

**Why iteration67's draft was replaced.**
- That draft post-filtered candidates, requiring every writer of a root to own a proven candidate.
- That rejected valid local classes: writes inside a confined class that another member overwrites, or that nobody
  reads.
- It also added candidate × operations × candidate scans.
- It is gone. `Proven` itself now proves each storage root once, in that root's own domain.

**Per root**, at its first written operand in group order (`metal_transform.mm` `Proven`):
1. **Not externally visible** (`ExternallyVisible`).
2. **Confined.** A non-erased operation outside the group with any retained `Accesses` view of the root keeps the root
   in its storage. This is root ownership, not a fusion rule: `Joins` and grouping are unchanged.
3. **Written first.** The first member touching the root writes it and does not read it. A later member reads it.
4. **Constraints on every member's access of the root:**
   - no scatter into it;
   - no quantization scale or zero stored in it;
   - its gathers keep the existing scalar, storage-scalar, quantization and pages checks, and their reads collect for
     the gathered proof.
5. **Geometry.** The existing `Partitioned` decoded geometry runs over every member's operand of the root, from the
   group's first member, including any before the first writer.
   - All operands need equal digits and scalar.
   - `lo` and `hi` are the union over them, so the extent covers the whole class.
6. **Gathered proof** (existing, unchanged): `lo` 0, no digits, one threadgroup. Each gathered read lies within the
   extent and is covered by a contiguous write of the class's actual writes.

Overwritten or unused writes inside a confined class need no proof of their own. Admitted, Internal, Realize, Time and
Profitable are unchanged and read the same candidate list.

**Expected effect for 652.** Ops 379 and 381 (kernel 8) access the root outside kernel 7's group, so the corrected proof cannot make its root internal to kernel 7. The next unchanged client observes actual placement; no corrected live placement has yet been reported.

**Not done:** no blanket realization, residency, copies, planner, formula or Python change, and no further diagnostic
run.

**Strict** (`-fsyntax-only -Wall -Wextra -Werror`, actual Makefile flags): rc=0 on all 8 METAL_IR_SRCS and both tools.
The original `metal_recording.m` is unchanged and was not compiled.

**Changed:** metal-microbench `metal_transform.mm`, `Proven` only, replacing iteration67's filter.

The live652 evidence is `repeat-live-local-facts.json`. The temporary private query and exception hook supplied these facts before native cleanup; they are not retained as another client evaluator.

Supervisor review corrected first-touch ordering to use the retained group index rather than source operation ID. The confinement scan checks actual outside uses, then the existing geometry pass follows group order. Compiler co-review and independent strict transform syntax passed before commit.

The matching build passed without source hash changes. `local-class-cpu.json` reports zero error. The outer archive reload in `local-class-archive.json` reports errors 0 and 6.902263703523204e-05, unchanged first result after the second invocation, distinct input/output storage, and the same retained signature. This validates the existing ordinary path; GPU eigensolver accuracy remains pending.
