# Current integration assignment

## Current storage-consumer correction (continuation 5)

`integration-storage-assignment.md` is the current unit. Codex compiler review has now settled these concrete
operands from the actual source; consume them before choosing an allocation implementation:

- Provider Bind (litert_mesh.cc4788) creates one value per original TFLite tensor index. Repeated interface entries
  reuse that ValueID. Its only value_view calls create fresh internal views, not arbitrary distinct interface roots.
  Capture external alias correspondence from each retained **unlowered** source/interface during export, before
  append frees it. Do not discover new aliases from post-lowering storage reuse.
- Project those declared alias classes to one planner storage index. `mesh_graph_components` on symmetric source
  correspondence edges is already the canonical transitive composition algorithm. No inline union-find.
- Remap `mesh_declared.value` and Operand/storage correspondence before mesh_plan, retaining transfer identity,
  from/to views and dependence records. Allocation span covers the exact views; lifetime covers their uses;
  registration follows any member's wire obligation. One relation must reach planned annotations, host addresses
  and final physical bindings.
- Recompute storage read-before-write from the existing ordered stage input/output accesses. Merely OR-ing the
  old per-Operand `first` bits is wrong after aliases join. Project per-phase owns/captures to the storage class.
- This caller's ONLY nonzero lag producer is `backing==token ? 1 : 0` near1112. Retain that explicit token recurrence
  owner for the class, including planner lag and the final ahead offset. Do not replace it with MAX(lag), an arbitrary
  first operand, or an invented compatibility policy. No second generation domain was found in this caller.
- Provider Bind creates contiguous offset-zero interface roots. Its offset-bearing Symbol and reshaped View are
  fresh INTERNAL values. Keep those logical views through append and bind the canonical physical storage. There
  is no current source basis for adding a host gather/view interpreter; `Operand.address` remains the storage base.

The simple composition is to export retained bodies before deriving external storage/access policy, form the
declared storage correspondence once, then bind/append at the original phase positions. Provider only branches
on CONSTANT for constant reuse; other interface roles do not select numerical operations, so an early export may
use its default interface roles and destination binding supplies the phase role. In that route, map ALL source
ValueIDs in each bound interface storage class through call.values to its phase backing. The existing append
contract accepts a backing with a different logical shape, clones each view's own layout/offset and applies the
supplied phase role. Otherwise provider View's inherited INPUT/OUTPUT roles would survive a TEMPORARY phase
binding. Independent constant/state storage stays untouched. No extra append heuristic or role schema is needed.

Implement this finite connection. Existing source locations identify operands and replacement sites; they are
not authority to preserve the inherited per-Operand allocation. Read the remainder as superseded work history
where it conflicts with this correction.

## Earlier composition assignment

The earlier signature/client and inherited-body repair assignments are superseded. Read and implement only
design/compiler-phase-composition-assignment-2026-11-13.md. Existing checkpoint3a2aa8c1 is review history,
not acceptance. The guessed export and numerical renovation were removed in817ff6a3.

Numerical worker owns metal_ir_append and Program retain/release; frontends owns opaque LiteRtMeshExported
producer. Actual header provider/litert_mesh.h is the contract. ExportedProgram borrows Program,
ExportedInterface returns ordered typed interface, ExportedDestroy releases the source Program reference.
The new append retains the source once, preserving reusable source and borrowed constant bytes.

E2B must export once, bind common interface storage before host fills, then append each retained body at its
original stage position. Remove the two provide paths. Frontends will delete the obsolete Declare/Publish public
APIs once your caller switches; report exact source readiness promptly for coordinated build.

Keep all edits to the concrete phase consumer. No reimplementation of provider numerical code, no new parser
or clone, no target-dispatch or Python-lowerer edits. Report progress/interface needs in integration-status.md.
Codex reviews the resulting connected calling path and evidence.

Concrete review correction from frontends/compiler: distinct manifest Operand objects can name the SAME source
ValueID or source-declared alias storage. In export_stage, don't allocate a different phase backing for each and
then last-write-win in append_stage's values[source_id]. Consume source alias/storage correspondence when binding
initial phase roots; reuse/coalesce the actual backing while preserving each logical offset/layout. The common
case of one KV Operand twice is insufficient coverage. Numerical append preserves source alias classes; it must
not receive a map that has already discarded this relationship. Report the exact binding correction before build.

## Frontend source review for root's retained E2B phase binding

Provider is final/held: `LiteRtMeshExportedProgram` returns const Program, interface ordered as source subgraph inputs then outputs, and repeated source tensor IDs share one `entries[k].value` (and storage). Legacy Declare/Publish entry paths are deleted. Final private compile/header/patch checks pass; root must build provider with append and new caller together.

Concrete review finding already sent root: `export_stage` presently creates a fresh phase value for each previously-unbound manifest Operand even when two source entries alias. `append_stage` then assigns `values[entries[k].value]` repeatedly, losing earlier correspondence before append can validate it. Preserve the actual source `value_info.storage` relation in initial phase binding, coalescing/reusing corresponding phase backing via the canonical storage-view owner where appropriate. Last-write-wins is not an alias handoff. The common KV case naming the same Operand already works, but source alias equivalence is not manifest object equality. Carry this through the existing backing/planning correspondence, rather than inventing a parallel alias scheme.

The exported source program owns model/body data; append retains it. Export Destroy after append is safe under compiler refcount. Do not copy owner callbacks or traverse model bodies again. Source constant identity deduplication is now actively preserved by compiler's Program.constants/backing index, not by E2B.
