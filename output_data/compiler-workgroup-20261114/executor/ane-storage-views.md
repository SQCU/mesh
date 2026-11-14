# ANE storage views — bounded source facts and proposed handoff

Read-only continuation after metal `ff3609f7`; no implementation/build/run yet.

Actual source authority:
- metal_ir.h94–103 / Value611: storage already names the canonical alias class; shape, offset and layout map
  logical indices to physical element offsets in that Value's scalar units. No alias discovery is required.
- metal_ir.mm132 Element evaluates offset + sum(Term(layout, index)); Bytes1036 gives that view's physical end.
  Value.bytes is initial-data length, not a general storage extent. Do not use it as the allocation span.
- Realize2004 folds actual realized view extents to class roots, but Artifact2390 runs it AFTER LowerNeural.
  The MIL constructor cannot read a future p.realized result as its input.
- The actual Python Builder.whole (metal.py617) already declares the full backing storage as 1-D nbytes/element_size;
  value() retains precise offset/stride aliases. General C class merging does not enforce root extent dominance.
- metal_neural.mm685 Need caches by ValueID. Store and final output emission also use that key. A store through
  view A does not advance view B's tensor even when their declared storage is identical. Branch restoration
  preserves that per-ValueID map correctly; it cannot repair this missing shared storage owner.

Smallest proposed representation in the SAME Lowering owner:
1. A storage-indexed SSA tensor holds the current contents for each actual class used by the neural function.
   It is one fixed 1-D physical element span, typed by the class's actual supported storage element type.
   The original ValueIDs remain immutable logical views and original operation identities remain unchanged.
2. Produce that span from retained typed view extents using existing Bytes/Element, while collecting actual
   function inputs/outputs/symbols/logical reads. Include the original declared root's extent; do not assume
   that root.shape alone covers every alias, and do not count erased/unrelated classes. This is a direct
   declared-layout projection, not another alias/lifetime solve. The final binding carries this actual span
   to the existing realization owner; no Native buffer-size or latest-writer inference.
3. A read composes the operation's actual logical coordinate tensors with the Value's existing offset/layout:
   byte address = Element(value, logical coordinates) * ScalarBits(value.scalar)/8.
   Gather those addresses from the SAME current storage tensor. Do not first materialize the whole logical
   view at static bounds and then slice it; Operand/LogicalRead can consume the composed actual-size indices.
   Need for a scalar symbol uses this same view read, so a newly written alias is visible immediately.
4. Store composes its actual output coordinates through that same layout and scatter-updates the current
   storage SSA tensor. Every update consumes the prior whole storage tensor, preserving disjoint and inactive
   regions; the next read of any alias observes the accumulated result. No last-writer search or overlap test.
   Cached immutable constant views may remain; mutable read views cannot survive a storage update as authority.
5. Branch/while lexical snapshots and cond outputs carry storage SSA values, once per actually written class.
   Parallel-empty branches yield the incoming storage tensor. Different output aliases of one storage must
   not independently replace its final state or bind contradictory old output snapshots.
6. The existing Neural binding/IO producer emits the fixed physical storage input/output view and original
   root identity, with offset0, fixed element count/stride1 and actual access role. This changes which compiled
   view its existing fields describe, not the public ABI or per-invocation shape. Original source Value views
   are projections of that result. NeuralModelOf/Bind already preserve these fixed descriptors and compiled
   files, so no new runtime storage cache, archive schema or output-copy policy is proposed.

Typed limits requiring paired review BEFORE code:
- For a uniform element representation, the mapping above is exact, including strided/offset/quasi-affine views.
- Numeric Mil::Cast is not a byte reinterpretation. Mixed F16/F32/I32 interpretations of the same written bytes
  cannot be made correct by numeric casts or by splitting one alias class into unrelated tensors. The current
  owner has no bit-reinterpret primitive; publish an exact unsupported case if encountered rather than invent
  a byte VM. I32/U32 retain common 32-bit storage bits, while unsigned numeric comparison remains separate.
- Existing immutable quantized constant decoding remains a read-only source path; do not reuse a decoded
  constant as mutable physical storage. Source dtype/quantization correspondence must remain explicit.
- Core ML compute partitioning, model input/output overlap behavior and actual physical work are not established
  by this proposal. One requested-ANE client remains gated by source review and matching compile.

Scope for a paired implementation, if accepted: metal_neural.mm only, existing Need/Operand/LogicalRead/Store,
conditional ownership and fixed IO emission. No new alias solver, scheduler, shape variants, native root guess,
operation placement change or numerical algorithm. Source original poisoned paths remain untouched.

Compiler acceptance for SAME e48 implementation:
- Produce the fixed physical span as an actual typed Value/view alias of the canonical root where needed;
  the existing later Realize must consume that SAME typed reach from the binding. Do not read future p.realized.
- Choose scalar representation from actual live reads/writes. A byte-typed backing declaration is not itself
  a conflicting semantic interpretation; a single live F32 interpretation over it must remain supported.
- Distinguish actual incoming read/source ownership from the syntactic prior argument of a MIL scatter.
  A genuinely internal fresh TEMPORARY first partial write must not create an external input which has no
  native backing once internal residency is selected. Existing source role, initial contents and real reads
  determine incoming contents. Report that exact decision and the fresh internal SSA representation.
- Branches merge written original storage classes once, retaining accumulated partial updates. No independent
  per-view output state. No new planner/alias solve, ABI, cache, placement rule, or numeric reinterpret cast.
- Source implementation and private strict syntax only, then HOLD; compiler owns the eventual requested-ANE
  public client and MLComputePlan observation after matching build.

Compiler follow-up: fresh internal prior authority
- Inspect and use the existing `Internal(p,value,kernel)` / corrected whole-class `Value.internal` authority (Proven430a9fe1) before introducing any read-history inference. An actual class proved internal to this kernel cannot become a Neural model input merely to seed scatter; later Realize excludes its backing. Noninternal classes retain actual native backing/prior across kernels. Role alone is insufficient: OUTPUT or TEMPORARY can have prior writers elsewhere. If MIL requires an initial full tensor for internally undefined bytes, represent an explicit local undef refinement consistent with source writes, and document it. Do not add a new alias/read-history/allocator analysis.

Final source-timing correction, accepted by compiler:
- Transform produces Proven/Internal only for CPU/GPU (metal_transform.mm1352-1357). ANE assigns internal later inside LowerNeural. Therefore use the SAME maximal-run operation membership and its existing outside/ExternallyVisible scope owner EARLY, including outside operation inputs, outputs, gathers AND symbols. A class touched by another kernel retains actual incoming backing even if TEMPORARY; no role-only inference. The existing corrected Internal predicate remains authoritative when already produced.
- Preserve declared initial contents/invariant source. Confined does not mean undefined if any actual class view/root carries initial data. Local zero-fill is only an undef refinement for a genuinely confined class with no declared initial contents. No new writer/history solver.
- Remove newly added natural-language source comments (Classes/Current etc.); use documentation citations. Do not clean unrelated existing prose.
