# Native Actor result lifetime

2026-11-13, executor source review; SAME Opus5.5 sessions e48f4bc7 (recorder relation/Actors48) and047e12be (legacy caller).

The recorder Actor result holds the actual canonical quotient relation, whose parent retains original dispatch effects and class correspondence. The native Target already keeps this result in Solution through planning and finalization. The legacy rdma caller now retains the same C result in its existing construction lifetime, with original host/work/edge pointers and accessible owner in the retained tuple. Three memcpy projections and immediate free are removed. Teardown frees Actor before its source Program.

Where the existing caller still copies input tensors into owned slots, it produces a derived host-cost array while retaining the original Actor and actual added measured copy cost. Original work/edge facts remain direct views. `_bound` reads those retained pointers. The old save recipe now serializes pointed-to contents by actual counts, avoiding pointer-address serialization after this change. That recipe still omits full Actor quotient provenance: canonical compiled-function/Actor persistence remains an explicit compiler-owned boundary, not closed by this checkpoint.

Private source/syntax checks pass. `_Actors` is now48 bytes (relation pointer at40), matching the changed recorder; no runtime result is claimed until the coordinated binary build. The earlier CPU client observation used the preceding Actors40 library and is recorded separately.
