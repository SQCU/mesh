# Immutable native archive owner handoff

2026-11-13, executor supervisor; implementation SAME Opus5.5 session047e12be, resume23.

`mesh_compile._archive_loaded` pins the actual immutable Python bytes object with one CPython strong reference. It passes the original buffer, static release callback and object owner to `metal_ir_archive_load`. Successful load adopts that reference; failure releases it at the caller. The callback acquires the GIL through ctypes. There is no byte copy or address registry.

The new native Target and legacy rdma restore path share this owner. Both retain/pass the actual archive handle to `MetalProgramLoad`; they no longer pass only a borrowed solved descriptor. The native program retains the archive, while independently retained graph relations retain their original byte owner through the canonical relation loader. Constants and caller resources continue to use their existing owners.

Source review and AST checks passed. The worker's ctypes-only ownership observation confirmed one added reference, callback release (including a foreign thread), and pointer identity/alignment without importing a target library. This checkpoint changes the caller ABI and requires the coordinated v32 native build; it has no post-change runtime claim. The preceding typed CPU observation is recorded separately in typed-cpu-review.md.

The native paired unit preserves full per-residue NEWEST relations and their parent provenance; load takes their edge/source views. Duplicate serialized dependence arrays are removed in that same v32 unit. Matching saved/load runtime and outer compiler archive checks follow the full build.
