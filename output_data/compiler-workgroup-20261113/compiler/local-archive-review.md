# Complete native local result retention

The executor's draft reduced Local to Python Layout tuples, freed the native result, and reconstructed Planned. The connected replacement retains Local, Function/value arrays and Actors owners through final emission; native save/load persists the actual result for later relocation.

Reviewed producer: mesh_local_save/load uses the existing plan_archive_bytes owner. It preserves full ValueID-indexed mesh_value records, compiled host/work/edge arrays, native call/order/cause/horizon, complete first/offset copy table and arena extent. Ordinary planning borrows existing input arrays; loaded arrays have one private owner released by mesh_local_free. No Bff/allocation/signature algorithm runs on load. Source producer has no local transfer/task inputs; those absent inputs are not fabricated in the loaded function. An allocation-failure pointer-arithmetic issue found in review was corrected before checkpoint.

Source/private checks: canonical C compilation, strict warning compilation, Python syntax and C/ctypes layout agreement Local144 (owned offset136). Runtime round-trip and actual Target consumer remain pending the coordinated libmesh rebuild. Existing Local136/Storage24 coherent build evidence is in local-plan-review.md; it does not validate the new Local144 ABI. Fresh imports are held until that rebuild.

Owned implementation: SAME Claude session dc4b988b-a100-4e17-8c80-ff3f90ac97e3, explicit and reported claude-opus-5-5, schedule.resume9/10 logs. Codex reviewed the producer and exact native consumer contract. No live bridge or E2B execution is claimed.
