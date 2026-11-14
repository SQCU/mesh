# Typed native CPU client observation

2026-11-13; executor supervisor. Existing `examples/engines.py`, one CPU machine with logical projection/part lanes.

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
DYLD_LIBRARY_PATH=/Users/mdot/metal-microbench/.build:/Users/mdot/dox/mesh/rdma \
ENGINES=cpu OP=ffn ROWS=2 K=32 N=48 LAYERS=1 \
/Users/mdot/.venv-mesh-uv/bin/python examples/engines.py --nodes 0 --device cpu
```

Result: exit 0, output `[2, 32]`, normalized maximum error **0.0**, reported call time 0.00015379198885057122 seconds. The two source channels are intra-machine and become ordinary values in one native entry. Preparation reports 55 values, 36,752 bytes, one schedule position, zero GPU commands and zero fence/host/transport waits. The signature digest is `56c07ef0f51c1db72e5843362869573c37083e3d5d80dacac4476d1a4fc48836`. This is one correctness observation, not a performance comparison.

The actual route is `compile_module → compiled → Program`, through typed machine construction, lowered Actors/Storages, one local plan, final native compilation, saved native bytes and relocation/load in `Target.prepare`. The load reports positions restored without solving a relation. This exercises native artifact reload; it does not yet exercise outer filesystem `archive.save/load`.

The first attempt needed the canonical library search path. The next reached native preparation but reported 56 values against the retained storage table. Compiler inspection identified `metal_ir_step_value` as a source constructor: the post-lowering query appended a STEP to this step-free FFN. The correction reads `ValueInfo.word == METAL_IR_STEP` in the existing metadata pass, preserving actual steps without constructing new ones. The legacy rdma `_lifetimes`/`_built` consumer now carries that same existing identity.

Source at the successful observation: metal `4db95f8d` plus held recorder/archive source changes (not rebuilt), mesh `9388eb3e` plus this STEP fix and current reviewed frontend construction. Canonical libraries were the retained-native Metal build and matching query/closure libmesh build; installed Actors ABI was 40 and RelationData was 152. Logs: `typed-cpu-ffn-step.json` and `.stderr`; prior failed observations remain in `typed-cpu-ffn-libs.*`. The same worker identities were preserved (archive047e12be, nativee48f4bc7), explicit/reported `claude-opus-5-5`.

Still separate: full causal/source discharge, recorder canonical relation handoff and full archive ownership, dynamic alias/mutation/repeated-result coverage, GPU argument binding, heterogeneous placement and live transport. This local observation makes no claim about them.

## Matching retained-owner build and outer archive

The matching build at mesh `04e1eed3` and metal `79d7e7a6` passed the same existing CPU client: exit 0,
normalized maximum error **0.0**, output `[2, 32]`, and the same signature digest above. Its actual ABI was
Actors48, Local144, Storage24 and RelationData160, with the v32 archive byte-owner interface and prepared phase
getter. The source build inventory is `compiler/full-retained-build-source.json`; build/ABI logs are
`compiler/full-retained-{libmesh-build,native-build,abi}.log`.

The filesystem archive path was then exercised using the existing Engines FFN model and production
`compile_module`, `compiled`, `archive.save`, `archive.load` and `Program`. The original Scheduled was deleted
and collected before loading. The restored entry was called with the original input and a distinct allocation
containing `x + 0.125`. Normalized maximum errors against the original model were **0.0** and
**6.902263703523204e-05**, respectively. This is the client's reported error metric; Engines defines no fixed
tolerance. The temporary invocation's `1e-2` stop condition is not a project acceptance standard.

The first returned tensor remained exactly unchanged after the second call. The two inputs had distinct storage,
the two fresh outputs had distinct storage, and the restored signature bytes equaled those saved. The archive was
182,754 bytes. Native load reported positions restored with no relation solved. Both processes exited 0 with
empty stderr. Logs are `typed-cpu-ffn-owned.{json,stderr}` and `typed-cpu-archive-owned.{json,stderr}`; the actual
archive is `typed-cpu-archive-owned/`. No new evaluator was retained.

SHA-256 of the libraries used:

| Library | SHA-256 |
| --- | --- |
| `.build/libmetal_ir.dylib` | `c230a7c384ad026b29ec594621bb2ad815c69d488f845dcfb841af282752fdfa` |
| `.build/libmetal_recording.dylib` | `fbb701c9c22295a1bdfdb66ab5d039743fd214cfa1a97c37ed5104e7da681090` |
| `rdma/libmesh.dylib` | `0f200419c927417c3e1789ba71406511d91fad668da8fbb864127ee7e65cae4b` |

This observation covers the local CPU entry, original archive-byte ownership, saved native relation views and
fresh result ownership under changed handles. It does not establish general alias/mutation, RNG, GPU/ANE,
transport or remaining source-cause and Actor/function persistence obligations. The reported single-call time is
not a performance comparison. Worker identities remain native `e48f4bc7` and archive `047e12be`, both explicitly
reported as `claude-opus-5-5`; these validation calls were run by the Codex supervisor.
