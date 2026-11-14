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
