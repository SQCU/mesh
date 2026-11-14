# Matching v36 build and ownership observation

The canonical `rdma/libmesh.dylib` build passed, followed by native IR, mesh Metal support, live capture/program library, provider, native_bench and E2B replay targets. Original malformed legacy sources were unchanged and excluded by the existing live source lists. The captured source hashes did not change during the build. `joint-v36-source.json` and `joint-v36-binaries.json` identify the inputs and outputs.

Fresh imports in `/Users/mdot/.venv-mesh-uv/bin/python` passed with the built libraries. Checked sizes: Function168, Planned152, Local376, Signature408, ScopeSource40, ScopesData64, ValueInfo120, Actors48 and WorkAssociation40. The source-binding and operation-axis symbols were present. Logs: `joint-v36-libmesh-build.log`, `joint-v36-native-build.log`, `joint-v36-abi.log`. An initial probe used the repository's unrelated `.venv`, which has no torch; it failed before importing the native owner. The corrected canonical-environment probe passed.

`scope-ownership.c` is a focused public API check for failures ordinary numerical client metrics cannot expose: two-sided scope retention, failure ownership, parent rebinding while a composed result survives, callback release exactly once, and saved source references restored explicitly unresolved. It passed against the canonical library (`joint-v36-scope-ownership.log`). This checks the graph ownership/record contract, not actual transported native writer/site mapping.

Reproduction of that bounded check from the mesh repository:

```sh
cc -std=c11 -Wall -Wextra -Werror -I rdma output_data/compiler-workgroup-20261114/compiler/scope-ownership.c -L rdma -lmesh -Wl,-rpath,/Users/mdot/dox/mesh/rdma -o /tmp/mesh-scopes-retention
/tmp/mesh-scopes-retention
```

The executor's existing CPU and outer archive observations passed after frontend `8df355c1` corrected an operand-variable shadow at construction. CPU error was zero. Archive reload with two different inputs gave errors 0 and 6.902263703523204e-05, preserved the signature and prior result, and used distinct result storage. The actual complete-signature producer/consumer switch is `73895200`; compact runtime evidence is executor checkpoint `93b465dc`.

The single existing cost-client retry compiled, reloaded, invoked and observed successfully. Earlier BMM product-factor and stage-cycle failures disappeared. It still returned no shares because 14 Work terms from four source operations lacked an operand association. That is the frontend's next concrete source correction, not a successful optimizer result or performance comparison.

One actual archive-load responsibility remains: full signature restore still copies primary/derived arrays and recomputes scoped answered pairs. SAME schedule20 is correcting that existing canonical owner to restore the complete saved result directly under its byte owner. No graph or numerical solve-on-load is accepted as completion. The public record conversion path remains separate. Transported scope execution coverage also remains unobserved: the existing E2B record-only path needs an actual prepared distributed context, which was not available in this gate; no replacement fixture or service was introduced.
