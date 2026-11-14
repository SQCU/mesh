# Current joint native local client and archive observation

2026-11-14. Native source checkpoint metal e7db540c, Python consumer d59f7b94, joint native producer checkpoint
mesh 04cd25af. Actual source and library hashes, including held Work/report changes, are in
joint-local-observed-source.json. Compiler's matching build and ABI logs are local-native-entry-build.log and
local-native-entry-abi.log in its workgroup evidence. Relevant ABI: Function168, Planned152, Local368, Actors48,
Storage24, WorkAxis24, WorkExtent24, WorkTerm128.

The existing examples/engines.py client ran with CPU projection and part0, ROWS=2, K=32, N=48, LAYERS=1, OP=ffn,
TORCH_DEVICE_BACKEND_AUTOLOAD=0 and the canonical library path. It exited0 with empty stderr, output[2,32] and
normalized maximum error0.0. Its reported single-call duration is not comparative performance evidence. Preparation
reports55 values/36752 bytes, one native entry, no crossings or waits, and signature
56c07ef0f51c1db72e5843362869573c37083e3d5d80dacac4476d1a4fc48836. The actual route is compile_module through the shared
compile_program path, original call analysis, one joint Local solve, native emission and Target.prepare artifact load.
Logs: joint-cpu-ffn-entry.json and .stderr.

The same existing model then passed archive.save, deletion and collection of the original Scheduled, archive.load,
and Program invocation with x and a distinct allocation containing x+0.125. Normalized maximum errors were0.0 and
6.902263703523204e-05 using the client's metric (which defines no fixed tolerance). Signature bytes matched, the first
returned tensor remained exactly unchanged after the second call, and both input and fresh-output storages were
distinct. The196901-byte archive loaded native programs and the full saved joint result; preparation reported no
relation solve. This process exited0 with empty stderr. Logs: joint-cpu-archive.json and .stderr; artifacts in
joint-cpu-archive/.

The first attempt failed during generated-native compilation because CompileTask read an entry name after Tint
Generate changed the typed function. The corrected producer retains that name before Generate; no emitted-source
parsing or suffix repair was added. Its failure log remains joint-cpu-ffn.stderr; entry-output-review.md records the
source ownership review and the rejected kernel-ordinal substitution.

These observations cover this ordinary local CPU function and its saved owner path. They do not establish the
unfinished cross-call network port/demand/outer-result connection, live transport, ANE execution, complete compiled
Work calibration or comparative performance. The separate frontend's exact noncontiguous mutation observation is
recorded by that owner; it is not folded into this client metric. No broader repeat was run.
