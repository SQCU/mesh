# Synthetic marker projection and actual Hermitian preparation

Native `ba92e7a8` changes only the existing placed-access producer: each host/residue/part bucket accumulates read/write bits and emits at most one synthetic key0 coherence marker per direction. Every real physical view and the overlap solver remain unchanged. Source review and private strict checks passed; executor evidence is `68ef1ce6`.

The canonical five-target native build completed in 22.51 seconds with exit0 and unchanged source hashes. The fresh ABI/import probe passed, unchanged from v37/Signature416. Exact source, binary and ABI manifests are `eigh-marker-{source,binaries,abi}.json`; full local log `eigh-marker-build.log`.

The unchanged MPS attempt4 passed all nine source-call replays. A read-only live query at 2m02s showed 53,613 views, outer index15,813 and a real key1 physical-access pair; the duplicate synthetic-marker domain had been crossed. Preparation then completed: 30 native rows/positions, one residue/phase, 28 plan order edges, 29 retained dependences, 15 fence waits, 14 program-thread waits and 21 start waits.

The process then crashed with SIGBUS138 in native LowerProgram, before compile_module returned and before numerical invocation. The actual instruction/registers identify observing() indexing program.bindings with bindingUINT32_MAX from a barrier incorrectly assigned unit1. Two MetalNativeWaits paths append sites/marks directly and omit unitOf; the paired native owner is correcting those callers through the existing MetalNativeSiteWhen helper. `eigh-native-attempt4.json` retains compact crash/query facts and original-report/log hashes. No spectral accuracy or repeat archive roundtrip is claimed.
