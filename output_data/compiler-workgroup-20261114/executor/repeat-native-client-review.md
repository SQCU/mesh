# Repeat v37: ordinary caller and outer archive

2026-11-14. Matching native build at metal e7dc2383, mesh e2cbf708, with RepeatRegion40 and Signature416. Compiler's repeat-native-source.json and repeat-native-binaries.json record the exact held source and linked libraries; repeat-native-abi.json records fresh imports and the four repeat symbols. The ordinary client below contains no repeat region and does not establish repeat numerical correctness.

The existing Engines CPU FFN client (ROWS2/K32/N48/LAYERS1, CPU0 projection and part0) passed with normalized maximum error0, exit0 and empty stderr. It produced54 values/36744 bytes, one native entry and no GPU commands, crossings or waits. Signature digest56c07ef0f51c1db72e5843362869573c37083e3d5d80dacac4476d1a4fc48836. The observed single-call time is not comparative performance evidence.

The same model passed outer save, deletion/GC, load and Program invocation on x followed by distinct x+0.125. Normalized maximum errors were0 and6.902263703523204e-05. Saved signature bytes matched; the first output remained unchanged after the second invocation; input and output backing pairs were distinct. Archive200997 bytes, exit0 and empty stderr. Actual native reload reported no relation solve.

Logs are repeat-native-cpu.json and repeat-native-archive.json. This check validates ordinary execution and loaded ownership after the v37 source changes. The compiler's separate small Hermitian MPS client is the next actual repeat execution; no GPU/ANE/network or repeat cost result is inferred from this CPU check.
