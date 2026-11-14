# Step replay: live residency facts

2026-11-14, source906ddc31 and matching v37 native libraries at e7dc2383. The same published 4x4 Hermitian MPS compile_module stops at replay of the first Step, before Program/numerical invocation. This is a diagnosis of that existing caller, not another evaluator or numerical result.

`repeat-placement-diagnostic.json` was collected by a read-only exception trace inside Target.placed while all native owners were live. It identifies row/call/instance1 among9 rows, source function1. Values and Storages both have4524 entries. Root652 is a float32 TEMPORARY/DATA root, with no payload, fixed buffer, callable argument, explicit interface, joined root, wire or generator occurrence. C Storage reports bytes0, lifetime0..8 and alignment4. The joint Function therefore reports run0 and has no copy offsets. GPU dispatch8, outside every Repeat, retains one read binding to root652 at slot4 with bytes0.

These facts place the discrepancy between the C construction/realization result and its emitted binding. They do not yet distinguish empty declared shape from an incorrectly retained local/global binding. No Python allocation change is justified. A broad rule allocating every bound root was rejected: if the writer still uses internal storage, a new global buffer would be uninitialized.

An earlier diagnostic tried reading Programs from a traceback after Owned cleanup. Those objects had been freed; all resulting counts and tentative explanations were discarded. The live capture above supersedes that invalid diagnostic. Source and numerical behavior were unchanged during both observations. The exact existing client command is compiler/eigh-native-command.md; production failure evidence is compiler/eigh-native-2.stdout/.stderr.
