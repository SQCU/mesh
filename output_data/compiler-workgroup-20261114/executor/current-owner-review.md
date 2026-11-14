# Current owner and archive observation

2026-11-14, after clone checkpoint metal d598ec53 / mesh88e93470 and the matching current-source canonical build.
Source and library hashes are in current-owner-source.json. Function168, Planned152, Local376 (Grouped at368),
ValueInfo120, Storage24, Actors48 and WorkAxis24/Extent24/Term128 matched the actual libraries. The current native
caller now accepts the driver's authoritative targets keyword and retains it during compilation.

The existing Engines CPU FFN client (ROWS2/K32/N48/LAYERS1, projection and part0 on CPU) exited0 with empty stderr
and normalized maximum error0.0. It produced55 values/36752 bytes, one native entry, no GPU commands, crossings or
waits; signature56c07ef0f51c1db72e5843362869573c37083e3d5d80dacac4476d1a4fc48836. Its single-call time is not comparative
performance evidence. Logs: current-owner-cpu.json and .stderr.

The same model passed archive.save, deletion/GC, archive.load and Program calls on x then a distinct x+0.125.
Normalized maximum errors were0.0 and6.902263703523204e-05 using the existing client's metric. Signature bytes
matched; input and fresh-output backings were distinct; the first output remained unchanged after the second call.
The200943-byte outer archive restored the complete native artifact and joint owner with no relation solve reported.
Exit0, empty stderr; logs current-owner-archive.json and .stderr. This covers the current v35 persistence on this
ordinary local program, not Work calibration, network publication, live transport or ANE execution.

No broader repeat was run. Next source work is the actual prefix/body/suffix port-copy consumer and aggregate writer
completion; these remain mandatory open responsibilities, not behavior inferred from this local observation.
