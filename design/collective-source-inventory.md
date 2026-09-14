# Collective source inventory

Baseline: mesh `6368860`, the baseline named in `collective-goals.md`. Current mesh: `de40122` plus the working tree. This is a later baseline, not a reconstruction of the original session start.

Counts are physical text lines, including comments, blank lines, embedded kernel source and generators. They are not a claim of structural halving. Both sides use the same filename rules. Deleted paths remain in the denominator; newly added paths are included. Binaries, data, documentation and untracked build artifacts are excluded from source counts. The source rule includes `.py`, `.c`, `.h`, `.m`, `.mm`, `.cpp`, `.cc`, `.swift`, `.metal`, `.sh`, `.bash`, `.zsh`, `.def`, `.modulemap`, `Makefile`, `CMakeLists.txt` and `setup.cfg`.

| Set | Before | Current | Change |
| --- | ---: | ---: | ---: |
| All tracked mesh source | 264714 | 262474 | -2240 |
| Mesh Python, native, generators, build and streaming example | 7785 | 5930 | -1855 |

## File inventory

This lists the complete Python/native/build/example set, including auxiliary RDMA programs rather than silently excluding them.

| File | Before | Current | Change |
| --- | ---: | ---: | ---: |
| `examples/streaming-chain.py` | 72 | 104 | +32 |
| `python/mesh/__init__.py` | 417 | 450 | +33 |
| `python/mesh/_native.py` | 84 | 72 | -12 |
| `python/mesh/collective.py` | 0 | 51 | +51 |
| `python/mesh/kernels.py` | 2869 | 1569 | -1300 |
| `python/mesh/nn.py` | 161 | 132 | -29 |
| `rdma/Makefile` | 17 | 17 | +0 |
| `rdma/mesh-algebra.h` | 73 | 71 | -2 |
| `rdma/mesh-algebra.m` | 1479 | 1037 | -442 |
| `rdma/mesh-dataflow.c` | 1016 | 804 | -212 |
| `rdma/mesh-dataflow.h` | 77 | 58 | -19 |
| `rdma/mesh-flow.c` | 396 | 433 | +37 |
| `rdma/mesh-kernel.h` | 23 | 23 | +0 |
| `rdma/mesh-memory.h` | 58 | 58 | +0 |
| `rdma/mesh-stat.c` | 34 | 34 | +0 |
| `rdma/mesh-verbs.h` | 239 | 239 | +0 |
| `rdma/mesh.h` | 155 | 163 | +8 |
| `rdma/mesh_coreml.py` | 48 | 48 | +0 |
| `rdma/module.modulemap` | 10 | 10 | +0 |
| `rdma/peers.py` | 341 | 341 | +0 |
| `rdma/workload.py` | 176 | 176 | +0 |
| `rdma/xonwire.def` | 13 | 13 | +0 |
| `setup.py` | 27 | 27 | +0 |

## Caller implementation included in integration

The external numerical library is the existing engine at `0e18c5c`, including its working-tree changes. Its full `ENGINE_SRCS` expansion is 27 files and 15297 physical lines. Counting only `mesh_matrix.swift` would conceal this dependency. The library also depends on the canonical mesh native source already counted above.

| Engine library source | Current lines |
| --- | ---: |
| `mesh_matrix.swift` | 43 |
| `gguf_loader.swift` | 321 |
| `safetensors.swift` | 143 |
| `model_file.swift` | 231 |
| `bootstrap.swift` | 2541 |
| `weights.swift` | 653 |
| `runtime.swift` | 18 |
| `common.swift` | 114 |
| `metal_runtime.swift` | 122 |
| `dense_execution.swift` | 314 |
| `diagnostics.swift` | 6 |
| `kernels.swift` | 4096 |
| `parameter_configuration.swift` | 25 |
| `matrix_shaders.swift` | 190 |
| `matrix_operations.swift` | 263 |
| `vision_tower.swift` | 798 |
| `vision_residency.swift` | 142 |
| `tokenizer.swift` | 226 |
| `lm_engine.swift` | 2400 |
| `stream_output.swift` | 37 |
| `pending_poll_response.swift` | 19 |
| `prefix_hash.swift` | 28 |
| `page_manager.swift` | 290 |
| `radix_trie.swift` | 436 |
| `kv_ssd_store.swift` | 189 |
| `ffi.swift` | 709 |
| `ffi_batch.swift` | 943 |

## Artifact and documentation disposition

Commit `e1e4540` removed 493 JSON, JSONL and compressed trace/provenance files totaling 58,041,333 stored bytes. None of those bytes or lines count toward source reduction. Documentation edits and removals are also excluded. Comment-to-documentation migration has not been independently quantified, so the physical-line totals must not be presented as structural implementation reduction.

Outstanding implementation includes the receive/storage-reuse contract in G4 and the full model collective call graph in G6. Dedicated transmit/receive threads and the working partial-tensor chain do not establish those remaining requirements.

The public server adds 14 tracked Python files and 3464 physical lines under `server/`. These are additional caller dependencies, not mesh library source.
