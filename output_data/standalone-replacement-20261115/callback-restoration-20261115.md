Mesh’s required architecture represents allocation, lifetimes, placement, dependencies, and schedules as explicit data because these are solved ahead of execution and must remain available to later transformations.
Producers publish complete results in paged storage, and consumers access those same results by view, preserving their identity and page-table references.
Compilation therefore composes functions of explicit operands and represented effects, with original objects or argument bundles passed through existing interfaces using `*args` and `**args` where appropriate.
This is a stronger functional discipline than ordinary Python programming: every agent must recognize copying or reconstructing existing results as a departure from the specification and correct its work.

Every descendant function accesses produced structures through views of the same storage, with their page-table references carried intact.
Passing a view preserves access to the complete structure while each consumer dereferences exactly what its algorithm uses.
Packing, unpacking, or selecting fields into smaller replacement structs along that chain is unnecessary because the complete structure is never passed by copy.
Throughout this integration, agents use access by view to avoid spending memory bandwidth and stack work on duplicated representations, especially across Apple Silicon’s bandwidth-limited interconnect.

# Callback restoration — 2026-11-15

Operator priority, verbatim: **“extremely low priority, lower priority than e2e testing”**.

Codex restored the deprioritized callback sweep changes, preserving the original implementations. This is mechanical restoration, not approval of every original callback design or completion of the compiler integration. Historical sweep inventories and review accounts remain unchanged. Root owns the current workgroup and composition specification.

## Preserved source history

| Repository | Marker source | Restored source commit | Exact comparison |
|---|---|---|---|
| Mesh | Named eight sweep commits below | `ef651076` | All 89 affected paths equal `24481f5f` before retaining the three central exceptions; 86 remain byte-identical afterward. |
| Metal | `1703f8df`, `2c634181` | `ac2bd2ca` | All 19 affected paths byte-identical to `1dca29d4`. |
| Dawn, canonical `.build/provider/dawn` | `09457b098` | `4e94d4b0a0757cb2b915bd91e292e89fa9c7ef75` | Complete Git tree identical to `d22798336`. |
| XNNPACK, `/tmp/xnnpack-native133/XNNPACK-878f1d8ca60d7f8fb6670ab7c8cbe39a6d7415f7` | `877af3f` | `80cfa9121d17e44f33ee2b5f5f9d750d268c0fdd` | Complete Git tree identical to `55a0ae4`. |
| MLDrift, `/tmp/ml-drift-native133` | `239994d` | `3f6d27b99a5cbbe6bc389a9347894c37499c6631` | Complete Git tree identical to `576d069`. |
| LiteRT, `/tmp/litert-native133b` and canonical `.build/provider/litert` | `fbde6d03` | `cf59d4d357ec4a4d56d60d75d79a26a2304aa818` | Complete Git tree identical to `767121aa`; canonical checkout fast-forwarded to this committed result. |

Mesh source changes were reversed in the specified order: `c906de24`, `abba01f8`, `cb7dfb50`, `d9a201d2`, `fd05ee5f`, `22d5cb96`, `1beab98e`, `4e5d6729`. Each reverse application passed its exact Git application check before mutation. Documentation paths were excluded. No reset, clean, stash, source backup, or transfer of uncommitted dependency source was used.

Each restoration commit was made with ordinary `git commit -m`, including the exact priority note above in its subject. No `core.hooksPath` override, `HUSKY=0`, `--no-verify`, or other hook bypass was used. The commands returned successfully without hook diagnostics; this does not claim that every repository has a configured review hook.

## Deliberate remaining markers

The tracked maintained-source scan finds exactly these unresolved-marker occurrences:

- `rdma/mesh_metal4.py`: ten occurrences across eight source lines, exclusively the central `HOSTED` construction, `attempt`, and send/receive/copy callback bodies/invocations. Its minor `weakref.finalize` marker is removed. The retained blocks are the existing sweep blocks, not new orchestration.
- `compiler/patches/torch-inductor-aot-lowered-result.patch`: one statement in the central deferred `_emission_context` implementation, after its original docstring.
- `compiler/patches/torch-inductor-cpu-kernel-results.patch`: one statement in Mesh's `CppTemplateKernel.printed` implementation.

Both compile-time markers have explicit phase/rationale comments. Their original bodies remain intact. Removing those three-line marker/comment additions and recounting unified-diff headers restores each pre-sweep patch byte-for-byte. All other compiler patch instrumentation is restored. No maintained Metal source or restored dependency source contains the sweep's unresolved marker.

`compiler/mesh_compiler/lower_coreml.py`, `rdma/mesh_compile.py`, and `examples/tp.py` remain byte-identical to `24481f5f`, including their pre-existing malformed dead definitions. No older usable-looking headers were resurrected.

## Source and patch evidence

The full path lists, commits, residual scan and exact checks are in [callback-restoration-evidence.json](callback-restoration-evidence.json).

- 56 changed Python files parse. `examples/tp.py:121` and `rdma/mesh_compile.py:7628` retain the same pre-existing syntax errors and exact baseline bytes; these are not new failures.
- All four changed shell scripts pass `bash -n`. All nine changed compiler patches pass `git apply --numstat` structure checking. Restored non-patch source passes scoped `git diff --check`.
- `provider/dawn.patch` equals the complete restored Dawn source diff from `v20260720.160313`; `provider/litert.patch` equals the complete restored LiteRT source diff from `f5d794861b1fafbc5854050520ecb9f3226f42f6`. Both reverse-application checks pass.
- LiteRT's nested XNNPACK and MLDrift patches equal their restored source diffs from pristine `e68c629` and `d404e39`, respectively. Reverse checks pass, and canonical LiteRT has identical nested patch bytes.
- Canonical `dawn.applied` and `litert.applied` metadata were updated to the identical restored maintained patches. Build recipes were not run; no build stamp or binary was regenerated.
- A read-only scan of the 27 affected original Torch files under `/Users/mdot/.cache/uv/archive-v0/xi1x_4wDn9qX_OVO` found no callback sweep markers. No Torch patch was installed. This is not a claim that the older installed compiler matches the newer carried producer patches.

These are source restoration and syntax/patch checks only. No Claude process was resumed, and no source import, callback execution, native build or workload ran. Existing binaries have their previous source attribution; this restoration does not establish new binary/source coherence or end-to-end success.
