# Memorandum: functional compilation

Mesh’s required architecture represents allocation, lifetimes, placement, dependencies, and schedules as explicit data because these are solved ahead of execution and must remain available to later transformations.
Producers publish complete results in paged storage, and consumers access those same results by view, preserving their identity and page-table references.
Compilation therefore composes functions of explicit operands and represented effects, with original objects or argument bundles passed through existing interfaces using `*args` and `**args` where appropriate.
This is a stronger functional discipline than ordinary Python programming: every agent must recognize copying or reconstructing existing results as a departure from the specification and correct its work.

# Memorandum: access by view

Every descendant function accesses produced structures through views of the same storage, with their page-table references carried intact.
Passing a view preserves access to the complete structure while each consumer dereferences exactly what its algorithm uses.
Packing, unpacking, or selecting fields into smaller replacement structs along that chain is unnecessary because the complete structure is never passed by copy.
Throughout this integration, agents use access by view to avoid spending memory bandwidth and stack work on duplicated representations, especially across Apple Silicon’s bandwidth-limited interconnect.

# Frontend / RDMA Python and shell callback sweep — 15 November 2026

Codex completed the source sweep directly after the operator authorized Codex completion of the held Claude slice. No Claude request, dependency installation, import, native build, handler execution or workload was used. Source baseline: `abba01f8401d94acdf40267f94144795be883244`. The inventory surveys **74 Python paths and 33 shell paths**; three Python paths and two shell safeguards are explicit exclusions below, leaving 71 Python and 31 ordinary shell paths admitted to source review. Vendor/dependency trees, generated/output_data code, compiler peer files and native/Swift files are excluded. Mesh-authored Xonotic tooling is included.

Final source contains **468 Python marker references in 46 files**, plus **6 shell markers in four files**. This includes the 16 previously committed Python markers. The current checkpoint changes 45 Python files and four scripts. Counts are source invalidation sites, not unique callback functions: a native callback can be marked both at construction and in its body.

## Actual callback families and exclusions

- Native outward callbacks: `RELEASE(_DECREF)`, `@RELEASE`, `@DELETER`, DLPack deleter, borrowed-storage releases, relation backing release, `ROW_KEY`, `EXTENT`, and `HOSTED(sent/received/copied)` fail in ordinary Python before crossing the ABI. Their original operands and bodies remain. `RELEASE()` is a null pointer and remains unchanged. CFUNCTYPE/PYFUNCTYPE declarations and CAPSULE_NEW/GET/PUT/_INCREF/_DECREF direct C-address wrappers are not callbacks by themselves.
- Graph/IR callbacks: adjacency/forward neighbour and transfer functions, view readers, FX arg transforms/Interpreter hooks, Rule.configure, actual registered lowering handlers, pytree callbacks and cost/placement functions. Represented FX/Torch operation keys remain data. Existing dead malformed IR code was surveyed and retained.
- Runtime/tool callbacks: thread targets, executor jobs, signal/HTTP hooks, weakref finalizers, sort keys, map/filter/reduce conversion functions, JSON/argparse/dataclass factories, property getters, regex replacement functions, cached asset lookup and asset geometry functions. Their phase is stated at each source use; compile-time callbacks are included. Ordinary direct functions, native API calls, and directly invoked local helper lambdas remain unmarked.
- Source programs are excluded: `Training.forward` and other numerical module/function operands supplied through export/compile/trace remain source inputs. For example, examples.parallel `compile_module(model, inputs, ...)`, rdma.mesh_compile `_exported`/`Tracer.trace(module)`, and coreml-models `ct.convert(program)` consume program data. Separate `mb.program` builder callbacks and registered fake/custom kernels are marked at their actual registrations.
- Shell EXIT/INT/TERM traps in mesh-run.sh, mesh-peers.sh, mesh-codesign-identity.sh and xonotic/payload/build.sh retain their original bodies. The undefined command emits the shell error, then `exit $?` preserves its failing status before the original handler body. Syntax alone was checked. Operational review/shell/kill safeguards and historical measurement artifacts remain byte-identical.

## Complete reviewed-file inventory

| File | Marker references | Disposition / source consumers |
|---|---:|---|
| `bench/gpu_gemm.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bench/meanfield.py` | 1 | `solve` |
| `bench/moe.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bench/moe_dispatch.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bench/moe_fast.py` | 3 | `bench` |
| `bench/moe_final.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bench/moe_shapes.py` | 2 | `bench` |
| `bench/orth.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bench/pairwise_distance.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bench/peak.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bench/prec.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bench/residual.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bench/residual2.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bench/sparse.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-application.py` | 4 | `command`, `deploy`, `main` |
| `bin/mesh-brief-gate.py` | 0 | Operational agent-assignment review gate unchanged. |
| `bin/mesh-item-tally.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-observe.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-post-commit.py` | 0 | Operational post-commit review: noted review callbacks unchanged. |
| `bin/mesh-runtime-id.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-stop.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-sync-monitor.py` | 2 | `main`, `trace_log` |
| `bin/mesh-told.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-top.py` | 21 | `Top.__init__`, `Top.bridge_link`, `Top.frame`, `Top.frame.both`, `Top.refresh`, `main`, `start_probe` |
| `bin/mesh_notes.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/strip-source-comments.py` | 3 | `html_text`, `xml_text` |
| `compiler/mesh_compiler/export.py` | 1 | `substitute` |
| `compiler/mesh_compiler/ir.py` | 11 | `evaluated`, `recv`, `send`, `send_fake`, `views` |
| `examples/coreml-models.py` | 15 | `activation.program`, `main`, `projection.program`, `residual.program` |
| `examples/cp.py` | 1 | `main` |
| `examples/crossing.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `examples/engines.py` | 1 | `arguments` |
| `examples/ep.py` | 1 | `main` |
| `examples/parallel.py` | 16 | `Eager.forward`, `arguments`, `chosen`, `execute`, `main` |
| `examples/pl.py` | 1 | `main` |
| `examples/tp.py` | 13 | `Eager.forward`, `Timed.run_node`, `arguments`, `chosen`, `execute`, `main`, `units` |
| `measurements/training-continuation-20260906/check-viewer.py` | 0 | Historical measurement/evidence artifact, excluded from maintained source. |
| `rdma/allocate.py` | 7 | `equal_finish`, `min_max`, `shift` |
| `rdma/mesh.py` | 2 | `extents` |
| `rdma/mesh_code.py` | 2 | `_shape` |
| `rdma/mesh_compile.py` | 171 | `<module>`, `Tracer.is_leaf_module`, `_Builder.attach`, `_Export.export`, `_Export.fake`, `_Export.fake.make`, `_Export.marked`, `_Export.run_node`, `_Program.run_node`, `_Symbolic.call_function`, `_Symbolic.call_method`, `_Symbolic.call_module`, `_Symbolic.get_attr`, `_actors`, `_adaptive_max`, `_all_to_all`, `_analyse`, `_archive_loaded`, `_attention`, `_attention_rule`, `_blelloch`, `_bound`, `_called`, `_collected`, `_collective_permute`, `_compared`, `_complete`, `_complex_fold`, `_convolution`, `_convolving`, `_counted`, `_decomposed`, `_delete`, `_dlpack_tensor`, `_dropped`, `_einsum`, `_epilogue`, `_exchange`, `_factory_rule`, `_factory_rule.configure`, `_finalized`, `_fold`, `_formed`, `_forms`, `_geqrf_metadata`, `_groups`, `_hosted`, `_initial_values`, `_instantiate`, `_joined`, `_lent`, `_lifted`, `_lowering`, `_lstsq_metadata`, `_matmul`, `_merged.configure`, `_meta`, `_metadata_asserted`, `_named`, `_norm`, `_operator`, `_outputs`, `_outputs.tensor`, `_over`, `_over.rule`, `_owned`, `_pad`, `_partition`, `_placed_buffer`, `_places`, `_pooling`, `_programs`, `_programs.call`, `_reduce_scatter`, `_reduced`, `_reduction`, `_released_constant`, `_replays`, `_restore_programs`, `_rewrite`, `_runner`, `_runner.save`, `_runner.timed`, `_runner.writable`, `_scan`, `_scanned`, `_selected`, `_sizes`, `_slice`, `_specialize`, `_specialized`, `_statistics`, `_stripe`, `_variance`, `_view`, `_view_rule`, `_wait`, `lower`, `partition`, `program`, `program._planned`, `shard` |
| `rdma/mesh_coreml.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `rdma/mesh_disk.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `rdma/mesh_graph.py` | 30 | `Context.__init__`, `Graph.adjacency`, `Graph.boundaries`, `Graph.edges`, `Graph.longest`, `Lifetimes.first`, `Lifetimes.last`, `Owned.__init__`, `Relation.__init__`, `Relation.edges`, `Relation.load`, `Relation.members`, `Relation.redundant`, `View.__getitem__`, `View.__iter__`, `adjacent`, `arrivals`, `closure_chains`, `components`, `forward`, `ordered`, `orderings`, `paths`, `rows` |
| `rdma/mesh_hourly.py` | 2 | `<module>`, `record` |
| `rdma/mesh_metal4.py` | 11 | `Function.__init__`, `Function.attempt`, `Function.copied`, `Function.pages`, `Function.received`, `Function.sent` |
| `rdma/mesh_preflight.py` | 7 | `beacons`, `discover.gather`, `main` |
| `rdma/peers.py` | 8 | `Node`, `Node.ranked`, `addresses`, `observe_mesh`, `realize`, `topology` |
| `rdma/workload.py` | 2 | `WorkloadMeter.publish_measures`, `WorkloadSpan.start` |
| `user/mesh-nodeinfod.py` | 1 | `<module>` |
| `user/mesh-telemetry.py` | 10 | `<module>`, `TelemetryHTTPServer.handle_error`, `TelemetryHTTPServer6.server_bind`, `TelemetryHandler.do_GET`, `TelemetryHandler.do_POST`, `TelemetryHandler.log_message`, `processes`, `serve` |
| `viz/serve.py` | 9 | `<module>`, `Handler.__init__`, `Handler.do_GET`, `Handler.log_message`, `SSHHTTPConnection.close`, `SSHHTTPConnection.connect`, `hosts`, `poll` |
| `xonotic/mapgen/nav.py` | 2 | `Body.height` |
| `xonotic/mapgen/spiralgen.py` | 31 | `Spiral.build`, `build_parser`, `strip`, `write_svg`, `write_svg.poly` |
| `xonotic/payload/runtime-package.py` | 3 | `main` |
| `xonotic/payload/tools/curve_reconcile.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `xonotic/payload/tools/fusegraph.py` | 7 | `load`, `navmesh_metrics`, `region_of`, `region_solve` |
| `xonotic/payload/tools/fusemeasure.py` | 1 | `main` |
| `xonotic/payload/tools/havocstate.py` | 11 | `clean`, `coordinates`, `main`, `source_inventory` |
| `xonotic/payload/tools/joinshot.py` | 5 | `main` |
| `xonotic/payload/tools/joinview.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `xonotic/payload/tools/mapfuse.py` | 8 | `hall_shell`, `leaf_assignment`, `main`, `navmesh_measure`, `transfer_site` |
| `xonotic/payload/tools/mapgen.py` | 3 | `Gen.gen_nav`, `bridge_tile`, `bridge_tile.snap` |
| `xonotic/payload/tools/mapsrc.py` | 4 | `Patch.translate`, `asset_paths`, `quake_texture_vectors` |
| `xonotic/payload/tools/mkentfile.py` | 6 | `balanced_spawn_points`, `localized_points`, `spawn_overlay`, `spread_points` |
| `xonotic/payload/tools/navmesh.py` | 7 | `components`, `kcenter`, `place_carts`, `plan_spans` |
| `xonotic/payload/tools/negspace.py` | 5 | `NegSpace._solid_bvh_relation_rows`, `NegSpace._solid_ray_relation_rows`, `NegSpace._solid_relation_rows`, `_indexed_face_pairs` |
| `xonotic/payload/tools/placement.py` | 7 | `map_sites`, `region_graph_solve`, `walk_nodes` |
| `xonotic/payload/tools/qcdump.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `xonotic/payload/tools/strategy_io_schema.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `xonotic/planner/plan.py` | 7 | `main` |
| `xonotic/render/frame-report.py` | 1 | `main` |
| `xonotic/render/glsl-pack.py` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `xonotic/render/pbr-materials.py` | 2 | `main` |
| `bench/orth_final.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bench/orth_sweep.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bench/orth_tune.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-beacon.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-bridge.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-codesign-identity.sh` | 1 | `trap EXIT` |
| `bin/mesh-devtools-init.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-fabric-init.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-keeper.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-kill-guard.sh` | 0 | Operational kill safeguard unchanged. |
| `bin/mesh-networks.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-nodeinfo.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-peers.sh` | 2 | `trap EXIT`, `trap INT TERM` |
| `bin/mesh-q3map2-build.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-rdma-init.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-router-init.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-run.sh` | 2 | `trap EXIT`, `trap INT TERM` |
| `bin/mesh-runtime-install.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-shell-guard.sh` | 0 | Operational chpwd/zle safeguard unchanged. |
| `bin/mesh-status.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bin/mesh-update.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `bootstrap.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `enable-autologin.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `install-user.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `install.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `serve.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `user/mesh-update-user.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `xonotic/mapgen/build-mesh256.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `xonotic/mapgen/build.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `xonotic/payload/build.sh` | 1 | `trap EXIT` |
| `xonotic/render/bench.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `xonotic/render/build-client.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |
| `xonotic/render/plc-run.sh` | 0 | Reviewed; no supplied callback use found in this file. Direct functions, source operands and data remain unchanged. |

## Exact marked-site index

Each entry names the current source line and its original containing function or registration scope. The source comment records the actual callback interface/phase. Counts do not rest on a regex-only survey: the static AST candidate inventory was checked against the actual passing, registration and invocation sites.

- `bench/meanfield.py`: 25 (solve).
- `bench/moe_fast.py`: 58 (bench), 58 (bench), 64 (bench).
- `bench/moe_shapes.py`: 13 (bench), 19 (bench).
- `bin/mesh-application.py`: 21 (command), 65 (deploy), 69 (deploy), 133 (main).
- `bin/mesh-sync-monitor.py`: 256 (trace_log), 385 (main).
- `bin/mesh-top.py`: 199 (Top.__init__), 202 (Top.__init__), 208 (Top.refresh), 290 (Top.bridge_link), 323 (Top.frame), 331 (Top.frame), 367 (Top.frame), 370 (Top.frame), 377 (Top.frame.both), 382 (Top.frame), 384 (Top.frame), 384 (Top.frame), 385 (Top.frame), 446 (Top.frame), 460 (Top.frame), 464 (Top.frame), 534 (start_probe), 540 (main), 543 (main), 548 (main), 575 (main).
- `bin/strip-source-comments.py`: 191 (html_text), 195 (html_text), 200 (xml_text).
- `compiler/mesh_compiler/export.py`: 121 (substitute).
- `compiler/mesh_compiler/ir.py`: 30 (send), 34 (send), 40 (send_fake), 44 (send_fake), 50 (recv), 54 (recv), 229 (evaluated), 399 (views), 404 (views), 419 (views), 423 (views).
- `examples/coreml-models.py`: 24 (projection.program), 28 (projection.program), 38 (activation.program), 42 (activation.program), 52 (residual.program), 56 (residual.program), 66 (main), 69 (main), 72 (main), 75 (main), 78 (main), 81 (main), 84 (main), 87 (main), 90 (main).
- `examples/cp.py`: 46 (main).
- `examples/engines.py`: 103 (arguments).
- `examples/ep.py`: 96 (main).
- `examples/parallel.py`: 114 (chosen), 120 (chosen), 125 (chosen), 131 (chosen), 147 (chosen), 147 (chosen), 153 (chosen), 166 (arguments), 170 (arguments), 174 (arguments), 178 (arguments), 181 (arguments), 184 (arguments), 212 (Eager.forward), 221 (execute), 258 (main).
- `examples/pl.py`: 109 (main).
- `examples/tp.py`: 100 (units), 113 (Timed.run_node), 141 (chosen), 158 (chosen), 161 (chosen), 165 (chosen), 178 (arguments), 182 (arguments), 186 (arguments), 190 (arguments), 210 (Eager.forward), 218 (execute), 243 (main).
- `rdma/allocate.py`: 67 (equal_finish), 79 (min_max), 79 (min_max), 83 (min_max), 105 (shift), 108 (shift), 108 (shift).
- `rdma/mesh.py`: 361 (extents), 361 (extents).
- `rdma/mesh_code.py`: 16 (_shape), 20 (_shape).
- `rdma/mesh_compile.py`: 190 (_sizes), 247 (shard), 377 (_named), 395 (_matmul), 415 (_einsum), 419 (_einsum), 459 (_convolution), 488 (_pooling), 512 (_pad), 523 (_slice), 532 (_slice), 540 (_slice), 550 (_slice), 569 (_view_rule), 591 (_attention_rule), 600 (_factory_rule), 610 (_factory_rule.configure), 617 (_factory_rule), 626 (_operator), 629 (_operator), 721 (_merged.configure), 721 (_merged.configure), 734 (_complete), 1019 (_partition), 1122 (_stripe), 1127 (_stripe), 1166 (partition), 1167 (partition), 1179 (partition), 1184 (partition), 1185 (partition), 1236 (_forms), 1236 (_forms), 1243 (_meta), 1248 (_meta), 1266 (_formed), 1275 (_formed), 1323 (_over.rule), 1337 (_over), 1393 (_groups), 1515 (_view), 1536 (_places), 1537 (_places), 1633 (_exchange), 1634 (_exchange), 1642 (_exchange), 1650 (_exchange), 1660 (_all_to_all), 1661 (_all_to_all), 1676 (_reduce_scatter), 1677 (_reduce_scatter), 1700 (_collective_permute), 1701 (_collective_permute), 1757 (lower), 1929 (_wait), 1932 (_wait), 1935 (_wait), 3180 (_epilogue), 3213 (_joined), 4055 (_geqrf_metadata), 4059 (_geqrf_metadata), 4062 (_geqrf_metadata), 4070 (_lstsq_metadata), 4074 (_lstsq_metadata), 4078 (_lstsq_metadata), 4082 (_lstsq_metadata), 4086 (_lstsq_metadata), 4091 (_lstsq_metadata), 4095 (_lstsq_metadata), 4101 (_lstsq_metadata), 4775 (_convolving), 4895 (_adaptive_max), 5202 (_reduction), 5224 (_reduced), 5251 (_fold), 5269 (_complex_fold), 5276 (_complex_fold), 5347 (_variance), 5382 (_statistics), 5433 (_blelloch), 5443 (_blelloch), 5475 (_scan), 5536 (_scanned), 5675 (_selected), 6257 (_attention), 6440 (_counted), 6561 (_norm), 6562 (_norm), 6670 (_lifted), 6689 (_lowering), 6696 (_lowering), 6696 (_lowering), 6701 (_lowering), 6707 (_lowering), 6714 (_lowering), 6719 (_lowering), 6732 (_lowering), 6732 (_lowering), 6736 (_lowering), 6795 (_delete), 6799 (_delete), 6818 (_placed_buffer), 6836 (_dlpack_tensor), 6919 (_called), 6922 (_called), 6926 (_called), 7016 (_collected), 7027 (_collected), 7032 (_collected), 7048 (_collected), 7051 (_collected), 7064 (_hosted), 7088 (_initial_values), 7092 (_initial_values), 7093 (_initial_values), 7096 (_initial_values), 7097 (_initial_values), 7127 (_specialized), 7146 (_specialized), 7187 (_instantiate), 7238 (_owned), 7251 (_dropped), 7269 (_released_constant), 7274 (_released_constant), 7335 (_Builder.attach), 7673 (_programs.call), 7708 (_programs), 7742 (_actors), 7807 (_finalized), 7902 (_replays), 7968 (_outputs.tensor), 7977 (_outputs), 8002 (_specialize), 8002 (_specialize), 8051 (Tracer.is_leaf_module), 8075 (_rewrite), 8091 (_Symbolic.call_function), 8097 (_Symbolic.call_method), 8103 (_Symbolic.call_module), 8109 (_Symbolic.get_attr), 8143 (_analyse), 8151 (_analyse), 8201 (_analyse), 8224 (_Program.run_node), 8229 (_Program.run_node), 8296 (_metadata_asserted), 8337 (_decomposed), 8395 (_Export.marked), 8405 (_Export.fake.make), 8410 (_Export.fake), 8415 (_Export.run_node), 8430 (_Export.run_node), 8455 (_Export.export), 8479 (_Export.export), 8519 (_bound), 8529 (_bound), 8638 (_compared), 8680 (program), 8790 (program._planned), 8965 (<module>), 8973 (_lent), 8979 (_lent), 8991 (_archive_loaded), 9008 (_restore_programs), 9027 (_restore_programs), 9039 (_restore_programs), 9107 (_runner.timed), 9123 (_runner.writable), 9152 (_runner.save), 9156 (_runner), 9162 (_runner).
- `rdma/mesh_graph.py`: 27 (Owned.__init__), 43 (View.__getitem__), 43 (View.__getitem__), 48 (View.__iter__), 55 (ordered), 185 (Graph.adjacency), 276 (Graph.longest), 287 (Graph.boundaries), 287 (Graph.boundaries), 301 (Graph.edges), 376 (forward), 376 (forward), 417 (Lifetimes.first), 421 (Lifetimes.first), 426 (Lifetimes.last), 430 (Lifetimes.last), 448 (paths), 466 (closure_chains), 741 (components), 755 (rows), 755 (rows), 1010 (Context.__init__), 1036 (Relation.__init__), 1054 (Relation.redundant), 1060 (Relation.edges), 1066 (Relation.members), 1101 (Relation.load), 1139 (orderings), 1155 (arrivals), 1165 (adjacent).
- `rdma/mesh_hourly.py`: 64 (<module>), 70 (record).
- `rdma/mesh_metal4.py`: 147 (Function.__init__), 147 (Function.__init__), 147 (Function.__init__), 159 (Function.pages), 173 (Function.attempt), 182 (Function.sent), 186 (Function.sent), 191 (Function.received), 195 (Function.received), 200 (Function.copied), 204 (Function.copied).
- `rdma/mesh_preflight.py`: 151 (beacons), 186 (discover.gather), 218 (main), 224 (main), 240 (main), 268 (main), 295 (main).
- `rdma/peers.py`: 56 (Node), 61 (Node.ranked), 107 (addresses), 182 (observe_mesh), 197 (topology), 213 (topology), 239 (topology), 276 (realize).
- `rdma/workload.py`: 37 (WorkloadMeter.publish_measures), 168 (WorkloadSpan.start).
- `user/mesh-nodeinfod.py`: 27 (<module>).
- `user/mesh-telemetry.py`: 205 (processes), 634 (TelemetryHandler.do_GET), 668 (TelemetryHandler.do_POST), 686 (TelemetryHandler.log_message), 702 (TelemetryHTTPServer.handle_error), 712 (TelemetryHTTPServer6.server_bind), 726 (serve), 732 (serve), 747 (<module>), 750 (<module>).
- `viz/serve.py`: 90 (hosts), 120 (SSHHTTPConnection.connect), 138 (SSHHTTPConnection.close), 418 (poll), 445 (Handler.__init__), 449 (Handler.do_GET), 470 (Handler.log_message), 478 (<module>), 482 (<module>).
- `xonotic/mapgen/nav.py`: 21 (Body.height), 25 (Body.height).
- `xonotic/mapgen/spiralgen.py`: 86 (strip), 150 (Spiral.build), 156 (Spiral.build), 162 (Spiral.build), 169 (Spiral.build), 173 (Spiral.build), 267 (write_svg), 267 (write_svg), 278 (write_svg.poly), 329 (build_parser), 334 (build_parser), 337 (build_parser), 340 (build_parser), 343 (build_parser), 346 (build_parser), 349 (build_parser), 352 (build_parser), 355 (build_parser), 358 (build_parser), 361 (build_parser), 364 (build_parser), 367 (build_parser), 373 (build_parser), 377 (build_parser), 380 (build_parser), 383 (build_parser), 386 (build_parser), 389 (build_parser), 392 (build_parser), 395 (build_parser), 398 (build_parser).
- `xonotic/payload/runtime-package.py`: 40 (main), 43 (main), 46 (main).
- `xonotic/payload/tools/fusegraph.py`: 15 (load), 36 (region_of), 85 (region_solve), 120 (region_solve), 153 (navmesh_metrics), 159 (navmesh_metrics), 159 (navmesh_metrics).
- `xonotic/payload/tools/fusemeasure.py`: 119 (main).
- `xonotic/payload/tools/havocstate.py`: 18 (clean), 54 (source_inventory), 57 (source_inventory), 61 (source_inventory), 94 (coordinates), 116 (main), 119 (main), 122 (main), 125 (main), 146 (main), 163 (main).
- `xonotic/payload/tools/joinshot.py`: 326 (main), 329 (main), 332 (main), 336 (main), 347 (main).
- `xonotic/payload/tools/mapfuse.py`: 31 (transfer_site), 36 (transfer_site), 157 (leaf_assignment), 513 (hall_shell), 872 (navmesh_measure), 1658 (main), 1669 (main), 1672 (main).
- `xonotic/payload/tools/mapgen.py`: 171 (Gen.gen_nav), 262 (bridge_tile.snap), 273 (bridge_tile).
- `xonotic/payload/tools/mapsrc.py`: 85 (Patch.translate), 231 (quake_texture_vectors), 563 (asset_paths), 567 (asset_paths).
- `xonotic/payload/tools/mkentfile.py`: 196 (spread_points), 209 (localized_points), 276 (balanced_spawn_points), 287 (balanced_spawn_points), 381 (spawn_overlay), 409 (spawn_overlay).
- `xonotic/payload/tools/navmesh.py`: 205 (components), 211 (kcenter), 217 (kcenter), 224 (kcenter), 355 (plan_spans), 760 (place_carts), 790 (place_carts).
- `xonotic/payload/tools/negspace.py`: 455 (_indexed_face_pairs), 667 (NegSpace._solid_bvh_relation_rows), 680 (NegSpace._solid_bvh_relation_rows), 700 (NegSpace._solid_relation_rows), 715 (NegSpace._solid_ray_relation_rows).
- `xonotic/payload/tools/placement.py`: 336 (map_sites), 341 (map_sites), 348 (map_sites), 414 (map_sites), 432 (walk_nodes), 437 (walk_nodes), 643 (region_graph_solve).
- `xonotic/planner/plan.py`: 48 (main), 51 (main), 54 (main), 57 (main), 62 (main), 65 (main), 101 (main).
- `xonotic/render/frame-report.py`: 31 (main).
- `xonotic/render/pbr-materials.py`: 107 (main), 110 (main).
- `bin/mesh-codesign-identity.sh`: 12 (trap EXIT).
- `bin/mesh-peers.sh`: 13 (trap EXIT), 13 (trap INT TERM).
- `bin/mesh-run.sh`: 42 (trap EXIT), 42 (trap INT TERM).
- `xonotic/payload/build.sh`: 12 (trap EXIT).

## Source-only verification and limits

All 72 syntactically valid surveyed Python files parse without imports. `examples/tp.py` and `rdma/mesh_compile.py` retain six pre-existing missing-`def` header lines exactly; an in-memory-only normalization allowed AST comparison of their remaining original bodies. Removing only undefined marker expressions/calls yields the same AST as baseline for all 74 Python files. Removing only trap comments and marker/error-status prefixes yields byte-identical baseline scripts. The four edited scripts pass `sh -n`; scoped diff whitespace checks pass. The marker has no definition, assignment or import in this slice.

The bounded owned-source sweep is complete. Foreign framework/native callback implementations remain their original owners; this slice invalidates their actual Mesh-side uses where present. Runtime-loaded third-party plugins and generated callables cannot be enumerated from these files and are not claimed as additional inspected source. No remaining uncertain owned registration was deferred. Original capture/compiler implementation work remains paused; these deliberate failures are not a working compiler or runtime release.
