# Production execution and dataflow audit, 2026-09-06

Follow-up: [static flow refactor](../policy-static-flow-20260906/README.md) consolidates ingress, emission and checkpoint interpretation, shares packed inputs and persists partial transport state. This record describes the preceding review.

This is a source audit of the active responder, model, learner, RPC worker, native view adapter, and reporting path. It records defects found before this cleanup and the corresponding changes now present. No test suite, substitute verification harness, service launch, remote workload, or backend primitive modification was used for this audit.

## Executed path

1. `strat_responder.make_policy` constructs one model per policy arm. Matrix arms use `Wally`; linear and FFN comparison arms use `BaselinePolicy`; the native default has no parameters. The two matrix ablations change fusion strength, retaining the same operator implementations.
2. Native `mesh_ipc.c:VM_mesh_view_publish` exports every retained view page. Each row contains the page identity/header and six 256-word arrays: observed state, current integrated residual, typed presence/layout, last read times, held velocity, and command residual. `state_steering.unpack_pages` preserves the page identities and all six arrays. The response remains a page identity plus command residual and velocity, not a candidate index.
3. The responder combines the page rows with literal OBS/CART/TEAM/EVENT rows. EVENT18 contains the native deposit-time episode in column 17. `RuntimeFrames` preserves each event batch's actual session, tick, and request identity; `ObservationMemory` retains histories by source session and literal deposited episode. Only the current episode's rows enter the model. `NavigationRows.prepare` supplies node/edge/cell rows and spatially restricted neighborhood indices, distances, support masks, and radii. `inputs.assemble` owns the resulting 30-field `ChorusArrays` contract; the unused duplicate `relations` field was removed.
4. `PolicyProgram.prepare` realizes sufficient capacity, pads the frame, and wraps its arrays for MLX. `strategy.embedded_rows` gives every literal row family its first learned projection. Its common row axis is `OBS + participant state pages + CART + TEAM`; events and navigation rows enter the learned sparse neighborhood operator before they affect these actor rows.
5. `encode_source` produces the row projection, local message, presence, team factors, rival factors, and normalized value rows. The global contraction is `rival_factors.T @ values`. `encode_finish` applies the global and team row interactions. The same already-computed factors are reused for mixing and factor reporting.
6. The row-local routed MoE calls the same `cast_header.scale_operator` locally, through RPC, and in the RPC fallback. `decode` applies the final shared SwiGLU to form H. `read_heads` is the sole learned head application for matrix and comparison policies: one projection of H produces state-rate coordinates and both value outputs. `Strategy.ir` is the exact owner OBS plus page H read by these heads.
7. `PolicyProgram._emit` samples the full Gaussian velocity and computes its likelihood and exact exponential residual. The responder selects the assigned policy's row for each bot, journals that issued velocity, and sends `StatePages.response`. Counterfactual policies run on the same input and random key and retain their full rate distributions and sampled outputs.
8. Native `MeshX_ViewConsume` installs residual and forcing into the bot-owned view pages. `PRVM_ViewTime` calculates the exact exponential hold and tail. `PRVM_ViewReadFast` adds the residual only to the value returned to the bot; it also records the literal source state and read time. Integer views round the integrated residual to an integer correction. This is an actuator quantization boundary, not another trainable model.
9. `ActionHistory.advance` joins observed application sequences to the exact issued decision. `OnlineLearner._tensor_loss` reevaluates current and successor source frames, uses the full-vector action likelihood, and learns winner/loser values. AdamW owns the single complete parameter tree for each arm. The worker has no optimizer or independent policy parameters.
10. The same `ActionHistory.source_features` extraction is used for live source reporting and for checkpoint/spill restoration. It enumerates every literal `ChorusArrays` coordinate, including structural masks and page/neighborhood indices. One vector and one label tuple are shared by all decision records from that frame. J output comes from the controlling policy's actual H, with unused prepared page columns removed.

## Every Strategy field and its consumer

All fields pass through `PolicyProgram.crop`. This table distinguishes a real numerical/report consumer from that structural copy.

| Field | Real consumer | Status |
| --- | --- | --- |
| `rate.mean` | Emission, full-vector likelihood, policy comparison | Retained |
| `rate.log_scale` | Sampling, likelihood, policy comparison | Retained |
| `rate.density` | Native-default versus Gaussian distribution, likelihood and comparison | Retained |
| `rate.present` | Exact word support for sampling, learning and comparison | Retained |
| `ir` | Responder source decisions, J artifacts and viewer row output | Retained; now the same final H read by action and value heads |
| `value_winnie`, `value_lou` | Current value errors, successor bootstrap, actor advantage, policy comparison | Retained |
| `coupling` | Responder factor record; `joracle.display.page_model` and viewer focus | Retained report-only; no feedback into input featurization |
| `local_neighborhood` | Responder `model.local_neighborhood`, field coverage and sampled viewer model | Retained report-only; the internal and product labels both name the actual local learned message |
| `scale_residual_stats` | Per-arm `moe.residual_stats` in telemetry and workload measurements | Retained; previously no live consumer beyond crop |
| `scale_expert_load` | Per-arm `moe.expert_load` in telemetry and workload measurements | Retained; previously no live consumer beyond crop |
| `scale_balance` | Ordinary balancing loss and per-arm MoE telemetry | Retained |
| Former `query` | No consumer beyond crop | Deleted |
| Former `aux_winnie`, `aux_lou` | Earlier authorized VERA self-distillation, superseded by the newer common-IR instruction | Deleted with their parameters and loss component |
| Former integrated-mean `residual` | No consumer beyond crop; live action/report used independently sampled integration | Deleted; sampled integration remains in emission |

## Defects corrected in this pass

- The old action readout had private learned state/residual projections after the encoder. Action rates could therefore use raw-state features absent from the displayed IR and value probes. The old final SwiGLU was also private to the action route. Both routes were removed in favor of the common H and one shared head projection.
- Rival factors were projected twice across `encode_source` and `row_gram_context`. The source factors now cross the execution boundary once and are reused. Orphan earlier participant Gram helpers were removed by the model cleanup.
- Whole-model output included unused query, auxiliary values, and an integrated mean action that was immediately recomputed from the real sampled action. The output contract now has nine fields.
- Cropping trimmed only the participant dimension of IR. Prepared page capacity leaked into J as artificial zero columns. Cropping now removes prepared page padding before the responder labels the actual page axis. Coupling uses the same page-axis convention.
- `RemoteCross.sample` was initialized false and never assigned; `measure_local_counterfactual` had no caller. Its pending arrays, timing state, local rerun path, and duplicate `RemoteScale` branch were deleted. The live comparison between different policies remains.
- Live feature reporting hand-concatenated a partial subset separately for every bot. It omitted other bots' pages and neighborhood structural inputs. Checkpoint and spill restoration then replaced even that subset with only the bot's state words. One generic extraction now supplies the same full source context before and after restore. Shared feature objects are counted once in history memory accounting.
- J's secondary source lens treated a learned neighborhood output, formal game hierarchy, and a new one-hot team encoding as input features. It now uses the actual source coordinates. Historical frames without that source record expose only their recorded raw state, rather than inventing missing inputs.
- Inference history previously had no retained source Frame. Inference now keeps the same literal Frame structure needed for reporting restoration.
- MoE load and residual statistics were calculated but never reached local telemetry. Every response now emits each arm's statistics, loads, and balancing loss; the viewer series and workload record receive them too.
- Region telemetry counted every Python encode call as a compilation trace while counting source/finish/decode traces differently. Only actual compiled-region trace counters are now reported.
- New-capacity preparation packed its first source twice and ran a second mean integration. The prepared first frame is reused and the redundant integration was removed.
- Comparison policies allocated the neighborhood Gram metric even though their static linear-neighborhood path never used it. `LocalNeighborhood(..., gram=False)` now omits that trainable matrix during model construction.
- Response timing previously omitted preparation and included source-report construction; comparison timing included optimization. Response now measures preparation through native send, with preparation/inference/source-report intervals separately named. Comparison timing covers `compare_policies` alone.

## Event ownership across rounds and restarts

The previous EVENT17 row had no episode identity. A delayed event from an earlier round was appended to whichever episode happened to be current when the batch arrived. Event batches were also concatenated in `RuntimeFrames.take`, erasing their session boundaries; retired-session events were discarded, and replay removed every queued batch below a tick watermark rather than only the recorded batches.

The completed path now preserves ownership at each boundary:

- The native producer stamps `payload_episode` when it deposits an EVENT18 row, before queueing or transport. That literal column reaches the event projection intact.
- Each interleaved EVENT batch assembles under its full `(session, tick, request)` key. Completed batch mappings reach the responder without concatenating away session metadata. Events from retired sessions remain available; future-tick batches remain pending until an eligible snapshot is consumed. Normal supersession of older state snapshots does not discard their event records.
- Memory partitions each eligible batch by its literal deposited episode and source session. Switching the current episode changes the selected history; it does not clear old histories. A future-episode row carried in an eligible batch remains in that future history until its episode is selected. Appends retain complete raw rows and their order within each history.
- Checkpoints serialize every retained history and unattributed legacy record, alongside pending transport batches and exact event identities. The step journal stores the complete event-batch mapping. Replay deduplicates only its exact recorded keys, so an additional delayed batch below the same tick watermark remains available.
- `realized_events` records each event's actual `episode_id`; `event_arrivals` retains the delivered raw batch rows and their original transport identity. Memory telemetry reports arrivals and retained row counts by episode. Current-episode J and round-result joins require matching episode ownership, preventing old-session request-number reuse from falsely joining a delayed event to a new action. Viewer round keys include the event episode and use recorded context for that episode instead of the controllers of the arrival round.

Limits are explicit. EVENT17 and old snapshots without reliable per-row episode ownership remain preserved as unattributed raw history; the code does not invent their episode from arrival time. Those rows do not enter a known episode's policy context. Late historical events remain reportable, but missing historical action/controller context is not reconstructed from the current round. The change preserves all retained histories in memory/checkpoints and does not introduce an eviction policy. Incomplete transport fragments remain transport state and are not represented as complete observed events.

## Remaining execution boundaries, stated precisely

- `inputs.pad_frame` still allocates NumPy buffers and `PolicyProgram.pack` still creates MLX arrays on every source preparation. The responder packs the same frame separately for each policy. This does not satisfy the stronger persistent-buffer interpretation of fixed execution shapes. Fixing it requires a shared prepared-frame/buffer owner; it is not solved by the first-frame reuse above.
- Capacity is rounded upward independently for ten row dimensions. A new capacity triggers a new traced model graph and preparation work. The custom matrix kernels use fixed tile/page extents, but the entire application is not one permanently allocated graph.
- Local inference and grouped training compile separate executions of the same model functions. This is graph duplication for distinct inference/backward/update programs, not separate model algebra. The staged RPC route necessarily has source/finish/decode boundaries in the current transport interface.
- RPC currently creates the custom-function/VJP wrapper at each call, converts tensors through host arrays, and sends all MoE parameter matrices with each request. `scale_rpc.compute` rebuilds MLX arrays and executes the shared operator eagerly at the worker. This is not resident remote parameter or graph ownership.
- `_tensor_loss` reevaluates a complete successor policy even though only successor values are consumed. A local compiler can remove unused outputs; a staged RPC run still performs the remote model operations needed before those values. There is no separately hand-written value model, and introducing one would recreate the defect this cleanup removes.
- Full source attribution has a large literal coordinate count. The one shared source vector eliminates per-bot host duplication, but later statistical calculations can materialize repeated observation rows. Exact-label J strata also separate source frames as their event-history dimensions change. Neither limitation justifies silently replacing raw features with a summary.
- The native adapter reports attempted and successfully applied views through different state. Its invalid-response path continues the previous view and logs the reason. Actual application attribution must continue to use the observed applied sequence; an emitted policy vector is not proof of completed native execution.

## Small modules and reachability

| Module | Production caller | Finding |
| --- | --- | --- |
| `capacity.py` | `curriculum` uses `engine_player_capacity` for server configuration | This is source-declared engine capacity, not prepared tensor capacity. `team_capacity` and `cart_capacity` are imported but have no production callsite. |
| `row_window.py` | `ActionHistory`, `joracle.probe.LiteralJWindow` | Live module. ActionHistory retains existing request entries and separately spills by bytes; J windows use row-budget eviction. `--measure-rows` is therefore not a hard outstanding-action-memory cap. |
| `checkpoint_release.py` | Its standalone CLI only; no in-repo production caller found | Byte-preserving atomic file copy, not checkpoint feature transformation. It has no live tensor-program role. |

Primary source files: `xonotic/solver/strat/{strat_responder,execution,strategy,cast_header,baselines,state_steering,inputs,online,scale_rpc,matrix_worker,action_history,neighborhood,replay}.py`, `xonotic/solver/strat/joracle/{probe,display,server,field_measures}.py`, and `xonotic/darkplaces-work/{mesh_ipc,prvm_view}.c`.
