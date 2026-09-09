# Connection recovery audit, 2026-09-05

The game client is connected over IP. The policy transport is not delivering
observations. Reconnection exists at both the bridge and shared-memory client, but
the Mini's bridge is blocked in the kernel and the deployed implementations had
drifted. An unplugged cable was not established as the cause.

## Runtime evidence

At inspection, laptop bridge PID 65397 continued printing pairing retries. Mini
bridge PID 94759 remained in `Us`; its log had not changed since 00:40:30 PDT.
Both shared regions reported zero sent/received frames and zero `up_ms`. The
current learner reported zero observations and zero new optimizer updates.

The Mini also had hundreds of `ibv_devinfo -d rdma_en3` processes in `U`. Their
parents were orphaned `mesh-nodeinfo.sh` shells. The node-info daemon ran a fresh
shell for every request with an eight-second `subprocess.run` timeout. Killing
that shell did not retire its blocked descendants; subsequent requests created
more. This explains the probe accumulation, not the original driver wedge. A
manual diagnostic probe in this session also blocked; further verbs probes stopped.

The final observed count was 478 and stayed constant across repeated checks after
the node-info fix. SSH and gameplay remained reachable throughout the audit.

`mesh-stat`'s `up:true` currently means that a shared-memory header exists. It does
not measure bridge pairing, peer receipt, or a fresh bridge heartbeat. This is a
remaining observability defect at that inspection. The subsequent
[kernel lifecycle change](RDMA-KERNEL-RECOVERY.md) adds fresh pairing, heartbeat,
owner and operation observations without changing the shared-memory ABI.

## Recovery owners and callsites

| Owner | Existing behavior | Limit |
| --- | --- | --- |
| `rdma/mesh-flow.c`, `main` / `verbs_up` | Repeats out-of-band pairing at full configured capacity; detects port loss, peer re-pair requests and stalled completions; retires the pair and retries. | A verbs call blocked in the kernel prevents this loop and signal handlers from progressing. |
| `rdma/mesh-client.c`, `refresh` / `reattach` | Checks named-region identity every 600 pump/read/stream turns; remaps a replacement region and restores client registration. | Submitted frames are not replayed. Interrupted streams become failed. Direct callers caching arena addresses also require care when the extent changes. |
| `rdma/mesh.py`, `Mesh.read` / `pump` | Calls the C read/pump APIs used by `strat_responder.incoming`. | Reattachment is exercised by the current API; it is not application-level delivery acknowledgement. |
| `xonotic/darkplaces-work/host.c`, `MeshX_Pump` | Pumps the transport every server frame, through `mesh_ipc.c` and the C client. | Receiving future frames does not reconstruct messages lost across a connection interruption. |

An isolated shared-memory experiment passed on both machines: submit ten frames
into an eight-slot arena, replace the named region, then pump. Both queued frames
survive and a new frame reaches the replacement region. The eight previously
submitted frames are not replayed. No verbs devices were opened by this experiment.

At the initial audit, INIT/RTS results were unchecked and completion errors did
not consistently force re-pairing. The subsequent kernel lifecycle change fixes
both. A local UC send completion still does not establish peer application receipt. Terminal outcome delivery
and pending episode restoration were subsequently repaired and exercised in
[training continuation](TRAINING-CONTINUATION.md).
The keeper, status command and RDMA initialization script also invoked
`ibv_devinfo` synchronously, allowing a blocked probe to prevent later repairs.
The subsequent kernel lifecycle change replaces these with passive link checks.

## Deployment drift

The Mini's old bridge source discarded QP/CQ pointers even when destruction failed,
incremented the MR count before checking registration success, lacked the current
signal handling, and did not return submitted arena credits after pair retirement.
These fixes already existed in the laptop bridge. The exchange structure, magic
and size checks match; this audit found no evidence of an exchange-layout mismatch.

The current bridge was compiled on the Mini, and the binary and corresponding
sources were installed at its existing service path. The replacement was atomic;
the stuck process was not signaled or restarted. It will use the current build at
the next service start. Both configured region fractions remain 25 percent.

## Changes applied and validation

`mesh-nodeinfo.sh` now reads interface carrier and neighbors without opening verbs
device contexts. Its output explicitly distinguishes IP carrier from unmeasured
RDMA port state. `mesh-nodeinfod.py` runs one sampler independently of requests and
serves partial/current output with status and timestamp. A blocked sampler cannot
turn request traffic into additional probe processes.

Both deployed services returned correctly framed responses. The regression test
holds a sampler open while twenty requests complete, verifies only one sampler was
started, then releases it and verifies the completed response. It passed on the
128 GB M5 Max laptop and 24 GB M4 Pro Mini, which run different macOS versions.

The unattended-cart correction also passed 39 scoring cases, six ownership
transitions, and six player-driven movement cases including 3,000 unattended
ticks. The corrected game code was loaded when the server resumed after reboot.

At the end of the initial audit, the Mini driver wedge remained and SSH still
worked. A remote reboot had not yet been attempted, so its failure was not
established. The operator subsequently requested that test; its evidence and
outcome are tracked in `SSH-REBOOT-20260905.md`.
`RDMA-RULES.md` records an earlier incident where shutdown hung on blocked verbs
processes and physical power cycling was necessary. That is evidence of a risk,
not proof that a remote reboot of the current Mini cannot finish.
No bridge SIGKILL, forced restart, region
reduction, or substitute learning workload was used. End-to-end learning resumed
after reboot and again after the lifecycle deployment, with both bridges paired
and optimizer updates advancing. See `RDMA-KERNEL-RECOVERY.md` and
`measurements/rdma-kernel-rca-20260905.json` for those later observations.

Detailed local evidence is in `/tmp/mesh-reconnect-20260905`; cart verification is
in `/tmp/mesh-cart-hold-20260905`. Aggregate evidence is retained in
`measurements/connection-recovery-20260905.json`.
