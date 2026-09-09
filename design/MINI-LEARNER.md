# Tethered learner placement

The September 5 runtime places training on the Mini and the dedicated game server
and renderer on the laptop. The Mini holds both policies, optimizer moments,
replay, and fixed-page Metal kernels. The model and objective remain unchanged.
Game observations and action responses cross the RDMA fabric; individual matrix
operations no longer transfer parameter snapshots and gradients between hosts.

The existing `solver.strat.curriculum` command supports this placement. Its
explicit bridge identities are `--strategy-node 1 --peer-node 0`, and its server
host is `Ms-MacBook-Pro.local`. It does not use `--distributed-scale`. Each node
retains every existing capability and its full configured bridge arena.

## Runtime ownership

| Component | Owner and durable location |
|---|---|
| Learner and curriculum | Mini system LaunchDaemon `io.mesh.cartlane.learner`, running as `mdot` |
| Policy viewer | Mini system LaunchDaemon `io.mesh.cartlane.viewer`, port 8796 |
| Source, native libraries, compiler, gamecode | Mini `/Users/mdot/mesh-workloads/cartlane/runtime` |
| Game assets | Mini `/Users/mdot/mesh-workloads/cartlane/Xonotic` |
| Training history and checkpoints | Mini `/Users/mdot/mesh-workloads/cartlane/runs/mini-learner-20260905` |
| Service logs | Mini `/Users/mdot/mesh-workloads/cartlane/logs` |
| Dedicated server | Mini supervisor's SSH child on the laptop; `/Users/mdot/dox/mesh/.build/mini-game-20260905` |
| Client | Laptop GUI LaunchAgent `io.mesh.cartlane.client`, connected to `127.0.0.1:26400` |
| Existing viewer URL | Laptop GUI LaunchAgent `io.mesh.cartlane.viewer-tunnel`, forwarding ports 8795 and 8796 to the Mini |

The launch services use `KeepAlive`, a five-second restart throttle, and normal
termination. The SSH server process and viewer tunnel use `ConnectTimeout=8`,
`ServerAliveInterval=5`, and `ServerAliveCountMax=3`. The tunnel also reports a
forwarding failure instead of holding a live-looking process without a listener.
The client retains its built-in DNS, handshake, and timeout reconnection behavior.
Its process lifetime is independent of the Mini supervisor.

The policy page is [localhost:8795/policy](http://127.0.0.1:8795/policy).
Port 8796 serves the same live viewer as a compatibility alias. Both forwards
belong to the same tunnel service; a runtime handover must update both together.
`/api/status` reports freshness; `/api/policy` reports optimizer progress and
checkpoint provenance. The same endpoints are available on the Mini's port 8796.

The viewer now follows `/Users/mdot/mesh-workloads/cartlane/runs/active`. The
responder publishes this alias atomically after a complete telemetry write, so
changing active run directories on the Mini does not require editing the viewer
service. A pending empty target retains the previous measured scope. Host changes
still require updating the tunnel; see [the reporting RCA](REPORTING-CONTINUITY-RCA.md).

The source package was built on the Mini, including gamecode, before the handover.
Both arm checkpoints transferred at update 4,149, with all 30 parameter arrays,
62 optimizer arrays, and 561 replay transitions per arm. All numeric arrays were
finite. The initial run keeps the same replay batch of one and model dimensions
128/341/8/top-2 so placement does not implicitly change the training contract.

## Checkpoint recovery

Launch checkpoint arguments seed a new run. Within an existing run, its latest
saved training checkpoints take precedence over those original arguments.
`curriculum.py` records `active-match.json` atomically before starting a match.
After an interrupted supervisor or host restart, that record makes the unfinished
match's saved checkpoints discoverable without claiming a completed game outcome.
Missing arm snapshots fall back independently to the latest completed snapshot,
then the original seed. Held-out matches never become training continuation.
New match ordinals do not overwrite the interrupted match's artifacts.

Normal supervisor termination saves the learners, ends the server through its
console, and lets launchd restart supervision. This lifecycle does not signal
either bridge or open another verbs context. A full host reboot was not requested
as a validation step.

## Evidence

Across 34 paired samples over 165 seconds before the restart test, Mini GPU
activity averaged 54.97% (median 56%), versus laptop activity of 8.76%
(median 7.5%). Mini GPU power averaged 2.31 W. This window used four teams,
four carts, and eight players, including the connected human, on `runningmanctf`.

The normal-stop recovery test saved both arms at update 4,532. Launchd replaced
supervisor PID 1625 with 6906; the next learner loaded those newer snapshots,
including all 62 optimizer arrays per arm, with no nonfinite checkpoint arrays.
Both arms subsequently reached 4,670 on `dance`. The viewer reported `advancing`,
zero errors, and one human row. Client PID 64267 stayed alive and reconnected;
bridge PIDs 60237 and 361 stayed unchanged throughout this test.

Deployment argument arrays, launchd property lists, checkpoint hashes, paired
host samples, and the restart observation are retained in
`.build/mini-learner-20260905/`. The recovery tests cover stale launch seeds,
interrupted matches, missing arm checkpoints, held-out matches, and obsolete
active records. Paged matrix and derivative tests also pass on the Mini.

GPU activity is a utilization observation, not a measured temperature or a
hardware-counter FLOP rate. Live timings across different maps are not a
controlled benchmark. The driver faults described in
[RDMA-KERNEL-RECOVERY.md](RDMA-KERNEL-RECOVERY.md) remain unresolved.
