# September 4 telemetry interruption

Scope: the corrected checkpoint-control lineage, not the discarded delivery-game
results. Times below are Pacific daylight time. Source changes are uncommitted.

## Causal chain and evidence

1. The Mini bridge PID 62938 crashed at **18:28:26.5665**, eight seconds after the
   last normal learner frame. The OS report is
   `/Library/Logs/DiagnosticReports/mesh-flow-2026-09-04-182828.ips`, incident
   `1208350F-935C-48F4-9F17-1A2A70FFBB14`. This was a userspace `SIGSEGV`, not a
   recorded kernel panic or an operator kill. Launchd replaced it with PID 57532.
2. The server and matrix worker remained alive. The replacement bridge had no
   attached userspace client and zero traffic. The server retained its mapping of
   the dead bridge's shared-memory object. `mesh-client.c` compared `st_ino` to
   detect replacement, but actual `/mesh0` descriptors return inode zero on **both**
   machines. The comparison can never distinguish these objects.
3. `FrameStream.exchange` retried without a completion budget. The learner was
   sampled sleeping inside the Python/MLX custom-operation path. It could neither
   finish its update nor append telemetry nor process queued outcomes. The
   supervisor treated a live PID as sufficient progress and awaited the learner's
   `outcome.json`. Engine rounds continued while the application reports stopped.
4. Independently, supervisor process launch used `open(log_path, "w")`. Replacing
   a persistent server could truncate its entire prior console history. Telemetry
   had another truncate mode. The logs were also the evidence needed to diagnose
   these same replacements. Fresh HTTP-cache timestamps did not imply fresh
   computation, despite the observer itself remaining responsive.

These are distinct failures: interruption, failure to reconnect, unbounded compute
wait, outcome/optimizer coupling, and destructive log reopening. The architectural
cause of the extended reporting outage was making observation depend on successful
distributed optimization, with no progress boundary or independent outcome journal.

## Corrections

- macOS shared-memory identity now comes from read-only VM backing-object metadata;
  other POSIX hosts keep their inode identity. The original named region is retained.
  Read/pump/stream calls periodically check replacement even during traffic, using
  the existing userspace remap path. An unavailable identity query is reported and
  does not interrupt work. No bridge header, verbs registration or device is changed.
- A full release ring yields to the caller and retains its pending page; it no
  longer spins forever before the replacement check can run.
- Remote operations have a five-second transaction budget, separate from the
  0.4-second strategy deadline and retransmit interval. Failure reports contain
  request identity, frame offers/replays, partial assembly, queue depths and elapsed
  time. The **same** Gram operation or VJP completes locally on an incomplete remote
  transaction. Every subsequent operation still attempts the remote worker; no
  permanent feature disable or reduced matrix/population configuration is introduced.
- The engine emits `MESH_OUTCOME` JSON containing map, episode, engine time, winner,
  threshold and all scores. The supervisor journals it independently of the learner,
  flushes/fsyncs `server-outcomes.jsonl`, and gives the learner its drain grace to
  acknowledge the outcome. Missing acknowledgement is reported, not called a learned
  terminal result. A source loop also reads the engine journal before checking the
  learner's acknowledgement, so that acknowledgement cannot skip the final read.
- Process logs and telemetry append. Per-match server extracts use the actual
  append offset. The shared incremental JSONL reader handles incomplete trailing
  lines. Application pages distinguish producer quietness from HTTP-cache freshness
  and read durable outcomes as well as observed learning frames.
- Responder restart prefers its current output checkpoint over the prior match's
  input, and does not overwrite the original initial snapshot. The first recovery
  attempt exposed that rewind: it consumed no new observations. The preserved
  current snapshots, strategy update 270 and terminal update 4, supplied deployment.

The added branches preserve reachability: retain pending work, retry the named
region, complete an equivalent operation, or report missing evidence. They do not
gate a capability by node class, invent a terminal reward, reduce configured capacity,
or operate on a verbs owner. One controlled userspace deployment applied the statically
linked engine and supervisor changes. Normal match/map transitions retain clients.

## What is not yet established

The provider crash itself is **not fixed by the userspace recovery corrections**.
Reading the Mini's already installed provider code, without opening a verbs device,
resolves `tbt_post_recv + 512` to `str w22, [x8, #8]`. The preceding instruction loads
`x8` from the provider QP's offset `0xd0`; the faulting target is `0x1005f0008`, in an
unmapped 16 KiB gap. This is the receive queue's producer-index publication, not a
walk beyond a game tensor. The crash report also records VM triage messages
`Returned success with no page`. This narrows the fault to an absent provider mapping;
it does not establish why the mapping disappeared or absolve every caller interaction.
Changing queue depth, patching a driver pointer or forcing another bridge crash would
not be an evidence-backed repair. Preserve this report for driver-level investigation.

No retrospective actions, value targets or wins were fabricated for the telemetry
gap. Older console messages can establish observed events, but cannot reconstruct
the missing behavior/version/exposure joins needed for a fair historical rating.

## Verification

Both live regions reported inode zero; two independent read-only mappings of each
region reported the same full backing-object identity. Client C compiles with
`-Wall -Wextra -Werror`; native engine and QC client/menu/server compile successfully.
This does not claim a deliberately induced bridge-replacement experiment.

At 19:20 the deployed runtime resumed remote forward/backward computation, learner
frames and native off-policy gameplay. Both existing bridge PIDs remained unchanged
during deployment, with `up=true`, attached live clients and `bad=0`. A real five-second
remote timeout then logged local completion, and later learning frames continued.
The corrected rank implementation matched all 273 pre-deployment recorded frames.
The first 123 post-deployment frames had zero game-coordinate/checkpoint residuals
and no score decreases in 484 same-episode comparisons. These are implementation
checks, not evidence that strategic learning or generalization has succeeded.

At 19:27:21 the independent journal captured team 4 reaching exactly 1200 at engine
time 429.53302 on `runningmanctf`, with scores
`[597.896118, 202.499847, 36.9332581, 1200]`. The learner acknowledged the same event
14 seconds later, reaching strategy update 407 and terminal update 8. The next
realization changed from three to two lanes on `dance`; client PID 68295 and Mini
server PID 21176 persisted. The revised HUD downloaded as CSQC size 4351193 and was
visually checked: team name, score/Q, score percentage and fixed-control ETA are
readable below the path bars, without the original top-timer clipping. The observer
client's welcome modal is disabled so it no longer covers the HUD at every map.

The first outcome join exposed decimal-versus-binary timestamp duplication:
`429.53302` and `429.53302001953125` are the same wire float32. The viewer keys that
coordinate by its float32 bits, preserving source metadata; the two actual records
now produce one round, including after observer restart. At 19:31 both application
APIs were advancing, J had two labelled strata, and the policy page showed strategy
446 updates, terminal 8, and one independently journaled post-recovery win. It does
not count the missing interval as losses, draws or nonexistent games.
