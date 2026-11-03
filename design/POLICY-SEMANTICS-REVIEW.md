# Policy prerequisites recovered from the September 4 session

This is a historical session review. The later
[full-row representation repair](STATE-REDUCTION-RCA.md) deleted the `LiveBelief`
observation summary mentioned below and replaced the model input/readout path.

This review reconstructs the interrupted work and distinguishes repaired defects from
questions that still prevent a policy-strength claim. The literal game remains
[checkpoint control integrated over time](CART-GAME-CONTRACT.md). The restored cart
constructor is documented in [CARTPATHS.md](../xonotic/payload/CARTPATHS.md).

## What the previous session was doing

The local transcript is
`~/.codex/sessions/2026/09/04/rollout-2026-09-04T12-24-04-01a06de0-ceb2-7153-a98d-90b7eb2515be.jsonl`.
Its later messages extend beyond the R46 agenda entry:

| Local time, September 4 | Work and evidence |
| --- | --- |
| 20:13 | Both objectives finally used the same fresh-transition update cadence. A telemetry integer cast was repaired; the preceding NaN computation was explicitly unresolved. |
| 20:21 | Working-tree repairs added scaled normalization, a zero-safe RMS derivative, an implicit conjugate-gradient derivative, and softmax over selected expert scores. These were arithmetic repairs, not evidence that continuation was clean. |
| 20:24–20:38 | Work shifted to battery/load investigation and removing full numerical-array rematerialization from JSON telemetry. `joracle/artifact.py` stores arrays with a small manifest. |
| 21:43 | The session reported nonfinite parameters, optimizer moments and replay, with zero restored weight tensors but old counters still advancing. |
| 21:48 | The operator corrected the evaluation premise: mixed-policy matches already supply outcomes for evaluation against policy history. A separate periodic checkpoint tournament is not required. |
| 21:51 | A retained observation reproduced NaN gradients with fresh parameters. Replacing only its nonfinite integration weights made that diagnostic finite. This identified reinfection through replay, not the original numerical cause. |
| 21:59 | Work moved into the regressed map/cart construction ontology. The subsequent cart-path restoration addressed that branch of the investigation. |

The resumed task was therefore numerical integrity, coherent continuation, spatial
problem synthesis and attribution. It was not policy hyperparameter optimization.

## Reproduced defects and repairs

### Checkpoint continuation combined unrelated states

`online.py::_load_full` independently restored parameters, optimizer, replay, random
state and counters. A failed parameter restore could leave a newly initialized model
paired with contaminated replay and the old update history.

The saved `00043-joint-00021-00001` checkpoints contain, for **each** policy:

| Component | Nonfinite arrays / total arrays |
| --- | --- |
| Parameters | 30 / 30 |
| Optimizer | 60 / 62 |
| Replay | 2,819 / 30,270 |

Of 1,009 retained transitions, 710 reference a nonfinite input or item array. Source
update counters were 1,797 and 1,097. These are observations of saved artifacts, not
counts of successful optimization. The compact audit is in
[`policy-semantics-review.json`](../measurements/policy-semantics-review.json).

Continuation now prepares the complete state before installing it. Tensor structure,
finite coordinates, architecture, objective and reward identity are checked together.
An incompatible or damaged source produces a complete fresh learner: initial
parameters and optimizer, empty replay, initial random state and zero counters.
The source path, source counter, audit and reason remain explicit under `continuation`.
Fresh recovery receives its own initial checkpoint identity. Initial saves now include
optimizer state even before the first update.

This branch prevents a broken learner from being represented as a working continuation;
it always supplies a learner. It neither rewrites damaged historical files during
inspection nor treats fabricated finite observations as training evidence. Arithmetic
failure during a live update remains a separate issue.

### Successor observations used the wrong player population

The responder previously scattered the next observation into the previous roster,
omitting arrivals and inserting zero-filled rows for departures. Since the policy
computes participant coupling and aggregate instrument quality, this changed the
next-state computation for every remaining participant.

The responder now retains the complete observed next frame plus one `successor_rows`
index vector. The learner evaluates that frame once, then gathers value and dynamics
outputs by participant identity. Missing successors receive no bootstrap or dynamics
target. Historical records retain compatibility with the former aligned layout;
their missing arrivals cannot be reconstructed retroactively.

### Replay serialization confused frame numbers with identity

Each replay owner has its own frame counter, while both learners can consume shared
frames. Two different frames with the same numeric counter were merged by `_frames`
and checkpoint export. Serialization now indexes actual frame objects; restored
counters advance beyond the frames they contain. A round-trip test preserves two
different observations both originally numbered zero.

### The belief layer erased height and mapped cells twice

`vcell_from_navigation` truncated 3-D nodes to XY. `LiveBelief` also truncated player
and event positions. Vertically stacked rooms could therefore select the same node,
even when their navigation components were disconnected.

There was an independent indexing error: static navigation supplied already merged
cell IDs, but `_slot_rows` and `chorus` treated those IDs as original node indices and
applied `node_cell` again. For owners `[0, 0, 1, 1]`, cell 1 was incorrectly remapped
to cell 0.

Navigation positions now retain XYZ, and the cell index points to a member node of
that cell. The regression fixture has stacked, disconnected floors with multiple
nodes per cell. It checks node assignment, player membership, observation support and
preserved target height.

Preserving disconnected components also exposes an infinite walking-distance value.
That is useful evidence of an unknown route, but it must not become an infinite
commitment sent to the engine. Commitment retains that distance, reports
`route_known=false`, and uses straight-line travel time plus the policy extension
as a finite estimate until a route is known. No target is removed.

## September 5 slate completed

The subsequent implementation is documented in
[POLICY-EXECUTION-CONTRACT.md](POLICY-EXECUTION-CONTRACT.md), with evidence in
[policy-cleanup-slate.json](../measurements/policy-cleanup-slate.json).

- Shared arithmetic fixes the reproduced PPO unused-branch overflow and global-norm
  overflow, and replaces raw exponential actuator scales with a versioned smooth
  reparameterization. Sampling and likelihood use one implementation.
- Registered navigation target IDs now reach the QC adapter through generated 3-D
  entity markers. Actual compiled QC fixtures distinguish vertically stacked goals.
- One action history supplies delayed first-application actor records, adjacent critic
  intervals, telemetry source joins and policy-version provenance. Actor-only groups
  do not duplicate episode rewards or historical critic intervals.
- Game state, equations and reward contracts have one Python owner. Compiled QC and
  Python agree on 39 float32 scoring cases and six ownership transitions.
- Mixed-match reports record executed version exposure and actual terminal outcomes,
  including attribution and observation-resolution gaps. Map asset realization has
  moved out of the supervisor; the responder delegates its former source-join block.

The original historical first invalid operation remains unidentified because no saved
intermediate captures it. Finite bounded verification does not establish numerical
stability on every live trajectory. Nor does consistent attribution establish policy
strength: changing multiplayer compositions still need a stated statistical comparison
model and uncertainty. There is no new tournament prerequisite.

## Module boundaries and verification

The useful boundaries are concrete data ownership:

| Data | Owner |
| --- | --- |
| Parsed map, navigation links, cart candidates and realized lanes | `payload/tools/navmesh.py` |
| Engine entities and measurement output | `payload/tools/mkentfile.py` |
| Literal cart-state value | `strat/game_value.py`; wire conversion in `runtime.py` |
| Spatial observations and their navigation cells | `strat/live_belief.py`, `featurize.py` |
| Full observation frames and participant successor indices | `strat/inputs.py` |
| Losses and complete learning-state continuation | `strat/online.py` |
| Replay identity, retention and serialization | `strat/replay.py` |
| Read-only stored-state numerical audit | `strat/checkpoint_audit.py` |

Request, execution and measurement ownership now lives in `strat/action_history.py`.
Map artifact ownership lives in `strat/map_assets.py`. The orchestration files remain
substantial, but no longer own those independent relations.

The following paragraph records verification of the earlier R47 repairs; the September
5 slate's expanded checks are in the linked execution contract and evidence file.

Historical verification on macOS arm64: ten regression tests passed, plus a reproduction on a
saved real observation. Six data/geometry tests also passed on Linux x86-64. An isolated
recovery exercise loaded each damaged checkpoint, reported a fresh learner, and ran
32 finite updates per objective on one saved observation with newly sampled behavior.
This exercises common arithmetic repeatedly; it is not an interactive match or a
policy comparison. No live process was manually restarted for these checks.

```sh
PYTHONPATH=xonotic bin/mesh-python -m solver.strat.checkpoint_audit path/to/checkpoint.npz
```
