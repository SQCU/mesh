# Reporting continuity and policy control: September 5, 2026

Current implementation status: [September 7 execution and reporting changes](POLICY-IMPLEMENTATION-20260907.md). Historical observations below predate these source changes.

Historical audit of the subsequently deleted categorical adapter. Its candidate
distributions, controls and viewer repairs below are historical observations, not
the current policy interface. The current source, Gaussian state-rate outputs and
reporting contract are in [POLICY-PROGRAM.md](POLICY-PROGRAM.md).

Follow-up: [policy feature and gradient bottlenecks](POLICY-ACTION-BOTTLENECKS.md)
records the operator-requested workload pause, a reproduced cart-ownership input
collision, projection-gradient checks and the mismatch with transcript-level WHAT/HOW.

The recurring failure is an ownership gap. Compute processes, run directories,
telemetry schemas and browser projections changed independently. We tested their
local operation without requiring the complete reporting path to survive:

```
produce -> retain -> discover -> read -> present -> identify missing coverage
```

A live PID or HTTP 200 does not establish that the requested report is advancing.
The previous placement repair also left the user's established port 8795 behind;
that was our deployment error, not an operator configuration mistake.

## Case studies

| Case | Mechanism and evidence | Repair and remaining limit |
| --- | --- | --- |
| September 4 lost outcomes | A replacement bridge left the engine mapped to an obsolete SHM object; the learner could wait inside a remote operation while the supervisor saw a live PID. Outcome handling depended on that learner progressing. Restarted logs could truncate evidence. | Backing-object remapping, bounded remote operations with local computation, append logs and independent `server-outcomes.jsonl` were added in earlier repairs. The old gap cannot be reconstructed as invented round results. See [the incident](TELEMETRY-LOSS-20260904.md). |
| Provider and kernel failures | Provider receive faults and a recurrent Mini Thunderbolt kernel fault interrupted application progress. Application supervision did not establish the closed driver's failure mechanism. | Flight records and recovery observations now retain more evidence. The driver fault remains unresolved; see [kernel recovery](RDMA-KERNEL-RECOVERY.md). |
| Backend moved, established URL stayed behind | 8795 served the old September 4 run while 8796 served the Mini learner. The old run's latest match was empty. | Both laptop ports now belong to one supervised tunnel to Mini port 8796. Changing the backend host still requires updating that service; fleet-wide application discovery is not implemented here. |
| New empty match erased measured scope | Selecting the newest directory treated its existence as proof of a new measurement. | The reader retains the previous measured match until a complete JSONL record exists and names the pending directory. Missing measurements remain missing, rather than becoming zeros. |
| Run path outlived its producer | A literal `--run-dir` remained valid after another run became active. Reconnection faithfully reconnected to obsolete data. | The responder now atomically publishes `runs/active` after a flushed telemetry record, and periodically republishes it. The viewer follows that alias and retains the prior scope while a new target is empty. This is same-host run discovery, not a persistent cross-host catalog. |
| Viewer rewrite dropped responsibilities | The working-tree replacement of `joracle/server.py` removed calls into `TelemetryFollower`, `RuntimeMeasure`, rolling probes and field coverage. Removing duplicate numerical decomposition also removed unrelated reporting responsibilities. | The current reader preserves producer J artifacts and now exposes explicit report coverage and isolated report errors. The old helpers' existence does not mean they are connected. Full field inventory and fleet diagnostics have not all been restored to the application page. |
| One malformed report froze unrelated reports | J or study parsing could abort refresh; cached application responses could keep the previous healthy status. | J/study failures now preserve their previous good artifact while optimizer telemetry advances. Reader errors propagate into both page caches. Tests exercise partial artifacts and run handovers. |
| Clock and generation confusion | Telemetry `t` is responder elapsed wall time, although labels called it engine/game time. A restarted responder could reset it within the same file. | Labels now identify responder time; a decreasing response counter or elapsed time starts a new plotted series. New records carry `recorded_at` and `producer_pid`. File mtime, viewer ingest time, producer time and J sample time remain distinct clocks. |
| Elo consumer expects a retired schedule | `study.summarize` groups mirrored comparison legs; the joint curriculum schedules mixed-team learning without those legs. | Existing paired-study coverage is now visible. The paired estimator cannot supply the promised composition-aware online rating just by reconnecting its output. |
| Counterfactuals were computed, then discarded | Both policies evaluated every player, but retained reports contained sampled action indices and values. The viewer dropped even that projection. | The new `policy_reports.py` reports same-input categorical divergence, conditional control parameters and sampled wire vectors; the page has corresponding tables and a full-vector endpoint. These are immediate policy comparisons, not simulated alternative game trajectories. |

The current application viewer still lives on the producer host. A full outage of
that host removes its HTTP service; an already open browser can retain its last
rendered values, but there is no durable laptop replica of the complete application
report. Source discovery, durable replication and connection recovery are separate
remaining responsibilities. Browser fetches also have no explicit completion budget
in the current implementation; a pending fetch can hold its single-flight polling
flag indefinitely. This is another transitive wait to repair, not proof of backend
failure. A background cmux browser remained in that pending state during this audit;
HTTP API checks succeeded, so visual verification is incomplete.

## Promised reporting versus implemented reporting

| Obligation | Actual state after this repair |
| --- | --- |
| Objective-labelled optimizer counts, losses, fresh/replay rows, rollout age and importance correction | Produced and exposed at `/api/policy`; both learners advance. Equal update counts do not establish equal actor exposure. |
| Literal J, feature coordinates, spectra, outcome covariance, delivery/route/event joins | Producer artifacts and `/j` exist. Coverage, identity and age must accompany every projection. |
| Both policies' outputs for every player, including other-policy and human rows | Now retained in `policy_comparison`. `/api/counterfactual/full` supplies the latest complete sampled probability and conditional-control tensors with the candidate catalog. |
| Policy divergence | Categorical Jensen-Shannon divergence in nats and total variation now computed per player on the same input. Expected raw-control differences are reported separately. This is not divergence of the full joint categorical/Gaussian action distribution. |
| Paired-study Elo | Implemented in `study.py`, conditional on paired comparison evidence; current joint scheduling supplies no mirrored blocks. The viewer exposes the coverage and study payload, not a newly fitted live team rating. |
| Seeded playerbot/team-composition Elo | Described in [application telemetry](APPLICATION-TELEMETRY.md), but no realized bot-configuration ledger or fitted conditional estimator exists. Explicitly marked `not_implemented`. |
| Playerbot performance versus strategy-controller contribution | Per-player damage, kills, deaths, pickups, push/contest and routed outcomes exist. `ExecutionEvaluation` summarizes version exposure for consumed streams. These are not a fitted separation of bot skill and controller effect. |
| Complete historical round denominator across interrupted runs and migrations | Partial: independent engine outcome journal, run index and consumed telemetry. `study.summarize` still obtains outcomes from its telemetry records, not a unified lineage-wide authoritative ledger. |
| Acquisition/recovery, retention, opposition, switching and coalition behavior with opportunity denominators | Promised, not supplied as a complete strategy report. Raw events and value losses do not fulfill this obligation. |

The paired Elo implementation converts observed paired score odds to
`400 * log10(p / (1-p))`, and fits a centered relative coordinate when the finite
comparison graph identifies it. Boundary evidence is retained separately; no wins
are fabricated to make a finite estimate. `realization_elo` adds checkpoint
provenance, not bot configuration traits. A missing paired rating and an absent
composition estimator are different conditions.

Bot configuration actually exists in `server/bot/default/bot.qc`: individual
movement, dodge, aim, thinking, weapon and aggression traits combine with the global
skill setting. The strategy wire observation contains player edict/team/control and
outcome counters, but does not identify the realized bot configuration and its trait
vector. A seed or the final roster size does not recover that missing identity.

The next rating implementation needs an authoritative roster/configuration ledger,
round and interruption identity, policy lineage/version exposure and complete
multi-team outcomes. Start with descriptive team/player-time exposure and coverage.
Then fit a declared composition model with uncertainty and calibration. The
[Plackett-Luce model overview](https://hturner.github.io/PlackettLuce/articles/Overview.html)
provides a multiway proportional-worth model family;
[TrueSkill 2](https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/)
is a primary reference for extending team outcome ratings with player performance.
Neither is already implemented here. Separating controller contribution requires
variation in which bot compositions receive which controllers; conditioning on
damage caused by a controller can remove part of the effect one intended to measure.

## What a policy action currently means

An action index is a temporary index into `InstrumentBatch.instruments` for one
observation. Each candidate combines an operation with a concrete target: push cart
3 and suppress cart 3 are different candidates. Item, rival and exploration targets
add more candidates, followed by spawn timing and idle. The index is neither an
engine entity ID nor a stable action identity across observations.

For P players and M candidates, each policy emits P×M logits and P×M×3 continuous
control means and log-scales. `act` samples one categorical candidate per player.
The responder then samples gain, raw commitment extension and spawn adjustment
conditional on that selection, using the assigned policy's output. Both policies
are evaluated on all players, but the team assignment selects which policy's rows
are sent. Optional uniform exploration has a separate behavior label; human rows
are labelled human and do not enter the bot actor mask.

The wire action has eight coordinates:

```
[instrument kind, target kind, target ID, cell X, cell Y,
 gain, commitment seconds, spawn adjustment]
```

`response_rows` copies the selected candidate's identity and sampled gain/spawn.
Commitment becomes estimated walking time plus softplus of the sampled extension.
The complete categorical distribution and all conditional controls stay in the
solver; the game receives one selected command per participant.

The engine applies this as a strategic preference inside Havocbot. It adds a goal
rating scaled by gain, while stock cart, item and waypoint priorities remain active.
Havocbot chooses navigation goals and controls movement, aiming, weapons and firing.
Thus a received command can be considered by goal rating and still lose the goal
selection. `ROUTE_SEQ` is set when the strategy target is resolved in goal rating;
it is not proof that the bot chose that goal, moved toward it or reached it.
Goal identity, goal match and touch are separate observations.

`ActionHistory.advance` joins that route acknowledgement to its retained source
observation, sampled action, raw controls, behavior probability and model version.
Its actor mask is the first matching acknowledgement for a bot whose behavior arm
is that learner. Repeated acknowledgement is not another actor sample. Both
learners receive value intervals from the observed game, and historical replay has
zero actor weight. Value gradients share parameters with action outputs, so a
counterfactual action distribution can change even without an actor example for
that player. These are behavior-mixture value estimates, not observed returns from
executing the counterfactual policy.

Two policies can select the same categorical candidate from different probability
vectors. They can then emit different continuous controls. The current comparison
uses common random keys; agreement in one draw is especially weak evidence of
policy equality. The relevant report is the full distribution plus the sampled
command, actual controlling identity and application stages.

## Execution contract discrepancies requiring follow-up

The existing hierarchical strategic interface is explicit in
[STRATEGY-IO.md](../xonotic/payload/STRATEGY-IO.md). It does not establish that the
current implementation perfectly realizes that contract:

- The name `first_execution` currently means first observed matching goal-rating
  application. The report should say so and separately count delivery, resolution,
  selected goal, persistence, touch and outcomes. Changing actor eligibility to
  require a successful goal or touch would also change the learned objective;
  that should not be slipped into a telemetry fix.
- The documentation calls human responses advisory, but the `PlayerPreThink`
  spawn adjustment in `sv_payload.qc` does not check bot control. The code can
  therefore affect a dead human's respawn timing. The documented blanket claim
  that no human control is applied is false for this coordinate.
- Spawn adjustment occurs outside the goal-rating acknowledgement path. Dead bots
  return before Havocbot goal rating, so that acknowledgement alone does not prove
  or account for application of the complete three-control action. Commitment
  also has no separate source acknowledgement in the observation schema.
- Candidate support currently uses team observation availability. It does not
  guarantee a target can be used by the selected engine operation; for example,
  rival application checks enemy-team membership. Resolution failures must remain
  visible and be explained, rather than being counted as successful control.

The desired data flow keeps a typed, versioned action record from model output
through wire conversion and each actual actuator application, with stable target
identity. It retains policy identity, sampling probabilities, raw and transformed
controls, application reasons and outcome joins. Ratings then distinguish the
strategic policy from the stock bot executor and its realized configuration.
This audit does not silently replace that hierarchical policy with direct movement
or firing control, nor change the actor objective to reward only successful routing.

## Verification and scope

The learner was normally terminated through its catchable handler and recovered by
the existing supervisor from update 7,814 per arm. No bridge was signalled. The
new reporting code subsequently produced 32 policy/player rows and 16 pairwise
comparisons for the eight-team, sixteen-player match. Five report/reader tests
passed on the laptop and Mini, including empty/partial handover, malformed report
isolation, responder counter reset, divergence endpoints and sampled vectors.

In saved response 1,602, all fifteen model-controlled rows matched their assigned
policy's sampled candidate and transmitted target/gain/commitment/spawn exactly.
Nine earlier commands were newly acknowledged in goal rating; four of those
matched the selected goal. The comparison computation took 1.856 ms in that frame.
This verifies an application path, not completion of every requested behavior.

A later 300-frame API window spanning responder seconds 105.901–438.679 contained
116 actor rows for matrix_fusion and 1,173 for terminal_win, with 7,948 reported
value rows for each. Both reached update 8,323. The current roster assigned one bot
to matrix_fusion, fourteen to terminal_win, and one human. Equal optimizer-step
counts therefore conceal very unequal actor exposure. These are counts in the
captured window, not a controlled comparison or lifetime totals.

Evidence is in `.build/reporting-rca-20260905/`: the source frame, API snapshot and
actor coverage summary. New model-report math does not perform additional policy
forwards; it reuses existing outputs and random keys after sending the live response.
Probability tensors are retained at the model-sampling cadence, not every frame.
