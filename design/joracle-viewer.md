# The J-lens and J-oracle

The live viewer measures the strategy representation and the server transitions it
causes. It does not decide whether a representation, policy, match, or release passes.
Every output is an empirical measure over values emitted by the running responder and
the Xonotic server state machine.

The computation is in `xonotic/solver/strat/joracle/probe.py`. The responder publishes
its measures through the same generic workload record as its FLOP, byte, row, and
deadline coordinates. The node telemetry ring on port 8788 retains the record and the
whole-mesh reporter on port 8787 presents it. No J-specific service or browser pane is
launched by the runnable demo.

## J

For participant `p`, the selected instrument is `a_p`. The measured J row is

```
J_p = Strategy.ir[p, a_p, :]
```

This is the participant–instrument representation consumed by the action sampler and
actuator. It is not the policy-distribution expectation `Strategy.pooled[p, :]` consumed by the two
value probes. A homogeneous native-width frame exposes the two matrices as `model.j`
and `model.pooled`. Every sampled frame also exposes `model.row_outputs`, a tagged row
union carrying the arm and literal native-width J, belief, and pooled coordinates. A
mixed matrix-fusion/baseline frame is never padded or broadcast into a fictional common J width.
The frame-level composer tensors and paired matrix-fusion intervention remain retained when J
widths differ; each participant's native J stays in its labeled source stratum.

The strategy response is the literal tuple

```
A_p = (target_kind, target_id, target_cell_x, target_cell_y, gain, commit, spawn)
```

Target kind, entity identity, and the two cell-lattice coordinates remain separately
typed coordinates. Gain, commitment, and spawn delay remain the three float controls
emitted by the actuator. The telemetry path does not pack, normalize, round, truncate,
or reinterpret these values.

Every policy output has one actuator mean tensor `(participants, instruments, 3)`, one
log-scale tensor of the same shape, and one participant-indexed density mass. Learned
`matrix_fusion`, `initial_policy`, `participant_fusion_ablated`,
`residual_fusion_ablated`, linear, and FFN arms have density mass one; the literal default arm
has density mass zero and emits its mean without noise. No tensor width selects policy
semantics. In a mixed-arm match, every reported participant row is gathered from the
same arm that sampled that participant's instrument and controls. `diag(K)` is therefore
participant-by-instrument: matrix-fusion rows carry their literal DPP diagonal and arms without
that computation carry zero contribution.

## J-lens

At every response, the lens forms the empirical joint measure

```
μ_lens = Σ_p δ_(F_p, J_p)
```

where the row-aligned feature coordinate `F_p` contains the exact `x`, learned belief,
closed-form hierarchy, team identity, selected instrument descriptor, incoming
integrated instrument weight, selected action-support mass, cadence, and the control
and exploration coefficients used for that selected row. The report
publishes mass, coordinate integrals, coordinate variances, J integrals, J variances,
and the complete `cov(F,J)` matrix.

For every feature schema and authoritative server-state schema, a separately named
derived measure computes the minimum-norm affine empirical L2 projection from J to those exact ground-truth
coordinates. This is an unregularized projection over the complete retained empirical
measure, not a train/test probe. The operator, offset, source singular values, numerical
rank, target and image square integrals, and residual square integral per coordinate are
all published. Finite and non-finite atom masses remain separate; the residual remains
a numerical measure.

The wire schema owns the coordinate types. Real coordinates remain real, each packed
weapon word expands losslessly to its 24 bit atoms, and entity, team, cell, target-kind,
target-identity, and response-sequence coordinates expand to observed categorical atoms.
The affine projection therefore never assigns metric meaning to an edict number, team
number, packed target, or sequence number.

Each feature/J coordinate schema and policy arm is a separate measure stratum. J labels
carry the arm name and native coordinate index. A responder restart, architecture
change, or a same-width baseline with different semantics therefore cannot make older
atoms disappear or merge unlike coordinates; the compact table renders the newest
stratum and `coordinate_strata` retains the moments of every observed schema.

When `matrix_fusion` and either independent fusion-ablation arm are present, the
parametric functions consume the same
authoritative frame with the same model object and realized checkpoint. A second
configured checkpoint cannot change the ablation's parameters. The lens also forms the paired
same-state intervention measure

```
μ_participant = Σ_t δ_(J_matrix_fusion - J_participant_fusion_ablated, TV(π_matrix_fusion, π_participant_fusion_ablated), Δactuator_mean, Δactuator_log_scale, Δvalue, Δparticipant_coupling)
μ_residual = Σ_t δ_(J_matrix_fusion - J_residual_fusion_ablated, TV(π_matrix_fusion, π_residual_fusion_ablated), Δactuator_mean, Δactuator_log_scale, Δvalue, Δresidual_fusion)
```

over every participant, instrument, and J coordinate. Its mass, integral, square
integral, mean, variance, minimum, and maximum are additive rolling-window measures.
No sampled action, fitted probe, reconstructed input, or later game outcome enters this
same-state computation. The oracle separately disintegrates later server outcomes by
the arm that was actually assigned, preserving the causal path from matrix-fusion intervention
through changed J and action distributions to realized behavior.

The responder passes all eleven composer inputs verbatim to the in-memory measure on
every response and serializes them when it samples a model frame:

```
x, z, cell_slots, gigi, hierarchy, team_ids,
w, action_mass, delta, control_weight, exploration_weight
```

The lens reports mass, integral, mean, variance, and observed shapes for every composer
family. Dynamic player, instrument, and cell axes remain dynamic; no study constant
pads them and every observed size retains its own stratum. The responder's live frame
retains the newest exact sampled arrays; the generic mesh interface carries their
empirical measures without learning the array schema.
The completed strategy response is scattered before selected-J copies, tensor moments,
telemetry serialization, or local counterfactual timing. Observer work therefore does
not occupy the server control deadline or inflate the delivered-response elapsed time.
The workload record carries `post_response_measure_elapsed_s` separately so observer
cost remains visible to the whole-mesh profiler instead of disappearing from host-load
accounting.

## J-oracle

The server exposes four sequence coordinates for each participant:

- `observed_response_seq`: the response delivered to gamecode;
- `route_seq`: the response currently used by navigation routing;
- `goal_seq` and `touch_seq`: the route that produced a goal and target contact.

They are not interchangeable. A delivered response may not yet be routed, while an
older route may still be producing movement and combat. The oracle therefore preserves
each named sequence relation rather than filtering them into the newest delivery.

At the source response it first forms

```
μ_source = Σ δ_(J_p, A_p, S_p)
```

where `S_p` is the authoritative server wire state: the complete 83-coordinate
participant observation row and every complete 16-coordinate cart row, all retained
verbatim before featurization. Learned belief, expanded weapon bits, selected instrument,
closed-form hierarchy, and integrated weight remain in the exact lens coordinate `F_p`.
The report retains state mass, integral, mean, variance,
`E[SJ]`, and `cov(S,J)` for every labeled state coordinate.

The authoritative state-reference measure is

```
μ_state = Σ_r Σ δ_(r, J_p, A_p, S_p, S′_p, ΔS_p, applied_target_resolution)
```

where `r` is `delivery`, `route`, `goal`, or `touch`, joined by each relation's own
`(stream epoch, response sequence, participant edict)`. `ΔS_p` is the subsequent
difference of the complete participant and cart rows supplied by the server state
machine. It is formed in the measure worker directly from the retained authoritative
source and successor rows rather than serialized as a second implementation by the
responder. Differences are formed only for real coordinates and losslessly expanded
weapon bits. Categorical coordinates produce literal `(coordinate, source, target)`
atoms, including unchanged atoms, with J and control integrals for each transition.
Every source retains its own coordinate labels. If a dynamic schema
changes, the delta is taken over the literal name intersection and both schema strata
remain measured. A delivered response that is not the current route remains a delivery
atom, and an older active route, goal, or touch remains an independently joined atom.
Cart coordinates are named by the server's cart ID rather than their temporary wire-row
ordinal, so a row reorder cannot masquerade as cart motion.
`applied_target_resolution` is the server's raw coordinate at each target observation;
it is not conjoined with route identity into a synthetic category. The report publishes
mass, exact-source mass, joined mass, and unjoined mass separately for all four relations.

The responder snapshots the complete row-aligned source coordinates, authoritative
server-state row, `J_p`, policy intervention, and literal action with each live sequence
reference before handing the frame to the measure worker. A row-counted in-memory source
window retains delayed response references across subsequent exchanges and roster changes;
its capacity, ingested mass, retained mass, evicted mass, and sequence mass are published.
State-reference and behavior joins
therefore do not depend on the source response still occupying the rolling feature
window. `source_atom_mass` counts unique `(epoch, response, participant)` atoms and
`source_state_all_atom_mass` counts the atoms carrying authoritative source state.
Changing cart counts produce separate state-coordinate strata rather than discarded
rows; `source_state_coordinate_mass` names the newest stratum rendered in the compact
summary while `source_state_strata` retains every stratum's moments.
The state-reference and behavior exact-source masses count later atoms joined directly to
those retained sources. A missing source remains unjoined mass; no sampled frame is
used to reconstruct it.

Every joined state-reference atom also carries the complete authoritative successor row and its
schema verbatim. `successor_state_strata` publishes its direct integrals, variance,
cross-moments with J, covariance with J, and the derived J-to-successor affine projection.
The successor row is not reconstructed by adding the reported delta to the source row;
the delta and categorical-transition measures are separate pushforwards of the same
literal source and successor atoms.

The behavior measure is

```
μ_behavior = Σ δ_(J_p, A_p, ΔY_p)
```

joined by `(stream epoch, route_seq, participant edict)`. `ΔY_p` contains the exact
successive server counter differences for enemy damage dealt and taken, kills, deaths,
pickups, cart push, and cart contest. This associates realized behavior with the route
that was actually active rather than the newest response merely present in memory.
Each counter also carries value-present, numeric, and finite mass. A non-finite counter
therefore remains an oracle atom and contributes to non-finite mass instead of being
omitted from the pushforward.

The event measure is

```
μ_event = Σ δ_(J_p, A_p, event_kind, event_value)
```

joined by the event row's own `(stream epoch, response_seq, actor edict)`. Damage,
kill, pickup, and round atoms come from the same authoritative event table consumed by
live featurization. Event-owned response sequences participate in source retention even
after the corresponding route leaves the current observation row, so the direct atom is
not replaced by an accumulated counter or a reconstructed source.
Event value presence, numeric mass, and finite mass are separate coordinates. An absent
or non-finite event value contributes to those masses and never becomes a fabricated
zero-valued outcome.

For every event, outcome, and state-delta coordinate the report publishes empirical mass,
integral, mean, variance, `E[JY]`, `cov(J,Y)`, `E[AY]`, and `cov(A,Y)`. It also publishes
the delivery, route, goal, touch, behavior, and event masses that joined and the masses whose source J row lies
outside the retained exact-source window. The same measures are disintegrated by
assigned policy arm and by actual behavior policy. `matrix_fusion`, both independent
fusion ablations, FFN, linear,
default, and uniform exploration therefore never disappear into one aggregate or
receive one another's outcomes. Missing joins remain counted as missing mass.

Every common outcome schema additionally forms a minimum-norm affine empirical L2
projection from the exact source J row to the complete joined outcome vector inside each
policy-arm/native-J stratum. Every continuous/bit state-difference schema forms the
corresponding stratified J-to-state-difference projection. Their per-coordinate residual
square measures are rendered beside the arm's direct moments. The source and target atoms
remain the server transition; these projections only measure how much of that observed
pushforward is linearly present in the corresponding J coordinate system.

This is the literal pushforward measure available from the controlled state machine:
the same process owns the authoritative source state, every featurization input, the
selected representation and response, and the later route-owned state and outcome
atoms. It is not a learned action-to-successor model and it does not substitute a
prediction for an engine transition. Policy-arm interventions are coordinates of the
source atom; mirrored match blocks provide repeated interventions under the same map,
seed, roster, team-side schedule, and perturbation.

## Spectral measures

J width, centered singular-value mass, integrals, participation-ratio effective rank,
nonzero input columns, and singular values are direct numerical summaries. Absent and non-finite
arrays are reported as their observed state. Numerical estimates use the finite-coordinate
submeasure and report both its mass and total row mass; neither condition suppresses the
literal lens or oracle. Width changes produce separate shape strata rather than deleting
older rows.

## Runtime interface

`LiteralJReporter` owns a bounded in-memory observation window inside the responder.
Ingestion of the already-materialized arrays remains on the game-response path; matrix
moments are computed by a separate low-cadence thread. A match or map reset clears
the transition window so response sequence reuse cannot join across server episodes.
The first frame after start or reset wakes an immediate computation; later frames use
the configured cadence. Graceful stop drains pending atoms before the measure thread
joins, submits the final measure to the node telemetry ring, and gives the generic
publisher a bounded final delivery attempt, so a short match cannot lose its only
literal source window merely because the process exits first.
The report names the window's row capacity and the ingested, retained, and evicted
coordinate and transition row masses. Its integrals are therefore explicitly measures
over the retained rolling domain rather than claims about unseen episode history. The
window bounds observer memory only; it does not bound player, instrument, residual,
expert, transport, or strategy workload rows.
The responder retains decision records by causal reach: the newest response and every
response sequence still named by a delivered response, active route, goal, touch, or
outcome interval. Once no live server coordinate can name a sequence again, its record
is removed rather than accumulating for the lifetime of the process.
If the live causal set itself exceeds the configured observation depth, the source
window expands to that measured set instead of evicting a response still named by the
server.

`rdma/workload.py` sends the small FLOP/byte/deadline heartbeat independently from a
latest-only asynchronous measure update. `user/mesh-telemetry.py` merges them by
producer identity in memory, so serialization or HTTP transport of the full covariance
objects never occupies the solver response path. The mesh observer does not import a
Xonotic schema: it transports every producer's `measures` dictionary and the phase-space
HUD renders every top-level scalar coordinate generically.

The full live measure is available at:

```
http://127.0.0.1:8788/v1/latest
http://127.0.0.1:8787/latest.json
```

under `workload.producers[].measures`. `/v1/visualization` carries the same newest
producer object for each node while compacting older phase-space samples. Consequently
both laptop and Mini measures appear in one 8787 response whenever both nodes are
meshed, and either remains a named stale node through a partition.
The curriculum's generic roofline sampler retains the newest nonempty `measures` object
for every matching producer on every node as `producer_measure_records` in the match
artifact. It copies the dictionary without importing this schema, so the exact J measure
remains available after the live in-memory ring advances.
