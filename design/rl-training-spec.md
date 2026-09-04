# Live strategy training

This document describes the executable strategy runtime. The verbatim requirements
remain in `SPECIFICATION.md`; realized behavior is measured from the running server and
responder.

The September 4 [joint-policy learning contract](JOINT-POLICY-LEARNING.md) adds
architecture-matched stratCGT-PPO and terminal-win PPO training within the same live
match, with separate complete optimization states and shared implementation. It also
defines the fresh-actor/historical-value data mixture and its predictive-target limits.

## Runtime

Training is the process that answers a running Xonotic payload server. The server
publishes participant observations, cart states, and perception events. The responder
selects one strategy instrument per participant, returns typed target kind and identity
plus gain, commitment, and spawn delay, then learns from the next observed server state. There is no simulator or separate
bootstrap environment in this path.

`solver/strat/curriculum.py` samples maps, team counts, players per team, cart counts,
bot/human controller mixtures, skills, perturbation regimes, and off-policy player
counts. The curriculum builds the payload overlay once, then gives each match its own
user directory and exact copies of `progs.dat` and `csprogs.dat`; a failed build is a
recorded match failure rather than silent stock gamecode.
Successive study cycles advance the map coordinate by the complete
repetition-by-perturbation span. The finite map catalog is therefore traversed rather
than resetting to the same prefix when a cycle changes only its random seed.
The optimization arms are `matrix_fusion`, `terminal_win`, FFN, and linear; the fixed default
and matrix-fusion intervention arms exist only in the separately mirrored study schedule, so
training does not duplicate held-out pairs or label a parameterless default as updated.

The server and responder belong on opposite mesh nodes. `--server-host` stages the
generated userdir and gamecode with SSH/SCP, launches the remote dedicated server, and
runs the responder and checkpoint locally. `--remote-engine`, `--remote-basedir`, and
`--remote-run-root` name the corresponding paths on the game node. Without
`--server-host`, local launch remains useful for non-mesh command diagnostics but is
not represented as two-node training evidence.

## State value

The observed cart projection supplies the current path-to-victory holder and a bounded
relative hierarchy for every team. It does not supply the unknown FPS transition
function.

Each gather carries route-sequenced behavioral counters for every participant. Those
counters remain literal observations used to measure aggression, robustness, pressure,
survival, and objective conversion. They do not enter an optimization reward.

The only reward atoms are exact cart-role boundary events:

```
b_i = -1  if projected winner i loses that role
       1  if non-winner i moves upward in loser rank
       0  otherwise

rW_i = -1  if projected winner i loses that role, otherwise 0
rL_i =  1  if non-winner i moves upward in loser rank, otherwise 0
```

Each atom is lifted to the rows belonging to that team. All other rows receive zero.
Sparse boundary rewards prevent hand-authored combat utilities from defining the
self-play objective. Held-out behavioral measures establish whether the learned policy
realizes more aggressive and robust competition under controlled perturbations.

`W` and `L` remain asymmetric value estimators selected by the player's current cart
role. They are separate linear probes of the exact matrix-fusion/SwiGLU representation consumed
by the policy. `W` receives only `rW`; `L` receives only `rL`. Role changes stop
cross-bootstrap between the heads.

## Linear algebra and dimensions

Let `k` be the team count, `l` the participant count, and
`P in {0,1}^{l x k}` the player-to-team incidence matrix. Let `n in N^k` be the
team-nimber vector, `q in {0,1}^k` the one-hot projected winner (or the zero vector when
there is none), `u = 1 - q`, and `C_ij = 1[n_i > n_j]`. The loser-rank vector is

```
rho = (I - diag(q)) C u + (k - 1) q                 in N^k
```

and the two role-boundary event vectors are

```
rW = -q * (1 - q')                                 in {-1,0}^k
rL =  (1 - q) * 1[rho' > rho]                      in {0,1}^k
```

`P rW` and `P rL` lift boundary events to player rows. Thus each transition carries two
sparse `l`-vectors, not a dense activity utility or one global scalar broadcast to
contradictory shared-policy rows.

For the final representation `H in R^{l x 128}`, the value projections are

```
vW = H thetaW + bW 1_l                             in R^l
vL = H thetaL + bL 1_l                             in R^l
```

Each head is a shared `128 -> 1` linear map applied independently to every player row;
the pair has a `128 x 2` weight interpretation, but there is no compression from all
players to one value. With player winner mask `m = Pq`, role changes terminate the old
return rather than cross-bootstrap the other head:

```
AW = m       * (r + gamma m'       * vW' - vW)  in R^l
AL = (1 - m) * (r + gamma (1 - m') * vL' - vL)  in R^l
```

The critic fits `vW` only on winner rows and `vL` only on loser rows. The shared policy
receives the compatible rowwise signal

```
L_policy = -(1/l) sum_p (AW_p + AL_p) log pi(target_p, gain_p, commit_p, spawn_p | s)
```

with categorical target probability and the three-dimensional learned Gaussian
actuator density forming one joint behavior probability. The output contract carries
mean, log scale, and density mass separately at fixed shapes; the default arm's zero
density mass denotes its literal deterministic controls rather than overloading a tensor
width with different sampling semantics. The implemented clipped
off-policy importance ratio multiplies each advantage.
Consequently shared parameters learn from many non-contradictory row objectives rather
than from one scalar that pretends every participant should take the same strategic
side.

## Observed dynamics

The server supplies the exact source observation and cart rows, complete row-aligned
featurization input, selected J, policy intervention, delivered action, subsequent
state, active route, and realized counters for every transition. The J-oracle retains each source atom by
`(epoch, response sequence, participant)` and forms the exact state-machine pushforward
measure over its later route-owned atoms, both jointly and disintegrated by assigned
policy arm and actual behavior policy.
The `matrix_fusion`, `participant_fusion_ablated`, and `residual_fusion_ablated`
functions are additionally evaluated on the identical source frame with one shared
model and one realized checkpoint even if multiple checkpoint paths were configured.
The J-lens integrates each independent intervention's literal J-coordinate delta,
action-distribution total variation, actuator mean and log-scale deltas, value delta, instrument-weight
velocity delta, and coupling delta before action sampling. These are direct paired
intervention measures; realized route, event, and outcome measures remain server-owned
coordinates joined afterward by response sequence.
The same additive matrix-fusion intervention is folded directly from every responder telemetry
row into each study record, independently of the lower-cadence whole-mesh reporter
snapshot retained alongside it.
Once the whole-mesh roofline sampler supplies a center, both training and held-out
study schedules preserve its measured total participant population while redistributing
that population across center-first team counts. Cart and team axes vary below and above
the center; changing team count no longer multiplies the measured participant target.
`LocalDynamics` fits two locally action-linear models
to the observed transitions. Their mean supplies corrective action coordinates and
their disagreement supplies an exploration coordinate. Telemetry records one-step
error, ensemble disagreement, and the smallest singular value of the learned local
action matrix as numerical measures.

This is the operational use of the control picture in
[Brown, Papadimitriou, and Roughgarden](https://arxiv.org/abs/2406.18805): preserve or
reach a target region while learning enough local response structure to correct
perturbations. No regret or controllability theorem from that paper is asserted for
Xonotic.

## Count-independent policy

Teams, participants, carts, rivals, items, cells, and instruments are rows. Shared
pointwise projections and invariant reductions produce one representation and one
scalar action score per participant/instrument relation. Adding a row never changes a
learned tensor shape. No absolute team, player, or cart positional encoding exists.

The final participant-instrument rows are RMS-normalized, form a symmetric Gram matrix,
mix by that matrix, and pass through one wide SwiGLU residual. W/L probe the
RMS-normalized expectation of those same rows under the policy's literal sampled-action
distribution. The value reduction is therefore invariant to instrument count and
changes when the realized allocation distribution changes. The DPP marginal signal and
integrated instrument weights feed the action head. `action_mass` is the literal team
observation-support measure; the policy learns a density over every action in its
support, without hand-authored team, cart-role, or alive-state exclusions.

## Runtime measurements

A running Xonotic server and its responder emit the measure. Each response exposes
`loser_ranks`, both length-`l` value vectors, both role-specific advantages and rewards,
winner/loser row counts, role-change fraction, tensor shapes and finiteness,
learned-dynamics error and disagreement, and the local-control singular values.
Checkpoint observations include architecture, optimizer state, update count, replay
mass, and update count before and after a handled stop and resume.
The response producer lease begins when a synchronized server frame enters strategy
composition and remains live until the action tensor is transmitted. Its in-progress
records carry a zero lower bound and an unknown upper bound; the completed response
record replaces those bounds with the analytic interval. The optimization producer
lease then covers the preceding transition update independently.
Replay retention is bounded by configured byte mass so training coexists with game and
audio workloads. The operator-requested initial count budget is 1,024 retained states;
`--replay-capacity 0` leaves retention to byte mass. Neither is a mesh workload extent.
Historical value sampling selects configuration, match and state. Completed observed
returns replace stale predictions as replay targets; ordinary zero-reward states are
retained alongside consequential states.

Policy comparisons are mirrored: two policies exchange team sides under the same map,
seed, roster, skill, and perturbation. A delivered cart is one observed round outcome.
All arms in a response receive the same categorical and actuator random keys. A mirrored
leg therefore exchanges policy functions across team slots without also exchanging or
advancing different random streams.
Repeated records at the same block coordinate are paired one-to-one by observed leg
order. Extra records remain unmatched-leg mass; they are not expanded into a Cartesian
product of fictional contests.
Per-player-time rates use a right-endpoint Riemann sum over the server's engine-time
coordinate and the participants actually present in each response. Runtime joins and
departures therefore change measured exposure instead of being expanded to configured
match duration. Selected-policy exposure and exact active-source exposure are separate
coordinates. An interval whose active source is absent contributes to global
unattributed player-time mass and is never reassigned to the newly selected arm.
Selected assignment rates use selected-policy row mass. Route-owned applied-action,
goal, touch, and execution rates use the active source-policy row mass, with the
source-matched-current submass reported separately; response lag therefore cannot put
one policy's numerator over another policy's denominator.
Map-space measures define an advanceable cart as the conjunction of a nondegenerate
path, a finite stock-playerbot navigation route from the shared spawn set, and zero
rider-volume gap segments. The report keeps each factor, their conjunction, spawn/cart
clearance residual mass, and the corresponding team–cart incidence measures separately.
The participant scatter/gather aligns each current chorus to every retained source roster.
Joining players do not close or erase older causal sources. When every source participant
remains present, the aligned successor retains bootstrap and dynamics mass; departures
produce numerical missing-successor mass while the source group retains its realized
action reward with zero bootstrap and zero dynamics mass.
Every transition publishes `joined_row_mass`, `successor_source_row_mass`,
`successor_present_row_mass`, and `departed_row_mass`. The first is the cardinality of
current stable participant identities absent from the source roster; the last is the
cardinality of source identities absent from the successor roster. Their distinct
domains prevent a join from being mislabeled as a departure or used to suppress an
older source transition.
The study carries frame mass, total engine-time coordinate mass, finite-coordinate
mass, non-finite mass, and non-monotone frame mass with every arm measure. Its finite
coordinate submeasure is therefore visible rather than a silent input filter. Mirrored
behavioral lifts are integrated whenever both legs carry arm measures; the presence or
absence of a capture event only changes the separate round-outcome measure.
Route-owned outcome and event measures likewise carry total row mass, source-attributed
row mass, numeric coordinate mass, finite coordinate mass, and their complementary
masses. These are disintegrated by exact source arm; separately named global masses
retain the complete observation population. An unavailable event source is not inferred
from the participant's current route. An unavailable source sequence or non-finite value
stays observable instead of disappearing from the study population.
Derived damage, kill, cart, and competition rates are defined only on their observed
coordinate submeasures. A genuinely observed zero remains numeric zero; an absent
coordinate has zero observation mass and a null derived value.
At a finite observation horizon, normalized cart depth and control are retained as
continuous outcome coordinates; ties remain ties. The study reports every record's
provenance and process observations, every block's leg multiplicity and round mass, Elo
ratings, paired score differences where the paired estimator is mathematically defined,
and empirical distributions by arm and perturbation. It does not discard a record or
turn these quantities into a policy ordering.

Every numeric arm coordinate is also retained as a finite submeasure overall, within
each perturbation, across perturbation expectations, and as a displacement from the
baseline expectation when baseline mass exists. This applies equally to routed pressure,
realized damage and kills, survival, cart control, objective conversion, competition
utility, and action entropy. Robust competition is therefore represented by its complete
perturbation-indexed outcome measure, its minimum score expectation, and the variance of
the perturbation expectations; aggression is represented by selected, applied, executed,
damage, kill, and objective-conversion coordinates. Neither word is a Boolean label.
Spawn schedules are disintegrated by assigned arm and expose scheduled, realized, and
pending event mass, latency, lane, spot, cohort, generation, participant count, physical
spot count, and simultaneous slot count. The mirrored legs exchange arm ownership of
team slots, so those spawn coordinates remain available beside every behavioral lift.

Elo coordinates use the unsmoothed observed score expectation. No fictitious win or
loss is added. A pair at empirical probability zero or one contributes an extended-real
coordinate with positive- or negative-infinity mass and no fabricated finite difference.
The directed coordinates retain `matrix_fusion` versus its initial state, versus
`participant_fusion_ablated`, versus `residual_fusion_ablated`, versus FFN, FFN versus
linear, and linear versus default even when a
finite multi-arm rating is not identifiable. Multi-arm ratings are emitted only when the finite
pairwise design plus its zero-sum coordinate has full numerical rank; otherwise the
report publishes rank deficiency and the exact pair measures without manufacturing an
identifiable rating.

Build return codes, process return codes, log diagnostic occurrences, telemetry mass,
checkpoint path/load/update coordinates, and configured versus realized populations
remain literal run measures. They do not rewrite a match into a failed study record and
do not prevent an emitted training checkpoint from becoming the next training input.
The study's `optimization_measures` disintegrates every numeric learner-update coordinate,
positive gradient mass, output checkpoint artifact, held-out source and live loaded-weight
fractions, checkpoint update count, source-only and live-only leaf mass, shape difference,
non-finite weight mass, load exception, and source/live schema coordinate by policy arm.
A checkpoint is one named parameter tree. The consumer attempts the whole tree with strict
names and shapes and restores the preceding complete tree if the operation raises; it never
intersects “compatible” leaves into a hybrid policy. Architecture, arm, version, and reward
metadata remain reported coordinates rather than gates on execution. Learner optimizer
moments are restored as their complete named tree rather than overlaid by matching name.
“Optimized” therefore names an observed update and checkpoint-consumption path rather than
an architectural label.

Before the first gradient update in every training match, the responder atomically writes
the complete starting parameter, optimizer, replay, update, architecture, arm, and reward
state. The curriculum retains the earliest such artifact for each training lineage.
Every later trained checkpoint carries that initial artifact's SHA-256, and the held-out
`initial_policy` arm loads the retained parameter tree strictly while using the same matrix-fusion
architecture and policy function as `matrix_fusion`. Mirrored matrix-fusion/initial legs therefore measure
the realized behavioral effect of the actual optimization lineage rather than comparing
training with a newly sampled random model. Both arms can use the same stateless Gram
worker: each call supplies its own residual rows and probe, without loading a second
model or sharing a separately trained parameter tree.
The study emits one direct lineage atom per matrix-fusion/initial record containing the trained
checkpoint hash, the initial hash embedded in that trained checkpoint, and the hash of
the initial checkpoint actually loaded by `initial_policy`.

The report retains assignment-level Elo and separately disintegrates Elo by each observed
realization coordinate. The latter coordinate is the empirical distribution over both
mirrored legs of source/live arm, version, architecture, reward contract, parameter-tree
masses, checkpoint SHA-256, initial-lineage SHA-256, non-finite mass, and load exception,
with checkpoint-update moments carried as a
scalar measure. A configured arm whose realized source differs therefore occupies a
different measure atom instead of borrowing the configured arm's interpretation.

The participant-fusion intervention loads the same checkpoint into the same model and
sets only participant coupling to zero. The residual-fusion intervention independently
sets only the residual-feature Gram-matrix contribution to zero. The report
retains realized competition utility, damage, routed pressure, objective state, and the
full perturbation distribution for `matrix_fusion` and both ablation arms. Optimized FFN,
optimized linear, and fixed-default arms remain parallel measured interventions.
Aligned damage and kill atoms compare the event subject directly with the emitted
`RIVAL` target kind and target identity retained for that response sequence; instrument
names are not reinterpreted as target types.
`matrix_fusion` and `participant_fusion_ablated` execute the same remote scale operator
under the same placement. `residual_fusion_ablated` executes the same operator and
multiplies only its returned residual contribution by zero.

Every arm returns the same canonical twenty-field policy output. Fields absent from an
architecture have literal zero contribution or an empty measure: baseline coupling,
DPP inclusion, dynamics guidance, uncertainty, and instrument-weight velocity are zero;
baseline scale-operator measures are empty; incoming instrument weights pass through
unchanged. FFN and linear J are their one-coordinate action score, while default J is
its idle-instrument indicator. Their pooled row is the expectation of that exact J under
their own categorical distribution. Primary values, auxiliary values, actuator mean,
and actuator log scale occupy distinct learned output coordinates. Native J, belief, and
pooled widths are tagged by arm and coordinate labels in mixed matches; they are never
broadcast or padded to the matrix-fusion width.

The [single-policy manifest](POLICY-STATE-CONTRACT.md) specifies current placement,
tensor framing, pullbacks, one optimizer/checkpoint owner, lifecycle and counterfactual
identity. It supersedes the former frozen-local/independently-trained-worker arrangement.
The loss and policy definitions above are unchanged.

The responder executes the parameterized policy and the matrix worker executes the
Gram contraction and its pullback. Runtime records retain distinct host/node roles,
full requested/processed input extents, returned context rows, widths, routed-bank load,
transfer bytes, compute time and deadline slack. The input/output row ratio is not a
completion fraction for a contraction; `remote_processed_row_fraction` records coverage.
No model identity is placed in floating-point feature or response cells.

The action is transmitted before optimizing the preceding transition. Response and
optimization work retain separate envelopes. Worker action forwards carry the action
deadline; background forwards and pullbacks do not. Cadence spans retain live workload
leases during long computation with zero known lower work and unknown upper work,
replaced by counted terminal records. Replay/retry intervals do not truncate tensors.

The 8787 stream supplies host FLOP/byte bounds and variance. Operating-point sampling
uses supported player/team/cart/instrument and per-host capacity coordinates, action
deadline loads and estimated local-only loads. Distributed points require distinct
responder and matrix producer nodes. Missing counters retain zero observation mass and
null values, not invented throughput or pressure toward a population ceiling.

The curriculum gives the engine relay and worker the same `MESH_EXPERT_SOCKET`.
The existing `--expert-command`/`--expert-socket` names and execution-log slot remain
launch-interface labels, not a second policy. The worker owns only an advisory socket
lease, reassembly buffers and a response cache. No scale checkpoint is staged or collected.

## Telemetry

Every response records the realized team/cart/player counts, cart identity and motion,
projected winner and succession, participant controller and behavior policy, target and
behavior log-probabilities, assignments, belief diagnostics, team health/armor/ammo/
weapons/alive/speed/power state, cross-team strategy focus, online losses, model error,
disagreement, local-control singular value, the reward coefficient contract, and the
separate delivered-response and active-route sequence joins to route, goal, touch,
server-feature changes, and realized outcomes. `solver.strat.joracle.metrics` aggregates the
retention, acquisition, hierarchy-flip, resource-change, focus, controller, behavior,
and learner measurements, including unclipped gradient norm and clip threshold, without
manufacturing missing outcomes.
