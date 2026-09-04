# Policy implementation claims

Controlling specifications: [`../SPECIFICATION.md`](../SPECIFICATION.md),
[`../rl-training-spec.md`](../rl-training-spec.md),
[`../joracle-viewer.md`](../joracle-viewer.md), and
[`../ALGORITHM-CONTRACTS.md`](../ALGORITHM-CONTRACTS.md), with the executable matrix
boundary controlled by [`../MATRIX-EXECUTION-SPEC.md`](../MATRIX-EXECUTION-SPEC.md).

Implementation surfaces:

- `xonotic/solver/strat/runtime.py` defines formal state rows and sparse W/L measures.
- `xonotic/solver/strat/game_value.py` defines the cart-game value and projected role
  coordinates consumed by the runtime.
- `xonotic/solver/strat/cast_header.py` and `strategy.py` define learned Gram fusion,
  residual fusion, DPP allocation, SwiGLU, integrated weights, and readout.
- `xonotic/solver/strat/matmul.py` defines the Metal matrix products and their reverse
  products; `dpp.py` composes feature covariance and conjugate-gradient execution from
  those products.
- `xonotic/solver/strat/online.py` and `strat_responder.py` define identity-preserving
  succession and optimization.
- `xonotic/solver/strat/expert_worker.py` and `xonotic/solver/xonwire.py` define
  mesh-owned residual-scale forward evaluation, input cotangents, parameter-gradient
  accumulation, batch commits, endpoint ownership, and the separately persistent scale
  parameter tree.
- `xonotic/solver/strat/instruments.py` and `sv_payload_strategy_io.qc` define the
  literal actuator boundary.
- `xonotic/solver/strat/featurize.py` supplies the canonical stock-waypoint walking
  distance consumed by each located instrument assignment.

The claims are substantiated only when runtime measures show sparse reward support,
separate W/L targets, positive-semidefinite participant and residual Gram spectra,
independent fusion interventions, stable-ID successor mass, distinct push/suppress
effects, and distance-derived commitment for every located assignment. A controlled
cartstate must report a null global nimber while the runtime PW, succession, portfolio
nimbers, and role ranks equal the coordinates carried by that same formal value.
The serialized formal value carries its discretization level count. `measure.py cgt`
reconstructs the value from the literal state and reports a residual measure for every
formal coordinate and for the PW, succession, and loser-rank runtime projections.
`runtime.py` owns the one formal-value and runtime-projection serialization interface;
the responder produces that record and the measurement command consumes it without an
independently inlined wire shape.
Stable-ID succession is exposed as `joined_row_mass`, `successor_source_row_mass`,
`successor_present_row_mass`, and `departed_row_mass`. Joins are measured over current
identities absent from the source roster; departures are measured over source identities
absent from the successor roster. The latter alone removes bootstrap and dynamics mass
for the departed row while preserving its realized sparse reward.
Distributed training is substantiated only when responder and expert producer records
occupy distinct live nodes and scale-update, gradient-atom, input-gradient, checkpoint,
FLOP, byte, and deadline measures are present for the same learner batch.
The curriculum preserves each remote lifecycle script as one quoted shell argument and
records the resulting worker PID. Endpoint ownership is acquired before model allocation,
survives only for the worker lifetime, and permits a successor to reclaim a stale socket
without confusing a queued datagram with process liveness.
The responder transmits the current action before applying the preceding transition's
optimization update. Its response-deadline workload record and post-response optimization
record are distinct. Response and optimization cadence spans keep the responder producer
live through long compilation or update intervals with a zero lower bound, an unknown
upper bound, and no claim of completed work. Retirement drains the responder, then the
expert, and only then closes the game relay.
Remote differentiation is open only inside an explicit gradient batch. Expert response
forwards carry the action deadline; training forwards, input cotangents, gradient
accumulation, and the batch commit do not.
Each reverse call carries the local loss's already normalized cotangent. The expert sums
all corresponding parameter-gradient atoms, clips the complete batch gradient once, and
performs one optimizer update; dividing those atoms by their count would apply a second
and incorrect batch normalization.
`measure.py matrix` exposes numerical and timing measures for all three forward address
modes, both reverse operands, and the DPP contraction against an FP64 dense reference. It
also names the exact execution schedule used by each product so relaxed MPP residuals are
not confused with threadgroup results.
Behavioral studies publish observed and missing coordinate masses beside aggression,
robustness, pressure, survival, and objective-conversion measures. They do not collapse
support into a completeness judgment or collapse behavior into an improvement score.

The participant Gram matrix has an explicit feature construction. Participant `i` has
the direct-sum feature `rival_metric(q_i) / r_e^(1/4)` together with
`one_hot(team_i) tensor team_metric(q_i) / r^(1/4)`. Its inner product with participant
`j` is exactly the implemented rival product plus the same-team-masked team product.
The residual-feature Gram matrix is exactly `residual.T residual / physical_rows`, and
the DPP feature Gram matrix is exactly the normalized instrument-row inner-product
matrix. Ridge-adjusted orthogonalization benchmark matrices are called normal matrices,
not Gram matrices, because their operational input is the adjusted matrix rather than
the augmented feature realization.

Commitment measurements report graph distance, stock walking time, learned extension,
and emitted horizon. The emitted horizon must equal walking time plus extension and may
exceed the previous thirty-second ceiling.

DPP inclusion is evaluated through its feature-side covariance and a dimension-derived
conjugate-gradient composition executed by repository-owned Metal matrix kernels. It
does not call a host linear-algebra fallback or build an instrument-by-instrument
inverse.

Policy velocity has one cadence. The strategy step computes
`tanh(weight + velocity * cadence)` once, uses that integrated tensor in the action
density, returns the same tensor as the next policy state, and persists it by stable
participant/instrument identity before the next request. The release measure compares
the action tensor, returned tensor, and persisted successor tensor coordinate by
coordinate at the strategy cadence.

The replay ring interns the eleven count-invariant chorus arrays consumed by the
composer. A transition stores endpoint references plus actions, behavior density, role
masks, sparse reward, and discount. Learned outputs and pair tensors are recomputed by
the strategy composition when sampled and are not replay records.
The learner has no generic reward-override interface. Immediate records call the formal
W/L transition measure directly, while delayed credit passes only a discounted sum of
those same sparse transition atoms through the explicitly named `sparse_return`
coordinate.

`xonwire.py` is the sole tensor framing and frame-wave scheduler for both RDMA and the
local mesh/expert datagram relay. The local envelope carries the destination node, live
page size, and literal page count; socket-buffer capacity controls only one scatter/gather
system call. An actual `EMSGSIZE` reduces that call's measured batch, never a later tensor
extent or the number of rows transported. Deadline slack is measured after delivery and
does not terminate a tensor, gradient, or batch-control response wait.
The expert retains each completed response wave verbatim and replays it on response-cadence
socket timeouts until the following request implicitly acknowledges delivery. Replays
perform no model or optimizer operation and retain the original request, tick, kind,
shape, and values.

Request waves also replay while the sender drains responses and credits. The reserved
header words carry a requester session, and every operation has a monotonically advancing
request ID, including gradient calls. The resident worker retains completed-session
watermarks and replays its most recent response instead of reapplying a completed
operation. This is loss recovery for a serial requester, not a durable transaction across
worker crashes. The proof and mixed-generation boundary are in
[`../RELEASE-CLOSURE.md`](../RELEASE-CLOSURE.md).

The existing local scale counterfactual is not yet a same-parameter comparison after
remote training or an independently restored expert checkpoint. The local tree is frozen,
while the worker updates its tree. Its output difference and duration are observations;
hardware necessity or same-checkpoint speedup remains unsubstantiated until parameter
identity is synchronized and recorded for both sides. This limitation does not retract
the demonstrated mesh integration or earlier two-host computation.

Checkpoint realization attempts the complete finite source parameter tree through the
strict module interface and restores the complete preceding tree if that attempt raises.
Optimizer restoration likewise assigns only the complete named source tree after exact
name, shape, and finite-coordinate realization. Source-only, live-only, shape-different, nonfinite, loaded,
and restored masses remain explicit measurements; none is used to splice a hybrid model
or optimizer state. The same whole-tree operation owns local and remote optimizer
moments.
