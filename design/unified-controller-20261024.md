# One controller for a complete function

`rdma/controller.py::WholeFunctionController` is the configuration and evidence
boundary around `whole.FunctionFlow`. A decision vector may contain TP shares,
per-node engine fractions, every physical route, and compiler-declared schedule
operands. They receive one whole-function objective and one finite-difference
update. There is no measured winner, acceptance threshold, standing allocation,
failure rollback, or separate per-operation engine optimizer.

The default cost remains `log(T/U)`: `T` is observed completion latency for the
entire declared function, and `U` is invariant useful work (one function if no
logical-output byte count is supplied). Intermediate deliveries, padding, and
extra relay hops never enlarge `U`. Relaying is encouraged whenever it lowers
the complete function's cost. Congestion observations guide physical models;
they do not reward slower delivery merely for having a smaller queue.

`proposal()` returns an immutable identity, continuous vector, quantized operands,
and phase. `completed(seconds, proposal=...)` requires the exact executed
proposal and workload. The first invocation can calibrate the current operands;
it does not create a zero-offset finite-difference sample. Four subsequent
complete evaluations form the existing `+--+` stencil. A slower observation is
still part of that derivative. A delayed observation under another proposal
cannot be relabeled. `failed(reason)` preserves failure evidence without
fabricating latency or restoring a historical allocation.

`completed_rollouts(records)` accepts delayed complete episodes at their original
applied operands. `rollout.RolloutInbox` joins rank records by episode, without
blocking newer complete episodes behind an older missing record. The in-process
`RolloutStats` worker owns statistics I/O and learning. Submission only enqueues
records and polls a local inbox; a missing peer does not stop function execution.
Several completed episodes can enter one update.

The delayed estimator fits a local response model with weights
`target_probability / behavior_probability` over the declared compiled-plan
sampling support. Probabilities use Q40 integer masses; behavior support must be
positive. Deterministic observations of the same policy have weight one. Age is
not an importance probability. Exact applied coordinates, rather than the newest
proposal's coordinates, enter the regression. A positive ridge handles an
unexcited design; flat costs yield exactly zero movement. This local model does
not claim an unbiased current-policy gradient for arbitrary stale trajectories.

Torch `partition.evaluations` keeps one realized generation for its submitted
sequence and exposes the learned proposal separately in `evaluation_state`.
It no longer performs a CPU all-gather or discards old-proposal timings on each
iteration. `realize_evaluations` explicitly applies a proposal during construction
of the next generation, after the previous generation's uses have retired. It is
not inserted between iterations or repeated `evaluations` sequences. Installing new generations concurrently with
execution still requires independently retained compiled bindings; the present
mutable communicator does not yet provide that activation interface.

`declare(comm, calls)` calls the common native `ncclMeshCommPlan` with the complete
typed graph, retaining count and explicit-demand arrays through configuration.
`from_comm` obtains physical topology and semantic directed bytes from
`ncclMeshCommPlanDemands`; it does not guess an all-pairs byte proxy from tensor
size. `Fabric.allocate` seeds every simple physical path, including alternatives
to a direct link. `realize(comm, calls, proposal)` installs the next path weights,
recompiles the same graph with its new allocation operands, and refreshes its
actual demand metadata before execution. No tensor operation replans a route.

## Migrated boundaries

- `allocate.Balancer` and `allocate.Coupled` have been removed, together with
  their independent model search, compass search, acceptance, rejection, and
  physical-engine exclusion algorithms. Old state can supply initial
  configuration operands through the shared adapter, never the obsolete policy.
- Torch engine sharing learns every operation/shape allocation jointly from
  an explicitly completed enclosing graph. Primitive timings are diagnostics.
  Shared physical resources are metadata rather than a prohibition on time
  sharing. The next vector's declared shards are prepared between functions.
- The microbench rates adapter jointly updates TP and normalized engine-fraction
  decisions. Fractions are projected onto the current node extent by a shared
  AOT lattice helper. Its measured boundary is the maximum rank latency across
  the complete compiled prefill/decode invocation, excluding configuration.
  End-to-end request dispatch latency is reported separately. It does not
  substitute a sum of tensor stretch times for that observation. A configuration
  still awaiting preparation is recorded without attributing its timing to the
  proposed operands. Preparation failure leaves a failure record and the intended
  configuration, without a performance rollback.
- Xonotic planner ticks and context solves retain immutable allocation operands
  throughout the function. Their existing numerical reductions carry the prior
  function's elapsed time. Rank-max completed timing updates the controller
  after the current function ends; stale repeated identities are omitted from
  the derivative. Their entire reduction/gather graphs are declared and their
  exact native demands initialize route vectors. Context-solve plans are
  suspended between functions so request broadcasts cannot consume graph slots.
  The gather source is produced directly into a region operand.

## Integration boundary and guarantees

The microbench consumer uses one full-fabric communicator and native per-call
contributor/recipient masks. Configuration admits every path-capable relay,
including alternatives to direct cables and nodes with zero numerical work.
The shared administrator forwards the existing configuration request to each
resident worker. It implements no tensor transport.

Each worker records its actual local crossings. The common native adapter merges
those typed records at configuration time, preserving each local dependency order
and inserting missing relay-only calls at the corresponding recording boundaries.
It declares the entire invocation sequence once, installs every proposed path
weight through the native interface, and reports the actual physical cables,
installed paths, compiler-derived useful demands and proposal identity. Missing
numerical contributors or contradictory recorded orders are configuration errors.
They are never filled by guessed tensor sizes or fabricated observations. Host
LiteRT and GPU-recorded classes issue the same native calls; zero-compute workers
pass NULL operands under the native role masks. No payload ACK or initial
one-word collective barrier is introduced.

Finite differences are empirical updates of a nonconvex, drifting function.
The implementation does not promise monotonic improvement or global optimality.
The AOT physical bound is a model bound, not proof of optimal execution. Reports
must retain measured whole-function latency, useful bandwidth, actual wire-byte
accounting, and the applicable physical roofline separately.
