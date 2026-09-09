# Policy program: native inputs, common IR, learning and outputs

The controlling requirements are the user quotes in [SPECIFICATION.md](SPECIFICATION.md),
including the September 6 common-IR and whole-program review instructions. This
page describes the implemented program; neither tests nor historical successful
runs supply its specification. Matrix policy version 21, baseline version 19,
architecture version 12. Older checkpoint and native-state layouts are incompatible.

## One data contract and one output computation

`inputs.py::native_inputs` joins literal native pages to occupied observation rows,
builds event visibility and local support, and returns the complete `ChorusArrays`
input tuple. This is one ingress function; there is no second generic assembler
that invents omitted native fields. `Frame.capture` records this shared source
independently of either learner. `Wally` owns
the main learned parameter tree. `strategy.py` composes its feature projections,
local neighborhood, row-Gram interaction, routed SwiGLU and final shared SwiGLU.
`strategy.read_heads` is the common readout used by the main model and both learned
comparison arms. `state_steering.py` owns numeric distribution, integration and
native page encoding. These are distinct responsibilities in one composed program.
There is no actor-private state encoder or alternate value representation.

Let I be state owners, O delivered observation slots, P pages per owner, J carts,
K teams, E event rows retained for the current episode, and V/A/C navigation
nodes/edges/cells. Physical extents include explicitly masked storage. Every
supplied source field reaches its first learned projection unchanged; invalid
physical storage is identified by the native presence/type metadata.

| Source before first learned projection | Shape | Meaning and downstream use |
|---|---|---|
| OBS plus requested duration and tau | O × 86 | All 84 native observation fields; PRESENT at native slot 29 distinguishes vacant client slots. Feeds shared player context, both policy and value outputs. |
| Native view pages | I × P × 1556 | At word width 256: source words, current residuals, address/type layout, requested timing, 15 headers, per-word read time, held velocity and command-anchor residual. Feeds the common IR. |
| CART | J × 18 | Complete cart state and geometry; shared strategic context, never direct authoritative-world writes. |
| TEAM | K × 7 | Complete team state; shared context and separately sourced role/reward accounting. |
| EVENT | E × 18 | Original 17 fields plus deposit-time EPISODE. Team ownership, geometry and age determine local integration after embedding. |
| Navigation nodes / edges / cells | V × 4 / A × 3 / C × 2 | Serialized map geometry through the shared local featurizer. |
| Presence, owner indices and neighborhood tables | Explicit separate axes | Structural ownership, addressing and spatial support; no output vocabulary or feature summaries. |

At general page width w, the learned page input width is 6w+20. The old unused
`relations` tensor is deleted; the actual neighborhood relation has one owner.
Per-word source values are the values captured at their last native read. Their
literal read times now accompany them; export time does not pretend they are a
simultaneous fresh VM snapshot.

## Shared representation

Each source family has its own learned linear embedding. The main model conditions
magnitudes after projection with asinh. Owner observation context enters its page
rows, and eligible local event/navigation messages enter observer and owner-page
rows. This preserves source/address distinctions through learned transformations.

Local integration uses learned destination/source metric factors and value
projections within navigable V-cell support. Distance and event age weight the
embedded messages. The learned event-age coefficient cannot be canceled by
normalizing the complete local message. Navigation itself does not age. Unowned
native topology events use team 0 and are available to present observers; positive
team IDs retain team ownership. Native live waypoint links and the offline metric
graph are distinct geometry sources, not falsely claimed identical duplicates.

The local integral and its VJP are sparse Metal operations lowered by `tensor_metal.py`.
They access projected key/value rows directly, without allocating an
observer × neighbor × feature tensor. For source address b in observer a's
neighborhood, let `s_ab = dot(q_a,k_b)/sqrt(D)` and let `w_ab` retain the existing
spatial/temporal weight. The operation is `U_a = sum_b w_ab*s_ab*v_b`. For an
incoming derivative g_a, its derivatives are:

```
dq_a   = sum_b w_ab * dot(g_a,v_b) * k_b / sqrt(D)
dk_b  +=       w_ab * dot(g_a,v_b) * q_a / sqrt(D)
dv_b  +=       w_ab * s_ab * g_a
dw_ab  =              s_ab * dot(g_a,v_b)
```

Repeated source addresses accumulate into the same source derivative. Comparison
arms without a metric use `s_ab=1` and zero metric derivatives. The existing
projection and temporal-weight programs receive these derivatives normally. The
kernel specialization consists of the realized model width and metric setting;
its source addressing uses the supplied index table. Each observer's SIMD group integrates its supplied neighbor slots. Backward
source-row contributions use floating-point atomic addition. Floating-point
association can therefore differ from the MLX expression; no neighbor subset or
page receives a separate normalization.

The global row axis is `N = O + I*P + J + K`. Row FFNs are independent along N.
The interaction operator is:

```
K[a,b] = dot(R[a],R[b])/sqrt(r_global)
       + same_live_team[a,b] dot(T[a],T[b])/sqrt(r_team)
message[a] = sum_b K[a,b] Z[b]
```

The global action is evaluated exactly as `R @ (R.T @ Z)`. Team membership groups
its additional cross factors. R and T are each projected once. The former separate
feature-Gram/probe/broadcast and unused alternate Gram implementations are deleted.
The row-local routed SwiGLU uses sigmoid top-k affinities, normalized selected gates,
ordinary auxiliary balancing and no shared expert. The final dense SwiGLU is shared
by all output heads, after the routed residual.

For each owner, H retains that owner's final observation row and every addressed
page row: `H[I,1+P,D]`. This is a structural view of the aggregate network output,
not an actor-specific encoder. Absent rows remain zero. The reported IR is this
same H with the final two axes flattened; reports remove physical page padding.

## Heads and output correspondence

One bias-free learned projection produces all four output groups at once:

```
Z = H @ A                   A[D,2w+2]
mu[i,p] = 0.01 * (Z[i,p+1,:w] + Z[i,0,:w])
a[i,p]  = Z[i,p+1,w:2w] + Z[i,0,w:2w]
log_sigma[i,p] = log(0.01) + asinh(a[i,p])
V_W[i], V_L[i] = sum_rows Z[i,:,-2:]
```

The mean and scale-coordinate maps and the two scalar value readouts are linear
in the same H. The distribution's positive-scale transformation is explicit and
parameter-free. The owner contribution permits the observation row to contribute
to every owned page's output; page contributions remain separate. The scalar
value contraction occurs after the complete learned representation. It is not a
pre-embedding average or a private observation-only critic. No state/residual
value array enters the heads through another learned route.

| Output | Input correspondence and consumer |
|---|---|
| Sampled rate, one coordinate per writable present native word | Same owner/entity/generation/page/word address as its source and residual inputs; only the bot's private view receives it. |
| Mean and scale | Parameters of that rate distribution, learned from H; sampled rates are acknowledged and become held-velocity inputs on following native captures. |
| Requested duration / tau and response identity | Explicit control and transport metadata, echoed to the native integrator; actual installed values are returned in STATE headers. They are not secretly learned outputs. |
| W/L values | Linear predictions of the distinct reward-defined returns. Successor states and realized rewards provide targets; values do not need fictitious native-state input counterparts. |
| Common IR, relationship factors, local messages, expert statistics | Reporting views of actual intermediate computation. They are not additional control channels or hidden recurrent state. |

World observations, events and topology are conditioning inputs. They do not imply
permission to mutate the authoritative cart/team/map state. When Havocbot reads a
world field through a captured state page, the corresponding addressed view word
is in the control surface. Unread memory is not invented as an output vocabulary.
No auxiliary autoencoder or reconstruction loss has been added to manufacture
input/output symmetry. Random sampling keys are exogenous sampling state, not
features supplied to let the policy predict its own exploratory draw.

The earlier VERA query-value imitation task had genuine user provenance. Its
pre-final-IR tap and loss are retired under the latest common-IR instruction;
they are not retained as aliases or misrepresented as invented historical work.

## Read-only Havocbot configuration

The September 7 user instruction makes the randomized Havocbot skill configuration
an input rather than an actuator. OBS carries the twelve realized bot-definition
traits and the global skill setting in the formerly reserved columns 30–38 and
43–46. Column 83 identifies an available realized configuration. These literals
reach the same observation projection as the other native fields.

Native STATE type tags reserve bit 21 for read-only words. The source value, tag
and address remain inputs. `rate_distribution` excludes those words from sampling,
log likelihood and entropy, with zero mean and scale coordinates. Emission and
wire encoding clear their velocity and residual, including recovered commands.
The native adapter masks incoming commands again and clears retained forcing on
read. Skill entity fields cannot receive stores during a view invocation.

`PRVM_ViewSkill` is the explicit native symbol inventory: the twelve definition
traits, `skill`, `autocvar_skill`, the reporting alias `bot_global_skill`, and the
stock aliases `skill_save` and `sk`. The `plc_ob_f_` observation aliases inherit
that classification. Field loads, scalar/vector copies, arguments and returns
carry read-only metadata across VM scratch words and function frames. This is an
exception to the previous unrestricted output surface, authorized by the latest
user instruction. It is not a restriction on learning effective movement or aim.
It does not establish an adversarial sandbox for arbitrary bytecode or complete
information-flow protection through every arithmetic/builtin transformation.

## Native integration and temporal ownership

STATE now has a 15-word header and six 256-word arrays: source, current residual,
type/bit encoding, last-read time, held velocity and command-anchor residual.
Its width is 1551. Header fields include owner/schema/export time/applied sequence,
entity/address/generations/session, command source time/duration/tau, received
sequence and per-page forcing. The schema fingerprint includes the packet layout.

STRATEGY remains width 524: twelve metadata words, current source residual and
sampled rate. On each native read, the installed command defines the residual
through the exact exponential law, then the residual modifies the private read:

```
r(t+dt) = exp(-dt/tau)*r(t) - tau*expm1(-dt/tau)*u
```

After the forcing duration, the residual decays without continued forcing. Native
integer/reference handling preserves native word representation and rounding.
Ordinary Havocbot writes and game consequences remain ordinary engine behavior.
A reported view fault unwinds the invocation. The callback is not replayed: its
earlier game effects already happened, and replay would duplicate them. Subsequent
bot invocations continue normally. This is not a transaction rollback.

EVENT carries its episode identity from deposit, not arrival. Complete transport
batches retain their source session/tick/request. Observation history is partitioned
by actual session and episode; selecting the current episode does not assign old
or future rows to it. Checkpoints and action journals carry that ownership. Late
rows remain attributed to their original episode in reports as well.

`RuntimeFrames` owns transport serialization too: pending synchronized snapshots,
completed events, partial reassemblers and their received-fragment bitmaps, outcome
records, deduplication identities and session watermarks. Runstate also retains
game packets queued during RPC waits. Live consumption and journal recovery call
the same snapshot-acceptance operation. Restored snapshots already consumed by the
journal cannot become new observations a second time. This preserves received
transport state at a checkpoint; it does not recover packets never received or
uncheckpointed fragments lost in a crash.

## Optimization and execution

Each learned policy owns its complete parameter tree and optimizer. W/L predictions
use their respective role rewards and value targets; only that policy's own first
observed action applications contribute actor likelihood gradients. Value learning
can use observed teams and retained historical experience. Bootstrap and advantage
are detached at their declared boundaries. Shared H receives both actor and critic
gradients; separate output columns retain their separate objective roles.
`ActionHistory.advance` supplies the actual source arm/version, applied-row mask,
successor identities, return and elapsed interval. The learner requires those fields;
it no longer invents ownership or successor rows when a caller omits them.

The loss consists of the actor objective, W/L value losses and ordinary expert
balancing. The inherited per-word Gaussian entropy floor, rate L2 penalty and
auxiliary query imitation loss are deleted. Entropy is reported. AdamW weight decay
remains an explicit optimizer hyperparameter. No invented state-category penalty,
rate cap or output field whitelist replaces those deleted losses.

`execution.emit_policies` owns the complete multi-policy emission: prepare source
frames, evaluate each policy, retain its full counterfactual distribution and
select only the assigned team's rows for publication. Journal recovery uses the
recorded issued vectors and log probabilities. A cache scoped to one emission or
one learner update shares each immutable source/capacity packing; it holds no
learned embeddings, output predictions or cross-update source copies.

Local and distributed execution use the same symbolic graph, derivatives and
optimizer arithmetic. `scale_rpc.py` declares placement of the actual cross-factor
or routed scale operator. `tensor_mesh.py` derives each region's imports, exports
and VJP from that graph. Worker program identity includes the source participant;
separate policies retain separate parameter generations. Model ablations set static
fusion strengths on the same composition. The persistent execution and lifetime
rules are described below.

`checkpoint_state.py` interprets policy checkpoints for both inference and training.
Both match native schema, architecture, policy version and reward semantics before
installing a whole parameter tree. Inference permits declared architecture-equivalent
comparison arms with the same reward contract; optimizer continuation additionally
requires the exact policy arm. Incompatible or unreadable sources are reported and
the complete existing model remains available. Equal tensor shapes alone are not
evidence of equal semantics.

Reports reconstruct the complete source feature vector from the same immutable
Chorus frame during live execution and after checkpoint/spill recovery. Source
arrays/labels are shared between decisions. They include all owners, context rows,
actuator state and structural tables, not merely the current owner's raw words.
Live and restored per-owner view reports use the same address selection. History
byte accounting includes issued velocity, likelihood and ownership arrays as well
as the shared source frame and reporting arrays.
Report-only source construction is timed separately from response preparation,
inference, native publication and policy comparison.

## Persistent execution and transport

`execution.py::PolicyProgram` realizes the input capacity and invokes
`persistent_policy.py`. The latter traces these same model functions and
`OnlineLearner._tensor_loss` once into `tensor.py::Graph`, including the common
representation, both heads, sampling, integration, VJP, contribution accumulation,
global gradient clipping and AdamW. Placement changes the owner of a graph region,
not its algebra. The former independent MLX compilation and tensor-packet worker
programs are removed from this runtime.

Sampling uses Philox4x32-10 and Box–Muller in the recorded graph. A retained
two-word host counter supplies each invocation key without a separate GPU key-split
operation. Checkpoints and run state retain that counter. This preserves the
Gaussian law, but does not reproduce the former MLX sampler bit for bit; the
training contract records both the generator and key schedule.

The ten source extents are symbolic dimensions. `tensor_metal.py` emits kernels
whose dimensions and addresses come from metadata; changing those extents does
not change shader source or pipeline identity. Matrix products, including routed
expert products and their adjoints, use 64 × 32 output tiles with a 32-wide inner
tile. Expert routing produces an integer row table without reducing input rows.
Reductions run across SIMD groups and retain their declared full logical axes.
The local neighborhood reads its supplied sparse index table directly.

`tensor_runtime.py::Executable` plans buffer lifetimes and records Metal indirect
commands. Inputs, outputs, parameters, moments and accumulated gradients retain
storage; compatible intermediate lifetimes share storage. Reshape and detached
views alias storage. Capacity growth repacks the arena and command dimensions
between invocations, retaining learned state. Kernel compilation is independent
of that growth. NumPy and checkpoint-facing MLX views retain the underlying Metal
allocation, including when a later realization replaces the active arena.

The trainer stages two frames and one transition. Each contribution completes its
VJP and ordered gradient/metric accumulation before those input slots are reused.
One complete contribution fold supplies one global clipping factor and one AdamW
update. Parameter updates for a remotely placed region execute with that region;
the learner receives the resulting parameter and moment state before advancing
its update count. AdamW retains the configured beta values, epsilon, weight decay
and bias-correction setting. Sampling uses Philox4x32-10 with Box–Muller conversion;
the supplied key and native-word index determine each draw. Different policy arms
use the same key and physical capacity for their counterfactual draws.

`tensor_mesh.py` sends graph/control descriptions through xonwire and tensor bytes
through scoped `mesh-functions.h` calls. A transfer plan coalesces tensor spans
into fixed receive/transmit windows. Metal gathers and scatters those windows;
the Python transport does not serialize floating-point operands into RPC packets.
A complete input bundle that fits a receive window retains its native page borrow
through the numerical region. The same arithmetic kernels read those pages through
their view metadata. The GPU also populates persistent storage for subsequent
parameter references. Larger bundles are gathered on the GPU into the persistent
arena before the region executes. No GPU command waits for an absent network
operand, and neither path allocates tensor buffers inside a solve.

Remote parameters have a generation per tensor. Forward/VJP references reuse that
generation; the corresponding remote optimizer region updates it. A new binding
clears the worker's generation claims and uploads the learner's current state.
Received results occupy separate persistent staging storage until the worker has
completed the entire invocation. A recorded GPU copy then publishes them to the
local graph. An interrupted network transfer therefore cannot overwrite inputs or
partially install an optimizer result before local recovery. A local GPU error is
reported as a failed calculation rather than retrying a partially executed commit.

Transport progress owns page acquisition, actual GPU-completion observation and
release. Compilation runs on a separate worker thread. Rebinding waits for prior
GPU borrows to retire before changing views or capacity. Idle compiled-worker and
control caches expire after four minutes; retained replay and checkpoints are
separate learner state. Runtime errors preserve game I/O, report the failed region
and permit the learner's recorded local region to complete it. Subsequent work
attempts a fresh remote binding. The bridge's verbs lifetime is unchanged.

Reports expose arena bytes/realizations, shader variants, submissions, completed
remote regions, direct-page regions, parameter uploads and local recoveries.
Worker arithmetic estimates label matrix FLOPs and logical operand bytes; they
are not hardware utilization counters. The current evidence is local compilation
and bounded numerical calculations, recorded in
[the implementation ledger](POLICY-IMPLEMENTATION-20260907.md). The new native
transfer path has not been exercised on a live mesh or deployed by this change.

## Physical and representational limits

Growing raw event histories can fragment exact-source-label J strata. No summary
feature is substituted to hide that reporting limit. Arena capacity, network
window size and model dimensions remain explicit physical choices; all occupied
source rows retain their addresses and presence information.

The default per-row IR width and Gram ranks remain finite model capacity choices.
At D=128 and w=256, each page's 512 mean/scale coordinates vary through a local
rank of at most 128 for fixed head weights. The owner/page addition uses the same
projection and does not double that bound. Exact integration
and full Gaussian sample support do not establish universal controllability or
playing strength. No width was chosen merely to preserve a benchmark FLOP count.

V-cell support currently measures metric-graph length, not walkable floor area.
The native view captures actual reads, not every possible future VM intermediate.
An applied sequence denotes an attempted view invocation and can include a fault
followed by stock retry; the fault counters remain visible, and failed attempts
are not silently removed from learning. Global delivered opponent rows remain
available to the Gram, so team-local event ownership alone does not establish a
partially observed or privacy-isolated game model.

Repository-owned tests and verification harnesses were deleted at the user's
instruction. Source review, compilation and operational observation remain evidence
about implementation, never a replacement specification. This change compiled the
native engine and gamecode locally in the [preceding review](../measurements/policy-whole-program-20260906/README.md).
The [subsequent static flow refactor](../measurements/policy-static-flow-20260906/README.md)
used production syntax and CLI import checks only. Neither deployed, started
training or loaded RDMA.
