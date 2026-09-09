# State reduction and execution RCA — September 6

This is a historical review of the representation repairs preceding the common
policy/value readout. [POLICY-PROGRAM.md](POLICY-PROGRAM.md) defines the current
program. The referenced test outcomes describe the earlier implementation; their
suite and harnesses have been removed and do not define completion criteria.

The operator's controlling requirements are quoted in
[SPECIFICATION §§17–21](SPECIFICATION.md): a learned velocity field over Havocbot's
state views, exact exponential relaxation, no semantic state inhibition, and fixed
numerical execution rather than data-dependent graph construction. Paging is a
storage and execution mechanism. It does not authorize summarizing away inputs.

## Critical errors and repairs

| Error | Lost distinction or failure | Repair and witness |
| --- | --- | --- |
| Averaging linearly encoded pages before the policy | Swapping values between fixed addresses preserved the global representation | Retain addressed pages through learned row encoding and interaction; fixed-address swap test |
| Broadcasting one policy vector into every page's additive readout | A policy change gave every page the same mean-rate increment | Read each page's own policy row; context-conditioned opposite-page optimization test |
| Omitting residual, duration and relaxation time from the rate predictor | The field could not condition on its current integrated state or timing | Include all three in learned inputs; finite nonzero derivatives and rate-change tests |
| Hand-built event summaries | Spatial filtering, decay and an eleven-feature observation vocabulary replaced the native data before learning | Delete the vocabulary; all seventeen native fields reach an embedding before local spatial integration or temporal contraction |
| Explicit cart/team frames replaced with eight formal-game values | Cart geometry and nonrepresented-team score changes could disappear from policy inputs | Raw OBS/CART/TEAM projections; first-projection spies and causal geometry/score tests |
| V-cell integration deleted with the event vocabulary | A valid spatial/temporal requirement disappeared with its invalid preprocessor | Live navigation rows, graph neighbor tables and learned local integration; complete raw event history and post-embedding contraction |
| Events replicated into every participant's global row block | Global multiplicity depended on roster size | Project each event once and gather its embedding only through local support |
| Independent scale-feature Gram/probe broadcast | An unrelated global compression was presented as the required Gram interaction | Delete that branch; retain real global/team row Gram and row-local sigmoid SwiGLU experts |
| Missing owner-observation connection to pages | Same-team bots with identical page metadata could not use different own observations | Add each owner's learned observation residual to its pages; opposite-page learning test catches this failure |
| Normalizing the complete neighborhood message | A uniform age/distance factor canceled under RMS normalization | Preserve message magnitude with postprojection `asinh`; full-policy old-event effect tends to zero |
| Baseline factory retained the deleted summary width | Direct constructor tests passed while the responder built a 25-field event projection for 17-field native rows | Remove the phantom width fields and test the actual factory for every policy arm |
| New constant balance output broke compiled stock policy | MLX compiled the first twelve outputs but raised `unordered_map::at` for the thirteenth constant scalar | Express the zero as an empty-array scalar reduction, retain compiled execution, test complete stock and baseline outputs |
| Unnormalized input SwiGLU | Native postembedding outliers near 100 became gate/value branch outliers near 2,000 | RMSNorm only inside that FFN branch; retain the full embedding residual and validate large-magnitude pass-through and gradients |
| Normalizing the combined local and Gram result | A shared aggregate suppressed local page distinctions; nonzero gradients alone missed the limitation | Preserve the unnormalized local residual path and normalize only the interaction branch; supervised opposite-page test now converges |
| Entropy floor after averaging across state words | High entropy in one word could conceal collapsed entropy in another | Apply the nonlinear penalty per represented word, then form the scalar loss |
| Treating stable Metal templates as a complete execution contract | Python rebuilt numerical graphs, and empty-page/routed-group branches changed kernel work | Prepared compiled local forward/emission and combined loss/backward/accumulation/optimizer; masked Metal gather, scatter, projection and contraction without occupancy-dependent returns or skips |
| Dense full-coordinate telemetry matrices | Retaining real state width made covariance and affine diagnostics quadratic in coordinate count | Store exact matrix factors and compute row/Frobenius norms in sample space; reconstruction and viewer tests |
| Repeated coordinate labels expanded into checkpoint JSON | The first eight-bot RDMA attempt overflowed NumPy's Unicode representation while saving | Intern strings, store label sequences as uint32 table-index arrays and metadata as UTF-8 bytes; large-table exact roundtrip and RDMA restart |
| J artifacts used a second label serializer | Repeated full-width labels expanded a manifest to about 3 GB and a viewer JSON document to 384 MB; shutdown exceeded the harness deadline | Share `array_tree.py` between checkpoints and reports; binary columnar viewer sidecars and exact coordinate pagination; retain legacy readers and failure evidence |
| Validation treated integrator rank as policy expressivity | Correct `d residual / d rate` said nothing about distinctions lost before the policy | Separate integrator numerics, representation sensitivity/learnability, execution reuse and actual training/application tests |

The [pre-repair audit](../measurements/state-representation-audit-20260906.json)
records numerical counterexamples for the first three defects. Earlier documentation
incorrectly treated full output width and nonzero gradients as sufficient evidence
of a general learned state mapping. Current owner pages now describe the retained
row representation and the limits of those proofs. Historical reviews remain
historical evidence, not the current implementation specification.

## What remains a reduction, and why

The [axis audit](POLICY-AXIS-AUDIT.md) now describes the repaired live topology.
The formal game calculations remain useful for rewards and reporting, but are no
longer a replacement input representation. Raw fields reach learned projections
before conditioning. Presence masks concern padding, not semantic filtering.

A matrix product sums products. The exact identity `(R Rᵀ) V = R (Rᵀ V)` avoids a
quadratic intermediate; it does not replace source rows with an unlearned average.
The model retains every encoded global row and adds learned interactions to it.
Team-restricted Gram products use the same identity, with literal team rows and
structural membership supplied as tensor data. Local neighborhood integration
contracts spatially supported source rows after their learned projections; the
destination observations and their owner pages survive. The former independent
scale feature-Gram broadcast is gone. The routed block processes each row through
sigmoid-selected SwiGLU experts and merges only that row's selected contributions.

The model has finite learned widths and ranks. Learned projections can lose
information; removing hard-coded pre-reduction is not a proof of injectivity or
universal control. The baseline models deliberately have less interaction capacity.
All parameterized arms receive the complete raw row families and use the same full-width rate interface. The linear/FFN comparators deliberately sum learned global rows and use linear local integration; the main model preserves its global row axis.

W/L values are scalar predictions read from each owner observation row after its declared processing; there is no final page/event sum in the main value heads. The joint Gaussian log likelihood sums per-word log densities. PPO and
critic objectives reduce weighted samples to the scalar being differentiated.
RMS normalization inside learned branches computes a scalar norm to rescale that
branch's vector; the local residual row remains available. It is an explicit model
operation with an inductive bias, not an input average replacing addressed pages.
The ordinary expert-balancing objective also measures token allocation and mean normalized router affinity; padding has no allocation or objective mass. Reported entropy, rates, drift and covariance have explicitly statistical meanings.
These operations occur at the declared learned output, objective or report; they
are not feature summaries substituted for the model's observations. The entropy
floor fix matters precisely because moving a reduction across a nonlinear function
changes the objective.

## Fixed execution means more than fixed templates

`inputs.py` owns lossless physical padding and presence masks. `execution.py` owns
prepared program capacities; `online.py` owns a combined tensor loss, backward pass, source-group accumulation
and optimizer program. Source-group capacity is prepared alongside frame capacity;
unused group slots carry zero objective mass.
Within a prepared capacity, values, addresses, presence, actor eligibility, returns
and expert occupancy change tensors rather than Python control flow or graph shape.
The paged kernels execute their configured page/tile loops regardless of occupancy.
Gather reads an explicit zero sentinel for inactive rows. Scatter writes inactive
rows into an allocated sink. Contraction selects each group's contribution in
cooperative registers instead of skipping unmatched pages. This spends arithmetic
on unused capacity; the execution guarantee takes precedence over that shortcut.

Sampling, joint likelihood and exact residual integration have their own prepared
emission program over the same physical capacity. The responder selects each
policy's already integrated outputs, eliminating its eager per-response tensor
construction and redundant integration.

The local update compilation captures both input and output model/optimizer state.
MLX's `nn.value_and_grad` installs parameter arrays in the module; omitting the
output capture left trace intermediates in that state. The combined update avoids
returning a full gradient tree to Python between backward and optimizer execution.
A numerical test compares a three-source update padded to four slots with separate
gradient accumulation, and verifies that preparation restores parameters and the
optimizer step exactly. The same prepared program also handles fewer active groups.

The subsequently removed measurement harness changed
state values, masks and model weights over eight prepared forward/emission and eight
prepared gradient/optimizer solves. Its process-local probe counted Metal device
`newBufferWithLength:options:`, `newBufferWithBytes:length:options:` and
`newHeapWithDescriptor:` calls, with a forced allocation as calibration. The recorded graph-reuse check held for forward, emission and the combined update. Earlier
narrower probes reported zero new device allocations. Expanding the workload to
include emission and multiple loss sources exposed intermittent allocations even
with unchanged shapes and traces. Combining backward and optimizer improved the
execution boundary but did not eliminate every allocation. Explicit synchronization
and additional cache warm-up did not reliably fix this; those attempted workarounds
were removed. The failed allocation observations remain evidence, not successful
qualification of the operator's full execution requirement.

The upstream [MLX Metal allocator](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp)
requests reusable cached buffers and creates a buffer on a cache miss; its
[buffer cache](https://github.com/ml-explore/mlx/blob/main/mlx/backend/common/buffer_cache.h)
uses a size window. Those mechanisms are not a fixed per-program buffer arena.
Attributing any particular observed miss to reuse order or timing is still an
inference: the probe records device calls and sizes, not MLX allocation call stacks.
A complete repair needs explicit buffer ownership/lifetimes and an arena allocated
at realization, including intermediates in the array backend. Repeated warm-up is
not a substitute for that guarantee. The harness exits nonzero when it sees a new
counted allocation, independently of its graph-reuse result.

The whole server is not yet allocation-free. Input packing and reporting allocate
host objects, and growth beyond a prepared capacity requires preparation of another
graph. Renaming that preparation would not remove its latency. Optional split-RPC
execution keeps RPC outside compiled row regions so tracing cannot freeze a remote
reply. Its full loss orchestration remains eager and reports `staged_loss_builds`;
it is not qualified by the compiled-local allocation result. The current tested
placement executes the whole learner on the Mini and uses RDMA for native game
state and response transport. Persistent input/intermediate arenas and a fully staged split-RPC
backward schedule remain execution work, not excuses to restore feature averaging.

## Actual validation and remaining evidence

[The earlier evidence directory](../measurements/state-representation-20260906/README.md)
contains both-machine test logs and execution probes, the native view fixture,
a complete socket-transport match with interruption/recovery, and resident-bridge
RDMA matches with the complete representation. The failed checkpoint attempt is
retained alongside successful recovery evidence. No bridge was restarted to repair
that application storage failure.

Those records describe the preceding graph. [Feature-integration evidence](../measurements/policy-feature-integration-20260906/) retains subsequent numerical observations and runtime attempts, including failures. The removed tests recorded agreement within their chosen tolerances and specific learned
representation distinctions, actual optimizer updates, source-attributed bot
applications and process recovery. They do not establish competitive playing
strength, useful game-reward convergence, sustained many-team throughput, a kernel
stall's cause, or recovery from an RDMA driver deadlock.

## Reporting follow-up from the combined-update RDMA run

The first combined-update match reached eight updates for both policies and saved
its checkpoint. Graceful shutdown then exceeded the harness's 60-second deadline.
The retained J NPZ had a 3,006,398,524-byte manifest, despite numeric covariance
already being factored: a separate artifact serializer repeated coordinate labels
throughout JSON. Its viewer document was about 384 MB. A second cleanup signal saw
an already exited learner and raised before the harness wrote its normal report.
This was an application/reporting failure, not evidence of a driver deadlock.

`array_tree.py` now owns lossless array/tree storage and interned labels for both
checkpoints and J artifacts. The viewer sidecar is a binary columnar report; HTTP
responses contain the explicitly selected exact coordinate slice. Filtering and
pagination do not average measurements or remove coordinates from the artifact.
Legacy NPZ trees and JSON views remain readable. The cleanup harness independently
attempts learner shutdown, engine quit and bridge observation, and retains their
errors in its report rather than losing the original failure.

The large-label codec test preserves all 10,000 labels in twenty strata while
keeping metadata below 500 KB. The HTTP test requests coordinates beyond the first
page and checks the actual final value, variance and J sample. Both test machines
exercise the shared serializer and viewer path. The final native/RDMA evidence
records whether this application failure was repaired at actual match width.

## Native magnitude review and input FFN normalization

The original operator explicitly named RMSNorm and SwiGLU for the learned
operator after a Gram intermediate (August 29, transcript line 9776, preserved
in [the provenance records](../measurements/policy-feature-integration-20260906/provenance.json)).
That statement supplies the components, not an explicit mandate about every
input-FFN placement. The ordinary pre-normalized residual construction here is
also supported by the following measured arithmetic failure and the current
instruction to preserve features until their first embedding.

The [read-only counterfactual](../measurements/policy-feature-integration-20260906/input-ffn-prenorm-counterfactual.json)
uses a retained native eight-bot frame with 2,031 pages per owner, 88,201 witnessed
words and identical checkpoint parameters after five updates. It compares only
`x + SwiGLU(x)` with `x + SwiGLU(RMSNorm(x))`. No fields, vocabulary, rate limits,
source values, game state or service configuration change in this comparison.

| Quantity | Unnormalized branch | Normalized branch |
|---|---:|---:|
| Input embedding RMS / maximum absolute value | 14.63 / 99.29 | 14.63 / 99.29 |
| FFN branch RMS / maximum absolute value | 38.79 / 2,058.02 | 0.129 / 0.490 |
| Rate-mean RMS / maximum absolute value | 0.212 / 6.03 | 0.096 / 1.50 |
| Gaussian scale RMS / maximum value | 0.143 / 4.56 | 0.058 / 1.14 |

The same comparison at initial parameter seed shows the same amplification.
Normalizing only the FFN branch preserves the complete learned embedding residual;
it does not repeat the earlier mistake of normalizing away the only local path.
The regression checks large-input scaling, row independence and the input Jacobian,
while the existing opposite-page optimization test still converges.

This does not eliminate unconstrained exploration or establish the cause of an
individual native view fault. For 26,883 reference-like witnessed words in this
frame, the modeled expected number whose newly forced residual crosses an integer
rounding boundary falls from about 11.3 to 7.1 per interval, remaining nonzero.
Previously accumulated residuals also persist until relaxation removes them.
`GetResourceLimit` switches on a Resource entity and reports invalid entity choices;
Havocbot can also take different float-dependent branches under small residuals.
The existing view-fault unwind and stock retry remain relevant. No reference or
field restriction is introduced to hide this remaining behavior.

Matrix-family policy version is now 19 because this changes its action distribution;
baseline version remains 17 and architecture version 10 because the tensor tree is
unchanged. The preceding version-18 counterfactual and runtime evidence remain
historical evidence. The new version must be validated on its own trajectory.
