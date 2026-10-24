# Whole-function performance objective

This document specifies the replacement for the congestion-reward objective
reviewed on 2026-10-24. The numerical function, its output contract, and its
accuracy tolerance stay fixed while placement, compute shares, routes,
collective realization, and schedules change. All physical links remain
available to every solver through the common transport realization. An
objective is not permission to remove links from that interface.

Extra relay hops are encouraged whenever their pipelined execution completes
the function sooner or fills useful ingress opportunities. There is no hop
count penalty and no preference for relay-free routes. Keeping the useful-work
numerator fixed prevents accounting inflation; it does not discourage moving
additional wire bytes to improve performance. A regression test multiplies
wire bytes by 100 without changing the score at identical completion time.

## The defect

The former cost was `log(T) + congestion_seconds / T + finish_variance`.
Two completed executions with identical useful work illustrate its reversed
preference:

| Execution | Whole-function time | Congestion seconds | Former cost |
|---|---:|---:|---:|
| A | 0.010 s | 0.010 s | -3.605170 |
| B | 0.011 s | 0 s | -4.509860 |

B is slower and has lower whole-function useful throughput, but has lower
cost. This is a property of the formula, not evidence that a particular
hardware measurement regressed for that reason. A finish-variance reward
has another defect: delaying an early delivery toward the last delivery
can improve its score without delivering any result sooner. Queue occupancy
and equal completion are explanations and possible means to performance;
neither is an independent reward.

## Quantities and priorities

Let `U > 0` be the fixed numerical function's useful work and `T > 0` its
complete execution time. For the DNN harness, `U = rows * width * 4` counts
one logical final float32 output. Replicas, intermediate partials, padding,
and relay hops do not increase it. A caller without a declared byte count
uses one completed function as the work unit. Thus its throughput is
functions/second, not a fabricated byte bandwidth.

The default objective is

```
B_function = U / T
J = log(T) - log(U) = -log(B_function).
```

An equivalent dimensionless form is `log(T / T_reference)` for fixed `U`.
The reference must stay fixed for the workload. Both formulations have the
same derivatives and ordering. The implementation fixes the useful-work
amount and unit at the first complete observation and rejects accounting
changes within that workload. In particular, a slower execution cannot
improve its score by having empty queues, less finish variance, more relay
bytes, or a less demanding configuration-specific roofline.

This objective simultaneously minimizes complete-function latency and
maximizes its useful-work throughput. It does not claim that every finite,
noisy optimizer step decreases latency.

The transaction metric answers a different question. Let `W` be the union
of intervals from a compiled range's first eligibility to its last landing,
including stalls while that transaction is unfinished. Let `U_transaction`
count useful deliveries under a declared, invariant transaction contract.
Then `B_transaction = U_transaction / W`. The observer reports bounds on
`W`; the conservative bandwidth uses its upper bound. It also reports
wire bandwidth separately, counting every hop and padded page.

A function can increase `B_transaction` while taking longer overall, for
example by delaying production and concentrating its network traffic into a
shorter window. Hence no scalar can promise independent global maximization
of `B_transaction` and minimization of `T` when those objectives conflict.
No finite positive weight makes a smooth scalar exactly lexicographic over
arbitrarily small differences either. A strict ordering or a latency budget
would be an additional operator policy, not something inferred from a queue
minimizer.

An explicitly requested fractional tradeoff is supported as

```
J_w = log(T / U) + w * log(W / U_transaction),  w >= 0.
```

The default is `w = 0`. A positive value must be supplied as
`transaction_weight`, or `--transaction-weight` in the DNN harness. Both
useful-work counts must remain invariant. For positive `w`, reducing both
throughputs can never improve the cost, but a gain in one can compensate
for a loss in the other. This is a declared tradeoff, not a guarantee of
strict priority. The upper observed window bound is used consistently;
its uncertainty remains visible in the report.

The existing DNN transaction count measures full partial deliveries. It is
suitable for comparisons of the same collective with different routes,
but it is not an algorithm-independent work count. Comparing full exchange
against reduce-scatter/all-gather must use the unchanged numerical output
contract for `U`, and separately expose each algorithm's transaction bytes.
The default objective already has this property.

## How congestion preferences become useful

The compiler can describe a directed physical multigraph with capacities,
parallel edges, propagation/start costs, queue and registration limits;
an operation and page-dependency DAG; producer releases; source, relay,
destination and reduction placements; storage lifetimes; and legal
collective algorithms under the numerical contract. A fitted compute model
can distinguish fixed call cost from work-dependent cost for each node and
operation. FLOPs, memory traffic and register/occupancy requirements are
inputs to that model, not interchangeable units of capacity.

For commodity `k` and physical path `p`, let `x[k,p]` be a continuous flow
fraction, with `sum_p x[k,p] = 1`. With demand `D[k]`, directed-link load is

```
L[e] = sum_(k,p containing e) D[k] * x[k,p].
service[e] = L[e] / C[e].
```

This shares a physical capacity among every own-output and relay flow using
it. Enumerating two paths through one cable does not create two cable
capacities. Dependencies and releases determine when that service can
begin. Useful simultaneous traffic should remain present on every usable
ingress; creating redundant wire traffic solely to raise utilization cannot
improve `U/T`.

A congestion minimizer's preferences can estimate which changes reduce a
link's service delay or head-of-line delay. They must pass through the
dependency model before becoming a performance gradient. In a simple
completion recurrence,

```
start[v]  = max_(u -> v) (finish[u] + transfer_delay[u,v])
finish[v] = start[v] + compute_duration[v]
T_model   = max_(required sink v) finish[v].
```

Scheduling on shared links adds resource-order predecessors to this DAG;
page-level dependencies permit overlap instead of whole-operand barriers.
For a differentiable surrogate, replace a maximum by log-sum-exp at a
declared temperature. Reverse differentiation weights each delay by its
contribution to the required sinks. A reduction in congestion on a wholly
overlapped, noncritical dependency has little or no effect; a small reduction
on the critical dependency can matter substantially. The relevant chain rule
is `dT/dx = sum_e (dT/ddelay[e]) * (ddelay[e]/dx)` together with compute and
release terms, not `d(sum queue area)/dx`.

The current implementation uses physical path incidence/capacity as a prior
and fits the last 64 complete functions' landing-time residuals against joint
route and compute fractions. These measured residuals include congestion
and release effects; queue area is retained as evidence, not charged again
in `J`. Its routing subproblem now uses a monotone smooth maximum of
predicted finish times plus a proximal movement term. The finish-variance
term has been removed. Increasing any one predicted finish while holding
the others fixed cannot lower the finish component. Smooth maximum with
sharpness 64 differs from the true maximum by at most `log(number_of_edges)
/ 64` in the normalized time units. This is an explicit approximation.

The compute equal-finish model and the routing residual correction remain
model-guided updates. They are not a proof of descent on the true execution
time; their predictions can be wrong. The DNN model retains every declared
serial layer's directed-link finishes and sums the layers' smooth makespans;
it does not average their successive bottlenecks away. It still does not
implement the full typed, page-dependent recurrence above. Within-layer
producer release and page-predecessor critical paths remain refinements.
The model must never reinterpret a lower queue area as compensation for a
slower measured function.

A stricter way to use a congestion model without adding another objective
is to form a positive-definite preconditioner `P` from its sensitivities and
update `z <- z - eta * P * gradient(J)`. For an exact gradient and a
sufficiently small step, the first-order change is
`-eta * gradient(J)^T * P * gradient(J) <= 0`. This guarantee concerns the
same performance objective; it does not require accepting or rejecting a
measured candidate. The current implementation's finite differences,
quantized realization, and independent model corrections do not establish
those exact-gradient assumptions. This alternative is a derivation, not a
claim that it is already implemented.

## Online update and limits

`FunctionFlow` preserves the fixed four-complete-function `+--+` finite
difference stencil, bounded continuous simplex coordinates, gradient moments,
and unconditional updates. There is no winner selection, slower-candidate
rejection, rollback, queue-empty threshold, or transport-side routing choice.
Only the measured cost and the routing model's finish objective changed.
Realization still rounds the continuous state to legal pages/tiles, so the
observed objective is not globally smooth even when the latent update is.

Old partial stencils and gradient moments used a different objective.
`FunctionFlow.of` therefore explicitly rejects unversioned old optimizer
states instead of silently mixing those derivatives with new ones. A saved
configuration can initialize a new controller. Old measurement reports keep
their historical cost labeled `congestion_v1`; new reports use the canonical
recorded `whole_function_throughput_v2` metrics, including fixed and centre
evaluations that never train the controller.

The graph completion boundary defines `T`. For the current harness, it
includes final GPU synchronization and excludes planning, warmup, correctness
checking, clock exchange and between-call model fitting. Those excluded
costs must also be reported when evaluating sustained invocation throughput.
A microsecond saved inside a call is not a claim about setup amortization.

## Lower bounds and evidence

Report the complete function against applicable critical-path, compute
resource, and information-flow cut bounds. Report a particular routing's
transaction window against both `wire_bytes / sum_e C[e]` and
`max_e wire_bytes[e] / C[e]`, with capacity constraints and finite pipeline
fill/drain costs exposed. A cut bound across different collective algorithms
must count the information required by the same numerical contract, not the
traffic of one chosen algorithm. A configuration-dependent lower bound is
useful for reporting its gap; dividing the optimizing cost by a different
bound for each configuration can reward a slower configuration and is not
the objective here.

The compiler's AOT knowledge makes these schedules expressible and removes
runtime address negotiation. It does not prove optimal routes, reduction
placement, page order, compute shares, or achieved performance. Neither all
links carrying bytes nor equal final completion proves every eligible link
was saturated. The measured gap and the validity of each bound remain
separate facts.

Local validation on 2026-10-24 passed 32 whole-function tests. They include
the old objective's counterexample; continuous movement to the latency
optimum when the queue minimum is deliberately elsewhere; invariant scores
under increased hop bytes; explicit transaction preference at equal graph
latency; immutable useful-work accounting; serialized replay; and the
existing unequal-capacity/release and cross-flow residual tests. DNN/report
syntax checks passed. Report compatibility checks covered old and new cost
formats, unknown versions, and two retained complete historical reports.
These checks are not new cluster performance measurements.

## Online node, link and function uncertainty

Hop delays are not known constants. `DelayPosterior` starts every node with
the same mean (`prior_mean`, zero by default) and equal prior standard
deviation. Directed-link and per-function random effects start at zero too.
The initial standard deviation is 1 ms, process standard deviation is 10 us
per complete invocation, and the observation-noise floor is 1 us. These are
explicit model hyperparameters, not measured hardware costs. A routing
model's `delay_prior` may supply other common values; no node-specific bias
is inferred from its machine name or expected compute capability.

After a completed invocation, `delay_observations` validates matching
directed physical page coverage at source and destination, rejecting missing
or duplicated intervals and inconsistent bytes per page. SEND and COPY entry
boundaries may differ: relay dependency refinement can split a SEND while
the receiver retains one COPY. The inference uses a first-frame sample only
where source eligibility and destination first landing were both observed
for the same first page. An interior page's arrival may be bounded by the
enclosing COPY's entire transfer, but that causal bound is not a measured
first-arrival bracket. Such samples are omitted, with observed and omitted
boundary counts and total validated wire bytes recorded in the batch and
posterior report. Entry subdivision never manufactures extra observations.

Each retained sample measures first-page eligibility to first-page arrival
and subtracts one page's known service time,
`page_bytes / directed_link_capacity`. It does not add the entire transfer
duration at every hop. Long transfers on a relay path can pipeline, whereas
their first-frame dependencies still have a latency. The same trace evidence
can show a large first-frame delay when another commodity is occupying the
queue; this is not a bare-cable propagation measurement. The joint posterior
updates once per complete training invocation, regardless of the number of
refined entries or retained boundary samples.

For observation `i`, row `H_i` names its source node, destination node,
directed physical link, and the current function effect. The joint state
is `theta`, with covariance `P`. The model is

```
theta_next = theta + process_noise
y = H theta + observation_noise
P_prior = P + Q
R = diag(bracket_variance + learned_link_observation_variance)
    + clock_design * diag(clock_offset_variance) * clock_design^T
P_new = inverse(inverse(P_prior) + H^T inverse(R) H)
mean_new = mean + P_new H^T inverse(R) (y - H mean).
```

This is a linear Gaussian Bayesian update in information form. The basic
linear filtering result is from [Kalman (1960), *A New Approach to Linear
Filtering and Prediction Problems*](https://people.duke.edu/~hpgavin/SystemID/References/Kalman-JBE-1960.pdf).
Here the delay decomposition, random-walk model, adaptive noise moments and
uniform-within-bracket observation assumption are modeling choices. They do
not establish global routing optimality or make real congestion Gaussian.

Clock-offset uncertainty is shared across observations from the same
invocation. Treating it as independent noise on every traced entry would
incorrectly average it away. The implementation uses a small Woodbury solve
for that correlated uncertainty, then a solve in the number of effects;
there is no cubic solve in the number of traced entries. Per-link observation
variance follows an exponentially weighted predictive residual moment. This
measures variability of the whole observation attributed to that link,
including upstream/endpoint/function congestion; it is not proof of an
isolated cable's variance.

Full covariance matters. A first-frame observation may identify only the
sum of two node effects, one link effect and a function effect. Its evidence
must not manufacture separate knowledge of those four causes. The report
therefore exposes the joint covariance, each effect's posterior parameter
standard deviation, and the observable per-link combination's mean,
parameter standard deviation, and predictive standard deviation. The latter
also includes process and observation variability. An individual random
effect can be negative while the combined delay is positive; such a contrast
is not a negative physical propagation latency. Equal configurable node
prior means preserve exchangeability before distinguishing evidence; they
do not resolve this attribution ambiguity or assert that the individual
physical node delays have been identified.

The DNN supplies this evidence after every complete training invocation.
Posterior predictive uncertainty controls the confidence of the physical
completion-residual fit. Posterior means also supply its pipeline-arrival
prior and analytic routing Jacobian. For a path's entry on link `e`, that
prior adds the first-frame excess delays of its preceding hops, their one-page
service times, and `e`'s excess delay. It does not add preceding hops' whole
transfer windows. The flow-weighted arrival on each link is
`sum_p bytes[p] * fraction[p] * first_frame_arrival[p,e] / link_bytes[e]`.
Its derivative changes continuously as competing source/relay shares move.
Shared-link bulk service remains in the capacity matrix. Observed finish
residuals correct this approximate arrival prior; it is not a new per-hop
objective penalty. Multiplying an operand's bytes leaves its first-frame
arrival prior unchanged. A test holds the measured finishes fixed and
reverses the learned ingress delays: the next continuous routing proposal
reverses as well, proving that these means affect routing, not just reports.
For the DNN's declared serial layers, the routing model now retains each
layer's directed-link finishes and minimizes the sum of their smooth
makespans. Averaging different layers' slow links first could incorrectly
erase their successive critical paths. Arbitrary DAGs still require their
own typed dependencies; this serial model does not claim to represent them.

Tests verify that identical priors stay exchangeable until evidence
distinguishes them, unidentifiable node contrasts retain uncertainty,
correlated clock uncertainty survives 64 repeated observations, JSON replay
is exact, changing congestion updates the state, and whole-transfer end
timestamps cannot change a first-frame observation. Ragged, crossing SEND/
COPY boundaries preserve only genuinely shared first-page observations;
missing and duplicate coverage still fail before inference, including in
omitted interior samples. A retained trace replay
processed 5,952 first-frame observations from 32 previously measured complete
DNN invocations, with a median posterior update of approximately 0.29 ms
after the first import. Its node effects remained substantially uncertain,
as expected from the attribution ambiguity. This was an offline inference
check against existing observations, not a fresh bandwidth measurement.

A subsequent live three-rank, four-layer DNN run supplied 1,536 first-frame
observations over eight training invocations, followed by two centre
measurements. The eight posterior updates and two routing updates completed;
the centre measurements did not train. Its twelve-direction measurement
design had rank 12 for 16 effects, retaining four unidentifiable dimensions.
Individual node means of -32.37, 23.58 and 186.82 us had parameter standard
deviations of 577.62, 577.62 and 565.72 us. Every observable endpoint/link/
function sum had a positive mean. These results demonstrate online inference
with honest attribution uncertainty, not separately identified hardware node
delays. The companion microbench artifact is
`output_data/review-20261024/interface-refactor/BAYES-INFERENCE.md`.
