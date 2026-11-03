# Fixed policy vocabulary: provenance and deletion inventory

September 5, 2026. The initial investigation identified the fixed vocabulary and
its dependents for deletion. The subsequent implementation replaced the Python
policy, loss and reporting bindings with the full state-rate interface and deleted
`instruments.py`. Native state capture and writeback, including the old QuakeC
consumer, remain unfinished. Deployed game binaries have not changed. Current
implementation and verification status: [POLICY-STATE-STEERING.md](POLICY-STATE-STEERING.md).
The inventory below identifies the pre-replacement source and remaining dependents;
it does not assert that every listed definition still exists.

The relevant operator requirements are the original messages below and
[`SPECIFICATION.md` §§9, 17, 18](SPECIFICATION.md). The implementation references in this
document locate the defect; they supply no requirements for its replacement.
The accepted state-vector and exponential-relaxation algebra is recorded in
[`POLICY-STATE-STEERING.md`](POLICY-STATE-STEERING.md).

## Finding

The available record shows a chain of assistant-authored restrictions. Gameplay
examples became an instrument vocabulary; prototype allocation code became a design
basis; a reachable goal-rating hook became the alleged unique control interface;
literature about discrete allocations supplied an analogy; documentation marked the
result firm; subsequent rewrites preserved it while changing the learned middle.
This is observable provenance, not evidence of anybody's private intent.

There is real operator precedent for a parametric adapter, fixed playerbot code and
skill parameters, stochastic policy training, and a separation between strategic
computation and stock navigation. None of those requirements supplies the seven
instrument kinds, the descriptor's field selection, three named actuator values,
the eight-field response, or the handwritten strategy interpreter.

The current candidate population is variable. The restriction is the fixed semantic
vocabulary and decoder, not a fixed number of rows. There is no reconstruction loss
or learned autoencoder in this path. The concrete information bottleneck is a
handwritten lossy encoder followed by a learned computation and a handwritten decoder.

## Original record

Source session:
`/Users/mdot/.claude-personal/projects/-Users-mdot/d3ad4328-26f8-44b9-a10d-b1c11e13f095.jsonl`.
[Selected complete messages](../measurements/policy-vocabulary-provenance-20260905.json)
preserve timestamps, message identifiers, line numbers and authorship. Task-result
notifications use a `user` envelope in that file but are agent-authored; they are
explicitly distinguished in the extraction.

| UTC time / transcript line | Speaker | Evidence and its significance |
| --- | --- | --- |
| Aug 28 06:10:18 / 2285 | Operator | Matrix results used as inputs to physics, planning or motion controllers over a delta-t function. |
| Aug 29 04:25:18 / 6724 | Operator | Interpolable goals or policies and separation from millisecond bot actions. |
| Aug 29 04:51:51 / 6818 | Assistant | Declares that every tick is an allocation vector: push, suppress or concede. This is an early explicit reduction in the inspected record. |
| Aug 29 08:43:07 / 8486 | Assistant | Reports a handwritten greedy assignment using dominance for combat posts and proximity for carts. |
| Aug 29 09:08:11 / 9020 | Assistant | Reports a learned logit field over 20 instruments added to closed-form priors. Learning was placed inside an already selected vocabulary. |
| Aug 29 20:51:28 / 9602 | Assistant | Proposes allocation weights over push-cart, suppress-cart, contest-post, hunt-player and grab-pickup, consumed through goal ratings. |
| Aug 29 21:14:49 / 9625 | Operator | Says the existing demonstration source is mere code and has nothing to do with the specification or algorithm. Requests per-player mixing/linear transformations. |
| Aug 29 21:18:21 / 9639 | Assistant | Writes instrument descriptors and learned key/value projections; assigns meanings such as aggressive, zoning and escort to value vectors. |
| Aug 29 21:37:38 / 9654 | Operator | Requests team/player weight derivatives rather than instantaneous strategy or tactic dictation. |
| Aug 29 21:51:04 / 9678 | Operator | Endorses DPP coupling and velocity on an integrated weight state. |
| Aug 29 21:57:17 / 9720 | Operator | Challenges the assistant's invented leader-character ontology. |
| Aug 29 21:58:10 / 9729 | Agent report | Calls Q-DPP's partition constraint exactly “one instrument per bot.” This is the agent's literature analogy, not an operator requirement or an independently verified theorem in this audit. |
| Aug 29 21:58:51 / 9737 | Assistant | Repeats that analogy to the operator. |
| Aug 29 22:05:44 / 9776 | Operator | Requires a learned operator, with RMSNorm and SwiGLU, to map intermediate signal to output behavior. |
| Aug 29 22:11:55 / 9790 | Operator | Requests a review of how the computed results become weighted changes in playerbot behavior. |
| Aug 29 22:23:18 / 9844 | Assistant | Calls `navigation_routerating` the “single” lever and equates instrument weights with its base-rating argument. |
| Aug 29 22:32:55 / 9848 | Operator | Corrects engine-tick integration and again rejects interpreting arbitrary source as specification. |
| Aug 29 22:47:52 / 9881 | Operator | Explicitly orders deletion of the earlier strategy hooks and solver implementation. |
| Aug 29 23:41:46 / 10116 | Operator | Describes a parametric adapter between playerbot state-action trajectories and the shared strategy policy. |
| Aug 29 23:44:25 / 10119 | Assistant | Narrows this to strategy in cartstate/instrument terms and an adapter rendering where frozen bots commit. |
| Aug 29 23:54:03 / 10131 | Operator | Clarifies that the C program is treated as stop-gradient/no-gradient because playerbots are frozen. |
| Aug 30 23:54:51 / 10734 | Operator | Requires strategic decisions in the learned matrix partition and ordinary playerbot navigation. The examples are followed by an ellipsis, not an exhaustive action enum. |
| Aug 31 00:38:10 / 10892 | Operator | Explicitly rejects feature-engineering behavior into the solver; requires wide learned projections from the final IR. |
| Aug 31 06:48:45 / 12981 | Operator | Challenges per-rival instrument shape reasoning and reiterates that each playerbot needs an action vector. |

The most direct contradiction is visible in the operator's original words:

> finally, moving on: how do we implement strategy and tactics as a state space which implements d_stratweight/d_t and d_tacticsweight/d_t (think team,player here) changes to weighting over change in time for each playerbot, rather than dictating an instantaneous tactic or strategy?

The later cadence correction also matters:

> "Engine-side per-tick integration" two scale integration both decoupled from engine tick for trivial reasons (woha, offloading, the rdma thesis, plugging in 17 more mac minis does not result in an allreduce inside of the engine core loop, only how often the buffer the engine reads from is updated)

An earlier reply in the present investigation repeated the engine-integration
assumption. The second quote corrects that reply too. The transcript does not require
putting strategy integration or a collective operation into each game tick.

## How an implementation became an alleged requirement

The branch history supplies the documentary link. These observations come from
`git log main --reverse -p`; no historical state was selected or deployed by hash.

On August 29 at 14:56:45 -07:00, `strategy-layers-and-modality.md` put a
“Control interface — allocation -> bot behavior, via a goal-rating bias plus a
spawn/travel commitment” under a section labeled `[FIRM]`. The same document
correctly said the demonstration source was not authoritative, while presenting
its restricted interface as settled. It also assigned instruments to the key/value
side of the operator. Both restrictions predate the interface review the operator
requested at 22:11 UTC.

At 18:26:15 -07:00, the “implement RL stack from spec” change added `head.py` with
input width `2 * n_instruments`, consisting of marginal-inclusion and appetite.
On August 30 at 15:18:42 -07:00, the “adopt Codex asymmetric W/L redesign” change
introduced `instruments.py` with eight enum members and hand-authored descriptor and
relation rows, and a `21 -> 32` head. Later rewrites removed the small head and some
heuristics, but retained the instrument ontology. The present enum has seven members.

This was not solely a subagent inventing an interface in isolation. The main
assistant introduced the vocabulary in its analysis; generated documentation
certified it; the interface and literature reports reinforced it; implementation
and later repairs inherited it. The current investigation's earlier defense of a
better-described candidate list repeated that same inheritance.

## What made the restriction sound plausible

| Actual premise | Unsupported restriction introduced from it |
| --- | --- |
| Cart-state analysis and examples such as collecting a rocket launcher | Treat those examples as an exhaustive set of policy output symbols. |
| Existing stock goal-rating functions | Treat a convenient control path as the only legal state transformation. |
| Frozen Havocbot program and seeded skill parameters | Conclude that the learned output can only be a destination plus a few scalar knobs. |
| A parametric adapter | Substitute a handwritten semantic interpreter for the parametric map. |
| DPP-related reasoning and logit sampling | Require one member of a handwritten action vocabulary per bot. |
| Different policy and game cadences | Hold a chosen goal for a computed timeout instead of representing the requested state evolution. |
| Wide IR and gradients through the learned middle | Claim the surrounding representation and decoding are now complete. |
| A working transport and existing checkpoints | Preserve the old output schema across successive rewrites. |

These are inferences visible in the record. They are not requirements supplied by
the operator. A numeric state schema may describe engine fields and types without
declaring what strategies those fields must mean.

## The defect as a composition

The existing restriction can be written as

```
observed state -> handwritten selection/descriptor E
              -> learned candidate scores F_theta
              -> sampled instrument and three controls
              -> handwritten strategic decoder D
              -> selected changes to Havocbot
```

If `E(s1) == E(s2)`, the learned middle cannot distinguish those states for any
parameter setting. Its outputs are also limited to the effects reachable through
`D`. Increasing the width of `F_theta` changes neither fact.

The [earlier measured ownership collision](POLICY-ACTION-BOTTLENECKS.md) already
demonstrates the input-side loss. The bot inventory now enumerates the actual
[runtime entity fields](../measurements/havocbot-state-20260905/runtime-entity-coordinates.csv).
Neither a temporary candidate index nor its descriptor constitutes that state.

The learning objective contains another restriction: `online.py` pools candidate
probabilities into the seven `KINDS` and applies an entropy-floor penalty to that
distribution. Thus the vocabulary is reinforced by the loss as well as by the
encoder, output dimensions and decoder.

## Code deletion inventory

All entries below are pending runtime deletions. “Delete” means remove the named
construction and its callers; it does not mean rename it, broaden its enum, retain
a compatibility adapter or wrap it in a larger network.

| Component to delete | Concrete owners and dependent code |
| --- | --- |
| Instrument ontology and construction | Entire `strat/instruments.py`: `InstrumentKind`, `KINDS`, descriptor schema, `Instrument`, `InstrumentBatch`, `build_instruments`, `_descriptor`, `_action_mass`, `response_rows`, commitment helpers and instrument-keyed weight tables. Its target dataclasses are representations of this rejected interface. |
| Candidate-axis policy composition | `strat/strategy.py`: the candidate-shaped `Strategy` result, per-candidate state integration, sampled index, conditional three-control selection, density and `act` path. The replacement is not this composer with a differently named row axis. |
| Fixed learned input/output bindings | `strat/cast_header.py`: descriptor-bound KAY/VAL bindings, scalar-per-candidate DOV and six-output ACTUATOR binding; `scale_config.py` realization from `DESCRIPTOR_WIDTH`; candidate-conditioned dynamics and candidate-weighted critic pooling in the composer. General learned matrix, Gram and SwiGLU operators do not imply these bindings. |
| Truncated state-to-model contract | `inputs.py::player_features`, the selected XAN schema in `strategy_io_schema.py`, and candidate fields in `ChorusArrays`/`assemble`. Eighteen scalar observations plus weapon bits are currently treated as the entire player input. Deleting this restriction includes the corresponding limited gather/model contract, not merely replacing a Python selector while leaving the engine unable to supply the other state. |
| Observation-to-instrument conversion | `LiveBelief.instrument_targets` in `live_belief.py`, `build_runtime_frame`'s target/candidate construction in `runtime.py`, and responder calls assembling those rows. Observation retention, identity joins and diegetic perception are separate requirements. |
| Eight-field command ABI | `strategy_io_schema.py`: `SC`, `INSTRUMENT_KIND`, `TARGET_KIND`, `CONTROL_FIELDS`, command decoder; matching `PLC_SC_*`, `PLC_INSTRUMENT_*`, command-kind constants and response fields in `sv_payload_strategy_io.qh`; response-specific scatter and reset code. Delete the response schema at producer and consumer together. |
| Handwritten strategic execution | `sv_payload_strategy_io.qc::havocbot_goalrating_strategy`; `sv_payload.qc::havocbot_goalrating_payload`, `havocbot_role_payload` and its role override. These include cart/item/waypoint priority constants and the suppress-cart rule that picks every rival within 1,024 units, otherwise the cart. |
| Point-command commitment and spawn interpretation | `assignment_commitment`'s walking-time plus softplus law, `plc_str_commit` timeout extension, and the `PlayerPreThink` block applying `plc_str_spawn` differences to respawn clocks. Travel distance, spawn events and normal engine clocks are not this policy interpreter. |
| Vocabulary-shaped baselines and intervention sampler | Entire `baselines.py` construction using `KINDS + DESCRIPTOR_WIDTH + 10`, its forced idle-candidate default, and the responder's uniform-by-kind off-policy selection. A comparison arm cannot independently restore the removed ontology. |
| Vocabulary-specific learning records and loss | `online.py`'s kind-entropy objective, index-plus-three-controls likelihood binding, chosen-candidate dynamics rows and corresponding replay fields. Remove semantic reinterpretation of old policy/optimizer/weight-table state as replacement-policy state. Ordinary optimizer arithmetic and historical files do not require this interface. |
| Command-based causal attribution | `action_history.py` target identity/commitment assignments and first-matching-goal-rating actor credit; corresponding `plc_str_applied_*`, route/goal/touch command matching and serializers. Delivery, applied state and subsequent outcome remain distinguishable measurements; matching the old command cannot define replacement state-transition credit. |
| Vocabulary-shaped reporting | Candidate construction in `measure.py`; candidate divergence/three-control response binding in `policy_reports.py::compare_policies`; selected-kind focus and assignment analyses in `joracle/probe.py`, `metrics.py`, `field_measures.py`, `study.py`; wire/control labels in `web/app.js`, `policy.html`, `index.html`. Remove interpretation bindings, not the reporting service, active-run publication or durable artifact transport. |
| Schema-pinning acceptance and cost models | Old tests and verification scripts asserted the invented menu as a required interface; candidate-width costs also appeared in `work_estimate.py` and responder estimates. The operator subsequently required deletion of all test suites and harnesses. Their assertions do not authorize policy requirements. |

The dependency closure includes indirect users. Direct imports of `instruments`
occur in `runtime.py`, `strat_responder.py`, `action_history.py`, `online.py`,
`baselines.py`, `live_belief.py`, `measure.py`, `work_estimate.py` and
`scale_config.py`. `inputs.py`, `strategy.py`, reporting and persistence also consume
its fields without importing its classes. Deleting only the enum file would leave
both broken callers and duplicate interpretations of the same vocabulary.

The generic `STRATEGY` tensor message tag in `rdma/xonwire.def` does not specify an
action vocabulary. Neither RDMA transport nor the stock Havocbot movement/aiming
implementation is the source of this restriction. The custom policy role/rater is.
Cart physics, checkpoint scoring, real map geometry, observation buffers, numerical
kernels and node accessibility remain independently specified capabilities.

## Documentation deletion inventory

Delete the prescriptive constructions below from their documents. Historical
measurements can remain explicitly identified as measurements of the old program;
their successful execution cannot be cited as acceptance of its representation.

| Document | Construction to delete |
| --- | --- |
| `CAST.md` | Instrument descriptor as given ontology; “the scalar the spec calls for”; named behavioral meanings of VAL; six-output ACTUATOR contract; scalar-per-instrument velocity and categorical pooled-IR equations as the prescribed architecture. |
| `rl-training-spec.md` | One instrument per participant; categorical-target plus three-control output contract; candidate-IR and baseline schemas; per-kind distribution treated as strategy semantics. Formal reward definitions do not require these statements. |
| `payload/STRATEGY-IO.md` | Permission to derive instruments as the action ontology; ZED descriptor prescription; eight-float scatter response; goal-rating, commitment and spawn interpreter as required control boundary. |
| `ALGORITHM-CONTRACTS.md` | Participant/instrument row contract and the selected-assignment/walking-time-plus-extension paragraph, including the false claim that QC does not reconstruct strategy semantics. |
| `claims/POLICY.md` | Instrument decoder as the required actuator boundary; push/suppress and prescribed commitment as policy acceptance; the one-cadence instrument-weight persistence rule. |
| `AGENDA.md` | Completion claims that per-instrument velocity, action sampling or decoder delivery demonstrate the requested policy-to-bot state map. Original dated evidence is not a replacement specification. |
| `POLICY-EXECUTION-CONTRACT.md`, `POLICY-STATE-CONTRACT.md`, `JOINT-POLICY-LEARNING.md` | Any candidate/action schema promoted from an implementation manifest into a required policy representation. Transaction ownership and actual reward identities are separate facts. |
| `joracle-viewer.md`, `xonotic/README.md` and reporting descriptions | Any claim that selected instruments, three controls or command-goal matches exhaust policy output or certify actual state transformation. |
| `POLICY-ACTION-BOTTLENECKS.md` | The prior “Repair contract to discuss before implementation” recommending preservation of the candidate adapter. **Deleted in this investigation.** Its measured collision and gradient evidence remain. |

The original quote index is not an implementation manifest. Its missing full
velocity/cadence statements and the present operator correction are now included.
This prevents the clipped “integrated weight” phrase from serving as a license to
reuse arbitrary instrument weights again.

## Completion boundary

The investigation and deletion inventory are complete. The follow-up Python
implementation and numerical checks are recorded in the steering document. The
native writeback boundary is awaiting clarification; the consumer replacement and
live validation are incomplete. No workload was restarted and no bridge was touched.
Existing deployed binaries and checkpoints still describe the old interface;
source deletion alone does not change them.

Eliminating the defect means the old vocabulary and semantic interpreter no longer
define any input reduction, model output, update law, wire contract, training target
or acceptance claim. The intended policy-state/Havocbot-state transformation must
be expressed directly, with its numeric schema and integration semantics explicit.
Another small named collection of strategy knobs would recreate this defect.
