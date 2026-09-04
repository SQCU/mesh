# J-space field measurement

The controlling contracts are [`SPECIFICATION.md`](SPECIFICATION.md),
[`rl-training-spec.md`](rl-training-spec.md),
[`ALGORITHM-CONTRACTS.md`](ALGORITHM-CONTRACTS.md), and
[`joracle-viewer.md`](joracle-viewer.md). Implementation claims are indexed by
[`claims/POLICY.md`](claims/POLICY.md).

## Purpose

The measurement describes how the strategy representation carries authoritative game
state and how policy actions alter later server state. It does not rank a checkpoint,
declare a policy acceptable, or reduce behavior to an improvement label. Aggression,
robustness, competitive pressure, survival, objective conversion, and their missing
coordinate masses remain separate measures.

The J-lens measures decodability from the exact representation emitted by the live
policy composition. The J-oracle uses the controlled server state machine to join each
source atom to its authoritative successor and to measure the realized response to the
source action. Neither interface reconstructs feature rows from console prose, invents
successors, or substitutes CartSim state.

## Producer interface

`xonotic/solver/strat/strat_responder.py` produces one response record from the literal
participant, cart, event, navigation, policy, action, and formal-value tensors used by
the strategy step. Stable participant and instrument identities accompany every tensor.
The record carries the representation, the complete source state, the complete
authoritative successor state once observed, the action density, selected and active
source arms, engine time, frame identity, and provenance.

`xonotic/solver/strat/joracle/field_measures.py` defines additive field measures over
those atoms. `xonotic/solver/strat/joracle/metrics.py` defines the behavioral measure
coordinates. `xonotic/solver/strat/joracle/liveness.py` contributes process and fabric
liveness coordinates without deciding whether a record belongs in the population.

Participant succession is an identity join, not a shape comparison. Each transition
publishes `joined_row_mass`, `successor_source_row_mass`,
`successor_present_row_mass`, and `departed_row_mass`. A newly joined identity has no
fictional predecessor. A departed source identity retains its realized sparse reward
and has zero successor bootstrap and dynamics mass.

## Consumer interface

`xonotic/solver/strat/joracle/probe.py` consumes the responder's literal records and
computes source strata, successor strata, cross-moments with J, covariance with J, and
the affine J-to-successor projection. The exact source and successor schemas travel
with their rows. The consumer never adds a reported delta to a source row to fabricate
a successor.

`xonotic/solver/strat/study.py` integrates these measures across mirrored perturbation
blocks. It preserves total, numeric, finite, non-finite, attributed, unattributed,
matched-leg, and unmatched-leg masses. Missing observations remain coordinates of the
measure rather than reasons to discard records.

## Feature and representation boundary

The encoder consumes the canonical participant state and belief tensors constructed by
the live responder. Instrument features and relation rows enter their declared strategy
composition boundaries. Participant Gram fusion, residual Gram fusion, and DPP feature
Gram construction have the exact meanings in `ALGORITHM-CONTRACTS.md`; unrelated
normal matrices and saved parameter trees are not called Gram matrices.

V-cell structure consumes stock navigation realization, streamed waypoint links, and
literal participant traversal observations as separately named sources. No nearest-
neighbor stand-in or unused plan object is part of the live interface. Cart belief and
commitment distance use the same navigation realization.

## Storage and reproducibility

Measurements flow through the in-memory telemetry rings and are written only to an
operator-selected ephemeral study directory when a durable capture is requested.
Historical run snapshots that predated these interfaces were removed because their
reconstructed rows and obsolete schemas could be mistaken for current release
evidence. Reproducible measurement starts at `joracle/probe.py` and the live responder,
not at a checked-in result file.

The result is a family of numerical measures with provenance and observation mass. No
boolean gate, judgment field, rejection filter, or fixed player/team/cart operating
point belongs to this interface.
