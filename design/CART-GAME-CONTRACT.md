# Literal checkpoint-control game

The controlling user quotes are in `SPECIFICATION.md` §16. This document gives the
algebra realized by `sv_payload.qc`, `game_value.py`, and `runtime.py`; it is not a
replacement game or an alternative policy objective.

## State, integration and victory

There are independently configured `k` teams and `j` cart lanes. Each lane contains
an ordered chain of scoring checkpoints, distinct from any additional path-shaping
vertices. Advancing past a checkpoint captures it for the cart's controller;
retreating behind it removes that control. Neither transition subtracts earned score.
The path endpoint is the last checkpoint, not a terminal delivery objective.

Cart motion requires live players in its capture area. An empty cart retains its
exact position, controller and captured checkpoints indefinitely; its checkpoints
keep accruing score. Opposing players must push it backward, crossing checkpoints
to remove their control and reaching the origin to neutralize the cart. Leaving
partway through that return stops the cart there. A neutral cart can then be claimed
by a team with a strict occupancy lead. There is no idle retreat or ownership decay.

For checkpoint depth `d_c`, controller `a_c`, positive checkpoint rate `alpha`,
accumulated score `S_i`, and victory threshold `Q`:

```
H_i = sum_c 1[a_c = i] d_c
dS_i/dt = alpha H_i
S_i(t + dt) = S_i(t) + alpha H_i dt
tau_i = (Q - S_i) / (alpha H_i)
```

`tau_i` is zero for an already reached threshold and infinite for a below-threshold
team with no held checkpoints. The unique smallest finite `tau_i` is the projected
winner with cart control held fixed. Equal first-passage times have no unique winner;
team ID never breaks the tie. The engine integrates only as far as the first threshold
crossing, including a simultaneous crossing as a draw. No delivery, timeout leader,
capture-count limit, or rollback deduction supplies another terminal condition.

Score integration happens before checkpoint ownership changes and before outcome
checks. Only an actual new episode resets the score ledger. Human and bot scoreboards
display the integer part of that same ledger; the solver receives its float values.
The native HUD receives the server's projected team, its float score, Q, projected
time and tie/finished state. Its victory percentage is `100 * S_PW / Q`, not lane
position, win probability or the largest raw score. The server projection routine is
shared with the score-ceiling crossing calculation. No unique projected leader is
labelled explicitly; an actual terminal winner overrides the projection.

## Denial and sparse rewards

The explicit counterfactual succession holds earned scores fixed and decrements the
current projected winner's held-checkpoint count, recomputing the first-passage winner
at every decrement. Every such decrement is realizable in the cart abstraction by
retreating one of that team's controlled checkpoint chains by one checkpoint. Which
lane is selected has the same immediate scoring effect because checkpoint weights
are equal. The FPS policy learns how to realize or resist that change; no synthetic
FPS rollout or hand-authored combat controller is introduced.

The succession records first appearances and marginal decrement counts. At a finite
tie, the counterfactual denies one checkpoint to each co-leading team as a symmetric
batch, recording the total decrements. The tie is never assigned an arbitrary winner;
its intermediate serial order is not asserted to be gameplay. The computation stops
when nobody can reach the threshold or the state has already ended. It is a literal
denial calculation, not an asserted Sprague–Grundy value of the multiplayer FPS game.
The unrelated heap-graph/portfolio-XOR implementation has been removed.

Ranks are read from weakly ordered first-passage tiers, including a tied bottom tier
for infinite times. This is the first-appearance ordering of the denial calculation
in an unfinished state: an unseen team's time remains unchanged; it cannot first
appear before an unseen team with a smaller time. Equal-time unseen teams enter
together. Every finite team eventually appears because each batch consumes held
checkpoints. Thus tie closure does not change the existing strict-rank reward.

With `q_i = 1[i = PW]`, rank counts teams in strictly later tiers. The unique winner
is above all nonwinners; tied teams have equal rank. The reward atoms remain:

```
rW_i = -q_i (1 - q'_i)
rL_i = (1 - q_i) 1[rank'_i > rank_i]
```

W predicts resistance to losing the winning role; L predicts upward loser-rank flips.
All previously nonleading teams that rise receive the positive atom, not just the
one becoming leader. A larger jump does not invent a larger reward atom. Advantage
is the TD return minus the role's value baseline, not an unconditional positive
number on every rewarded sample. Holding W has zero immediate cost versus minus
one for losing it; this is a resistance objective, not an added survival bonus.
Their role-specific TD returns do not cross-bootstrap between heads. Score, damage,
cart speed and checkpoint gains are observations, not substituted rewards.
Terminal-win PPO remains the separately requested ablation on actual score wins.
Its role rows select equivalent terminal-target heads, not W/L transition targets.

Fresh actor eligibility is the policy's own applied behavior, excluding human and
explicitly off-policy actions. Actor regularization uses that same mask. Historical
records train values through the shared backbone, not an imitation objective.
Auxiliary W/L probes are fitted only on the role where their source value is trained.

## Configuration and process lifetime

`--team-counts`, `--cart-counts`, and `--players-per-team` are independent sampling
axes. `--checkpoints-per-lane`, `--score-limit`, and `--checkpoint-score-rate` realize
the scoring configuration. The initial defaults are four, 1,200, and one respectively:
two fully controlled lanes need 150 seconds of accrual, in addition to acquisition.
The selected axes and roster are not rewritten by hardware operating-point search.
Capacity telemetry measures the requested workload without changing its population.

The supervisor retains a server at one endpoint and retains native clients across
episodes. It stages the next lane realization and uses the engine's `changelevel`
command. Match artifacts, learner input/output checkpoints and observation contexts
change without recreating the player application. A match advances after an observed
score outcome, not after a wall-clock process timeout.

## Reset and evidence

On September 4, 2026, the old supervisor was drained and 277 identified policy-state
files were permanently removed from both machines, including initial/trained weights,
optimizer/replay state and responder runstate: 22,524,790,908 bytes. No resume archive
was retained. Geometry caches, maps, source and diagnostic logs were preserved.
The old delivery-variant results are not evidence for this policy objective or a
valid rating history. An additional 10 verification resumables (442,590,521 bytes)
were deleted after a supervisor command-name collision prevented the intended map
handoff. That run is also excluded. The fresh lineage is `checkpoint-control-live-20260904`.

Wire observations contain actual checkpoint depths/counts and a synchronized complete
team table: score, held checkpoints, rate, threshold, episode identity and terminal
state. `measure.py cgt` checks these records against the literal projection, checkpoint
accounting and within-episode score monotonicity. Such checks establish implementation
consistency, not learned strategic improvement.

## Representation and executable equivalence

`game_value.py` owns `GameContext`, `CartSnapshot`, checkpoint ownership, score
integration, the projected winner, rank/succession, sparse/terminal reward contracts
and their records. `runtime.py` converts wire observations into this game state.
Python projection and score integration use the engine's float32 operation order,
including subtraction and division near the first threshold. A double-precision
recalculation is not the engine's tie rule.

Historical comparisons used extracted server projection, integration and
checkpoint-control functions over multiple team counts and ownership transitions.
The fixture suite was removed at the operator's instruction; its cases do not
define the game. See [POLICY-PROGRAM.md](POLICY-PROGRAM.md) for delayed-action
credit, interval accounting and evaluation over executed policy versions.
