# RL training specification — asymmetric control of the path to victory

Authoritative definitions for the strategy-learning stack. Where another design
document disagrees, this document is intent. Modality is `[FIRM]`, `[OPEN]`, or
`[BUILD]`.

## 0. Spine `[FIRM]`

- There are two games. Game 1 evaluates the observed cart position: `PW(s)` is the team
  with a path to victory and `SUCC(s)` is the cartstate hierarchy exposed to the policy.
  Game 2 is the FPS realization that turns player allocations into a later cartstate.
- The learner starts without a transition oracle for Game 2. It is not given
  `s, action -> s'`, a prior over the history until the next cartstate flip, or a label
  saying which FPS behavior realizes a particular cartspace move. It learns these
  effects from sampled actions and observed successor states, and continually fits a
  local transition estimator.
- One shared-weight policy acts for every player. Teams and players are activation rows,
  not separate parameter sets.
- The value stack is asymmetric. `W` evaluates preservation for a team that currently
  has a path to victory. `L` evaluates promotion and acquisition for every team that
  does not. `W` is not the sign-reversed or otherwise symmetric dual of `L`.
- The neural encoding and every learned parameter shape are independent of the number
  of teams, players, carts, posts, rivals, and observation cells.

## 0.1 Training is the server process `[FIRM]`

The primary environment is not `CartSim`. Training means running the Xonotic dedicated
server, sampling match configurations from the curriculum, assigning strategy actions
to participant rows, observing the later server state, and updating from those real
transitions. `strat_responder --train` performs the update in the same loop that answers
the server. `CartSim` is only a bootstrap prior, unit-test environment, and cheap policy
smoke test; success there is not evidence about Game 2.

The curriculum distribution is realized at match boundaries. Its sampled variables
include map, team count, players per team, controller mixture, cart count, skill, spawn
state, and perturbation regime. The scheduler records the realized configuration with
the transition stream. A map determines which cart graph is actually available, so a
requested cart count is a map selection or map-generation choice, not a fictitious
tensor-padding setting.

Every active participant is an action row. Any subset can use the current policy,
another checkpoint, a scripted controller, uniform exploration, or a human controller
in the same match. The behavior source and its action probability are recorded per row;
off-policy correction is therefore participant-local rather than match-wide. A
strategy assignment may be attached to a human row too. Until the game presents that
assignment to the human or infers which discrete strategy their behavior realized, the
result measures response to an advisory signal, not forced execution of a bot action.
That distinction belongs in telemetry rather than in a separate training topic.

## 1. State evaluation and unknown realization `[FIRM]`

The cartstate evaluator may compute the following after a state has been observed:

- `PW(s)`: the present path-to-victory holder.
- `N_i(s)`: team `i`'s cartstate standing in the current NIM-like hierarchy.
- `SUCC(s)`: the ordered counterfactual succession obtained by demoting the incumbent
  cart position.
- `H_i(s)`: a bounded relative hierarchy score for team `i`, increasing when `i` is
  promoted and decreasing when any rival is promoted.

These are state scores, not an a-priori dynamics model. Computing them at `s_t` and `s_{t+1}`
does not imply knowledge of which player action will cause `s_{t+1}`, when the next
cartstate flip will occur, or which sequence of physical states constitutes one
cartspace move. Replay records the empirical tuple
`(observation, action, later observation, cartstate before, cartstate after)` and gives
credit across the actual elapsed history. Those tuples also supervise an ensemble of
local action-linear models
`Δy = b_eta(y) + A_eta(y) u`, where `y` is a fixed-width reduction of the
team-relative cart hierarchy and `u` is a fixed-width reduction of the joint player
allocation. The model is local and continuously revised; unknown dynamics are an
estimation problem, not a permanent limitation.

The current deterministic `PW`/`SUCC` implementation is a candidate Game-1 evaluator.
Whether its integer heap labels are the correct nimbers of the realized partizan FPS
game remains an empirical/modeling claim, not a fact supplied to the policy.

## 2. The two value estimators `[FIRM]`

For team `i` and observation history `o_t`:

`W_phi(o_t, i) = E[sum_h gamma^h r^W_{t+h}(i) | PW(s_t) = i]`

`L_psi(o_t, i) = E[sum_h gamma^h r^L_{t+h}(i) | PW(s_t) != i]`

`W` is a state-preservation estimator. Its target is resistance to the other teams'
perturbations: retaining the winning region and restoring its margin after pressure.
It does not reward cart progress per unit time, minimum completion time, or a larger
nimber merely for being larger. Those objectives can surrender strategically useful
state space to a policy that induces fast but brittle progress.

`L` is a state-acquisition estimator. Its target increases when the acting team rises
relative to every rival, when rivals are demoted, and especially when the acting team
acquires `PW`. Once it acquires `PW`, subsequent states are evaluated by `W`; `L`
does not become a winner-preservation objective by symmetry.

A concrete bounded reward realization is:

`r^W_t(i) = retain_i(s_t,s_{t+1}) + eps * margin_recovery_i(s_t,s_{t+1})`

`r^L_t(i) = H_i(s_{t+1}) - H_i(s_t) + acquire_i(s_t,s_{t+1})`

where `retain` distinguishes holding from losing the path, `margin_recovery` is a small
secondary term, and `acquire` is paid only when a loser becomes the path holder. This
asymmetry is purposeful: the incumbent protects a target set; a challenger traverses
the hierarchy toward that set.

The role-gated baseline for player `p` on team `i` is:

`V_p(s_t) = W_phi(o_t,i)` if `PW(s_t)=i`, otherwise `L_psi(o_t,i)`.

Both heads emit one scalar for each activation row. They never emit an `l`-wide vector,
because an output width tied to the player count prevents reuse at another count.

## 3. Policy gradient `[FIRM]`

For an observed transition:

`A_p = r_role(p) + gamma * V_p(s_{t+1}) - V_p(s_t)`

`L_pg = -E[sum_p stopgrad(A_p) log pi_theta(a_p | o_t)]`

The critic loss fits `W` only on winner rows and `L` only on loser rows. The next-state
baseline follows the next role, so acquisition changes from `L` to `W` and loss of the
path changes from `W` to `L`. `theta` is shared across all rows and all population
shapes.

The policy gradient remains valid without differentiating through the transition model.
The learned model adds counterfactual local targets: `W` can choose corrections that
return to the winning region, `L` can choose a reachable promotion step, and
ensemble disagreement can schedule small information-gathering probes. The TD residual is not itself a
potential-based reward-shaping theorem, its empirical mean need not be zero, and critic
fit does not prove policy improvement. Improvement must be measured separately on
winner retention under rival perturbation, loser acquisition, terminal outcomes, and
held-out population/map shapes.

## 4. Count-invariant encoding `[FIRM]`

The input is a typed graph or collection of row sets:

- player rows, grouped into teams;
- cart, post, rival, and observation-cell rows;
- candidate instrument rows and their relations to the acting player/team.

Shared row encoders act pointwise. Graph messages are reduced with permutation-
invariant operators such as mean, max, and normalized sum. The reduced messages mix
back into each player/instrument row, and a shared scalar head emits one logit per
candidate edge. Adding a team, player, cart, or candidate therefore adds rows; it never
adds coordinates to a weight matrix.

Forbidden encodings include team-ID one-hot vectors whose width is the roster size,
flattened `2*j` cart arrays, a `2*k+1` succession vector, an `l`-wide critic output,
and a dense `2*M -> M` action head. They bind learned parameter shapes to one scenario.

No absolute team/player/cart positional encoding is used. Identity needed for action is
expressed as relations such as self-team, rival-team, controlled-by-self, and target-of-
instrument. This gives permutation equivariance by construction. It gives sensible
interpolation a strong inductive bias, not a proof of arbitrary extrapolation.

The server curriculum samples many team counts, uneven player counts, controller
mixtures, cart counts, maps, and observation graphs. Evaluation withholds complete
count/map combinations and includes small extrapolations outside the training range. A
checkpoint is accepted only if the same tensors run all of them without resizing or
reinitialization.

## 5. Brown–Papadimitriou–Roughgarden connection `[FIRM]`

[Online Stackelberg Optimization via Nonlinear Control](https://arxiv.org/abs/2406.18805)
supplies the useful control picture: optimize a target state/region while anticipating
adaptive responses, and preserve reachability margin so adversarial disturbances can be
corrected. That is the right inductive explanation for `W`, and it explains why a
simple linear or fastest-progress policy can be inferior to a state-targeting policy.

The paper's regret results are not imported as a theorem for this system. Its known-
dynamics algorithm queries an action-to-target oracle, which is precisely the role the
learned local model should approach. Its unknown-dynamics result
assumes time-invariant, locally action-linear, locally controllable dynamics plus an
initial near-stabilizing action. The FPS learner has not established those assumptions.
It can, however, try to realize them: locally action-linear parameterization, deliberate
probing, ensemble error estimates, and singular-value/reachable-neighborhood diagnostics
turn the assumptions into measurable design goals. Roth et al.'s locally learned
preferences supply the analogous lesson: re-express a non-convex surface in the state
or agent-action space where local learning and exploration expose usable convex
structure. Here the paper motivates the target-region and disturbance objectives;
online system identification supplies the initially unknown realization.

## 6. Loss and acceptance `[FIRM]`

`L = L_pg + c_W L_W + c_L L_L + c_D L_dynamics + c_reg L_reg`

- `L_W`: masked regression of `W` to winner-preservation returns.
- `L_L`: masked regression of `L` to hierarchy-promotion/acquisition returns.
- `L_dynamics`: ensemble prediction loss for the observed reduced-state transition.
- `L_reg`: logit and weight regularization compatible with broad weighted sampling.
- Selection remains categorical sampling, not MAP.

Required measurements are winner-retention probability after controlled rival
perturbations, time-to-recovery after a temporary demotion, loser acquisition
probability, terminal win outcome, and the same metrics on held-out counts/maps. A
single fixed-shape self-play win-rate curve does not establish this contract. Dynamics
acceptance additionally requires held-out one-step error, calibration of ensemble
disagreement, local action-Jacobian rank/smallest singular value, and direct reachability
tests around sampled states. A full-rank learned Jacobian alone is not evidence that the
real game is locally controllable.

## 7. Implemented / open

- `[OPEN]` the calibrated hierarchy score `H`, retention margin, credit horizon, `gamma`,
  and loss coefficients.
- `[DONE]` real perception-gated observations, persistent per-team histories, temporal
  contraction, and a live V-cell graph from Game 2.
- `[DONE]` a match supervisor that samples and records maps, rosters, carts,
  controllers, skills, seeds, perturbations, held-out tuples, and checkpoint lineage.
- `[OPEN]` execute the perturbation and held-out schedules at sufficient scale to fill
  the outcome acceptance matrix.
- `[BUILD]` a visible human strategy channel or an inverse classifier that identifies
  the discrete strategy realized by human movement.
- `[DONE]` exact mex/XOR evaluation for complete impartial graphs and an empirical
  option graph that keeps incomplete, cyclic, or partizan values explicitly unresolved.
- `[OPEN]` acquire enough real transitions to validate or replace the candidate cart
  nimber evaluator.
- `[DONE]` bounded ensemble-disagreement probes and local action-to-target proposals
  refine W/L actions after the dynamics learner has observed transitions.
