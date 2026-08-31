# Live strategy training

This document describes the executable strategy runtime. The verbatim requirements
remain in `SPECIFICATION.md`; behavior is accepted through code, QuakeC builds, and
realized server telemetry.

## Runtime

Training is the process that answers a running Xonotic payload server. The server
publishes participant observations, cart states, and perception events. The responder
selects one eligible strategy instrument per participant, returns eight control values,
then learns from the next observed server state. There is no simulator or separate
bootstrap environment in this path.

`solver/strat/curriculum.py` samples maps, team counts, players per team, cart counts,
bot/human controller mixtures, skills, perturbation regimes, and off-policy player
counts. The curriculum builds the payload overlay once, then gives each match its own
user directory and exact copies of `progs.dat` and `csprogs.dat`; a failed build is a
recorded match failure rather than silent stock gamecode.

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

For the current winning team, `W` has a sparse preservation reward:

```
r_W(s, s') = -1  if the team loses the path to victory
              0  otherwise
```

The optimal `W` estimator is the expected discounted negative incidence of losing the
path before the player's role changes. It is closest to zero in winner states robust to
the other teams' perturbations and more negative where that path is vulnerable. The
winning player rows therefore optimize the policy toward actions with positive `W`
advantage: actions that preserve the winning region or restore its robustness, without
an invented preference for winning quickly.

For every other team, `L` has a sparse upward-rank reward. Let `rho_i(s)` be the
ordinal rank of team `i` among the non-winning teams, with acquisition of the projected
winner position above every loser rank:

```
r_L_i(s, s') = 1  if rho_i(s') > rho_i(s)
                 0  otherwise
```

Holding rank and losing rank both have zero immediate reward. Their policy consequences
differ through the learned successor value: a state moving away from an upward flip has
lower `L`, producing negative temporal-difference advantage relative to a state approaching
one. The optimal `L` estimator is the expected discounted count of upward loser-rank
events before the player's role changes. Non-winning player rows therefore optimize the
policy toward actions with positive `L` advantage: approaches to the next upward rank
boundary, culminating in acquisition of the path as the final upward transition.

`W` and `L` are deliberately asymmetric. They are separate linear probes of the exact
128-dimensional Gram/SwiGLU representation consumed by the policy. The actor uses the
role-appropriate temporal-difference advantage; it is not rewarded for cart speed or
fast progress.

## Linear algebra and dimensions

Let `k` be the team count, `l` the participant count, and
`P in {0,1}^{l x k}` the player-to-team incidence matrix. Let `n in N^k` be the
team-nimber vector, `q in {0,1}^k` the one-hot projected winner (or the zero vector when
there is none), `u = 1 - q`, and `C_ij = 1[n_i > n_j]`. The loser-rank vector is

```
rho = (I - diag(q)) C u + (k - 1) q                 in N^k
```

and the two team-event vectors are

```
rW = -q * (1 - q')                                 in {-1,0}^k
rL =  (1 - q) * 1[rho' > rho]                      in {0,1}^k
```

`P rW` and `P rL` lift team events to player rows. Thus each transition carries an
`l`-vector of scalar row rewards, not one global scalar broadcast to contradictory
shared-policy rows. Teammates receive the same event, while different teams can receive
different events.

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
AW = m       * (P rW + gamma m'       * vW' - vW)  in R^l
AL = (1 - m) * (P rL + gamma (1 - m') * vL' - vL)  in R^l
```

The critic fits `vW` only on winner rows and `vL` only on loser rows. The shared policy
receives the compatible rowwise signal

```
L_policy = -(1/l) sum_p (AW_p + AL_p) log pi(a_p | s)
```

with the implemented clipped off-policy importance ratio multiplying each advantage.
Consequently shared parameters learn from many non-contradictory row objectives rather
than from one scalar that pretends every participant should take the same strategic
side.

## Unknown dynamics

The learner receives no action-to-successor oracle. `LocalDynamics` fits two locally
action-linear successor models from live transitions. Their mean proposes corrective
actions; their disagreement bounds exploratory proposals. Telemetry records one-step
error, ensemble disagreement, and the smallest singular value of the learned local
action matrix. These are diagnostics, not a claim that the real server dynamics are
globally known or controllable.

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

The final participant rows are RMS-normalized, form a symmetric Gram matrix, mix by
that matrix, and pass through one wide SwiGLU residual. W/L probe those final rows.
The DPP marginal signal and integrated instrument weights feed the action head.
Ineligible actions are masked only at sampling; their integrated weights remain finite.

## Executable evidence

The acceptance artifact is a running Xonotic server, its cross-node responder, and the
telemetry they produce. Each response exposes `loser_ranks`, both length-`l` value
vectors, both role-specific advantages and rewards, winner/loser row counts, role-change
fraction, tensor shapes and finiteness, learned-dynamics error and disagreement, and the
smallest local-control singular value. Checkpoint evidence includes the architecture,
optimizer state, update count, replay contents, and a subsequent increase in updates
after a handled stop and resume. A local assertion or manufactured transition is not
substituted for this evidence.

## Telemetry

Every response records the realized team/cart/player counts, cart identity and motion,
projected winner and succession, participant controller and behavior policy, target and
behavior log-probabilities, assignments, belief diagnostics, team health/armor/ammo/
weapons/alive/speed/power state, cross-team strategy focus, online losses, model error,
disagreement, and local-control singular value. `solver.strat.metrics` aggregates the
retention, acquisition, hierarchy-flip, resource-change, focus, controller, behavior,
and learner measurements without manufacturing missing outcomes.
