# Live strategy training

This document describes the executable strategy runtime. The verbatim requirements
remain in `SPECIFICATION.md`; behavior is accepted through code, tests, QuakeC builds,
and realized server telemetry.

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

For every other team, `L` rewards its change in relative hierarchy and adds an
acquisition reward when it becomes the projected winner:

```
r_L_i(s, s') = H_i(s') - H_i(s) + 1[i acquires the path]
```

`W` and `L` are deliberately asymmetric. They are separate linear probes of the exact
128-dimensional Gram/SwiGLU representation consumed by the policy. The actor uses the
role-appropriate temporal-difference advantage; it is not rewarded for cart speed or
fast progress.

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

| Behavior | Implementation | Check |
|---|---|---|
| asymmetric W/L rewards | `solver/strat/runtime.py` | `RuntimeTests.test_asymmetric_rewards` |
| 128d Gram/SwiGLU and equivariance | `solver/strat/gram.py`, `estimator.py` | `test_gram_is_permutation_equivariant` |
| W/L are linear probes of final IR | `solver/strat/value.py` | `test_value_heads_are_linear_probes_of_ir` |
| one checkpoint spans row counts | shared row modules | `test_one_checkpoint_runs_different_counts` |
| finite masked updates | `estimator.py`, `online.py` | `test_ineligible_weights_do_not_become_sentinels` |
| one live belief path | `live_belief.py` | `test_live_belief_is_the_only_spatial_pipeline` |
| gamecode and schedule realization | `curriculum.py`, `payload/build.sh` | `test_curriculum_realizes_counts_controllers_and_gamecode` |
| resource/focus observability | responder telemetry, `metrics.py` | `test_resource_and_focus_metrics_are_computed_from_runtime_rows` |
| cart identity across gather rows | `sv_payload_strategy_io.qc` | isolated client/menu/server QuakeC build |

Run the checks from `xonotic/`:

```
PYTHONPATH=.:payload/tools python3 -m unittest solver.strat.test_runtime
payload/build.sh
```

The unit suite proves tensor-shape, gradient-path, curriculum-command, and telemetry
invariants. The isolated QuakeC build proves that the payload overlay compiles. Neither
is represented as a completed live match. Winner retention, loser acquisition,
terminal outcomes, held-out count/map results, recovery after perturbation, and
calibration of the learned dynamics require telemetry from executed curriculum matches.

## Telemetry

Every response records the realized team/cart/player counts, cart identity and motion,
projected winner and succession, participant controller and behavior policy, target and
behavior log-probabilities, assignments, belief diagnostics, team health/armor/ammo/
weapons/alive/speed/power state, cross-team strategy focus, online losses, model error,
disagreement, and local-control singular value. `solver.strat.metrics` aggregates the
retention, acquisition, hierarchy-flip, resource-change, focus, controller, behavior,
and learner measurements without manufacturing missing outcomes.
