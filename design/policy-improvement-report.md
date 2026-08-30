# Policy-improvement report — evidence status

## Current conclusion

No live curriculum result in this repository yet demonstrates the policy-improvement
contract in `rl-training-spec.md`. The end-to-end implementation now exists; live
outcome evidence remains distinct from implementation evidence.

The earlier 600-iteration run remains a useful execution trace for the superseded
single-critic, fixed-shape implementation. It was trained only at two teams, four
players, and three carts. Its `win_rate` and `pw_control_frac` curves therefore do
not answer any of the newly required questions:

- whether a current winner resists rival perturbations;
- whether a loser climbs the full hierarchy and acquires the path;
- whether the asymmetric `W` and `L` critics are calibrated;
- whether the same checkpoint works at unseen team/player/cart counts;
- whether the learned local dynamics predicts held-out transitions or identifies a
  controllable neighborhood.

The former report also called a learned TD residual potential-based shaping and treated
its near-zero empirical mean as a policy-invariance fingerprint. That inference is not
valid. A critic baseline can reduce variance, but a TD residual is not thereby an
Ng–Harada–Russell reward transformation, and a centered advantage does not prove
improvement.

## Superseded-run provenance

- Trace: `xonotic/solver/strat/runs/train_log.jsonl`
- Checkpoint: `xonotic/solver/strat/runs/policy_ckpt.npz`
- Shape: `k=2, j=3, l=4`
- Objective: one relative terminal projected-winner reward
- Critic: one player-count-sized value output

These artifacts are retained as historical evidence and are not loaded by the current
responder. The revised CartSim bootstrap trainer writes `train_log_v3.jsonl` and
`policy_ckpt_v3.npz`. The live responder writes `policy_online_v3.npz` only when run
with `--train`; this is the primary training path.

## Acceptance matrix

| Property | Required evidence | Status |
|---|---|---|
| Winner policy `W` | retention under controlled rival perturbations; recovery after demotion | not measured |
| Loser policy `L` | hierarchy gain and path-acquisition rate, by initial rank | not measured |
| Terminal play | win outcome against fixed and adaptive opponents | not measured for revised policy |
| Count invariance | identical checkpoint tensors run trained and held-out count combinations | shape/equivariance smoke only |
| Interpolation/extrapolation | metrics on withheld combinations and small out-of-range counts | not measured |
| Learned dynamics | held-out one-step error and calibrated ensemble disagreement | not measured |
| Local controllability | real reachability probes plus local Jacobian diagnostics | not measured |
| Map generalization | withheld maps/observation graphs | not measured |

## Required experiment

Run Xonotic server matches over a distribution of team, uneven-player, controller,
cart, and map counts while the responder runs with `--train`. Use CartSim only to
initialize or smoke-test the checkpoint. Withhold complete combinations and maps.
Report `W` and `L` losses separately,
winner retention, loser acquisition, terminal outcome, dynamics prediction error,
uncertainty calibration, and reachability tests. Compare against fastest-progress,
greedy-denial, random, and no-dynamics-ablation policies.

The result is successful only if outcome metrics improve on trained and held-out
settings without resizing or reinitializing any learned tensor.

## Implementation evidence

The repository now exercises the complete synthetic loop from perception events through
belief reduction, dynamic instrument construction, shared-policy inference, 8-column
scatter decoding, multi-step role credit, and local-dynamics updates. The test uses bot
and human participant rows together and performs two credited optimizer steps without
resizing a learned tensor. Exact game-value tests cover terminal-zero/mex recursion,
XOR composition, belief uncertainty, and the rule that an incomplete empirical option
graph has no asserted nimber. The curriculum dry-run test realizes varying count tuples
and records every match.

A one-iteration bootstrap run passed across `(2,1,2)` and `(3,2,5)` training shapes and
the held-out `(4,3,7)` shape, producing `train_log_v3.jsonl` and
`policy_ckpt_v3.npz`. Clean stock-source overlays compiled both server QC and client QC.
These checks establish executability and shape transfer, not policy improvement; the
acceptance matrix above deliberately stays outcome-based.
