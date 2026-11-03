# Asynchronous policy learning

The learner uses AReaL's decoupled PPO objective and proximal-log-probability reuse,
with verl's per-action truncated importance sampling preset. The equations and
correction defaults are ported to MLX. The engine continues executing while requests
and observations travel between it and the responder.

This selection follows a September 5, 2026 review of current implementations. It is
a fit for this application's existing single-step actor/learner schedule; it is not
a claim that one estimator wins across all reinforcement learning workloads.

## Reviewed approaches

| Approach | Treatment of rollout/training lag | Fit here |
| --- | --- | --- |
| Synchronous PPO/GRPO | Generate a batch with one version, then train | Useful comparison, but the running game continues during training |
| One-step asynchronous rollout | Overlap the next rollout batch with the current update | Bounds version lag through scheduling; does not remove probability mismatch |
| Fully asynchronous decoupled PPO | Preserve behavior probabilities and versions; separate rollout correction from the proximal update | Selected |
| CISPO | Detach and truncate importance coefficients in a REINFORCE-style gradient | Closely related single-step gradient; reviewed as an alternative objective |
| IMPALA/V-trace | Correct multi-step value targets and actor updates from asynchronous trajectories | Requires a coherent ordered trajectory owner beyond the current delayed execution groups |

[verl's asynchronous trainer documentation](https://verl.readthedocs.io/en/latest/advance/v1_async_trainer.html)
describes the scheduling choices. [AReaL's paper](https://arxiv.org/abs/2505.24298)
and [asynchronous RL guide](https://areal-ai.io/docs/en/algorithms/async.html) combine
version-lag control with decoupled PPO. [MiniMax's account of CISPO](https://www.minimax.io/news/post-training-experience-and-insights-for-agent-models)
explains its detached coefficient weighting. [IMPALA](https://arxiv.org/abs/1802.01561)
provides the V-trace alternative for ordered trajectories.

## Exact selected equations

For the recorded action, let `b` be its behavior log probability, `p` its detached
proximal log probability, and `c` its current differentiable log probability:

```text
w = stop_gradient(min(exp(clamp(p - b, -20, 20)), 2))
r = exp(c - p)
L_actor = -mean(w * min(r * A, clamp(r, 1 - epsilon, 1 + epsilon) * A))
```

The first line is [verl's token-TIS correction](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/rollout_corr_helper.py),
with threshold 2, numerical log bound 20, and no batch renormalization. The loss is
[AReaL's decoupled PPO equation](https://github.com/areal-project/AReaL/blob/main/docs/en/best_practices/algo_perf.md).
[verl documents the separation of the three policies](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html).
Our PPO epsilon remains 0.2.

Truncation introduces bias to reduce variance. The numerical lower bound also
floors very small coefficients at `exp(-20)`. These are explicit upstream choices,
recorded in the training contract. No sequence rejection or age-based discard is
enabled by the selected preset. The eligibility mask still identifies whose action
was actually observed executing.

One game action corresponds to one token in the pointwise estimator. Its log
probability includes the instrument choice and conditional continuous controls.
Ratios are not multiplied across players, teams, or an entire match. This correction
does not claim to reconstruct the joint distribution of all match trajectories.
The competitive teams are not exchangeable GRPO samples of one prompt; their
existing game rewards and advantage definitions remain the inputs to this actor
estimator.

## Proximal policy and version identity

`OnlineLearner.learn` makes one optimizer update after accumulating the entire
execution group. Each source frame is evaluated under the same pre-update model.
The proximal value is `stop_gradient(c)`, using AReaL's
[`REUSE_TRAIN_LOGP` implementation](https://github.com/areal-project/AReaL/blob/main/areal/trainer/ppo/actor.py).
Thus `r` has value 1 at this update, while its derivative still provides the policy
gradient. There is no additional model copy or forward pass.

With this single-step schedule, the actor gradient is the detached truncated
importance-weighted policy gradient. PPO clipping only becomes active if an
optimization batch is reused for multiple updates against a fixed proximal policy.
Such a schedule would need to retain its proximal log probabilities across those
updates instead of refreshing them each step.

`ActionHistory` retains the actual issued behavior log probability, source update
counter, participant, action, and request identity through delayed execution.
Training does not overwrite that probability with the current one. Engine execution
lag is corrected at consumption time. The responder publishes requests from its
latest model on each decision; this change does not introduce a second worker queue
or a staleness rejection policy.

The training schedule is `first-execution-v4`. Loading an otherwise compatible v3
checkpoint retains its model, optimizer, and replay and records the current update
as the new schedule boundary. The sampling distribution did not change, so its
checkpoint policy-distribution version is unchanged.

## Data owners and measurements

- `policy_contract.py` owns the correction constants.
- `policy_math.py::rollout_importance_weights` owns the detached TIS calculation.
- `online.py::_item_loss` combines that coefficient with the proximal PPO surrogate.
- `online.py::learn` owns the single update and its proximal version counter.
- `ActionHistory` owns behavior identity and first observed execution.

`importance_ratio` reports applied coefficient quantiles, raw log-ratio quantiles,
the TIS truncation fraction, numerical floor fraction, and effective sample fraction
`sum(w)^2 / (n * sum(w^2))`. Raw drift stays in log space; reporting does not
exponentiate it into infinity. `behavior_age_updates` counts the lag from behavior
to proximal version, weighted by eligible actor rows. A same-version action has age
zero. `proximal_updates` identifies that pre-update version explicitly.

The only added conditional selects an absent age measurement when a group has no
actor rows. It does not skip training or engine execution. Existing eligibility
masks continue excluding advisory or unmatched actions from actor attribution.

## Verification and remaining scope

The reference-equation tests compare values and analytical gradients across both
advantage signs, proximal ratios inside and outside the PPO interval, and rollout
log gaps from -100 to 100. Separate tests exercise the actual single-step detached
proximal graph, full learner updates, checkpoint contents, and schedule migration.

The original finite-input reproduction now reports finite loss and gradient norm,
zero nonfinite parameter tensors, and zero nonfinite saved arrays. Results are in
[policy-async-correction.json](../measurements/policy-async-correction.json).

This repairs the reproduced stale-probability failure. The remaining
[prelaunch findings](POLICY-PRELAUNCH-REVIEW.md) concern pending episode/request
continuation, durable terminal outcomes, and spawn execution attribution. None is
resolved by an importance coefficient. A live engine/restart/outcome integration
run remains necessary after those repairs.
