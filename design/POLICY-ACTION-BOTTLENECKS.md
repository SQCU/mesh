# Policy action representation and gradient bottlenecks

September 5, 2026. This audit follows the operator's request to pause the playerbot
servers and examine the feature-to-action boundary. It supersedes any inference in
the [reporting RCA](REPORTING-CONTINUITY-RCA.md) that demonstrating transmitted model
samples establishes the intended strategic semantics.

The subsequent [vocabulary provenance and deletion inventory](POLICY-VOCABULARY-RCA.md)
records the operator's state-vector correction. The former candidate-adapter repair
proposal has been deleted. The equations below describe the measured implementation.

## Workload pause

The Mini's `io.mesh.cartlane.learner` LaunchDaemon was disabled and booted out.
Its supervisor received SIGTERM, the responder saved both policies at update 9,049,
and the dedicated server exited through the supervisor's console shutdown path.
The supervisor recorded normal completion at 2026-09-05T20:49:06Z. Both saved
checkpoints' numeric arrays were finite. No learner, supervisor or dedicated game
server remained; UDP port 26400 had no listener. Bridge PIDs 361 on the Mini and
60237 on the laptop were unchanged. The read-only viewer remains available with
the final, now stale, measurements.

Automatic workload restart is disabled until explicitly resumed. The service plist
and all checkpoints remain in place. Resumption, when requested, consists of enabling
`system/io.mesh.cartlane.learner` and bootstrapping its existing plist on the Mini.
No bridge or machine accessibility setting was changed.

## Indices are addresses; candidate features must supply meaning

The current scorer is shared across candidate rows. It does not learn a permanent
meaning for integer 17. Its executable path in `strategy.py` is:

```
q_p = norm(QUINN(player_features_p, spatial_belief_p, team_semantics_p))
k_m = norm(KAY(candidate_descriptor_m))
v_m = norm(VAL(candidate_descriptor_m))
score_pm = dot(q_p, k_m) / sqrt(d)
H_pm = learned participant mixing + score_pm * inclusion_m * IR_VALUE(v_m)
H_pm = norm(H_pm + residual_fusion(H))
dw_pm = SwiGLU_projection(H_pm)
w_next_pm = tanh(w_previous_pm + delta * dw_pm)
logit_pm = (w_next_pm + control_weight * guidance_pm
            + exploration_weight * uncertainty_pm) / temperature
           + log(action_mass_pm)
```

The final categorical normalization is an action distribution; it does not turn
the Gram-based representation operator into a softmax-attention replacement.
The selected integer addresses the candidate that supplied its descriptor.
Reordering candidates, their support, weight-state columns and identities together
should reorder the corresponding outputs, preserving semantic choices. That is the
right indexing contract to verify, rather than making an integer label meaningful
by itself.

A categorical selection is therefore learnable in principle. The more serious
question is whether the features distinguish situations requiring different actions.
Let F be all preprocessing and pi_theta the model. If F(s1) = F(s2), then
pi_theta(F(s1)) = pi_theta(F(s2)) for every parameter setting. More width, training
or compute cannot recover the missing distinction.

## Reproduced information loss

`instruments._descriptor` keeps the instrument-kind tag, availability, position,
path position/length, speed, respawn timestamp, health, observation time and
checkpoint depth/count. It omits `Instrument.team` and subject identity. Identity
is retained for decoding, but there is no explicit candidate-to-entity feature join
restoring the target's remaining attributes into this descriptor.

The omission of raw IDs alone is not a defect: IDs should support joins, and need
not be treated as ordered real-valued features. The loss of the joined attributes
is the defect.

An algebraic check through the actual `build_runtime_frame` demonstrates a complete
input collision. Put two carts at different fixed positions with equal captured
checkpoint depths, owned by different teams. Exchange their owners while preserving
player state, map observations, scores and the integrated policy weight state:

| Quantity | Before versus after ownership exchange |
| --- | --- |
| Physical cart ownership | Different |
| Per-team held checkpoint totals, projected role and rank | Identical |
| Eight-coordinate team semantic rows | Identical |
| Candidate descriptors and support | Identical |
| Player, belief, team-ID and previous-weight inputs | Identical |

The policy cannot distinguish which physical cart belongs to which team in this
pair of states. The formal cart snapshot contains that information, but preprocessing
discards its association with the target. The test is an input transformation check,
not a simulated game outcome. Evidence is retained in
`measurements/policy-action-bottlenecks-20260905.json`.

Other concrete omissions:

- Item candidates do not describe item class, weapon identity, ammo type, quantity
  or health/armor benefit. Item observation slots likewise identify presence/gone
  and literal position/timestamps, not that resource inventory. Position may allow
  map-specific memorization; it does not expose the resource semantics.
- Rival candidates retain location and health but drop their team from the model
  descriptor and do not join that candidate to the target player's weapons/ammo.
  Global player mixing and spatial observations can provide indirect cues. They
  are not an explicit identity-preserving target join.
- Cart state is compressed into eight per-player team-summary coordinates plus
  candidate geometry. A team total cannot reconstruct the ownership-to-cart mapping.

These omissions are especially material given [SPECIFICATION.md §3](SPECIFICATION.md):

> the POLICY is integrating FULL RELEVANT GAME STATE FEATURES like THE HEALTH OF
> ALL PLAYERBOTS AND THEIR AMMO COUNTS AND GUNS therefore the POLICY is implicitly
> a LEARNED FILTER on whether a GUY WITH A ROCKET LAUNHER IN A TEAM SHOULD RUN
> TOWARDS THE CART OR TWOARDS MORE HEALTH AND AMMO, alongside WHICH cart to run
> towards

The current player input really does include health, armor, ammo and weapon bits.
The defect is not total absence of those inputs. It is losing the target-side facts
and relations needed to connect that player's condition to a particular destination.

## What gradients exist

For the categorical component, the basic score-function update is
`A * grad_theta(log pi_theta(m|s))`. At the logit boundary,

```
d log pi(m|s) / d logit_j = 1[j=m] - pi(j|s)
```

Thus a selected candidate provides a gradient through all supported candidate
scores, not a derivative through the integer index or through the engine. The
current learner also differentiates the selected conditional Gaussian log density
for the three raw controls. Its rollout correction and proximal surrogate modify
the weighting of these gradients. This mathematical possibility does not establish
correct causal credit for this particular adapter. The
[policy-gradient paper](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
provides the general estimation result; the executable source defines our actual
sampling and loss.

An automatic-differentiation check of the canonical forward at an initialized model
with the deployed widths found:

| Parameter projection | Value-loss gradient | Action-log-likelihood gradient |
| --- | --- | --- |
| Player/context query | Nonzero | Nonzero |
| Candidate key and behavioral value | Nonzero | Nonzero |
| SwiGLU candidate score projection | Nonzero | Nonzero |
| Continuous actuator projection | Exactly zero | Nonzero |
| Primary winner/loser value projections | Nonzero | Zero |

This is a connectivity check, not evidence of trained semantic performance. The
actuator gradient norm was 0 for value loss and approximately 13.144 for the sampled
action likelihood in this check. No optimizer update or game rollout was performed.

The structural reason is that value is a probe of a probability-weighted pooled IR.
Actuator parameters are projected from IR downstream and are not inputs to that
value computation. Value learning can change the shared IR, and hence change control
outputs indirectly. It does not directly train the actuator projection's parameters.
The learned dynamics used for guidance also consume the candidate IR rather than
the actually sampled gain/commitment/spawn vector; they do not supply a direct
action-conditioned value gradient for those controls.

This qualifies the earlier counterfactual explanation: other-policy and human value
rows can change counterfactual outputs through shared representation learning.
They are not independent evidence that those counterfactual controls work, and they
do not become actor examples for that policy.

## Where credit and execution are narrowed

The engine still chooses WHAT in addition to HOW. `havocbot_role_payload` combines
the model's additive rating with stock cart, item and waypoint priorities. An issued
model preference may never become the navigation goal. `ROUTE_SEQ` acknowledges
resolution/application inside goal rating; it is not a selected-goal or movement
acknowledgement.

This does not fully honor [SPECIFICATION.md §12](SPECIFICATION.md):

> 'what navigation targets should the bot be going towards' is something that has to
> be encoded in the big matmul code partition. 'how do i move around the map to get
> to a target?' is something that is in normal playerbot code

Some fixed semantics are necessary to execute a command. Existing stock aim,
pathfinding and movement implement HOW. A second unreported competition over cart
versus item destinations changes WHAT, so it needs correction against this quote;
its presence should not be defended merely because it is documented in a later
implementation description.

The training consequences require separate treatment:

- Actor credit currently selects first observed matching goal-rating applications.
  Unresolved commands are omitted. Since resolution depends on the sampled action,
  this selects the actor data by an action-dependent event. Calling it ordinary
  unconditioned policy-gradient evidence would require justification we do not have.
- The delayed actor interval runs from issue time to first observed application.
  It includes latency and can include rewards before the command affected behavior.
  The code does not establish a full causal interval from application through outcome.
- Commitment and spawn do not have their own application acknowledgements. Spawn
  runs outside goal rating, including while a bot is dead. That route acknowledgement
  cannot certify the entire sampled control vector.
- Some controls are irrelevant for some selected instruments or participant states.
  They still enter the joint likelihood; absent additional structure, those factors
  contribute sampling variance without a corresponding causal effect in that state.
- Other-policy and replay value updates can advance optimizer counters while own
  actor exposure remains small. Shared gradients do not repair missing observations.

The DPP calculation supplies marginal-inclusion features that influence logits.
The final action sampler samples player rows categorically. It does not sample a
joint DPP allocation of players to objectives. Context-dependent coordination of
preferences exists; stronger claims about enforced diverse or complementary team
allocations require separate evidence.
