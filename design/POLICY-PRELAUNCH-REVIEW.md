# Online learning prelaunch review — September 5, 2026

This review found four defects to repair before another sustained online learning run.
These findings come from source inspection and isolated MLX/QCVM reproductions.
They do not establish how frequently the failures occur in live matches or identify
the first invalid operation in the September 4 run. No server was launched or
restarted for this review.

Finding 1's reproduced stale-probability update has subsequently been repaired with
[the established asynchronous rollout correction](ASYNC-POLICY-LEARNING.md).
Findings 2–3 are repaired by [durable training continuation](TRAINING-CONTINUATION.md),
including actual engine matches with responder interruption and lost EVENT frames.
Finding 4's spawn/target adapter was deleted by the
[full-state interface replacement](POLICY-STATE-STEERING.md).
The numbered findings below preserve the original September 5 evidence, not the
current implementation or a request to reinstate that adapter.

Historical machine-readable results and source fingerprints are in
[policy-prelaunch-review.json](../measurements/policy-prelaunch-review.json).

## 1. A finite stale-probability update corrupts the learner and its checkpoint

`policy_math.py::clipped_policy_surrogate` still exponentiates an unbounded selected
log ratio for negative advantages. The earlier repair addressed overflow in the
unused clipped branch; this failure occurs in the selected branch.

An actual `OnlineLearner.learn` call with finite source features and behavior log
probabilities, log ratio 100, and advantage -1 produced infinite loss and NaN
gradient norm. The optimizer changed 30 parameter tensors to nonfinite values and
advanced its update counter. Saving that learner wrote 90 nonfinite arrays.

Repair the stale-probability update mathematics, retain a usable coherent learner
generation, and capture the first failing computation while continuing service.
Clamping both PPO branches arbitrarily would change the objective and needs an
explicit mathematical justification.

Relevant owners: `strat/policy_math.py::clipped_policy_surrogate`,
`strat/online.py::learn`, and `strat/online.py::save`.

## 2. Mid-round resume loses pending returns and request history

The learner keeps pending episode returns separately from retained replay. `save`
serializes retained replay but omits the pending episode. A save/load reproduction
reported mode `resumed`, preserved update count 1, and changed pending episode
length from 1 to 0 without recording a truncation.

The responder also creates a new empty `ActionHistory` on restart. Its runstate
restores RNGs and cursors, without restoring pending request attribution. A
mid-round restart therefore cannot continue the pre-restart return history and
execution joins it claims to resume. This especially affects the terminal objective.

Make continuation a coherent match generation that includes pending returns and
request identities. Account explicitly for any history that cannot be recovered.

Relevant owners: `strat/online.py::save`, `strat/online.py::_load_full`,
and responder runstate initialization.

## 3. A missed transient event loses terminal learning

The responder derives `winning_team` only from received outcome events. The engine
removes event entities after publishing their rows. A later durable `FINISHED`
observation does not repair a missed event.

For a finished two-team snapshot with scores [100, 0] and no outcome event,
`OnlineLearner.transition` returned terminal-arm rewards [0, 0] and bootstrap
discounts [1, 1]. `ActionHistory.advance` stops generating transitions after its
previous snapshot is finished. The supervisor separately retains engine outcomes
in `server-outcomes.jsonl`, but no learner consumes that journal.

Persist the engine's actual episode/winner outcome and consume it idempotently in
learning and reporting across restarts. Preserve the engine's winner identity;
rounded score observations need not reconstruct its exact decision.

Relevant owners: `strat/strat_responder.py` outcome handling,
`strat/online.py::transition`, `strat/action_history.py::advance`,
`sv_payload_strategy_io.qc::payload_strategy_gather`, and `strat/curriculum.py`.

## 4. Spawn execution bypasses attribution and affects advisory human rows

The engine applies spawn timing in `PlayerPreThink`, independently of the bot route
application that stamps `ROUTE_SEQ`. Dead bots return before route application, so
a spawn request can take effect without ever receiving the execution acknowledgement
used for actor credit. The spawn hook also applies to humans, although the responder
labels their policy rows human/advisory and excludes them from actor samples.

A QCVM fixture compiled the actual hook body with an identity `Spawn_SwizzleTime`
stub. For a human entity it moved respawn time from 10 to 15, recorded applied spawn
control 5, and left route sequence 1 unchanged. This checks dispatch and arithmetic;
it does not exercise the engine's full spawn scheduling implementation.

Give applied spawn controls explicit request attribution and make actual human
control behavior agree with the advisory contract.

Relevant owners: `sv_payload.qc::PlayerPreThink`,
`sv_payload.qc::havocbot_role_payload`, strategy response scattering,
and `strat/action_history.py::advance`.

## What the previous verification established

The prior 128-update checks exercised common learner arithmetic and saved
model/optimizer/replay continuation. The subsequently removed numerical harness
resampled actions on already-prepared transition rewards and called `learn`
directly for both arms. It did not exercise each arm's complete transition construction, pending episode
resume, engine execution, or terminal event delivery. The prior phrase "complete
resume" in the execution contract exceeds that coverage.

The subsequent integration check covers full-state rates, applied sequence
attribution, restored history and terminal labels on both machines. Its controlled
tie verifies terminal plumbing; it does not establish learned playing strength.
