# Two objectives, one learning implementation

The current input/IR/head/loss program is defined in
[POLICY-PROGRAM.md](POLICY-PROGRAM.md). This page retains the objective distinction,
experience-retention design and dated operational evidence. Historical launches
below do not describe a currently running or verified deployment.

The [September 4 prerequisite review](POLICY-SEMANTICS-REVIEW.md) records the recovered
session, continuation and spatial repairs, executable coverage and remaining defects.

September 4, 2026. This is the operator-requested mixed-team training intervention,
not a claim that either objective has already demonstrated strategic superiority.

## Governing distinction

Value learning predicts realized consequences from observations over an explicit
mixture of fresh and historical experience. Its gradients train the shared backbone.
Policy learning changes action selection against its stated objective. Historical
outcome labels are not instructions to imitate historical actions. Actual competitive
outcomes evaluate the usefulness of each objective independently of its training loss.

These distinguish losses, targets and data distributions, not independent components
called experts, frozen parameter groups, or host-dependent ownership.

## Realization

`matrix_fusion` is the existing stratCGT-PPO objective. `terminal_win` is terminal-win
PPO. Both use `Wally`, `strategy`, identical parameter shapes and the same architecture
seed. Each has its own whole parameter tree, AdamW state and replay lineage because
they are genuinely different learned policies. They share all implementation code and
the stateless remote matrix worker.

`curriculum --joint-training` samples real maps, teams, carts and game seeds. It draws
a team count for the first policy uniformly from 1 through teams minus 1, fills the
remaining slots with the second policy and shuffles the assignment. Every team gets
one behavior policy for that match. Both policies evaluate the same observed state;
only the assigned policy's sampled output is sent for each participant.

Per-policy actor masks select its own behavior actions at their first observed
application, excluding native-default and human controllers. The actual
applied state sequence identifies the source request, including an older delayed request. Value masks include all observed teams. The other
policy's actions are not treated as supervised demonstrations or as on-policy samples.
The shared-backbone value gradient nevertheless changes counterfactual action outputs.
Off-policy actor improvement from the other policy's actions is not implemented by
silently applying an ordinary PPO objective to them.

The initial runtime exposed a queued-observation defect: response 146 was being
optimized while the observed players still used response 58. The responder now acts
on the newest complete synchronized observation, counts coalesced older observations,
and preserves all intervening perception and outcome events. PPO eligibility requires
the server's applied state sequence to match the source action; an emitted response is
not by itself evidence of application. All pre-restoration policy resumables have
been permanently deleted; their diagnostic logs are not a valid rating history for
the checkpoint-control objective. That historical lineage was `checkpoint-control-live-20260904`.
This checks application at the observed endpoint, not an exact causal decomposition
of every intervening engine tick or multi-agent outcome.

## Loss and targets

Both policies use the single loss and optimizer implementation described in
[POLICY-PROGRAM.md](POLICY-PROGRAM.md#optimization-and-execution). The former
dynamics-ensemble, auxiliary query-imitation, entropy-floor and rate-penalty losses
are absent. Actor, W/L value and ordinary MoE balancing losses train the same
parameter tree.

When history is empty its weight is zero and fresh value weight is one. The initial
mixture gives historical and fresh value losses equal weight, independently of their
sample counts. `--replay-weight 0` is the loss-level replay ablation, with the same
retention and sampling implementation. It is not a second trainer.

The existing strategy reward remains unchanged: projected-winner role loss receives
minus one, an upward loser-rank flip receives plus one, and other transitions receive
zero. Fresh strategy updates retain the existing role-specific TD objective. Retained
strategy targets accumulate observed discounted rewards until a role change or actual
round end. They are not old model predictions.

Terminal-win rewards are zero on nonterminal transitions, one for participants on the
server-reported winning team at termination, and zero for the other teams; a real draw
gives zero to every team. On every fresh transition, including before the first observed
win, its critic target is `r + continuation * V(next_state)` and its PPO advantage is
that target minus `V(state)`. Both existing value heads estimate the same undiscounted
terminal outcome on their respective role rows. The next state's role selects the next
head even when leadership changes. The unit bootstrap discount preserves the specified
eventual-win objective; strategy retains its discounted, role-terminated return.

Both arms take exactly one AdamW step per attributed observation group, with the same
fresh-state grouping, historical batch size, fresh/history loss weights, actor/value/
balancing coefficients, clipping and learning rate. Every observed row participates in
value learning; only each arm's own applied fresh behavior participates in its PPO loss.
There is no sparse-reward warmup gate, outcome-only optimizer or terminal-only epoch
budget. Uncalibrated values can produce unhelpful early advantages; that is measured,
not used to postpone this arm's optimization.

At a real outcome, retained terminal observations receive the actual winning-team label
for later historical value learning. They are not reused for actor learning, and ending
an episode takes zero additional optimizer steps in either arm. A terminal event known
with a fresh transition supplies its reward and stops bootstrapping in that same shared
update. A delayed label still annotates history without reclassifying old actions as
fresh. Missing outcomes do not manufacture terminal zero targets.

A shutdown, missing outcome or observation-context discontinuity is unlabelled, not a
loss or draw. The joint supervisor waits for a published real outcome instead of
assuming a wall-clock process lifetime is a completed game. The literal terminal
condition is a held-checkpoint score threshold, including simultaneous-crossing draws.
No round timer or delivered cart supplies a terminal label.

Historical value targets describe the observed behavior-policy mixture, not exact
current-policy counterfactual returns. They are an explicit predictive auxiliary bias
through the shared value/backbone parameters. No V-trace or joint-policy importance
correction is claimed. The fresh critic is therefore being regularized toward this
historical predictive task; the behavioral effect must be measured.

## Retention and provenance

Each learner has a current-round observation pool and a persistent historical pool,
each initially bounded at 1,024 records and 256 MiB. Shared immutable observation
frames are interned. The byte bound can yield fewer records at large team/cart counts.
Recorded rewards, actions, behavior log probabilities, team IDs, behavior update count,
configuration, match identity and observed return targets travel with the examples.
Learned embeddings are recomputed from the recorded inputs.

At a completed round, up to roughly five percent of the historical count budget is
drawn randomly from the current pool and inserted into history. Overflow eviction is
random within the most represented configuration. Sampling selects configuration,
then match, then state. This prevents a longer match from automatically receiving more
historical value-loss weight. Ordinary zero-reward states remain eligible; the pool
does not select only successes or large errors. Reward/target variance is measured,
not guaranteed or fabricated.

History, sampler RNG and optimization state persist in each policy checkpoint. An
unfinished current-round pool is reported as unlabelled at orderly retirement rather
than checkpointed as a completed training example. Cross-match continuation selects
both preceding policy checkpoints, including after supervisor restart. Recorded
behavior age in optimizer updates is distinct from insertion age in the replay pool.

## Evidence and limitations

Each running match publishes `learning.json`, detailed `telemetry.jsonl`, and an
`outcome.json` after outcome labelling and retention. `policy_updates` and `learning`
are keyed by policy; losses, gradient norms, actor/value row masses, target variance,
update counts, historical configurations/matches and pending-state counts remain
separate. `matches.jsonl` retains both policy checkpoint artifacts and realized game
configurations. The server and worker logs remain part of the same match directory.

The first rollout uses matched residual width 128, hidden width 341, eight routed FFN
banks and top-k two. This is an integration-scale comparison, not the default-width
capacity result. An implementation defect initially skipped all fresh terminal updates
and took four sampled actor steps at each outcome. The old 4 → 4 → 8 → 8 → 12 history
therefore documents an unmatched training schedule, not evidence about sparse terminal
PPO's relative strength. The `fresh-attributed-v2` correction established matched observation cadence.
The September 5 `first-execution-v3` schedule retains one combined step per observation
while joining delayed actor sources and counting each critic interval once. See
[POLICY-EXECUTION-CONTRACT.md](POLICY-EXECUTION-CONTRACT.md). Checkpoints retain their real cumulative counts and record
`schedule_start_updates`; `schedule_updates` counts updates since this correction.
Continuation preserves the already-unequal optimization history. Matched cadence from
this boundary does not retroactively establish equal-budget initialization or Elo.

The retired `--replay-steps` responder argument is accepted with an explicit diagnostic
while an already-running supervisor can still send it. It no longer creates a second
optimizer schedule; newly started supervisors do not pass it.

Random policy-team counts change each policy's opportunity to win. Mixed-policy
matches already supply the evaluation stream: retain actual outcomes, policy/version
history, team-slot assignments, exposure and uncertainty. This is not automatically a
two-player Elo dataset, and it does not require a separate checkpoint tournament.
Leader suppression alone is insufficient: recovery
into contention and the controlled team's eventual success are the relevant outcomes.
No claim of generalization, equal-budget superiority or learned coalition behavior
follows from nonzero gradients or a declining loss.

## Live monitor and userspace participation

The September 4 restoration samples 4/8 teams and 2/3/4 lanes independently, with two
participants per team, four checkpoints per lane, score limit 1,200 and rate one.
One persistent native client is requested through `--human-counts 1` and
`--human-client-command`. The endpoint, server process and client process persist
across `changelevel`; match artifacts and learner contexts change. Client startup
uses `cl_hook_gamestart_plc` to join after game initialization, not a guessed connection
delay. `{directory}` resolves to a match artifact directory, so persistent client
storage should instead use a run-level directory. A requested client is not counted
as joined until server observations contain its human-controller row.

Before game restoration, a verified client match had 32 teams, six carts and 33 participants,
including one human-controlled player. Another had eight teams, twelve carts and nine players.
The now-deleted terminal policy checkpoint after the second launch retained 21 human-state rows
and zero human actor-eligible rows. Both value learners retain those observed states;
solver suggestions and their probabilities are not represented as human behavior
probabilities. This is a historical record of the retired suggestion interface;
the current native state-rate output addresses captured bot views.

The generic whole-mesh telemetry channel receives `match`,
`outcomes`, `learning.matrix_fusion`, and `learning.terminal_win` through the generic
workload-measure publisher, alongside the literal J reports. Update counters, replay
configuration/match coverage, target variance, actor/value row counts, loss components
and gradient norms stay objective-labelled. Terminal updates remain visible between
outcomes with their own sample time. Row counts describe the last optimizer step;
gradient-step counts describe that response. Outcome wins and team-round exposures
are responder-session counts attributed to the assigned team arm, not causal credit
or Elo, especially when a team includes a human controller.
The application dashboards are separate: `http://127.0.0.1:8795/j` for representation
and server relationships, and `http://127.0.0.1:8795/policy` for learning, replay,
checkpoint history and outcomes. They share one incremental reader and cached HTTP
responses. The mesh page on 8787 retains its infrastructure responsibility.

The node ring and mesh observer retain full measures only in their newest sample;
older samples preserve lightweight workload/phase history and explicitly identify
that retention domain. Interactive polling requests `measures=scalars`, preserving
scalar coordinates and array lengths without transmitting every covariance matrix.
`http://127.0.0.1:8788/v1/latest` still provides the complete node record; append
`?measures=scalars` for its compact projection. The mesh dashboard and its JSON
endpoints use the compact projection. Application pages retain selector state across
refreshes; background tabs poll at a lower cadence. Missing optimizer steps are gaps,
not zeros. Historical rating requirements and known coverage limits are described in
[APPLICATION-TELEMETRY.md](APPLICATION-TELEMETRY.md).
The existing J measurement thread atomically retains the complete latest report per
responder/episode as `j-measures.PID.GENERATION.npz` in the match directory, so compact
interactive transport does not discard the literal numerical artifact.
The full report is binary NumPy arrays plus a JSON metadata manifest. Telemetry
carries scalar/array-shape summaries and an artifact descriptor; it no longer
parses and republishes a second copy of the J covariance tensors. The application
download and generic node artifact route stream the existing bytes.

This corrects an observed monitoring failure: the old MacBook telemetry process grew
to approximately 32 GB RSS, and a full node record measured 174,883,949 bytes. A compact
poll measured 17,550 bytes and approximately 1 ms during verification. The full record
is an on-demand numerical artifact, not an appropriate two-second interactive poll.
Measure publication merges namespaces and allows a full asynchronous HTTP transfer to
complete rather than retrying a large body after a half-second timeout. The same node
telemetry changes were installed and queried on the Mini. Only userspace monitoring
services were gracefully restarted; both RDMA bridges remained up with `bad=0`.

During live verification, a missing terminal-update record initially raised an
exception in the new monitor adapter. The supervisor's restart records retain that
failure; absent updates are now represented without calling methods on `None`.
These integration observations do not establish strategic improvement.

## Source ownership and control flow

Continuation restores the complete parameter/optimizer/replay/random/counter state.
A damaged or semantically incompatible source starts a complete fresh learner and
reports its source counter separately; it does not continue the old update history.
`checkpoint_audit.py` supplies the same compact numerical audit for offline inspection.
Next observations retain their complete roster; `successor_rows` selects participant
outputs after policy evaluation. Replay frame identity is independent of owner-local
frame counters.

`policy_contract.py` identifies objectives and architecture equivalence; `runtime.py`
owns reward contracts; `online.py` owns both losses and the one optimizer application;
`replay.py` owns retention, sampling and serialization; `strat_responder.py` owns the
dictionary of learners, actual behavior selection and server outcomes; `curriculum.py`
owns mixed-team schedules, checkpoint continuation and process lifetime. Unused older
single-step/segment learner entry points were removed instead of maintained alongside
the actual attributed-transition path.

New branches select explicitly requested objectives, loss participation, actual
episode labels, measurement availability or normal service lifecycle. They neither
withhold a node capability nor change bridge, firewall, power or accessibility policy.
