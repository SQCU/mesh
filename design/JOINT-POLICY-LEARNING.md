# Two objectives, one learning implementation

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

Per-policy actor masks select its own freshly generated behavior actions, excluding
explicit uniform exploration and human controllers, and require the next observation's actual route sequence
to identify that action. Value masks include all observed teams. The other
policy's actions are not treated as supervised demonstrations or as on-policy samples.
The shared-backbone value gradient nevertheless changes counterfactual action outputs.
Off-policy actor improvement from the other policy's actions is not implemented by
silently applying an ordinary PPO objective to them.

The initial runtime exposed a queued-observation defect: response 146 was being
optimized while the observed players still used response 58. The responder now acts
on the newest complete synchronized observation, counts coalesced older observations,
and preserves all intervening perception and outcome events. PPO eligibility requires
the server's applied route sequence to match the source action; an emitted response is
not by itself evidence of application. The initial `joint-20260904` artifacts remain
available but their optimization lineage is not reused by `joint-live-20260904`.
This checks application at the observed endpoint, not an exact causal decomposition
of every intervening engine tick or multi-agent outcome.

## Loss and targets

```text
for policy in policies:
    fresh = objective_defined_training_records(policy)
    history = sample_configuration_then_match_then_state(policy.replay)
    loss = mean(actor_loss(fresh))
         + (1 - replay_weight) * mean(value_loss(fresh))
         + replay_weight * mean(value_loss(history))
         + existing_fresh_regularizers_and_dynamics_loss
    gradient = globally_clip(derivative(loss, policy.parameters))
    AdamW.update(policy.parameters, gradient)
```

When history is empty its weight is zero and fresh value weight is one. The initial
mixture gives historical and fresh value losses equal weight, independently of their
sample counts. `--replay-weight 0` is the loss-level replay ablation, with the same
retention and sampling implementation. It is not a second trainer.

The existing strategy reward remains unchanged: projected-winner role loss receives
minus one, an upward loser-rank flip receives plus one, and other transitions receive
zero. Fresh strategy updates retain the existing role-specific TD objective. Retained
strategy targets accumulate observed discounted rewards until a role change or actual
round end. They are not old model predictions.

Terminal-win targets are one for participants on the server-reported winning team and
zero otherwise; an actual server-declared draw gives zero to every team. They are
undiscounted terminal outcomes, span changes of leader, and do not inherit the strategy
critic's role-change termination rule. Terminal policy optimization begins only after
an outcome arrives, using retained current-round observations/actions. Both existing
value heads fit that same terminal target on their respective role rows.

A shutdown, missing outcome or observation-context discontinuity is unlabelled, not a
loss or draw. The joint supervisor waits for a published real outcome instead of
assuming a wall-clock process lifetime is a completed game. The engine's own round
timer may declare a winner or draw; that is an actual game outcome.

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
`outcome.json` after outcome-labelled optimization. `policy_updates` and `learning`
are keyed by policy; losses, gradient norms, actor/value row masses, target variance,
update counts, historical configurations/matches and pending-state counts remain
separate. `matches.jsonl` retains both policy checkpoint artifacts and realized game
configurations. The server and worker logs remain part of the same match directory.

The first rollout uses matched residual width 128, hidden width 341, eight routed FFN
banks and top-k two. This is an integration-scale comparison, not the default-width
capacity result. Strategy updates happen online; terminal updates happen after round
outcomes. Update counts and compute budgets consequently differ and are not yet a
budget-matched algorithm comparison.

Random policy-team counts change each policy's opportunity to win. Training outcomes
are not automatically a two-player Elo dataset. Evaluation must rotate team slots,
account for policy team exposure, use stable unseen configurations and retain actual
matchup outcomes and uncertainty. Leader suppression alone is insufficient: recovery
into contention and the controlled team's eventual success are the relevant outcomes.
No claim of generalization, equal-budget superiority or learned coalition behavior
follows from nonzero gradients or a declining loss.

## Live monitor and userspace participation

The September 4 continuation of `joint-live-20260904` runs three-minute rounds and
requests one native userspace client per match through `--human-counts 1` and
`--human-client-command`. Joint schedules now honor those existing options. Client
commands substitute the actual `{port}` for every launch; clients join the match and
retire through the same curriculum lifecycle as their server. `{directory}` also
resolves to the match artifact directory. A requested client is not counted as joined
until the server observations contain its human-controller row.

The first verified client match had 32 teams, six carts and 33 participants, including
one human-controlled player. The next had eight teams, twelve carts and nine players.
The terminal policy checkpoint after the second launch retained 21 human-state rows
and zero human actor-eligible rows. Both value learners retain those observed states;
solver suggestions and their probabilities are not represented as human behavior
probabilities. The server sends suggestions to humans without applying the bot route.

The existing whole-mesh dashboard at `http://127.0.0.1:8787/` receives `match`,
`outcomes`, `learning.matrix_fusion`, and `learning.terminal_win` through the generic
workload-measure publisher, alongside the literal J reports. Update counters, replay
configuration/match coverage, target variance, actor/value row counts, loss components
and gradient norms stay objective-labelled. Terminal updates remain visible between
outcomes with their own sample time. Row counts describe the last optimizer step;
gradient-step counts describe that response. Outcome wins and team-round exposures
are responder-session counts attributed to the assigned team arm, not causal credit
or Elo, especially when a team includes a human controller.

The node ring and mesh observer retain full measures only in their newest sample;
older samples preserve lightweight workload/phase history and explicitly identify
that retention domain. Interactive polling requests `measures=scalars`, preserving
scalar coordinates and array lengths without transmitting every covariance matrix.
`http://127.0.0.1:8788/v1/latest` still provides the complete node record; append
`?measures=scalars` for its compact projection. The mesh dashboard and its JSON
endpoints use the compact projection. Expandable measure groups preserve their open
state across refreshes; background tabs keep a low-cadence scalar heartbeat.
The existing J measurement thread atomically retains the complete latest report per
responder/episode as `j-measures.PID.GENERATION.json` in the match directory, so compact
interactive transport does not discard the literal numerical artifact.

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
