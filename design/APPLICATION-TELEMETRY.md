# Application viewers and historical policy reports

Current execution, durable reporting and producer discovery are documented in
[joracle-viewer.md](joracle-viewer.md) and
[POLICY-IMPLEMENTATION-20260907.md](POLICY-IMPLEMENTATION-20260907.md).
The records below describe earlier source and deployments.

## Restored responsibilities, September 4, 2026

| Page | Responsibility | Source |
| --- | --- | --- |
| `8787/` | Mesh pages, links, machine activity, workload capacity and deadlines | Generic node telemetry |
| `8795/j` | Selected native J, labelled representation strata, geometry, feature/state relationships, outcomes and applied routes | Existing literal J producer plus current game telemetry |
| `8795/policy` | Objective-labelled updates, losses, actor/value rows, replay mixture, checkpoint history and observed rounds | Game telemetry and completed-launch index |

Run from `xonotic`:

```
../bin/mesh-python -m solver.strat.joracle.server --run-dir solver/strat/runs/checkpoint-control-live-20260904
```

The reader follows new launch directories. Both pages use the same reader and cached
JSON; opening more browser tabs does not multiply numerical analysis. This process
is a userspace observer and never opens a verbs device or changes bridge state.

The game restoration invalidates prior delivery-variant policy results and all of
their resumables have been deleted. Old logs are diagnostic artifacts, not a rating
baseline. Outcome reports must identify `checkpoint-control-integral-v1`; the viewers
are pointed at the new lineage, not a mixture of pre/post-restoration directories.

The old application server reran rolling analysis independently of the producer,
retained hundreds of large raw frames, and did decomposition work in request handling.
Its J selection could choose the last matching feature stratum without identifying
policy semantics or native shape. Browser interval polling could overlap slow requests.
The preceding generic HUD also repeated application scalar trees inside infrastructure
presentation. These were distinct errors, not reasons to remove application viewers.

The replacement retains 300 compact trajectory rows and the newest sampled model,
reads complete JSONL records incrementally by file/inode/offset, and caches serialized
responses on a two-second cadence. A partial trailing record waits for its newline.
Browser requests never load the full covariance artifact. Native J labels and feature
schema identify a selection; different policies are never padded into one heatmap.
The full producer artifact remains an explicit streaming download. The producer's
pending-frame FIFO now evicts in constant time instead of shifting a Python list.
The unused secondary `RollingProbe` worker and its duplicate spectral decomposition
were removed; the existing literal producer remains the numerical source of truth.

Source-file age, reader-cache age, J artifact age and sampled response identify
different clocks. Geometry is the producer's centered-J affine-projection spectrum;
coordinate variance is separately labelled. Neither is a held-out representation test.
The historical launch index is read once, retaining compact records rather than full
manifests. The current UI is a rolling operational view, not a historical analytics DB.

## Numerical artifacts and polling cost

The J producer keeps numerical results as NumPy arrays and writes uncompressed NPZ
artifacts with a small JSON tree manifest (`numpy-npz-tree-v1`). Array values, shapes,
dtypes and nonfinite coordinates survive without decimal conversion or Python float
lists. `solver.strat.joracle.artifact.read_report(path)` reconstructs the report;
loading never enables pickle. The compact `.npz.view.json` retains the existing
dashboard fields. `/api/j/full` streams the artifact bytes, without decoding them.
Historical JSON reports remain readable by the viewer.

The workload channel now publishes labelled section-level scalar/array-shape summaries plus an
`artifacts.j` descriptor, not another full covariance tree. The generic node service
adds a `/v1/artifact` URL resolved from that producer's registered descriptor and
streams its bytes. This is an explicit numerical export, not a file-polling telemetry
bus. Nested coordinate maps are labelled by their entry counts, with their complete
contents in the artifact; they are not another multi-megabyte scalar string dump in
the infrastructure stream. The J page retains its detailed application projection.
Learning and outcome namespaces retain their existing values and cadence.

Each immutable node-ring record caches its full and scalar JSON bytes. History reads
join cached records with a small sequence envelope; multiple readers do not repeatedly
convert the same floats. Advancing the ring replaces the preceding record with its
labelled historical projection and releases that record's full-measure cache.
Legacy full JSON workload publications remain supported.

The reporter waits when there are no new frames instead of recomputing the same
covariances. This branch removes duplicate work, not an observation or capability.
Artifact routing adds access to registered exports; missing files are reported as
errors rather than replaced with invented measurements. Producer generation changes
still clear the previous report, and publication follows the atomic artifact write.

## What the policy page does and does not establish

Value targets, actor eligibility and policy identity retain the learning contract in
[JOINT-POLICY-LEARNING.md](JOINT-POLICY-LEARNING.md). Historical behavior-mixture returns
train value learning; human or other-policy rows are not on-policy actor samples.
Loss coordinates stay objective-labelled. A terminal update that has not happened is
missing, not a zero. Last-update row counts describe one optimizer step, not an entire
match or the batch of gradient steps. Update counts are not equal compute budgets.
The policy cards also report fresh/history batch sizes and `schedule_updates` since
the shared `fresh-attributed-v2` cadence began. Lifetime counts include the earlier
outcome-only terminal optimizer defect and must not be treated as a matched experiment.
The measured value-target variance includes fresh bootstrapped TD targets, not merely
the variance of immediate rewards; zero terminal reward does not imply zero TD signal.

Completed launch history and the new `server-outcomes.jsonl` survive a viewer restart.
The round table joins that engine-originated ledger with launch streams consumed by
the viewer, including their existing lines at first read. This does not retroactively
fill the earlier telemetry gap; see [the incident RCA](TELEMETRY-LOSS-20260904.md). A launch
can contain several terminal rounds; neither a launch count nor a final outcome file
is a round denominator. No Elo is fitted by these viewers.

An earlier supervisor continuation reused textual match IDs after restarting its cycle
counter. The resume lookup is repaired, but existing data must be joined by run root,
launch ordinal/directory, responder generation and engine event identity—not the short
match ID alone. Replay's existing match-group count is not a count of unique launches.

## Reports needed for an effective historical Elo

First materialize one retrospective round ledger from all retained telemetry streams
and authoritative terminal events, with explicit source coverage and duplicates:

- Run, launch ordinal/directory, responder generation, round start/end, engine event
  identity, actual winner or tie, interruption/truncation, and reporting wall times.
- Map and geometry, team/cart/player counts throughout the round, starting positions,
  randomization seed, team assignment and each team's bot/human composition over time.
- Complete policy provenance and checkpoint lineage; optimizer version at action
  sampling, delivered/applied route sequences, and version range over the round.
- Per-arm team-round and player-time exposure, actor/value training rows, optimizer
  steps, examples and measured compute. Unequal step schedules remain visible.

The first report is descriptive: wins, ties, participation and exposure by policy,
time, map, team/cart counts and controller composition. Include coverage and uncertainty,
not just running percentages. Do not turn a human-assisted team's win into exclusive
credit for its assigned bot policy. Preserve the full multi-team round as the unit of
observation; expanding one round into many independent pairwise wins invents evidence.

A proposed *effective* rating can then be an explicitly conditional model. For a
single-winner round with comparable autonomous teams and strengths `r_A`, `r_B`, choose

```
w_A = 10^(r_A / 400)
w_B = 10^(r_B / 400)
P(winner assigned A) = n_A w_A / (n_A w_A + n_B w_B)
```

This defines an exposure-adjusted multinomial model, not a finding that these strategic
games satisfy its assumptions. It is the single-choice proportional-worth form
described in the [PlackettLuce authors' model overview](https://hturner.github.io/PlackettLuce/articles/Overview.html),
with equal-worth teams grouped by arm; the overview also treats ties and conditional
worth models. That reference supplies a model family, not evidence about our policies.
With equal strengths the expected arm win share is
`n_A / (n_A + n_B)`, not one half. The 400 scale is a reporting convention. Estimate a
relative difference with uncertainty, conditioned or stratified by time/checkpoint,
team/cart counts, map, composition and arm mixture. Teams can form coalitions, so a
single constant strength may fail badly as the mixture changes. Report that model's
calibration and residuals; do not hide them behind a smooth Elo curve. Handle ties and
incomplete rounds explicitly before fitting, and cluster uncertainty at the actual
dependent round/launch unit. Sparse slices remain sparse evidence.

Online checkpoints change within rounds. A rating fitted to that history describes
training trajectories under the observed opponent mixture; it is not the Elo of the
checkpoint saved at the end. An equal-architecture, equal-budget comparison needs
version-matched reports. Mixed-policy matches are the requested evaluation stream;
they do not require periodic fixed-checkpoint tournaments. Report each policy's actual
version and initialization lineage over the observed interval, including fresh
recovery after damaged continuation. A held-out generalization experiment would
answer a separate question if requested.

Finally report strategy directly: loser acquisition/recovery, winner retention,
leader-directed opposition, objective switching and coalition outcomes, with source
route joins and opportunities as denominators. Neither a value loss nor a relative
win rating alone establishes that the policy learned the intended strategic behavior.
