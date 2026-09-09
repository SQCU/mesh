# J-space and policy viewers

The application viewer serves `/j` and `/policy` on port 8795. Infrastructure
telemetry is a separate service. Policy inputs, native state views and execution
limits are documented in [POLICY-PROGRAM.md](POLICY-PROGRAM.md).
The controlling operator requirements are in [SPECIFICATION.md](SPECIFICATION.md).

## Owners and data flow

| Data or operation | Owner |
| --- | --- |
| Complete issued source frame, sampled rate, behavior likelihood, policy version and native application join | `strat/action_history.py` |
| Addressed state/residual words, metadata, complete event rows, timing and per-row learned representation | `strat/strat_responder.py` |
| Retained source/transition window, moments, exact covariance factors and empirical affine projections | `strat/joracle/probe.py` |
| Lossless arrays, shared references and interned coordinate labels | `strat/array_tree.py`, used by checkpoints and `joracle/artifact.py` |
| Columnar viewer sidecar and exact coordinate slices | `strat/joracle/display.py` |
| Retained producer records, HTTP views and source handover | `strat/joracle/server.py` |
| Coordinate selection, plots and tables | `strat/web/app.js` |

There is no target/instrument candidate table, manual behavioral vocabulary,
observation-slot average or decayed belief summary in this data flow. The policy
retains addressed state pages and native event rows through learned encoding.
The viewer does not synthesize a replacement policy input or action decoder.

## Literal representation and source measurements

Each source retains its own feature and J labels, policy arm/version, participant,
response sequence and state schema. Labels identify actual entity generations,
page/word addresses, residuals, packed metadata, native events and timing. Unused
physical padding is excluded from measurement. Different schemas form explicit
strata rather than being averaged into a shared-width surrogate.

The lens measures coordinate means, variances, cross moments and covariance on the
retained source rows. These are statistical outputs, not summaries fed back as
policy observations. The empirical affine projection fits source or outcome
coordinates from centered J samples. It reports the observed rank, singular values
and residual mean squares. This describes linear information in a retained sample;
it does not prove causal control or universal representation capacity.

Large covariance matrices are stored exactly as `left.T @ right / mass`. Affine
operators are stored as factors obtained in sample space. Row norms and Frobenius
norms can be computed without constructing the full coordinate-by-coordinate
matrix. The full artifact retains the factors and their coordinate labels. Finite
sample mass, original mass and alignment residuals remain explicit.

The oracle joins issued sources to first observed native state applications,
repeated/delayed applications, outcomes and observed state changes. An application
acknowledges a Havocbot invocation under the issued state view; it does not assert
arrival at a location or causal responsibility for a later outcome. Actor eligibility
and durable outcome identity are described in
[TRAINING-CONTINUATION.md](TRAINING-CONTINUATION.md).

## Storage, pagination and continuity

The producer atomically writes `j-measures.PID.GENERATION.npz` and a smaller
`*.npz.view.npz` sidecar. Both use the shared array/tree codec, with coordinate label
sequences stored as integer indices into a string table. Shared arrays and labels
are preserved without repeatedly expanding them into JSON. Legacy NPZ trees and
`*.view.json` documents remain readable.

`/api/j` accepts `stratum`, `filter`, `offset` and `width`. It returns the requested
exact feature-coordinate slice and J-coordinate window, along with full native
widths, matched-coordinate counts and retained row counts. The UI exposes these
controls. Filtering searches all labels in the selected stratum. No averaging is
used to make the viewport smaller; `/api/j/full` provides the complete numerical
artifact. Gram-factor plots likewise display an explicitly selected coordinate
window rather than a fabricated participant-by-participant matrix.

The view remains on its last measured source until the next source has a complete
telemetry frame. Empty directories, partial records and producer restarts do not
replace measured values with zero. The header reports source identity, age, pending
handover and reader/report errors. The viewer does not request new RDMA contexts or
restart a bridge to read application telemetry.

`/policy` reports optimizer updates, loss components, gradient norms, replay and
actor/value row counts, source age, application joins and checkpoint provenance.
Same-input policy comparisons distinguish output distributions, sampled rates and
integrated residuals. A deterministic stock policy and a Gaussian policy have
mutually singular distributions; their KL is not assigned a fabricated finite value.
The full-vector endpoint retains the sampled comparison data independently of the
compact tables. `ratings.py` joins literal bot configurations, acknowledged
controller exposure and completed outcomes in the durable report store. Its
regularized Bradley–Terry estimates report conditional team/factor ratings,
posterior uncertainty, identified rank and excluded rounds. These estimates do
not establish causal separation between bot capabilities and their controllers.
The estimator is described in the
[implementation ledger](POLICY-IMPLEMENTATION-20260907.md#4-configuration-exposure-and-conditional-ratings).

The viewer retains reports, optimizer history, source contexts and rating inputs in
SQLite under `.joracle`, or `--state-directory`. Journal identities survive rsync
replacement and incomplete tails. A producer publishes
`~/.local/share/mesh/xonotic-active.json` after its first measured frame. With
`--replicate-from mesh:xonotic`, the viewer discovers these registries locally and
on mesh nodes, follows the advancing producer and retains the previous complete
replica during handover or connection loss. Discovery candidates and failures
appear in replication status. An explicit `host:path` remains an explicit source.

## Viewer and historical evidence

```sh
PYTHONPATH=xonotic:rdma bin/mesh-python -m solver.strat.joracle.server --run-dir "$HOME/.cache/mesh/joracle-replica" --replicate-from mesh:xonotic --port 8795
```

To inspect a fixed local run, omit `--replicate-from` and supply that run as
`--run-dir`. The replica directory above is a local cache, separate from producer
run directories.

[STATE-REDUCTION-RCA.md](STATE-REDUCTION-RCA.md) records the full-width report failure:
a duplicated label manifest reached about 3 GB and a viewer JSON document about
384 MB, delaying graceful learner shutdown. The shared codec and columnar sidecar
repair that representation problem. Historical observations included lossless
arrays/labels, legacy reading, producer handover and an HTTP request reaching
coordinates beyond the first page. The verification suite was subsequently removed.
Match evidence is retained in
[the representation evidence directory](../measurements/state-representation-20260906/README.md).
