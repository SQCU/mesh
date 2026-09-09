# Application deployment and ongoing implementation evidence

This directory records actual builds, deployment, and a native two-bot, two-team,
four-cart match across the laptop and Mini. It is not a test suite or specification.
The application deployment contract is in
[APPLICATION-DEPLOYMENT.md](../../design/APPLICATION-DEPLOYMENT.md).

## Deployment failure and repair

The first isolated deployment copied solver and payload tools but symlinked `rdma` to
the Mini's independently maintained backend checkout. Startup successively exposed a
missing `xonwire.def`, missing `workload.py`, and an older Python `Mesh.read` signature.
These were repaired inside that isolated application only. The manual recipe was then
replaced by `bin/mesh-application.py`, with its source/resource closure in `pyproject.toml`.

The new path built its own native client from staged sources, installed a fresh locked
environment on the Mini, imported every declared entrypoint, published a generation,
and ran the responder CLI. It subsequently launched the real learner on the Mini and
the viewer on the laptop. Both processes reported application ID
`52163540f4b9a12d3646d6c56b906faf50accc09d2eabb71ec0ca674d0bb68d5`, native client digest
`51ce8445a0e9e8c7f7a6df99cb9bffc4f8c351fcb659a7ee1ff864bc2ee42e78`, and wire digest
`c9ea4038ec149c8f0a89e88c4b5ec2de6d89747f9f7b54e674ab449d97774340`.
The Mini's application package is independent of `/Users/mdot/mesh/rdma`.

A later deployment published the launcher refinements while the learner continued
using its original complete generation. The two bridge processes remained PID 55709
and PID 93690; both were paired/responsive with `bad=0` in the recorded census.
No bridge restart or backend-library replacement was performed.
A repeated deployment selected the same intact generation and dependency environment.
After recording the measurement, the game exited through its console and the learner
completed ordinary TERM checkpointing. The local viewer on port 8797 remains available
with its durable replica; ending the producer does not delete the observations.

Local operational logs, ignored by Git, include `application-deploy.log`,
`application-launch.log`, `application-update.log`, `application-refresh.log`,
`viewer-launch.log`, `native-build.log`, and `qc-build.log`. Source syntax parsing,
shell parsing, JavaScript parsing, and `git diff --check` completed successfully.
No repository-owned tests or verification harnesses were added.

## Live observations

[observations.json](observations.json) summarizes the recorded viewer responses.
[viewer-status.json](viewer-status.json), [viewer-policy.json](viewer-policy.json), and
[viewer-j.json](viewer-j.json) preserve the actual HTTP output. At response 320, both
policies had performed 318 optimizer updates; each fresh group credited one actor row
and two value rows. The viewer reported optimizer data, 32 J strata, two policy-pair
comparisons, and four counterfactual output rows. Its replica had recovered automatically
after an rsync race with an unpublished `.new` J file. The replica now excludes temporary
publication files before including public artifact families.

Input and transition buffers were reused between capacity changes: the recorded shared
input arena had 320 loads and seven allocations, and each learner had 636 training frame
loads and twelve allocations. These counts include growth and realization. They do not
prove that every kernel has one lifetime shape or that MLX allocates no intermediates.
Response timing includes preparation, with a separate preparation duration.

The native engine and QC PK3 were built from current sources. The learner's intended
small dimensions were rank 128, hidden 341, eight experts, top two. An earlier invocation
mistakenly omitted these flags and realized the large defaults. It spent minutes compressing
multi-gigabyte checkpoints; ordinary TERM completed and released its client slot. Its
checkpoints and logs remain in the remote `00000-live` directory for diagnosis.

## Findings still open

- Before the map transition, a long-lived map emitted STATE snapshots requiring 7,046
  transport fragments. Reassembly repeatedly received roughly 4,100–4,500 before the next
  snapshot superseded the incomplete one. A normal `changelevel runningmanctf` restored
  complete 59-fragment snapshots and learning began. Resetting the map is evidence about
  the failure condition, not a fix for long-lived-map starvation.
- The optimizer is executing, but actor importance weights collapsed in this run. At the
  recorded response, numerical-floor fractions were 0.94141 and 0.98047, with very low
  effective sample fractions. This does not establish useful actor learning. Probability
  provenance and full-dimensional likelihood dynamics require further review.
- Navigation realization was omitted from this particular launch and correctly reported
  unavailable. It does not establish the integrated geometry path operationally.
- Recorded historical replay batches were zero: no completed-match replay/restart outcome
  claim follows from these samples.
- Capacity growth, compiled distributed reduction integration, and bot/team/controller
  composition ratings remain unfinished. The viewer explicitly reports the latter as
  `not_implemented`.
