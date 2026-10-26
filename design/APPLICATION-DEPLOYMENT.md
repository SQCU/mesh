# Canonical application deployment

`bin/mesh-application.py` selects a committed revision and asks each participant
to fetch it from that participant's Git `origin`. Each node builds its `rdma`
source and activates the checkout through a `current` symlink.

The hourly uses the same source path for its ordinary examples: each node fetches
the selected mesh and microbench commits from their Git upstreams. Git worktrees
hold the build directories. The installer removes an older worktree after its
replacement is active. SSH starts processes and passes configuration operands;
it does not carry source archives. Commits must be published before the run.

`deploy` synchronizes/builds and activates the checkout. `activate` publishes an
existing canonical checkout. `run` executes a retained module from the activated
checkout; `launch` performs deployment before execution. Engine binaries, assets,
recorded observations and other application data remain separately owned.

The old persistent-policy responder, remote worker, curriculum and their demo
launchers were deleted. They are not deployment examples. The retained
`solver.strat.joracle.server` serves existing recorded observations; its live
service is independent of those removed numerical bindings. See
[the Xonotic README](../xonotic/README.md) and [caller migration](caller-migration.md).

Historical packaging observations remain in
[the September 6 run record](../measurements/policy-tranches-20260906/README.md).
They do not describe the current implementation or certify compatibility with a
new shared-memory ABI. Bridge protocol changes are coordinated by canonical mesh.
