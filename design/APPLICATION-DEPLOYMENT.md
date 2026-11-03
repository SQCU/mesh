# Canonical application deployment

`bin/mesh-application.py` synchronizes committed `main` checkouts and builds each
participant's canonical `rdma` source. It transfers Git objects, verifies matching
revisions, and activates the checkout through a `current` symlink. It does not
copy application source snapshots or create alternate numerical runtimes.

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
