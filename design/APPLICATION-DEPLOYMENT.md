# Complete application deployment

The September 6 failure was a packaging defect. The dependency installer only installed
Python, MLX, and NumPy. The demo and curriculum separately copied selected application
directories. An isolated deployment then borrowed the Mini's backend checkout: it lacked
`xonwire.def`, lacked `workload.py`, and supplied a `Mesh.read` without `max_batches`.
The repair is to remove independent file synchronization from normal launches.

`pyproject.toml` owns one application source/resource closure: the complete `rdma`,
`xonotic/solver`, and `xonotic/payload` trees and the runtime installer/launcher inputs.
Runtime output directories and bytecode caches are excluded. New code and resources in
these trees automatically travel with the application; launch sites contain no file lists.
Engine builds, game PK3s, maps, checkpoints, and telemetry are separate artifacts, with
their existing build and wire contracts. They are not substituted by this Python package.

`bin/mesh-application.py` owns snapshot, transfer, activation, and execution:

1. Copy the source trees into an isolated staging directory and build `libmesh.dylib`
   there from those sources. No build runs in the bridge checkout. No backend symlinks
   survive into the application.
2. Transfer into a new generation directory. Its inventory covers every packaged file,
   including native client, Python client, wire definition, viewer assets, and lockfile.
3. Resolve the locked dependency environment. The existing environment is reused only
   after `uv sync --frozen --check` and imports succeed. Repairs create another environment;
   they do not mutate the environment of a running process.
4. Import the application entrypoints with isolated Python import paths, then atomically
   publish `current`. Identical, intact application/environment pairs reuse their generation.
5. Resolve `current` once and execute from that concrete directory and Python environment.
   Later imports stay in that generation even when the deployment pointer changes. Foreground
   launch replaces its process so TERM still reaches the actual local application.

`launch` always attempts deployment before execution. If updating fails, it reports the
failure and attempts the previous complete generation. An incomplete transfer, failed
dependency realization, or import/link failure cannot replace `current`. These checks
prevent demotion of the available installation; they do not disable it for being old.
`deploy` exposes failure to the caller while preserving `current`; node installers continue
their independent work. The next launch or periodic named-branch installation retries.
On a first installation with no usable generation, errors identify the failed realization;
there is no fictional fallback application to report as running.

The demo uses `launch` directly. Curriculum defaults use it for local responders and workers;
remote worker deployment finishes before any existing worker receives TERM. Both node
installers realize `applications/cartlane` after opening access and provisioning services.
Explicit custom curriculum commands remain operator overrides and report `unbundled` unless
they use the launcher. A direct `python -m` from an arbitrary checkout is development execution,
not deployment; it cannot inherit completeness from an unrelated installed environment.

Example from the working source tree:

```sh
bin/mesh-python bin/mesh-application.py launch \
  --host mesh-mini-local --python /usr/local/mesh/bin/mesh-python \
  --target /Users/mdot/mesh-workloads/cartlane/application \
  --log /Users/mdot/mesh-workloads/cartlane/responder.log \
  -- solver.strat.strat_responder --train \
  --train-arms matrix_fusion,terminal_win --team-policy-arms matrix_fusion,terminal_win \
  --scale-rank 128 --scale-hidden 341 --scale-experts 8 --scale-topk 2
```

`run` is the primitive for executing an already published generation. Supervisors should
use `launch` when their source tree is available. Deployment does not hot-reload a training
process or restart a bridge. Application content digests identify bytes and cache entries;
they never select a Git commit. Repository convergence still fetches the newest named branch.

Policy records carry application ID, immutable source directory, concrete Python environment,
native-client digest, and wire digest. Viewer `/api/status` identifies both producer and viewer;
different process generations are visible. They may coexist during upgrades. An application
generation alone does not prove engine wire compatibility or compatibility with arbitrary
future shared-memory ABI changes. `MESH_VERSION` remains the backend ABI contract; incompatible
native protocol changes need coordinated compatibility support in that owner, not a launcher
that quietly rewrites a live bridge.

Operational evidence and outstanding findings are in
[the September 6 run record](../measurements/policy-tranches-20260906/README.md).

The September 7 transport ownership change requires a matching rebuilt engine and
Python application generation. The native library adds `mesh_try_open` and
`mesh_close`; the local game/runtime envelope declares `LOCAL_VERSION` from the
shared wire definition. The curriculum launches the numerical runtime for every
server and retains it across maps, including runs with local-only policy arithmetic.
The engine is still a separately built deployment artifact; the application
manifest does not currently certify its source correspondence. No deployment or
build of this ownership change has been performed.
