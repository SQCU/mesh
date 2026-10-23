# Source-to-claim citation index

This file maps each source surface to an implementation claimfile, and each claimfile
to the documents it cites.

| Source surface | Claimfile | Documents cited |
|---|---|---|
| `xonotic/solver/strat/`, `xonotic/solver/xonwire.py` | [`POLICY.md`](POLICY.md) | `SPECIFICATION.md`, `rl-training-spec.md`, `joracle-viewer.md`, `MATRIX-EXECUTION-SPEC.md`, `ALGORITHM-CONTRACTS.md` |
| `xonotic/payload/tools/`, `xonotic/mapgen/` | [`GEOMETRY.md`](GEOMETRY.md) | `NAV-SPEC.md`, `FUSION-SPEC.md`, `ALGORITHM-CONTRACTS.md` |
| `xonotic/qcsrc/`, `xonotic/darkplaces-work/bot_batch*` | [`ENGINE-SCALE.md`](ENGINE-SCALE.md) | `xonotic-bot-compute.md`, `rl-training-spec.md`, `ALGORITHM-CONTRACTS.md` |
| `rdma/`, `user/mesh-telemetry.py`, `bin/mesh-observe.py` | [`MESH-TELEMETRY.md`](MESH-TELEMETRY.md) | `mesh-coprocessor-demo.md`, `rl-training-spec.md`, `ALGORITHM-CONTRACTS.md` |
| `viz/`, `xonotic/render/`, `xonotic/payload/runtime-package.py` | [`RUNTIME.md`](RUNTIME.md) | `mesh-coprocessor-demo.md`, `FUSION-SPEC.md`, `NAV-SPEC.md`, `ALGORITHM-CONTRACTS.md` |
| `xonotic/solver/strat/joracle/client-keep.sh` | [`RUNTIME.md`](RUNTIME.md) | `mesh-coprocessor-demo.md`, `FUSION-SPEC.md`, `ALGORITHM-CONTRACTS.md` |
| `install*.sh`, `enable-autologin.sh`, `serve.sh`, `bin/mesh-bridge.sh` (deleted in 67f959bf, 2026-10-01), `bin/mesh-networks.sh` | [`ACCESSIBILITY.md`](ACCESSIBILITY.md) | `THREAT-MODEL.md`, `design/rubrics.md`, `design/transport.md`, `ACCESS-TOPOLOGY-SPEC.md` |
