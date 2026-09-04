# Source-to-claim citation index

This file is the repository-level citation carried by source after local prose is
removed. Each source surface below cites an implementation claimfile; every claimfile
in turn cites its controlling data-flow and control-flow specifications. The relation
is deliberately centralized so executable files cannot redefine their contracts by
restating them locally.

| Source surface | Claimfile | Controlling specifications |
|---|---|---|
| `xonotic/solver/strat/`, `xonotic/solver/xonwire.py` | [`POLICY.md`](POLICY.md) | `SPECIFICATION.md`, `rl-training-spec.md`, `joracle-viewer.md`, `MATRIX-EXECUTION-SPEC.md`, `ALGORITHM-CONTRACTS.md` |
| `xonotic/payload/tools/`, `xonotic/mapgen/` | [`GEOMETRY.md`](GEOMETRY.md) | `NAV-SPEC.md`, `FUSION-SPEC.md`, `ALGORITHM-CONTRACTS.md` |
| `xonotic/qcsrc/`, `xonotic/darkplaces-work/bot_batch*` | [`ENGINE-SCALE.md`](ENGINE-SCALE.md) | `xonotic-bot-compute.md`, `rl-training-spec.md`, `ALGORITHM-CONTRACTS.md` |
| `rdma/`, `user/mesh-telemetry.py`, `bin/mesh-observe.py` | [`MESH-TELEMETRY.md`](MESH-TELEMETRY.md) | `mesh-coprocessor-demo.md`, `rl-training-spec.md`, `ALGORITHM-CONTRACTS.md` |
| `viz/`, `xonotic/render/`, `xonotic/payload/runtime-package.py` | [`RUNTIME.md`](RUNTIME.md) | `mesh-coprocessor-demo.md`, `FUSION-SPEC.md`, `NAV-SPEC.md`, `ALGORITHM-CONTRACTS.md` |
| `xonotic/solver/strat/joracle/demo.sh`, `xonotic/solver/strat/joracle/evaluate-distributed.sh`, `xonotic/solver/strat/joracle/client-keep.sh` | [`RUNTIME.md`](RUNTIME.md) | `mesh-coprocessor-demo.md`, `FUSION-SPEC.md`, `ALGORITHM-CONTRACTS.md` |
| `install*.sh`, `enable-autologin.sh`, `serve.sh`, `bin/mesh-bridge.sh`, `bin/mesh-kill-guard.sh`, `bin/mesh-networks.sh` | [`ACCESSIBILITY.md`](ACCESSIBILITY.md) | `THREAT-MODEL.md`, `AGENTS.md`, `RDMA-RULES.md`, `ACCESS-TOPOLOGY-SPEC.md` |
