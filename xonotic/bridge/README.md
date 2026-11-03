# Xonotic server ↔ mesh bridge

The bridge has one engine implementation and one transport implementation.

- `../darkplaces-work/mesh_ipc.c` implements the QuakeC builtins.
- `../../rdma/mesh.h` and `../../rdma/mesh-client.c` implement the transport used by the engine.
- `../qcsrc/common/gamemodes/gamemode/payload/mesh_ipc.qh` declares the builtins used by game code.
- `../solver/strat/strat_responder.py` consumes requests and publishes policy responses.

Darkplaces includes `mesh_ipc.c` from `prvm_cmds.c`. The bridge source includes
the canonical transport files directly from `rdma/`; there are no copied headers,
copied C files, generated patches, or patch-application scripts.

The active builtins are:

| number | function |
|---|---|
| 644 | `mesh_open` |
| 648 | `mesh_gather` |
| 649 | `mesh_scatter` |
| 650 | `mesh_publish` |
| 651 | `mesh_poll` |
| 653 | `mesh_stat` |

Both server and client VM builtin tables register the same functions. Payload uses
non-blocking publish and poll operations; solver latency cannot block the server
frame.
