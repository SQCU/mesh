# Xonotic server ↔ mesh bridge

The bridge has one engine implementation and one transport implementation.

- `../darkplaces-work/mesh_ipc.c` implements the QuakeC builtins and the engine's end of
  its local socket (`MESH_EXPERT_SOCKET`).
- `../solver/xonwire.py` (`Engine`, `RuntimeFrames`) is the worker's end of that socket
  and the wire's framing; between nodes the rows ride libnccl-mesh
  (`../solver/worker.py`, xonotic/README.md "The solver").
- `../qcsrc/common/gamemodes/gamemode/payload/mesh_ipc.qh` declares the builtins used by game code.

Darkplaces includes `mesh_ipc.c` from `prvm_cmds.c`. Its wire constants come from
`rdma/xonwire.def`; there are no copied headers, copied C files, generated patches, or
patch-application scripts.

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
