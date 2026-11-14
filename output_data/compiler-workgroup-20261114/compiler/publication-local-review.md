# Publication and joint-local result checkpoint

The source checkpoint retains one joint local result across actual function instances, original call relation/closure, shared storage roots and all-copy offsets. Local save/load preserves those native arrays and their byte owner without re-solving. Python `analyse`, `declare_program` and `solve_program` retain original call/slot/root correspondence; the current network branch is still unfinished.

Independent publications now reserve page-table rows and arena extents around live reservations through the existing greedy allocator. A stable root-row generation publishes each program; retirement closes only that generation before the existing hazard/held reclamation check. Discovery happens on preparation/publication changes. No per-call allocation or handle walk was added. The bridge owns each loaded mapping separately; escaped arena backing retains it, and failure cleanup retires only adopted results.

The diagnostics copy only the bounded holder fields, then validate the same open row generation after an acquire fence. `mesh-stat` emits its client list in one scan with no intermediate array or allocation, avoiding the previous count/fill race. The heartbeat uses the same protected snapshot and the existing holder-liveness authority.

Reviewed boundaries: publication→hazard announcement→same-row reread→held ownership; retirement→closed generation→hazard/held discharge→row release; reserved extents→load relocation; loaded byte owner→native result views; actual binding slot→source storage root→joint local copies. Original deliberately malformed files were not repaired, moved or used as imports.

Evidence is bounded:

- Canonical `make -C rdma libmesh.dylib` completed before the final diagnostic-only correction; log `libmesh-publication-joint-build.log`, source hashes `libmesh-publication-joint-source.sha256`. Fresh live imports observed Function168, Planned152 and Local288. Built libmesh SHA256: `1ab463d3ab4cc88015a2a04261fd91fc1bd71c2c197923a7c12703794983757b`.
- The local worker's current-source six-call/three-function observation placed shared roots in 176 bytes and round-tripped the full 4792-byte local archive. This was a private planner observation, not a native machine-entry execution.
- Publication/holder/header consumers passed their canonical strict private syntax checks. The final one-pass mesh-stat correction passed its rule and `-Wall -Wextra -Werror`; `git diff --check` passed.
- The actual frontend mutation-stage observation against this libmesh preserved the old-read→write edge and returned alias layout (`frontends/torch/observe/mutation-stages-final.stdout`, frontend941db09b).

Still open: network shared-root/port planning and original source-family observer projection; native joint callable/archive integration; canonical whole-network result load into Published; actual concurrent live-publication execution. Payload bytes and wire backing extent must remain distinct in the next port handoff. This checkpoint does not claim those consumers or general numerical coverage.
