# Change requests: which are outstanding

As of 2026-11-08, after mesh e6eda244 and metal-microbench 03298a79. Each entry names a change request, whether it is outstanding, and what of it is open. The standard
every change request follows is [change-requests.md](change-requests.md).

## Outstanding

- [integration-finishing-change-request.md](integration-finishing-change-request.md): outstanding in part.
  - Item 1: every bridge is configured as the repository installs it and runs. The job io.mesh.bridge was disabled on
    all five machines and is enabled; the mini's stale /usr/local/mesh/bridge.conf (2026-09-17) is removed; each
    machine's ~/.mesh-bridge.conf states region=/mesh0, its arena (82 GiB on the Ultras, 16 GiB on the mini, as the
    deleted etc/bridge profiles set them) and mesh_block_pages=1, and no window, so the window is mesh-bridge.sh's
    default, a bank's pages; the mini's configuration.plist is rewritten by the same start. The window was refused on
    every port at the provider's advertised max_mr_size; device_up now registers it in 1 GiB regions as M09 states
    (mesh b59c579a, e6eda244), four a port, and all five maps list all five nodes. The five-rank E2B decode
    (metal-microbench tools/mesh/placements/even-5.json, 03298a79) is prepared on every node and planned on every rank
    (mesh_plan needs 199,500 wired pages, which the window holds). Open, a stop: every rank stops at the prefill,
    "prefill: Operation timed out: no feedback of its commits" (output_data/e2b-five-2026-11-08).
  - Items 2-6: implemented. Item 2 in metal-microbench c0a067fc and 6b8f7c3c with mesh b482a825; item 3 in mesh
    01632014; item 4 in mesh e34e7a5e and metal-microbench c0a067fc, which C9 passes for MetalMaskedPass; item 5 in
    metal-microbench 5d5efc09; item 6 here.
- [torch-capture-change-request.md](torch-capture-change-request.md): outstanding in part.
  - Item 3 in the LiteRT scope: implemented in metal-microbench 5600d84b, 66a18fab, c0a067fc and 6b8f7c3c, and mesh
    56e33484, e34e7a5e and b482a825. A received operand's reader is masked, the threadgroups whose pages have not
    landed are compacted into a worklist and run by a later pass's indirect dispatch, and the later passes are placed
    by the heap order over the commands' dependence levels and their ranges' published completion times. Checked on
    this Mac by the scratch mask harness; the E2B check on the five machines stops at the prefill (above).
  - Open, a stop: item 2's execution-time estimates t(v), on which its buffer bound depends. `HOST_LAG` stands until
    then.
  - In the torch scope, item 9 waits on the lowering request's item 4, which is withdrawn.
  - Implemented: items 1 (metal-microbench a3792f13), 4 (mesh 992f6389), 5, 6 (mesh a2c1532c and b880b808,
    metal-microbench 6b3c29e5, completed in mesh 356bed56) and 7, and item 2's first write. Item 8 is written by the lowering
    request's items 1–3.
- [problems-of-14b9a757-change-request.md](problems-of-14b9a757-change-request.md): outstanding in part.
  - Implemented: the requirements written from its stops in mesh 593dd1c0. These are the second stops' item 8 and the
    third stops' "loading only the slices a node reads", by Megatron-LM's `_initialize_affine_weight_cpu`, and the
    third stops' collective over a view of an invariant, by GSPMD §3.1 and §3.5. They are in mesh 2155c2b8 and
    60f43ddb, completed in 76d26e3b.
  - Settled: the second stops' u copies, by its appendix item 14. Superseded: the first stops' item 8.
  - Problems 4 and 11: their sections of sources say what is written. The request records no commit implementing
    them.
- [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md): outstanding in part.
  - R4's consumption of a received operand one partition at a time: implemented, in the torch scope by the lowering
    request's item 1 and in the LiteRT scope by the torch-capture request's item 3.

## Implemented

- [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md): implemented; checked on this Mac,
  and built on the five mesh machines.
  1. Each function typed and scheduled as plan-to-bridge R3 and R4 write it, on every Apple silicon backend, the CPU
     included (rewritten 2026-11-08). Implemented in mesh 502bf42c and 8cac6d69 and metal-microbench 3bc0e2a4: the IR's
     kinds, one task per page for the asynchronous consumer, the same source compiled for the CPU, and A3's weighting
     operand, a weight on each of mesh_plan's tasks and on each edge (mesh 01632014). Reductions, softmax and attention
     as R4's homomorphisms, one task per page combined by their sources' ⊕ (design/homomorphism-sources-2026-11-08.md),
     and a move between backends split into functions ordered by an event: mesh 2abc7125, metal-microbench 71a23822.
     The all-reduce root's and reduce-scatter's sum of received partials, one task per received page, in the order of
     their published completion times: mesh 3e176a9d, a582ce3c and 91786869.
  2. The capture is a static dataflow graph. Implemented in mesh b4a6e67d.
  3. Self-timed lowering of dependences into Metal orderings. Implemented in mesh 38e2b7e6 and 775f142c and
     metal-microbench e4800f65, 270c6e5f and 43f37e96. The order is plan_schedule's: its completion times are
     published with the plan (mesh a582ce3c), the second linear program over the GPU's commands is deleted
     (metal-microbench 0a910421), and the replay's list is the heap order over those keys (integration-finishing
     item 2).
  4. Withdrawn 2026-11-08: plan-to-bridge R4 writes this consumer, and item 1 lowers it. Its code is deleted
     (metal-microbench 5d5efc09).
- [aot-compilation-change-request.md](aot-compilation-change-request.md): A1–A10 are implemented, A3's operand as
  integration-finishing item 3 writes it (mesh 01632014), and its appendix's departures were addressed in commits of
  2026-11-05. Appendix item 7, the torch path on torch's existing kernels, is
  superseded by the lowering request's item 1.

## Superseded or closed

- [disjoint-pages-change-request.md](disjoint-pages-change-request.md): D.2, D.3.1 and D.5 are superseded by the
  plan-to-bridge request; the rest is implemented.
- [page-queue-change-request.md](page-queue-change-request.md): superseded in part; the rest is implemented.
