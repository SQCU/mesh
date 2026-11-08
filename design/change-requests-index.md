# Change requests: which are outstanding

As of 2026-11-07. Each entry names a change request, whether it is outstanding, and what of it is open. The standard
every change request follows is [change-requests.md](change-requests.md).

## Outstanding

- [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md): outstanding.
  1. Each function typed and scheduled as plan-to-bridge R3 and R4 write it, on every Apple silicon backend, the CPU
     included (rewritten 2026-11-08). Open: the lowering to R4's tasks. The kernels of metal-microbench c24308dd and
     2e262f1f and mesh 014fe04a and 40e4189d lower every operator as one task over a whole operand.
  2. The capture is a static dataflow graph. Implemented in mesh b4a6e67d.
  3. Self-timed lowering of dependences into Metal orderings. Implemented in mesh 38e2b7e6 and 775f142c and
     metal-microbench e4800f65, 270c6e5f and 43f37e96. Open: each task's length from the weighting operand (A3), where
     the replay now orders by dependence depth.
  4. Withdrawn 2026-11-08: plan-to-bridge R4 writes this consumer, and item 1 lowers it.
- [torch-capture-change-request.md](torch-capture-change-request.md): outstanding in part.
  - Open, written and not yet implemented: item 6, its stop settled in mesh d62f205d from
    design/stops-sources-2026-11-07.md.
  - Open, written and not yet implemented: item 3 in the LiteRT scope, with a partition a page (plan-to-bridge R4).
  - Open, a stop: item 2's execution-time estimates t(v), on which its buffer bound depends. `HOST_LAG` stands until
    then.
  - In the torch scope, item 9 waits on the lowering request's item 4.
  - Implemented: items 1 (metal-microbench a3792f13), 4 (mesh 992f6389), 5 and 7, and item 2's first write. Item 8 is written by the lowering
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
  - Open: R4's consumption of a received operand one partition at a time. In the torch scope this is the lowering
    request's item 4. In the LiteRT scope it is the torch-capture request's item 3, written in d62f205d and not yet
    implemented.

## Implemented

- [aot-compilation-change-request.md](aot-compilation-change-request.md): A1–A10 are implemented, and its appendix's
  departures were addressed in commits of 2026-11-05. Appendix item 7, the torch path on torch's existing kernels, is
  superseded by the lowering request's item 1.

## Superseded or closed

- [disjoint-pages-change-request.md](disjoint-pages-change-request.md): D.2, D.3.1 and D.5 are superseded by the
  plan-to-bridge request; the rest is implemented.
- [page-queue-change-request.md](page-queue-change-request.md): superseded in part; the rest is implemented.
