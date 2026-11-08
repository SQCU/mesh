# Change requests: which are outstanding

As of 2026-11-07. Each entry names a change request, whether it is outstanding, and what of it is open. The standard
every change request follows is [change-requests.md](change-requests.md).

## Outstanding

- [torch-capture-change-request.md](torch-capture-change-request.md): outstanding in part.
  - Item 3 in the LiteRT scope: implemented in metal-microbench a8ed7bd6, with a partition a page. Open: holding a run
    of threadgroups for the supplied words of the pages it reads (plan-to-bridge R4), where a8ed7bd6 stops; and C9's
    finding on 2c802b93.
  - Open, a stop: item 2's execution-time estimates t(v), on which its buffer bound depends. `HOST_LAG` stands until
    then.
  - In the torch scope, item 9 waits on the lowering request's item 4.
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
  - Open: R4's consumption of a received operand one partition at a time. In the torch scope this is the lowering
    request's item 4. In the LiteRT scope it is the torch-capture request's item 3, written in d62f205d and not yet
    implemented.

## Implemented

- [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md): implemented; checked on this Mac
  only, since no mesh machine has answered.
  1. Each function typed and scheduled as plan-to-bridge R3 and R4 write it, on every Apple silicon backend, the CPU
     included (rewritten 2026-11-08). Implemented in mesh 502bf42c and 8cac6d69 and metal-microbench 3bc0e2a4: the IR's
     kinds, one task per page for the asynchronous consumer, the same source compiled for the CPU, and each capture
     weighed by A3's task weights (mesh 14b2061d). Reductions, softmax and attention as R4's homomorphisms, one task
     per page combined by their sources' ⊕ (design/homomorphism-sources-2026-11-08.md), and a move between backends
     split into functions ordered by an event: mesh 2abc7125, metal-microbench 71a23822. The all-reduce root's and
     reduce-scatter's sum of received partials, one task per received page: mesh 3e176a9d.
  2. The capture is a static dataflow graph. Implemented in mesh b4a6e67d.
  3. Self-timed lowering of dependences into Metal orderings. Implemented in mesh 38e2b7e6 and 775f142c and
     metal-microbench e4800f65, 270c6e5f and 43f37e96; the order from the completion-time linear program with each
     task's length from the weighting operand, in metal-microbench 3b719f10 and mesh 14b2061d.
  4. Withdrawn 2026-11-08: plan-to-bridge R4 writes this consumer, and item 1 lowers it.
- [aot-compilation-change-request.md](aot-compilation-change-request.md): A1–A10 are implemented, and its appendix's
  departures were addressed in commits of 2026-11-05. Appendix item 7, the torch path on torch's existing kernels, is
  superseded by the lowering request's item 1.

## Superseded or closed

- [disjoint-pages-change-request.md](disjoint-pages-change-request.md): D.2, D.3.1 and D.5 are superseded by the
  plan-to-bridge request; the rest is implemented.
- [page-queue-change-request.md](page-queue-change-request.md): superseded in part; the rest is implemented.
