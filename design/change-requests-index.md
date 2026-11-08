# Change requests: which are outstanding

As of 2026-11-07. Each entry names a change request, whether it is outstanding, and what of it is open. The standard
every change request follows is [change-requests.md](change-requests.md).

## Outstanding

- [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md): outstanding, all four
  requirements.
  1. No torch kernel; every function from source.
  2. The capture is a static dataflow graph.
  3. Self-timed lowering of dependences into Metal orderings.
  4. Consumption of received data inside functions.
- [torch-capture-change-request.md](torch-capture-change-request.md): outstanding in part.
  - Open: the stops of items 1, 2, 3 (in the LiteRT scope), 4 and 6. In the torch scope, item 9 waits on the lowering
    request's item 4.
  - Written: items 2, 5 and 7, and item 1's copies and fills. Item 8 is written by the lowering request's items 1–3.
- [problems-of-14b9a757-change-request.md](problems-of-14b9a757-change-request.md): outstanding in part.
  - Open: the stops on problems 4, 7, 8 and 11, in its three sections of sources for the stops.
- [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md): outstanding in part.
  - Open: R4's consumption of a received operand one partition at a time. In the torch scope this is the lowering
    request's item 4. In the LiteRT scope it is the torch-capture request's item 3, a stop.

## Implemented

- [aot-compilation-change-request.md](aot-compilation-change-request.md): A1–A10 are implemented, and its appendix's
  departures were addressed in commits of 2026-11-05. Appendix item 7, the torch path on torch's existing kernels, is
  superseded by the lowering request's item 1.

## Superseded or closed

- [disjoint-pages-change-request.md](disjoint-pages-change-request.md): D.2, D.3.1 and D.5 are superseded by the
  plan-to-bridge request; the rest is implemented.
- [page-queue-change-request.md](page-queue-change-request.md): superseded in part; the rest is implemented.
