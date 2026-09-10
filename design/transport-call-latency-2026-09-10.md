# Provider call costs and message latency, September 10

Direct verbs stream (`rdma/mesh-verbs-stream`, `CPPFLAGS=-DMESH_TRANSPORT_TIMING`),
1 GiB as 65,536 SENDs of 16,384 bytes, M5 (`rdma_en2`) to M4 (`rdma_en3`),
canonical `049bfb5`, no bridges running. Times are the receiver's or sender's
own monotonic clock; per-call means are aggregate instrumented seconds divided
by call counts and include the clock reads.

| Window | First-to-last | Per page | Rate | `ibv_post_send` | `ibv_post_recv` | `ibv_poll_cq` (95% empty) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1023 | 114.07 ms | 1.741 us | 9.41 GB/s | 398 ns (1.02 WR/call) | 668 ns | 44 ns / 39 ns |
| 4 | 115.25 ms | 1.759 us | 9.32 GB/s | 435 ns | 711 ns | 45 ns / 36 ns |
| 1 | 252.85 ms | 3.858 us | 4.25 GB/s | 379 ns | 664 ns | 29 ns / 28 ns |

Window 1 keeps exactly one message in flight, so 3.858 us is the sender's
post-to-completion-to-next-post period for one 16 KiB SEND: about 2.1 us of
provider and hardware turnaround beyond the 1.74 us the page occupies the wire.
Three or more messages in flight hide it entirely (window 4 is within 1% of
window 1023). At window 1023 the sender's thread spends 25.7 ms posting and
60.1 ms polling out of 114.4 ms; none of that is on the transfer's critical
path. All payload words verified at the receiver in every run.

Consequences for the block transport (61 pages, 999,424 bytes, 244 frames):
about 106 us of wire per block, 16 blocks in flight per QP, so the 2 us
turnaround is under 2% even unpipelined and 0 when pipelined.

Page-table cost on the same day (`scratchpad/table_bench.c`, M5, no bridge):
an FFN generation's table work — 32 blocks landed, two claims of 976 rows,
16 reduce occurrences issued and completed, hold consumed, 32 blocks freed —
is 8.2 us; a not-ready issue scan of 16 occurrences is 32 ns; landing one
block in the bridge is 17 ns; a send-completion OR is 3 ns.
