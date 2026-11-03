# A7 page echo baseline, September 10

The measurement caller is `rdma/mesh-page-echo.c`, introduced in `abea3b5`.
Both machines checked out identical canonical commits for each measurement.
The numerical repository was `52c0a966d5c8be615a9bb2ed2b2b0a6aee1a12db` on both
machines; its callers built successfully but did not execute in these runs.

M5 used active `rdma_en2`; M4 used active `rdma_en3`. Both `ibv_devinfo -v`
reports showed `active_width: 8X (4)` and `active_speed: 10.0 Gbps (4)`.
This advertised product is not a measurement of application bandwidth.
The old M5 `rdma_en6` was down and was not the measured link.

Each managed bridge registered 278,937,600 bytes: 8192 arena pages and 8192
receive pages, with 16,384-byte pages. The 4096-page caller actually used
4151 arena pages and retained 4096 receive pages. No numerical kernels, operand
copies, checksums, or remote clock synchronization were used. Exact uint64
payload comparison occurred after the timed exchange. Every run on both nodes
reported zero mismatches, successful status and `detached=0`.

| Canonical commit | Pages | Initiator cohort time | Mean observed RTT | RTT/2 estimate | Returned payload rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `abea3b5` | 1 | 0.881 ms | 881 us | 440.5 us | 18.60 MB/s |
| `abea3b5` | 4096 | 19.820 ms | 10,211.282 us | 5,105.641 us | 3.386 GB/s |
| `abea3b5` | 1 | 0.034 ms | 34 us | 17 us | 481.88 MB/s |
| `85f9e5b` | 4096 | 107.670 ms | 59,576.157 us | 29,788.078 us | 0.623 GB/s |

4096 pages is 67,108,864 payload bytes in each direction. The returned-payload
rate counts that quantity once; aggregate link payload counts it twice. Neither
is a bare one-way link bandwidth measurement. Bulk page RTT includes queuing and
client observation delay. Two single-page samples are not a latency distribution.
The echo host's elapsed interval includes waiting for the initiator process and
is deliberately not used for throughput.

Both builds requested `CPPFLAGS=-DMESH_TRANSPORT_TIMING`. In `abea3b5`, timing
was printed only at shutdown, and no timing record was captured in the managed
logs. This absence is unresolved; those runs must not be labeled a verified
uninstrumented baseline. `85f9e5b` also publishes the instrumented aggregates
through the existing telemetry interval, allowing capture before managed stop.
The following snapshots include startup and idle processing as well as traffic:

| Node | Passes | Pass seconds | CQ polls | Poll seconds | Mean measured pass | Mean measured CQ poll |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M5 | 90,813,436 | 10.362143922 | 81,760,905 | 2.435268962 | 114.10 ns | 29.79 ns |
| M4 | 139,076,498 | 10.113519999 | 132,487,632 | 2.555371000 | 72.72 ns | 19.29 ns |

These are instrumented aggregate durations, including clock-read overhead,
dominated by idle polls. They are not loaded-path dispatch latency or evidence
that all bridge overhead is below a threshold. The large bulk-run variation
has not been attributed to instrumentation, scheduling, or another cause.
A1–A6 comparisons must use matching build flags and repeat both one-page and
bulk measurements; a warmed distribution and an uninstrumented repeated bulk
rate remain necessary before claiming a stable performance improvement.

## Reproduction

Build committed source on both machines with:

```
make -C rdma -B CPPFLAGS=-DMESH_TRANSPORT_TIMING mesh-flow mesh-page-echo
```

Use the existing `tools/mesh/participants.py:bridge_pair` lifecycle to start and
ready both bridges with `arena_pages=8192, receive_pages=8192`. Launch
`rdma/.build/mesh-page-echo echo 0 N 60` on M4; after its configuration-time
`page_echo_ready` output, launch `rdma/.build/mesh-page-echo send 1 N 60` on M5.
Read both outputs and bridge logs before ordinary managed stop. No timing
coordination enters the transfer path. N is 1 or 4096 for the records above.

Both bridges were stopped after measurement. Both machines were rebuilt with
`make -C rdma -B mesh-flow`, restoring the default build without measurement
clock reads. No bridge or numerical process was left running.
