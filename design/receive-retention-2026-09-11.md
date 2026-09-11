# Receive retention qualification, September 11

Implementation: `8c047bf`, built from matching committed main on M5 Max and
M4 Pro. The caller build also passed on both. Mechanism and API contract:
[receive storage before consumer binding](algorithm-sources.md#receive-storage-before-consumer-binding).

## Actual numerical run

The existing 4096-row, 20-invocation FFN runner used caller `1007ea8` and canonical
`8c047bf`, with the same configuration and real input as the preceding run.
Both nodes received 32 blocks, with zero bad blocks. Read-only inspection of the
landed bitmap, inverse rows and binding table confirmed association on both:
binding 1 began at logical row 21635 on M5 and 5795 on M4. Previously M4 discarded
all 32 successful driver completions for absent binding. That discard is gone.

The numerical invocation still did not finish. The caller gives reduction
occurrences overlapping whole-residual dependencies and a shared READ plane;
completion of an early occurrence makes later occurrences unavailable. This is
separate caller work, not a transport qualification result. Both clients were
terminated normally and their exit observed. Logs and manifest are under
`metal-microbench/.build/nfe-configured/f10-8c047bf-1*` and
`metal-microbench/.build/f10-8c047bf.log`.

## Existing one-GiB page-stream measurement

Both machines built `rdma/.build/mesh-page-stream` with the existing Makefile
target. No new evaluator was introduced. Transport geometry was 61 pages per
block, 16384 bytes per page, with 60 payload pages and existing tag storage.
The tool rounds its one-GiB budget down to 1092 complete blocks: 1,073,479,680
actual payload bytes, or 1,091,371,008 bytes including tag pages.

M5 had 66612 arena pages and 1952 receive pages. M4 had 61 arena pages and 66612
receive pages. Normal bridge startup registered 1,126,350,848 and 1,095,368,704
bytes respectively. All allocations were explicit; no percentage allocation.

First, M5 ran `rdma/.build/mesh-page-stream send 1 60` to completion with no M4
consumer application. M4's bridge reported client 0, 1092 received blocks and
zero bad blocks. Only then did M4 run
`rdma/.build/mesh-page-stream receive 0 60`. It bound and read all original
arrivals with zero mismatched words. Both commands exited zero. This establishes
arrival before binding and retention across sender exit without a retransmission.

Second, M4 launched the receive command before M5 launched the same send command.
Both completed 1092 blocks with zero mismatched words and zero exit status.
The receiver measured 858,193,920 interior payload bytes over 0.092853 seconds:
9.2425 GB/s. The sender measured the same interior payload over 0.092846 seconds:
9.2432 GB/s. Full first-to-last block spans were about 116 ms.

The late-binding receiver's reported roughly 0.2 ms span measures association
and observation of already-delivered memory. It is not link bandwidth and must
not be reported as such. These runs qualify storage/transport, not numerical MFU
or FFN speedup. Evidence files are `metal-microbench/.build/f10-late-bind-*` and
`metal-microbench/.build/f10-bound-*`.

## Explicit invalidation

The calling context used the built library directly: `mesh_attach`,
`mesh_receive_invalidate`, then `mesh_detach`, all on M4. Both status returns
were zero. No numerical function invoked invalidation.

After invalidating the posted receive window, M5 sent another 1092 blocks with
the same existing stream command. All sends completed. Read-only table inspection
found 1076 retained landed blocks, plus 16 newly posted HOT blocks. The original
16 invalidated posted receives had completed without restoring ownership or
becoming retained arrivals. A second invalidation changed the retained landed
count to zero and the owned block count to zero, leaving 16 HOT posted accesses
for the driver to complete. Neither operation pushed FREE from the client.

The invalidation calls returned without waiting. Their effects were inspected
after subsequent bridge progress; this is not instantaneous cancellation of
already issued numerical work or future remote transmissions. Finally both
bridges stopped through normal teardown, releasing their registered regions.

Source review independently covered association concurrent with detach:
ROW_BOUND protects captured bindings until the bridge has established
ROW_LANDED, before end-of-pass binding reclamation. Invalidated unassociated
blocks never enter logical-row reclamation. Consumer detach preserves their
physical ownership; only explicit invalidation discards them.
