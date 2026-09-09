# Transport guarantees belong at their supplying layer

## Indexed retention under queue pressure, September 8, 2026

The previous bridge acknowledged a SUB page when `link_submit` could not accept
it, released a forwarding page on the same condition, and released received
application data when CMP approached capacity. The 4,096-row Gemma FFN run
exposed a permanent receive deficit with fully published local outputs and no
remaining client send work. Queue pressure must retain ownership; it is not a
transport completion or an instruction to discard the operand.

SUB and each link's completion ring now retain their existing descriptors until
the next destination accepts them. Their single consumer selects by numeric
index. For occupied indices `[t,h)` and accepted index `i`, removal is:

```
entry[i] = entry[t]    (i != t)
t' = t + 1
multiset(entry[t':h]) = multiset(entry[t:h]) - {old_entry[i]}
```

Both entries belong to the consumer's occupied interval. The copy precedes the
release store of the new tail. The producer acquires that tail before reusing
the freed slot. Thus the existing ring supplies both storage and ownership;
there is no additional pending queue or numerical payload copy. The usual case
`i == t` performs no descriptor copy. A rotating index revisits retained work
while allowing other peers and send completions to pass it. Ordering is not an
implicit operand index.

A successful receive changes its existing completion kind to `L_READY` and its
existing page-owner byte from `RECV` to `READY`. The posted-receive count is
decremented exactly once. A blocked delivery retains that descriptor and page.
Acceptance transfers the page to `APP` or `SEND`; its corresponding release or
send completion returns it to `FREE`. `READY` is an internal fifth state, not a
change to the shared allocation split or ABI. Telemetry includes it with held
application pages. Link retirement releases unfinished `RECV`/`SEND` pages but
does not recycle completed `READY` bytes that await another destination.

Client death discards that client's unsubmitted SUB entries and drains its
submitted arena pages through the existing send completions. It no longer resets
every link. Registered storage belongs to the surviving bridge, so a vanished
client does not invalidate the DMA source mapping or the work being forwarded
for other peers. Hardware link failure still follows the existing retirement
path.

Receive posting accounts for occupied completion entries and outstanding receive
requests, leaving QD send-completion entries and the worker's existing 128-entry
emission headroom. Retained receives therefore cannot fill the completion ring
and prevent the send completions needed to drain it. Send/receive command counts
are each bounded by QD; their sum is below LINK_QUEUE. The command producer is
the bridge alone, so an available command reservation cannot be consumed by
another producer before a terminal reply is encoded and submitted.

All of these changes are in canonical mesh. Numerical clients still publish
indexed page ranges and consume indexed arrivals through the existing API.
This removes software loss under temporary queue pressure; it does not turn a
UC send completion into proof of remote receipt after an actual link failure.

Mesh trusts successful receive work completions for transport-valid bytes. It
does not add a payload checksum. CPU/GPU ownership and numerical correctness
remain distinct responsibilities.

## Evidence and selected boundary

Apple's [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
specifies UC send/receive operations, failure reporting in completion status,
messages up to 16,773,120 bytes and a 4095-frame queue limit. Send completion does
not establish successful remote receipt. Posts are nonblocking and make no
syscall. Message and receive-buffer frame counts must agree.

The upstream [Thunderbolt/USB4 driver](https://github.com/torvalds/linux/blob/master/drivers/net/thunderbolt/main.c)
consumes the hardware `RING_DESC_CRC_ERROR` flag in `tbnet_check_frame`, without
rescanning payloads. This demonstrates the controller reporting mechanism, not
an audit of Apple's proprietary driver. Apple's
[JACCL receive implementation](https://github.com/ml-explore/mlx/blob/main/mlx/distributed/jaccl/lib/jaccl/mesh_impl.h)
also transfers completed buffers without an application payload checksum.
Mesh retains its explicit `IBV_WC_SUCCESS` check before emitting a receive event.

This is an API trust boundary, not a claim that short numerical runs prove every
physical bit error is detected. An overwrite after completion is an ownership
defect; a wrong numerical result is a computation defect. A second transport
checksum is not the implementation of either responsibility.

## Removed and retained work

The former format computed CRC32C at the sender, at the destination bridge merely
to classify the frame, and at the receiving client. Both software implementations,
the checksum field, extra header rewrite and checksum-specific test were deleted.
Header encode/decode now access only fixed-size metadata. Routing relays already
forward canonical pages without touching numerical payloads.

| Responsibility | Owner |
|---|---|
| Physical-frame integrity and completion error status | Controller/driver; mesh checks completion status |
| DMA lifetime and registered memory | Existing per-port verbs owner and canonical arena |
| Invocation, plan, endpoints and extent identity | Mesh metadata; hardware cannot identify numerical programs |
| Missing UC delivery, route changes and software relay drops | Endpoint FIN/REQ repair and duplicate suppression |
| Remote receipt and cancellation acknowledgement | Endpoint protocol; local send completion is insufficient |
| Publication, contributor uniqueness and reduction association | Native completion and reduction primitives |
| Permanent numerical-state loss | Caller checkpoint and fresh invocation |

The fault evaluator now loses a delivery despite local send completion, instead
of flipping application memory after a simulated successful receive. Stale epochs,
missing terminal receipts, plan mismatch, borrowed storage and numerical reductions
through Y/cyclic lesions remain covered.

The scoped wire marker advances from `0x4d580000` to `0x4d590000`. Both scoped
clients and bridge terminal responders need matching implementations. The old
marker is not accepted as a new scoped frame; no legacy checksum implementation
is retained. Legacy unscoped framing is unchanged. The internal C frame remains
40 bytes because of alignment; the deletion removes payload work, not eight wire
bytes.

## Next broad simplification

One verbs work request per 4-KiB page is a mesh implementation choice. The API
supports multi-frame messages. Realization can choose matching receive/send
extents per edge and tile shape while retaining canonical ownership until
completion. This is the next submission/layout intervention; it needs neither
another allocator nor a software model of hardware CRC.

Deleting repeated payload scans removes an O(bytes) CPU cost per numerical
transfer. It does not change mesh diameter or establish an infinite-cluster
failure rate. Degree-three routing, finite outstanding work and measured
capability still bound useful recruitment.

## Evaluation

The [recorded comparison](../measurements/transport-boundary-20260906.json) uses
the existing 128 × 64-KiB, four-stage resident GPU reduction on the physical
M5 Max/M4 Pro pair. Each cohort keeps its clients and Metal kernels resident for
ten invocations and omits the first invocation from the timing summary.

| Implementation | Median of the greater peer-local duration |
|---|---:|
| Hardware-accelerated software CRC | 86.242 ms |
| No software CRC | 86.843 ms |
| No software CRC, repeated cohort | 86.944 ms |

The checksum deletion demonstrates no throughput gain in this workload. All
20 invocations without it produced exact final values on both machines. The
existing ASan/UBSan and ThreadSanitizer evaluations passed, including the revised
delivery-loss case. Maintained source shrank by 29 lines, and all three redundant
payload scans disappeared. Both bridges remained at their configured 32-GiB and
6-GiB registered capacities and finished paired with zero ownership errors.
