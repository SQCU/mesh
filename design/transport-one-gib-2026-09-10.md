# One GiB transport comparison, September 10

Each transfer below delivered 1,073,741,824 bytes (1 GiB), one way from M5 to
M4, as 65,536 16-KiB pages. These are receiver-clock delivery measurements,
not echo RTT, process-start timing, or small-message extrapolations.

## Accepted comparison

Both machines used canonical `bdc1a52` with default `-O2` builds and no
`MESH_TRANSPORT_TIMING` instrumentation. M5 `rdma_en2` and M4 `rdma_en3`
were active; both advertised width 8X and speed 10.0 Gbps. The numerical
repository remained `52c0a966d5c8be615a9bb2ed2b2b0a6aee1a12db`; no numerical
kernel ran. Shared provider realization came from the same `mesh-verbs.h`.

| Path/run | Receiver first-to-last | Interior observed bytes | Interior time | Interior GB/s |
| --- | ---: | ---: | ---: | ---: |
| Direct verbs 1 | 114.414 ms | 858,996,736 | 91.546 ms | 9.383225 |
| Direct verbs 2 | 114.421 ms | 858,996,736 | 91.549 ms | 9.382918 |
| Mesh, first after bridge start | 143.012 ms | 858,865,664 | 115.566 ms | 7.431820 |
| Mesh, next full transfer | 115.053 ms | 858,996,736 | 91.989 ms | 9.338038 |
| Mesh, next full transfer | 115.084 ms | 858,931,200 | 92.064 ms | 9.329718 |

All five completed all 65,536 pages, with every payload word correct. Both
direct endpoints also passed completion-ID order checks. Every process exited
successfully through normal teardown; both bridges were stopped afterward.
Two repeat samples per path establish this comparison, not a tail-latency
distribution or a guarantee about every future run.

The interior is approximately the 10–90% delivery interval. Numerators use the
actual observed page counts at threshold crossings, avoiding completion-batch
rounding errors. No setup delay or remote clock comparison enters that rate.
The first-to-last numerator excludes the pages already present at the first
observation: 1 page for each direct run and the first mesh run, 5 and 4 pages
for the two following mesh runs. Every run nevertheless transfers the full GiB.

The sender publishes already-realized source pages in 101, 109 and 110 us in
the three mesh runs. This interval is separately reported and is not folded
into receiver interior rate. Received values remain available for exact
verification after timing. The receiver uses an advancing observation cursor;
it does not rescan already-observed pages. Canonical mesh's own binding and
retirement scans are still present and are part of the path being measured.

## What can be attributed

The mean repeat rates are 9.383071 GB/s direct and 9.333878 GB/s mesh: a
0.5243% payload-rate deficit. Normalizing those measured rates to one GiB gives
114.434 ms versus 115.037 ms, or about **0.603 ms additional mesh time**.
These normalized times are calculations from the interior rates, not additional
measured end-to-end latencies.

The current mesh encoding necessarily adds 4,194,304 header bytes to each GiB
of payload: 64 bytes for each 16,384-byte page. That is 0.390625% extra bytes.
At the measured direct rate, those bytes alone represent about **0.447 ms** of
additional service. Subtracting that estimate leaves about **0.156 ms/GiB**;
this residual is not an isolated measurement of rings, polling, or any other
individual function. Noise, additional framing and software costs also belong
in that residual. The experiment does not establish an exact additive timing
decomposition of overlapping work.

Static source analysis identifies these removable costs:

- Two SEND WRs and two completions per payload page, versus one in the direct
  path: 131,072 versus 65,536 over a GiB.
- Five frame credits per payload page versus four: 327,680 versus 262,144
  cumulative credits. At the configured 4095-frame capacity this admits 819
  mesh page pairs versus 1023 direct pages, not five padded pages on the wire.
- SUB/CMP/ACK handoffs, repeated binding/retirement scans and page zeroing in
  the current mesh path. These consume work, but their costs overlap transport;
  the measured repeat gap does not support assigning each a large serial delay.

The first mesh transfer remains slower even with setup excluded: 7.432 GB/s,
equivalent to about **30.0 ms/GiB more service time** than direct. Later complete
GiB transfers remove most of that difference. This establishes first-use
sensitivity; it does not yet distinguish memory preparation, CPU scheduling,
or another cause. It must not be called irreducible link latency or hidden by
averaging it into the repeated transfers.

At a nominal 80 Gbit/s, payload serialization alone requires at least
107.374 ms per GiB. Real framing and the achieved provider rate add cost. The
direct measurement establishes an achievable operating point, not proof that
all 114.434 ms is physically irreducible. This experiment therefore separates
the payload serialization floor, the observed avoidable mesh gap, and the
unattributed remainder rather than inventing a precise hardware/software split.

## Benchmark bugs corrected before acceptance

The first direct implementation posted receives only after both endpoints had
completed QP setup. It reported 65,536 send completions but only 63,488 receive
completions. Read-only inspection of the retained receiver found source page
2048 in receive page 0, and source page 4096 in receive page 2048. All payload
words failed the expected-location check. Those runs are rejected evidence.

Posting in QP INIT fixed full-byte delivery but produced wrong completion IDs
after subsequent QP transitions. `bdc1a52` fixes the benchmark's configuration
ordering: receiver reads the sender's existing QPI record, completes RTR and
RTS, posts its initial registered receive descriptors, then sends its existing
QPI record. The sender cannot finish its existing setup exchange before that
point. There is no additional message, echo, acknowledgement, sleep, clock
coordination or wait in the stream. Both payload and completion identities
then pass. Broken TCP setup also now ignores SIGPIPE so normal teardown owns
the device; unreachable hostname-selected setup addresses were replaced by the
numeric LAN address in the measurement command.

## Reproduction and allocation

Build on both machines from the same committed source:

```
make -C rdma mesh-flow mesh-page-stream mesh-verbs-stream
```

With managed bridges stopped, start the direct sender on M5 and receiver on M4:

```
rdma/.build/mesh-verbs-stream send ms-mac-mini.local rdma_en2 30
rdma/.build/mesh-verbs-stream receive 10.0.0.5 rdma_en3 30
```

The receiver's numeric address is only for QP metadata exchange. All payload
uses the named RDMA interfaces. After both JSON results, send ordinary TERM to
the exact reported PIDs and collect their exit statuses. Endpoints retain
registered storage until that outer teardown; no peer teardown signal is part
of the data path. Each direct endpoint allocates and registers its one GiB
before timing, with a continuously replenished 1023-page descriptor window.

For mesh, use the existing `participants.py:bridge_pair` lifecycle with:

| Node | Arena pages | Receive pages | Registered bytes |
| --- | ---: | ---: | ---: |
| M5 | 70,000 | 2,048 | 1,198,063,616 |
| M4 | 1,024 | 70,000 | 1,181,155,328 |

The sender actually uses 66,213 arena pages; the receiver uses 122 arena pages
and retains 65,536 receive pages. These bounded allocations follow the transfer
shape, not a fraction of either machine's RAM. Start and ready both bridges,
then launch M4's receiver and observe its configuration-ready output before
launching M5's sender:

```
rdma/.build/mesh-page-stream receive 0 120
rdma/.build/mesh-page-stream send 1 120
```

Repeat the full transfer without restarting the bridges to retain the explicit
first-use/repeat distinction. After collecting successful detach statuses,
stop both bridges through the existing managed lifecycle. No verbs owners or
their registered allocations were left running after this measurement.

## Backing-memory separation measurement

Canonical `4d1892c58a025604c6e35bc9bd89896f02cfe49a` and caller
`cc3a4ae91be1da085aa622c81e1c2c6f8f6f3494` were synchronized and built on both
machines. ABI 10 removes arena context rows, release-time zeroing, retirement
stamp mutation in the bridge, and arena retirement FIFO traffic. The existing
separate metadata/operand transfers remain. Three full 1 GiB transfers used
the same configured capacities as the earlier comparison:

| Transfer | Receiver first-to-last | Interior bytes | Interior seconds | GB/s |
| --- | ---: | ---: | ---: | ---: |
| First after bridge start | 114.985 ms | 859,422,720 | 0.092065 | 9.334956 |
| Repeat 1 | 114.956 ms | 858,963,968 | 0.091955 | 9.341134 |
| Repeat 2 | 114.896 ms | 859,111,424 | 0.091949 | 9.343347 |

Every transfer delivered all 65,536 backing extents and every payload word
matched. Both endpoints returned status 0 and detach 0. Registered sizes were
1,189,068,800 bytes on M5 and 1,179,910,144 bytes on M4. Sender realization used
66,085 arena pages rather than 66,213 because outbound records are smaller.
Bridges stopped normally afterward. No direct-verbs rerun was made; the previous
direct measurement is historical context, not a new matched comparison.
First-use slowdown was absent in these three samples, without establishing
causality or a general latency guarantee.

The interrupted ABI-9 attempt is excluded: its receiver timed out before the
sender started, so sender completions did not establish consumed receiver data.

The existing real-input FFN evaluator then ran with canonical `4d1892c` and
caller configuration commit `c1954f9`, 4,096 rows, six measured samples and one
and two invocations in flight. Both peers completed ten invocations per case,
reported no errors/nonfinite values, and destroyed their contexts. Relative RMS
against the local baseline was 0.0003935084 (limit 0.002), with exact peer
agreement in both cases. M5/M4 medians were 34.5306/34.5999 ms for one in flight
and 54.6755/54.0809 ms for two. The local MPS baseline was 32.37475 ms, so the
required TP latency improvement is still absent. The M5 two-in-flight maximum
was 111.4771 ms; medians must not conceal that sample. Native internal storage
remains unverified. Both bridges were stopped after evaluation.

## Removal of the local SEND acknowledgement

Canonical `d4d8d72c6a4bea2d26a8978d2a353ebf94037f4f`, built on both machines,
removes the ACK ring, client ACK drain, ACK submission-capacity gate and
`arena_pending` counter. Configuration stores a region-relative source-row
offset in the existing outbound record. The dataflow completion operation
releases the source NIC use directly after unlinking the completed transfer.
Each region loses 2,097,152 bytes of ring storage. The same change removes
setup ping probes and artificial pending returns after successful teardown.

Three complete 1 GiB streams on ABI 11 each delivered all 65,536 extents with
zero payload mismatches, status 0 and detach 0 at both endpoints:

| Transfer | Receiver first-to-last | Interior bytes | Interior seconds | GB/s |
| --- | ---: | ---: | ---: | ---: |
| First | 114.961 ms | 858,996,736 | 0.091956 | 9.341389 |
| Repeat 1 | 114.954 ms | 858,963,968 | 0.091918 | 9.344894 |
| Repeat 2 | 114.903 ms | 859,095,040 | 0.091921 | 9.346015 |

The existing 4,096-row FFN evaluator used numerical configuration commit
`8b49749fab8ca758a831b5c85036e83388c12a41`. One and two invocations in flight
each completed ten invocations on both peers, with no reported errors or
nonfinite values and successful context destruction. Both peers agreed exactly;
relative RMS against the local output was 0.0003935084, within the 0.002 limit.
These checks cover actual source reuse after deletion of the client ACK drain.
Both bridges were stopped normally after measurement. Receive CMP publication
and the existing worker lifecycle remain; this result does not establish their
removal or compliance.
