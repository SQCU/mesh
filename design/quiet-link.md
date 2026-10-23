# A quiet link

This note goes with design/data-model.md, section 3. It says what a link's first SEND after a quiet stretch costs,
where mesh runs meet that cost, and how large a transfer must be before the cost is lost in it. Each number names its
source. The files measurement.md and the folders output_data/ and tools/ are in the metal-microbench repo.

## What was measured

A Thunderbolt controller that has sent nothing for a while is slow on its next frame. On the Mac mini, after 40-50 µs
with nothing sent, a SEND took 6.2 µs to complete. With a frame every 5 µs it took 1.9 µs (measurement.md:3165-3178,
2026-10-17). Since version 133 the bridge sends no keepalive, by the operator's ruling of 2026-10-22
(design/HETEROGENEOUS-LINKS.md section 10). So a link's first SEND after a quiet stretch pays this cost.

tools/mesh/hourly/quiet.c times it through the bridge's interface (measurement.md:3451-3467; output_data/quiet-20261023,
mesh 3d849a16, 2026-10-23 20:58Z). Each trip sends one 16 KiB page and gets one back. Before each trip the sender waits
0, 200, 2,000 or 5,000 µs, taken in turn trip by trip, about 198 trips a gap. "SEND" is the time from the page marked
ready to its SEND done. "Trip" runs until the reply lands. A gap of 0 is the warm case. Medians, µs:

| Link | SEND, warm | SEND, after 200-5,000 µs quiet | Trip, warm | Trip, after 200-5,000 µs quiet |
|---|---|---|---|---|
| m5ult1 → m5ult2 | 4.12 | 6.62-8.83 | 12.46 | 18.54-20.00 |
| m5ult2 → Mini | 4.50 | 6.29-6.50 | 13.33 | 18.92-21.62 |
| Mini → m5ult2 | 6.29 | 11.04-11.21 | 16.42 | 21.62-25.33 |

So quiet adds 1.8 to 4.9 µs to a SEND at the median. It adds 5.2 to 8.9 µs to a trip, 32 to 62% of a warm 16 KiB trip
(output_data/quiet-20261023).

On the m5ult2-Mini link alone, after 2 ms or more of quiet, about one trip in ten took 0.6 to 2.7 ms
(measurement.md:3464-3467). After 0 or 200 µs its p90 trip was under 29 µs. From m5ult2 the SEND itself still ended in
time (p90 9.88 µs after 2 ms), but the trip's p90 was 2,648 µs. From the Mini the SEND's p90 was 1,758.83 µs after
2 ms (output_data/quiet-20261023). Its cause is not known.

## How it ranks

The operator ruled on 2026-10-23, of the sentence "It pays nothing in steady state": "this almost certainly
generalizes for almost all models we will ever run and will always generalize for very large single matrix*matrix
operations used for any reason at any scale that matters; we can file this under 'idiosyncracies which do not affect
the (throughput,latency) 2-coordinate pareto superiority or inferiority of kernels at all plausible shapes. we don't
care about artifacts tah only appear at one tensor shape ever with no plausible origin or reason for study; this is why
the lstm-chain-demo demands that multiple shapes be run at fastest possible speed rather than a single shape."

This note files the quiet link there. Every step and every transfer that is measured already holds its cost. It needs
no rule and no mechanism of its own.

## 1. Do we care?

Not within a run of the chain or the LSTM. Each crosses every step. The chain on two Studios takes 28.5-28.8 µs a step.
The LSTM on three nodes takes 71.3-84.0 µs a tick (output_data/hourly/log.jsonl, run 20261023T2113Z). So no link they
use is quiet 200 µs while they run. Shorter stretches can still cost. After 40-50 µs of quiet the Mini's SEND took
6.2 µs, 4.3 µs more than with a frame every 5 µs (above). After 0-10 µs a 3 KB round trip read 12.8-13.8 µs, and
after 40-160 µs it read 16.5-18.8 µs (gapprobe2, measurement.md:3171-3172). Stretches of 10-40 µs are not measured.
Whatever they cost comes every step, so it is inside the measured step time.

Four cases meet a quiet link. Costs are medians on the SEND unless the row says otherwise.

| Case | Quiet before it | Cost | How often |
|---|---|---|---|
| A run's first crossing on a link | since the last run | +1.8 to +4.9 µs | once a run |
| A cable a placement rule skips | as long as the skip | +1.8 to +4.9 µs, 6-17% of one chain step | once a switch |
| Bulk transfers spaced apart | the spacing | +1.8 to +4.9 µs, fixed | once a transfer |
| The Mini's tail | 2 ms or more, m5ult2-Mini only | 0.6-2.7 ms on about 1 in 10, about 0.06-0.27 ms on average | once a transfer |

- **A run's first crossing.** It pays once. A steady step leaves it out, as the hourly run's steady figures do.
- **A cable a placement rule skips.** A rule that keeps a chain on its cable pays nothing. A rule that moves it onto a
  cable quiet 200 µs or more pays +1.8 to +4.9 µs on the first SEND there. That is 6 to 17% of that one 28.5-28.8 µs
  step. A switch every k steps costs a k-th of that a step. The rule's measured step time shows it, and the rule is
  ranked on that.
- **Bulk transfers spaced apart.** The cost is fixed per transfer. Section 2 gives the size above which it is under 1%.
- **The Mini's tail.** It is the one case that costs milliseconds. It needs 2 ms of quiet on that one link. The chain
  and the LSTM never leave a link quiet that long while they run. It can meet a run's first crossing on that link, or a
  program whose crossings there are 2 ms or more apart.

## 2. The size above which it doesn't matter

"Doesn't matter" here means the cost is under 1% of the transfer's time on one cable. One cable moves 9.21-9.25 GB/s a
direction (measurement.md:364, 2026-10-01). At 9.25 GB/s, each µs of cost needs 0.925 MB of transfer to fall under 1%.
Only the wire time is counted. A real transfer takes longer, so these sizes are upper bounds.

The cost is fixed per transfer, paid once at its first page. It does not grow with size. The bridge sends every page,
16 KiB, as a SEND of its own (design/transport.md sections 1 and 14). Only a transfer's first SEND follows the quiet.
The link is busy from then on, a page every 1.8 µs at 9.25 GB/s. A frame every 5 µs kept the Mini's SEND at 1.9 µs
(measurement.md:3176-3178). So pages 1.8 µs apart stay warm. This is read from the records, not measured: no record
times a quiet transfer of more than one page.

Each cell is the cost, then the size above which it is under 1% (output_data/quiet-20261023):

| Link | SEND, median | SEND, p90 | Trip, median |
|---|---|---|---|
| m5ult1 → m5ult2 | 4.7 µs: 4.4 MB | 6.1 µs: 5.6 MB | 7.5 µs: 7.0 MB |
| m5ult2 → Mini | 2.0 µs: 1.9 MB | 5.9 µs: 5.4 MB | 8.3 µs: 7.7 MB |
| Mini → m5ult2 | 4.9 µs: 4.6 MB | 8.3 µs: 7.7 MB | 8.9 µs: 8.2 MB |
| The Mini's tail, 2 ms or more | | 0.6-2.7 ms: 0.55-2.5 GB | |

- A median's cost is the largest quiet median over the gaps of 200-5,000 µs, less the warm median.
- A p90's cost is the quiet p90 less the warm median. On the Mini's link it is the 200 µs gap's alone. After 2 ms its
  p90 is the tail.
- A trip's cost covers both ends, so it bounds a one-way transfer's cost from above.

So above about 8 MB the cost is under 1% on every link, at the median and at the p90. Below that it is a large share:
32 to 62% of a 16 KiB trip (above). Only the Mini's tail needs more: 0.55 GB for 0.6 ms, and 2.5 GB for
2.7 ms.

## Sources

- metal-microbench docs/measurement.md:363-364: round trips by size and the one-cable bandwidth, 2026-10-01.
- metal-microbench docs/measurement.md:3165-3178: the Mini's controller after 40-50 µs of quiet, gapprobe2, and the
  keepalive's figures, 2026-10-17.
- metal-microbench docs/measurement.md:3451-3467 (#the-quiet-probe): the quiet probe and its reading, 2026-10-23.
- metal-microbench tools/mesh/hourly/quiet.c and output_data/quiet-20261023/: mesh 3d849a16, 2026-10-23 20:58Z.
- metal-microbench output_data/hourly/log.jsonl, run 20261023T2113Z: the chain's steady step and the LSTM's steady tick.
- design/HETEROGENEOUS-LINKS.md section 10: the keepalive deleted at version 133, and the operator's ruling.
- design/transport.md sections 1 and 14: a row is one 16 KiB page, and every SEND is one page.
- design/data-model.md section 3: the quiet controller, 6.2 µs against 1.9 µs.
