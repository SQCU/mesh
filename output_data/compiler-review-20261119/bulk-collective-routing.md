# Bulk collective routing on 19 November 2026

The 37.35 GB/s direct measurement gave Studio-3 and Studio-2 four parallel cables and two ranks. A four-rank communicator added Studio and m5ult2. Its physical subgraph had ten cables. The ordinary broadcast on the deployed Mesh transport at `b9fdb770` used all ten to move one 64 MiB operand. It sent 53,723,136 bytes straight to the destination and 13,385,728 bytes over three hops. In the forward direction those hops used two cables, then three, then one. With the Mini also participating, the five-rank plan used twelve of thirteen cables. It delivered 44,793,856 bytes directly and 22,315,008 through relays. Both directions selected the same byte shares. [Received bytes, revisions and per-cable counters](../../../../compiler-emit/artifacts/scheduled-broadcast-20261119/installed-public.jsonl)

The compiler gives the bridge the selected peer hops and DMA span sequence. Relays read the completion words of the ranges they forward. For a full four-rank broadcast the source issued two sends and the three remote ranks received 192 MiB in total per 64 MiB source operand. The median of the slowest remote call in each generation was 4.15 ms from Studio-3 and 4.58 ms from Studio-2. Four-rank uneven all-gather also passed with full and masked participants, distinct source lengths and cuts between page boundaries. These experiments used the public collective entry points on the five installed bridges. [Broadcast measurements](../../../../compiler-emit/artifacts/scheduled-broadcast-20261119/public-broadcast.md) · [Uneven all-gather](../../../../compiler-emit/artifacts/scheduled-broadcast-20261119/installed-four-rank-allgatherv.jsonl)

## Sustained delivery with an RDMA receipt

The source timed four admissions of 48 consecutive transfers per direction and rank count. The receiver sent the eight bytes it observed at the end of each landed operand back over RDMA. The source verified those bytes before issuing its next transfer. Admission, four warmup transfers and the final payload scan ran outside the timed interval. The measured loop included both public collective calls and every receipt. Across the 192 timed transfers per cell, useful payload divided by total source-clock time gave the following decimal GB/s. [Timed transfers and physical byte counts](../../../../compiler-emit/artifacts/scheduled-broadcast-20261119/sustained-public.jsonl)

| Participating ranks | Studio-3 to Studio-2 | Studio-2 to Studio-3 |
|---|---:|---:|
| Two | 35.35 | 35.23 |
| Four | 42.80 | 43.03 |
| Five | 47.82 | 48.91 |
