# The mesh as a matrix coprocessor: Xonotic payload demo

Supersedes the crossover analysis in `xonotic-bot-compute.md`. Every number here was
measured on Mac16,11 and Mac17,6 unless marked otherwise.

## What is being demonstrated

Not "two computers are faster than one." The claim is narrower and harder: **a Mac mini
is usable as a tile-streaming matrix coprocessor**, addressed over an RDMA fabric, on
problems that can be swizzled across several matrix units as a higher-level SIMD
universe. Sparsity, routing and compaction are the swizzle. The direct analogies are to
the matrix operations that perform best on strictly superior silicon — and the point is
that the *algorithmic* dividend is available to us even though the *hardware* support is
not.

## Measured hardware

| | Mac17,6 (M5 Max) | Mac16,11 (M4 Pro) | ratio |
|---|---|---|---|
| GPU cores | 40 | 16 | 2.5× |
| dense GEMM fp32, N=4096 | 27,374 GF/s | 5,182 GF/s | **5.28×** |
| same, bf16 | 44,728 GF/s | 5,827 GF/s | 7.68× |
| gain from bf16 over fp32 | +63% | +12% | — |
| memory bandwidth (memcpy r+w) | 362 GB/s | 238 GB/s | 1.52× |
| CPU/AMX GEMM (Accelerate) | 1,695 GF/s | — | — |
| fabric, one TB5 cable | 8.7 GB/s (69.3 Gbit/s), ~10 µs RTT | | |

Two conclusions fall straight out.

**The M4 Pro has no tensor cores.** The M5's 63% gain from reduced precision is
dedicated matmul hardware; the M4 Pro's 12% is just cheaper ALU traffic. So the honest
claim is saturating its GPU ALUs, and MLX already does — 5,182 GF/s against a plain-ALU
peak of 5.3–6.1 GF/s is **84–97% of theoretical**. There is no hand-tuned kernel worth
writing, and more importantly no skeptic can argue the single-machine baseline was
crippled.

**Stay in fp32.** bf16 widens the machine gap to 7.68× while buying the mini only 12%.

**The mini must host.** If the MacBook hosts the server and the mini assists, the mesh
buys 1.17×. Inverted — mini hosts, MacBook is the coprocessor — it is **6.96×**. That is
also the honest deployment story: minis are the always-on appliances, the MacBook is the
workstation that shows up.

## The workload, and what each piece survives

A design is only as good as the attack it survives. Three were fatal to earlier drafts:
**precomputation collapse** (`Σγᵏ Aᵏ` folds 32 iterations into one operator), **low-rank
collapse** (a smooth dense kernel factors as `UVᵀ`), and **sparsity collapse** (a
visibility-masked operator is 90% zeros, and sparse work is memory-bound, which shrinks
the machine gap from 5.28× to 1.52× and destroys the demo).

| piece | lever | scaling | measured asymmetry | survives because |
|---|---|---|---|---|
| MoE routed experts | bots | linear | 2.83× | routing is data-dependent per solve; experts are independent matrices; sort→block→GEMM→scatter *is* the optimal kernel |
| Gram over residual basis | **R** | **quadratic** | 2.0× → **6.6×** | dense by construction, full rank, and the standard algorithm for all-pairs |
| team count | **k** | **quadratic** | — | k(k−1)/2 pairwise couplings; R ∝ k so Gram cost ∝ k² |
| 2:4 compaction | — | — | — | halves bytes, doubling the coprocessor frontier per interval |

Measured Gram scaling, n=4096, R as the lever:

| R | M5 Max | M4 Pro | ratio | FLOPs |
|---|---|---|---|---|
| 512 | 0.55 ms | 1.56 ms | 2.8× | 2.15 GFLOP |
| 1024 | 1.07 ms | 2.18 ms | 2.0× | 8.59 |
| 2048 | 1.77 ms | 6.67 ms | 3.8× | 34.36 |
| 4096 | 3.88 ms | 25.69 ms | **6.6×** | 137.44 |

The asymmetry *widens* with the lever: the M4 Pro flatlines near 5.3 TF/s while the M5
climbs to 35.4. This is the opposite of what sparsity would have done, and it is why the
lever is worth having.

MoE compute-to-communicate is **4096 FLOP per byte** — the property that makes a link
30× slower than local memory survivable.

## Compaction is a reach optimisation, not a memory one

Distribution is bandwidth-bound, so halving the bytes per node doubles the number of
coprocessors reachable inside a fixed wall clock:

```
edge 8.7 GB/s, degree 3, 100 ms budget
tile/node    dense    2:4 compacted    gain
   16 MB       163        326          2.0×
   64 MB        41         82          2.0×
  256 MB        10         20          2.0×
```

Constant 2.0× regardless of tile size, or **+0.63 hops of reachable depth** at degree 3.
This is why 2:4 belongs here despite Apple having no silicon for it: we take the
algorithmic dividend that NVIDIA's hardware was built to exploit and spend it on fabric
reach rather than on throughput.

## The ABI

Today the workload is a C symbol resolved at static link (`make F=f-yourthing.c`). That
is the wrong boundary, and three independent arguments give the same replacement.

- **Functor law.** `F(id) = id` only holds if the transport genuinely cannot inspect the
  payload. A shared address space makes that a discipline; a process boundary makes it
  structural.
- **Affine ownership.** A page is held by exactly one stage and freeing is consumption.
  Crossing to another process is a transfer, which is what a submission/completion ring
  *is* — and a page never returned becomes detectable rather than silent.
- **Reachability.** `RDMA-RULES.md`: a verbs process that dies badly can wedge the driver
  into a physical power cycle. Keeping `mesh_f` inside that process makes a workload
  pointer bug into a node outage.

**Shared registered memory + SPSC submission/completion rings carrying span
descriptors.** The workload maps the same pages the NIC wrote, so zero copy survives the
process boundary. Page-table statistics become another mapped region rather than a
parameter, which dissolves that question instead of answering it. And the workload can
then be MLX — which matters, because the solver needs the GPU and a C symbol cannot
reach it.

## Xonotic integration

Payload: a cart on a track whose speed is a function of contested occupancy. Chosen
because it makes coordination quality **spatial and continuous** — the cart's position is
a running integral of how well the planner is working, so a viewer needs no overlay and
no prior knowledge of good play. Halve the compute and the cart stalls; rejoin the second
machine and it resumes.

Tractable: `func_train` already exists (`qcsrc/common/mapobjects/func/train.qc`), and
`domination` is a 740-line mode to copy from.

Teams: `qcsrc/common/teams.qh:3` has `const int NUM_TEAMS = 4`, with team IDs that are
Quake palette colour indices (5, 14, 13, 10) rather than sequential — but the file already
carries a `TEAMNUMBERS_THAT_ARENT_STUPID` path giving 1..4, left disabled. Extending past
4 is anticipated rather than fought.

Multi-team payload is not a bigger two-team game. Five teams contesting one cart creates
coalition structure — trailing teams share an interest in blocking the leader without
allying permanently — and the correlation between teams' unmet objectives *is* that
coalition signal. The Gram matrix and the game mode want the same object.

Bot AI to replace: `qcsrc/server/bot/default/` is 7201 lines, a module to substitute
rather than a system to fully understand.

## Open and unmeasured

- MoE dispatch computes expert boundaries in Python with a `.tolist()`. 2.83× is a
  **floor**; a real kernel moves it toward the dense 5.28×.
- The orthogonalisation step must stay on GPU. Measured with `mx.linalg.cholesky` on CPU,
  it consumed half the wall time at R=2048 and flattened the machine ratio to 1.09×. This
  would have quietly ruined the demo, and only showed up because the first measurement
  looked *too even*.
- Whether raising `NUM_TEAMS` stays contained. Loops use `for (i = 0; i < NUM_TEAMS; ++i)`
  which is promising, but colour tables and menu entries are where 4 is usually hardcoded
  separately.
- The ABI is designed, not built.
