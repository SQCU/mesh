# What "Amdahl improvement" means here — gemma-4-E2B on an M5 Max, made marginally faster by a mesh

This is the tedious version. It exists because every previous measurement of "the mesh"
compared a slow, wrongly-shaped two-node run against a solo run and then explained the
result. The objective is the opposite and it is narrow: take an **otherwise optimal**
gemma-4-E2B — a 2.3B-effective model whose whole decode step is ~5 ms on the fastest
laptop chip sold — running on an **already fast** machine (M5 Max, 614 GB/s), and make it
**marginally faster** by offloading part of each step onto peered machines over
Thunderbolt 5 RDMA. The model is absurdly small; the FFN is 1536×6144; a single layer is
~150 µs. Every microsecond of mechanism is visible against that, and none of it is
forgiven. The point of this note is that (a) that is the bar, (b) extremely low latency
is nonetheless *possible* on this transport, and (c) there is a regime — long KV — where
the mesh wins *anyway*, whose existence justifies exactly zero overhead.

## 1. The reference class (what "otherwise optimal" means, in numbers)

Public, single-stream, int4-class weights, gemma-4-E2B:

| runtime | chip | prefill tok/s | decode tok/s | ms / decode forward |
|---|---|---|---|---|
| Google LiteRT-LM, GPU (official table) | MacBook Pro M4 Max (546 GB/s) | 7,835 | **160** | 6.3 |
| Google LiteRT-LM, GPU | iPhone 17 Pro | 2,878 | 56 | 18 |
| MLX 0.32.2 (mlx-vlm text path), QAT-4bit, 4,085-token prompts | M3 Max (410 GB/s) | 6,117 | **100.6** | 9.9 |
| llama.cpp Metal, UD-Q4_K_XL | base M4, 16 GB | 82 (short prompt) | 55.7 | 18 |
| **this engine**, solo, Q4_K_M | **M5 Max (614 GB/s)** | 3,666 (8×580 tokens, concurrency 2) | **182** at B=1 (5.48 ms); 68–84 in two other runs | 5.5 (–14.6) |

Sources: LiteRT-LM overview table (developers.google.com/edge/litert-lm/overview);
unsloth-zoo PR #1246 (M3 Max MLX numbers, ABBA-blocked); magnitude #82 (llama.cpp on M4);
`output_data/mesh_e3/solo_report_t4.out`, `output_data/mesh_reports/20260916T012857…-solo.json`.

Two readings. Decode: this engine at 5.5 ms/token on the M5 Max is at the reference
class (LiteRT's 160 tok/s on a 546 GB/s chip scales to ~5.6 ms at 614 GB/s). The two
slower solo runs (11.9, 14.6 ms) are not a baseline; a comparison that uses them is
refused (I24). Prefill: 3,666 tok/s on the fastest chip against 6,117 (M3 Max, MLX) and
7,835 (M4 Max, LiteRT) is **~2× below the reference class** — the solo prefill path is not
"otherwise optimal" yet, and until it is, a prefill speedup is a speedup over a slow
baseline and counts for nothing.

## 2. The model, as bytes per decoded token

`~/models/gemma-4-E2B-it/config.json`: hidden 1536; intermediate 6144 (some layers
double-wide, 10,504 columns in the engine's realization); 35 layers; 8 query heads, **1 KV
head** (MQA); sliding head_dim 256, global head_dim 512; sliding window 512; layer types:
**7 global** (every fifth layer: 4, 9, …, 34), 28 sliding; `num_kv_shared_layers = 20`
(layers 15–34 reuse the KV of layers 0–14 — so there are 3 global caches and 12 sliding
caches, but **7 global layers read a full-length cache** each step); PLE dim 256; vocab
262,144.

Per decoded token, per sequence, f16 KV:

- global layers: 7 × (K + V) × 512 × 2 B = **14 KiB per context position** → `14 KiB × L`
- sliding layers: 28 × 2 × 256 × 2 B × 512 positions = **14 MiB, constant**
- weights: 2.3B effective at Q4_K_M ≈ 1.4 GB + lm_head 262144×1536 (~0.3 GB at Q6) ≈
  **1.7 GB, constant** (f16: 4.6 GB; the PLE row is 35 × 256 × 2 B, nothing)

So, on one M5 Max at 614 GB/s, with the weight term taken as measured (5.5 ms includes
its launch gaps):

```
T_solo(B, L) ≈ T_w + B · L · 14 KiB / 614 GB/s
             ≈ 5.5 ms + B · L · 23 ns
```

That is the "time per decoded token increases linearly with KV length": +3.0 ms per
sequence at L = 128k. It is the only term of a decode step that grows.

## 3. The overhead of the current mesh path, against that

`bindMeshLayer` runs, per layer, `A → reduceScatter(add) → allGather → B →
reduceScatter(add) → C → allGather`: 5 command buffers and 4 wire crossings on the
critical path, ×35. Measured crossing cost 0.28 ms (GPU end → completion handler ~60 µs
and commit → GPU start ~70 µs on both sides, wire ~18 µs, RDMA-runtime metadata ~1–2 µs
at ABI 75). Local dependency boundary ~0.13 ms.

```
overhead / forward ≈ 35 × (4 × 0.28 + 0.13) ms ≈ 44 ms      (≈ 140 host round trips)
```

Against §1–§2: **one crossing costs one layer of compute.** The overhead alone is 4–8×
the whole solo forward, which is the measured pair result (5.5 tok/s vs 182). 96% of it
is Metal host round trips, 4% wire, <1% the RDMA runtime. The H campaign (25 commits,
ABI 50→75) moved this number by about 0.1%.

## 4. How fast a transaction *should* be — the floor, stage by stage

Today's crossing cost is irrelevant to this section. The RDMA link is an RDMA link; the
collectives are push-only (every node SENDs its partial into the consumer's posted
RECV pages the moment the bytes exist; nobody asks, waits, or checks); the partial is
exactly the bytes the consumer's kernel reads (Definition); the consumer is resident.
Under those requirements one transaction — producer kernel's last store → consumer
kernel's first load of the remote partial — is this budget and nothing else:

| stage | mechanism | floor |
|---|---|---|
| 1. last GPU store → CPU spinner sees the completion word | unified memory; the spinner polls one 32-B line the kernel writes last | ≤ 0.5 µs |
| 2. spinner: ring pop → `ibv_post_send` doorbell | prebuilt WR/SGE in the record (H2); store→doorbell ≤ 200 ns (H7) | ≤ 0.2 µs |
| 3. NIC DMA → TB5 fabric → remote DMA → remote CQ | measured 12 µs one-way for 32 KiB on this stack, of which serialization at 10 GB/s is 3.3 µs; a 16.4 KiB `(m,l,O)` partial or a 3 KiB decode vector rides the ~8–9 µs fabric base | **8–12 µs** |
| 4. remote CQ poll → presence stamp store | CQ→ring ≤ 100 ns (H7); the stamp is the ring store | ≤ 0.1 µs |
| 5. resident consumer kernel sees the stamp | one threadgroup spinning on a device-coherent word (X9/F7) | ≤ 1 µs |

**Total: ~10–14 µs for a decode-sized partial, i.e. the wire plus ≤ 2 µs.** That is the
number. Not 280 µs; not 25 µs; the fabric, plus one shm→L1 on each side, plus one GPU poll.
F7's pass value ("arrival-to-first-consuming-kernel ≤ 2× the wire estimate") and H7's
store→doorbell / CQ→ring bounds are this table; E7 uses `c ≤ 15 µs`.

What the floor does to the model:

- Megatron per layer (o_proj partial + down_proj partial, both pushed as they complete):
  2 × ~12 µs = ~25 µs exposed per layer, **~0.9 ms per forward** on a 5.5 ms decode step.
- With FFN columns and attention placed ∝ bandwidth on M5 Max + M4 Pro (`Σ BW/BW_leader =
  1.44`), the bandwidth-bound weight term is 5.5/1.44 = 3.8 ms; 3.8 + 0.9 = **4.7 ms vs
  5.5 ms: 1.17× at B = 1 decode, on a 2.3B model, with an M4 Pro as the peer.** Small,
  positive, and therefore the objective (I21). The crossover of §5 collapses to `7c ≈
  85 µs → B·L× ≈ 12k`, so at B = 1 the KV term is a second, independent win past ~12k
  tokens, compounding to ~1.3× at 128k.
- Four M5 Ultra-class nodes (`Σ BW/BW_leader ≈ 4`): weight term ~1.4 ms; reduce-scatter of a
  3 KiB decode vector among 4 is one hop each way, so still ~25 µs per layer, ~0.9 ms per
  forward: **~2.3 ms vs 5.5 ms, ~2.4× at B = 1** — on a model whose whole step was 5.5 ms.
  That is what "marginally faster on an already-fast machine" buys at the floor; every
  microsecond above the floor comes straight out of it.

Anything between these numbers and 280 µs is not "the transport": it is a command-buffer
commit, a completion handler, a dependent metadata load, a layout copy, or a wait, and
each one has a row in this document whose status is ✗ until it is gone.

## 5. The crossover: where the mesh wins *anyway*

The growing term in §2 is attention over the KV, and it is the one term that splits
without touching the weights: **sequence-split attention** (flash-decoding split-K over
positions). Node `i` holds positions `[p_i, p_{i+1})` of the 3 global caches; for a query
it computes, per head, `(m_i, l_i, O_i)` — running max, running sum, unnormalized output
— over its positions only; the combine is `m = max m_i; l = Σ l_i e^{m_i − m};
O = Σ O_i e^{m_i − m} / l`, associative, exact in fp32, and tiny: 8 heads × (512 + 2) × 4 B
= **16.4 KiB per sequence per layer**. No K or V ever moves. The KV is placed once, when it
is written (prefill writes each node's positions to that node), so decode moves nothing
but the 16 KiB partials: **7 crossings per token** (one per global layer), not 140.

With bandwidths `BW_i` (M5 Max 614, M4 Pro 273 GB/s), positions placed ∝ `BW_i`:

```
T_mesh(B, L) = T_w + B · L · 14 KiB / Σ BW_i + 7 · c
```

and the mesh is ahead iff

```
B · L · 14 KiB · (1/BW_leader − 1/Σ BW_i)  >  7 · c
```

For M5 Max + M4 Pro: `1/614 − 1/887 GB/s = 0.50 ps/B` → the left side is `B · L × 7.2 ns`.

| crossing cost c | 7c | B·L at crossover | B = 1 | B = 8 | B = 32 |
|---|---|---|---|---|---|
| **12 µs (the floor, §4)** | **85 µs** | **12k** | **12k** | **1.5k** | **370** |
| 25 µs | 175 µs | 24k | 24k | 3k | 760 |
| 0.28 ms (the current path, for the record) | 1.96 ms | 272k | never (context is 128k) | 34k | 8.5k |

That is the crossover: **at today's crossing cost it does not exist inside the model's
context window for a single stream**, and appears only for batched decode past ~34k
tokens per sequence. At 25 µs it is at 24k tokens for one stream — a long document, an
ordinary agent trace — and at 5 µs it is at 5k, i.e. most real conversations. The other
crossover, where the KV read equals the weight read and the split therefore pays for
itself at the *weight* term's scale, is `B · L = 5.5 ms / 23 ns ≈ 236k` (B = 1: beyond the
context; B = 8: 29k; B = 32: 7.4k).

Add the FFN column split on top (r_M4 ≈ 0.17 at large rows, 0.73 at B = 8) and the constant
term also shrinks, by the E2B bound (1.11 prefill, ≤1.42 at B = 8 decode); the two
compose because they touch different terms.

## 6. Why the crossover justifies no overhead

Every row of the table above is a *loss* for every `(B, L)` below it. A mechanism that
adds 0.28 ms per crossing does not "move the crossover"; it deletes it from the usable
range. The crossover is where an implementation with *no* overhead starts to win; it is
computed from the measured `c`, never chosen, and the pass values that bound `c`
(H7: ≤ 200 ns store→doorbell, ≤ 100 ns CQ→ring; F7: arrival-to-kernel-start ≤ 2× wire) apply
whether or not a run happens to land past it. "It will win at 128k" is not a result; it
is the statement that the implementation loses everywhere else.

## 7. The regression rows (deliverables.md)

- **I24** — the baseline is at reference class or the comparison is refused; the
  objective is the modelled margin at every `(B, L)` past `L×(c_measured)`; the
  crossover justifies no overhead.
- **F11** — sequence-split attention as a partial with an associative combine: KV placed
  once by position ∝ bandwidth, `(m, l, O)` partials, 7 crossings per token on E2B,
  16.4 KiB each, no K/V movement.
- **E7** — the measured crossover: `T_solo(B, L)` and `T_mesh(B, L)` on the public
  endpoint at `L ∈ {4k, 16k, 64k, 128k} × B ∈ {1, 8}`, `c` measured per crossing, the
  predicted and observed crossover reported side by side.
