# Page readiness for a gathered output

Mesh `b11baac6d`, compiler-emit `53783a4`. A four-site placement splits a float32 matrix product of shape (1024, 32768) × (32768, 1024) into four 256-column products. The root's TorchInductor program writes its own product into its output slice. Three copy kernels, each with a deferred pass, read the received slices into the other columns. The public callable returns a tensor view of that output allocation.

**A:** The FX placement declared a whole-operand join for each received input to `aten.cat`. Reading row 512 of one shard then depended on receiving row 0 of that shard. Torch concatenation maps each output element to one input element. The two rows read different 16 KiB pages. [ATen concatenation](https://docs.pytorch.org/docs/stable/generated/torch.cat.html)

**B:** Placement declares independent pages when all consumers on the destination node concatenate that input. The numerical compiler uses the input access map to compute which pages each Metal threadgroup reads. A caller can supply its own transfer policy through the existing placement interface. [Compilation requirements](../design/aot-compilation-change-request.md) · [Transfer requirements, R4](../design/plan-to-bridge-change-request.md)

## Delayed-page comparison

The experiment ran the root's saved and reopened Metal program on an M5 Max. The harness supplied each remote shard's eager values at the byte offsets in its compiled extent plan. On the fourth call it held one arrival word for source 1's first page. It sampled the shared output at rows 0 and 512 while the GPU was running, then released that page. A 250 ms deadline bounded the observation. Every completed invocation alternated between two inputs and compared its entire returned matrix with eager PyTorch.

| Output before releasing source 1's first page | Whole-operand join | Per-page declaration |
|---|---|---|
| Source 1, row 0 | Previous value | Previous value |
| Source 1, row 512 | Previous value through both deadlines | Current value in both runs |
| Sources 2 and 3, row 512 | Current value | Current value |

Each form passed two 16-call reopened-image runs. The changed form also passed 12 alternating calls with a 512 × 8192 by 8192 × 3079 product and unequal output widths.

## All-ready control

Four interleaved 20-call batches used the same root program and eager values with every page ready before submission. The entries below are medians over the last 12 calls of each batch. GPU times for copies and compactions sum the three corresponding dispatches within each sample.

| Root measurement | Whole-operand join, two batches | Per-page declaration, two batches |
|---|---:|---:|
| Three GPU copy dispatches, µs | 74.0, 73.9 | 38.9, 39.0 |
| Three GPU compaction dispatches, µs | 26.8, 26.8 | 26.7, 27.4 |
| GPU unit containing the local matrix product, µs | 1810, 1815 | 1985, 2064 |

The observed full calls had median times of 2.26 ms in the two whole-operand batches and 2.40 and 2.50 ms in the per-page batches. The recorded matrix unit varied more than the copy savings. This comparison measures root execution on one GPU. The physical network and the other three GPUs require separate measurements.
