# FP32 accumulation through streamed contractions

Implementation and measured driver: `da28c7e`, September 13, 2026. This completes
the contraction-storage precision item in the
[previous implementation report](literature-pipeline-2026-09-13.md#remaining-boundaries).

## Library behavior

Following the [Pallas mixed-precision matmul](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#bfloat16-matrix-multiplication),
`mesh.nn.linear` retains FP32 K-panel products and FP32 reduction-tree values.
It converts once per completed output tile to the input dtype by default.
`output_dtype="float32"` retains the result for a surrounding contraction sum:

```python
partial = linear(program, x, weight, tile_rows=64, tile_k=64,
                 tile_columns=64, output_dtype="float32")
```

FFN up-projection sums across input partitions stay FP32. Swish reads FP32 and
writes the activation dtype in its existing launch. Down-projection partials and
sums stay FP32 until the final FFN output conversion. Each tile has only its own
mathematical dependencies; neither conversion nor reduction creates a barrier
across output tiles. There is no mutable accumulator shared by concurrent kernels.

The CPU and MPS paths already support the required output storage. Their kernel
implementations and direct canonical-page operand bindings are preserved. The
change resides in library composition and allocation types, plus the existing
Core ML generator. Core ML now promotes operands before matmul and removes its
internal fixed-32-K half-product splitting. Its FLOAT32 conversion setting alone
had not promoted operations that were already typed FP16.

The [algorithm-source contract](algorithm-sources.md#kernelsdot)
records the publications, rounding boundaries, implementation mechanisms and
limits of backend claims. FP32 tree summation need not be bitwise identical to a
serial FP32 accumulator; both avoid intervening FP16 storage rounding.

## Operational evidence

The existing streaming-algebra example gains `--dtype float16` and optional
`--coreml PYTHON GENERATOR CACHE`. The ordinary FP32 invocation remains available.
No separate evaluation harness was added.

Its additional tiny linear expression has two rows and two K panels. Expected
outputs are exactly 2 and 0. The first detects a residual lost by FP16 panel
rounding; the second detects panel overflow before cancellation. All recorded
runs produce both outputs exactly.

The existing FFN → RMSNorm → summed embedding → FFN → RMSNorm chain also passes
with an unrelated input section withheld until a downstream section completes.
FP32 retains the float64 reference and original tolerance. FP16 reference values
round at declared numerical output boundaries; its tolerance is 3e-3 absolute
plus 3e-3 relative. The exact cancellation expression supplements that random
chain with failures its tolerance could otherwise miss.

| Execution | Measured invocations after one warmup | Maximum absolute gold error |
| --- | ---: | ---: |
| Local CPU, FP32 | 3 | 1.653720e-6 |
| Local Metal, FP32 | 3 | 1.845012e-6 |
| Two-node CPU, FP16 | 3 | 0.001953125 |
| Two-node Metal, FP16 | 3 | 0.00390625 |
| Local Core ML, FP16 | 1 | 0.00390625 |

Error maxima include warmup. The exact cancellation expression runs on rank zero;
the distributed gold exercises both M5 Max and M4 Pro. Full logs and paired
compute/transfer traces are retained in
[measurements](../measurements/fp32-accumulation-2026-09-13/).
The CPU/Metal packages use the unchanged native library from the preceding work
and Python composition from this commit. Core ML uses the changed generator.

These runs establish numerical behavior and continued partial progress. Their
batch timings are retained as observations, not a matched old/new speedup claim.
The FP16 CPU path still uses its existing scalar contraction loop, while FP32
uses Accelerate; that performance difference remains a separate kernel issue.

Core ML recorded 326 native submissions and 326 matching caller-owned output
backings. It selected zero ANE operations in its compute plan for this case.
The precision fix is not an ANE acceleration claim. Compiler-internal cast storage
is opaque: public output identity does not establish internal copy-free execution.
All setup and model compilation finish before the numerical batch.

Both bridges retain the existing bounded 549,650,432-byte registered region.
Measurement clients exited normally or by handled SIGTERM; neither bridge was
restarted or killed. No bindings, compatibility layers or tensor-wide waits were
introduced.
