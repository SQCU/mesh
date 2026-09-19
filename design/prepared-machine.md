# Prepared machine

## Reference baseline

The target is a measured `S >= 1.10` over fast Gemma-4-E2B B=1 decoding by overlapping useful computation across the RDMA-connected machines. The existing executor is not a preservation constraint. Source structure and compilation alone do not establish the speedup.

The unchanged `litert-lm==0.17.1` GPU benchmark measured **180.54 decode tokens/s** and **8,963.62 prefill tokens/s** on M5 Max with 1,024 prefill tokens, 256 decode tokens, 2,048 context, speculative decoding disabled, warmup and three measured iterations. The engine's `make e2b-reference` repeats that command. `output_data/e2b-reference-20260918/{benchmark.log,run.json,generation.log,generation.json}` records the benchmark and coherent generation. The package uses WebGPU over Metal. The model revision is `b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1`; SHA-256 is `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c`.

Google's [reference results](https://developers.google.com/edge/litert-lm/overview) and [released model card](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) define the upstream implementation and measurement configuration. This is not a same-weight comparison with the old safetensors executor. The old local 182.378 tokens/s citation was delta-coalesced (`output_data/mesh_e3/run.sh`) and is not a baseline. The 49.146 ms resident numerical rewrite and its constructor have been deleted, including the engine target that built it. Its previous layout and instruction tables are removed with that implementation.

## Prepared operands

`A` is the client's shared-memory arena address; `P` is the system page size; `q(r,s,o)` is the physical page bound to operand `o` for rank `r` and invocation slot `s` during preparation. These integers are resolved before GPU buffer creation. Each live operand occupies its own canonical pages. Transport registration and the numerical buffer use those same pages.

| Object | Prepared address | Extent | Constructor | Runtime use |
| --- | --- | --- | --- | --- |
| <a id="M01"></a>Numerical input/output backing | `A + P*q(r,s,o)` | FFN boundary: 3,072 bytes, allocated in whole pages | `rdma/mesh-call.c:mesh_section_create`, `mesh_backing_bind` | Numerical shader reads/writes payload directly. |
| <a id="M02"></a>Receive backing | `A + P*q(r,s,o)` | Caller-specified operand extent, allocated in whole pages | `rdma/mesh-call.c:mesh_transfers_prepare` | Native receive writes payload into the final pages. Connection to the reference FFN is not yet exercised across nodes. |
| Native GPU and LiteRT operand handles | Prepared client's `mesh_webgpu_operand` address | 16 bytes, `_Static_assert` in `rdma/mesh-webgpu.h` | `rdma/mesh-webgpu.c:mesh_webgpu_bind` | Setup passes native handles to LiteRT. These are not transport event records and no per-event load bound is claimed for LiteRT. |
| GPU view of canonical backing | Same `A + P*q(r,s,o)` allocation | Page-rounded physical extent, 3,072-byte logical FP16 operand | `mesh_webgpu_bind`, Dawn `BufferHostMappedPointer`, LiteRT `CreateTensorBufferFromWebGpuBuffer` | No payload copy, map/unmap operation, allocation, or Mesh callback is added by this binding during numerical execution. |
| Reference FFN graph and weights | Prepared upstream model/native objects | 35 graphs, retaining original operators, dimensions, packed INT4/INT2 bytes and quantization metadata | Engine `tools/prepare_e2b.py:prepare`, upstream `LiteRtCreateCompiledModel` | Existing LiteRT kernels execute the arithmetic. Model parsing and operand remapping are setup operations. |

Dawn's [host-mapped-pointer contract](https://dawn.googlesource.com/dawn/+/HEAD/docs/dawn/features/host_mapped_pointer.md) permits shared-memory backing and requires the `HostMappedPointer` device feature. The tested provider additionally requires the `allow_unsafe_apis` device-creation toggle. Its [Metal implementation](https://dawn.googlesource.com/dawn/+/refs/tags/v20260720.160313/src/dawn/native/metal/BufferMTL.mm) uses `newBufferWithBytesNoCopy`. LiteRT's [WebGPU import](https://github.com/google-ai-edge/LiteRT/blob/95b6fb70caf0ba6c42cb0498f9388c13e9470434/ml_drift_delegate/delegate/buffer_handler_webgpu.cc) wraps that native buffer with `owns_tensor=false`. Mesh retains ownership of the actual pages. The required Dawn disposal callback does not free borrowed pages; native views must be destroyed before the owning program's pages are released.

`make e2b-prepare` generates the upstream schema headers, extracts all 35 reference FFNs and builds the native page-binding library. The extracted graphs are full FFNs, not a completed tensor-parallel partition. In the released quantized graph, the final projection is followed by output requantization: summing separately requantized shard outputs would change the function. The partition boundary must expose the linear contribution before that operation and apply the original output quantization after combination.

## Executed connection

`output_data/e2b-rdma-preparation-20260918/operand-layers.json` and `operand-layer-{0,15,34}.log` record execution of reference FFNs on real canonical Mesh pages through LiteRT's WebGPU backend. All three produced 1,536 finite FP16 outputs; the native clients and isolated bridges exited 0. Both 6,144-column INT4 and 12,288-column INT2 FFNs were exercised. The arena was 76,922,880 bytes; model/upload allowance was 64 MiB and compiler allowance 512 MiB. No network device or persistent dispatch was opened. Elapsed process times include setup and are not throughput measurements.

Native queue completion was observed explicitly by the measurement client before it read the shared output. External-buffer execution supplied no LiteRT tensor completion event; returning from `LiteRtRunCompiledModel` did not establish CPU-visible completion. The page-binding library contains no invocation, wait, completion callback or scheduler. Production publication at the numerical boundary, preposted cross-node transfers and the complete reference decode chain remain unimplemented. No crossing latency or `S >= 1.10` result exists yet.
