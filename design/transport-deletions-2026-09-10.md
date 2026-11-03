# Transport deletions and operational evidence, September 10, 2026

Measured canonical source: `454cd1c082c282126c8187f9bba9fab5133cb61b`.
Numerical implementation: `7faa404caefd00b33e1d22b24383dd2075c9f6c7`.
Committed real-input configurations: `5bb215e19b49598bcdaa762b8da92e2c7a340a28`.
Both machines checked out matching main revisions and built canonical mesh,
the numerical caller, and the existing one-GiB executables. No uncommitted source
was transferred. The connected source rewrite preceded those builds and runs.

## Removed implementations

The worker ownership handshake, reconnect/reset loop, routing layer, flight
tracing, client completion pump, receive-completion ring, and send-binding scan
are deleted. Completion submits configured outgoing records directly; the sole
verbs owner consumes local descriptors and actual CQ completions.

The table-wide reference counter and deferred-retirement proposal were removed
before building. Invalidation changes the table index and allocation block mask.
Actual registered requests have index masks, projected only when configuration
allocates backing. There is no receipt, waiting-for-release loop, cancellation
walk, or EBUSY retirement ladder.

Selection and numerical configuration arrays live with their configured caller.
Command-specific extent-address slices remove concurrent CPU writes to a prior
command's address inputs. A closure cycle retaining the function and its buffers
was deleted. Reduction and normalization now share one encoder and threadgroup
FP32 storage; the intermediate global accumulator and second encoder are gone.

The generic tensor-command interpreter, tensor dylib, flight modules and unused
DLPack definitions are deleted. Python ABI declarations are generated from the
canonical C headers. Live application encoders were migrated to concrete
configured functions rather than silently deleting their supported application.
Python send layout/reservation gates, transport pending queue, artificial slicing,
inflight scans, teardown retry loop and numerical-worker idle sleep are removed.

## One-GiB streams

Three consecutive one-way transfers reused the registered bridge. Each sent
1,073,741,824 bytes as 65,536 literal 16,384-byte payloads. Both endpoints reported
all pages complete and zero mismatched words; every detach returned zero.

| Run | Receiver interior bytes/s | Receiver first-to-last seconds |
| --- | ---: | ---: |
| 1 | 9,336,109,208 | 0.114990 |
| 2 | 9,347,080,914 | 0.114867 |
| 3 | 9,346,877,500 | 0.114865 |

The reported interior interval covers 858,996,736 bytes. Submission took
0.608–0.647 ms at the sender. These are streaming measurements, not small-message
latency measurements. The raw and bridge now use the same RTS-before-receive-post
ordering with the existing QPI exchange; there is no additional setup message.

## Existing real-input FFN evaluator

`NFE_CFG=ffn tools/mesh/nfe.sh guardless-454cd1c 910600000 4096 6 1,2`
ran the existing layer-zero FFN evaluation on 4,096 real input tokens. Each
participant completed ten invocations per configuration, including six measured
samples. Both configurations had zero nonfinite outputs, empty error metadata,
exact peer agreement, and successful context destruction. Relative RMS against
the local numerical reference was 0.0003935029637187864, below its 0.002 limit.

| Invocations in flight | M5 median invocation ms | M4 median invocation ms |
| --- | ---: | ---: |
| 1 | 31.57225 | 31.61458 |
| 2 | 49.76535 | 49.98231 |

The bridge was registered once for the maximum configuration, reused for both
cases, and stopped normally at the end. This is FFN evidence, not a full 48-layer
NFE result. The evaluator still reports `native_internal_storage_verified=false`;
these measurements do not establish the private backend's internal storage mapping.
The migrated Xonotic generated-encoder path still requires its own real-input
application execution evidence.
