# Prepared native crossings

These are the emitted instructions in `rdma/.build/mesh-flow.s`, produced by `make -C rdma native-audit` with `-O2 -Wall -Wextra -Werror`. They describe the current native transport, not the deleted Swift/generated numerical executor. They do not establish crossing latency or distributed speedup. Provider-internal instructions and native numerical shader instructions are not counted as zero.

| Value-ready to SEND post, one prepared stream | Memory operation | D0 object |
| --- | --- | --- |
| GPU payload visibility | Aligned 16-byte system-coherent payload reads and writes, system-scope fence, threadgroup barrier in the measured 256-lane publication workgroup | M01, M07 |
| GPU destination | Read prepared publication record's destination and argument | M10; `prepared_publication`, 32 bytes |
| GPU publication | Store argument to the already prepared SEND ready cell; system-scope fence | M04 |
| TX poll | `ldapr x9, [x8]` | M04; current cell address held in `x8` |
| Native request and next cell | `ldp x1, x21, [x8, #16]` | M04; both fields in that same 32-byte cell |
| Accept publication | `str xzr, [x8]` | M04 |
| Native queue | `ldr x0, [x8, #8]` | M04 |
| Post | `blr x20`; native function already held in a register | M08 native queue |
| Advance | Move the already loaded successor into `x8` | M04; no memory operation |

The one-stream selection occurs at thread entry. For multiple independent streams, the loop additionally stores the accepted successor at its prepared cursor position and loads the next stream's cursor; the number of probes is bounded by configured streams times invocation concurrency, not the 35-layer graph. No later receive can change the prepared send order. Empty polling checks the explicit shutdown word; accepted publications do not read it before posting.

| RECV completion to consumer input | Memory operation | D0 object |
| --- | --- | --- |
| Native CQ polling | Provider writes the 48-byte completion at the poller's fixed output address | M11; native `ibv_wc`, constructed by `mesh-flow.c:404` |
| Completion status | Load word at completion +8 | Same completion |
| Native receive identity | Load pointer at completion +0 | Same completion; native `wr_id` |
| Destination and value | Load pair at receive record +80 | M08 |
| Publish native completion | Release-store the value at that destination | M07's directly bound completion word |
| Receive repost binding | Load next request and QP at receive record +64 | M08 |
| Repost | Call the register-held provider function with the prepared next request | M08 |
| GPU observation | System-scope fence, system-coherent read inside each actual numerical reader until the actual dependency arrives; explicit shutdown read only while absent | M07, M12 |
| Numerical consumer entry | Original numerical kernel observes its directly bound M07 word; no reset or intervening dispatch | M07, M06 |
| Numerical input | That same upstream kernel reads the canonical received payload | M02, M06 |

The receive destination/value and next-request/QP fields occupy one aligned 32-byte line at +64 within the 128-byte receive record. Native TX discovery depth is 0 for one stream; native RX discovery depth is 1 from `wr_id`. TX has three accepted-path load instructions, all from one M04 cell. RX has four through repost: two from the native completion and two from its directly addressed record. Neither accepted path reloads a spilled register. The provider's own loads, coherence cost, and full store-to-consumer latency remain unmeasured.

| Deleted execution | Replacement |
| --- | --- |
| Packed two-bit weights extracted by float conversion, repeated division, floor and subtraction | M06 compiles equivalent unsigned shifts and masks before the original scale/bias and multiply-accumulate; the packed storage and native output are unchanged |
| Normalization sum's per-stride shared-memory barriers within one SIMD group | M06 lowers the exact original descending sum to register shuffles and one final shared broadcast; the recorded native pipeline replaces the old reduction, without a new dispatch |
| Projection reduction's shared-memory stores, loads and barriers for strides contained in one SIMD group | M06 compiles register shuffles for those strides, retaining the descending sum order and original cross-SIMD exchanges; recording replaces the native pipeline rather than adding a dispatch |
| WebGPU scratch zero stores and entry barrier before kernels that assign their own scratch | M06 configures `disable_workgroup_init` before compiling the native tensor program; original numerical scratch stores, reduction identities and data dependencies remain |
| Cable-loss detection depending only on a later CQ error or control-socket EOF | M24 subscribes to native data-link loss before pairing, using the existing control thread and M12 cancellation; no successful-crossing load or additional polling thread |
| Device-input completion publication from SEND preparation | Deleted; M07 is published only by its native receive completion, and M10 contains SEND destinations only |
| Standalone receive kernel, completion reset and subsequent numerical dispatch | M07 is directly bound inside the actual numerical entry; immutable completion words are prepared for each requested step, and M08 directly preposts the next request |
| Per-record provider-function load and repeated receive identity | Register-held provider function plus precomputed next-request binding in the same record line |
| Scan every layer's SEND cell on each empty probe | Register-held next cell, plus bounded independent-stream cursors |
| One native QP per SEND and receive transfer | Prepared FIFO per independent channel and invocation slot |
| Per-completion invocation decrement, generation advance and status update | No replacement; these values had no reader in the decode chain |
| RX completion-count branch and call | Direct completion store followed by native repost |
| RX thread stack-address publication and controller readiness spin | No replacement: M08 requests and M11 completion storage exist before thread creation; the stack address had no reader other than the deleted guard |
| TX completion `wr_id` decoding and completion-counter call | Native CQ drain and native error handling |
| Fixed 1 GiB registration extent | Setup derives extents from the provider's `max_mr_size` |
| Deleted generated-kernel and Swift-worker instruction claims | This table describes only current source and emitted native instructions |
| Per-step host command construction, completion wait, position/parameter/mask stores | M13 fixed boundary copies, prepared before the one interval submission |
| Separate prefill command buffer without an encoded KV dependency on decode | M14 prefill and decode ranges share M19's prepared compute sequence and native dispatch dependencies; no host wait or resource traversal |
| Metal 3 interval submission and implicit barriers on both sides of every ICB range | M19 native Metal 4 submission, with one encoded dependency at each numerical/boundary-copy cut; M20 brackets the measured decode steps after the first completed collective step, reporting their count explicitly, and M21 reports final retirement once |
| Exported cache-index temporaries across incompatible host/GPU physical layouts | Original expressions retained with consumers during graph preparation (M15) |
| Resource-array declaration at numerical submission | One prepared native residency set (M06) |
| Native command bindings unused by the reflected shader, and unoptimized repeated command state | M06 encodes only used binding slots and runs Apple’s native ICB optimizer during preparation, before transport starts; replay uses the complete optimized ranges |
| Exported split-half rotary slices, phase temporaries and separate arithmetic dispatches | M06 lowers the expression to upstream `odml.rope` before extraction; prepared sequence/head transposes and a sequence-shaped position operand match the native interface. The original expression remains the composite decomposition, including inverse layout presentation. A full geometric frequency vector or its prefix followed by zeros determines the native timescale and proportion only at preparation. |
| Separate native Metal numerical delegate and its queue-residency bookkeeping | Actual benchmark WebGPU provider; prepared Metal commands and canonical host-mapped operands |
| Vocabulary BITCAST, concatenation, slicing and candidate-array construction | M16 direct gather/pair store and direct MAXLOC comparison, bound before recording |
| Per-binding WebGPU allocation and separate numerical Metal aliases | One native GPU resource per M01 operand; typed LiteRT views and native commands share it |
| Mesh-owned numerical-provider handles and import wrappers | Native imports and typed views are realized by the engine compiler setup |
| Separate original prefix, FFN and finishing graph interfaces | One region between collective publication boundaries; original local edges stay inside upstream compilation |
| Unconditional barrier before every recorded numerical command | M06 compiles RAW/WAR/WAW conflicts from pipeline reflection and indirect-resource declarations; access sets are absent from execution |
| Separate publication buffer alias | The transport receives the existing canonical native buffer during setup; receive completion declares the external write to that resource |
| One Metal metadata resource aliasing independent SEND cells and receive completions | M07 and M17 expose disjoint, page-aligned parts of the same canonical mapping; publication no longer conflicts with completion observation merely because their addresses share an arena |
| GPU consumer waiting after native/control-link failure until a separate client signal | M12's existing cancellation operand is bound to `link_stop` during preparation; only the error branch records abandonment, checked by the caller after GPU retirement |
| Default WebGPU robustness and integer division/modulo polyfills in generated numerical code | Reference compiler toggles realized in M06 before compilation; native subgroup-matrix feature and limits exposed at setup |

The GPU handoff is inside the actual numerical kernel in the prepared indirect program. Entry-prefix compilation and selection happen only during recording; no compiler description is traversed at execution. Each requested step has a prepared native range with distinct completion bindings, while pipelines and numerical storage remain shared. The requested steps and M13 boundary copies are submitted together; numerical dispatches are still replayed, not across-step resident. An accepted CQ publishes the actual transport completion; no additional generation identity is computed. A new generation cannot reuse a layer contribution before both ranks have consumed the preceding step's dependent layer chain. This graph fact is not implemented as a reuse guard. Paired execution is recorded in D0; it does not establish the required speedup or crossing latency.

| Device boundary copy, source operations | D0 object |
| --- | --- |
| Read `state[0]`, `state[1]`: two `int4` values within one 32-byte record | M13 |
| Read token | M01 token operand |
| Store emitted token at the bound step output | M13 |
| Store position and two parameter vectors | M01 |
| Store prepared mask word at its bound offset | M01, M13 |

This boundary table states the shader source accesses, not an emitted GPU instruction count. It is outside both RDMA crossing transitions. After submission the host only observes completion of the requested interval and reads its outputs.

M22 diagnostic samples are native timestamp commands with indices and command-range cuts realized during preparation. `TRACE_CROSSINGS` samples one publication-to-first-numerical-consumer boundary per step and exposes the associated numerical-region duration; its precise samples perturb the sampled step and do not establish an H7 pass. With the option absent, no cut storage, subrange replay or additional timestamp is submitted; M20 remains the two endpoint samples.
