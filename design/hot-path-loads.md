# Constructed transitions

| Resident embedding access; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Dispatch entry: load width at +12. | M26 | `metal-microbench/kernels.swift:105` |
| Dispatch entry: load slot stride at +16. | M26 | `metal-microbench/kernels.swift:105` |
| Dispatch entry: load embedding scale at +20. | M26 | `metal-microbench/kernels.swift:106` |
| Dispatch entry: load completed generation at +0; add register stride. | M26 | `metal-microbench/kernels.swift:108` |
| Lane 0 loads explicit shutdown word at +0; SIMD broadcast. | M40 | `metal-microbench/kernels.swift:111` |
| Lane 0 loads the directly bound input generation; SIMD broadcast. | M10 | `metal-microbench/kernels.swift:113` |
| Read token from the directly bound canonical input, rank 0 byte offset 12 and other ranks byte offset 0. | M01/M02 | `metal-microbench/kernels.swift:115`, binding at `metal-microbench/mesh_layer.swift:33` |
| Store embedding components into the fixed canonical hidden operand. | M03 | `metal-microbench/kernels.swift:117` |
| After the numerical stores and fence, lane 0 stores completed generation at +0; advance generation in registers. | M26 | `metal-microbench/kernels.swift:120` |

| Resident FFN global control accesses; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Stage entry loads gate-done at +4. | M27 | `metal-microbench/kernels.swift:1635` |
| Operand poll loads the explicit shutdown word at M40 +0. | M40 | `metal-microbench/kernels.swift:1641` |
| Operand poll loads input generation at +16. | M27 | `metal-microbench/kernels.swift:1641` |
| Operand poll loads completed generation at +24. | M27 | `metal-microbench/kernels.swift:1641` |
| Numerical normalization first reads the bound input payload. | M03 | `metal-microbench/kernels.swift:1647` |
| Gate loop loads gate-done at +4. | M27 | `metal-microbench/kernels.swift:1656` |
| Gate loop loads explicit shutdown word at M40 +0. | M40 | `metal-microbench/kernels.swift:1657` |
| Gate claim loads gate-next at +0. | M27 | `metal-microbench/kernels.swift:1659` |
| Gate claim compare/exchange reads and writes gate-next at +0. | M27 | `metal-microbench/kernels.swift:1660` |
| Completed numerical gate tile increments gate-done at +4. | M27 | `metal-microbench/kernels.swift:1683` |
| Down loop loads down-done at +12. | M27 | `metal-microbench/kernels.swift:1690` |
| Down loop loads explicit shutdown word at M40 +0. | M40 | `metal-microbench/kernels.swift:1691` |
| Down claim loads down-next at +8. | M27 | `metal-microbench/kernels.swift:1693` |
| Down claim compare/exchange reads and writes down-next at +8. | M27 | `metal-microbench/kernels.swift:1694` |
| Completed numerical down tile increments down-done at +12 before direct publication. | M27 | `metal-microbench/kernels.swift:1712` |
| Final tile stores completed generation at +24 after publication. | M27 | `metal-microbench/kernels.swift:1714` |

| Resident attention global control accesses; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Input poll loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3015` |
| Input poll loads completed generation. | M28 | `metal-microbench/kernels.swift:3015` |
| Input poll loads embedding completion or the fixed layer input generation. | M26 or M28 | `metal-microbench/kernels.swift:3015` |
| Normalization reads its first bound hidden component. | M03 | `metal-microbench/kernels.swift:3022` |
| Projection claim loads next column tile. | M28 | `metal-microbench/kernels.swift:3028` |
| Projection loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3029` |
| Projection claim compare/exchange reads and advances next tile. | M28 | `metal-microbench/kernels.swift:3030` |
| Completed column tile increments its head's count. | M29 | `metal-microbench/kernels.swift:3053` |
| Last tile stores head readiness after norm/RoPE/KV work. | M29 | `metal-microbench/kernels.swift:3078` |
| Split claim loads next split ordinal. | M28 | `metal-microbench/kernels.swift:3086` |
| Split loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3087` |
| Split claim loads its query-group readiness. | M29 | `metal-microbench/kernels.swift:3090` |
| Split claim loads the bound KV producer head readiness. | M29 | `metal-microbench/kernels.swift:3091` |
| Split claim compare/exchange reads and advances next ordinal. | M28 | `metal-microbench/kernels.swift:3092` |
| Completed split increments its query-group count. | M29 | `metal-microbench/kernels.swift:3109` |
| Last split stores reduced generation after the existing numerical reduction. | M29 | `metal-microbench/kernels.swift:3117` |
| Output-input poll loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3125` |
| Output-input poll reads each query group's reduced generation once before the output tile loop. | M29 | `metal-microbench/kernels.swift:3127` |
| Output tile loop loads completed tile count. | M28 | `metal-microbench/kernels.swift:3137` |
| Output tile loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3138` |
| Output claim loads next column tile. | M28 | `metal-microbench/kernels.swift:3140` |
| Output claim compare/exchange reads and advances next tile. | M28 | `metal-microbench/kernels.swift:3141` |
| Completed output tile increments output-done. | M28 | `metal-microbench/kernels.swift:3154` |
| Last tile stores FFN input generation after post-attention norm/residual output. | M27 | `metal-microbench/kernels.swift:3162` |
| Last tile stores attention completion generation. | M28 | `metal-microbench/kernels.swift:3162` |

| Resident partial consumption and layer finishing; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Entry loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:1744` |
| Entry loads completed generation. | M33 | `metal-microbench/kernels.swift:1744` |
| Local partial poll loads explicit shutdown word. | M40 | `metal-microbench/mesh_layer.swift:181` |
| Local partial poll loads its literal availability word. | M10 | `metal-microbench/mesh_layer.swift:181` |
| Read first local contribution directly into numerical threadgroup scratch. | M01 | `metal-microbench/mesh_layer.swift:187` |
| Remote partial poll loads explicit shutdown word. | M40 | `metal-microbench/mesh_layer.swift:194` |
| Read literal peer stamp; peers are unrolled at setup. | M10 | `metal-microbench/mesh_layer.swift:170` |
| Read first peer contribution through the literal payload pointer in matrixAdd. | M02 | `metal-microbench/mesh_layer.swift:173` |
| Without PLE, output claim loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:1760` |
| Without PLE, output claim compare/exchange reads and advances down-next. | M33 | `metal-microbench/kernels.swift:1760` |
| Without PLE, claimed threadgroup stores the finished hidden components. | M03 | `metal-microbench/kernels.swift:1767` |
| Without PLE, claimed threadgroup stores next-layer input generation, then completion. | M28/M33 | `metal-microbench/kernels.swift:1770` |
| PLE input loop loads completed tiles/final-normalization count. | M33 | `metal-microbench/kernels.swift:1780` |
| PLE input loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:1781` |
| PLE input claim loads next tile. | M33 | `metal-microbench/kernels.swift:1783` |
| PLE input claim compare/exchange reads and advances next tile. | M33 | `metal-microbench/kernels.swift:1784` |
| Completed input tile increments input-done. | M33 | `metal-microbench/kernels.swift:1799` |
| Read canonical token for the supplied PLE embedding calculation. | M01/M02 | `metal-microbench/kernels.swift:1803` |
| Last input tile increments input-done again after normalized/combined input stores. | M33 | `metal-microbench/kernels.swift:1812` |
| Gate loop loads completed tiles. | M33 | `metal-microbench/kernels.swift:1820` |
| Gate loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:1821` |
| Gate claim loads next tile. | M33 | `metal-microbench/kernels.swift:1823` |
| Gate claim compare/exchange reads and advances next tile. | M33 | `metal-microbench/kernels.swift:1824` |
| Completed gate tile increments gate-done. | M33 | `metal-microbench/kernels.swift:1839` |
| Down loop loads completed tiles. | M33 | `metal-microbench/kernels.swift:1845` |
| Down loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:1846` |
| Down claim loads next tile. | M33 | `metal-microbench/kernels.swift:1848` |
| Down claim compare/exchange reads and advances next tile. | M33 | `metal-microbench/kernels.swift:1849` |
| Completed down tile increments down-done before final norm/residual/scale. | M33 | `metal-microbench/kernels.swift:1861` |
| Last down tile stores finished hidden components. | M03 | `metal-microbench/kernels.swift:1867` |
| Last down tile stores next-layer input generation, then completed generation. | M28/M33 | `metal-microbench/kernels.swift:1871` |

| Resident vocabulary and sampler accesses; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Vocabulary loads completed tile count. | M34 | `metal-microbench/kernels.swift:4575` |
| Vocabulary loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:4576` |
| Vocabulary claim loads next tile. | M34 | `metal-microbench/kernels.swift:4578` |
| Vocabulary reads fixed final-layer availability. | M33 | `metal-microbench/kernels.swift:4579` |
| Vocabulary claim compare/exchange reads and advances next tile. | M34 | `metal-microbench/kernels.swift:4579` |
| Completed vocabulary tile increments done. | M34 | `metal-microbench/kernels.swift:4598` |
| Last tile directly stores publication values; no record or sequence load. | M04/M10/M18 | `metal-microbench/mesh_layer.swift:246` |
| Last vocabulary tile stores completed generation. | M34 | `metal-microbench/kernels.swift:4600` |
| Partial loop loads explicit shutdown word. | M40 | `metal-microbench/mesh_layer.swift:315` |
| Partial loop loads completed partial count. | M35 | `metal-microbench/mesh_layer.swift:315` |
| Scope claim loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:4620` |
| Scope claim loads its compiled local/remote stamp. | M10 | `metal-microbench/kernels.swift:4620` |
| Scope claim loads next numerical tile. | M36 | `metal-microbench/kernels.swift:4621` |
| Scope claim compare/exchange reads and advances next tile. | M36 | `metal-microbench/kernels.swift:4622` |
| Read first bound logit element; no operand record. | M01/M02 | `metal-microbench/kernels.swift:4630` |
| Completed numerical partial/capture tile increments done. | M35 | `metal-microbench/kernels.swift:4635` |
| Mass loop loads explicit shutdown word. | M40 | `metal-microbench/mesh_layer.swift:324` |
| Mass loop loads done count. | M35 | `metal-microbench/mesh_layer.swift:324` |
| Mass claim loads next numerical tile. | M35 | `metal-microbench/mesh_layer.swift:326` |
| Mass claim compare/exchange reads and advances next tile. | M35 | `metal-microbench/mesh_layer.swift:327` |
| Completed mass tile increments mass-done. | M35 | `metal-microbench/mesh_layer.swift:336` |
| Final claim loads explicit shutdown word. | M40 | `metal-microbench/mesh_layer.swift:342` |
| Final claim loads completed generation. | M35 | `metal-microbench/mesh_layer.swift:342` |
| Final claim compare/exchange reads and advances finish. | M35 | `metal-microbench/mesh_layer.swift:342` |
| Output visibility reads the existing scalar output word before its coherent store. | M38 | `swift/Mesh.swift:676` |
| Store completed generation, then fixed retirement event. | M35/M37 | `metal-microbench/mesh_layer.swift:354` |

| T1 instruction: cursor → native SEND acceptance / advance | D0 row | Source |
| --- | --- | --- |
| `ldr x25,[x24]`: fixed slot's current cell pointer. | M16 | `rdma/mesh-flow.c:235` |
| `ldapr x8,[x25]`: publication cell. | M04 | `rdma/mesh-flow.c:236` |
| `ldp x9,x0,[x8,#-32]`: native entry/QP. | M05 | `rdma/mesh-flow.c:241` |
| `ldp x1,x2,[x8,#-16]`: WR/failure output. | M05, M06, M15 | `rdma/mesh-flow.c:241` |
| `blr x9`: native post. | M05–M07, M15 | `rdma/mesh-flow.c:242` |
| `stlr xzr,[x25]`: accepted cell clear. | M04 | `rdma/mesh-flow.c:247` |
| `ldr x9,[x24,#16]`: slot end. | M16 | `rdma/mesh-flow.c:248` |
| `ldr x8,[x24,#8]`: slot first, only on wrap. | M16 | `rdma/mesh-flow.c:248` |
| `str x8,[x24]`: slot cursor advance. | M16 | `rdma/mesh-flow.c:248` |

| T2 instruction: native completion → availability publication | D0 row | Source |
| --- | --- | --- |
| `ldr w1,[x22,#8]`: native completion status. | M11 | `rdma/mesh-flow.c:289` |
| `ldr w8,[x22]`: completed canonical row integer. | M11 | `rdma/mesh-flow.c:290` |
| `ldr x28,[x9,#24]`: previous stamp at page base + 32*row + 24; add register F. | M10 | `rdma/mesh-flow.c:292` |
| `stlr x9,[x10]`: publish availability at that same stamp address. | M10 | `rdma/mesh-flow.c:293` |

| RX continuation after availability publication | D0 row | Source |
| --- | --- | --- |
| `ldp x8,x0,[x23,#32]`: prepared native RECV post entry and QP; WR is x23, bad-WR output is SP+24. | M08, M09, M25 | `rdma/mesh-flow.c:297` |
| `blr x8`: repost prepared WR; payload consumer can already read M10/M02. | M08, M09 | `rdma/mesh-flow.c:297` |
| `ldp w10,w11,[x23,#52]`: compiled destination range. | M08 | `rdma/mesh-flow.c:300` |
| `ldp x9,x8,[sp,#8]`: optional destination base/generation offset at RFP-104/-96. | M25 | `rdma/mesh-flow.c:301` |
| `ldp x11,x12,[x9,#-16]`: destination and value from flat record. | M13 | `rdma/mesh-flow.c:301` |
| `ldr x13,[x9],#32`: multiplier; advance to next consecutive record, no linked traversal. | M13 | `rdma/mesh-flow.c:303` |
| `stlr x12,[x11]`: fixed CPU/forward destination after multiply/add in registers. | M04 or M18 | `rdma/mesh-flow.c:302` |
| `ldr w23,[x23,#48]`: cold retirement row. | M08 | `rdma/mesh-flow.c:306` |

| Link polling lifecycle outside accepted-event handoff | D0 row | Source |
| --- | --- | --- |
| TX `ldapr w8,[x8]`: link progressing word, only on empty publication cell. | M17 | `rdma/mesh-flow.c:238` |
| RX `ldapr w8,[x8]`: link progressing word on polling backedge. | M17 | `rdma/mesh-flow.c:285` |

| Input-preparation worker instructions; emitted AArch64, outside partial handoffs | D0 row | Source |
| --- | --- | --- |
| `ldapr x9,[x9]`: prepared input event cell. | M18 | `rdma/mesh-call.c:177` |
| `str xzr,[x23,x8]`: clear event cell. | M18 | `rdma/mesh-call.c:179` |
| `ldr w9,[x28,#28]`: end. | M19 | `rdma/mesh-call.c:181` |
| `ldr x0,[x28,#16]`: call. | M19 | `rdma/mesh-call.c:183` |
| `ldr w8,[x28,#24]`: mask. | M19 | `rdma/mesh-call.c:184` |
| `ldp w11,w10,[x0,#32]`: pending/sequence. | M20 | `rdma/mesh-call.c:184` |
| `stp w9,w8,[x0,#32]`: pending/sequence update. | M20 | `rdma/mesh-call.c:185` |
| `ldp x8,x20,[x28]`: CPU entry/context, followed by branch. | M19 | `rdma/mesh-call.c:186` |
| `ldr x22,[sp,#24]`: target base at FP-88 on probe backedge. | M25 | `rdma/mesh-call.c:174` |
| `ldr x19,[sp,#16]`: call-group base at FP-96 before cold drain. | M25 | `rdma/mesh-call.c:191` |
| `ldapr w8,[x8]`: call-group running word. | M24 | `rdma/mesh-call.c:171` |

| Removed TX source load site | Replacement |
| --- | --- |
| `reader->inputs` | M16 register cursor |
| `reader->count` | M16 register end |
| `reader->cursor` | M16 register cursor |
| `input->slots` | M16 register cursor |
| `input->position` | M16 register cursor |
| `input->mask` | M16 register end |
| `send_edges[first].end` | One M04 cell per native request |
| `source->span.addr` for a wire tag store | Deleted wire tag |
| `source->tag_row` | Deleted wire tag |
| `source->queue` | Native QP already in M05 |
| Retry `ready->head` | Deleted retry queue |
| Retry `ready->tail` | Deleted retry queue |
| Retry `ready->first` | Deleted retry queue |
| Retry `ready->mask` | Deleted retry queue |
| Retry range `first` | M16 cursor stays on an unaccepted request |
| Retry range `end` | Deleted retry range |
| Retry range `invocation` | Deleted wire tag |

| Removed numerical arrival / rearm work | Replacement |
| --- | --- |
| `arrivals.inputs` | M18 fixed shared base |
| Per-input `slots` | Affine M18 address |
| Per-input `position` load | Register ordinal |
| Per-input `mask` load | Fixed one-cell stream |
| Per-input `position` store | Register ordinal |
| CPU-arrival union/find and stream merging | Fixed M18 cells; retirement grouping remains |
| Resident step notification/submit, M21 descriptors, mesh_metal_submit, meshMetalRearm, MeshResidentCommands | Deleted; setup-only body and M37 retirement publication |
| Vocabulary GPU publication record/sequence reads (3) | Compiled stores with register generation |
| Sampler descriptor and sequence reads (2) | Compiled payload/stamp bindings |
| Final per-step norm sequence read (1) | Resident generation |
| Native command submit loads (2), rearm selection/descriptor/ABI-restore loads (11) | Deleted with native decode rearm |


| Removed receive/device publication load site | Replacement |
| --- | --- |
| RX `m->target_off` | Prepared receive record; no receipt-time publication lookup |
| RX `m->target_stride` | Fixed 64-byte M08 stride |
| RX `provider.queues` | QP/post entry in M08 |
| RX per-slot `receive[slot].requests` | WR inline in M08 |
| RX forwarding `targets[i].count` | Each native TX cell already has its own M13 record |
| RX CPU-event `stream->position` | Fixed M18 cell |
| RX CPU-event `stream->mask` | Fixed M18 cell |
| FFN publication record load in participating threads | Compiled store operands in resident FFN |
| FFN publication overflow-loop record load | Compiled store operands in resident FFN |
| GPU publication header `counts.x` | Function constant 79 |
| GPU publication header `counts.y` | Function constant 78 minus 79 |
| GPU publication header `sequence` pointer | Native sequence buffer argument |
| GPU send record `count` | One M13 record per native request |
| GPU CPU-event `destination.position` pointer | Fixed destination in M13 |
| GPU CPU-event `*destination.position` | No reservation or position update |
| GPU CPU-event `destination.mask` | Fixed destination in M13 |

| FFN remote payload address in `mesh_add` | Compiled literal operand in the resident consumer |
| FFN remote stamp address in `mesh_add` | Compiled literal M10 address |
| FFN consumer invocation sequence | Resident generation register |
| Layer-finish publisher invocation sequence | Resident generation register |

| Deleted structures / stores | Replacement |
| --- | --- |
| Per-slot `mesh_receive` arrays and retained `rows`/`count` | Flat canonical-row M08/M09 arrays; setup-only posting order is freed before start returns |
| GPU `target`, `publication` header, and `push` routine | The same 32-byte prepared-publication record for each destination |
| RX duplicate availability store through `mesh_publish` | One M10 store immediately after the native completion |
| GPU split high/low CPU-event stores and ring-position write | One prepared 64-bit store |
| Unread link `send_count` field and assignment | Deleted |
| Per-slot producer allocations | One allocation per operand, fixed slot stride |
| Per-step decode embedding and scale dispatches | Resident `mesh_embed_resident`; neither launch remains in the step recording |
| Per-step mesh decode attention recording, QKV, norm/RoPE, KV scatter, attention/reduction, output projection and post-attention residual dispatches | Resident attention stages share the persistent FFN dispatch; native and resident entries call the same RoPE/KV/attention/reduction arithmetic |
| Per-step post-attention FFN-input publisher | Direct M27.input store by the resident attention producer |
| Per-step mesh decode FFN program construction and dispatches | Resident `mesh_ffn_resident` across all 35 layers; existing dot/quantization routines reused |
| FFN collective cohere pass and publication traversal | Coherent numerical output stores followed by direct compiled publication stores |
| Per-step `meshDecodeLayerProgram`, `mesh_add`, and layer-finish output publisher | Resident partial sum, post-FFN norm/residual, PLE and scale; direct M28 input publication |
| Decode `PerLayerInputs.begin` and its private input/embedding buffers | Resident PLE projection/embedding into canonical M03 input storage |
| Unused decode normalized-attention section, global FFN-sum section and FFN M12 argument table | Removed; numerical sum uses existing threadgroup scratch |
| Embedding/QKV/final-norm FC80 handoffs | Removed; all decode numerical stages are resident |
| Per-step final norm, unembedding, mesh_signal/coherent dispatches and samplingCommands call | Resident vocabulary and sampler; native callers share the numerical helpers |
| Sampler prepared_arguments, sampling_span and sampling_values | Deleted; literal bindings in the resident program |
| Renewal, two banks, lease observation, ICB replay, timers, lifecycle traces and probe switch | Deleted; one setup-time command and explicit shutdown word |
| Uninstantiated `MeshCommands` class | Deleted |
| Whole-program entered counter and entry flags | Deleted |
| Lease-generation entry loads, restart reads, embedding claim/reload and entry acknowledgements | Deleted |

| Count / construction boundary | Value |
| --- | --- |
| Loads listed | 119 sites: 26 emitted AArch64 loads and 93 GPU source-level accesses. Native: TX 6, RX handoff 3, RX continuation 6, link lifecycle 2, input-preparation entry 6, input lifecycle/spills 3. GPU: embedding 7, FFN 15, attention 21, partial-consumption/finish 27, vocabulary 6, sampler 17. Atomic read-modify-write sites count once. Numerical weight/input/gather/reduction loops remain supplied arithmetic; these counts do not assert GPU ISA loads or spills. |
| Deleted load sites | Prior 44 + vocabulary/sampler/final-norm metadata sites 6 + native command-submit/rearm loads 13 + startup retirement-enable 1 + whole-program entry exchange/counter 2 + lease-generation/restart/embedding-reuse loads 7 = 73. Replaced availability/payload reads are not counted as deleted. Compiler reassignment of input-worker spills is not a structural deletion. |
| Native entry depth T1/T2 | `2/1`: TX cursor → cell → dispatch; RX completion integer → M10 stamp. Device payload/stamp bindings and publication destinations are compiled literals. Native provider internals are outside this ABI depth. |
| ABI frames | Input/retirement worker and RX worker each reserve 128 bytes with FP=SP+112 in emitted -O2 assembly. Input-worker target/call-group reloads at FP-88/-96. No meshMetalRearm frame exists. RX base/generation reload remains after availability publication. |
| Retirement | M37 is a dedicated prepared one-cell stream. The existing return reader selects its cold frame, performs refcount cleanup and later releases the invocation. It does not select, construct or submit a device command. |
| Construction / compilation | 37/37 listed objects constructed; M12, M14, M21 retired. C/Swift build passes. The generated MSL signatures were read; this revision has not been dispatched. Numerical scope spans the whole step. Decode uses one setup command, no per-step command allocation and no renewal. The remaining source violations below prevent claiming invariant satisfaction from this construction count. |

| Governing invariant on the decode submission path | Source result |
| --- | --- |
| I2 | Holds for completed-partial publication: after visibility, FFN/vocabulary write the compiled TX/stamp destinations and return; `mesh_publish` has stores/range walks and no wait, semaphore or poll. `kernels.swift:1714`, `kernels.swift:4598`, `mesh_layer.swift:60`, `rdma/mesh.h:172`. |
| I4 | Does not hold: `mesh_calls_submit` selects `index % extent` and checks `instance->available` before admitting an invocation (`rdma/mesh-call.c:465`). Numerical tile writes also retain CAS claims (`kernels.swift:1694`). Renewal removal does not remove those paths. |
| I17 | Does not hold end to end: bindings and ranges are prepared, but TX follows M16 -> M04 -> M05 (`rdma/mesh-flow.c:234`), and CPU use dispatch follows M19 -> M20 (`rdma/mesh-call.c:180`). Those exceed the one-dependent-record clause. |
| I18 | Does not hold: the layer-finish consumer waits for its local term before probing ready remote terms (`mesh_layer.swift:178`); TX can only inspect the current cell of each cursor (`rdma/mesh-flow.c:234`). Removing successor waits does not remove these ordering gates. |
| I22 | Does not hold: TX accesses cursor, event cell and send record; submission follows calls -> context -> memory and instance state (`rdma/mesh-flow.c:234`, `rdma/mesh-call.c:460`). The T1/T2 depth numbers cover those transitions, not all runtime decisions. |
