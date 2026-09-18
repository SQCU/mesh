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
| Operand poll loads the layer-wide completed generation at the first batch row’s +24. | M27 | `metal-microbench/kernels.swift:1641` |
| Numerical normalization first reads the bound input payload. | M03 | `metal-microbench/kernels.swift:1647` |
| Gate loop loads gate-done at +4. | M27 | `metal-microbench/kernels.swift:1656` |
| Gate loop loads explicit shutdown word at M40 +0. | M40 | `metal-microbench/kernels.swift:1657` |
| Gate claim loads gate-next at +0. | M27 | `metal-microbench/kernels.swift:1659` |
| Gate claim compare/exchange reads and writes gate-next at +0. | M27 | `metal-microbench/kernels.swift:1660` |
| Completed numerical gate tile increments gate-done at +4. | M27 | `metal-microbench/kernels.swift:1683` |
| Down loop loads explicit shutdown word at M40 +0. | M40 | `metal-microbench/kernels.swift:1691` |
| Down claim loads down-next at +8. | M27 | `metal-microbench/kernels.swift:1690` |
| Down claim compare/exchange reads and writes down-next at +8. | M27 | `metal-microbench/kernels.swift:1693` |
| Completed numerical down tile increments the layer-wide publication count at the first batch row’s +12. | M27 | `metal-microbench/kernels.swift:1709` |
| Final tile stores completed generation at +24 before direct publication, so no old-generation store follows publication. | M27 | `metal-microbench/kernels.swift:1711` |

| Resident attention global control accesses; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Input poll loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3012` |
| Input poll loads completed generation. | M28 | `metal-microbench/kernels.swift:3012` |
| Input poll loads embedding completion or the fixed layer input generation. | M26 or M28 | `metal-microbench/kernels.swift:3012` |
| Normalization reads its first bound hidden component. | M03 | `metal-microbench/kernels.swift:3019` |
| Projection claim loads next column tile. | M28 | `metal-microbench/kernels.swift:3025` |
| Projection loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3026` |
| Projection claim compare/exchange reads and advances next tile. | M28 | `metal-microbench/kernels.swift:3027` |
| Completed column tile increments its head's count. | M29 | `metal-microbench/kernels.swift:3050` |
| Last tile stores head readiness after norm/RoPE/KV work. | M29 | `metal-microbench/kernels.swift:3075` |
| Split claim loads next split ordinal. | M28 | `metal-microbench/kernels.swift:3083` |
| Split loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3084` |
| Split claim loads its query-group readiness. | M29 | `metal-microbench/kernels.swift:3087` |
| Split claim loads the bound KV producer head readiness. | M29 | `metal-microbench/kernels.swift:3088` |
| Split claim compare/exchange reads and advances next ordinal. | M28 | `metal-microbench/kernels.swift:3089` |
| Completed split increments its query-group count. | M29 | `metal-microbench/kernels.swift:3106` |
| Last split stores reduced generation after the existing numerical reduction. | M29 | `metal-microbench/kernels.swift:3114` |
| Output-input poll loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3122` |
| Output-input poll reads each query group's reduced generation once before the output tile loop. | M29 | `metal-microbench/kernels.swift:3124` |
| Output tile loop loads completed tile count. | M28 | `metal-microbench/kernels.swift:3134` |
| Output tile loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3135` |
| Output claim loads next column tile. | M28 | `metal-microbench/kernels.swift:3137` |
| Output claim compare/exchange reads and advances next tile. | M28 | `metal-microbench/kernels.swift:3138` |
| Completed output tile increments output-done. | M28 | `metal-microbench/kernels.swift:3151` |
| Last tile stores FFN input generation after post-attention norm/residual output. | M27 | `metal-microbench/kernels.swift:3159` |
| Last tile stores attention completion generation. | M28 | `metal-microbench/kernels.swift:3159` |

| Resident partial consumption and layer finishing; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Entry loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:1741` |
| Entry loads completed generation. | M33 | `metal-microbench/kernels.swift:1741` |
| Read the first available contribution directly into numerical threadgroup scratch. | M01/M02 | `metal-microbench/mesh_layer.swift:179` |
| Partial poll loads explicit shutdown word. | M40 | `metal-microbench/mesh_layer.swift:192` |
| Read literal local/peer stamp; all terms are unrolled at setup. | M10 | `metal-microbench/mesh_layer.swift:172` |
| Read each subsequent contribution through its literal payload pointer in matrixAdd. | M01/M02 | `metal-microbench/mesh_layer.swift:180` |
| Without PLE, output claim loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:1757` |
| Without PLE, output claim compare/exchange reads and advances down-next. | M33 | `metal-microbench/kernels.swift:1757` |
| Without PLE, claimed threadgroup stores the finished hidden components. | M03 | `metal-microbench/kernels.swift:1764` |
| Without PLE, claimed threadgroup stores next-layer input generation, then completion. | M28/M33 | `metal-microbench/kernels.swift:1767` |
| PLE input loop loads completed tiles/final-normalization count. | M33 | `metal-microbench/kernels.swift:1777` |
| PLE input loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:1778` |
| PLE input claim loads next tile. | M33 | `metal-microbench/kernels.swift:1780` |
| PLE input claim compare/exchange reads and advances next tile. | M33 | `metal-microbench/kernels.swift:1781` |
| Completed input tile increments input-done. | M33 | `metal-microbench/kernels.swift:1796` |
| Read canonical token for the supplied PLE embedding calculation. | M01/M02 | `metal-microbench/kernels.swift:1800` |
| Last input tile increments input-done again after normalized/combined input stores. | M33 | `metal-microbench/kernels.swift:1809` |
| Gate loop loads completed tiles. | M33 | `metal-microbench/kernels.swift:1817` |
| Gate loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:1818` |
| Gate claim loads next tile. | M33 | `metal-microbench/kernels.swift:1820` |
| Gate claim compare/exchange reads and advances next tile. | M33 | `metal-microbench/kernels.swift:1821` |
| Completed gate tile increments gate-done. | M33 | `metal-microbench/kernels.swift:1836` |
| Down loop loads completed tiles. | M33 | `metal-microbench/kernels.swift:1842` |
| Down loop loads explicit shutdown word at M40 +0. | M40 | `metal-microbench/kernels.swift:1691` |
| Down claim loads next tile. | M33 | `metal-microbench/kernels.swift:1845` |
| Down claim compare/exchange reads and writes down-next at +8. | M27 | `metal-microbench/kernels.swift:1693` |
| Completed down tile increments down-done before final norm/residual/scale. | M33 | `metal-microbench/kernels.swift:1858` |
| Last down tile stores finished hidden components. | M03 | `metal-microbench/kernels.swift:1864` |
| Last down tile stores next-layer input generation, then completed generation. | M28/M33 | `metal-microbench/kernels.swift:1868` |

| Resident vocabulary and sampler accesses; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Vocabulary loads completed tile count. | M34 | `metal-microbench/kernels.swift:4572` |
| Vocabulary loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:4573` |
| Vocabulary claim loads next tile. | M34 | `metal-microbench/kernels.swift:4575` |
| Vocabulary reads fixed final-layer availability. | M33 | `metal-microbench/kernels.swift:4576` |
| Vocabulary claim compare/exchange reads and advances next tile. | M34 | `metal-microbench/kernels.swift:4576` |
| Completed vocabulary tile increments done. | M34 | `metal-microbench/kernels.swift:4595` |
| Last tile directly stores publication values; no record or sequence load. | M04/M10/M18 | `metal-microbench/mesh_layer.swift:244` |
| Last vocabulary tile stores completed generation. | M34 | `metal-microbench/kernels.swift:4597` |
| Partial loop loads explicit shutdown word. | M40 | `metal-microbench/mesh_layer.swift:313` |
| Partial loop loads completed partial count. | M35 | `metal-microbench/mesh_layer.swift:313` |
| Scope claim loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:4617` |
| Scope claim loads its compiled local/remote stamp. | M10 | `metal-microbench/kernels.swift:4617` |
| Scope claim loads next numerical tile. | M36 | `metal-microbench/kernels.swift:4618` |
| Scope claim compare/exchange reads and advances next tile. | M36 | `metal-microbench/kernels.swift:4619` |
| Read first bound logit element; no operand record. | M01/M02 | `metal-microbench/kernels.swift:4627` |
| Completed numerical partial/capture tile increments done. | M35 | `metal-microbench/kernels.swift:4632` |
| Mass loop loads explicit shutdown word. | M40 | `metal-microbench/mesh_layer.swift:322` |
| Mass loop loads done count. | M35 | `metal-microbench/mesh_layer.swift:322` |
| Mass claim loads next numerical tile. | M35 | `metal-microbench/mesh_layer.swift:324` |
| Mass claim compare/exchange reads and advances next tile. | M35 | `metal-microbench/mesh_layer.swift:325` |
| Completed mass tile increments mass-done. | M35 | `metal-microbench/mesh_layer.swift:334` |
| Final claim loads explicit shutdown word. | M40 | `metal-microbench/mesh_layer.swift:340` |
| Final claim loads completed generation. | M35 | `metal-microbench/mesh_layer.swift:340` |
| Final claim compare/exchange reads and advances finish. | M35 | `metal-microbench/mesh_layer.swift:340` |
| Output visibility reads the existing scalar output word before its coherent store. | M38 | `swift/Mesh.swift:708` |
| Store completed generation, then fixed retirement event. | M35/M37 | `metal-microbench/mesh_layer.swift:352` |

| T1 instruction: cursor → native SEND acceptance / advance | D0 row | Source |
| --- | --- | --- |
| `ldr x26,[x25]`: current cell pointer. | M16 | `rdma/mesh-flow.c:227` |
| `ldapr x8,[x26]`: publication cell. | M04 | `rdma/mesh-flow.c:228` |
| `ldr x0,[x25,#24]`: QP from the same cursor line. | M16 | `rdma/mesh-flow.c:233` |
| `add x1,x24,x8,lsl #7`: native WR address; x24 retains X-128. | M06 | `rdma/mesh-flow.c:233` |
| `add x2,sp,#8`; `blr x22`: stack failure output and register-held native post. | M06, M07, M15 | `rdma/mesh-flow.c:233` |
| `stlr xzr,[x26]`: accepted cell clear. | M04 | `rdma/mesh-flow.c:235` |
| `ldr x9,[x25,#16]`: end. | M16 | `rdma/mesh-flow.c:236` |
| `ldr x8,[x25,#8]`: first, only on wrap. | M16 | `rdma/mesh-flow.c:236` |
| `str x8,[x25]`: cursor advance. | M16 | `rdma/mesh-flow.c:236` |

| T2 instruction: native completion → availability publication | D0 row | Source |
| --- | --- | --- |
| `ldr w1,[x22,#8]`: native completion status. | M11 | `rdma/mesh-flow.c:277` |
| `ldr w8,[x22]`: completed canonical row integer. | M11 | `rdma/mesh-flow.c:278` |
| `ldr x28,[x9,#24]`: previous stamp at page base + 32*row + 24; add register F. | M10 | `rdma/mesh-flow.c:280` |
| `stlr x9,[x10]`: publish availability at that same stamp address. | M10 | `rdma/mesh-flow.c:281` |

| RX continuation after availability publication | D0 row | Source |
| --- | --- | --- |
| `ldp x8,x0,[x23,#32]`: prepared native RECV post entry and QP; WR is x23, bad-WR output is SP+24. | M08, M09, M25 | `rdma/mesh-flow.c:285` |
| `blr x8`: repost prepared WR; payload consumer can already read M10/M02. | M08, M09 | `rdma/mesh-flow.c:285` |
| `ldp w10,w11,[x23,#52]`: compiled destination range. | M08 | `rdma/mesh-flow.c:288` |
| `ldp x9,x8,[sp,#8]`: optional destination base/generation offset at RFP-104/-96. | M25 | `rdma/mesh-flow.c:289` |
| `ldp x11,x12,[x9,#-16]`: destination and value from flat record. | M13 | `rdma/mesh-flow.c:289` |
| `ldr x13,[x9],#32`: multiplier; advance to next consecutive record, no linked traversal. | M13 | `rdma/mesh-flow.c:291` |
| `stlr x12,[x11]`: fixed CPU/forward destination after multiply/add in registers. | M04 or M18 | `rdma/mesh-flow.c:290` |
| `ldr w23,[x23,#48]`: cold retirement row. | M08 | `rdma/mesh-flow.c:294` |

| Link polling lifecycle outside accepted-event handoff | D0 row | Source |
| --- | --- | --- |
| TX `ldapr w8,[x8]`: link progressing word, only on empty publication cell. | M17 | `rdma/mesh-flow.c:230` |
| RX `ldapr w8,[x8]`: link progressing word on polling backedge. | M17 | `rdma/mesh-flow.c:273` |

| Input-preparation worker instructions; emitted AArch64, outside partial handoffs | D0 row | Source |
| --- | --- | --- |
| `ldapr x9,[x9]`: input generation at cell+8. | M18 | `rdma/mesh-call.c:168` |
| `str xzr,[x8,#8]`: clear event cell. | M18 | `rdma/mesh-call.c:170` |
| `ldp w10,w8,[x8]`: prepared call integer and invocation mask. | M18 | `rdma/mesh-call.c:172` |
| `add x0,x22,x10,lsl #7`: directly indexed call address. | M20 | `rdma/mesh-call.c:172` |
| `ldp w11,w10,[x0,#32]`: pending count and generation. | M20 | `rdma/mesh-call.c:173` |
| `stp w9,w8,[x0,#32]`: dependency count/generation update. | M20 | `rdma/mesh-call.c:174` |
| `ldp x8,x20,[x0,#72]`; `blr x8`: entry/context in that same line. | M20 | `rdma/mesh-call.c:175` |
| `ldapr w8,[x8]`: call-group running word. No dispatch-loop stack reloads. | M24 | `rdma/mesh-call.c:163` |

| Prepared invocation publication and result; emitted AArch64 | D0 row | Source |
| --- | --- | --- |
| `ldp w9,w8,[x1,#108]`: stride and root count. | M41 | `rdma/mesh-call.h:21` |
| `ldr w0,[x1,#104]`: submitted generation. | M41 | `rdma/mesh-call.h:21` |
| `str w9,[x1,#104]`: advance generation; no slot calculation. | M41 | `rdma/mesh-call.h:22` |
| `ldr x9,[x1,#96]`: invocation destination. | M41 | `rdma/mesh-call.h:23` |
| `str w0,[x9,#20]`: write invocation, without resetting result status. | M42 | `rdma/mesh-call.h:23` |
| `ldr w12,[x10,x9,lsl #2]`: root range extent. | M41 | `rdma/mesh-call.h:23` |
| `ldr x13,[x1,x9,lsl #3]`: root range address; stamp offset 8. | M41 | `rdma/mesh-call.h:23` |
| `stlr x11,[x13]`: generation+1; advance 16 bytes with register extent. | M18 | `rdma/mesh-call.h:23` |
| `ldp x8,x9,[x3]`; `dmb ishld`: directly held result snapshot. Success and pending paths return without a frame. | M42 | `rdma/mesh-call.h:34`, `swift/Mesh.swift:347` |

| Deleted submission/result and consumer load sites (20 source sites, not an ISA count) | Replacement |
| --- | --- |
| Submit `calls->context`, `context->M`, `context->client` (3) | Prepared M41 handle |
| Submit first and second global result loads (2) | Faults stay in M42; publication never resets them |
| Submit `root_workers`, `extent`, `instances`, `instance->available`, `first` (5) | M41 destinations and fixed M42 address |
| Submit `target_off`, `target_stride`, `page_off`, publication `sends` and `uses` (5) | Inline M41 root range addresses/counts |
| Result `calls->instances`, `calls->extent` (2) | Direct M42 result address |
| Local-first shutdown/presence poll (2) | One symmetric numerical reduction loop |
| FFN down-completion poll (1) | Continuation after work assignment; last actual completion publishes the full batch |

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
| `source->queue` | Native QP in M16 |
| Retry `ready->head` | Deleted retry queue |
| Retry `ready->tail` | Deleted retry queue |
| Retry `ready->first` | Deleted retry queue |
| Retry `ready->mask` | Deleted retry queue |
| Retry range `first` | No hot-path retry; native refusal concludes the affected execution |
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

| Removed direct-dispatch source loads | Replacement |
| --- | --- |
| M19 `end` and `call` (2) | Independent M18 cells carry the final M20 integer; no use range or call pointer |
| M05 native `post`, `request`, `bad` (3) | Thread-held native entry, affine WR address, stack-native output; QP shares M16 |
| Input-worker target/call-group spill reloads (2 emitted sites, excluded from source deletion total) | Bases retained in registers after deleting the use table |

| Deleted structures / stores | Replacement |
| --- | --- |
| Per-slot `mesh_receive` arrays and retained `rows`/`count` | Flat canonical-row M08/M09 arrays; setup-only posting order is freed before start returns |
| GPU `target`, `publication` header, and `push` routine | The same 32-byte prepared-publication record for each destination |
| RX duplicate availability store through `mesh_publish` | One M10 store immediately after the native completion |
| GPU split high/low CPU-event stores and ring-position write | One prepared 64-bit store |
| Unread link `send_count` field and assignment | Deleted |
| M05 prepared_send allocation and per-slot failure-output heap cells | Native WR array, existing M16 spare word and M15 stack output |
| M19 mesh_use, duplicate function entry/context, setup event union/remap/sort, publisher field | One M20 record and final M18 cells; cold return streams are constructed at their final addresses |
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
| Loads listed | 117 sites: 27 emitted AArch64 loads and 90 GPU source-level accesses. Native: TX 5, RX handoff 3, RX continuation 6, link lifecycle 2, input-preparation entry 4, input lifecycle 1, prepared submit/result 6. GPU: embedding 7, FFN 14, attention 21, partial-consumption/finish 25, vocabulary 6, sampler 17. Atomic read-modify-write sites count once. Numerical weight/input/gather/reduction loops remain supplied arithmetic; these counts do not assert GPU ISA loads or spills. |
| Deleted load sites | Previous 93 + M19 end/call 2 + M05 native-entry/request/failure-output 3 = 98 source/previously enumerated sites. Two additional emitted input-loop spills disappear but are excluded from this total. Relocated mask/QP reads and retained availability/payload reads are not counted as deleted. |
| Native entry depth T1/T2 | `1/1`: TX cursor → cell, affine native WR; RX completion integer → M10 stamp. No M05 dispatch descriptor. CPU M18 → M20; native entry and context are in that same M20 line. Provider internals are outside this ABI depth. |
| ABI frames | Input worker 112 bytes, FP=SP+96, no dispatch-loop spills. TX 96 bytes, FP=SP+80, bad-WR output SP+8. RX 128 bytes, FP=SP+112; publication base/generation reload remains after availability publication. No meshMetalRearm frame exists. |
| Retirement | M37 is a dedicated prepared one-cell stream. The existing return reader selects its cold frame, performs refcount cleanup and later releases the invocation. It does not select, construct or submit a device command. |
| Construction / compilation | 37/37 listed objects constructed; M05, M12, M14, M19, M21 retired. C/Swift build passes. The engine and four updated chain/example clients compile; generated MSL changes were read, not dispatched or runtime-compiled. Emitted submit/result code has no stack frame on success or pending. Numerical scope spans the whole step. Decode uses one setup command, no per-step command allocation and no renewal. The remaining source violations below prevent claiming invariant satisfaction from this construction count. |

| Governing invariant on the decode submission path | Source result |
| --- | --- |
| I2 | Holds for completed-partial publication: after visibility, FFN/vocabulary write the compiled TX/stamp destinations and return; `mesh_publish` has stores/range walks and no wait, semaphore or poll. `kernels.swift:1714`, `kernels.swift:4598`, `mesh_layer.swift:60`, `rdma/mesh.h:176`. |
| I4 | Submission modulo, availability state and admission are deleted. Independent handles bind disjoint storage at setup; reuse follows graph lifetimes. Full closure is still unproven: fixed receive backing is reposted immediately, and numerical work assignment still uses CAS. M41/M42 do not establish those lifetimes across arbitrary graphs. |
| I17 | M18 directly indexes the complete M20 call record; M04 directly indexes native WRs. M05/M19 and setup remapping are deleted. Generic CPU output publication still loads operand and publication metadata and must be flattened; no end-to-end claim yet. |
| I18 | The local-first poll and FFN down-completion gate are deleted. A free workgroup may consume a remote term while another finishes the local term. TX still inspects only the current cell of each cursor; the fixed native receive order and its independent-work implications remain to replace. |
| I22 | TX reads M16 plus the M04 queue cell; CPU dispatch reads M18 plus one M20 line. Submission/result read M41/M42 respectively. Generic CPU publication still follows call → operand → publication; that competing path prevents end-to-end closure. |
