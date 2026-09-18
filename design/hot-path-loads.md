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
| Last tile stores completed generation, then direct publication values. No record or sequence load. | M04/M10/M18 | `metal-microbench/mesh_layer.swift:244` |
| Last vocabulary tile stores completed generation. | M34 | `metal-microbench/kernels.swift:4598` |
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
| Output visibility reads the existing scalar output word before its coherent store. | M38 | `swift/Mesh.swift:670` |
| Store completed generation, then fixed execution-completion event. | M35/M37 | `metal-microbench/mesh_layer.swift:352` |

| T1 instruction: ready publication → native SEND post | D0 row | Source |
| --- | --- | --- |
| `ldapr x8,[x23]`: publication ready word, with record address in a register. | M04 | `rdma/mesh-flow.c:195` |
| `str xzr,[x23]`: clear this accepted event before the native call. | M04 | `rdma/mesh-flow.c:196` |
| `ldp x0,x1,[x23,#8]`: native QP and WR from that same aligned 32-byte line. | M04 | `rdma/mesh-flow.c:197` |
| `add x2,sp,#8`; `blr x22`: stack failure output and register-held native post entry. | M06, M07, M15 | `rdma/mesh-flow.c:197` |
| `add/cmp/csel`: advance to the next record; base/end retained in registers, no cursor load or store. Empty and accepted events use this same advance. | register | `rdma/mesh-flow.c:200` |

| T2 instruction: native completion → availability publication | D0 row | Source |
| --- | --- | --- |
| `ldr w1,[x21,#8]`: native completion status. | M11 | `rdma/mesh-flow.c:247` |
| `ldr w9,[x21]`: completed canonical row integer. | M11 | `rdma/mesh-flow.c:248` |
| `ldr x8,[x8,#24]`: previous stamp at page base + 32*row +24; add register F. | M10 | `rdma/mesh-flow.c:250` |
| `stlr x8,[x10]`: publish availability at that same stamp address. | M10 | `rdma/mesh-flow.c:251` |

| RX continuation; inline publications precede native repost | D0 row | Source |
| --- | --- | --- |
| `mul/add`: record address from register base/stride and completion integer. | M08 | `rdma/mesh-flow.c:253` |
| `ldr w9,[x22,#56]`: inline instruction count. | M08 | `rdma/mesh-flow.c:255` |
| `ldp x11,x12,[x10,#-8]`: destination and multiplier in one aligned 32-byte instruction. | M13 | `rdma/mesh-flow.c:256` |
| `cmp/csinc`; `stlr x12,[x11]`: select generation or 1 without a branch, then store. | M04 or M18 | `rdma/mesh-flow.c:257` |
| `ldp x8,x0,[x22,#32]`: native RECV entry/QP; WR x22, bad-WR SP+8. | M08, M09, M25 | `rdma/mesh-flow.c:261` |
| `blr x8`: native repost after all declared publications. | M08 | `rdma/mesh-flow.c:261` |
| `ldr w8,[x22,#60]`: prepared execution-completion extent. No loop spill reloads before publication or repost. | M08 | `rdma/mesh-flow.c:263` |

| CPU completion publication and operand access | D0 row | Source |
| --- | --- | --- |
| `ldr w8,[x0,#64]`: inline instruction count. | M20 | `rdma/mesh-call.c:386` |
| `ldr w9,[x0,#40]`: invocation. | M20 | `rdma/mesh-call.c:385` |
| `ldp x12,x13,[x10,#-8]`: destination/multiplier at call+128+32*a. | M13 | `rdma/mesh-call.c:387` |
| `madd`; `stlr x13,[x12]`: prepared store; advance 32 bytes, then tail-branch to cleanup. No operand or publication-object traversal and no frame. | M04/M10/M18 | `rdma/mesh-call.c:388` |
| Read operand.data at +0, then numerical payload at the supplied offset. One metadata source access; caller-dependent Swift instruction count is not asserted. | M43 | `swift/Mesh.swift:13` |

| Link polling lifecycle outside accepted-event handoff | D0 row | Source |
| --- | --- | --- |
| TX `ldapr w8,[x8]`: link progressing word, only on empty publication cell. | M17 | `rdma/mesh-flow.c:209` |
| RX `ldapr w8,[x8]`: link progressing word on polling backedge. | M17 | `rdma/mesh-flow.c:243` |

| Input-preparation worker instructions; emitted AArch64, outside partial handoffs | D0 row | Source |
| --- | --- | --- |
| `ldapr x9,[x9]`: input generation at cell+8. | M18 | `rdma/mesh-call.c:142` |
| `str xzr,[x8,#8]`: clear event cell. | M18 | `rdma/mesh-call.c:144` |
| `ldp w10,w8,[x8]`: prepared call integer and invocation mask. | M18 | `rdma/mesh-call.c:146` |
| `madd x0,x23,x10,x22`: directly indexed call address using start-bound stride. | M20 | `rdma/mesh-call.c:146` |
| `ldp w11,w10,[x0,#36]`: pending count and generation. | M20 | `rdma/mesh-call.c:147` |
| `stp w9,w8,[x0,#36]`: dependency count/generation update. | M20 | `rdma/mesh-call.c:148` |
| `ldr w8,[x0,#68]`; `str w8,[x0,#36]`: restore the recurring dependency count before invoking the function. | M20 | `rdma/mesh-call.c:149` |
| `ldp x8,x20,[x0,#48]`; `blr x8`: entry/context in that same line. | M20 | `rdma/mesh-call.c:150` |
| `ldapr w8,[x8]`: call-group running word. No dispatch-loop stack reloads. | M24 | `rdma/mesh-call.c:137` |

| Prepared invocation publication and result; emitted AArch64 | D0 row | Source |
| --- | --- | --- |
| `ldp w9,w8,[x1,#100]`: stride and root count. | M41 | `rdma/mesh-call.h:19` |
| `ldr w0,[x1,#96]`: submitted generation. | M41 | `rdma/mesh-call.h:19` |
| `str w9,[x1,#96]`: advance generation; no slot calculation. | M41 | `rdma/mesh-call.h:20` |
| `ldr w12,[x10,x9,lsl #2]`: root range extent. | M41 | `rdma/mesh-call.h:21` |
| `ldr x13,[x1,x9,lsl #3]`: root range address; stamp offset 8. | M41 | `rdma/mesh-call.h:21` |
| `stlr x11,[x13]`: generation+1; advance 16 bytes with register extent. | M18 | `rdma/mesh-call.h:21` |
| `ldp x8,x9,[x3]`; `dmb ishld`: directly held result snapshot. Success and pending paths return without a frame. | M42 | `rdma/mesh-call.h:31`, `swift/Mesh.swift:328` |

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
| `reader->inputs` | register-held M04 scan |
| `reader->count` | register-held M04 end |
| `reader->cursor` | register-held M04 scan |
| `input->slots` | register-held M04 scan |
| `input->position` | register-held M04 scan |
| `input->mask` | register-held M04 end |
| `send_edges[first].end` | One M04 record per independently published partial |
| `source->span.addr` for a wire tag store | Deleted wire tag |
| `source->tag_row` | Deleted wire tag |
| `source->queue` | Native QP in M04 |
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
| CPU-arrival union/find and stream merging | Fixed M18 cells; device completion cells remain |
| Resident step notification/submit, M21 descriptors, mesh_metal_submit, meshMetalRearm, MeshResidentCommands | Deleted; setup-only body and M37 execution-completion publication |
| Vocabulary GPU publication record/sequence reads (3) | Compiled stores with register generation |
| Sampler descriptor and sequence reads (2) | Compiled payload/stamp bindings |
| Final per-step norm sequence read (1) | Resident generation |
| Native command submit loads (2), rearm selection/descriptor/ABI-restore loads (11) | Deleted with native decode rearm |


| Removed receive/device publication load site | Replacement |
| --- | --- |
| RX `m->target_off` | Prepared receive record; no receipt-time publication lookup |
| RX `m->target_stride` | Register-held start-bound M08 stride |
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
| M05 native `post`, `request`, `bad` (3) | Deleted intermediate dispatch; native operands now share the directly polled M04 record |
| M16 `cell`, `first`, `end` (3) | Direct record scan; no per-slot cursor |
| Input-worker target/call-group spill reloads (2 emitted sites, excluded from source deletion total) | Bases retained in registers after deleting the use table |

| Deleted structures / stores | Replacement |
| --- | --- |
| Per-slot `mesh_receive` arrays and retained `rows`/`count` | Flat canonical-row M08/M09 arrays; setup-only posting order is freed before start returns |
| GPU `target`, `publication` header, and `push` routine | The same 32-byte prepared-publication record for each destination |
| RX duplicate availability store through `mesh_publish` | One M10 store immediately after the native completion |
| GPU split high/low CPU-event stores and ring-position write | One prepared 64-bit store |
| Unread link `send_count` field and assignment | Deleted |
| M05 prepared_send allocation and per-slot failure-output heap cells | Native WR array and M15 stack output |
| M16 cursor allocation and updates; shared-QP cross-partial SEND order; temporary receive-order array | Each independent publication has a prepared QP and native fragment chain; directly polled M04 contains ready/QP/WR |
| Unread target index and queried-capacity array | Deleted; publication addresses and native queue extents fixed at setup |
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
| Native commands array, rearm closure and function entry/context, cold readiness reset | Single-use commands belong to the ordinary nonresident adapter invocation; M20 carries the recurring dependency count; resident decode remains one lifetime command |
| Lease-generation entry loads, restart reads, embedding claim/reload and entry acknowledgements | Deleted |

| Deleted CPU/RX publication and operand source loads | Replacement |
| --- | --- |
| CPU completion memory, operand array, input count and output count (4) | M20 invocation/store count and inline M13 instructions |
| Output publication pointer and page pointer (2) | Direct inline publication instructions |
| Publication sends and uses (2) | Prepared instruction count |
| Separate TX/CPU target stream and count reads (4) | One prepared instruction per actual destination |
| RX first ordinal and literal value (2) | Inline offset and constant 1; end/count is retained |
| CPU load transport quantum and indirect page address (2) | Direct contiguous M43 data pointer |
| RX optional-array base/generation reloads (emitted sites, excluded from source deletion total) | Inline tail and register-held stride/generation |
| Duplicate Swift destination construction; mesh_publish; raw page accessor; completion_publication flag; receive_targets allocation/reallocation | One setup constructor, direct operands and inline programs |

| Count / construction boundary | Value |
| --- | --- |
| Loads listed | 116 sites: 25 emitted AArch64 loads, 1 direct CPU-operand source access and 90 GPU source-level accesses. Native: TX 2, RX handoff 3, RX continuation 4, link lifecycle 2, input-preparation entry 5, input lifecycle 1, prepared submit/result 5, CPU completion 3. GPU: embedding 7, FFN 14, attention 21, partial-consumption/finish 25, vocabulary 6, sampler 17. Atomic read-modify-write sites count once. Numerical weight/input/gather/reduction loops remain supplied arithmetic; these counts do not assert GPU ISA loads or spills. |
| Deleted load sites | Previous 101 + CPU publication 12 + RX first/value 2 + CPU operand quantum/page-address 2 = 117 source/previously enumerated sites. Emitted spill removals are excluded from this total. Destination, count and generation reads that remain are listed above. |
| Native entry depth T1/T2 | `0/1`: TX directly polls M04 and reads native operands in the same line; RX completion integer → M10/M08, with M13 at fixed inline offsets. CPU M18 → M20, with M13 at fixed inline offsets. Provider internals are outside this ABI depth. |
| ABI frames | Input worker 96 bytes, FP=SP+80, no dispatch-loop spills. TX 80 bytes, FP=SP+64, bad-WR SP+8. RX 112 bytes, FP=SP+96, bad-WR SP+8, no hot-loop spills. CPU completion has no frame and tail-branches to cold cleanup. |
| Execution completion | Per-buffer references, consumed-input walks, output-return cells, cyclic resets, the reference CAS helper and the transport retirement thread are deleted. M37 reports a completed resident function. M42 counts actual numerical/transport completions for result reporting; no completion releases storage. RX result reporting follows publication and native repost. |
| Construction / compilation | 37/37 listed objects constructed; M05, M12, M14, M16, M19, M21, M44 retired. C/Swift, engine and four chain clients compile. Generated MSL changes were read, not dispatched or runtime-compiled. Numerical scope spans the whole step. Decode uses one setup command, no per-step command allocation and no renewal. Construction count does not establish the remaining whole-graph lifetime conditions below. |

| Governing invariant on the decode submission path | Source result |
| --- | --- |
| I2 | Completed-partial publication writes the prepared TX/stamp destinations without waits. CPU completion writes inline M13 instructions and tail-calls result accounting. RX writes availability and all CPU/forward destinations before reposting. |
| I4 | Submission modulo, availability admission and bank/lease renewal are deleted. Independent handles bind disjoint storage at setup. Whole-graph reuse closure remains unresolved: fixed receive backing is immediately reposted, and root completion alone has not established remote sampler completion before the next generation. Canonical backing has program lifetime; no per-execution buffer reference or reset remains. |
| I17 | M18 directly indexes M20; M04 directly supplies native QP/WR. Generic CPU operands contain direct contiguous pointers. CPU/RX publication instructions occupy fixed inline offsets and use the same setup constructor as GPU bindings. |
| I18 | Independent M04 queues post ready partials without cross-partial ordering. RX CPU/forward publications precede native repost. Local-first and FFN down-completion gates stay deleted. Numerical readiness resets before invocation; result accounting does not admit the next call. The software availability-stamp relay remains and is incompatible with the operator’s latest requirement; native CQ polling remains required. |
| I22 | Direct M04, M08/M13, M18/M20, M41/M42 and M43 records replace pointer-chased publication/operand objects. This records the constructed paths; it does not establish every numerical caller's metadata footprint. |

| Removed per-execution storage mechanism | Source replacement |
| --- | --- |
| Buffer retain/release/reset and cyclic reference CAS | Whole-program arena ownership; detach after native aliases are released |
| Consumed-input index array and per-input/per-output release loops | Direct M20 execution result and worker-completion destinations |
| Per-output return streams and output-refzero countdown | Function completion reports once, after numerical output publication |
| Link return readers, buffer-frame reconstruction and retirement pthread | Prebound M08 execution-completion destination; SEND CQ remains drained |
| Submit's result-pointer load and invocation store | M42's start-bound sequence advances on actual step completion |
| Queue arenas reserved for buffer returns | Removed from the shared-memory layout |
