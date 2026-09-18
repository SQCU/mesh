# Constructed transitions

| Resident embedding access; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Dispatch entry: load rows at +8. | M26 | `metal-microbench/kernels.swift:103` |
| Dispatch entry: load token offset at +24. | M26 | `metal-microbench/kernels.swift:103` |
| Dispatch entry: load width at +12. | M26 | `metal-microbench/kernels.swift:103` |
| Dispatch entry: load slot stride at +16. | M26 | `metal-microbench/kernels.swift:103` |
| Dispatch entry: load embedding scale at +20. | M26 | `metal-microbench/kernels.swift:104` |
| Dispatch entry: load completed generation at +0; add register stride. | M26 | `metal-microbench/kernels.swift:104` |
| Lane 0 loads explicit shutdown word at +0; SIMD broadcast. | M40 | `metal-microbench/kernels.swift:107` |
| Lane 0 loads the operand argument from its fixed input cell; SIMD broadcast. | M10 | `metal-microbench/kernels.swift:109` |
| Read token through the delivered operand pointer, rank 0 byte offset 12 and other ranks byte offset 0. | M01/M02 | `metal-microbench/kernels.swift:111`, binding at `metal-microbench/mesh_layer.swift:35` |
| Store embedding components into the fixed canonical hidden operand. | M03 | `metal-microbench/kernels.swift:113` |
| After the numerical stores and fence, lane 0 stores completed generation at +0; advance generation in registers. | M26 | `metal-microbench/kernels.swift:116` |

| Resident FFN global control accesses; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Lane 0 clears the consumed input cell after the batch. | M10 | `metal-microbench/kernels.swift:120` |

| Stage entry loads gate-done at +4. | M27 | `metal-microbench/kernels.swift:1634` |
| Operand poll loads the explicit shutdown word at M40 +0. | M40 | `metal-microbench/kernels.swift:1640` |
| Operand poll loads the producer attention completion directly. | M28 | `metal-microbench/kernels.swift:1640` |
| Operand poll loads the layer-wide completed generation at the first batch row’s +24. | M27 | `metal-microbench/kernels.swift:1640` |
| Numerical normalization first reads the bound input payload. | M03 | `metal-microbench/kernels.swift:1646` |
| Gate loop loads gate-done at +4. | M27 | `metal-microbench/kernels.swift:1655` |
| Gate loop loads explicit shutdown word at M40 +0. | M40 | `metal-microbench/kernels.swift:1656` |
| Gate claim loads gate-next at +0. | M27 | `metal-microbench/kernels.swift:1658` |
| Gate claim compare/exchange reads and writes gate-next at +0. | M27 | `metal-microbench/kernels.swift:1659` |
| Completed numerical gate tile increments gate-done at +4. | M27 | `metal-microbench/kernels.swift:1682` |
| Down loop loads explicit shutdown word at M40 +0. | M40 | `metal-microbench/kernels.swift:1690` |
| Down claim loads down-next at +8. | M27 | `metal-microbench/kernels.swift:1689` |
| Down claim compare/exchange reads and writes down-next at +8. | M27 | `metal-microbench/kernels.swift:1692` |
| Completed numerical down tile increments the layer-wide publication count at the first batch row’s +12. | M27 | `metal-microbench/kernels.swift:1708` |
| Final tile stores completed generation at +24 before direct publication, so no old-generation store follows publication. | M27 | `metal-microbench/kernels.swift:1710` |

| Resident attention global control accesses; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Input poll loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3006` |
| Input poll loads completed generation. | M28 | `metal-microbench/kernels.swift:3006` |
| Input poll loads embedding completion or preceding layer-finish completion directly. | M26 or M33 | `metal-microbench/kernels.swift:3006` |
| Normalization reads its first bound hidden component. | M03 | `metal-microbench/kernels.swift:3013` |
| Projection claim loads next column tile. | M28 | `metal-microbench/kernels.swift:3019` |
| Projection loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3020` |
| Projection claim compare/exchange reads and advances next tile. | M28 | `metal-microbench/kernels.swift:3021` |
| Completed column tile increments its head's count. | M29 | `metal-microbench/kernels.swift:3044` |
| Last tile stores head readiness after norm/RoPE/KV work. | M29 | `metal-microbench/kernels.swift:3069` |
| Split claim loads next split ordinal. | M28 | `metal-microbench/kernels.swift:3077` |
| Split loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3078` |
| Split claim loads its query-group readiness. | M29 | `metal-microbench/kernels.swift:3081` |
| Split claim loads the bound KV producer head readiness. | M29 | `metal-microbench/kernels.swift:3082` |
| Split claim compare/exchange reads and advances next ordinal. | M28 | `metal-microbench/kernels.swift:3083` |
| Completed split increments its query-group count. | M29 | `metal-microbench/kernels.swift:3100` |
| Last split stores reduced generation after the existing numerical reduction. | M29 | `metal-microbench/kernels.swift:3108` |
| Output-input poll loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3116` |
| Output-input poll reads each query group's reduced generation once before the output tile loop. | M29 | `metal-microbench/kernels.swift:3118` |
| Output tile loop loads completed tile count. | M28 | `metal-microbench/kernels.swift:3128` |
| Output tile loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:3129` |
| Output claim loads next column tile. | M28 | `metal-microbench/kernels.swift:3131` |
| Output claim compare/exchange reads and advances next tile. | M28 | `metal-microbench/kernels.swift:3132` |
| Completed output tile increments output-done. | M28 | `metal-microbench/kernels.swift:3145` |
| Last tile stores attention completion generation. | M28 | `metal-microbench/kernels.swift:3153` |

| Resident partial consumption and layer finishing; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Entry loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:1740` |
| Entry loads completed generation. | M33 | `metal-microbench/kernels.swift:1740` |
| Read the first available contribution directly into numerical threadgroup scratch. | M01/M02 | `metal-microbench/mesh_layer.swift:181` |
| Partial poll loads explicit shutdown word. | M40 | `metal-microbench/mesh_layer.swift:196` |
| Read local/remote operand pointer from the fixed input cell; probes are unrolled at setup. | M10 | `metal-microbench/mesh_layer.swift:174` |
| Read each subsequent contribution through its delivered pointer in matrixAdd. | M01/M02 | `metal-microbench/mesh_layer.swift:182` |
| Reduction/norm job claim compare/exchange reads and advances gate-next. | M33 | `metal-microbench/kernels.swift:1757` |
| Without PLE, claimed threadgroup stores the finished hidden components. | M03 | `metal-microbench/kernels.swift:1757` |
| Without PLE, reduction owner stores completion. | M33 | `metal-microbench/kernels.swift:1757` |
| With PLE, reduction owner increments gate-done after writing normalized/residual values into the existing output. | M33 | `metal-microbench/kernels.swift:1760` |
| Clear each consumed operand input after its final batch row. | M10 | `metal-microbench/mesh_layer.swift:183` |
| PLE input loop loads completed tiles/final-normalization count. | M33 | `metal-microbench/kernels.swift:1772` |
| PLE input loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:1773` |
| PLE input claim loads next tile. | M33 | `metal-microbench/kernels.swift:1775` |
| PLE input claim compare/exchange reads and advances next tile. | M33 | `metal-microbench/kernels.swift:1776` |
| Completed input tile increments input-done. | M33 | `metal-microbench/kernels.swift:1791` |
| Read canonical token for the supplied PLE embedding calculation. | M01/M02 | `metal-microbench/kernels.swift:1795` |
| Last input tile increments input-done again after normalized/combined input stores. | M33 | `metal-microbench/kernels.swift:1804` |
| Gate loop loads completed tiles. | M33 | `metal-microbench/kernels.swift:1812` |
| Gate loop loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:1813` |
| Gate claim loads next tile. | M33 | `metal-microbench/kernels.swift:1815` |
| Gate claim compare/exchange reads and advances next tile. | M33 | `metal-microbench/kernels.swift:1816` |
| Completed gate tile increments gate-done. | M33 | `metal-microbench/kernels.swift:1831` |
| Down loop loads completed tiles. | M33 | `metal-microbench/kernels.swift:1837` |
| Down loop loads explicit shutdown word at M40 +0. | M40 | `metal-microbench/kernels.swift:1690` |
| Down claim loads next tile. | M33 | `metal-microbench/kernels.swift:1840` |
| Down claim compare/exchange reads and writes down-next at +8. | M27 | `metal-microbench/kernels.swift:1692` |
| Completed down tile increments down-done before final norm/residual/scale. | M33 | `metal-microbench/kernels.swift:1853` |
| Last down tile stores finished hidden components. | M03 | `metal-microbench/kernels.swift:1859` |
| Last down tile stores numerical completion. | M33 | `metal-microbench/kernels.swift:1863` |

| Resident vocabulary and sampler accesses; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Vocabulary loads completed tile count. | M34 | `metal-microbench/kernels.swift:4566` |
| Vocabulary loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:4567` |
| Vocabulary claim loads next tile. | M34 | `metal-microbench/kernels.swift:4569` |
| Vocabulary reads final-layer numerical completion. | M33 | `metal-microbench/kernels.swift:4570` |
| Vocabulary claim compare/exchange reads and advances next tile. | M34 | `metal-microbench/kernels.swift:4570` |
| Completed vocabulary tile increments done. | M34 | `metal-microbench/kernels.swift:4589` |
| Last tile stores completed generation, then direct publication values. No record or sequence load. | M04/M10/M18 | `metal-microbench/mesh_layer.swift:247` |
| Last vocabulary tile stores completed generation. | M34 | `metal-microbench/kernels.swift:4592` |
| Partial loop loads explicit shutdown word. | M40 | `metal-microbench/mesh_layer.swift:316` |
| Partial loop loads completed partial count. | M35 | `metal-microbench/mesh_layer.swift:316` |
| Scope claim loads explicit shutdown word. | M40 | `metal-microbench/kernels.swift:4611` |
| Scope claim loads the operand pointer from its compiled input cell. | M10 | `metal-microbench/kernels.swift:4611` |
| Scope claim loads next numerical tile. | M36 | `metal-microbench/kernels.swift:4612` |
| Scope claim compare/exchange reads and advances next tile. | M36 | `metal-microbench/kernels.swift:4613` |
| Read first logit element through the delivered operand; no version comparison. | M01/M02 | `metal-microbench/kernels.swift:4622` |
| Completed numerical partial/capture tile increments done. | M35 | `metal-microbench/kernels.swift:4629` |
| Mass loop loads explicit shutdown word. | M40 | `metal-microbench/mesh_layer.swift:325` |
| Mass loop loads done count. | M35 | `metal-microbench/mesh_layer.swift:325` |
| Mass claim loads next numerical tile. | M35 | `metal-microbench/mesh_layer.swift:327` |
| Mass claim compare/exchange reads and advances next tile. | M35 | `metal-microbench/mesh_layer.swift:328` |
| Completed mass tile increments mass-done. | M35 | `metal-microbench/mesh_layer.swift:337` |
| Final claim loads explicit shutdown word. | M40 | `metal-microbench/mesh_layer.swift:343` |
| Final claim loads completed generation. | M35 | `metal-microbench/mesh_layer.swift:343` |
| Final claim compare/exchange reads and advances finish. | M35 | `metal-microbench/mesh_layer.swift:343` |
| Output visibility reads the existing scalar output word before its coherent store. | M38 | `swift/Mesh.swift:668` |
| Clear consumed input cells at their compiled addresses after the final sample/capture read. | M10 | `metal-microbench/mesh_layer.swift:305` |
| Store completed generation, then fixed execution-completion event. | M35/M37 | `metal-microbench/mesh_layer.swift:355` |

| T1 instruction: ready publication → native SEND post | D0 row | Source |
| --- | --- | --- |
| `ldapr x8,[x23]`: publication ready word, with record address in a register. | M04 | `rdma/mesh-flow.c:195` |
| `str xzr,[x23]`: clear this accepted event before the native call. | M04 | `rdma/mesh-flow.c:196` |
| `ldp x0,x1,[x23,#8]`: native QP and WR from that same aligned 32-byte line. | M04 | `rdma/mesh-flow.c:197` |
| `add x2,sp,#8`; `blr x22`: stack failure output and register-held native post entry. | M06, M07, M15 | `rdma/mesh-flow.c:197` |
| `add/cmp/csel`: advance to the next record; base/end retained in registers, no cursor load or store. Empty and accepted events use this same advance. | register | `rdma/mesh-flow.c:200` |

| T2 instruction: native completion → operand delivery | D0 row | Source |
| --- | --- | --- |
| `ldr w1,[x21,#8]`: native completion status. | M11 | `rdma/mesh-flow.c:236` |
| `ldr x22,[x21]`: directly bound receive-record pointer. | M11 | `rdma/mesh-flow.c:238` |
| `ldp x8,x9,[x22,#64]`: fixed input destination and literal operand. | M08 | `rdma/mesh-flow.c:239` |
| `stlr x9,[x8]`: deliver that operand. | M10 | `rdma/mesh-flow.c:239` |

| RX continuation after delivery | D0 row | Source |
| --- | --- | --- |
| `ldp x8,x0,[x22,#48]`: native RECV entry/QP. | M08 | `rdma/mesh-flow.c:241` |
| `add x2,sp,#8`; `mov x1,x22`; `blr x8`: native repost, no intervening lookup. | M08/M09/M25 | `rdma/mesh-flow.c:241` |
| `ldr w1,[x22,#88]`: actual transfer-completion extent. | M08 | `rdma/mesh-flow.c:243` |
| `ldr x0,[x22,#80]`: directly bound result destination. | M08/M42 | `rdma/mesh-flow.c:243` |

| CPU completion publication and operand access | D0 row | Source |
| --- | --- | --- |
| Load publication count at call+64. | M20 | `rdma/mesh-call.c:385` |
| Load destination and literal argument together at call+128+32*a. | M13 | `rdma/mesh-call.c:386` |
| Release-store that argument; advance 32 bytes. Tail-branch to result accounting after all stores. | M04/M10/M18 | `rdma/mesh-call.c:387` |
| Read operand.data at +0, then numerical payload. | M43 | `swift/Mesh.swift:11` |

| Link polling lifecycle outside accepted-event handoff | D0 row | Source |
| --- | --- | --- |
| TX `ldapr w8,[x8]`: link progressing word, only on empty publication cell. | M17 | `rdma/mesh-flow.c:209` |
| RX `ldapr w8,[x8]`: link progressing word on polling backedge. | M17 | `rdma/mesh-flow.c:237` |

| Input-preparation worker instructions; emitted AArch64, outside partial handoffs | D0 row | Source |
| --- | --- | --- |
| `ldapr x9,[x9]`: input argument at cell+8. | M18 | `rdma/mesh-call.c:142` |
| `str xzr,[x8,#8]`: clear event cell. | M18 | `rdma/mesh-call.c:144` |
| `ldr w8,[x8]`: prepared call integer. | M18 | `rdma/mesh-call.c:146` |
| `madd x0,x23,x8,x22`: directly indexed call address using start-bound stride. | M20 | `rdma/mesh-call.c:146` |
| `ldr w8,[x0,#36]`: pending input count. | M20 | `rdma/mesh-call.c:146` |
| `str w8,[x0,#36]`: decremented input count. | M20 | `rdma/mesh-call.c:147` |
| `ldr w8,[x0,#68]`; `str w8,[x0,#36]`: restore the recurring dependency count before invoking the function. | M20 | `rdma/mesh-call.c:148` |
| `ldp x8,x20,[x0,#48]`; `blr x8`: entry/context in that same line. | M20 | `rdma/mesh-call.c:149` |
| `ldr x8,[sp,#8]`; `add x8,x8,#1204`; `ldapr w8,[x8]`: saved call-group base then running word on the polling backedge. | M25/M24 | `rdma/mesh-call.c:136` |

| Resident execution completion; emitted AArch64 | D0 row | Source |
| --- | --- | --- |
| `ldapr x8,[x26]`: completion event at the register-held flat record address. | M37 | `rdma/mesh-call.c:154` |
| `str xzr,[x26]`: consume that event. | M37 | `rdma/mesh-call.c:155` |
| `ldr x0,[x26,#8]`: final result destination in the same line; branch directly to the existing execution-count operation. | M37/M42 | `rdma/mesh-call.c:156` |

| Prepared invocation publication and result; emitted AArch64 | D0 row | Source |
| --- | --- | --- |
| `ldp w9,w8,[x1,#100]`: stride and root count. | M41 | `rdma/mesh-call.h:19` |
| `ldr w0,[x1,#96]`: submitted generation. | M41 | `rdma/mesh-call.h:19` |
| `str w9,[x1,#96]`: advance generation; no slot calculation. | M41 | `rdma/mesh-call.h:20` |
| `ldr w12,[x10,x9,lsl #2]`: root range extent. | M41 | `rdma/mesh-call.h:21` |
| `ldr x13,[x1,x9,lsl #3]`: root range address; argument offset 8. | M41 | `rdma/mesh-call.h:21` |
| `stlr x11,[x13]`: literal 1; advance 16 bytes with register extent. | M18 | `rdma/mesh-call.h:21` |
| `ldp x8,x9,[x3]`; `dmb ishld`: directly held result snapshot. Success and pending paths return without a frame. | M42 | `rdma/mesh-call.h:31`, `swift/Mesh.swift:326` |

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
| CPU-arrival union/find and stream merging | Fixed M18 cells and flat M37 completion records |
| Resident step notification/submit, M21 descriptors, mesh_metal_submit, meshMetalRearm, MeshResidentCommands | Deleted; setup-only body and M37 execution-completion publication |
| Vocabulary GPU publication record/sequence reads (3) | Compiled literal stores |
| Sampler descriptor and sequence reads (2) | Compiled operand-input bindings |
| Final per-step norm sequence read (1) | Resident generation |
| Native command submit loads (2), rearm selection/descriptor/ABI-restore loads (11) | Deleted with native decode rearm |


| Removed receive/device publication load site | Replacement |
| --- | --- |
| RX `m->target_off` | Prepared receive record; no receipt-time publication lookup |
| RX `m->target_stride` | Direct native wr_id pointer |
| RX `provider.queues` | QP/post entry in M08 |
| RX per-slot `receive[slot].requests` | WR inline in M08 |
| RX forwarding `targets[i].count` | M08 carries its sole D2 input destination |
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
| FFN remote stamp address in `mesh_add` | Directly bound M10 input cell |
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
| M19 mesh_use, duplicate function entry/context, setup event union/remap/sort, publisher field | One M20 record for nonresident functions and final M18 cells; resident results use M37 directly |
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

| Deleted in this replacement | Runtime load change |
| --- | --- |
| RX page generation | Delete one generation load and its arithmetic/store |
| RX notification count | Delete one count load and the notification loop |
| RX instruction destination and multiplier | Delete two source field reads; the inline instruction-array load disappears |
| CPU publication invocation | Delete one invocation load and multiplier arithmetic |
| CPU arrival mask and call invocation | Delete two field reads; paired mask/index and pending/invocation loads become single index and pending loads |
| CPU operand.sequence and operand.invocation | Delete the public indirection and counter; not used by D2, excluded from hot-load reduction |
| Attention→FFN and finish→attention input copies | Delete duplicate generation stores and fields; bind the existing numerical completion directly |
| Optional hidden/logit trace consumers | Delete competing decode consumers and their copied outputs |

| Count / construction boundary | Value |
| --- | --- |
| Loads listed | 119 sites: 26 emitted native loads, one direct CPU operand access and 92 GPU source accesses. Atomic read-modify-write sites count once; numerical arithmetic loops are not an ISA-load count. |
| Deleted load sites in this replacement | 7 source fields/sites. Accepted native paths lose 3 emitted load sites: RX notification count/array and CPU invocation. The RX generation load is replaced by its direct operand-record load; the removed CPU mask and invocation shared instructions with retained call index and pending count. |
| Native entry depth T1/T2 | `0/1`: TX reads its directly polled record; native RX wr_id supplies one record whose +64 line contains destination and argument. Payload pointers are actual operands. |
| ABI frames | Input worker 112 bytes, FP=SP+96: call-group backedge reload at FP-88; worker base FP-96 only at shutdown; no accepted-event spills. TX and RX 80 bytes, FP=SP+64, bad-WR SP+8; no RX hot-loop spills. CPU completion has no frame. |
| Compilation | C/Swift library and engine build; Metal source plus explicit finish/FFN/sampling template instantiations compile through makeLibrary without dispatch. The whole-model generated specialization was not executed. |
| Construction | 37/37 rows constructed; construction does not establish whole-graph lifetime closure or measured latency. |
| I2 / I17 / I18 | Native producers write prepared operands/events without waiting. RX performs one direct operand store before native repost. No buffer generation comparison, notification traversal or receive-time destination selection remains on the D2 path. Dedicated native CQ polling remains. |
| I4 | Independent invocation handles have disjoint program-lifetime backing. The inter-rank recurrence still needs a graph lifetime construction establishing that next-step writes cannot overtake a previous remote sampler's reads; no reuse guard is added or claimed. |
| I22 | M04 and M08 carry the native event operands directly. The supplied numerical stages still have cooperative arithmetic counters; this table does not claim their full machine-code footprint fits one metadata line. |
