# Constructed transitions

| Resident embedding access; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Dispatch entry: load width at +12. | M26 | `metal-microbench/kernels.swift:102` |
| Dispatch entry: load slot stride at +16. | M26 | `metal-microbench/kernels.swift:102` |
| Dispatch entry: load embedding scale at +20. | M26 | `metal-microbench/kernels.swift:103` |
| Dispatch entry: load completed generation at +0; add register stride. | M26 | `metal-microbench/kernels.swift:104` |
| Lane 0 loads lease stop at +8; SIMD broadcast. | M26 | `metal-microbench/kernels.swift:107` |
| Lane 0 loads the directly bound input generation; SIMD broadcast. | M10 | `metal-microbench/kernels.swift:109` |
| Read token from the directly bound canonical input, rank 0 byte offset 12 and other ranks byte offset 0. | M01/M02 | `metal-microbench/kernels.swift:111`, binding at `metal-microbench/mesh_layer.swift:33` |
| Store embedding components into the fixed canonical hidden operand. | M03 | `metal-microbench/kernels.swift:113` |
| After the numerical stores and fence, lane 0 stores completed generation at +0; advance generation in registers. | M26 | `metal-microbench/kernels.swift:116` |
| First remaining numerical consumer loads invocation from its native sequence argument. | M20 | `metal-microbench/kernels.swift:87` |
| First remaining numerical consumer polls completed generation through its native state argument. | M26 | `metal-microbench/kernels.swift:87` |

| Resident FFN and layer-finish global control accesses; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Layer-finish producer loads its bound sequence. | M20 | `metal-microbench/kernels.swift:85` |
| Layer-finish producer stores the next attention input generation after coherent operand stores. | M28 | `metal-microbench/kernels.swift:85` |
| Dispatch entry loads final layer's completed generation at its compiled offset. | M27 | `metal-microbench/mesh_layer.swift:162` |
| Stage entry loads gate-done at +4. | M27 | `metal-microbench/kernels.swift:1639` |
| Operand poll loads the lease stop at M26 +8. | M26 | `metal-microbench/kernels.swift:1645` |
| Operand poll loads input generation at +16. | M27 | `metal-microbench/kernels.swift:1645` |
| Operand poll loads completed generation at +24. | M27 | `metal-microbench/kernels.swift:1645` |
| Numerical normalization first reads the bound input payload. | M03 | `metal-microbench/kernels.swift:1652` |
| Gate loop loads gate-done at +4. | M27 | `metal-microbench/kernels.swift:1661` |
| Gate loop loads lease stop at M26 +8. | M26 | `metal-microbench/kernels.swift:1662` |
| Gate claim loads gate-next at +0. | M27 | `metal-microbench/kernels.swift:1664` |
| Gate claim compare/exchange reads and writes gate-next at +0. | M27 | `metal-microbench/kernels.swift:1665` |
| Completed numerical gate tile increments gate-done at +4. | M27 | `metal-microbench/kernels.swift:1688` |
| Down loop loads down-done at +12. | M27 | `metal-microbench/kernels.swift:1696` |
| Down loop loads lease stop at M26 +8. | M26 | `metal-microbench/kernels.swift:1697` |
| Down claim loads down-next at +8. | M27 | `metal-microbench/kernels.swift:1699` |
| Down claim compare/exchange reads and writes down-next at +8. | M27 | `metal-microbench/kernels.swift:1700` |
| Completed numerical down tile increments down-done at +12 before direct publication. | M27 | `metal-microbench/kernels.swift:1718` |
| Final tile stores completed generation at +24 after publication. | M27 | `metal-microbench/kernels.swift:1720` |

| Resident attention global control accesses; MSL source, not emitted GPU instructions | D0 row | Source |
| --- | --- | --- |
| Input poll loads lease stop. | M26 | `metal-microbench/kernels.swift:2867` |
| Input poll loads completed generation. | M28 | `metal-microbench/kernels.swift:2867` |
| Input poll loads embedding completion or the fixed layer input generation. | M26 or M28 | `metal-microbench/kernels.swift:2867` |
| Normalization reads its first bound hidden component. | M03 | `metal-microbench/kernels.swift:2875` |
| Projection claim loads next column tile. | M28 | `metal-microbench/kernels.swift:2881` |
| Projection loop loads lease stop. | M26 | `metal-microbench/kernels.swift:2882` |
| Projection claim compare/exchange reads and advances next tile. | M28 | `metal-microbench/kernels.swift:2883` |
| Completed column tile increments its head's count. | M29 | `metal-microbench/kernels.swift:2906` |
| Last tile stores head readiness after norm/RoPE/KV work. | M29 | `metal-microbench/kernels.swift:2931` |
| Split claim loads next split ordinal. | M28 | `metal-microbench/kernels.swift:2940` |
| Split loop loads lease stop. | M26 | `metal-microbench/kernels.swift:2941` |
| Split claim loads its query-group readiness. | M29 | `metal-microbench/kernels.swift:2944` |
| Split claim loads the bound KV producer head readiness. | M29 | `metal-microbench/kernels.swift:2945` |
| Split claim compare/exchange reads and advances next ordinal. | M28 | `metal-microbench/kernels.swift:2946` |
| Completed split increments its query-group count. | M29 | `metal-microbench/kernels.swift:2963` |
| Last split stores reduced generation after the existing numerical reduction. | M29 | `metal-microbench/kernels.swift:2971` |
| Output-input poll loads lease stop. | M26 | `metal-microbench/kernels.swift:2980` |
| Output-input poll reads each query group's reduced generation once before the output tile loop. | M29 | `metal-microbench/kernels.swift:2982` |
| Output tile loop loads completed tile count. | M28 | `metal-microbench/kernels.swift:2993` |
| Output tile loop loads lease stop. | M26 | `metal-microbench/kernels.swift:2994` |
| Output claim loads next column tile. | M28 | `metal-microbench/kernels.swift:2996` |
| Output claim compare/exchange reads and advances next tile. | M28 | `metal-microbench/kernels.swift:2997` |
| Completed output tile increments output-done. | M28 | `metal-microbench/kernels.swift:3010` |
| Last tile stores FFN input generation after post-attention norm/residual output. | M27 | `metal-microbench/kernels.swift:3018` |
| Last tile stores attention completion generation. | M28 | `metal-microbench/kernels.swift:3018` |

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

| GPU access site; MSL source, not an emitted GPU instruction count | D0 row | Source |
| --- | --- | --- |
| FFN load remote input address before polling local contribution. | M12 | `metal-microbench/matrix_shaders.swift:236` |
| FFN load remote stamp address before polling local contribution. | M12 | `metal-microbench/matrix_shaders.swift:236` |
| FFN consumer loads invocation sequence. | M20 | `metal-microbench/matrix_shaders.swift:238` |
| FFN consumer polls directly bound local availability. | M10 | `metal-microbench/matrix_shaders.swift:240` |
| FFN final tile stores publication values to destinations compiled at setup; no publication record is loaded. | M04, M10 or M18 | `metal-microbench/mesh_layer.swift:61`, `metal-microbench/kernels.swift:1720` |
| FFN poll generation at the preloaded stamp address. | M10 | `metal-microbench/matrix_shaders.swift:249` |
| FFN first payload read by supplied matrixAdd, address selected from resident bindings by SIMD shuffle. | M02 | `metal-microbench/matrix_shaders.swift:257` |
| Vocabulary signal loads TX record; count is a function constant and generation is not loaded for TX. | M13 | `swift/Mesh.swift:768` |
| Vocabulary signal stores TX value to its prepared cell. | M04 | `swift/Mesh.swift:755` |
| Vocabulary signal loads invocation after TX publication. | M20 | `swift/Mesh.swift:769` |
| Vocabulary signal loads local-availability/CPU record. | M13 | `swift/Mesh.swift:770` |
| Vocabulary signal stores generation/value at the prepared destination. | M10 or M18 | `swift/Mesh.swift:755` |
| Sampler loads its tile's fixed 32-byte record before polling. | M12 | `metal-microbench/kernels.swift:4182` |
| Sampler loads sequence through the start-bound argument before polling. | M20 | `metal-microbench/kernels.swift:4186` |
| Sampler polls the prebound availability word. | M10 | `metal-microbench/kernels.swift:4187` |
| Sampler reads its first tile element through the returned coherent data pointer. | M01 or M02 | `metal-microbench/kernels.swift:4172` |

| T3 instruction: numerical notification → native Metal commit | D0 row | Source |
| --- | --- | --- |
| `ldapr x9,[x9]`: cell at base + 128*register ordinal + 8. | M18 | `rdma/mesh-call.c:200` |
| `str xzr,[x11,x8]`: clear the accepted cell. | M18 | `rdma/mesh-call.c:202` |
| `ldr w9,[x22,#28]`: target range end. | M19 | `rdma/mesh-call.c:204` |
| `ldr x0,[x22,#16]`: call address. | M19 | `rdma/mesh-call.c:206` |
| `ldr w8,[x22,#24]`: invocation mask. | M19 | `rdma/mesh-call.c:207` |
| `ldp w11,w10,[x0,#32]`: pending and sequence. | M20 | `rdma/mesh-call.c:207` |
| `stp w9,w8,[x0,#32]`: decrement pending and write sequence; the zero-count gate remains. | M20 | `rdma/mesh-call.c:208` |
| `ldp x8,x20,[x22]`: native submit entry and prepared context. | M19 | `rdma/mesh-call.c:209` |
| `blr x8`: bound function entry. | M19 | `rdma/mesh-call.c:209` |
| `ldp x0,x2,[x20]`: command and native commit entry. | M21 | `rdma/mesh-call.c:165` |
| `ldr x1,[x20,#16]`: selector. | M21 | `rdma/mesh-call.c:165` |
| `br x2`: native commit, without a Swift command-array lookup. | M21 | `rdma/mesh-call.c:165` |

| T3 instruction: completed frame → next native command | D0 row | Source |
| --- | --- | --- |
| Load completed call's native record pointer at +64. | M20 | `rdma/mesh-call.c:224` |
| Load native rearm entry at +72, then call it. | M21 | `rdma/mesh-call.c:226` |
| `stp x24,x23,[sp,#-64]!`: rearm save at FP-176. | M25 | `swift/Mesh.swift:799` |
| `stp x22,x21,[sp,#16]`: rearm save at FP-160. | M25 | `swift/Mesh.swift:799` |
| `stp x20,x19,[sp,#32]`: rearm save at FP-144. | M25 | `swift/Mesh.swift:799` |
| `stp x29,x30,[sp,#48]`: rearm save at FP-128. | M25 | `swift/Mesh.swift:799` |
| `ldr x19,[x0,#24]`: queue; native retain and command-buffer creation follow. | M21 | `swift/Mesh.swift:801` |
| `ldp x20,x21,[x24,#56]`: add-completion entry and selector. | M21 | `rdma/mesh-call.h:75` |
| `ldr x22,[x24,#48]`: start-created completion block; invoke native attachment. | M21 | `rdma/mesh-call.h:75` |
| `ldr x21,[x24]`: previous command. | M21 | `rdma/mesh-call.h:76` |
| `str x23,[x24]`: new command. | M21 | `rdma/mesh-call.h:77` |
| `ldp x8,x20,[x24,#32]`: encoder entry/context. | M21 | `rdma/mesh-call.h:79` |
| `blr x8` with `x0=x24`: supplied encoder receives the address of M21.command directly. | M21 | `rdma/mesh-call.h:79` |
| `ldp x29,x30,[sp,#48]`: rearm restore. | M25 | `swift/Mesh.swift:804` |
| `ldp x20,x19,[sp,#32]`: rearm restore. | M25 | `swift/Mesh.swift:804` |
| `ldp x22,x21,[sp,#16]`: rearm restore. | M25 | `swift/Mesh.swift:804` |
| `ldp x24,x23,[sp],#64`: rearm restore. | M25 | `swift/Mesh.swift:804` |

| T3 polling instructions outside accepted-event → commit | D0 row | Source |
| --- | --- | --- |
| `ldr x8,[sp,#8]`: call-group pointer at FP-104. | M25 | `rdma/mesh-call.c:195` |
| `ldapr w8,[x8]`: running word after adding 1196. | M24 | `rdma/mesh-call.c:195` |
| `ldp x23,x11,[sp,#16]`: invocation/arrival bases on probe backedge, FP-96. | M25 | `rdma/mesh-call.c:197` |
| `ldr x11,[sp,#24]`: arrival base after cold returns, FP-88. | M25 | `rdma/mesh-call.c:217` |

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
| Resident invocation's `mesh_call_index` and managed command-array traversal | M19 argument points directly to M21 |
| Rearm frame-array lookup | M21 retained completion block |
| Rearm encoder-array lookup | M21 encoder entry/context |
| Per-rearm Swift closure allocation and `_Block_copy` | Block created once during preparation |
| Per-rearm Objective-C protocol cast | Prepared queue identity |
| Encoder argument store/reload on the stack | M21.command is the native argument address |

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
| Uninstantiated `MeshCommands` class | Deleted |

| Count / construction boundary | Value |
| --- | --- |
| Loads listed | 100 sites: 40 emitted AArch64 load instructions and 60 GPU source-level accesses. GPU: resident embedding/handoff 9, resident FFN control 16, layer-finish handoff 1, resident attention control/handoff 21, collective/vocabulary/sampler 13. Atomic read-modify-write sites count once. Numerical weight/input loops (including the existing M31/M32 position/page/mask gathers) and threadgroup arithmetic remain supplied numerical work, not ABI load instructions. AArch64: TX 6, RX handoff 3, RX continuation 6, link lifecycle 2, numerical notification-to-commit 8, rearm selection 2, rearm descriptor 5, rearm ABI restores 4, numerical lifecycle/backedge 4. No GPU ISA load or spill count is inferred from MSL. |
| Deleted load sites | Prior 38 plus the two FFN publication-record sites = 40. M08 post/QP and retirement-row loads replace earlier equivalents and are not counted as deletions. M08 range bounds replace header send/use counts and are not counted as deletions. Stores and setup-only structures are separate above. |
| Native entry depth T1/T2 | `2/1`: TX cursor → cell → dispatch; RX completion integer → M10 stamp. Record bases remain in registers; no RX stack reload precedes M10 publication. Consumer payload/stamp addresses are preloaded before polling; native provider internals are outside this ABI depth. |
| ABI frame sizes | AArch64 `-O2` C and `-O -whole-module-optimization` Swift: numerical worker 128 bytes with FP=SP+112; native rearm 64 bytes; RX worker 128 bytes with RFP=SP+112. M25 covers the numerical FP-176..FP+15 and RX RFP-112..RFP+15. RX reload at RFP-104 occurs only for a declared notification range, after availability and repost. Cold frame-release call remains out of line; no spilled base is reloaded on the completion-to-stamp path. |
| Retained work | Native SEND/RECV/Metal internals, supplied numerical code and encoder bodies, cold numerical and transport refcount/reset work, and the transport retirement scan. Optional CPU/forward publications execute their flat prepared stores after M10; they do not gate the resident consumer. |
| Construction / compilation | 31/31 listed objects constructed. Swift engine, native transport and generated 35-layer attention + FFN resident body compile; the combined body uses 12256 threadgroup bytes at H=1536. RoPE, KV write, split attention/reduction and both specialized layer-finish handoff entries compile. Embedding/scaling, attention and FFNs are resident; finish/PLE, vocabulary and sampler remain per-step. T3's 26 derivation instructions and four Mesh.swift command-buffer allocation sites remain until the final per-step encoder is removed. The whole-step requirement is unmet; GPU execution and throughput are unmeasured. |
