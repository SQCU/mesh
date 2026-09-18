# Constructed transitions

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
| FFN load remote input address before polling/cohering local contribution. | M12 | `metal-microbench/matrix_shaders.swift:237` |
| FFN load remote stamp address before polling/cohering local contribution. | M12 | `metal-microbench/matrix_shaders.swift:237` |
| FFN load invocation sequence before its final coherent payload stores. | M20 | `metal-microbench/matrix_shaders.swift:239` |
| FFN load one publication record per participating thread before those stores. | M13 | `metal-microbench/matrix_shaders.swift:242` |
| Additional records beyond thread count: direct consecutive record load. Pair E2B uses the preceding prefetched records. | M13 | `metal-microbench/matrix_shaders.swift:247` |
| FFN store publication value to the prebound destination; no header load after final coherent payload store. | M04, M10 or M18 | `swift/Mesh.swift:755` |
| FFN poll generation at the preloaded stamp address. | M10 | `metal-microbench/matrix_shaders.swift:259` |
| FFN first payload read by supplied matrixAdd, address selected from resident bindings by SIMD shuffle. | M02 | `metal-microbench/matrix_shaders.swift:268` |
| Vocabulary signal loads TX record; count is a function constant and generation is not loaded for TX. | M13 | `swift/Mesh.swift:768` |
| Vocabulary signal stores TX value to its prepared cell. | M04 | `swift/Mesh.swift:755` |
| Vocabulary signal loads invocation after TX publication. | M20 | `swift/Mesh.swift:769` |
| Vocabulary signal loads local-availability/CPU record. | M13 | `swift/Mesh.swift:770` |
| Vocabulary signal stores generation/value at the prepared destination. | M10 or M18 | `swift/Mesh.swift:755` |
| Sampler loads its tile's fixed 32-byte record before polling. | M12 | `metal-microbench/kernels.swift:3722` |
| Sampler loads sequence through the start-bound argument before polling. | M20 | `metal-microbench/kernels.swift:3726` |
| Sampler polls the prebound availability word. | M10 | `metal-microbench/kernels.swift:3727` |
| Sampler reads its first tile element through the returned coherent data pointer. | M01 or M02 | `metal-microbench/kernels.swift:3712` |

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
| `stp x24,x23,[sp,#-64]!`: rearm save at FP-176. | M25 | `swift/Mesh.swift:804` |
| `stp x22,x21,[sp,#16]`: rearm save at FP-160. | M25 | `swift/Mesh.swift:804` |
| `stp x20,x19,[sp,#32]`: rearm save at FP-144. | M25 | `swift/Mesh.swift:804` |
| `stp x29,x30,[sp,#48]`: rearm save at FP-128. | M25 | `swift/Mesh.swift:804` |
| `ldr x19,[x0,#24]`: queue; native retain and command-buffer creation follow. | M21 | `swift/Mesh.swift:806` |
| `ldp x20,x21,[x24,#56]`: add-completion entry and selector. | M21 | `rdma/mesh-call.h:53` |
| `ldr x22,[x24,#48]`: start-created completion block; invoke native attachment. | M21 | `rdma/mesh-call.h:53` |
| `ldr x21,[x24]`: previous command. | M21 | `rdma/mesh-call.h:54` |
| `str x23,[x24]`: new command. | M21 | `rdma/mesh-call.h:55` |
| `ldp x8,x20,[x24,#32]`: encoder entry/context. | M21 | `rdma/mesh-call.h:57` |
| `blr x8` with `x0=x24`: supplied encoder receives the address of M21.command directly. | M21 | `rdma/mesh-call.h:57` |
| `ldp x29,x30,[sp,#48]`: rearm restore. | M25 | `swift/Mesh.swift:809` |
| `ldp x20,x19,[sp,#32]`: rearm restore. | M25 | `swift/Mesh.swift:809` |
| `ldp x22,x21,[sp,#16]`: rearm restore. | M25 | `swift/Mesh.swift:809` |
| `ldp x24,x23,[sp],#64`: rearm restore. | M25 | `swift/Mesh.swift:809` |

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

| Count / construction boundary | Value |
| --- | --- |
| Loads listed | 54 sites: 40 emitted AArch64 load instructions and 14 GPU source-level accesses. AArch64: TX 6, RX handoff 3, RX continuation 6, link lifecycle 2, numerical notification-to-commit 8, rearm selection 2, rearm descriptor 5, rearm ABI restores 4, numerical lifecycle/backedge 4. GPU source accesses are counted separately from native instructions; no GPU ISA load count is inferred from MSL. Conditional sites count once. |
| Deleted load sites | Prior 24 plus the 14 receive/device sites above = 38. M08 post/QP and retirement-row loads replace earlier equivalents and are not counted as deletions. M08 range bounds replace header send/use counts and are not counted as deletions. Stores and setup-only structures are separate above. |
| Native entry depth T1/T2 | `2/1`: TX cursor → cell → dispatch; RX completion integer → M10 stamp. Record bases remain in registers; no RX stack reload precedes M10 publication. Consumer payload/stamp addresses are preloaded before polling; native provider internals are outside this ABI depth. |
| ABI frame sizes | AArch64 `-O2` C and `-O -whole-module-optimization` Swift: numerical worker 128 bytes with FP=SP+112; native rearm 64 bytes; RX worker 128 bytes with RFP=SP+112. M25 covers the numerical FP-176..FP+15 and RX RFP-112..RFP+15. RX reload at RFP-104 occurs only for a declared notification range, after availability and repost. Cold frame-release call remains out of line; no spilled base is reloaded on the completion-to-stamp path. |
| Retained work | Native SEND/RECV/Metal internals, supplied numerical code and encoder bodies, cold numerical and transport refcount/reset work, and the transport retirement scan. Optional CPU/forward publications execute their flat prepared stores after M10; they do not gate the resident consumer. |
| Construction / compilation | 24/24 D0 objects constructed. Bridge, Swift library and engine compile; combined Metal library plus specialized mesh_add/mesh_signal pipelines compile without executing kernels. No inference or benchmark was run. |
