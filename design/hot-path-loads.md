# Constructed transitions

| T1 instruction: cursor → native SEND acceptance / advance | D0 row | Source |
| --- | --- | --- |
| `ldr x25,[x24]`: fixed slot's current cell pointer. | M16 | `rdma/mesh-flow.c:204` |
| `ldapr x8,[x25]`: publication cell. | M04 | `rdma/mesh-flow.c:205` |
| `ldp x9,x0,[x8,#-32]`: native entry/QP. | M05 | `rdma/mesh-flow.c:210` |
| `ldp x1,x2,[x8,#-16]`: WR/failure output. | M05, M06, M15 | `rdma/mesh-flow.c:210` |
| `blr x9`: native post. | M05–M07, M15 | `rdma/mesh-flow.c:211` |
| `stlr xzr,[x25]`: accepted cell clear. | M04 | `rdma/mesh-flow.c:216` |
| `ldr x9,[x24,#16]`: slot end. | M16 | `rdma/mesh-flow.c:217` |
| `ldr x8,[x24,#8]`: slot first, only on wrap. | M16 | `rdma/mesh-flow.c:217` |
| `str x8,[x24]`: slot cursor advance. | M16 | `rdma/mesh-flow.c:217` |

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
| `ldp x20,x21,[x24,#56]`: add-completion entry and selector. | M21 | `rdma/mesh-call.h:43` |
| `ldr x22,[x24,#48]`: start-created completion block; invoke native attachment. | M21 | `rdma/mesh-call.h:43` |
| `ldr x21,[x24]`: previous command. | M21 | `rdma/mesh-call.h:44` |
| `str x23,[x24]`: new command. | M21 | `rdma/mesh-call.h:45` |
| `ldp x8,x20,[x24,#32]`: encoder entry/context. | M21 | `rdma/mesh-call.h:47` |
| `blr x8` with `x0=x24`: supplied encoder receives the address of M21.command directly. | M21 | `rdma/mesh-call.h:47` |
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

| Count / construction boundary | Value |
| --- | --- |
| Loads listed | 29 instruction sites: TX 6 including cursor advancement; numerical notification-to-commit 8; native rearm selection 2; rearm descriptor reads 5; rearm ABI restores 4; numerical polling lifecycle/backedge 4. Conditional sites are counted once, not weighted by frequency. |
| Deleted load sites | 17 prior TX source sites plus 7 numerical source sites: arrivals.inputs, input.slots, input.position, input.mask, resident call index, rearm frame-array lookup, rearm encoder-array lookup = 24. Allocations, stores and eliminated preparation algorithms are reported separately above. |
| Native entry depth | TX accepted path: cursor → cell → dispatch = 2 dependent address edges. Numerical notification-to-commit: event → M19 → M20/M21 = 2. No stack reload occurs between accepted M18 read and native commit. |
| ABI frame sizes | AArch64 `-O` output, with the library and assembly using Swift whole-module optimization: numerical worker reserves 128 bytes with FP=SP+112; rearm reserves 64 bytes. M25 covers FP-176 through FP+15. Setup publishes FP before start returns. The word-size assertion does not substitute for these compiler frame sizes. |
| Retained work | Native SEND/Metal internals, supplied CPU preparation and encoder bodies, numerical refcount/reset loop, transport retirement scan. These are not included in the 29-site primitive table and are not claimed to have zero cost. |
| Unconstructed transition portions | M13 and the complete receive/consumer transition remain outside the constructed claims; full T1/T2 depth and D2 timing remain unestablished. |
