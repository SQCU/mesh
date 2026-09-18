# Prepared machine

| Start-time integer | Value |
| --- | --- |
| `r, p` | Rank in `{0,1}`; `p` is the dense local index of the configured peer link. Rank 0 is M5 Max, rank 1 is M4 Pro. |
| `s` | Invocation slot in `[0,F)`; `F` is caller-supplied concurrent invocations, independent of layer count. Each simultaneous slot has its own native QP and TX spinner. |
| `l` | Physical partial in `[0,L_r)`; `0..34` are FFN outputs, `35..L_r-1` tile the single vocabulary phase. |
| `H, V, B` | `1536`, `262144`, `1`. |
| `v_r` | Number of vocabulary columns assigned to rank `r` by the caller. |
| `L_r, L` | `35 + ceil(v_r/60416)`; `L = max(L_0,L_1)`. |
| `b(r,l)` | `3072` for `l < 35`; otherwise `2*min(60416,v_r-60416*(l-35))`, at most `120832` bytes (118 KiB). |
| `W` | `524288` bytes, the reserved contiguous extent of each output or receive operand. |
| `Q` | Native payload extent selected during connection setup, fixed for this prepared execution. |
| `C` | `ceil(W/Q)`, reserved native requests per logical push. |
| `c(r,l)` | `ceil(b(r,l)/Q)`, actual native requests for that push. |
| `k` | Native request in `[0,c(r,l))`; private to transport. |
| `e(s,l,p)` | `(L*s+l)*2+p`. |
| `P_r(l)` | `sum(c(r,i), i < l)`, resolved at preparation; never loaded by the TX spinner. |
| `j(s,l,p,k)` | `t(r,s,l,k)`, within the peer-specific SEND arena `X_rp`. |
| `n(r)` | `sum(c(r,l), l < L_r)`, native requests per rank per slot. |
| `t(r,s,l,k)` | `s*n(r)+P_r(l)+k`. |
| `A_r, U_r, T_rp, X_rp, Y_r, Z_r, I_r, CQ_output_r, TX_output_rp` | Producer arena, receive arena, TX ring, SEND descriptors, RECV descriptors, availability/operand records, native argument storage, native completion output, native failure output; allocated and pinned before `start()` returns. |
| `CPU_r(offset), GPU_r(offset), NIC_r(offset)` | The three start-bound aliases of the same registered pages; no runtime translation between aliases. |
| `source_row(r,s,l)` | Canonical producer row integer, resolved by the caller’s tensor allocation at setup. |
| `g` | Invocation generation held in the participating native execution and transport registers before the first producer store. It is a stored value, never an address component. |
| Constructor state | `7/15`: M04, M05, M06, M07, M11, M15, M16. Rows still labelled “D1 replacement” remain unconstructed. |

| ID / object | Address | Bytes and required `_Static_assert` | Constructed at `start()` by | Read by which event | Loads to reach it at runtime |
| --- | --- | --- | --- | --- | --- |
| <a id="M01"></a>M01 producer output | `A_r + W*(L*s+l)` | Reserved `W`; active `b(r,l)`; `_Static_assert(sizeof(struct prepared_operand_storage)==524288,"M01");` | D1 replacement: `metal-microbench/mesh_layer.swift:93`; allocation realized by `swift/Mesh.swift:649` | value-ready | 0 — bound producer destination |
| <a id="M02"></a>M02 receive backing | `U_r + W*e(s,l,p)` | Reserved `W`; active `b(p,l)`; `_Static_assert(sizeof(struct prepared_operand_storage)==524288,"M02");` | D1 replacement: `rdma/mesh-call.c:528` | consumer read | 0 — bound consumer input |
| <a id="M03"></a>M03 numerical consumer output | `A_r + L*F*W + W*(L*s+l)` | Reserved `W`; active `b(r,l)` for FFN and the caller's vocabulary result extent; `_Static_assert(sizeof(struct prepared_operand_storage)==524288,"M03");` | D1 replacement: `metal-microbench/mesh_layer.swift:93` | consumer read | 0 — bound numerical destination |
| <a id="M04"></a>M04 producer publication cell / TX ring | `T_rp + 8*t(r,s,l,k)` | `8` per entry; `8*F*n(r)` total; `_Static_assert(sizeof(_Atomic uint64_t)==8,"M04");` | `rdma/mesh-call.c:572`, direct producer binding at `swift/Mesh.swift:648` | TX post | 0 — ring cursor held by transport |
| <a id="M05"></a>M05 SEND dispatch record | `X_rp + 32*j(s,l,p,k)` | `32`: native post entry, native QP, prepared WR address, native failure-output address; `_Static_assert(sizeof(struct prepared_send)==32 && _Alignof(struct prepared_send)==32,"M05");` | `rdma/mesh-flow.c:142` | TX post | 1 — M04 integer |
| <a id="M06"></a>M06 native SEND work request | `X_rp + 32*F*n(r) + 128*j(s,l,p,k)` | `128`; `_Static_assert(sizeof(struct ibv_send_wr)==128,"M06");` | `rdma/mesh-flow.c:141` | TX post | 1 — M04 integer; its affine address is also fixed in M05 |
| <a id="M07"></a>M07 native SEND scatter/gather entry | `X_rp + 160*F*n(r) + 16*j(s,l,p,k)` | `16`; `_Static_assert(sizeof(struct ibv_sge)==16,"M07");` | `rdma/mesh-flow.c:140` | TX post | 1 — M04 integer |
| <a id="M08"></a>M08 native RECV work request | `Y_r + 32*j(s,l,p,k)` | `32`; `_Static_assert(sizeof(struct ibv_recv_wr)==32,"M08");` | D1 replacement: `rdma/mesh-flow.c:226` | RX completion | 1 — native completion integer |
| <a id="M09"></a>M09 native RECV scatter/gather entry | `Y_r + 32*(2*L*F*C) + 16*j(s,l,p,k)` | `16`; `_Static_assert(sizeof(struct ibv_sge)==16,"M09");` | D1 replacement: `rdma/mesh-flow.c:226` | RX completion | 1 — native completion integer |
| <a id="M10"></a>M10 receive availability record | `Z_r + 32*j(s,l,p,k)` | `32` aligned bytes; the first `8` bytes are the generation; `_Static_assert(sizeof(struct prepared_available)==32 && _Alignof(struct prepared_available)==32,"M10");` | D1 replacement: `rdma/mesh-call.c:528`, GPU alias at `swift/Mesh.swift:247` | consumer read | 0 — availability address bound before receipt |
| <a id="M11"></a>M11 completion output | `CQ_output_r + 48*p` | `48`; `_Static_assert(sizeof(struct ibv_wc)==48,"M11");` | `rdma/mesh-flow.c:430`; shared native CQ at `rdma/mesh-verbs.h:238` | RX completion | 0 — native poll destination held by transport |
| <a id="M12"></a>M12 consumer operand view / native argument bytes | `I_r + 32*e(s,l,p)` | `32`: bound input, local contribution, output, extent; `_Static_assert(sizeof(struct prepared_arguments)==32 && _Alignof(struct prepared_arguments)==32,"M12");` | D1 replacement: `metal-microbench/mesh_layer.swift:48`, native encoding at `metal-microbench/mesh_layer.swift:189` and `swift/Mesh.swift:876` | consumer read | 0 — encoded native argument location |
| <a id="M13"></a>M13 producer publication argument bytes | `I_r + 32*(2*L*F) + 32*j(s,l,p,k)` | `32`: bound M04 address, event integer, generation argument, extent; `_Static_assert(sizeof(struct prepared_publication)==32 && _Alignof(struct prepared_publication)==32,"M13");` | D1 replacement: `swift/Mesh.swift:733` | value-ready | 0 — encoded native argument location |
| <a id="M15"></a>M15 native post failure output | `TX_output_rp + 8*s` | `8`; `_Static_assert(sizeof(struct ibv_send_wr *)==8,"M15");` | `rdma/mesh-flow.c:79` | TX post | 0 — native output address held by transport |

| <a id="M16"></a>M16 TX cursor | Prepared image `cursor_rp + 40*s`; live `(l,k)` flattened to the register address `T_rp + 8*t(r,s,l,k)` | `40` preparation bytes, no runtime memory cursor; `_Static_assert(sizeof(struct prepared_cursor)==40,"M16");` | `rdma/mesh-flow.c:94`; thread-entry loads at `rdma/mesh-flow.c:192` | TX post | 0 — current cell, first cell, end and record base remain in transport registers |

| Fixed native binding | Start-time contents |
| --- | --- |
| M06 | `wr_id=(s<<32)+source_row(r,s,l)` (the row is fixed at setup); `next=NULL`; `sg_list=M07(j)`; `num_sge=1`; opcode `SEND`; only the final request of each partial signals TX completion; flags fixed at setup. |
| M07 | Source `NIC_r(A_r + W*(L*s+l) + Q*k)`; length `min(Q,b(r,l)-Q*k)`; registration key resolved at setup. |
| M08 | `wr_id=j`; `next=NULL`; `sg_list=M09(j)`; `num_sge=1`. |
| M09 | Destination `NIC_r(U_r + W*e(s,l,p) + Q*k)`; length `min(Q,b(p,l)-Q*k)`; registration key resolved at setup. |
| Native queue | One ordered sequence per `(r,s,p)`: increasing layer, then increasing native request within that layer. Different invocation slots have distinct QPs, cells, cursors and backing; all slot QPs on `(r,p)` share the peer CQ. No slot scan. Acceptance advances the cursor to the next native request, then the next layer/tile, wrapping only that slot. |
| M12 | Direct GPU aliases of M02, M01 and M03, plus the numerical extent; loaded by native execution before the corresponding receipt transition, retained as operand registers. |
| M13 | Direct GPU alias of M04 and the constant `j+1`; no target list, stream header or sequence-pointer traversal. |
| Native execution of M12/M13 | Supplied numerical code and direct bindings; no receipt-time command construction, commit, callback, or operand-address modification. |
| Full operand availability | M10 at `j(s,l,p,c(p,l)-1)`; earlier native chunks are in the same ordered native receive sequence. Each completion writes its own M10 cell without a last-chunk branch. |
| Reuse | Each layer has distinct backing. A slot's next decode step uses its next input token, produced after that slot's previous vocabulary consumption. No receipt-time storage selection or occupancy word. |

| T1: value ready → SEND posted | Instruction | Object |
| --- | --- | --- |
| T1.01 | Producer's final system-visible payload store at its already bound output address. | M01 |
| T1.02 | Release-store `j+1` to the bound publication cell. | M04, with M13 operands already held |
| T1.03 | Acquire-load the cell at `T_rp + 8*t(r,s,l,k)`; decode `j = value - 1`. | M04 — ring read |
| T1.04 | Load bytes `0..15` of `X_rp + 32*j`. | M05 — first half of one record line |
| T1.05 | Load bytes `16..31` of `X_rp + 32*j`. | M05 — second half of the same record line |
| T1.06 | Invoke the setup-bound native post entry with the setup-bound QP, M06 and M15. | M05, M06, M07, M15 |
| T1.07 | Release-store zero to the consumed cell after native acceptance; advance M16 using register arithmetic only. | M04 |

| T2: RECV completion → consumer first read | Instruction | Object |
| --- | --- | --- |
| T2.01 | Load the completed native `wr_id` into `j` from the fixed native poll destination. | M11 — completion integer |
| T2.02 | Release-store the slot generation already held by transport to `Z_r + 32*j`. | M10 |
| T2.03 | Acquire-load the availability generation at the consumer's already bound M10 address. | M10 — one record line |
| T2.04 | Load the first operand element from the M02 address already held by the native consumer. | M02 — payload |

