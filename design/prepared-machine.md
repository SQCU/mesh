# Prepared machine

| Start-time integer | Value |
| --- | --- |
| `r, p` | Rank and peer in `{0,1}`, `p != r`; rank 0 is M5 Max, rank 1 is M4 Pro. |
| `s` | Invocation slot in `[0,F)`; a distinct native queue pair for each simultaneous slot on the link. |
| `l` | Push in `[0,36)`; `0..34` are FFN outputs, `35` is the vocabulary output. |
| `H, V, B` | `1536`, `262144`, `1`. |
| `v_r` | Number of vocabulary columns assigned to rank `r` by the caller. |
| `b(r,l)` | `3072` for `l < 35`; `2*v_r` for `l = 35`. |
| `W` | `524288` bytes, the reserved contiguous extent of each output or receive operand. |
| `Q` | Native payload extent selected during connection setup, fixed for this prepared execution. |
| `C` | `ceil(W/Q)`, reserved native requests per logical push. |
| `c(r,l)` | `ceil(b(r,l)/Q)`, actual native requests for that push. |
| `k` | Native request in `[0,c(r,l))`; private to transport. |
| `e(s,l,p)` | `(36*s+l)*2+p`. |
| `j(s,l,p,k)` | `C*e(s,l,p)+k`. |
| `n(r)` | `35*c(r,0)+c(r,35)`, native requests per rank per slot. |
| `t(r,s,l,k)` | `s*n(r)+l*c(r,0)+k`. |
| `A_r, U_r, T_r, X_r, Y_r, Z_r, I_r, O_r` | Producer arena, receive arena, TX ring, SEND descriptors, RECV descriptors, availability/operand records, native argument storage, native executable storage; allocated and pinned before `start()` returns. |
| `CPU_r(offset), GPU_r(offset), NIC_r(offset)` | The three start-bound aliases of the same registered pages; no runtime translation between aliases. |
| `g` | Invocation generation held in the participating native execution and transport registers before the first producer store. It is a stored value, never an address component. |
| Constructor state | `0` constructed rows. Constructor locations below are replacement sites at `43122d1` and engine `aeb5c96`, not claims that the existing constructors implement these layouts. |

| ID / object | Address | Bytes and required `_Static_assert` | Constructed at `start()` by | Read by which event | Loads to reach it at runtime |
| --- | --- | --- | --- | --- | --- |
| M01 producer output | `A_r + W*(36*s+l)` | Reserved `W`; active `b(r,l)`; `_Static_assert(sizeof(struct prepared_operand_storage)==524288,"M01");` | D1 replacement: `metal-microbench/mesh_layer.swift:93`; allocation realized by `swift/Mesh.swift:649` | value-ready | 0 — bound producer destination |
| M02 receive backing | `U_r + W*e(s,l,p)` | Reserved `W`; active `b(p,l)`; `_Static_assert(sizeof(struct prepared_operand_storage)==524288,"M02");` | D1 replacement: `rdma/mesh-call.c:528` | consumer read | 0 — bound consumer input |
| M03 numerical consumer output | `A_r + 36*F*W + W*(36*s+l)` | Reserved `W`; active `b(r,l)` for FFN and the caller's vocabulary result extent; `_Static_assert(sizeof(struct prepared_operand_storage)==524288,"M03");` | D1 replacement: `metal-microbench/mesh_layer.swift:93` | consumer read | 0 — bound numerical destination |
| M04 producer publication cell / TX ring | `T_r + 8*t(r,s,l,k)` | `8` per entry; `8*F*n(r)` total; `_Static_assert(sizeof(_Atomic uint64_t)==8,"M04");` | D1 replacement: `rdma/mesh-call.c:206`, native producer binding at `swift/Mesh.swift:735` | TX post | 0 — ring cursor held by transport |
| M05 SEND dispatch record | `X_r + 32*j(s,l,p,k)` | `32`: native post entry, native QP, prepared WR address, native failure-output address; `_Static_assert(sizeof(struct prepared_send)==32 && _Alignof(struct prepared_send)==32,"M05");` | D1 replacement: `rdma/mesh-flow.c:97` | TX post | 1 — M04 integer |
| M06 native SEND work request | `X_r + 32*(2*36*F*C) + 128*j(s,l,p,k)` | `128`; `_Static_assert(sizeof(struct ibv_send_wr)==128,"M06");` | D1 replacement: `rdma/mesh-flow.c:151` | TX post | 1 — M04 integer; its affine address is also fixed in M05 |
| M07 native SEND scatter/gather entry | `X_r + 160*(2*36*F*C) + 16*j(s,l,p,k)` | `16`; `_Static_assert(sizeof(struct ibv_sge)==16,"M07");` | D1 replacement: `rdma/mesh-flow.c:263` | TX post | 1 — M04 integer |
| M08 native RECV work request | `Y_r + 32*j(s,l,p,k)` | `32`; `_Static_assert(sizeof(struct ibv_recv_wr)==32,"M08");` | D1 replacement: `rdma/mesh-flow.c:226` | RX completion | 1 — native completion integer |
| M09 native RECV scatter/gather entry | `Y_r + 32*(2*36*F*C) + 16*j(s,l,p,k)` | `16`; `_Static_assert(sizeof(struct ibv_sge)==16,"M09");` | D1 replacement: `rdma/mesh-flow.c:226` | RX completion | 1 — native completion integer |
| M10 receive availability record | `Z_r + 32*j(s,l,p,k)` | `32` aligned bytes; the first `8` bytes are the generation; `_Static_assert(sizeof(struct prepared_available)==32 && _Alignof(struct prepared_available)==32,"M10");` | D1 replacement: `rdma/mesh-call.c:528`, GPU alias at `swift/Mesh.swift:247` | consumer read | 0 — availability address bound before receipt |
| M11 completion output | `CQ_output_r + 48*s` | `48`; `_Static_assert(sizeof(struct ibv_wc)==48,"M11");` | D1 replacement: `rdma/mesh-flow.c:326` | RX completion | 0 — native poll destination held by transport |
| M12 consumer operand view / native argument bytes | `I_r + 32*e(s,l,p)` | `32`: bound input, local contribution, output, extent; `_Static_assert(sizeof(struct prepared_arguments)==32 && _Alignof(struct prepared_arguments)==32,"M12");` | D1 replacement: `metal-microbench/mesh_layer.swift:48` | consumer read | 0 — encoded native argument location |
| M13 producer publication argument bytes | `I_r + 32*(2*36*F) + 32*j(s,l,p,k)` | `32`: bound M04 address, event integer, generation argument, extent; `_Static_assert(sizeof(struct prepared_publication)==32 && _Alignof(struct prepared_publication)==32,"M13");` | D1 replacement: `swift/Mesh.swift:735` | value-ready | 0 — encoded native argument location |
| M14 native consumer command binding | `O_r + D*(36*s+l)`; `D` is the native encoder's fixed start-time command stride | `D`; `_Static_assert(sizeof(struct prepared_command_binding)==32,"M14 binding");`; native executable extent is supplied by its encoder at setup | D1 replacement: `metal-microbench/mesh_layer.swift:189`, installation at `swift/Mesh.swift:897` | consumer read | 0 — native executable already submitted |
| M15 native post failure output | `TX_output_r + 8*s` | `8`; `_Static_assert(sizeof(struct ibv_send_wr *)==8,"M15");` | D1 replacement: `rdma/mesh-flow.c:97` | TX post | 0 — native output address held by transport |

| Fixed native binding | Start-time contents |
| --- | --- |
| M06 | `wr_id=j`; `next=NULL`; `sg_list=M07(j)`; `num_sge=1`; opcode `SEND`; native send flags fixed at setup. |
| M07 | Source `NIC_r(A_r + W*(36*s+l) + Q*k)`; length `min(Q,b(r,l)-Q*k)`; registration key resolved at setup. |
| M08 | `wr_id=j`; `next=NULL`; `sg_list=M09(j)`; `num_sge=1`. |
| M09 | Destination `NIC_r(U_r + W*e(s,l,p) + Q*k)`; length `min(Q,b(p,l)-Q*k)`; registration key resolved at setup. |
| Native queue | One ordered sequence per `(r,s,p)`: increasing layer, then increasing native request within that layer. Different invocation slots have distinct queues and distinct backing. |
| M12 | Direct GPU aliases of M02, M01 and M03, plus the numerical extent; loaded by native execution before the corresponding receipt transition, retained as operand registers. |
| M13 | Direct GPU alias of M04 and the constant `j+1`; no target list, stream header or sequence-pointer traversal. |
| M14 | Supplied numerical code and direct M12/M13 bindings; no receipt-time command construction, commit, callback, or operand-address modification. |
| Full operand availability | M10 at `j(s,l,p,c(p,l)-1)`; earlier native chunks are in the same ordered native receive sequence. Each completion writes its own M10 cell without a last-chunk branch. |
| Reuse | Each layer has distinct backing. A slot's next decode step uses its next input token, produced after that slot's previous vocabulary consumption. No receipt-time storage selection or occupancy word. |

| T1: value ready → SEND posted | Instruction | Object |
| --- | --- | --- |
| T1.01 | Producer's final system-visible payload store at its already bound output address. | M01 |
| T1.02 | Release-store `j+1` to the bound publication cell. | M04, with M13 operands already held |
| T1.03 | Acquire-load the publication cell into `j`. | M04 — ring read |
| T1.04 | Load bytes `0..15` of `X_r + 32*j`. | M05 — first half of one record line |
| T1.05 | Load bytes `16..31` of `X_r + 32*j`. | M05 — second half of the same record line |
| T1.06 | Invoke the setup-bound native post entry with the setup-bound QP, M06 and M15. | M05, M06, M07, M15 |
| T1.07 | Store zero to the consumed publication cell after native acceptance. | M04 |

| T2: RECV completion → consumer first read | Instruction | Object |
| --- | --- | --- |
| T2.01 | Load the completed native `wr_id` into `j` from the fixed native poll destination. | M11 — completion integer |
| T2.02 | Release-store the slot generation already held by transport to `Z_r + 32*j`. | M10 |
| T2.03 | Acquire-load the availability generation at the consumer's already bound M10 address. | M10 — one record line |
| T2.04 | Load the first operand element from the M02 address already held by the native consumer. | M02 — payload |

| Transition construction still required | Concrete unconstructed item |
| --- | --- |
| T1.06 | Native post execution reads M06/M07 as well as M05. The one-ring/one-32-byte-record pass is not established by the dispatch record above; the native submission loads remain to be flattened and listed. |
| T2.03 | The native consumer must already hold M12 and M14 operands before receipt; the present source does not establish this for the replacement layout. |
