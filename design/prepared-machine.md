# Prepared machine

| Start-time integer | Value |
| --- | --- |
| `r, p` | Rank in `{0,1}`; `p` is the dense local index of the configured peer link. Rank 0 is M5 Max, rank 1 is M4 Pro. |
| `s` | Invocation slot in `[0,F)`; `F` is caller-supplied concurrent invocations, independent of layer count. Each simultaneous slot has its own native QP and cursor; one TX spinner serves the link. |
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
| `a, h, f, w` | Start-bound numerical notification ordinal, invocation-record ordinal, function identity, and worker index. `a` and `h` include the invocation slot. Normal decode has `2F` notification cells per rank: input preparation and resident step; optional trace calls add their declared cells. |
| `E_r, D_r, C_r, B_rf, K_r, J_rf, G_r` | Numerical notification arena, invocation records, call frames, native command records, worker records, function metadata, call-group allocation; each base is an integer fixed by preparation. |
| Constructor state | `16/24`: M04–M07, M11, M15–M25. Rows still labelled “D1 replacement” remain unconstructed. |

| ID / object | Address | Bytes and required `_Static_assert` | Constructed at `start()` by | Read by which event | Loads to reach it at runtime |
| --- | --- | --- | --- | --- | --- |
| <a id="M01"></a>M01 producer output | `A_r + W*(L*s+l)` | Reserved `W`; active `b(r,l)`; `_Static_assert(sizeof(struct prepared_operand_storage)==524288,"M01");` | D1 replacement: `metal-microbench/mesh_layer.swift:93`; allocation realized by `swift/Mesh.swift:649` | value-ready | 0 — bound producer destination |
| <a id="M02"></a>M02 receive backing | `U_r + W*e(s,l,p)` | Reserved `W`; active `b(p,l)`; `_Static_assert(sizeof(struct prepared_operand_storage)==524288,"M02");` | D1 replacement: `rdma/mesh-call.c:528` | consumer read | 0 — bound consumer input |
| <a id="M03"></a>M03 numerical consumer output | `A_r + L*F*W + W*(L*s+l)` | Reserved `W`; active `b(r,l)` for FFN and the caller's vocabulary result extent; `_Static_assert(sizeof(struct prepared_operand_storage)==524288,"M03");` | D1 replacement: `metal-microbench/mesh_layer.swift:93` | consumer read | 0 — bound numerical destination |
| <a id="M04"></a>M04 producer publication cell / TX ring | `T_rp + 8*t(r,s,l,k)` | `8` per entry; `8*F*n(r)` total; `_Static_assert(sizeof(_Atomic uint64_t)==8,"M04");` | `rdma/mesh-call.c:607`, direct producer binding at `swift/Mesh.swift:652` | TX post | 1 — M16 supplies the cell address; the bounded slot scan uses register arithmetic |
| <a id="M05"></a>M05 SEND dispatch record | `X_rp + 32*j(s,l,p,k)` | `32`: native post entry, native QP, prepared WR address, native failure-output address; `_Static_assert(sizeof(struct prepared_send)==32 && _Alignof(struct prepared_send)==32,"M05");` | `rdma/mesh-flow.c:146` | TX post | 1 — M04 integer |
| <a id="M06"></a>M06 native SEND work request | `X_rp + 32*F*n(r) + 128*j(s,l,p,k)` | `128`; `_Static_assert(sizeof(struct ibv_send_wr)==128,"M06");` | `rdma/mesh-flow.c:145` | TX post | 1 — M04 integer; its affine address is also fixed in M05 |
| <a id="M07"></a>M07 native SEND scatter/gather entry | `X_rp + 160*F*n(r) + 16*j(s,l,p,k)` | `16`; `_Static_assert(sizeof(struct ibv_sge)==16,"M07");` | `rdma/mesh-flow.c:144` | TX post | 1 — M04 integer |
| <a id="M08"></a>M08 native RECV work request | `Y_r + 32*j(s,l,p,k)` | `32`; `_Static_assert(sizeof(struct ibv_recv_wr)==32,"M08");` | D1 replacement: `rdma/mesh-flow.c:226` | RX completion | 1 — native completion integer |
| <a id="M09"></a>M09 native RECV scatter/gather entry | `Y_r + 32*(2*L*F*C) + 16*j(s,l,p,k)` | `16`; `_Static_assert(sizeof(struct ibv_sge)==16,"M09");` | D1 replacement: `rdma/mesh-flow.c:226` | RX completion | 1 — native completion integer |
| <a id="M10"></a>M10 receive availability record | `Z_r + 32*j(s,l,p,k)` | `32` aligned bytes; the first `8` bytes are the generation; `_Static_assert(sizeof(struct prepared_available)==32 && _Alignof(struct prepared_available)==32,"M10");` | D1 replacement: `rdma/mesh-call.c:528`, GPU alias at `swift/Mesh.swift:247` | consumer read | 0 — availability address bound before receipt |
| <a id="M11"></a>M11 completion output | `CQ_output_r + 48*p` | `48`; `_Static_assert(sizeof(struct ibv_wc)==48,"M11");` | `rdma/mesh-flow.c:431`; shared native CQ at `rdma/mesh-verbs.h:238` | RX completion | 0 — native poll destination held by transport |
| <a id="M12"></a>M12 consumer operand view / native argument bytes | `I_r + 32*e(s,l,p)` | `32`: bound input, local contribution, output, extent; `_Static_assert(sizeof(struct prepared_arguments)==32 && _Alignof(struct prepared_arguments)==32,"M12");` | D1 replacement: `metal-microbench/mesh_layer.swift:48`, native encoding at `metal-microbench/mesh_layer.swift:189` and `swift/Mesh.swift:876` | consumer read | 0 — encoded native argument location |
| <a id="M13"></a>M13 producer publication argument bytes | `I_r + 32*(2*L*F) + 32*j(s,l,p,k)` | `32`: bound M04 address, event integer, generation argument, extent; `_Static_assert(sizeof(struct prepared_publication)==32 && _Alignof(struct prepared_publication)==32,"M13");` | D1 replacement: `swift/Mesh.swift:733` | value-ready | 0 — encoded native argument location |
| <a id="M15"></a>M15 native post failure output | `TX_output_rp + 8*s` | `8`; `_Static_assert(sizeof(struct ibv_send_wr *)==8,"M15");` | `rdma/mesh-flow.c:80` | TX post | 0 — native output address held by transport |
| <a id="M16"></a>M16 TX cursors | `cursor_rp + 32*s`; cell value is `T_rp + 8*t(r,s,l,k)` | `32F`; `_Static_assert(sizeof(struct prepared_cursor)==32 && _Alignof(struct prepared_cursor)==32,"M16");` | `rdma/mesh-flow.c:97` | TX post | 0 — slot index and cursor-array base held in registers; one cursor load before M04 |
| <a id="M17"></a>M17 thread handles / population | `link_base_r + sizeof(struct mesh_link)*p + offsetof(struct mesh_link,workers) + 8*d`, `d<4`; controller at that link's `controller` offset | `8` per pthread handle; `_Static_assert(sizeof(pthread_t)==8,"M17");` | `rdma/mesh-flow.c:337`, `rdma/mesh-flow.c:456`; numerical workers at `rdma/mesh-call.c:470` | TX post | 0 — native thread entry argument. Chosen: one TX thread per link scans at most F flat M16 cursors, one pointer load then one cell load per probe, instead of F spinning threads. Per arena with P links and W active numerical workers: `spinners <= 4P+W+1`, `threads <= 5P+W+1`; the extra 1 is the spinning bridge main, and P controllers block in kqueue. `W<=8`, independent of `MESH_QPS` and F; decode and prefill each configure W=1. The engine constructs both arenas: for the pair, P=1 per arena, total bound 12 spinners and 14 Mesh threads per rank, independent of F. Native driver, Metal and caller-owned threads are outside this Mesh-owned count. |
| <a id="M18"></a>M18 numerical notification cell and publication header | `E_r + 128*a`; the event cell is at `+8` | `128` stride, `8` event bytes; `_Static_assert(sizeof(struct mesh_stream)==128 && offsetof(struct mesh_stream,slots)==8,"M18");` | `rdma/mesh-call.c:280`, fixed worker base at `rdma/mesh-call.c:323` | step rearm | 0 — base plus register cursor times 128; M25 accounts for the prebound-base reloads; no input array, per-input position or mask load |
| <a id="M19"></a>M19 numerical invocation record | `D_r + 32*h` | `32`; `_Static_assert(sizeof(struct mesh_use)==32 && _Alignof(struct mesh_use)==32,"M19");` | `rdma/mesh-call.c:390`; native entry binding at `rdma/mesh-call.c:180` | step rearm | 1 — M18 low event integer; the record includes submit, argument, call, invocation mask and end |
| <a id="M20"></a>M20 call frame | `C_r + 128*(f*F+s)` | `128`, aligned 64; `_Static_assert(sizeof(struct mesh_call)==128 && _Alignof(struct mesh_call)==64,"M20");` | `rdma/mesh-call.c:111`, frame initialization at `rdma/mesh-call.c:135` | step rearm | 1 — prepared M19 call pointer; return processing indexes the same array by its event integer. Pending/sequence at +32/+36; native rearm record pointer at +64. |
| <a id="M21"></a>M21 native command and rearm arguments | `B_rf + 96*s` | `96`, allocation aligned 32; `_Static_assert(sizeof(struct prepared_metal)==96 && offsetof(struct prepared_metal,encode)==32,"M21");` | `swift/Mesh.swift:823`, `swift/Mesh.swift:852`; native methods bound at `rdma/mesh-call.c:178` | step rearm | 1 — M19 argument for submission, M20 native pointer for rearm. First line: command, commit entry, selector, queue. Following lines: encoder entry/context, retained completion block, add-completion entry/selector, rearm entry. The descriptor and initial command exist at start; later command identities are native API return values stored in this same descriptor. |
| <a id="M22"></a>M22 numerical worker state | `K_r + 128*w` | `128`; `_Static_assert(sizeof(struct mesh_call_worker)==128 && _Alignof(struct mesh_call_worker)==128,"M22");` | `rdma/mesh-call.c:90` | step rearm | 0 — thread entry argument; arrival base/count are loaded before polling; cold return-reader state remains here |
| <a id="M23"></a>M23 function lifetime / rearm metadata | `J_rf` (one base integer per function, fixed at binding) | `128`; `_Static_assert(sizeof(struct mesh_function)==128,"M23");` | `rdma/mesh-call.c:114` | step rearm | 1 — M20 function pointer, used by cold refcount/reset work; native submission does not read it |
| <a id="M24"></a>M24 call-group lifecycle state | `G_r`; running at `+1196`, worker array at `+128` | `1280`; `_Static_assert(sizeof(struct mesh_calls)==1280,"M24");` | `rdma/mesh-call.c:85` | step rearm | 0 — captured by the numerical thread at entry; lifecycle polling is outside notification-to-commit instructions |
| <a id="M25"></a>M25 numerical worker / native rearm ABI frames | `FP_r-176` through `FP_r+15`; `FP_r` is the worker's frame address, published before `start()` returns | `192` in this AArch64 build: 128-byte worker frame plus 64-byte rearm frame; `_Static_assert(24*sizeof(uintptr_t)==192,"M25");` pins the word extent; compiler frame sizes are recorded in D1 | `rdma/mesh-call.c:194`, startup publication observed at `rdma/mesh-call.c:476` | step rearm | 0 — frame/SP registers; bases are fixed before submission. Worker spills are at `FP_r-104`, `FP_r-96`, `FP_r-88`; rearm saves start at `FP_r-176`. |


| Fixed native binding | Start-time contents |
| --- | --- |
| M06 | `wr_id=(s<<32)+source_row(r,s,l)` (the row is fixed at setup); `next=NULL`; `sg_list=M07(j)`; `num_sge=1`; opcode `SEND`; only the final request of each partial signals TX completion; flags fixed at setup. |
| M07 | Source `NIC_r(A_r + W*(L*s+l) + Q*k)`; length `min(Q,b(r,l)-Q*k)`; registration key resolved at setup. |
| M08 | `wr_id=j`; `next=NULL`; `sg_list=M09(j)`; `num_sge=1`. |
| M09 | Destination `NIC_r(U_r + W*e(s,l,p) + Q*k)`; length `min(Q,b(p,l)-Q*k)`; registration key resolved at setup. |
| Native queue | One ordered sequence per `(r,s,p)`: increasing layer, then increasing native request within that layer. Different invocation slots have distinct QPs, cells, cursors and backing; all slot QPs on `(r,p)` share the peer CQ. The single link spinner probes one M16 cursor at a time, rotating across at most F slots. Acceptance advances only that slot to the next native request, then the next layer/tile. Native refusal leaves its cursor unchanged and proceeds to the next slot. |
| M12 | Direct GPU aliases of M02, M01 and M03, plus the numerical extent; loaded by native execution before the corresponding receipt transition, retained as operand registers. |
| M13 | Direct GPU alias of M04 and the constant `j+1`; no target list, stream header or sequence-pointer traversal. |
| Native execution of M12/M13 | Supplied numerical code and direct bindings; no receipt-time command construction, commit, callback, or operand-address modification. |
| Full operand availability | M10 at `j(s,l,p,c(p,l)-1)`; earlier native chunks are in the same ordered native receive sequence. Each completion writes its own M10 cell without a last-chunk branch. |
| Reuse | Each layer has distinct backing. A slot's next decode step uses its next input token, produced after that slot's previous vocabulary consumption. No receipt-time storage selection or occupancy word. |

| T1: value ready → SEND posted | Instruction | Object |
| --- | --- | --- |
| T1.01 | Producer's final system-visible payload store at its already bound output address. | M01 |
| T1.02 | Release-store `j+1` to the bound publication cell. | M04, with M13 operands already held |
| T1.03 | Load the current cell pointer from `cursor_rp + 32*s`; advance the register slot index modulo F. | M16 |
| T1.04 | Acquire-load that M04 cell; decode `j=value-1`. | M04 |
| T1.05 | Load bytes `0..15` of `X_rp + 32*j`. | M05 |
| T1.06 | Load bytes `16..31` of `X_rp + 32*j`. | M05 |
| T1.07 | Invoke the prepared native post entry with QP, M06 and M15. | M05, M06, M07, M15 |
| T1.08 | Release-store zero to the accepted cell. | M04 |
| T1.09 | Load end from `cursor_rp + 32*s + 16`. | M16 |
| T1.10 | On wrap, load first from `cursor_rp + 32*s + 8`. | M16 |
| T1.11 | Store the advanced cell pointer at `cursor_rp + 32*s`. | M16 |

| T2: RECV completion → consumer first read | Instruction | Object |
| --- | --- | --- |
| T2.01 | Load the completed native `wr_id` into `j` from the fixed native poll destination. | M11 — completion integer |
| T2.02 | Release-store the slot generation already held by transport to `Z_r + 32*j`. | M10 |
| T2.03 | Acquire-load the availability generation at the consumer's already bound M10 address. | M10 — one record line |
| T2.04 | Load the first operand element from the M02 address already held by the native consumer. | M02 — payload |


| T3: step notification → native commit, then next-command rearm | Instruction | Object |
| --- | --- | --- |
| T3.01 | Acquire-load event from `E_r+128*a+8`; the rotating ordinal is held in a register. | M18 |
| T3.02 | Store zero to that cell; decode `h=low32(event-1)` and generation `high32(event-1)` in registers. | M18 |
| T3.03 | Load end from `D_r+32*h+28`. Normal decode has one target per notification. | M19 |
| T3.04 | Load call pointer from `D_r+32*h+16`. | M19 |
| T3.05 | Load invocation mask from `D_r+32*h+24`. | M19 |
| T3.06 | Load pending and sequence from `C_r+128*(f*F+s)+32` as one pair. | M20 |
| T3.07 | Store decremented pending and updated sequence to that pair. The pending gate remains; decode's single varying prerequisite has pending=1. | M20 |
| T3.08 | Load submit entry and argument from `D_r+32*h` as one pair. | M19 |
| T3.09 | Enter the bound submit entry. Input-copy/derivation calls execute their supplied CPU function; the resident step enters `mesh_metal_submit`. | M19 |
| T3.10 | Load command and commit entry from `B_rf+96*s` as one pair. | M21 |
| T3.11 | Load selector from `B_rf+96*s+16`. | M21 |
| T3.12 | Tail-call the prepared native commit entry. | M21 |
| T3.13 | After cold lifetime release selects the completed M20 frame, load its native rearm pointer at `+64`. | M20 |
| T3.14 | Load rearm entry from `B_rf+96*s+72`, then call it with that descriptor address. | M21 |
| T3.14a | Store saved x24/x23 at `FP_r-176`. | M25 |
| T3.14b | Store saved x22/x21 at `FP_r-160`. | M25 |
| T3.14c | Store saved x20/x19 at `FP_r-144`. | M25 |
| T3.14d | Store saved frame/link registers at `FP_r-128`. | M25 |
| T3.15 | Load queue from `B_rf+96*s+24`; call native command-buffer creation. | M21 |
| T3.16 | Load add-completion entry and selector from `B_rf+96*s+56` as one pair. | M21 |
| T3.17 | Load completion block from `B_rf+96*s+48`. | M21 |
| T3.18 | Call the prepared native attachment entry with the new command and existing block. | M21 |
| T3.19 | Load previous command from `B_rf+96*s`. | M21 |
| T3.20 | Store the new command at `B_rf+96*s`. | M21 |
| T3.21 | Load encoder entry/context from `B_rf+96*s+32` as one pair. | M21 |
| T3.22 | Invoke the prepared encoder with the address `B_rf+96*s` as its native argument; release the old command through the native object runtime. Native object retain/release calls surround command creation and replacement. | M21 |
| T3.23 | Load saved frame/link registers at `FP_r-128`. | M25 |
| T3.24 | Load saved x20/x19 at `FP_r-144`. | M25 |
| T3.25 | Load saved x22/x21 at `FP_r-160`. | M25 |
| T3.26 | Load saved x24/x23 at `FP_r-176`. | M25 |
| T3.27 | Between probe sweeps, load the call-group pointer from `FP_r-104`. | M25 |
| T3.28 | Acquire-load the call-group running word at `G_r+1196`. | M24 |
| T3.29 | On the probe backedge, load invocation/arrival bases from `FP_r-96` as one pair. | M25 |
| T3.30 | After the cold return sweep, load the arrival base from `FP_r-88`. | M25 |

| T3 work / budget boundary | Account |
| --- | --- |
| Invocation population | Input copy on rank 0 or input derivation on rank 1, then one resident GPU step per slot. Both are included in M18/M19; optional trace consumers add their bound notifications. |
| Polling | At most `2F` cell probes per normal decode sweep; 1 cell load plus the listed M25 base-pair reload per probe; normal sweep also reads M24 running state. Counter gate and range handling are explicitly retained in T3.03–08. |
| Cold lifetime work | The numerical return loop still drains return events, decrements output references, resets frames and invokes T3.13–22 before returning the frame. `link_retire_progress` and its event scan remain unchanged. |
| Native work per rearm | One new command buffer, one attachment of the start-created completion block, and the supplied encoder call; engine `MetalProgram.bind` replays its prepared ICB through a new native compute encoder and native resource declarations. These native operations are work, not zero-cost instructions. |
| D2 accounting | Thread population and per-step launch/rearm are now explicit. The complete 0.5 ms budget remains unestablished while the unconstructed D0 rows and native encode/submit costs remain unresolved. No benchmark claim is made. |
