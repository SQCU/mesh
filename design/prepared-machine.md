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
| `S(r,l)` | `Q*ceil(b(r,l)/Q)`, reserved contiguous bytes per invocation of that operand. |
| `Q` | Native payload extent selected during connection setup, fixed for this prepared execution. |
| `c(r,l)` | `ceil(b(r,l)/Q)`, actual native requests for that push. |
| `k` | Native request in `[0,c(r,l))`; private to transport. |
| `j_rx(r,s,l,p,k)` | `first_rx(r,l,p)+s*stride_rx(r,l,p)+k`; canonical receiver row. Both section integers are fixed at binding, independent of SEND arena indices. |
| `P_r(l)` | Setup prefix within varying native requests in SEND order, including declared varying input metadata before layer 0; excludes the once-only prefix and is never loaded by the TX spinner. |
| `j(s,l,p,k)` | `t(r,s,l,k)`, within the peer-specific SEND arena `X_rp`. |
| `n(r), o(r), N(r)` | Native requests per varying slot, once-only requests, and `F*n(r)+o(r)` total, including declared input metadata; fixed by `mesh_transfers_prepare`. |
| `t(r,s,l,k)` | `o(r)+s*n(r)+P_r(l)+k`; the once-only prefix precedes varying requests. |
| `A_rl, U_rlp, O_rlu, T_rp, X_rp, Y_rp, V_rp, Z_r, I_rsl, P_rsl, N_rp, CQ_output_r, TX_output_rp` | Start-bound producer, receive, numerical output, TX cells, SEND records, RECV records, RECV SGEs, canonical page table, operand arguments, GPU publication records, CPU publication records, CQ output and native failure-output bases. All CPU/GPU/NIC aliases are fixed at setup. |
| `CPU_r(offset), GPU_r(offset), NIC_r(offset)` | The three start-bound aliases of the same registered pages; no runtime translation between aliases. |
| `source_row(r,s,l)` | Canonical producer row integer, resolved by the caller’s tensor allocation at setup. |
| `g` | Invocation generation held in the participating native execution and transport registers before the first producer store. It is a stored value, never an address component. |
| `a, h, f, w` | Start-bound numerical notification ordinal, invocation-record ordinal, function identity, and worker index. `a` and `h` include the invocation slot. Normal decode has `2F` notification cells per rank: input preparation and resident step; optional trace calls add their declared cells. |
| `E_r, D_r, C_r, B_rf, K_r, J_rf, G_r` | Numerical notification arena, invocation records, call frames, native command records, worker records, function metadata, call-group allocation; each base is an integer fixed by preparation. |
| Constructor state | `25/25`. M14 is retired. Resident numerical scope: embedding and scaling; the remaining numerical graph and T3 are still per-step. |

| ID / object | Address | Bytes and required `_Static_assert` | Constructed at `start()` by | Read by which event | Loads to reach it at runtime |
| --- | --- | --- | --- | --- | --- |
| <a id="M01"></a>M01 producer output | `A_rl+s*S(r,l)` | Reserved `S(r,l)`; active `b(r,l)`; scalar width `_Static_assert(sizeof(uint16_t)==2,"M01 M02 M03 scalar storage");` in `rdma/mesh-call.h:47`; extent is the start-bound section byte count. | `metal-microbench/mesh_layer.swift:176`; contiguous slot allocation/binding at `rdma/mesh-call.c:688` and `:695` | value-ready | 0 — native producer binding |
| <a id="M02"></a>M02 receive backing | `U_rlp+s*S(p,l)` | Reserved `S(p,l)`; active `b(p,l)`; same scalar assertion as M01; the registered extent is fixed at setup. | `rdma/mesh-call.c:633`, `rdma/mesh-call.c:640` | consumer read | 0 — native consumer binding |
| <a id="M03"></a>M03 numerical consumer output | `O_rlu+s*Q*ceil(bytes(r,l,u)/Q)`; `u` selects the caller-declared output section | Active `bytes(r,l,u)`, Q-rounded backing; FFN sum and finished hidden state each have `2*B*H` bytes; same scalar assertion as M01. | `metal-microbench/mesh_layer.swift:176`, `rdma/mesh-call.c:688` and `:695`; numerical destinations bound at `metal-microbench/mesh_layer.swift:138` | consumer read | 0 — native numerical destination |
| <a id="M04"></a>M04 producer publication cell / TX ring | `T_rp + 8*t(r,s,l,k)` | `8` per entry; `8*N(r)` total; `_Static_assert(sizeof(_Atomic uint64_t)==8,"M04");` | `rdma/mesh-call.c:607`, direct producer binding at `swift/Mesh.swift:657` | TX post | 1 — M16 supplies the cell address; the bounded slot scan uses register arithmetic |
| <a id="M05"></a>M05 SEND dispatch record | `X_rp + 32*j(s,l,p,k)` | `32`: native post entry, native QP, prepared WR address, native failure-output address; `_Static_assert(sizeof(struct prepared_send)==32 && _Alignof(struct prepared_send)==32,"M05");` | `rdma/mesh-flow.c:152` | TX post | 1 — M04 integer |
| <a id="M06"></a>M06 native SEND work request | `X_rp + 32*N(r) + 128*j(s,l,p,k)` | `128`; `_Static_assert(sizeof(struct ibv_send_wr)==128,"M06");` | `rdma/mesh-flow.c:151` | TX post | 1 — M04 integer; its affine address is also fixed in M05 |
| <a id="M07"></a>M07 native SEND scatter/gather entry | `X_rp + 160*N(r) + 16*j(s,l,p,k)` | `16`; `_Static_assert(sizeof(struct ibv_sge)==16,"M07");` | `rdma/mesh-flow.c:150` | TX post | 1 — M04 integer |
| <a id="M08"></a>M08 native RECV work request | `Y_rp+64*j_rx(r,s,l,p,k)` | `64`, aligned 32; native WR at +0, post/QP at +32/+40, retirement row and publication range at +48/+52/+56; `_Static_assert(sizeof(struct prepared_receive)==64 && offsetof(struct prepared_receive,post)==32,"M08");` plus `sizeof(struct ibv_recv_wr)==32` in `rdma/mesh-flow.c:18`. | `rdma/mesh-flow.c:120`, `rdma/mesh-flow.c:174` | RX completion | 1 — canonical row carried by native completion |
| <a id="M09"></a>M09 native RECV scatter/gather entry | `V_rp+16*j_rx(r,s,l,p,k)` | `16`; `_Static_assert(sizeof(struct ibv_sge)==16,"M07 M09");` in `rdma/mesh-flow.c:17` | `rdma/mesh-flow.c:122`, `rdma/mesh-flow.c:171` | RX completion | 1 — completion row; native WR contains the same start-bound address |
| <a id="M10"></a>M10 receive availability record | `Z_r+32*j_rx(r,s,l,p,k)`; generation at `+24`; producer availability uses `Z_r+32*source_row(r,s,l)+24` | `32`; canonical `mesh_page_entry`, asserted in `rdma/mesh.h:39`, including `offsetof(stamp)==24`; arena base and stride are multiples of 32. | Backing at `rdma/mesh-call.c:640`; initial generation `s+1-F` at `rdma/mesh-flow.c:168`; native GPU alias at `swift/Mesh.swift:249` | consumer read | 0 — consumer holds the stamp address; RX addresses it by the completion integer |
| <a id="M11"></a>M11 completion output | `CQ_output_r + 48*p` | `48`; `_Static_assert(sizeof(struct ibv_wc)==48,"M11");` | `rdma/mesh-flow.c:473`; shared native CQ at `rdma/mesh-verbs.h:238` | RX completion | 0 — native poll destination held by transport |
| <a id="M12"></a>M12 consumer operand view / native argument bytes | `I_rsl+32*p`; vocabulary `I_rs_vocab+32*(row*ceil(V/1024)+tile)` | `32`: direct GPU address, element offset, availability address, sequence address; `_Static_assert(sizeof(struct prepared_arguments)==32 && offsetof(struct prepared_arguments,sequence)==24,"M12");` in `rdma/mesh-call.h:28`; Metal `static_assert` in `swift/Mesh.swift:157`. | `metal-microbench/mesh_layer.swift:129`, `metal-microbench/mesh_layer.swift:265`; fixed local/output arguments at `metal-microbench/mesh_layer.swift:138` | consumer read | 0 — native buffer binding; input/stamp addresses loaded before polling, retained in consumer registers |
| <a id="M13"></a>M13 producer publication argument bytes | GPU: `P_rsl+32*u`; CPU: `N_rp+32*u`, where `u` is a start-bound destination ordinal | `32`: destination, value, generation multiplier, reserved; `_Static_assert(sizeof(struct prepared_publication)==32 && offsetof(struct prepared_publication,scale)==16,"M13");` in `rdma/mesh-call.h:43`; Metal `static_assert` at `swift/Mesh.swift:746`. GPU buffers are page-aligned; native array allocated at alignment 32. | `swift/Mesh.swift:657`, `:662`, `:669`, `:673`; native destinations at `rdma/mesh-flow.c:188`, `:193`, final array at `:213`; counts specialized at `swift/Mesh.swift:694` | value-ready | 0 — native binding or retained CPU array base plus ordinal; one record, no header |
| <a id="M15"></a>M15 native post failure output | `TX_output_rp + 8*s` | `8`; `_Static_assert(sizeof(struct ibv_send_wr *)==8,"M15");` | `rdma/mesh-flow.c:87` | TX post | 0 — native output address held by transport |
| <a id="M16"></a>M16 TX cursors | `cursor_rp + 32*s`; cell value is `T_rp + 8*t(r,s,l,k)` | `32F`; `_Static_assert(sizeof(struct prepared_cursor)==32 && _Alignof(struct prepared_cursor)==32,"M16");` | `rdma/mesh-flow.c:103` | TX post | 0 — slot index and cursor-array base held in registers; one cursor load before M04 |
| <a id="M17"></a>M17 thread handles / population | `link_base_r + sizeof(struct mesh_link)*p + offsetof(struct mesh_link,workers) + 8*d`, `d<4`; controller at that link's `controller` offset; polling lifecycle word at `offsetof(struct mesh_link,progressing)` | `8` per pthread handle; 4-byte lifecycle word; `_Static_assert(sizeof(pthread_t)==8,"M17");` | `rdma/mesh-flow.c:377`, `rdma/mesh-flow.c:498`; numerical workers at `rdma/mesh-call.c:470` | TX post | 0 — native thread entry argument. Chosen: one TX thread per link scans at most F flat M16 cursors, one pointer load then one cell load per probe, instead of F spinning threads. Per arena with P links and W active numerical workers: `spinners <= 4P+W+1`, `threads <= 5P+W+1`; the extra 1 is the spinning bridge main, and P controllers block in kqueue. `W<=8`, independent of `MESH_QPS` and F; decode and prefill each configure W=1. The engine constructs both arenas: for the pair, P=1 per arena, total bound 12 spinners and 14 Mesh threads per rank, independent of F. Native driver, Metal and caller-owned threads are outside this Mesh-owned count. |
| <a id="M18"></a>M18 numerical notification cell and publication header | `E_r + 128*a`; the event cell is at `+8` | `128` stride, `8` event bytes; `_Static_assert(sizeof(struct mesh_stream)==128 && offsetof(struct mesh_stream,slots)==8,"M18");` | `rdma/mesh-call.c:280`, fixed worker base at `rdma/mesh-call.c:323` | step rearm | 0 — base plus register cursor times 128; M25 accounts for the prebound-base reloads; no input array, per-input position or mask load |
| <a id="M19"></a>M19 numerical invocation record | `D_r + 32*h` | `32`; `_Static_assert(sizeof(struct mesh_use)==32 && _Alignof(struct mesh_use)==32,"M19");` | `rdma/mesh-call.c:390`; native entry binding at `rdma/mesh-call.c:180` | step rearm | 1 — M18 low event integer; the record includes submit, argument, call, invocation mask and end |
| <a id="M20"></a>M20 call frame | `C_r + 128*(f*F+s)` | `128`, aligned 64; `_Static_assert(sizeof(struct mesh_call)==128 && _Alignof(struct mesh_call)==64,"M20");` | `rdma/mesh-call.c:111`, frame initialization at `rdma/mesh-call.c:135` | step rearm | 1 — prepared M19 call pointer; return processing indexes the same array by its event integer. Pending/sequence at +32/+36; native rearm record pointer at +64. |
| <a id="M21"></a>M21 native command and rearm arguments | `B_rf + 96*s` | `96`, allocation aligned 32; `_Static_assert(sizeof(struct prepared_metal)==96 && offsetof(struct prepared_metal,encode)==32,"M21");` | `swift/Mesh.swift:816`, `swift/Mesh.swift:845`; native methods bound at `rdma/mesh-call.c:178` | step rearm | 1 — M19 argument for submission, M20 native pointer for rearm. First line: command, commit entry, selector, queue. Following lines: encoder entry/context, retained completion block, add-completion entry/selector, rearm entry. The descriptor and initial command exist at start; later command identities are native API return values stored in this same descriptor. |
| <a id="M22"></a>M22 numerical worker state | `K_r + 128*w` | `128`; `_Static_assert(sizeof(struct mesh_call_worker)==128 && _Alignof(struct mesh_call_worker)==128,"M22");` | `rdma/mesh-call.c:90` | step rearm | 0 — thread entry argument; arrival base/count are loaded before polling; cold return-reader state remains here |
| <a id="M23"></a>M23 function lifetime / rearm metadata | `J_rf` (one base integer per function, fixed at binding) | `128`; `_Static_assert(sizeof(struct mesh_function)==128,"M23");` | `rdma/mesh-call.c:114` | step rearm | 1 — M20 function pointer, used by cold refcount/reset work; native submission does not read it |
| <a id="M24"></a>M24 call-group lifecycle state | `G_r`; running at `+1196`, worker array at `+128` | `1280`; `_Static_assert(sizeof(struct mesh_calls)==1280,"M24");` | `rdma/mesh-call.c:85` | step rearm | 0 — captured by the numerical thread at entry; lifecycle polling is outside notification-to-commit instructions |
| <a id="M25"></a>M25 numerical worker / native rearm and RX ABI frames | Numerical `FP_r-176..FP_r+15`; RX `RFP_rp-112..RFP_rp+15`; frame addresses published before `start()` returns | Numerical `192`, RX `128`; `_Static_assert(24*sizeof(uintptr_t)==192,"M25");` and `_Static_assert(16*sizeof(uintptr_t)==128,"M25 RX ABI frame");` pin word extents; compiler frame sizes are recorded in D1. | `rdma/mesh-call.c:194` and `:476`; `rdma/mesh-flow.c:284` and `:384` | step rearm / RX completion | 0 — FP/SP registers. Numerical spills at FP-104/-96/-88; rearm saves from FP-176. RX publication base and generation offset at RFP-104/-96, native bad-WR output at RFP-88. Neither RX spill is read before publishing M10. |
| <a id="M26"></a>M26 resident embedding continuation | `R_r+s*Q*ceil(32*B/Q)+32*b`, `b<B`; `R_r` is the final canonical storage section allocated for decode | `32` per batch row: completed generation at +0, lease stop +8, width +12, slot stride +16, scale +20. `_Static_assert(sizeof(struct prepared_residency)==32 && offsetof(struct prepared_residency,stop)==8,"M26");` in `rdma/mesh-call.h:35`; matching Metal assertion at `metal-microbench/kernels.swift:72`. | `metal-microbench/mesh_layer.swift:176`, initialized at `:22`; native bindings at `:32`, initial dispatch committed during start-time finalization at `:37` | consumer read | 0 — native buffer binding plus fixed batch-row offset. Width, stride, scale and completed generation load once on dispatch entry, remain in registers across steps. One SIMD group per `(s,b)`; one concurrent command per lease. Lease requests exit after 100 ms, with a 2^22-poll finite backstop; every poll checks exit. Renewal preserves completed generations and does not depend on token arrivals. This bounds the new dispatch's work; it does not establish progress of the remaining per-step graph. |


| Fixed native binding | Start-time contents |
| --- | --- |
| M06 | `wr_id=(s<<32)+source_row(r,s,l)` (the row is fixed at setup); `next=NULL`; `sg_list=M07(j)`; `num_sge=1`; opcode `SEND`; only the final request of each partial signals TX completion; flags fixed at setup. |
| M07 | Source `NIC_r(A_rl+s*S(r,l)+Q*k)`; length `min(Q,b(r,l)-Q*k)`; registration key resolved at setup. |
| M08 | `wr_id=j_rx`; `next=NULL`; `sg_list=M09(j_rx)`; `num_sge=1`; post entry and QP fixed at setup. |
| M09 | Destination `NIC_r(U_rlp+s*S(p,l)+Q*k)`; length `min(Q,b(p,l)-Q*k)`; registration key resolved at setup. |
| Native queue | One ordered sequence per `(r,s,p)`: increasing layer, then increasing native request within that layer. Different invocation slots have distinct QPs, cells, cursors and backing; all slot QPs on `(r,p)` share the peer CQ. The single link spinner probes one M16 cursor at a time, rotating across at most F slots. Acceptance advances only that slot to the next native request, then the next layer/tile. Native refusal leaves its cursor unchanged and proceeds to the next slot. |
| M12 | GPU aliases of payload and availability, element offset, and invocation sequence address. FFN loads input/stamp addresses before its poll. Sampler loads its tile record and sequence before its poll. Local contribution and output are separate fixed native buffer arguments. |
| M13 | One record per native TX cell, local availability word, or CPU notification. TX: destination=M04, value=j+1, scale=0. Local availability: value=1, scale=1. CPU notification: destination=M18+8, value=target+1, scale=2^32. Function constants fix record counts; no header, position, mask or reservation. |
| Native execution of M12/M13 | Supplied numerical code and direct bindings; no receipt-time command construction, commit, callback, or operand-address modification. |
| Full operand availability | M10 at `Z_r+32*j_rx(r,s,l,p,c(p,l)-1)+24`; earlier native chunks are in the same ordered native receive sequence. Each completion increments its own M10 by F without a last-chunk branch. |
| Reuse | Each layer has distinct backing. A slot's next decode step uses its next input token, produced after that slot's previous vocabulary consumption. No receipt-time storage selection or occupancy word. |

| T1: value ready → SEND posted | Instruction | Object |
| --- | --- | --- |
| T1.01 | Producer's final system-visible payload store at its already bound output address. | M01 |
| T1.01a | Vocabulary signal loads its M13 TX record at the encoded address; FFN has loaded its record before T1.01. | M13 |
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
| T2.01 | Load native completion status at `CQ_output_r+48*p+8`. | M11 |
| T2.02 | Load completed `wr_id` into `j_rx` at `CQ_output_r+48*p`. | M11 |
| T2.03 | Load previous generation at `Z_r+32*j_rx+24`; add F held in a register. | M10 |
| T2.04 | Release-store that generation to the same address. No intermediate stack reload or binding lookup. | M10 |
| T2.05 | Consumer loads generation at its prebound stamp address; peer selection is register/SIMD arithmetic over the preloaded bindings. | M10 |
| T2.06 | Consumer loads first payload element from its prebound M02 address. | M02 |

| RX continuation after T2.04; independent of consumer's T2.05–06 | Instruction | Object |
| --- | --- | --- |
| T2.07 | Load native post entry and QP at `Y_rp+64*j_rx+32`; invoke with WR at `Y_rp+64*j_rx` and bad-WR output at `RFP_rp-88`. | M08, M09, M25 |
| T2.08 | Load first/end destination ordinals at `Y_rp+64*j_rx+52`. Normal resident FFN/vocabulary receives have an empty range. | M08 |
| T2.09 | For a declared CPU/forward destination range, reload array base and generation offset from `RFP_rp-104`. | M25 |
| T2.10 | Load destination/value at `N_rp+32*u`. | M13 |
| T2.11 | Load generation multiplier at `N_rp+32*u+16`. | M13 |
| T2.12 | Release-store `value+(generation-1)*scale` to that prepared address. | M04 or M18 |
| T2.13 | Load retirement row at `Y_rp+64*j_rx+48`; existing cold buffer/frame release follows. | M08 |
| T2.14 | On polling backedge load link lifecycle word; empty TX probe also reads this word. | M17 |

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
