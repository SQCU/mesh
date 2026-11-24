# Flat Page-Table Async DMA Bridge Specification

This document specifies the flat page-table memory model, register-transfer semantics, wire framing, and C11 client API for `rdma/mesh-flow` and `libmesh` under Apple Thunderbolt RDMA (`TN3205`).

## 1. Scope and Hardware Premises

The bridge is an asynchronous DMA memory container over `/mesh0` that moves 4096-byte pages across a switchable graph of Thunderbolt RDMA links as fast as possible. Clients never see ports, queue pairs, firmware objects, or routes.

Real hardware rules (Apple TN3205, `ibv_post_send(3)`, `ibv_post_recv(3)`, `ibv_poll_cq(3)`, and macOS 26/27 measurements):

- **H1:** `IBV_WR_SEND` / `ibv_post_recv` only over `IBV_QPT_UC` (no RDMA WRITE, no immediate data).
- **H2 / H10:** Work queues are sized in 4096-byte frames (`max_send_wr`, `max_recv_wr <= 4095`). Matching SEND and RECV must cover the same frame count (`f = 1`, 4096 B uniform across all QPs).
- **H9 / H11:** `ibv_post_recv` fixes the receive address before arrival; hardware credit flow control holds a SEND until a matching RECV is posted.
- **H13:** `ibv_post_send` and `ibv_post_recv` accept linked lists of work requests in a single provider call.
- **H14:** A work request's buffer may be reused only after its CQ completion is polled.
- **`max_sge = 1`:** One contiguous `(addr, length, lkey)` per work request.
- **N7:** `tbt_post_recv` uses the low 32 address bits with the extent's `lkey`; `/mesh0` is mapped inside 4 GiB banks and registered in 1 GiB (`2^18` page) `ibv_mr` extents.
- **UC FIFO:** Within each queue pair, SEND `k` fills RECV `k` in post order.

## 2. Cited Algorithms

1. **Unbounded lock-free pending sets:** Linux `include/linux/llist.h` and `lib/llist.c` (`llist_add`, `llist_empty`, `llist_del_all`, `llist_reverse_order`). Multiple producers with `llist_add` and multiple consumers with `llist_del_all` require no locks.
2. **Lock-free refcount from shared table entry:** Linux `include/linux/kref.h` (`kref_get_unless_zero`, CAS refusing `0 -> 1`).
3. **Free-page & contiguous allocation:** 3-level lock-free free bitmap (`l0`, `l1`, `l2`) with Dekker `seq_cst` summary-bit re-check (O(1) scatter allocation; Linux CMA `bitmap_find_next_zero_area` first-fit over `l0` for typed contiguous spans).
4. **Dynamic Storage Allocation complexity guard:** Garey & Johnson `[SR2]` (Dynamic Storage Allocation is NP-complete); typed contiguous allocations are capped at 45% of the arena, aborting with `ERR:NPHARD` above that bound.
5. **Register-transfer notation and memory model:** Bell & Newell (ISP 1971), Heuring & Jordan (RTN 1997), Lamport (PlusCal 2009), ISO/IEC 9899:2011 §7.17 (C11 atomics), POSIX.1-2017 §4.12 (`waitpid` memory synchronization).

## 3. Shared Registers in `/mesh0`

Let `N = (length - data_off) / 4096` be the arena's 4096-byte page count (`data_off` aligned to `getpagesize()` = 16384 B).

| Register | Location | Width | Meaning | Writers | Readers |
|---|---|---|---|---|---|
| `P[0..N-1]` | `data_off + 4096*p` | 4096 B | Raw byte pages. Zero metadata inside data pages. | Clients, `D_p` (RECV) | Clients, `D_p` (SEND) |
| `W[0..M-1]` | `table_off + 8*i` | 8 B | Page table (`M = 2^20`): `0` = free index; `EMPTY` (`0xffffffff`) = allocated empty; `(p + 1) \| (bytes << 32)` = mapped to `P[p]`, `bytes <= 4096`. | Clients, `T_p` (`LAND`) | Clients, `T_p` |
| `u[0..N-1]` | `ref_off + 4*p` | 4 B | Physical page refcount: bit 31 = `CONTIG`, bits 0..30 = hold count `c`. `c == 0` iff free. | Clients, `T_p`, `T_E` | Clients, `T_p` |
| `contig_pages` | `hdr` (`spare`) | 8 B | Count of allocated pages with `CONTIG` bit set (`<= floor(9N/20)`). | Clients, `T_p` | Clients |
| `free_p`, `free_i` | `bmp_off` | `~N/8` B | 3-level free bitmaps hinting `u[p] == 0` and `W[i] == 0`. `u[p]` and `W[i]` are the sole sources of truth. | Clients, `T_p`, `T_E` | Clients, `T_p` |
| `pending` | `hdr` (`attached`) | 4 B | Lock-free `llist` head (`q + 1`, `0` = empty) of client transfer-batch pages. | Clients, `T_k` | Clients, `T_k` |
| `port_q[0..7]` | `switch_off` | 4 B | Per-port `llist` head (`d + 1`, `0` = empty) of wire-ready `DESC` pages. | `T_k`, `T_q`, `T_E` | `T_k`, `T_q`, `T_E` |
| `tx_head[0..7]` | `switch_off` | 4 B | Per-port single-writer local `DESC` queue head, drained by `T_E` after `waitpid`. | `T_p`, `T_E` | `T_p`, `T_E` |
| `lane[L].flight[D]` | `switch_off` | 4 B * 4096 | The one fixed-depth ring per QP (`SQ_j` in-flight page ids). | `T_p`, `T_E` | `T_p`, `T_E` |
| `lane[L].rx_ring[D]` | `switch_off` | 4 B * 4096 | Posted `RQ_j` receive page ids for O(D) dead-child reclamation. | `T_p`, `T_E` | `T_p`, `T_E` |

There is **no `mesh_pd` pool**, **no owner byte array**, **no client write ring**, **no inbound ring**, **no lap number**, **no arrival counter (`landed`)**, and **no release on client exit**.

## 4. Wire Format (`MESH_VERSION = 168`, `MESH_LAYOUT = 200`)

Client pages `P[0..N-1]` are raw 4096-byte pages with no inline headers and no pool classes. On the wire, each port partitions its 8 RC queue pairs (`j = 0..7`) into three fixed-frame classes (`TN3205`, exact `f * 4096 B` SEND/RECV matching, `max_sge = 1`):

- **Class 0 (`j = 0, 1`, `f = 3`, 12 KiB, `MESH_WIRE_INLINE`):** 1-WR inline lane for tiny transfers (`<= 2` unique pages, e.g., 6 KiB). Page `h+0` carries `struct mesh_wire_desc` (`e[i].src_page = slot ∈ {1, 2}`); pages `h+1..h+2` carry raw 4 KiB data pages that land zero-copy via `XCHG(W[dst_idx], pack(h + slot, b))`.
- **Class 1 (`j = 2, 3`, `f = 1`, 4 KiB):** Scattered zero-copy lane (`1 DESC + m DATA` linked `ibv_post_send`, 1 page per WR).
- **Class 2 (`j = 4..7`, `f = 8`, 32 KiB, `MESH_WIRE_BULK8`):** Bulk zero-copy lane (`1 DESC + m DATA`, 8 contiguous pages = 32 KiB per WR). Each 8-page RECV lands 8 raw 4 KiB pages zero-copy into `W[dst_idx_0..7]` and relays zero-copy at 32 KiB WR granularity.

Multicast groups sharing a `src_page` (or 8-page contiguous run) route direct neighbors first, then multi-hop destinations piggyback for free (`MESH_WIRE_SAME_DATA`) on any already-chosen port that reaches them.

```c
#define MESH_WIRE_SAME_DATA 1u
#define MESH_WIRE_INLINE    2u
#define MESH_WIRE_BULK8     4u

struct mesh_wire_entry {
  uint32_t dst_node;
  uint32_t dst_index;
  uint16_t bytes;      /* 1..4096 */
  uint8_t  flags;      /* bit 0: SAME_DATA; bit 1: INLINE; bit 2: BULK8 */
  uint8_t  hops;       /* hop count (dropped past MESH_SWITCH_N - 1) */
  uint32_t src_page;   /* sender-local physical page id (or slot 1..2 for INLINE) */
};                     /* 16 bytes */

struct mesh_wire_desc {
  _Atomic uint32_t next;         /* local llist link (q + 1, 0 = end); ignored by receiver */
  uint32_t count;                /* 1..255 */
  uint32_t origin_node;          /* originating node (cycle avoidance) */
  uint32_t from_node;            /* immediate sender node (cycle avoidance) */
  struct mesh_wire_entry e[255]; /* 16 + 255*16 = 4096 bytes exact */
};
```

## 5. Register-Transfer Steps

`pack(p, b) = (uint64_t)(p + 1) | ((uint64_t)b << 32)`. `EMPTY = UINT32_MAX`.

### 5.1 Allocator and Elective Refcounts

```
AP1  T:   p <- first set bit in M[free_p]                                  [relaxed]
AP2  T:   ok <- CAS(M[u[p]], 0, 1); !ok -> clear_hint(free_p, p); AP1     [acq_rel, fail relaxed]
AP3  T:   clear_hint(M[free_p], p); return p                               [seq_cst on empty word]

AC1  T:   c <- M[contig_pages]                                             [relaxed]
AC2  T:   (c + k > floor(9*N/20)) -> write("ERR:NPHARD"), abort()
AC3  T:   ok <- CAS(M[contig_pages], c, c + k); !ok -> AC1                 [acq_rel, fail relaxed]
AC4  T:   s <- first-fit run of k 1-bits in free_p.l0; none -> sub k, ENOMEM
AC5  T:   for r = 0..k-1: ok <- CAS(M[u[s+r]], 0, CONTIG | 1)             [acq_rel]
          any !ok -> release_range(s, r); retry AC4
AC6  T:   clear_hints(M[free_p], s, k); return s

H1   T:   v <- M[u[p]]; ((v & ~CONTIG) == 0) -> return false               [relaxed]
H2   T:   ok <- CAS(M[u[p]], v, v + 1); !ok -> H1; return true             [relaxed]

R1   T:   v <- M[u[p]]; c <- v & ~CONTIG; (c == 0) -> return               [relaxed]
R2   T:   nv <- (c == 1) ? 0 : (v - 1)
R3   T:   ok <- CAS(M[u[p]], v, nv); !ok -> R1                             [acq_rel, fail relaxed]
R4   T:   (c == 1 && (v & CONTIG)) -> fetch_sub(M[contig_pages], 1)        [relaxed]
R5   T:   (c == 1) -> set_hint(M[free_p], p)                               [release / seq_cst on 0->1]
```

### 5.2 Client Verbs (`T_C`)

```
AI1  T_C: i <- first set bit in M[free_i]                                  [relaxed]
AI2  T_C: p <- back ? alloc_page() : ABSENT; val <- back ? pack(p,0) : EMPTY
AI3  T_C: ok <- CAS(M[W[i]], 0, val); !ok -> (release(p)); retry AI1       [release]
AI4  T_C: clear_hint(M[free_i], i); out_idx[r] <- i

E1   T_C: w <- M[W[i]]; return page_of(w), bytes_of(w)                     [acquire]

HI1  T_C: w <- M[W[i]]; p <- page_of(w); (p == ABSENT) -> ABSENT           [acquire]
HI2  T_C: !hold(p) -> retry HI1
HI3  T_C: w' <- M[W[i]]; (page_of(w') != p) -> release(p); retry HI1       [acquire]

CI1  T_C: old <- XCHG(M[W[i]], EMPTY)                                      [acq_rel]
CI2  T_C: (page_of(old) != ABSENT) -> release(page_of(old))

FI1  T_C: old <- XCHG(M[W[i]], 0)                                          [acq_rel]
FI2  T_C: (page_of(old) != ABSENT) -> release(page_of(old))
FI3  T_C: (old != 0) -> set_hint(M[free_i], i)                             [release]

DF1  T_C: s <- alloc_contiguous(k)                                         (ERR:NPHARD > 45%)
DF2  T_C: for r = 0..k-1:
            p <- hold_index(idx[r]); b <- bytes_of(M[W[idx[r]]])           [acquire]
            (p != ABSENT && b > 0) -> copy b B: M[P[p]] -> M[P[s+r]]
            old <- XCHG(M[W[idx[r]]], pack(s+r, b))                        [acq_rel]
            release(p); release(page_of(old))

W1   T_C: local xfers: hold(src_page); old <- XCHG(M[W[dst_idx]], pack(src_page, bytes)); release(old)
W2   T_C: remote chunks (<= 63, wire cap 255): q <- alloc_page(); hold each src_page; fill M[P[q]]
W3   T_C: h <- M[pending]                                                  [relaxed]  llist.c:29
W4   T_C: M[P[q].next] <- h                                                [relaxed]  llist.c:32
W5   T_C: ok <- CAS(M[pending], h, q + 1); !ok -> h <- read; W4           [seq_cst, fail relaxed]
```

### 5.3 Port Poller (`T_p`)

```
LR1  T_p: n <- poll CQ(completion) -> wc[0..n); error -> child_halt(EIO)
LR2  T_p: (n > 0) -> fence                                                 [acquire]
          for each wc[i] = (qp j, slot s):
            h <- M[rx_ring[j][s % D]]; rx_done[j] += 1
LR3  T_p:   (rx_rem[j] == 0) -> rx_desc[j] <- h; rx_pos[j] <- 0; rx_rem[j] <- clamp(P[h].count, 0, 255)
LR4  T_p:   (rx_rem[j] > 0) -> consume DATA frame h for entry rx_pos[j] and any following SAME_DATA entries:
              dst == self -> old <- XCHG(M[W[dst_idx]], pack(h, bytes)) [acq_rel]; release(old)
              dst != self -> route(dst) -> port q; append to relay_desc[q] (0 copy)
            (rx_rem[j] == 0) -> release(rx_desc[j])
LR5  T_p: flush relay_desc[q] via llist_add to M[port_q[q]] (or local tx_head[p])

SR1  T_p: refill each serving QP j up to D with alloc_page_batch + post RECV (H9, H13)

TK1  T_p: (M[pending] == 0) -> skip                                        [relaxed]  llist.h:218
TK2  T_p: list <- llist_reverse_order(XCHG(M[pending], 0))                 [seq_cst]  llist.h:266
TK3  T_p: route each batch q: all-own-port -> append q directly to tx_head[p];
          else partition by route(dst) into per-port DESC pages (merging SAME_DATA), release(q), llist_add to port_q[r]

SD1  T_p: (M[port_q[p]] != 0) -> append llist_reverse_order(XCHG(M[port_q[p]], 0)) to tx_head[p] [seq_cst]
SD2  T_p: while tx_head[p] != 0:
            pop DESC d; pick least-loaded QP j with SQ room >= 1 + m_data;
            post SEND(j, linked list [d, data_0..data_{m-1}]) (H12, H13); record pages in flight[j]

SC1  T_p: n <- poll CQ(sent) -> wc[0..n); error -> child_halt(EIO)
SC2  T_p: for each wc[i]: p <- flight[j][s % D]; sq_done[j] += 1; release(p)  (H14)
```

## 6. Dead-Child and Crash Safety

- **Killed client (`T_C`):** Before `W5`, `M[pending]` is untouched. After `W5`, `P[q]` is fully populated and every `src_page` is already held `+1`. No permanent gap is possible.
- **Killed port child (`T_p`):** After `waitpid` (`H18`), parent `T_E` walks `flight[j][sq_done..sq_posted)`, `rx_ring[j][rx_done..rx_posted)`, `rx_desc[j]` (if `rx_rem > 0`), `tx_head[p]`, and `llist_del_all(port_q[p])`, releasing every held page once in `O(D + queued)` time.
