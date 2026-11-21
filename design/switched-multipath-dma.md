# Switched multipath DMA (memorandum, 2026-11-19)

The operator's requirements are quoted in design/STATE-20261119.md section 1 (branch compose-20261119), and the
derivation is in STATEMENT-routing-and-switching-20261119.md. This memorandum states the design those leave in place
and the order of work.

## The three scopes
- **The program** declares *what* moves: which pages exist, which node produces each one, and which nodes need it. It
  also decides what changes the computation (TP shares, layouts), and recompiles them from the measured per-rank
  residual. It waits only for its input pages to be present. A consumer that needs contiguity says so, as a typed
  claim, and an asynchronous defragmenter copies its pages into one span as they land.
- **The bridges** decide *how* each piece moves: which cable, which direction, which relay. They decide while the piece
  moves, from live state. They are the only stateful party, and they never wait on the program after a page is
  published.
- **Measurement** judges the result: `tp_fused.py` ms/step with tokens unchanged, each transaction's bandwidth and
  latency mean and variance, and per-cable, per-direction utilisation.

## The algorithm: multipath DMA, switched hop by hop
The operator's statements of fact about this hardware (SEND/RECV only, receives matched in order on each queue pair,
no reset of a queue pair) leave one algorithm in place, close to multipath TCP:

| Multipath TCP | Here |
|---|---|
| a subflow | a queue pair on a cable direction |
| the scheduler | each node's bridge, per piece, as it sends or forwards |
| the data-sequence mapping | the piece's name, carried in the same SEND |

- **A piece's name** is its transaction, execution and page, plus the set of destinations still to be served through
  the node it lands on. The name sits in the landing slot directly beside the page frame, because a request names one
  contiguous range. That makes one DMA crossing per piece: one SEND into one preposted receive. There is no second
  crossing and no reply.
- **No home pages.** A transfer is its pages. A page is present wherever it landed, and the page table resolves its
  name. No receive is tied to a particular page: a piece lands in the next free slot preposted on its queue pair.
- **Forwarding is local.** A node holding a piece with destination set D sends it on for each destination d ∈ D:
  - it chooses the outgoing lane with the earliest estimated arrival at d, computed as
    (backlog + bytes) / rate + latency + dist(next, d);
  - it only considers lanes whose far end is strictly closer to d. Distance is the fabric's shortest delay, a metric
    with positive edges;
  - destinations that choose the same lane share one SEND, whose name carries their subset.

  Every hop strictly lowers the distance, so no piece can cycle. Each destination is in exactly one subset at every
  hop, so each page reaches each destination exactly once. Neither property needs checking at run time.
- **A relay** is a node whose bridge forwards. It needs no client.
- **The bridge's state per node** is one backlog count per outgoing lane, the pieces it holds with their destination
  sets, and the free-slot pool. Its inputs are a page published, a slot landed, and a lane completion. Its outputs are
  posts. A transaction ends at a node once every page it needs is present there. That is the whole machine.
- **All-reduce** is the same mechanism over two kinds of page:
  1. each partial page goes to its reducing owner;
  2. the owner publishes the reduced page once its last partial has landed, and that page goes to every rank.

  There is no barrier between the two: each reduced page leaves as soon as its own inputs are present.

## Work, in order
1. **The switching core, pure** (rdma/mesh-switch.h): the fabric's distances, the local forwarding rule, the name. A
   discrete-event simulator of any fabric (tests/switch.c), run in `make check` on the measured five-node cluster. It
   checks every page reaching every destination exactly once, termination, and no cycle. It reports completion time
   against a lower bound and direct-only sending, and per-lane utilisation, for the all-gather and the all-reduce at the
   decode sizes.
2. **Region and wire layout:** landing slots of name plus frame, receives preposted into free slots, the pool sized by
   the declaration.
3. **The bridge:** the core in mesh-link.c, taking the fabric from the bridges' records and measured rates.
4. **The library:** declarations of pages and contiguity claims; consumers wait for presence.
5. **Delete what this replaces, in the same change:** the shared pair order, segments and SEGMENT, rounds composition,
   agreement restarts, the gone-client restart, declaration-time route tables and origin lowering (branch
   scheduled-tables-20261119), the relay-only rank client, and their tests and docs.
6. **Tests:** the N-node simulator in `make check`.
7. **Deploy by git, then measure** `tp_fused.py` at B = 2, 8, 32 and 128, at TP-4 and with all 5 nodes. Tokens must
   equal the single-node run's. Report ms/step against 4.15 / 5.27 / 7.17 / 12.1, the triplet, and per-cable
   utilisation.
8. **Then:** the saturation demonstration, TP-share rebalancing, and the H200 Pareto sweep.

## Where the work stands (2026-11-20)
- Items 2-4 are in rdma/mesh-switched.h, driven by the port children and the parent in mesh-link.c. Pieces are
  staged into a pool of slots (name frame, then page frame) and drawn per hop from the routing weights. Relays
  forward by name, delivery goes by key into the receiving program's span, and a receive window of credits bounds
  each sender to what its receiver's pool holds. The weights are refreshed off the hot path as a residual toward a
  policy's target (Thompson sampling by default; struct mesh_switch_policy is where the trained policy plugs in).
  Each draw is logged in rdma/tests/route's record format (mesh-switch-log drains it).
- Item 5: ORDER, SEGMENT, epochs, segments, rounds composition, agreement restarts, the gone-client restart and the
  declaration-time route tables (mesh-write-routes.h, ncclMeshCommWriteRoutes) are deleted. Still there: the
  relay-only rank support in libnccl-mesh, the Python controller's route weights in its fluid model, and libmesh's
  generic multi-hop lowering (mesh_write_compile, mesh_routed_compile), which nothing compiles routes into now.
- Item 6: tests/sim-switched.c (five nodes, timed wire) is in `make check`, and tests/sim-two-nodes.sh runs over it.
