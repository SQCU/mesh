# Declared programs (wire 140, layout 158)

Every unit on every wire is placed by a pure function of declarations that both ends hold alike. The placement is
computed ahead of time, so at run time a client raises only one monotone counter. This replaces the shared order of
[node-addressed-transfers.md](node-addressed-transfers.md) section 2. In that order, each transfer was appended at run
time to a 4096-slot ring per peer node and direction, slots were taken by compare-and-swap, and per-queue-pair walkers
read the transfers back out of slots that were later reused. That document's section 1 (what clients name: a node and
bytes) still stands.

## 1. Why

Ring position decided every piece's placement at run time. Under run-ahead load the two ends credited units to
different transfers. A fused all-gather hung at 36 of 48 pieces while its sender was complete and every queue pair's
posted count equalled its done count at both ends. Other runs, among them a TP-2 decode run, failed with
`IBV_WC_LOC_LEN_ERR`. The repro is ~/dox/models/gemma-4-e2b-pareto/coll_pct.py: a root running about 3000 Broadcasts
ahead of its receiver.

The operator's requirement (2026-11-19), and the coordinator's notes on it:

- The placement and order of every unit are a pure function of shared declarations, computed ahead of time.
- There is no ring, no slot CAS, no retirement or anchor, and no EAGAIN.
- Backpressure is only what the hardware imposes: posted receive capacity and the gates' cumulative counts.
- No implicit state machine or scheduler is hidden in walkers, polling loops or handshakes, and nothing is designed
  for callers treated as unknown code. The programs run are linear algebra, and their transfers appear in our own
  code.

## 2. The contract

**A program** (mesh.h `struct mesh_declaration`, `struct mesh_crossing`) is what a client declares toward a node,
once:
- Its period: the steps in one execution.
- Its key: equal at both ends.
- Its count: executions a round (section 3).
- For each direction (0: bytes this node sends, 1: bytes it receives), an immutable table of transfers.

Each transfer in a table has:
- `units` spans in the client's own leases, each at most one piece (one allocation block).
- `first`: the pieces of the table before it in one execution.
- The step of the period it belongs to.
- A gate: SEND span s goes once its word reaches `gate_base + s + 1 + e·gate_step` (with `MESH_GATE_WHOLE`, `gate_base
  + e·gate_step` for every span). A receive goes once its word reaches `gate_base + e·gate_step`.
- A transfer with `gates` terms addresses an array of `(word, threshold)` pairs at `gate`. Each term supplies its own
  base threshold. The same execution and span increments apply to each term. The posting lane reads these operands
  directly from their lease and posts once all have reached their thresholds. A failed prerequisite marks the
  transfer's completion word lost and holds its payload.
- A done word, raised to `done_value + e·done_step` once every byte of execution e has crossed.

The peer's client declares the same program with its directions swapped, at the same point of its own code. Nothing
about any transfer is sent.

**Declaring** (mesh-call.h `mesh_declare`): the client's program toward the node before this one ends at the steps
it issued, which must be a whole number of executions. The node's next segment then holds the programs that go on
and the new one. Declarations toward a node are made one at a time, through the node's `declaring` word.

**At run time** the client raises one counter, the steps it has issued (`mesh_issue`), and nothing else.

**Transfer numbering.** Transfer t of a direction is entry t mod n of its table, in execution t / n. A step k is in
execution k / period.

## 3. The placement (mesh-order.h)

- **Queue pair of a piece.** The epoch holding piece n is the latest whose `first[d]` is at most n. Piece n of
  direction d is on queue pair `(n − first[d]) mod Q` of that epoch, where Q is the epoch's queue pairs. They are taken
  lane by lane across its cables (`order_qp`), so consecutive pieces cross every cable before any cable's second queue
  pair. Unit u of a queue pair is therefore `first + q + u·Q` in closed form (`order_unit_piece`), summed over the
  epochs that name its cable.
- **Program, execution, transfer and span of a piece.** The node's pieces are its segments in turn. A segment holds
  rounds from its `start[d]`. A round is `count` executions of each of its programs, in key order (`order_locate`:
  the segment from a hint, then the program, then the execution, then a binary search on the table's `first`). The
  program, execution, transfer and span of every piece are therefore fixed by the declarations alone.
- **Posting.** A queue pair's next unit is posted once three things hold: its step is issued, its gate holds and the
  lane has room (`order_next`, `order_ready`). The lane then waits at its own next piece. Nothing is searched for,
  skipped or retired. Measured with a throwaway loop over `order_next` on an M-series Mac (not committed): 5.8 ns a
  piece, from unit index to a posted span's address.
- **Done words.** Every port's child raises them, in each pass right after it reads its completion queues
  (mesh-link.c `order_done`), so the process that sees a completion first raises its words and no completion waits for
  another process's pass. Each port maps the shared queue counters to piece positions when the cable set changes
  (`order_project`). As counters advance, `order_progress_read` computes the first unfinished piece.
  `order_raise_through` publishes each complete transfer below that position and advances `raised`. Past it, each done word has a cursor of its own (`struct order_raiser`) and rises through its own
  transfers that are wholly done, passing over other words' transfers. A word several transfers count on therefore
  rises only as their prefix, and a transfer not yet done holds back its own word alone: one communicator's words never
  wait for another's transfers (the earlier look-ahead of 64 transfers coupled them). Each port sizes its cursor
  table from the distinct completion words in the declarations. Every done count only rises,
  each word is raised to the greatest value any raiser computes, and `raised` moves only up.
- **Posting is bounded by the issued steps.** A piece of execution e is posted only once its program's count of
  issued steps passes it, whatever its gate word holds, so the pieces either end posts are those of executions its
  client issued, and a program's end closes there. (81bc7e0d let a gate word stand for the issue on the Metal path;
  a communicator's gate word, a lease's prior contents, read far ahead, so receives of executions that never ran were
  posted, and the next, larger program's SENDs met them: IBV_WC_LOC_LEN_ERR on hardware, reproduced by
  tests/sim-two-nodes.sh's programs in turn.)
- **Composition.** Several clients' programs toward one node (a different process each) share its rounds. Each round
  is `count_p` executions of each program, in key order. A program joins or leaves only at a round that every program
  of the segment has reached. Clients agree that round between themselves, ahead of time (section 4).
- **Cable epochs.** A cable coming up mid-run is the one hardware fact that needs the two bridges to agree on a
  boundary (mesh-order.h `order_agree_turn`). The end of the lower node number proposes the cables it holds up whenever
  they are not its latest epoch's, with its bound each way. It sets `limit` to the current epoch's first piece before reading the lane posting
  counts, then opens the range through the highest piece already posted. Both ends' bounds thus protect work in
  flight. When cables join before either end posts, both ends stripe from piece zero over the agreed cables. The other
  end takes the latest proposal and accepts it once every cable it names is up there. It bounds its own posting,
  appends the epoch from the later of the two bounds each
  way, and answers ACCEPT with its bound and the proposer's echoed. The proposer appends what the ACCEPT names. An end
  appends only a proposal its peer accepted, and the peer accepts one proposal an epoch, so both append the same
  epochs whenever cables come up. (The earlier symmetric exchange let one end append an epoch over one cable while the
  other, three cables up by then, never did: seen on the first deploy.) The first epoch begins at piece 0.
- **Restart.** Losing a cable of the epoch restarts the order: the generation moves, every program is lost (its done
  words go to UINT64_MAX) and both ends' cables are lost. A declaration also asks for a restart when it meets a
  program of a client gone with pieces outstanding or mid-execution, because only a restart realigns the two ends.
- **Segments agreed** (wire 140, layout 158; mesh-order.h `order_segments_turn`). The two ends place alike only while
  their clients keep the contract, and a death breaks it: a client gone at one end while its peer client lives on at
  the other leaves one end's next segment closing the program and the other's keeping it going. Every piece is then
  placed apart, and both ends' GPUs wait their deadlines for pieces that never come (the M5 Ultra pair, 2026-11-19: a
  REAL=1 run declared while a dead logit_check's peer lived on, both ranks waiting at entries 16 and 0). So the ends
  compare what they place. An end's segments in turn, less each one closed before any of its rounds (clients sharing a
  node may declare in a different order at each end and join only at a round both reach, so such a segment places
  nothing), are its placing segments, the n-th of each end the n-th of the other's. Each end tells the peer each placing
  segment's digest in a SEGMENT: the order's session (epoch 1's first cable's identity, new at every restart), the
  segment's place, its digest (its start, round and programs: keys, bases, counts, periods, tables), whether its clients
  issued a step in it, and, once closed, its rounds; the open segment's again as it changes. A segment's pieces are
  posted only once the peer's digest at its place is the end's own (`agreed`, `order_next`). Where the ends cannot
  come to agree (their segments at a place differ and both ends' clients issued in them, or the peer's closed at fewer
  rounds than this end's clients began in it), the end restarts the order, and both ends' clients fail as values.
  The cost is a control message a declaration, before its first piece.
- **A piece never sent.** A program closed at a whole number of executions may still hold a piece its client never
  published (its SEND gate never reached: a fused crossing's frame whose producer never ran; logit_check left its
  last step's arg-max frame so, its output never evaluated). That piece holds the queue pair it is on, so every program
  after it waits behind it (the M5 Ultra pair, 2026-11-19: three of six logit_check pairs waited their deadlines at
  entry 0 on a settled order, `raised` one piece short of the closed decode's end). A declaration decides it from the state alone, no clock (mesh-order.h order_gate_shut): a piece of a
  gone client's program, closed or going on, whose gate is not open can never be posted (only its client opens its
  gates), so the declaration asks a restart and fails as a value at once (mesh-call.c declare_held), and the run after
  it runs; every other piece of it is posted or postable and completes or fails through its own completion, and the
  next program's pieces follow it on their queue pairs, so nothing is waited for.
- **Recovery.** A communicator made while an order toward a member's node is being made again (a node slot held, not
  live: its cables coming up, its epochs agreed) waits for it before its bootstrap, at every rank alike, and is made
  once it is live, not refused with ENOLINK; its bootstrap's phases then begin with the orders live, none of their
  deadlines spent on the recovery. What ends the wait is the order's state (`live`); its bound, 30 s
  (`NCCL_MESH_ORDER_WAIT_NS`), is only a recovery that never comes. A restart then loses the run it happened in, not
  the ones after it.

## 4. Who declares what (the precondition)

- **One process, several communicators.** All the communicators one process issues on in a known order (micro-batches,
  tensor-parallel groups) form one program. Its table interleaves their calls in the order the code issues them
  (libnccl-mesh: `ncclMeshDeclareBegin`, the calls, `ncclMeshDeclareEnd`). Call k is step k mod period, and a call
  that is not its step fails as a value. A program changes shape only by an explicit, symmetric declaration at the
  same point at both ends. Nothing detects a shape and nothing negotiates one.
- **Counted calls.** An AlltoAllV captured by a declaration gathers its ranks' counts over the bootstrap there
  (`prepare_counted`), once for the program's life: a declaration is a point of configuration every rank reaches alike.
- **Separate processes toward one node** compose by rounds. Each one's count is part of its declaration, and the round
  at which a program joins or leaves is the clients' own agreement, reached through their own atomics, barriers or
  semaphores. The bridge discovers nothing at run time. tests/sim-runahead.c shows one such agreement: a barrier of
  files after declaring, and another before ending.
- **Configuration is not a program.** Split and Grow exchange over the communicator's bootstrap, and so does Barrier
  in mesh_mpi and torch-mesh (`ncclMeshCommConfigGather`). None of them is a step.

## 5. What was deleted

- **Region.** `struct mesh_slot`, `MESH_RING` and the rings' arena lease (`mesh_bridge_pages.rings`), `struct
  mesh_gate` and multi-gate transfers, and `mesh_crossing_valid`. `mesh_order_append` and `mesh_order_outstanding`.
- **Bridge.** The walkers (`struct order_walker`, `order_walk`, `order_anchor`), the slot completion counts
  (`order_completed`, `order_publish`, `ORDER_WINDOW`), retirement (`order_retire`, `retired`, `anchor`, `anchored`,
  `retiring`) and `order_rings_new`. Also: the freeze and its acknowledgements (`freeze`, `ack`, `frontier`),
  `orders_unfilled`, the per-lane slot maps (`slots`, `lanes_room`) and the old `order_place` rule.
- **libnccl-mesh.** `room_for`, `room_wait`, `channel_present`, the append loop and `present_group`. Group-set-aside
  (`aside`) and the RDMA all-gathers of Split and Grow.
- **Clients.** The peer service's ring appends. The quiet probe's per-round appends and its EAGAIN loops.

## 6. Literature

Re-read for this document: none. Each entry below is recalled, and says where a detail matters.

- **Bulk-synchronous parallel (Valiant, CACM 1990; recalled).** Computation proceeds in supersteps whose
  communication is fixed before the barrier that ends them. A declared execution is a superstep whose crossings are
  known in advance. This design keeps BSP's fixed communication per step but not its barrier: run-ahead past it is
  bounded only by the gates and by posted receive capacity.
- **MPI persistent requests and MPI-4 persistent collectives (MPI_Send_init/MPI_Start, MPI_Allreduce_init; recalled).**
  A communication is set up once and started many times, so per-start overhead drops to a start. Our "raise one
  counter" is that start. Unlike MPI, the counter starts a whole period of operations, and the matching order is
  fixed by declaration rather than by MPI's per-communicator ordering rule.
- **Static scheduling of collectives (MSCCLang, PLDI 2023; TACCL, NSDI 2023; Blink, MLSys 2020; recalled).** A
  schedule is compiled once and every rank executes it identically, with no run-time agreement. The placement
  function here is such a schedule at the bridge's level, evaluated in closed form rather than stored.
- **Synchronous dataflow (Lee and Messerschmitt, Proc. IEEE 1987; recalled).** Fixed production and consumption rates
  per firing let a schedule be computed at compile time, as a periodic schedule whose buffers are known. A program's
  period, and the rounds composition with `count_p` executions per program per round, are this idea. A round is SDF's
  repetition vector, written down by the clients instead of solved for.
- **Two-sided verbs with pre-posted receives (InfiniBand practice; recalled; see node-addressed-transfers.md section
  1).** The n-th SEND on a queue pair fills the n-th posted receive. Everything here follows from making that n-th
  receive the same declared piece at both ends.
- **CUDA Graphs (recalled).** A captured stream of work is replayed with its buffers fixed. libnccl-mesh's
  `ncclMeshDeclareBegin/End` is a capture of the crossing groups, and every later group is checked against its
  captured step rather than replayed blindly.

## 7. Testing

- `make -C rdma all` (every warning an error) and `make -C rdma check`.
- **tests/order.c** runs the placement in memory, over tests/wire.h's simulated wire. The wire runs the bridge's own
  steps and ends at any SEND that meets a receive of another piece, transfer, execution, program or length. The test
  covers:
  - a period-4 program of mixed shapes, run ahead with gates opened in different orders;
  - a second client's program joining at a round boundary;
  - a cable joining mid-run;
  - 17,220 transfers with one queue pair served once in 64 turns;
  - redeclaration at an execution boundary;
  - prefix done words;
  - refusals (a span past a piece, a declaration mid-execution);
  - the restart a gone client's outstanding program asks for.
- **tests/sim-two-nodes.sh** runs two nodes' regions over the same wire, each served by the bridge's steps:
  - every tests/peer-groups.c mode;
  - libnccl-mesh communicators (1 and 2 in one program) at 6 KB, 98 KB and 1.5 MB, every value checked;
  - the same through mesh_mpi (tests/sim-mpi.py);
  - a Metal-path program (tests/sim-metal.m);
  - a one-way WRITE plan and an AlltoAllV of uneven counts (tests/sim-programs.c);
  - two process pairs, each running its own one-step program of 12,000 Broadcasts ahead of its receiver over a queue
    pair served once in 64 turns, the two programs composed by rounds;
  - restarts (sim-wire `RESTARTS`: the bridges' parents run as order_turn runs them, cables coming up in turn and
    epochs agreed by ORDER over a message queue, segments by SEGMENT, every restart losing every cable, clients gone
    retired): a client gone at one node as the other loses a cable, then a pair made at once (it waits for the order
    and runs, cables joining mid-run); a client gone at one node while its peer lives on, then a pair declared
    meanwhile (it fails as values at both ranks, no deadline met), then one more (it runs); a declaration at a node
    whose bridge sees the other's restart 300 ms late, then one more; a program left with a piece never sent
    (tests/sim-dangling.c), then a pair (fails as values), then one more (runs); a gone pair's last pieces still
    crossing (sim-dangling inflight, 200,000 executions), then a pair at once (runs behind them, no restart). No pair may wait a deadline, and the
    wire's check must stand.
- **Python.** The native extraction tests (`test_native_byte_plans`, `test_indexed_write`, `test_diagnostics`) take
  the new production functions.

## 8. What may fall short on hardware

- **The ORDER exchange for a cable joining mid-run** (the bound, its proposal, the extension) is exercised in
  tests/order.c's in-memory form and by the simulator's restarts (cables joining under a running pair), but over the
  simulator's message queue, not the bridges' sockets.
- **Striping.** Consecutive pieces stripe over every queue pair (all 6 on the M5 pair), not one lane a transfer per
  cable as before. A transfer's latency is therefore its slowest queue pair's. A transfer held by its gate holds every
  queue pair its pieces are on, and a transfer longer than Q pieces is on all of them. tests/peer-groups.c's
  late-receive-window mode now expects the second group to complete after the first.
- **Several raisers.** Every port's child now raises done words. The simulator runs one raiser a cable in turn
  (tests/wire.h), not concurrently. A restart racing a child's raise has the window it had with one raiser, now on
  every cable.
- **Program buffers are fixed per step.** Callers that allocate a collective's buffers anew each call fail with
  ncclInvalidUsage: torch.compile graphs replayed through `torch_mesh.record`, `partition`'s ops, examples/chain*.py and
  lstm-service. They need a declaration over tensors they keep. xonotic/solver/worker.py's crossings are data-dependent
  (each request's `rows[:n]`, and a solve's shapes follow n), so it has no program to declare until its crossings are
  of a fixed size.
- **Process composition.** A program of a client that died mid-execution is only realigned by a restart, which loses
  every cable toward the node. Separate processes sharing a node must agree their rounds. One that issues more rounds
  than its partner programs waits for them.
