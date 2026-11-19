# Declared programs (wire 137, layout 156)

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
  another process's pass. From every queue pair's done count, `order_done_through` takes the least piece not done, and
  the done words of every transfer wholly below it are raised in the pieces' order (`order_raise_through`). Beyond that
  prefix, up to the latest piece any queue pair has done, a transfer that is wholly done has its word raised if no
  earlier transfer not yet done names the same word (`order_raise_pass`). A word several transfers count on therefore
  rises only as their prefix, and a transfer held by its gate holds back no other transfer's word. Raisers need no
  agreement: every done count only rises, so each raiser's reading is true of the queue pairs, each word is raised to
  the greatest value any raiser computes (`order_raise`, a compare-and-swap up), and `raised` moves only up
  (`order_raised_to`). Measured the same way: 36 ns a pass.
- **Composition.** Several clients' programs toward one node (a different process each) share its rounds. Each round
  is `count_p` executions of each program, in key order. A program joins or leaves only at a round that every program
  of the segment has reached. Clients agree that round between themselves, ahead of time (section 4).
- **Cable epochs.** A cable coming up mid-run is the one hardware fact that needs the two bridges to agree on a
  boundary. Each end bounds each direction's posting `ORDER_SLACK` pieces past everything it has posted (`limit`,
  stored before the counts are read again: lane_post counts a unit before it reads the bound). Each end sends one ORDER
  naming the cables and its bound, and the epoch begins at the later of the two bounds. The first epoch begins at
  piece 0.
- **Restart.** Losing a cable of the epoch restarts the order: the generation moves, every program is lost (its done
  words go to UINT64_MAX) and both ends' cables are lost. A declaration also asks for a restart when it meets a
  program of a client gone with pieces outstanding or mid-execution, because only a restart realigns the two ends.

## 4. Who declares what (the precondition)

- **One process, several communicators.** All the communicators one process issues on in a known order (micro-batches,
  tensor-parallel groups) form one program. Its table interleaves their calls in the order the code issues them
  (libnccl-mesh: `ncclMeshDeclareBegin`, the calls, `ncclMeshDeclareEnd`). Call k is step k mod period, and a call
  that is not its step fails as a value. A program changes shape only by an explicit, symmetric declaration at the
  same point at both ends. Nothing detects a shape and nothing negotiates one.
- **Steps alongside one another.** Independent work the code runs side by side (one micro-batch's fused crossing
  beside another's, each on a stream of its own) is declared so: `ncclMeshDeclareAlongside` between two groups of a
  declaration puts the next group in the run of the one before it. Toward each node the table is then ordered by run,
  then repeat (a fused crossing's frame, nccl.h `ncclMeshFused` repeats; 0 for a group of one), then step, then capture
  order (nccl-mesh.c `recs_order`): frame j of every group of the run precedes frame j + 1 of any. That order is
  computed from the declarations alone, alike at both ends. The client's obligation is to produce the data in that
  order: tp_fused (gemma-4-e2b-pareto) builds each micro-batch's step and then encodes it frame by frame across them.
  Built one after another instead, MLX's throttle on command buffers in flight held micro-batch 1's encoding until
  micro-batch 0's step had nearly run, so the two never overlapped whatever the wire's order, and with frames
  interleaved on the wire that wait is a deadlock.
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
  - a one-way WRITE plan, an AlltoAllV of uneven counts and two communicators declared alongside (tests/sim-programs.c);
  - two process pairs, each running its own one-step program of 12,000 Broadcasts ahead of its receiver over a queue
    pair served once in 64 turns, the two programs composed by rounds.
- **Python.** The native extraction tests (`test_native_byte_plans`, `test_indexed_write`, `test_diagnostics`) take
  the new production functions.

## 8. What may fall short on hardware

- **The ORDER exchange for a cable joining mid-run** (the bound, its proposal, the extension) is exercised only in
  tests/order.c's in-memory form, not by the simulator's processes or by hardware.
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
