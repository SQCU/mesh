# Change request: pages queued per node, carried by the bridge over every link

The operator's statements of 2026-11-03 on the bridge and the collective, verbatim; the state of the code they change
(mesh b89e6362, metal-microbench f82ec268); and the data structures and functions in scope.

## The operator's statements

### Collective size

> "What we don't have:
> - Any E2B number on a Studio pair."
> buddy pairs don't exist anymore, it's already been said that all code now must operate on collectives of arbitrary
> node count and there won't be any compromises here.

### Relays

> "8. Relays. A group whose members aren't all cabled to each other needs a non-member to forward its traffic. "
> routing within tb5 linked collectives is supported; attempting to involve nodes with no tb5 dma connection to any node
> within the collective is not allowed. do not expect for every node to be directly connected to all other nodes
> without the use of other nodes as relays; this emerges from the extremely small limit on the number of inbound links
> per studio or mini (6,3)

### Interleaving every transfer

> "4. Several collectives sharing cables at once, for example a tensor-parallel all-reduce while gradients reduce.
> Proposal: route them jointly with the ported concurrent-flow solve (Garg–Könemann), rather than each in isolation."
> this, btw, is one of the things we really ogtta do... the point of having aot schedulign and compilation of schedules
> is to allow us to interleave any combination of transport requirements concurrently with one another, s.t. we are
> never underusing the bandiwdth on any cable routing when we could counterfactually swizzle together more
> communications to cause pipeline-blocking operations to begin or conclude faster, no matter waht they are!

### Queue pairs and links belong to the bridge

> "5. The queue-pair budget (3 on the Studios) caps the streams on a cable. Proposal: time-share a cable between
> collectives through one transfer order per link, " find previous discussion on this topic in this repo and the basic
> misidentification of whether 'queue pairs' or 'streams' or any attempt to expose ortransduce the global object
> representing the literal connection between two physical and different-clock-domain literal computers are ever to be
> exposed to collective or data processing code

Earlier statements found in that search:
- 2026-10-19: "it is the job of hte bridge to negotiate stuff like the queue pairs."
- 2026-10-23: "why would any agent think they need to expose global api objects like queue pairs to anything outside of
  the bridge service this time"
- 2026-10-23: "clients don't get to compile 'orders'. only pages->pages scatters."
- 2026-10-23: "the bridge is the one that knows about queue pairs after all, so it might only possibly be the bridge's
  responsibility to decide how to swizzle and tile transactions which can use multiple links at a time to reach a
  target."

> "- Not inlining dangerous code. Not met. The new arm contains pieces nobody has qualified:
>   - it creates queue pairs until the device refuses, up to 64, to learn each device's budget, with unknown side
>     effects on the Studios;"
> … why are you hypothesizing what the hardware does when the failport line contians complete documentation of how the
> collective code *must* be written at the bridge level in so many ways? why are you trying to write code of this form
> at all? do you think you need to write original auteur code that has never been seen ebfore or uesd before in the
> context of supporting a rdma driver that has only one acceptable calling style by design bc it is a piece of commodity
> hardware we are stuck with?
> "  - it destroys queue pairs while a peer may still be posting, which may or may not upset the driver;"
> just btw this is a total noncomprehension of what a 'queue pair' is, which is kind of why the codebase was originally
> segmented into a 'bridge' object that opaquely presents the effects of the rdma standard without allowing collective
> code or data processing code to interact iwth the low level abi which supports no creativity,discovery,
> self-actualization, or authorship in the exercise of an extremely limited and single-correct-use-api.

### The E2B code, the bridge and two layers

> do you think you're being told to invent some totally new test, or being told to change the e2b running code so that
> it can interact with a bridge connected to some number of peers within a total collective of 5 machines? is the bridge
> being totally deleted or is the bridge also required to present a usable interface for collective operations over 5
> total machines and some number of immediately tethered peers?

> "It has to present a usable interface for collective operations across all five machines. " is this a vague topic
> that involves inventing a totally new piece of code that has never been written before or is this rote, mechanical,
> and totally unoriginal?
> "Today it binds to one peer's queue channel." what did the user write earlier about 'queue channels' as a prohibited
> object for data processing or collective code of any kind? What role does the bridge now implicitly or explicitly
> have as a router when there are many tb5 interface edges active per node and many nodes in the overall collective? it
> is clear that many requiremetns are being introduced all at once, but requirements that only relate to 2 specific
> layers of abstraction in the code...

### The page queue

> collectives clearly need to submit work to a runtime. some part of the runtime needs to enqueue and dequeue pages (the
> unit of memory used in this repo btw) s.t. they are transmitted to their final destination within a collective as
> fast as possible. this means using absolutely all inbound links (and perhaps also using routing across intermediate
> nodes as switches in some networking solutions to a scatter operation) which connect a sender and receiver, not using
> 'queue pairs'. no code but the bridge can actually be responsible for transmitting data, so it is clear that there
> must be some kind of zero copy queue, consumption of the queue by the bridge's functions for interacting with the send
> and receive dma interfaces, and that this is very very very boring code, which might have gone unspecified because of
> how basic and simple it should have been (and how implicit the requirement for its writing.). nevertheeless, this very
> boring abstraction for maximizing use of interconnect totally conflicts with any code which would attempt to pass
> 'queue pairs or anything representing them' to collectives or data processing code, for several obvious reasons, as
> any code which was misplacedly interacting with hardware links would now need to be an inlined streaming concurrent
> multiplexer in addition to previous misplaced inlining of networking.

### Replacing the --link form

> "Deleting now means main's only collective program, the E2B pair driver, stops working until item 2 lands." you
> don't hve the discipline to delete code then implement replacements. this has been proven many times.
> "Meeting it means deleting the --link form in the same change, so no flag, environment variable or code path can reach
> it."
> well you can't do that so the full sentence needs to be appraised.
> "s.t. there are no fallbacks and only one mode is ever in force"
> this can clearly be satisfied in semantics by finishing the implementation of the replacement 'modality' (the one that
> isn't '--link'), ensuring it *covers all features and requirements the previous implementation covered*, and only
> *then* removing the now-ablatd obsolete code. attempting to front run deletion ebfore impolementation is such an
> obvious mistake at this point that it should not be repeated. no 'fallbacks' means that only one sort of
> configuration can run within one runtime, with no inversion of control or callback-decided-use of one implementation
> or the other, with, e.g., inlining of handlers for both in all callsites forever bc a subagent was told to do
> something evil like 'retain the --link form of course'.

On how parity is shown:
- "1. Inventory before anything else, kept in the repo. It lists everything the --link bridge does, " — No.
- "2. Implement, then check parity, then delete, in that order and as separate steps. " — No.
- "3. Parity is shown by the normal program, " — yes.

### The bridge's process lifecycle

> "- Left behind: the children, the supervisor, ' why do you think you can get rid of the hcild processes and why do
> you think all of the work in the failport here was incorrect? code that doesn't respect sigterms shouldn't run. code
> that habitually breaks kernel state and requires full system restarts shouldn't be run or ablated just for fun either.

> "What the history actually shows: the failport fixed those failures one at a time, and its final form had the clean
> record." then you have to verbatim reuse that process lifecycle code since you have no alternative to present or cite
> which sin't odious and a waste of time to test or run.

### Failing by error, not by hardware fault

> … this is to be addresed by not writing bad code and deleting bad code / revising it s.t. launches fail-by-error
> rather than fail-by-hardware-fault while more detailed wrok is ongoing.

> some spinwaits are allowed, as a treat, if we know they conclude in bounded time even if someone messes up their
> complementary async event originators

> "So the bridge main runs today has a machine-crash path." so delete a function header s.t. the code fials on compile
> instead of failing at runtime, then do the reading and background i just told you to do.

## State of the code

- The bridge, mesh `rdma/mesh-flow.c`, is started for each run with `-I <node>` and one `--link
  device,peer,local-address,remote-address[,service]` per cable. metal-microbench `tools/mesh/run_pair.py` starts it on
  two nodes (`start_bridge0`, `start_bridge1`, `bridges_up`, `bridges_down`) over the one link `discover_link` and
  `links()` choose to the one peer `MESH_PEER` names.
- The teardown `down_pair` (mesh `rdma/mesh-verbs.h`) has its header deleted (mesh b89e6362), so `rdma/` does not
  compile.
- The region keeps one transfer list per link, queue pair and direction: mesh `rdma/mesh.h` `mesh_order_length(m, owner,
  queue, direction)`, indexed `link * qps + queue`, with the geometry fields `links` and `qps`.
- Clients name a link and a queue pair: mesh `rdma/mesh-dataflow.h` `mesh_peer_channel(c, peer, queue)` and
  `mesh_publish_bind(c, row, queue)`; mesh `rdma/mesh-call.h` `mesh_transfer_bind(c, queue, receive, identity, section,
  invocation_pages)`.
- Data code calls them directly: metal-microbench `tools/e2b_replay.m`, lines 497-500,
  `mesh_transfer_bind(..., mesh_peer_channel(..., peer, queue), ...)`.
- The collective: mesh `rdma/mesh-collective.h` `struct mesh_link_map` (kinds), `mesh_link_map_read` (from a file),
  `mesh_allreduce_plan` (direct exchange, ring, or reduce to a root and broadcast, by the map's kind) and
  `mesh_allreduce_bind` (binds the steps onto `mesh_transfer_bind`).
- The E2B programs are prepared for two ranks: `tools/mesh/run_pair.py`'s `--split-dir` default
  `.build/e2b-ffn/arms-20260923/decode-tp2-{split}`.
- In progress, uncommitted in mesh `rdma/`: a resident bridge that pairs its own ports and keeps a map of the fabric
  (`mesh-flow.c`, `mesh-verbs.h`, `mesh-stat.c`, new `mesh-bridge.h`, `mesh-map.h`, `mesh-graph.h`). The failport's
  process lifecycle (tag `pre-return-20261103`: `rdma/mesh-heartbeat.c`, and in `rdma/mesh-link.c` the per-port child
  processes, their stop, reaping, flush and exit, and the holder table) is being brought into it verbatim.

## Scope

Two layers.

The bridge, mesh `rdma/`:
- `mesh-flow.c`: `link_prepare`, `link_send_progress`, `link_receive_progress`, `link_close`, `link_run`;
- `mesh.h`: the per-link, per-queue transfer lists (`mesh_order_length`, the geometry's `links` × `qps` lists);
- `mesh-verbs.h`: pairing and teardown.

The collective, mesh `rdma/`:
- `mesh-collective.{c,h}`: `mesh_link_map`, `mesh_link_map_read`, `mesh_allreduce_plan`, `mesh_allreduce_bind`;
- `mesh-dataflow.h`: `mesh_peer_channel`, `mesh_publish_bind`;
- `mesh-call.h`: `mesh_transfer_bind`.

Their callers in metal-microbench:
- `tools/e2b_replay.m`, lines 497-500;
- `tools/mesh/run_pair.py`: `discover_link`, `links`, `start_bridge0`, `start_bridge1`, `bridges_up`, `bridges_down`;
- `tools/mesh/grid.py`: preparation for two ranks.

## Claude's proposal (2026-11-03)

This section is Claude's, not the operator's. With it, this document is the unit of work.

### The page lists (the queue)

- Each node's region holds its page lists. The collective writes them, once a plan, and the bridge alone reads them.
  They replace the per-link, per-queue transfer lists (`mesh.h` `mesh_order_length`, the geometry's `links` × `qps`).
- An entry is one hop of one page: `{ page, node, direction, word, position, serial }`.
  - `page`: a page of this node's region. It is the page sent, the page a receive lands in, or a relay's page.
  - `node`: the adjacent node of this hop, by its index in the map: the next hop for a send, the previous hop for a
    receive.
  - `direction`: send or receive.
  - `word`, `position`: the readiness word and the value that releases a send (a supplied word, or the landed word of
    the hop it relays), or that a receive's landing advances.
  - `serial`: the map serial the plan was made against.
- Entries refer to pages already in the region. Nothing is copied. They are not a ring: a plan's lists are written once
  and replayed by every execution of the plan.
- A plan whose `serial` is not the map's current serial is refused when it is loaded. The call fails by error and its
  caller plans again on the map as it is.

### What the collective decides, and what the bridge decides

- **The collective** plans every transfer of a program together: every collective call and pipeline hand-off, over
  the map's Thunderbolt-5-connected nodes, for any node count. It does this ahead of time with the ported packing and
  concurrent-flow solve, and puts the transfers that block a pipeline stage first. Its output is per-hop page lists on
  every node a page passes through:
  - the source node's sends;
  - each relay's receive into a relay page, and a send released by that receive's landed word;
  - the destination's receive into the destination page.
- **Why the collective fixes the node-level path:** a SEND can only fill a receive already posted on that hop, so every
  node a page passes through must hold that page's entries before the call.
- **Relays are members.** Every live node of the Thunderbolt-5-connected collective runs a rank, so each node's
  collective writes its own relay entries. No node outside the collective takes part.
- **The collective sees** node-level adjacency and each node pair's capacity, the sum of that pair's cables, read from
  the map. It sees no cable, queue pair or link identity.
- **The bridge** decides, for each hop between two adjacent nodes, which cable and which queue pair carry each page,
  and posts and sends accordingly.

### The rule both ends compute

- For each ordered pair of adjacent nodes (a → b), the plan gives both a and b the same ordered list of a's sends to b
  and b's receives from a.
- Both bridges know the cables between a and b from the map, each cable's bandwidth, and the queue-pair count
  negotiated at that cable's pairing, which is the same at both ends.
- Both assign the list's pages, in the list's order, to the (cable, queue pair) whose cumulative bytes over the cable's
  bandwidth finishes earliest, ties broken by cable id and then queue-pair index. This is Graham's list scheduling over
  the pair's links. It is weighted by capacity, so unequal parallel cables both finish together (the failport's grid
  spread, ea247333, split equally).
- b posts each queue pair's receives in that order, and a sends in that order as each page's word releases it. Both
  ends compute the assignment from the same inputs when the plan is loaded, so nothing is negotiated per page.
- Any ready page waits only behind earlier pages of its own queue pair, so spreading over every cable and queue pair is
  what keeps a ready page from waiting.

### The functions

- **The bridge** (`mesh-flow.c` `link_prepare`, `link_send_progress`, `link_receive_progress`, `link_run`, `link_close`):
  load the page lists, apply the rule, post receives ahead, send on release, advance landed words, forward relay pages.
  - The process lifecycle is the failport's, verbatim (tag `pre-return-20261103`: `rdma/mesh-heartbeat.c`;
    `rdma/mesh-link.c` per-port children, their stop, reaping, flush and exit; the holder table).
  - The queue-pair budget is the failport's rule (`max_qp` − `MESH_PROVIDER_RESERVED_QPS`, both ends' smaller,
    bd85613e).
  - `down_pair` stays deleted.
- **The collective** (`mesh-collective.{c,h}`, `mesh-dataflow.h`, `mesh-call.h`):
  - a plan call takes a program's declared transfers (operand ranges and destination ranks) and the map, and writes the
    page lists;
  - `mesh_peer_channel`, `mesh_publish_bind`'s queue argument, `mesh_transfer_bind`, `mesh_link_map`,
    `mesh_link_map_read`, `mesh_allreduce_plan` and `mesh_allreduce_bind` are replaced by that call.
- **The callers** (metal-microbench):
  - `tools/e2b_replay.m` lines 497-500 pass the program's declared receives and sends (operands.json's names, ranks
    and ranges) to the plan call instead of binding transfers;
  - `tools/mesh/run_pair.py`'s `discover_link`, `links`, `start_bridge0`, `start_bridge1`, `bridges_up` and
    `bridges_down` go: the driver starts no bridge, chooses no link, and runs one rank on every node of the collective;
  - `tools/mesh/grid.py` prepares the programs for the collective's rank count with an explicit placement operand
    (shares per rank, even by default). Placement chosen by topology is separate work.

### Order, and what judges it

- The replacement above is finished until it covers everything the `--link` form and the old calls did. Then the
  `--link` form (`-I`, `--link`, `oob`, `dial`, `listener_up`, the per-session exchange) and the replaced calls are
  removed.
- One configuration runs in a runtime. No call site handles both forms.
- Parity is shown by the normal program, the E2B decode on the collective, in an end-to-end run when the operator
  authorizes runs. No tests are written.
- `make -C rdma` and `make -C rdma native-audit` build clean in mesh, and `make .build/e2b_replay` in
  metal-microbench.
- Every wait is bounded, and every failure is an error value, never a hardware fault.
