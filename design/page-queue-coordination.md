# Page-queue change: the coordinator's contract and its routing record

The notes the coordinating agent kept while carrying out design/page-queue-change-request.md (2026-11-03), verbatim: the contract every sub-agent worked to, then the record of every sub-assignment and review finding and where each went.

## The contract (CONTRACT.md)

### Page-queue change: the coordinator's contract for every sub-agent

This is a coordination note for the sub-agents carrying out one document. The requirements are in that document, not here:
**/Users/mdot/dox/mesh/design/page-queue-change-request.md** (mesh main at bd1581df). Read it in full first. Its
operator's statements are the requirements, "Claude's proposal" is the design, and "State of the code" and "Scope" bound
what is touched. Below are only the file split, the names files share, and the rules.

Failport source = mesh tag `pre-return-20261103` (90677543). Read it with `git -C /Users/mdot/dox/mesh show
pre-return-20261103:rdma/<file>`. Bring what your brief names over verbatim, adapting only interfaces (names of region
fields, types this layout has). Each verbatim piece gets a one-line comment giving its source (tag file, function, and the
commit sha where the brief gives one).

## Rules (all sub-agents)

- Work only in /Users/mdot/dox/mesh and /Users/mdot/metal-microbench, on main. No worktrees, no /tmp checkouts, no stash
  (never `git stash`, never read stashes).
- Edit only the files your brief says you own. If you need something in another owner's file, do not edit it: write
  what you need in your final report and I route it. You may read any file.
- Run nothing: no bridge, no program, no ssh, no tests. Compiling your own file is allowed (`cc -fsyntax-only`, or a
  `-c` to an object file under the scratchpad). Every compile finishes within ~8 s.
- Do not open docs/facts.md at any revision. Do not use design/transport.md as a source (the tag's comments cite it; it is
  not a source for this change). Do not touch rdma/mesh_hourly.py, rdma/mesh_preflight.py or rdma/mesh_graph.py. Never edit
  xonotic/.
- Every wait you write is bounded (a deadline set when the wait begins). Every failure returns an error value: never a
  hardware fault, never an unbounded retry. `down_pair` stays deleted: no queue pair is destroyed while a peer may post.
- No invented mechanisms (no hashing, receipts, credits, gates, filters or locks the tag or the document does not have).
  Where the code has to differ from the proposal or from the tag, write a comment at the spot saying what and why, and
  list it in your report.
- Order: build the replacement first. Do not delete the `--link` form or the old calls unless your brief says so
  (sub-assignment 16 does that last). No call site handles both forms.
- Commits: commit only your own files with an explicit pathspec (`git -C <repo> commit -m "..." -- <your files>`), never
  `git add -A` or `git add .`. A new file is added by pathspec first (`git add -- <file>`). The message says which of the
  document's builds the commit passes (`make -C rdma`, `make -C rdma native-audit`, `make .build/e2b_replay`), or "builds:
  none yet (the file compiles alone with cc -fsyntax-only)". It ends with the line
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  Commit as soon as your file compiles on its own. If `.git/index.lock` exists, wait 1 s and retry, at most 10 times.
  Push with `git push origin main`. **If the push is rejected, do not pull, rebase or stash** (other agents' uncommitted
  edits share this working tree): leave the commit local and say so in your report; I push at integration.
- Report (final message, short): files and functions you wrote; what came over verbatim from the tag (function, sha);
  every difference from the proposal or the tag, and why; what you need from other owners; commit shas; compile result.

## Where things live (owner = sub-assignment number)

mesh repo, `rdma/` unless noted:

| file | owner | contents |
|---|---|---|
| mesh.h | 9 | the region: page lists (the queue), map area, holder table, bridge pages |
| mesh-bridge.h (new) | 0, then the coordinator | the bridge's process-local types and every cross-file prototype |
| mesh-holders.h (new) | 4 | `mesh_started`, `mesh_holds`, `mesh_gone`, the holder table's operations |
| mesh-heartbeat.c (new), bin/mesh-bridge.sh | 1 | the heartbeat, whole; the launchd job form |
| mesh-link.c (new) | 2M | the bridge's `main`: configuration_read, region_same, region_running, region_orphans_end, region_open, region_adopt (they share the tag's static `region_children`), control_watch, turn_due, the event loop, the parent's stop (FAREWELL, halts, the windows' hold), then `children_end()` |
| mesh-children.c (new) | 2 | parent side of the per-port children: message_send, child_send, child_elsewhere, child_spawn, child_job, child_heard, child_turn, child_reaped, children_end (main's tail at the tag: CHILD_END to all, the exits queue, SIGKILL past the bound, holders named) |
| mesh-child.c (new) | 3 | the child: child_main, child_read, child_command, child_jobs, child_reports, child_tell, lanes_flush, child_leave |
| mesh-ports.c (new) | 5 | discovery and self-pairing: ports_find, ports_found, port_at, port_open, port_opened, port_hello, hello_heard, port_greet, control_heard, port_heard, link_heard, message_heard, port_say (+ port_flush, port_coalesce, port_watch), port_lost, link_set, link_turn, link_lost, link_tell, record_reach |
| mesh-events.c (new) | 6 | network_heard, addresses_heard, peer_arrived, peer_terminated, peer_port, terminations_watch, terminations_turn, arrivals_turn, quiet_all, fault_heard, faults_install |
| mesh-verbs.h, mesh-lanes.c (new) | 7 | verbs level: device_up with the budget, port_gid/device_gid, wire_map, wire_span; mesh-lanes.c: device_open, link_job, link_make, link_connect (the child's verbs jobs) |
| mesh-records.c (new), mesh-stat.c | 8 | the map: map_put, map_grow, map_adopt, record_changed, record_retell, record_heard, record_held, known_at, the serial; mesh-stat prints the map |
| mesh-flow.c | 10 | the data path in the child: link_prepare (load page lists, apply the rule), link_send_progress, link_receive_progress, link_run, link_close; guard_raise/guard_lower/guard_wake, port_wait/port_tell/port_pass, port_threads; streams_open/streams_cancel/streams_let_go (as cancellation of this node's words) |
| mesh-graph.h, mesh-graph.c (new) | 11g | the tag's graph algorithms the plan and the rule use, verbatim |
| mesh-collective.h, mesh-collective.c | 11 | the plan call |
| mesh-call.h, mesh-call.c, mesh-dataflow.h, mesh-dataflow.c, mesh-metal.h, mesh-metal.m | 12 | the client API; the map reader (node pairs, summed capacity) |
| Makefile | coordinator | |

metal-microbench: `tools/e2b_replay.m` (13), `tools/mesh/run_pair.py` (14), `tools/mesh/grid.py` (15).

The binary `mesh-flow` becomes the resident bridge, built from mesh-link.c, mesh-children.c, mesh-child.c,
mesh-ports.c, mesh-events.c, mesh-lanes.c, mesh-records.c, mesh-flow.c, mesh-dataflow.c and mesh-graph.c. Its child is the
same image (`mesh-flow --child VERSION DEVICE INDEX ONES`, as at the tag). `mesh-heartbeat` is its own binary
(mesh-heartbeat.c, mesh-dataflow.c). mesh-flow.c keeps no `main`.

## Names files share (fixed now; owners define them exactly so)

Region, in mesh.h (owner 9). Main's `struct hdr` and `mesh_layout` are the base; these are added:
- `struct mesh_hop { uint32_t page, node, direction, padding; uint64_t word, position, serial; }`: one hop of one page, as
  the proposal's entry. `page` is a page of this region's arena; `node` the adjacent node's index in the map; `direction`
  MESH_SEND or MESH_RECEIVE; `word` the byte offset in the region of an `_Atomic uint64_t` readiness or landed word;
  `position` its value; `serial` the map serial the plan was made against. 9 decides and documents in mesh.h how
  repeated executions of one plan (main's invocations and ring depth) map onto `word`/`position` and pages, keeping the
  GPU's existing word protocol (mesh-metal.m), with any per-plan values kept beside the list (not per entry).
- `mesh_hops(m)` → the plan's entries; `hdr.hop_off`, `hdr.hop_capacity`; `_Atomic uint64_t hdr.hops` (entries of the
  plan written, 0 none); `_Atomic uint64_t hdr.plan` (the serial of the plan written).
- The map: `struct mesh_record` and `struct mesh_reach` as at the tag (adapted to main), at `hdr.map_off` with
  `_Atomic uint64_t hdr.map, hdr.map_nodes` (map seqlock, record count), `mesh_records(m)`. A node's **index** is its
  position in ascending node-id order among the map's records. `_Atomic uint64_t hdr.map_serial`: moves at every change
  of the map's nodes or cables (8 moves it). Per cable a reach entry gives the peer node, both ends' port index, the
  bandwidth and the queue pairs the pairing negotiated (the same at both ends).
- The bridge pages, the tag's structs verbatim: `struct mesh_holder { uint32_t pid, unit; uint64_t start; }` and
  `struct mesh_bridge_pages { uint32_t link[MESH_PORTS],ones,map,map_pages,padding; uint64_t bridge_start;
  struct mesh_holder child[2*MESH_PORTS]; }`, at an offset of its own in the layout, `mesh_bridge_pages(m)`;
  `MESH_PORTS` the most ports. `_Atomic uint64_t hdr.client_start` (the attached client's mesh_started value).
- The rule's inputs (proposal, "The rule both ends compute"), the same at both ends, read from the map: the cables
  between a and b are the reach entries naming that pair. A cable's key is (the lower node's port index, the higher
  node's port index), and cables are ordered by that key. A cable's bandwidth is the lesser of the two records' entries
  for it. Its queue pairs are the count the pairing negotiated (both ends' smaller budget, so the same at both ends).
  Each (cable, queue pair) link's rate is the cable's bandwidth over its queue-pair count. Links are listed by
  (cable key, queue-pair index), and the assignment is `mesh_graph_list_schedule` (mesh-graph.h).
- `MESH_PROVIDER_RESERVED_QPS` comes from the build (`-DMESH_PROVIDER_RESERVED_QPS=1`, the tag's
  etc/bridge/thunderbolt.mk). No header defines it. The tag's etc/bridge/layout.h holds MESH_CHILD_VERSION, TAG_CHILD,
  MESH_*_NS, MESH_KNOWN and MESH_LINK_STREAMS, which come over verbatim into mesh-bridge.h.
- `stop` is one shared `extern _Atomic sig_atomic_t stop;` (the tag's form), defined once in mesh-link.c.
- Registration stays main's registered window (`wire_pages`, `mesh_wired`), as in the document's scope, which touches
  pairing and teardown, not registration. Every page a plan names, relay pages included, is in the window. The plan call
  refuses, with EINVAL, a page outside it.
- `struct mesh_link_info` (main's, per port, at `hdr.link_off`): keeps peer, device, bandwidth, port phase/code.

Bridge process-local, in mesh-bridge.h (owner 0): the tag's `struct port`, `struct link`, `struct message`,
`struct child_message`, `struct known`, `MESH_CHILD_VERSION`, `TAG_CHILD`, `LINK_STREAMS_ENDED`, `link_plan`,
`hello_budget`, `mesh_say`, `clock_ns`, `MESH_PLATFORM_NS`, `MESH_PAUSE_NS`, `MESH_FLUSH_QUIET_NS`,
`MESH_PROVIDER_RESERVED_QPS`, the shared globals (`M`, `wire`, `ports`, `port_count`, `events`, `listener`, `hello`,
`network`, `addresses`, `now`, `stop`, `stop_reason`, `exit_at`, `faulted`, `terminated`, `self_path`, `ones`), and one
prototype for every function above that another file calls, with the tag's signature. Lanes become main's queue pairs of
one cable: a port is one cable; its child holds that cable's queue pairs (the budget's count).

The child's command line: `mesh-flow --child VERSION DEVICE INDEX ONES REGION`, the tag's plus the region's name as
the sixth word. Main's region is named by configuration, not by MESH_NAME alone. 2 spawns it so, and 3 parses it so.

A function you call that mesh-bridge.h does not yet declare: declare it at the top of your own file with the tag's
signature, marked `/* declared here until mesh-bridge.h carries it */`, and list it in your report. The coordinator moves
it into mesh-bridge.h at integration. Owners of the extra tag statics: `port_halt`, `lane_lost` and `link_failed` →
mesh-flow.c (10); `windows_end` and `quiet_all` → mesh-events.c (6); `ports_stalled` → mesh-ports.c (5).

A plan reaches the child's data path without changing the child's loop: the progress threads mesh-flow.c starts
(`port_threads`, once the link is up) watch `hdr.plan` with bounded waits. When it moves, and the plan's serial is the
map's current serial, they load this port's share of the page lists (`link_prepare`) and serve them. The data path keeps
the tag's per-lane counts (`l->lane[k].sent/done/posted/got`) that `lanes_flush` reads.

Pairing messages: per the coordinator, a pairing uses the TCP control socket for one exchange (HELLO, then LINK, and the
records the map holds then), and the socket is then closed. After pairing, `port_say` carries RECORD and FAREWELL on the
datagram socket (`hello`) to the peer's link-local address; a link's loss is heard from the kernel's link events
(network_heard), the peer interface's termination (peer_terminated) and the verbs completions (CHILD_BROKEN), not from
the socket's end. This differs from the tag (which kept the socket); 5 writes the note at the spot.

## Wave-2 interfaces (fixed by the coordinator; mesh.h 17612857's execution comment is the data-path contract)

Read mesh.h's comment above `struct mesh_hop` (17612857): executions, depth, `hop_stride`, which page and which word
execution t uses, what releases a send, what a landing stores, cancellation (UINT64_MAX), and the order of writing a plan.

The region is the node's one region, MESH_NAME, as at the tag: one resident bridge per node. The child's sixth word
names it all the same.

The configuration (2M reads it with the tag's configuration_read; 1's bin/mesh-bridge.sh writes it): the canonical
property list, with the keys the tag's rdma/bridge_configuration.py writes (`name`, `region_bytes`) plus main's geometry
as integers: `block_pages`, `rows`, `window_pages`, `hops` (the hop list's capacity). An absent geometry key takes
mesh_layout's default. No node index and no link is configured: the bridge finds its ports and its node id itself.

The plan call (11, mesh-collective.h), called by every rank with the same declared transfers:
```c
/* one declared transfer: an operand range from rank `source` to rank `destination` (ranks are map indices);
   `order` lower first (a transfer that blocks a pipeline stage); `section` this node's pages of it where it is the
   source or the destination (section.first == MESH_ABSENT elsewhere), section.bytes the range's bytes everywhere */
struct mesh_declared { uint32_t source, destination, identity, order; struct mesh_section section; };
int mesh_plan(struct mesh_ctx *, const struct mesh_declared *, uint32_t count, uint32_t executions, uint32_t depth);
```
It reads the map only through 12's reader. It routes all transfers together over the node graph (mesh-graph.h). Each
node writes only its own hops: its sends, its relay receives (into relay pages it allocates in the registered window
with mesh_section_create) and the sends those release, and its receives. It allocates the plan's words (a section
outside the window, zeroed). For each hop page's row, it records the hop's words in that row's
`struct mesh_publication`, so mesh-metal.m and the host functions find them as they do today:
- a landed word: `device_input` = the execution-0 word's region offset, `device_stride` = `hop_stride`;
- a send's release word at its source: `sends` = 1, `targets[0]` = {`.stream` = the execution-0 word's region offset,
  `.count` = 1, `.stride` = 1}, and `argument` = `position`. `hop_stride` = sizeof(struct mesh_send), so the GPU's
  existing `+ t * sizeof(struct mesh_send)` addressing holds.
It returns an error value: ENOSPC past `hop_capacity`, ESTALE where the map moved while planning, EHOSTUNREACH where a
transfer has no Thunderbolt-5 path within the collective, EINVAL for a page outside the registered window.

The map reader (12, mesh-dataflow.h). The collective sees node pairs and summed capacity, nothing else:
```c
#define MESH_NODES_MOST 32
struct mesh_nodes { uint64_t serial; uint32_t count, self; uint32_t node[MESH_NODES_MOST];
                    uint64_t capacity[MESH_NODES_MOST][MESH_NODES_MOST]; /* bits/s, a pair's cables summed; 0 none */ };
int mesh_nodes_read(struct mesh_ctx *, struct mesh_nodes *);   /* 0, or an error value; reads under the map seqlock */
```

A plan loaded (10 stores, 12 waits): when a port's data path has loaded the plan `hdr.plan` names, it stores that
serial in `mesh_links(M)[port].port.prepared` (release). On refusal, it sets phase MESH_STOPPED and `code` instead. The
client's start (12, in place of mesh_transfers_start's configured latch and its unbounded spin) waits, bounded by
MESH_PLATFORM_NS, for every port whose peer some hop names, and returns the port's code, or ETIMEDOUT.

mesh-stat (8) prints the map for the drivers with `mesh-stat --map`, one JSON object on one line:
`{"serial":S,"self":I,"nodes":[{"index":i,"node":id,"model":"...","reach":[{"node":j,"bandwidth":b}]}]}`. Indices are
the map's (ascending node id). It is a bridge-side tool, so it may also print cables under `"cables"`, but the drivers
read only the above.

Link pages: `struct link`'s `page`/`pages`, `child_message.page` and `mesh_bridge_pages.link[]` exist (the tag's
structs, verbatim), but this design leases no link pages and posts nothing on lanes that carry nothing. `l->page` stays
MESH_ABSENT and `link[p]` stays MESH_ABSENT. Code that leased or posted on them is left out, with a note at the spot.
mesh.h owns `mesh_notify` and `mesh_map_whole`: no other file defines them. mesh-bridge.h (54ac5541) owns
`quiet_until`, `page_bytes`, `clock_ns`, `mesh_say`, MESH_WIRE_MAGIC and the layout.h constants.

## The routing record (ROUTING.md)

### Coordinator's routing notes

## Done
- 15 grid.py: 2242e7bb (pushed). --nodes required; shares even by default; stages() no clamp.
- 4 mesh-holders.h: 1f751647 (pushed). mesh_gone (2310753d), mesh_started (b346ef2b; exit case 8ac5c9fb, reverted c18f7a0e, restored 07949b28), mesh_holds (b346ef2b).

- 11g mesh-graph.{h,c}: b37d0073 (pushed). Verbatim: flow+flow_same 4163911c; pack+mesh_packed_heavier+pack_same 9bb6972a; tree_loads feef1156; trees_rate, tree_depth, fill d30f02f5; grid_cut ea247333; exp/log 4163911c; keyed_find/add d8d861b5; arborescence ab823357; arborescence_within 9bb6972a; depth 8cd0c480; order ec2ad141. New: mesh_graph_list_schedule(links, rate, units, bytes, link, carried) -> double. Diff: pack comment cited facts.md -> 9bb6972a msg.

- 7 step 1 mesh-verbs.h: 615aa4e7 (pushed). down_pair orphan removed; max_pairs (tag 167-168, f35a0447/76be3217/bd85613e); device_gid (de96b545,672f6f7f,5eabe91e); pair_make (ceff26ec); pair_up (76be3217); queue_receive/queue_send (ceff26ec); port_bandwidth (5eabe91e); GID/speed constants from tag etc/bridge/thunderbolt.h (5eabe91e). Added fields to mesh_device/mesh_queue/mesh_verbs; two CQs in pair_up; unused attrs on wire_map/die/down_device/down/onsig/verbs_up.

- R1 round 1 done. Findings: H1 (holders ops stay inline: confirmed), H2 (12 includes mesh-holders.h), G1 (rule inputs -> CONTRACT fixed), G2 (11 may need reach/climb/cycle_length from 11g), V1 (whole-arena registration: NOT taken, out of scope + wired-memory risk; note in report), V2 (device_up must fill device->port for device_gid; tag device_up has no mutex) -> 7 step 2, V3 stop extern -> 7, V4 Makefile -D, V5 note, P1 member source for 14/15, P2 window sizing in grid.plan/run_pair -> 16/14, C1 bridge pages fields -> sent to 9, C2 constants in tag etc/bridge/layout.h -> sent to 0, tension: tag stop bounds 13 s/18 s vs 8 s (keep verbatim, note), C3 reach per-cable fields in client-mapped region: 12's reader exposes only node pairs.

## To route
- MESH_PROVIDER_RESERVED_QPS: tag value 1 via -D from etc/bridge/thunderbolt.mk. Decide: 7 defines in mesh-verbs.h #ifndef =1 (cite thunderbolt.mk); 0 must not define it.
- to 7 step 2: mesh-verbs.h line 111 `static stop` -> tag's extern form (defined in mesh-link.c 2M); shm/down static are --link form (16). device_open: main device_up has no refused out -> recover from errno (port_opened backoff).
- MESH_BANK: tag thunderbolt.h spelling differs; don't bring that header whole.
- Makefile: mesh-graph.c into mesh-flow link (+libmesh if collective uses); mesh-graph.h deps.
- to 10: rule rates — per (cable,qp) link rate = bandwidth[c]/qps[c], listed by cable id then qp index (11g suggestion; 10 decides).
- to 14 (run_pair.py): --split-dir default decode-tp2; PEER default mesh-mini; nodes are an operand (no config names members: etc/mesh-nodes.json is a 2-machine roster incl. workstation).
- to 12 (dataflow): include mesh-holders.h, no redefinition; mesh_attach stores hdr.client_start=mesh_started(getpid()) after CAS (tag mesh_slot_take:284); re-read version seq_cst after swap -> EINVAL if expired (heartbeat replace_turn relies); line 72 kill/ESRCH -> mesh_gone.
- to 1 (heartbeat): replace_turn client: hdr.client low 32 + hdr.client_start.
- to 8 (mesh-stat): mesh_gone(bridge_pid) -> -ESRCH (tag mesh-stat.c:45).
- contract change: region_running + region_orphans_end move to 2M (mesh-link.c) beside region_open (static region_children).
- to 11/14: relay pages need window room; grid.plan sizes only own sends/receives.
- placement files are 2-rank (tools/mesh/placements/*.json); collective of N needs its own file; topology placement separate work.
- hourly passes --nodes local,<peer> (2 of the live members) — mesh_hourly.py untouchable; note in report.

## Wave 2 landings
- 9 second pass: 2ffbbe8e (boot, health, busy, idle_since, stopping; MESH_VERSION 106).
- 0 second pass: 54ac5541 (link page fields back; MESH_CHILD_VERSION 2u; layout.h consts; MESH_WIRE_MAGIC; quiet_until/page_bytes; externs).
- 7 step 2: ae3c36f9 mesh-lanes.c (link_lapsed+link_numbers 3f72cbde; link_make 3f72cbde; lane_reset 4bb01da0; link_connect 4bb01da0; link_job 60b00774; device_open abeffc1b) + mesh-verbs.h stop extern, device_up(dev,wire,m,int *refused) fills device->port, mutex moved to verbs_up, name char[32].
- 3: 53e3935c mesh-child.c (tag 1536-1740; 437e2c38, 60be5eb7, 34a0d80c, bc17811b). Kept l->page lines (verbatim). Needs link_lane (tag 270-274, c24265e0) -> INTEGRATION: add static inline to mesh-bridge.h, remove S3's local decl.
- Integration: check who defines exit_at/faulted/terminated (6 vs 2M) — link errors will show.
- Makefile written locally (not committed): -D QPS, BRIDGE list, heartbeat, -Wall -Wextra -Werror on bridge link.
- 6: 32155bf8 mesh-events.c (windows_end 8bbc01a5 d62c879b; network_heard 6b33cc33 ceff26ec 5b49fbb9 8bbc01a5 d62c879b 0e34c13e 128d0e84; addresses_heard ba674fbe b505bae4 ea04d19f; arrived* 8bbc01a5 d71970f8 a8952de9 ba674fbe abeffc1b; quiet_all d62c879b; peer_port 8bbc01a5; peer_terminated 8bbc01a5 d62c879b d71970f8 abeffc1b; peer_arrived ba674fbe; terminations_watch 8bbc01a5 ba674fbe abeffc1b; fault_heard 8bbc01a5 2f4c511f b11621e6 ba674fbe 60b00774; faults_install 2f4c511f 1f9be1d5; terminations_turn 8bbc01a5 2f4c511f a8952de9 0e34c13e abeffc1b; arrivals_turn ba674fbe). Diffs: statics dropped; terminations_watch logs (abd476a5 lines); faults_install die -> mesh_say+exit(1). Locals: port_halt, windows_end, MESH_KERNEL_EVENT_BYTES 512 -> INTEGRATION move to mesh-bridge.h. arrivals_turn control<0 test -> change to 5's paired state.
- 2: f0d2c2ff mesh-children.c (message_send 111d42f7 90179f5e d556370e; child_send 1f9be1d5 111d42f7; child_elsewhere/child_spawn b346ef2b 8ac5c9fb c18f7a0e 07949b28 (+1f9be1d5); child_job 1f9be1d5 76be3217 128d0e84; child_heard 1f9be1d5 76be3217 b53804bf 306b22be; child_turn 1f9be1d5 b78c7712 90179f5e d556370e; child_reaped b346ef2b 0d14fb73 90179f5e 2548cbce; children_end 3f72cbde 90179f5e 07949b28). child_reaped waitpid(…,0) unbounded: operator's ruling 90179f5e took 111d42f7's deadline back -> keep, report.
- INTEGRATION: MESH_CHILD_VERSION 2u -> 3u (command line gained REGION word).
- 2M: d77eec2a mesh-link.c (wake 3f72cbde; control_watch 90179f5e…6b33cc33; region_children/running/orphans_end 07949b28; due_at/turn_due 90179f5e…923d73fd; main 1894-2078). Diffs: configuration_read +geometry keys; region_same main fields; region_open length check, MESH_VERSION, bridge_off bounds for holder copy; region_adopt main link table + one client; main .links=MESH_PORTS qps/orders 0 (old binds EINVAL); no link_reserve; one client EVFILT_PROC; tail children_end. Locals: windows_end, ports_stalled, stalled, MESH_CONFIGURATION_BYTES 65536, MESH_REGION_NAME_BYTES 256, MESH_REQUEST_BYTES 16773120u -> INTEGRATION move to mesh-bridge.h. control_watch no-timeout wait: verbatim (90179f5e ruling) -> report. --link main's shm_unlink(MESH_NAME) hazard until 16 removes it.
- 8: 6955a3c0, 7468cee5 mesh-records.c + mesh-stat.c (all tag functions; 3f72cbde, record_heard 60b00774 abeffc1b, map_put abeffc1b). Diffs: includes mesh-dataflow.h; busy_tell = client attached; map_grow mesh_arena_alloc(...,0) outside window; map_adopt serial; map_put ascending insert + mesh_graph_reach prune (C9 fix) + serial on cable change; record_links_same whole entries; record_changed region = arena - bridge's; clients_turn one client word. mesh-stat --map per-cable reach. Locals: mesh_joined (-> 12), mesh_graph_reach (-> 11g), MESH_HEALTH_NS 10 s (-> mesh-bridge.h INTEGRATION).
- HAZARD routed: plan end with requests outstanding -> 10 (LINK_STREAMS_ENDED, recycle child, prepared=0); 12 retains retired rows while port.prepared == plan; INTEGRATION/2: child_reaped clears mesh_links(M)[p].port.prepared=0 (+ unreleased=1 already).
- 11g resumed: mesh_graph_reach, climb, cycle_length.
- 1: f50dcfd9 mesh-heartbeat.c (tag whole, 20 shas) + bin/mesh-bridge.sh (tag job form). Heartbeat local copies of mesh_say/MESH_PLATFORM_NS/LOG/DIAGNOSTIC/mesh_clock_ns -> INTEGRATION dedupe (mesh.h gets MESH_PLATFORM_NS via 9). bridge.conf needs mesh_arena_pages or mesh_pct (old form required too). Docs (README, AGENTS.md, mesh-kill-guard.sh) describe old restart -> note only.
- 12: 7da3c125, 3117456f (+ resumed: mesh_joined, retained rows). mesh_nodes_read; mesh_attach client_start/version recheck/mesh_gone; mesh_transfers_start bounded in mesh-dataflow.c; mesh-metal wraps plan words (send_off/send_bytes); receive_prepare one page; hop_stride 256 enforced; MESH_PLATFORM_NS local #ifndef. Declined C9 move of mesh_cables etc. (reads cable records) — note.
- Contract change: hdr.plan = generation (sent to 9, 10, 11).
- 11g step 2: 449f2097 (reach 6379e868; climb 7e7f8726; cycle_length+climb_return f21a4d24). -> 8's local decl of mesh_graph_reach: INTEGRATION include mesh-graph.h.
- 5: b46bea07 mesh-ports.c. Added port_paired (control>=0||greeted), FAREWELL datagram on loss, hello-from-paired-peer = loss (ECONNRESET), peer_at[] getpeername. Locals: ports_stalled, stalled, MESH_TOPOLOGY_MESSAGE_BYTES (64<<20), port_paired -> INTEGRATION mesh-bridge.h. INTEGRATION: mesh-link.c:474 accept test -> port_paired; :313 turn_due -> !port_paired; mesh-events.c:260 arrivals_turn -> !port_paired.
- 9 third pass: 61a8653d (plan generation; hops default rows; MESH_PLATFORM_NS in mesh.h). CHECK: its comment says "hdr.port.prepared" — should be mesh_links(M)[p].port.prepared.
- INTEGRATION 9f09c812: mesh-bridge.h link_lane, port_paired, port_halt, windows_end, ports_stalled/stalled, constants; MESH_CHILD_VERSION 3; locals removed; port_paired in mesh-link.c (accept, turn_due) and mesh-events.c arrivals_turn.
- R2 launched on bridge lifecycle files.
- 12 resumed: c01fa79d (mesh_paired/mesh_join/mesh_joined verbatim, MESH_STREAMS->MESH_PORTS; mesh_nodes_read capacity = join count x join bandwidth; deleted mesh_cables; mesh_retired_release returns kept rows, retains while ANY port prepared != 0 (deviation: no retired_plan field) ). C9 flagged mesh_paired/mesh_join in dataflow.c (tag's place; graph module "knows no cable") -> left; report.
- INTEGRATION 74461a30: mesh-records.c mesh_joined local removed; clients_turn unreleased=mesh_retired_release(M)!=0 (tag line).
- 10 told: clear prepared on client retire / plan end / refusal.
- 11: c5bffd0c, c9ffedcf, 80e2e4d3 mesh_plan (map via mesh_nodes_read; demands grouped (source, identity); mesh_graph_reach per source -> EHOSTUNREACH; one joint mesh_graph_flow (eps 0.1); mesh_graph_rows; mesh_graph_fill latency 0 cuts pages over trees; mesh_graph_climb parents; position 1; words outside window, hop_stride 256; ESTALE if map moved). Diffs: blocking-first via list order only; flow not pack for multicast; no grid_cut (ranges page-aligned); extra EINVALs; new relay/word sections each call (no free); mesh_host_inputs returns 0; flow memory ~ room x nodes. Needs mesh_graph_rows (11g step 3). 13: ranges page-aligned. 10: relay page reuse waits forward send t-depth.
- 13 launched.
- 10: c193b11a, dad7b49d mesh-flow.c (port_halt 3f72cbde 306b22be; lane_lost d556370e; link_failed d556370e; port_tell 60b00774; port_pass 8b757084; port_progress 3a82ede9; guard_wake 60b00774; port_threads 3a82ede9; port_wait e3c4c986; guard_lower/raise 306b22be; streams_open/cancel 11807bec; streams_let_go 60b00774; map_read from tag mesh_records_read e3c4c986). Diffs: one progress thread per port (tag guard) not d6dceac's two spinning threads; batch polling of two CQs; receive window d6dceac's depth-1; EMSGSIZE refusal; cancellation never overwrites landing; plan end w/ outstanding -> LINK_STREAMS_ENDED; no plan while client 0; ledger per lane. Gap: both ends' maps not proven equal (serial local) -> report.
- DECISION relay hazard: cell(t)+8 = page's sent count of forwards; receive t waits cell(t-depth)+8 == forwards (deferral). Sent to 10, 9.
- 12 resumed: mesh_cables (C9), mesh_transfers_start serial fix, mesh_retire wakes &hdr.plan.
- 11g step 3: 311dd5f6 mesh_graph_rows (d8d861b5).
- INTEGRATION cc22ead9: Makefile + mesh-collective.c local decl. BUILDS: make -C rdma PASS, native-audit PASS (18:42).
- 9 fourth pass: 90962999 (cell layout +8 sent count; per-port prepared wording).
- 12 third: 13733942 mesh_cables (hidden; one copy; mesh_join built on it; mesh_paired gone); mesh_nodes_read capacity = sum of cable bandwidths; start compares entry serial + waits prepared==generation; mesh_retire wakes &hdr.plan; local MESH_PLATFORM_NS removed. C9 flags mesh_cables / mesh_join as graph-module code -> left in dataflow (tag's place; graph module knows no cable) -> REPORT.
- 10 final-ish: 39e2364d (temp ENOTSUP, superseded), be5643e3 (sent-count rule; mesh_cables; include mesh-dataflow.h). Build mesh-flow + native-audit pass. C9 flags forward matching in link_prepare.
- CHAIN: 9 renames mesh_hop.padding -> forwards; 11 writes forwards + own 256 B cell per word (plan_word packed 32/line bug); then 10 reads forwards, deletes matching.
- 14: dc454f50 run_pair.py (deleted discover_link, links, link_interfaces, _active, start_bridge0/1, bridges_up/down, wait_for, run_block generations, size_geometry/ARENA/WINDOW, PORT, MESH_PEER, MESH1, --ledger; REGION /mesh0; --nodes required (grid.nodes_from); rank = mesh-stat --map self; refusals: unreadable map, differing records, dup index, members not covered, world_size != ranks; bounded waits (run 120 s, stop 8 s, node read 8 s, preflight 45 s); preflight = mesh_preflight.py --target per node only; split-dir decode-tp{ranks}). Diff: "one map" = matching records, not serials (serial local) -> CONFIRMED. Record fields pair->collective (no parsers). HOURLY: mesh_hourly.py calls run_pair.py with MESH_PEER, no --nodes -> exits 2 (untouchable) -> REPORT. grid.py docstring: ranks follow --nodes order vs map index.
- 9 fifth: 52aeae89 (padding -> forwards).
- 11 second: cc2c75c3 (own 256 B cell per word per execution; forwards written; C9 passed).
- 10 resumed: read forwards, delete matching.
- NETWORK CUT ~19:1x. 10 had committed d7b59fa4 (forwards read; done). 13 and R2 resumed. Build at d7b59fa4: make -C rdma PASS, native-audit PASS.
- R1 round 2 launched: mesh.h, mesh-flow.c, collective, graph, call/dataflow/metal, run_pair.py, grid.py.
- 13: 058ea619 e2b_replay.m (mesh_plan with every route x page x destination; order = region; ranges = page cut; rank check vs mesh_nodes_read self -> exit 2; removed replay_transfers_prepare; stop_replay mesh_cancel+notify; manifest_bytes: tflite operand bytes = shape x 2 (F16) checked against LiteRT binding -> exit 2). make .build/e2b_replay PASS. Gaps: map change between check and plan; RECORD_ONLY w/ rank plans. Recorder even-split -> 17. prepare_e2b.py should write bytes (note only).
- 16 (removal) launched; 17 (metal_recording range size) launched.
- 17: eadc957b metal_recording.{h,m} publish: rangeBytes param, element 8 stores it; e2b_replay.m call passes backing->range_bytes. 1749 assumption holds (release words spaced executions x 256). make .build/e2b_replay PASS.
- R1 round 2 done. Blocking: X1 FP contraction -> Makefile CFLAGS_EXACT=-ffp-contract=off (local, NOT COMMITTED yet; verify 0 fmadd after 16's mesh.h settles); X2 byte_len check -> 10; block size mismatch already refused by verbatim port_heard (heard->block != page_bytes -> EPROTO); map agreement across nodes = tag's assumption + run_pair check -> REPORT. Non-blocking: S1 refusal generation -> 10 then 12; S2 torn read -> 10 (reader) then 11 (writer fence); M1 cancel overwrites +8 counts (note only: after cancel, run stops); P1 sections leak per mesh_plan -> 11 after 16; C1 identity in client headers -> mesh_observe/link_view removal (16), mesh_cables/mesh_reach move to bridge-only header -> S18 after 16; D1 notes; R1/R2 run_pair -> 14; G1 grid plan() dead -> 15.
- PENDING after 16: 11 (writer fence S2, P1 retire sections), 12 (S1 start/retention), S18 header move (C1).
- 10 review fixes: b3cc9990 (byte_len EMSGSIZE; refusal stores generation last; torn-read re-check EAGAIN).
- 14 review fixes: 5ba3b7be (REACH compare node,bandwidth,port,peer_port,qps; own-pid stop with ps guard; pid/said fields).
- 15 G1: 9bb797c0 (deleted plan, with_kv_counts, kv_slots, extent, stages, PAGE, GIB, math; rank = map index wording).
- 16 DONE: mesh 85fd621a (link map + all-reduce + mesh.py AllReduce), 1444f409 (mesh_transfer_bind, mesh_transfers_prepare, mesh_peer_channel, mesh_publish_bind), 4ac5cbdc (mesh.h old lists, notices, MESH_QPS, hdr qps/orders/configured/length_off/order_off/notice_*; 106->107; fields kept as vacant0..6 padding so bridge_off stays 296), b1c676bf (mesh-verbs.h qpi, XMAGIC, pairing_active, dial, listener_up, exchange, oob, verbs_up, port_gid, down_device, down, onsig, shm, mesh_verbs --link fields, mesh_device.setup), 98d23983 (mesh_observe, mesh_link_view, Swift observe), a37d5a74 (peers.py bridge_arguments; etc/bridge.conf node=/links=), 98db73fe (examples/links-*.conf); metal-microbench 8375cb22 (configs/links/pair-ring.txt; stalls.py constant). Remaining refs: mesh-flow.c:11 comment; user/mesh-telemetry.py:217-219 + bin/mesh-top.py parse --link (tolerate empty); history notes; docs (design/ 7 files, docs/elastic.md etc.).
- INTEGRATION e17d9be0 Makefile -ffp-contract=off: 0 fused ops in libmesh, mesh-flow. All 3 builds PASS.
- 11 (S2 fence, P1 retire) and 12 (S1 start/retention STOPPED) resumed. Then S18 header move.
- R2 done. BLOCKING P1/P2: closed control socket misses dead peer (UC QPs no error) and mid-exchange crash -> DECISION: revert to tag's socket-kept form (still one exchange per pairing; no per-session exchange) -> 5. REPORT this deviation from coordinator's instruction prominently. P3 FAREWELL order (tag form fixes), P4 drop link->device write -> 5. L1 region fit, L2 streams_cancel(NULL) on adopt -> 2M. C1 re-cancel no-op -> 10 (+ mesh-flow.c:11 comment). H1 mesh-child.c comment, R1 MESH_HEALTH_NS, B2 dup defs (MESH_PLATFORM_NS x4, MESH_MODEL_BYTES x2, mesh_say x2) -> S18. R2 mesh-stat prints port/peer_port/qps (driver compares; keep, note). B1 link_plan differs (budget 1 -> 1 QP; tag lanes 0/1 idle) -> note. K1 heartbeat guard, S1 launchd tag domain user/<uid> + LimitLoadToSessionType Background, S2 120 s -> 8 s -> 1. Unbounded waits: control_watch, child_reaped waitpid, heartbeat kevent NULL timeout = tag's (90179f5e) -> report.
- AFTER 5 lands: me: port_paired -> control>=0 in mesh-bridge.h.
- S18 after 12 and 1: header hygiene (C1 move record/cable/link types to bridge-only header; B2 dedupe; MESH_HEALTH_NS; H1).
- 12 S1: 4000eba9 (start: prepared==plan then phase STOPPED->code/ENOTCONN; retention still counts STOPPED ports, because lost-while-serving ports are STOPPED with prepared set and receives posted until reap; refused ports cleared on next pass) -> ACCEPTED.
- 10 C1: 71a6f19b (cancelled[MESH_PORTS] record of last peer; re-cancel uses it; header comment history).
- 11 round-2 fixes: 2da5a85b (release fence after hops=0; one relay section per plan; plan_release marks closed + mesh_rows_release; released on failure and on replacement; process-local record of previous plan's sections; send_off/bytes 0 after failure). Limit: closed sections held while any port serves a plan -> grows per re-plan -> REPORT.
- 2M L1/L2: be0e1e48 (tag mesh_layout_within 378dbd5f as static in mesh-link.c; region_open(&geometry, shape, &fd) fits; pages in whole blocks; region_adopt streams_cancel(NULL)). S18: move mesh_layout_within to mesh.h.
- 5 revert: ade422e4 (socket kept for link life; tag form; device write dropped).
- INTEGRATION 9e8f8453: port_paired = control>=0 (push race with S1's 34c643b2, retried OK).
- 1 fixes: 34c643b2 (K1 guard; user/<uid> + LimitLoadToSessionType Background; system scope removed; 8 s readiness; local MESH_PLATFORM_NS dropped). stop waits 30 s (heartbeat TIMER_END 25 s verbatim) -> accepted, report. etc/bridge.conf no region size -> start exits 64 (old form also needed arena pages) -> report.
- 18 launched: header hygiene.
- 18: e99bfc21 (mesh-map.h: mesh_link_info, mesh_links, MESH_MODEL_BYTES, mesh_health, mesh_reach, mesh_record, mesh_records, mesh_geometry, mesh_layout, mesh_joined decl, mesh_cable, mesh_cables), 9d826caf (MESH_PLATFORM_NS mesh.h only), 66beb887 (mesh_say/LOG/DIAG/clock_ns one copy in mesh-map.h; heartbeat mesh_clock_ns -> clock_ns 2 calls), ecf6fd72 (MESH_HEALTH_NS -> mesh-bridge.h), 3c6657d3 (mesh_layout_within -> mesh-map.h), f1fe6509 (mesh-child.c comment). Also edited rdma/Makefile HEADERS + deps (committed in one of these). Still in client headers: MESH_PORTS, mesh_bridge_pages (link[]) via mesh-holders.h, mesh_holder, mesh_send.pair + <infiniband/verbs.h>, MESH_PORT, port phase enum, mesh_cancel link arg, hdr links/link_off/port, mesh_port_info -> REPORT.
- FINAL BUILDS at mesh f1fe6509 / mm 8375cb22 (clean rebuild): make -C rdma rc 0; native-audit rc 0; make .build/e2b_replay rc 0; 0 fused ops.
- R2 final pass launched (blocking only).
