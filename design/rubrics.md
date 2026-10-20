# Three rubrics: mesh, collective, data

Each rubric says what its code is like at its best and at its worst, and scores the code on several qualities. A
quality is not met or missed. The code sits somewhere between a virtuous pole and a derelict one, and every change moves
it one way or the other. Neither pole is neutral: a virtue is like a particular good thing done well, and a dereliction
is like a particular way of failing the people who depended on it. The signs under each quality say which way a change
moves the code.

The three rubrics invert one another. What is virtue in one layer is, in another, a dereliction (Inversions, below).
Code that reads as virtuous under two rubrics is in the wrong layer.

| | Mesh | Collective | Data |
|---|---|---|---|
| Code | `rdma/mesh-flow` and what it publishes (mesh-link.c, mesh-verbs.h, mesh-records.c, mesh.h's region layout); the node services | libnccl-mesh (nccl.h, nccl-mesh*.c/.m), the schedule compiler (mesh-collective.c, mesh-plan.h), the device functions kernels call (mesh.metal), the bindings (torch-mesh backend.py, mesh_mpi.py) | everything that computes values: metal-microbench's engine, LiteRT's program, a rank's operands and crossings (mesh_rank.m, rank.swift), goal_a.swift, torch programs, examples/, torch-mesh's kernels; and the configuration they load |
| Lives | as long as the node | as long as a communicator, in its client's process | one call; its configuration, one model load |
| Hides | Apple's driver and the cables' hardware | the cabling and the membership, which change daily | the model, its formats and kernels, which change weekly |

## M. Mesh code: the common carrier, or the customs house

At its best the bridge is a common carrier. It takes anyone's sealed goods and carries them without opening them. It is
there every day, whatever happens to any one shipper, and it keeps an honest ledger of what it lost. At its worst it is
a customs house. It opens parcels, demands forms, and holds goods until someone answers. It closes the road when one
cart breaks down, and it will not leave when its shift is over.

**M1. Discretion: what it knows of what it carries.**
- Virtue, the sealed-letter postman: it moves bytes between rows and never learns what they mean; a new model or a new
  collective changes nothing in it.
- Dereliction, the customs inspector: it opens the parcels (dtypes, checksums, digests, a collective's name), and must
  be retrained, which here means restarted on every node, for every new kind of parcel.
- Signs: toward the postman, deleting anything the bridge reads of payload or of a position's purpose; toward the
  inspector, any check of content or branch on what a position is for.

**M2. Steadfastness: what other parties' failures do to it.**
- Virtue, the lighthouse: ships founder and the light stays on; a client's failure costs that client its rows, its
  unsent positions go out as duds, and nothing else changes.
- Dereliction, the drawbridge that goes up whenever a cart breaks down: a client's end concludes a link, restarts its
  queue pairs and delays every other client on the cable, so one failure becomes everyone's.
- Signs: toward the lighthouse, duds and released rows; toward the drawbridge, any path from a client's end to a link's
  conclusion.

**M3. Tenure: how it begins and ends.**
- Virtue, the night watchman: on duty all night, gone at once when relieved. Any signal from anyone ends it, and it
  leaves nothing locked behind it.
- Dereliction has two faces. The deserter walks off mid-shift on its own: it stops itself when its binary changes, on
  a peer's silence, on a timer. The squatter barricades the door: guards that make it unkillable, teardown it insists
  on finishing before it will go.
- Signs: toward the watchman, exits reduced to a startup failure and an outside signal, release calls deleted; toward
  the deserter, any stop it starts itself; toward the squatter, anything that delays or refuses a signal.

**M4. Austerity: how much state it keeps.**
- Virtue, the monk's cell: rows, requests, words; nothing it does not need, all of it allocated at start.
- Dereliction, the hoarder's house: sessions, caches, predictions, policies, per-client bookkeeping, each living as long
  as the node and rotting there.
- Signs: toward the cell, deleting a table, a phase or a timer; toward the hoarder, any feature, however useful it
  sounds to some client.

**M5. Silence: how it coordinates.**
- Virtue, the pneumatic tube: things arrive where they were sent without a word exchanged; the only conversation is the
  pairing a new cable needs.
- Dereliction, the bureaucracy: requests, receipts, acknowledgements, appointments, sessions, meetings, stamps. Each is
  a second message about the first, and it can be late, lost or false.
- Signs: toward the tube, deleting a frame kind; toward the bureaucracy, any message whose content is the state of
  another message.

**M6. Custody: who touches the device.**
- Virtue, the vault teller: clients hand over slips naming their own accounts, and only the teller enters the vault. A
  slip naming someone else's account cannot even be written.
- Dereliction, keys under the mat: clients holding work requests, queue-pair pointers and the addresses of words the
  bridge writes, and grabbing queue pairs for themselves.
- Signs: toward the teller, client-visible types reduced to row indices and readiness words; toward the mat, any device
  object in a client's memory.

**M7. Composure: whether it ever waits.**
- Virtue, the air-traffic controller: eyes on every plane, never stepping away; every loop polls and can stop.
- Dereliction, the clerk who goes to lunch with the only key: a kernel call that waits on a device's termination, a
  drain that waits for markers never promised, a wait on a peer.
- Signs: toward the controller, only calls that return and a stop flag read every pass; toward the clerk, any call that
  can wait on another party.

**M8. Honest accounts: its reliability, counted.**
- Virtue, the auditor's ledger: every frame counted and every loss attributed to its cause, a thousand clean frames for
  each loss, and links lost only to physical events.
- Dereliction, the embezzler's books: its own conclusions of links booked as flaky hardware, client errors booked as
  transport errors, losses not counted at all.
- Signs: toward the ledger, counters per port by cause; toward the embezzler, any loss it caused and did not count as
  its own.

**M9. Directness: what it costs a frame.**
- Virtue, the relay runner's clean handoff: no hop, copy or wake beyond the hardware's 2-3 µs.
- Dereliction, the middleman taking a cut: an extra hop, a staging copy, a thread woken to pass something on.

## C. Collective code: the ensemble reading one score, or the committee

At its best a collective is an ensemble playing from one score. Every entry is written down before the performance and
each player knows every other's part. Nobody speaks while playing. If a player collapses, the music stops, and the next
piece starts clean. At its worst it is a committee. Its members confer before each step, and each decides alone when it
feels like it. A missing member is waited for indefinitely, and the minutes carry over to haunt the next meeting.

**C1. Fidelity: how close it stays to its sources.**
- Virtue, the faithful translator, the musician playing as written: NCCL's calls and semantics, the papers' algorithms
  near verbatim (Rabenseifner; Patarasuk & Yuan; Thakur, Rabenseifner & Gropp; Trivance).
- Dereliction, the cover band rewriting the song: invented semantics, home-grown algorithms, features NCCL never had.

**C2. Universality: what it can serve.**
- Virtue, standard-gauge track: any train on any line; every dtype and op, any member count from two, ragged counts,
  any connected map.
- Dereliction, the bespoke part machined for one engine: a variant per model or layer, a form chosen for one shape.

**C3. Unison: how members know each other's parts.**
- Virtue, one score: every length, order and position known to every member from a schedule each compiles identically.
- Dereliction, the committee and the soloist: conferring at run time (meets, agreements, stamps), or one member
  improvising alone and leaving the rest out of time.
- Signs: toward the score, deleting a message between members; toward the committee, any wait for another member's
  word that is not its data.

**C4. Foresight: when the schedule is decided.**
- Virtue, the printed timetable: routes, forms and chunks fixed from the map and the communicator's configuration
  before the call.
- Dereliction, the dispatcher deciding each train's route as it pulls in: routes inferred, forms priced, chunks
  measured during the call.

**C5. Clean endings: what a loss does to it.**
- Virtue, the referee's whistle: on a loss the call stops within its deadline with an error naming what was lost.
  Everyone leaves the field holding nothing, and the next call starts on the next map.
- Dereliction, the hostage negotiation and the ghost: waiting out a missing peer, retrying, resuming, or keeping a
  communicator alive so that its state haunts the next call.

**C6. Tenancy: what it touches.**
- Virtue, the good tenant: its own rows, its supplied and landed words, and device functions offered as a header for
  others to call.
- Dereliction, the burglar, and the surgeon who sews his instruments into the patient: reaching for device objects,
  posting, claiming queue pairs; splicing its code into other programs' kernels.

**C7. Candor about numbers.**
- Virtue, the recipe that states its measures: accumulation precision and reduction order declared, last-bit
  differences between members accepted.
- Dereliction, the nervous inspector stamping every crate: checksums and digests, which become gates.

**C8. Economy.**
- Virtue, water finding the shortest path: latency plus bytes over bandwidth per link, overlapped with compute.
- Dereliction, the tour bus stopping at every gift shop: extra rounds, staging copies, host round trips between steps.

## D. Data code: the tailor with mise en place, or the improvising cook

At its best a program is bespoke work prepared before service. It is fitted to this model and these machines at load,
and every station is prepped. Each cook does the work that suits them, and the kitchen is judged by the plate that
leaves it. At its worst it is an improvising cook in someone else's kitchen. It rummages for ingredients mid-order and
serves the same generic dish to everyone. It inspects each step instead of tasting the result. It grabs the delivery
driver's wheel, and it blames the weather when it is slow.

**D1. Fit.**
- Virtue, the bespoke tailor, the racing car set up for this track: kernels, row classes, splits and attention plans
  made for this model on these machines.
- Dereliction, off the rack: generic kernels, branches at run time choosing among shapes, "works for any model" as a
  goal.

**D2. Mise en place: when its choices are made.**
- Virtue, the chef's station before service: every choice synthesized when the configuration loads, and a missing one
  stops the kitchen before the first order.
- Dereliction, the cook rummaging mid-order: parameters derived inside functions that take tensors, defaults, guesses,
  fallback forms.
- Signs: toward the station, a choice moved into configuration synthesis; toward the rummaging, any branch or
  computation of a parameter where a tensor is in scope.

**D3. Judged by the plate.**
- Virtue, the dish judged by the diner: the whole model's KL at the unembedding, over temperature-1 draws, measured
  after the call.
- Dereliction, the hypochondriac taking his pulse every minute, and the cheat tasting only the garnish: gates inside
  the program, demands for bit identity, greedy outputs taken as evidence.

**D4. Honest pace.**
- Virtue, the athlete against their best time: tok/s end to end against the last best form, differences under 3%
  treated as equal.
- Dereliction, the excuse-maker: slowness blamed on other load or on noise, a different distance run (a component timed
  in place of the call), a regression kept because it reads cleaner.

**D5. The brigade: how members share the work.**
- Virtue, a kitchen brigade: each member does the work that suits it. Shares are ragged, from the calls' evidence;
  every member hosts; nodes run different engines; the second computer is always working.
- Dereliction, the chain gang or the bench: identical shares forced on unequal members, or the weaker member benched at
  0%.

**D6. The passenger: what it knows of the road.**
- Virtue, the passenger who buys a ticket: crossings through the API, and the route unknown and unneeded.
- Dereliction, the backseat driver: reading the bridge's region, watching its process, naming links, peers or routes,
  so that a recabling means a rewrite.

**D7. Provisions: how it gets its memory.**
- Virtue, the provisioned expedition: buffers sized from its own configuration and held from load.
- Dereliction, the lodger on rations or the looter: ceilings imposed by someone else's estimate, or memory taken per
  call.

**D8. Sending the dish back: what a failure does.**
- Virtue, the dish remade: the call fails, returns its error, and the caller runs it again; only caches keyed by content
  outlive it.
- Dereliction, the cook who serves it half-done with an excuse (fallback forms, partial results), or who sets the
  kitchen on fire (shared state held across the failure, cleanup protocols, recovery inside the call).

**D9. One master recipe.**
- Virtue, one recipe for each dish (one layer program for target and drafter, one RMSNorm), and others' recipe books
  respected (LiteRT and torch called through their APIs).
- Dereliction, twenty-three variants of the same sauce, and the vandal scribbling in someone else's recipe book
  (monkeypatching, text splices).

## Inversions

| What the code does | Virtue in | The same act, in another layer |
|---|---|---|
| Ignores what it carries | M: the sealed-letter postman | D: the cook indifferent to the ingredients |
| Never ends on its own | M: the lighthouse, the watchman | C and D: the hostage negotiation, the guest who will not leave |
| Ends at the first loss | C: the referee's whistle | M: the drawbridge that rises for every broken cart |
| Knows every other party's part | C: the ensemble's score | M: the bureaucrat who must be consulted; D: the backseat driver |
| Fits one case exactly | D: the bespoke tailor | C: the part that fits one engine; M: the carrier that favours one shipper |
| Fits every case | C: standard-gauge track | D: off the rack |
| Members do different work | D: the brigade | C: the soloist out of time |
| Adds a feature that measures faster | D: the athlete's new technique | M: the hoarder's new shelf |

They invert for these reasons:
- **Lifetime.** The node outlives a communicator, which outlives a call. Steadfastness in the longest-lived is
  stubbornness in the shorter-lived. A clean ending in a call is desertion in the carrier. The operator, 2026-10-14:
  "functions can fail from any one operand becoming invalidatd, servers should not fail when any operand is invalidated
  (ideally)".
- **What each hides.** The driver and the cables (M), the cabling and the membership (C), the model (D) [Parnas 1972].
  Knowing another layer's secret is meddling, and it makes the meddler change whenever the secret does.
- **Opposite objectives.** Reliability comes by subtraction (M), generality by refusing to specialize (C), throughput
  by specializing (D). Code that serves two objectives serves neither.
- **No protocol.** Agreement exists only by construction (C). A carrier that agrees is a bureaucrat. Members that
  differ do so by configuration they all load, never by negotiation.
- **Privilege** [Anderson 1972; Saltzer & Schroeder 1975]: design/transport.md section 2.
- **End to end.** Correctness is judged at the plate, never at the post office [Saltzer, Reed & Clark 1984].
- **Crash-only.** The way to stop is to end, and the way to recover is the next start [Candea & Fox 2003].
- **Contagion.** A mechanism one layer adds to survive another layer's failure is a second failure. It presumes the
  first will happen, does not remove it, fails in its own way, and hides the first from its owner, and then the two
  spread together. Examples:
  - The squatter: a guard that made processes unkillable against a deadlock any exit could enter.
  - The drawbridge: links concluded to clear receives that a communicator had left posted.
  - The ghost: communicators kept across calls to avoid that conclusion.
  - The committee: meets and stamps against a missing peer.
  - The lodger on rations: ceilings against a window the program could size itself.

  Each failure is removed by its owner and nowhere else.

## Where HEAD sits (2026-10-19)

- **The bridge.**
  - M2, near the drawbridge: links are concluded when a communicator ends.
  - M3, near the deserter: it stops itself when its binary changes.
  - M6, near keys under the mat: clients hold `struct mesh_send` and claim queue pairs on `hdr->lanes`.
  - M8, near the embezzler's books: 100 of the M5 bridge's 176 link losses from 2026-10-16 to 10-19 were its own
    conclusions, and nothing counted them as its own.
  - M3's squatter is gone: the kill guard was deleted on 2026-10-19.
- **libnccl-mesh.** Near the committee on C3 (sessions, meets, stamps) and near the ghost on C5 (communicators kept
  across calls).
- **metal-microbench.**
  - D6, near the backseat driver: mesh_rank.m reads the bridge's region and watches its process.
  - D7, near the lodger on rations: `withinPlan` and `plans` ceilings.
  - D1 and D2, close to the tailor's mise en place since attention plans and instances are synthesized at load
    (2b75b6a, d3d0f4b).

## References

- Anderson, J. P. (1972). Computer security technology planning study. ESD-TR-73-51.
- Candea, G. & Fox, A. (2003). Crash-only software. HotOS IX.
- Parnas, D. L. (1972). On the criteria to be used in decomposing systems into modules. CACM 15(12).
- Saltzer, J. H. & Schroeder, M. D. (1975). The protection of information in computer systems. Proc. IEEE 63(9).
- Saltzer, J. H., Reed, D. P. & Clark, D. D. (1984). End-to-end arguments in system design. ACM TOCS 2(4).
