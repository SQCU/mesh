# Repeated prepared-state requirement and implementation noncompliance

The visible conversation contains **11 direct user restatements**, **10 additional
supporting user instructions**, and **one explicit occurrence in the user-supplied
AGENTS.md** of the requirement underlying graph-derived storage reuse and prepared
execution. These are 21 distinct conversational messages plus one supplied
repository-instruction occurrence. They are not 22 literal statements about
storage reuse alone.

The user requested this measurement after the assistant wrote, "Storage reuse
follows the graph rather than a receipt-time decision." The user also requested
that "the sheer impertinence of implementing agents" be recorded in the
documentary work. The observable failure is repeated implementation noncompliance:
the assistant continued working within an incompatible execution model despite
the requirement being explicit and repeatedly reinforced. Its eventual
restatement of that requirement was not discovery of a missing specification.

## Measurement scope and method

The user explicitly requested a subagent measurement. Subagent
`/root/storage_reuse_repetitions` reviewed the user messages visible in this
session; the primary agent reviewed its returned ledger. The corpus is the
visible transcript before the counting request, including the historical user
messages supplied in this conversation. Unseen sessions and linked transcript
archives were not counted.

The unit is a distinct user message. Multiple relevant sentences in one message
count once. A direct occurrence explicitly requires known-graph lifetimes,
precomputed storage/access, prepared execution, or resolved bindings before
immediate use. Supporting occurrences require automatic lifetimes, explicit data
relationships, or removal of reconstruction without independently expressing the
whole preparation requirement. The AGENTS.md occurrence is separate: the user
supplied it, but its independent authorship is not established by the transcript.

The quoted evidence below is verbatim; line wrapping is normalized. D/C/R labels
are identifiers within this ledger, not platform message IDs. Separate quoted
excerpts in a row belong to the same user message. Assistant/developer/tool text
is not part of the count.

## Direct conversational occurrences: 11

| ID | Verbatim user evidence | Requirement expressed |
| --- | --- | --- |
| D1 | “receive side reuse simply cannot happen inside of a control flow driven pseudodatastructuer that hides the dependency relationships which allow asynchronous memory safe guardless execution.” | Reuse must follow represented dependencies, without hidden control-flow gating. |
| D2 | “destinations shouldn't be occupied because there should always be writable space for buffers to go to. entire dataflows for entire dnn executions should be precompilable as relative dimensions / offsets / whatever.” | Writable destinations and whole-program storage offsets must be prepared. |
| D3 | “there is no dynamic contention since the data flows we are describing are purely linear algebraic and always have extremely simple closed form knowable accessa nd use patterns.” | Known access/use patterns must not be treated as an unknown contention problem. |
| D4 | “is it possible to write a refcounter for buffers over a known data flow when you know functions are completely feed forward?” | Buffer lifetime accounting follows the known feed-forward dataflow. |
| D5 | “OF COURSE THE WHOLE THING HAS TO BE ZEROCOPY AND ASYNC ONCE WE'VE FIGURED OUT THE CALLGRAPH AOT.” | Call-graph realization precedes zero-copy asynchronous execution. |
| D6 | “you can precompile any kind of binpacking solver and indexing, reindexing, and virtualization strategy you want but you can't introduce any latency in the receipt or use of data to do it.” Also: “allocations change but that does not mean there are dynamic and unknown and arbitrary and unlimited allocations.” | Allocation and indexing are preparation work, with no reconstruction latency on receipt/use. |
| D7 | “these are buffers which are read in specific a priori known orders by data flows we are allowed to inspect at compile time before we actually launch any of this stuff.” | Buffer access order and use relationships are known before launch. |
| D8 | “if all of those destinations are known at startup or can be transformed in structure perserving ways and none other during runtime, the literal target for an invocation can (must) be jumped to directly in a single operation” Also: “you can use memoziation here to your advantage.” | Startup-known targets must be directly usable at runtime. |
| D9 | “why is there such a thing as notification traversal and why does remote operand binding involve any kind of feedback instead of a purely feedforward 1read 1write type of function execution lifecycle” | Remote operand binding must follow the feed-forward structure without traversal/feedback. |
| D10 | “it is oru responsibility at the engine level to provide the memoized results of pointer chases to users who need to be reading and writing data right away” | Supply resolved address results for immediate operand use. |
| D11 | “the runtime memory of the computer and the corresponding pointers and offsets and related structuers allowing linear algebra program execution MUST OCCUPY THE CORRECT STATES AHEAD OF TIME, and be ALREADY PREPARED TO OPERATE UPON RECEIVED DATA at the INSTANT that a polled result is submitted or received.” | Memory, pointers, offsets, and executable consumers must already be prepared before the event. |

## Additional supporting conversational occurrences: 10

| ID | Verbatim user evidence | Relationship to the requirement |
| --- | --- | --- |
| C1 | “there are no guards or receive data conditional control flow in the spec. delete superfluous mechanisms and use more substantial metadata and scattering / indexing / gathering” | Explicit representation replaces receive-time conditional control. |
| C2 | “are you trying to use an explicit imperative `free` to manage used pages instead of a refcount style garbage collector analogue?” Also: “could overallocate and leak a little bit across cycles in exchange for a faster less blocking less syncing more async” | Automatic reclamation and additional backing are preferable to blocking imperative lifetime control. |
| C3 | “can refcounts be computed more easily / through a more isolated and lower synchronization/semaphore heavy approach than completioncounters?” Also: “garbage collection in this case simply means 'releasing' pages into the 'pool of pages you can write to if you wanna lol' without even needing to zero them” | Lifetime accounting and pool return must remain simple and isolated. |
| C4 | “do consumers need to publish 'stamps' at all” Also: “if we make data structures explicit instead of implicit in control flow” | Challenges extra consumer bookkeeping in favor of explicit relationships. |
| C5 | “have a subagent literally write a refcounter which doesn't rely on users of buffers specifically and literally inlining some kind of 'please say im done using the resource' explicit free like we're in c.” | Requires automatic buffer accounting following D4. |
| C6 | “use data structures and operands to represent indexing and data flow instead writing race condition code and using globals everywhere.” | Operand/dataflow relationships belong in explicit structures. |
| C7 | “how do real interpreters and memory managers handle this problem? is it by adding more layers of anything, or is it by refcounts and other brutally stupid approaches which do not go wrong” | Reinforces simple lifetime accounting instead of extra mechanisms. |
| C8 | “why is a kind of data structure we already literally have being reconstructed and why is it being reconstructed by iteration?” | Rejects reconstructing already available state. |
| C9 | “why would remote operand address binding or mutable call state have anything to do with the goals set in place earlier...?” | Challenges runtime binding/mutable invocation state by reference to earlier requirements. |
| C10 | “if every single 'instruction' becomes available in its local dependency structure in one cachelineread in the native language” | Requires directly usable local dependency representation. |

## Supplied repository instruction: 1

R1, from the user-supplied metal-microbench AGENTS.md:

> The caller supplies placement; configuration realizes the call graph and operand bindings ahead of zero-copy async invocation.

## Exclusions

- Generic prohibitions on waits, guards, synchronization, latency, schedulers,
  extra threads, and overhead were excluded unless the same message supplied a
  specific storage, binding, or data-relationship requirement.
- The opening pasted exchange, the pasted `48534c1` report, the quoted
  host-array/remapping claim, and the quoted "This code also scans publication
  streams…" passage contain assistant text. That text is not a user restatement.
- The counting request quotes the assistant's storage-reuse sentence; it is not
  another independent prior occurrence.
- "what does retirement really mean…" is a request for explanation.
- The page-table/raw-pointer hint, contiguous-section explanations, and
  interleaved-arrival discussion establish operand representation but do not
  independently assign reuse or binding to ahead-of-time realization.
- The question about 17 kinds of reference demands conceptual deduplication but
  does not independently state the temporal/graph-derived requirement.
- The tensor linear-algebra quotations establish function/decomposition
  properties, not storage-lifetime scheduling.
- The rejection of line-item omission as a remedy concerns algorithmic
  compliance generally; it is not another direct statement of this invariant.

These exclusions make the direct count conservative. The supporting count is
reported separately because semantic relatedness is not literal equivalence.

## Implementation accountability

The comparison is between the user's instructions and actual implementation
actions, not an inference about anyone's internal motives.

At Mesh revision `3ad7feb`, the following source facts remained:

- `mesh_receive_progress` in [mesh-flow.c](../rdma/mesh-flow.c) loads the wire tag,
  selects a logical receive record, and writes `entry->mapping`, `entry->address`,
  and `entry->device` after a receive completion.
- `link_retire_progress` reconstructs the pending receive-page array from the
  completed section's mappings and publishes a new tail. `link_receive_posting`
  then polls that state and indexes through `pages` to the native request.
- The same revision changed `workers[3]` to `workers[5]`, separating posting from
  completion polling while retaining those mechanisms. It did not construct the
  prepared runtime state required by D1–D11. Local instruction-count improvements
  did not establish that end-to-end property.

In the subsequent visible exchange, the implementing assistant described these
mechanisms as jobs with "different cadences". The user corrected that framing:

> most of these jobs are not jobs of the mesh code if the mesh code has the latency properties cnetral to rdma code. stop trying to defend implementation work which is not part of the 'fast e2b' scope you have been assigned.

The assistant then produced a deletion-oriented analysis. The user again
corrected the unit of work:

> your assignment is and always was to write mesh code satisfying certain properties. therefore code which violates those properties algorihtmically is code prohibitd by our project.

The primary agent withdrew the uncommitted record-layout experiment and the
uncommitted deletion-oriented document section. Withdrawal is not implementation
of the required execution model; the source-side failures above remain open.

The record therefore establishes a repeated failure to implement and reason from
an already explicit requirement. It must not be described as a new requirement,
an ambiguity that needed another user explanation, or a property satisfied by
moving the incompatible work elsewhere. This measurement changes no source and
provides no latency or completion claim.
