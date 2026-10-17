# Collective goal

The current user instructions define the work. The excerpts below are verbatim
from the September 14 conversation. Papers describe mechanisms; existing code,
agent plans, inventories and old specifications do not add requirements.

Superseded by the operator's 2026-09-22 telos in [collective-goals.md](collective-goals.md).

> 1: SOME FUCKING HIGHER ORDER FUNCTIONS FOR PARTIAL TENSORS 2: THE PARTIAL TENSORS NEED TO HAVE COLLECTIVE COMMS TOO 3: OF COURSE THE WHOLE THING HAS TO BE ZEROCOPY AND ASYNC ONCE WE'VE FIGURED OUT THE CALLGRAPH AOT.

> the *runtime* is necessarily extremely ismple, ebcause any artificially imposed control flow puts more cache reads and loads in between 'work is ready on peer' and 'work has arrived to this meshnode from peer and is being used'.

> all responsibilty lands on the CALLER to supply a MESH AND TENSOR PLACEMENT CONFIG which is USEFUL instead of USELESS.

> we are here to do IMPLEMENTATION OF DISTRIBUTED COLLECTIVE PARTIAL TENSOR STREAMING CODE. we DO NOT NEED TO IMPLEMENT EVERY SINGLE ALGORITHM EVER INLINE INSIDE OF OUR MESH CODE. we can in fact SEPARATE THE TRANSPORT AND PARTIAL PUBLICATION AND SO ON from the FUNCTION WHICH IS RUN OVER THE TENSOR PARTIALS

> implement streaming independently of each collective comms pattern, implement all of the collective communication patterns as async non blocking streaming operations, and worry about which you'll need later (at calling function def time)

> these hardware constraints simply mean that all queues must be drained IMMEDIATELY WITHOUT EXCEPTION WITH NO WAITS SEPARATING QUEUES FROM BEING FILLED AND EMPTIED.

> there is no reason for the send/receive rdma work to share a thread with any other work under any situation for any reason

> your task here isn't to introduce incompatibility with core ml or ANE

> a partial tensor can literally be made of several contiguous sectinos which are globally contiguous but are also locally contiguous so that tehy can be individually consumed by different contiguous-typed function launches.

> have a subagent literally write a refcounter which doesn't rely on users of buffers specifically and literally inlining some kind of 'please say im done using the resource' explicit free like we're in c.

> remember that garbage collection in this case simply means 'releasing' pages into the 'pool of pages you can write to if you wanna lol' without even needing to zero them lol

> if we want to supply a 'sync_on_remote_fill' barrier, this is library code that needs to be used EXPLICITLY AND DIEGETICALLY IN CALLING CONTEXTS, it can NEVER be a default behavior.

> we must now implement an explicit 'sync on remote fill' call that our library users can use, and a peice of demonstration code which shows it is slower and can deadlock forever

> the goal has not been achieved until the requirements are implemented in source and *demonstrated* ergo *used* for a streaming producer/consumer operation chain.

> runtime testing is not part of any requirement and does not inform any actions

(Superseded 2026-09-18 and 2026-09-22: measuring Goal A and Goal B on the pair is the task.)

> never ask for permission to handle integration and build steps

> successful compilation is not important compared to the literal textual goals, which are all about producer async streaming consumer async streaming tensor ops that finish tensor parallel operations (the trivial obvious case of async producer/consumer meshing) faster than the same operations run standalone at worldsize 1.

> you might want to upgrade the prefill kernel to be ready for continued prefill tasks and also specdec draft verifier shaped tasks while yo'ure at it

> delete. the fucking. 'inherited'. implementation. delete it immediately and commit the deletion.

## The September 13 and 14 sessions' lines deleted with collective-goals.md (restored 2026-10-17)

mesh a07d7f66 (2026-09-14, "Remove obsolete collective specifications and anchor scope in user instructions") deleted
the operator lines below with design/collective-goals.md's normative section; they are restored verbatim with the
session and time the record gave each (AGENTS.md: "Verbatim operator quotes outrank everything else here"). Lines of
that record still held elsewhere are not repeated: "pallas is to provide an api header style ..." (2026-09-14T16:51:47Z,
[SPECIFICATION.md](SPECIFICATION.md) section 25) and "successful compilation is not important ..." (above).

Session `01a09bf3`, 2026-09-13T18:21:35Z:

> how do we implement the streaming producers and streaming consumers we want, which
> explicitly take chunks of a tensor which can be partially processed elementwise, and
> begin that partial processing without waiting for a big operand (which would have been
> fed into a matmul implementation one 'tile' at a time anyways once it was ready)?

Session `01a09bf3`, 2026-09-13T18:25:15Z:

> mesh can give us a way to use the rdma link through generalized linear algebraic
> operations and scatters and allgathers and reduces and stuff like that; a dnn ops
> library can then implement dnn modules using compositions of that linear algebra

> "I would make the first acceptance program streamed producer → peer sum → elementwise
> transform → contraction. Deliberately delay an unrelated extent" yeah so true bestie i
> agree, this is a good starting scope

Session `01a09bf3`, 2026-09-13T19:06:48Z:

> we should have a trivially deducible ring-of-buffers structure where we use MORE of
> the shmem on our peer computers by passing DIFFERENT OUTPUTs into DIFFERENT TARGET
> BUFFERS ... remember that we're here to implement the pallas async tp collective code,
> not to rewrite or reaffirm whatever ended up in mesh.

Session `01a09bf3`, 2026-09-13T19:27:14Z:

> it's time to add this inner 'mesh send function use' inside of the linear algebra
> kernels we use for our collectives so we never give users the impression they have the
> *option* to make or use mesh implementait[ions that stall]

Session `01a0a0ca`, 2026-09-14T16:42:04Z:

> we're workign on implementing a pallas like interface for distributed collective
> operations which don't block, don't expose low level synchronization or memory table
> implementation details to callers, and induce streaming operations by default for both
> producers and consumers of linear algebraic work being passed along meshes for work
> sharing. you're descirbing a whole lot of code which isn't part of hte mesh repo as it
> was last bashed out in specification, and also isn't necessary for the pallas like
> interface, suggesting most of the code you're 'fixing' incrementally is totally
> superfluosu

Session `01a0a0ca`, 2026-09-14T16:49:23Z:

> why are there cost models lol we're talking about a scituation where we specified
> asnyc producers, async consumers, and all responsibilty lands on the CALLER to supply
> a MESH AND TENSOR PLACEMENT CONFIG which is USEFUL instead of USELESS.

Current session, 2026-09-14:

> a definite list of goals that conclude in streaming tensor partial collective ops
> that, trivially, interleave and don't have a bunch of wrapping abstractions or control
> flow that stop them from being literally used

> it is so much simpler to literally implement distributed collective code instead of
> doing all of these other interventions though. the goal is really not open ended at
> all; it's *literally* 'there is a need for overlapped streaming tensor partials.
> implement overlapped streaming tensor partials. no, seriously, still streamed, still
> tensor, still partials, still overlapped.'

Current-session clarification:

> we do not even need to run the programs to know whether they satisfy our
> distributed collective async properties

(Superseded 2026-09-18 and 2026-09-22: measuring Goal A and Goal B on the pair is the task.)

Current-session constraint on implementation work:

> we are talking about RDMA CODE ON WEAK PROCESSORS! we cannot HIDE any LATENCY
> behind SOME OTHER OPERATION and ASSUME we can WASTE A BUNCH OF OPERATIONS OR
> CONTEXT SHIFTS. that is why the requirements are SIMPEL and mention SIMPLICITY SO MUCH

From the same record's implementation inventory, 2026-09-14 (the current conversation then):

> views aren't intrinsically contiguous

> did the transcript say to write a refcounter and garbage collector? is it
> possible to write a refcounter for buffers over a known data flow when you know
> functions are completely feed forward?

The subsequent deployment clarification is also authoritative:

> we should note that writing something which sounds like it satisfies all contraints but is slow and doesn't utilize flops on two computers concurrently would be 'fake'. so would anything which is depthwise recurrent, like a resnet with 4 ffn-residual-layers, running slower with the mesh collectives than without them, because some asshole decided to add a global sync or global guard or global wait again

> the importable path is that oyu are writing a tensor function invariant streaming partial function library. what do you think the linear algebra notes earlier meant?

The user's repeated-block workloads motivate actual use of this invariant
library. They do not authorize a model-specific API or numerical implementation
inside Mesh. Block counts and function compositions belong to the caller.

The subsequent transport clarification distinguishes two abstraction levels:

> there's no relationship between the transport level idea of streaming and our idea of partial tensors to have early emit by producers and asynchronous concurrent placement for consumption btw; tehse are two harmonious and different notions of streamedness which affect two different elvels of abstraction.

> remember that a streaming transportmessagesize transport layer implementation should be totally transparent to all users of a send or receive api call exposed through our mesh api; this is something which is purely internal and have no effect on any caller or user of this api at any point in the future.

The numerical partial is not a transport request. A tensor's byte count must not
be limited by one request, and internal fragmentation must not change its shape,
partition, numerical function or public send/receive declaration.

The subsequent exact-tail clarification is authoritative:

> an exact tail change has to implement exact tails by choosing what to push to the send interface instead of trying to do weird truncation logic inside of the send or recv interface. throw out the requirement if the requirement is stupid / cooked / washed / chopped

Tail selection belongs to preparation of the messages submitted to transport.
It does not authorize SEND or RECV to reinterpret or shorten those messages.
The current implementation prepares matching native request extents and retains
padding; it does not implement padding-free wire tails. The incorrect requirement
to shorten a SEND against an incompatible preposted RECV is withdrawn. See
[prepared native requests](pages-and-functions.md#prepared-native-requests).

The user-selected continuation objective is the attachment
`5701aa94-8d2a-4238-851b-7ad3af7e91bc/pasted-text-1.txt`. It reiterates these
requirements and actual integration through existing numerical implementations.

The old executor, frontend and callers were deleted in mesh `5762898` and
metal-microbench `e108f4b`. The replacement higher-order interface, collective
relations and source operation chains are described in [the implementation](async-collectives.md).
Finite indexed submissions now reuse the realized functions and routes.
The small matrix example and its build target have been deleted at the user's
instruction. They no longer serve as evidence of integration.
The fixed scalar chain has also been deleted. Its replacement is the
[configured function chain](function-chain.md), which uses supplied Core ML
functions, reduce-scatter and supplied consumers at each declared depth. The
function representation is shared by calls, maps and reductions; model stages
remain caller code.
The earlier completion claim was too broad: the single-use value extent and
example-only integration remain
implementation restrictions, not user-authorized definitions of the deployment
target. The implementation description records these gaps and actual transport
costs. Deletion, documentation and successful builds alone are not completion.

The subsequent peer-count clarification is authoritative:

> special casing one peer is an error in library design for the same reason as special casing exactly one tensor function ever and no guardless continuation of work between multiple tensor functions.

> note that peer>1 prohibits the use of type inference of broadcasts or alls rather than defined scatters and defined gathers

The one-peer restriction has now been removed from the Swift rank bound, endpoint
binding, bridge configuration, connection ownership and send notification layout.
The [configured links](async-collectives.md#configured-peers) share canonical
storage and device registrations, with independent TX/RX progress. Collective
verbs remain explicit and are not inferred from rank count or tensor types.
