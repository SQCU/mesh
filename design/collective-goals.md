# Collective goal

The current user instructions define the work. The excerpts below are verbatim
from the September 14 conversation. Papers describe mechanisms; existing code,
agent plans, inventories and old specifications do not add requirements.

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

> never ask for permission to handle integration and build steps

> successful compilation is not important compared to the literal textual goals, which are all about producer async streaming consumer async streaming tensor ops that finish tensor parallel operations (the trivial obvious case of async producer/consumer meshing) faster than the same operations run standalone at worldsize 1.

> you might want to upgrade the prefill kernel to be ready for continued prefill tasks and also specdec draft verifier shaped tasks while yo'ure at it

> delete. the fucking. 'inherited'. implementation. delete it immediately and commit the deletion.

The user-selected continuation objective is the attachment
`5701aa94-8d2a-4238-851b-7ad3af7e91bc/pasted-text-1.txt`. It reiterates these
requirements and actual integration through existing numerical implementations.

The old executor, frontend and callers were deleted in mesh `5762898` and
metal-microbench `e108f4b`. The replacement higher-order interface, collective
relations and source operation chains are described in [the implementation](async-collectives.md).
Finite indexed submissions now reuse the realized functions and routes.
The source chains use existing numerical implementations, including a contraction-
partitioned matrix producer, all-reduce and a column-partitioned numerical consumer.
The earlier completion claim was too broad: the single-use value extent, one-peer
transport, example-only integration and example-local native queue sizing remain
implementation restrictions, not user-authorized definitions of the deployment
target. The implementation description records these gaps and actual transport
costs. Deletion, documentation and successful builds alone are not completion.
