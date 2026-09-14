# Removal tickets: transaction semantics in an RDMA linear-algebra substrate

Status: open work tickets, filed 2026-09-13 by the metal-microbench session. Every ticket is a deletion. None adds a
guard, retry, timeout, readiness message, check or test. A ticket is complete when the named condition is gone from
source, whoever removes it, or when an equal or better removal lands.

Line numbers are as of mesh commit 01f3920. Confirm them by function name, because the tree changes quickly.

## Operator authorization (2026-09-13, verbatim)

> this is rdma code, there is no such thing as 'refusing transactions'. inputs are written to shared paged memory
> immediately, addressed for other operations by paging as an indirection or abstraction over pointer literals to
> encourage zero copy reuse of written values, and have well defined feed forward non recurrent memory allocation use
> and free cycles by construction as we are never doing anything but linear algebra.

> get rid of every guard and sync on publication instead of treating this as a control flow puzzle. this is a memory
> mapping 'puzzle', if it can be considered a puzzle at all, and is not a recurrent control flow determined buffer
> reuse microcontroller 'puzzle' and never will become one.

> nodes do not need to be kept in step by any mechanism because they only need to consume local inputs and write local
> outputs to the remote node to always resynchronize on the same linear algebraic program state within a
> millisecond-scale catchup window.

> you do not need to add guards, tests, regression testers, checks, unit tests, or anythin of the sort; only remove
> abstractions and control flow that you have already derived as unnecessary.

## Model the tickets implement

- **Planned pages.** Every value instance (per invocation, block and participant) has pages planned before execution.
  They are allocated before the first write and released after the last reader. The program is feed-forward linear
  algebra, so these intervals are static. Memory is abundant: an E2B B=8 decode step needs about 9 MiB of transfer
  pages per participant, and a 100-step run with no reuse needs under 1 GiB. The M4 has 24 GiB of shared memory.
- **Publication.** A publication is a page-table presence update plus a SEND post. Nothing is refused.
- **Receives.** Receives are posted on the consumer's planned pages ahead of arrival. A consumer reads its operands
  wherever the page table maps them.
- **No lockstep between nodes.** Each node consumes local inputs and writes outputs to the peer's pages.

## Literature (quotes as verified in collective-dependency-ledger.md)

> The Thunderbolt controller hardware ensures this send work request isn't processed until the receiving Mac has posted a matching receive work request using a credit-based flow control system.
> — [TN3205] (D2)

> In order to prevent getting into RNR, one needs to always make sure that there are enough Receive Requests in the Receive Queue.
> — [RDMAmojo] (D2)

> Host B allocates B's inputs, transmits the input buffer addresses to host A, and performs most of the preparatory work to launch node B's function. When node A completes, its outputs are sent via the accelerator interconnect directly into node B's input buffers, and then host B starts node B. The latency between one node completing and the next node starting can be made to be little more than the data transfer time.
> — [Pathways] §4.5 (D4)

> (ii) enqueueing network sends to remote accelerators for the buffer futures output by function executions
> — [Pathways] §4.4 (D7; ledger license: "Transport completion never gates a producer's next issue.")

> Completion of a send operation indicates that the peer has posted sufficient receive requests to receive all sent frames and that the Thunderbolt hardware has sent the buffers to the peer. Completion does not indicate the receiver has successfully received the sent data or that data was not corrupted in flight because Thunderbolt does not perform Acks in hardware.
> — [TN3205] (D12; ledger cuts: "acks, readiness frames, handshakes, health checks in the progress loop")

> At this point, applications should use an out-of-band communication mechanism to share the GID and queue pair number (QPN) with a peer.
> — [TN3205] (D13)

> This approach splits both GEMMs in the MLP block across GPUs and requires only a single all-reduce operation in the forward pass (g operator)
> — [Megatron] (D11)

Pallas collective matmul (algorithm-sources.md#streaming-algebra):

> we can't begin computing the matrix multiplication while the all-gather is running! That means we're underutilizing our hardware!

## Tickets

### R1. Pair once per bridge process, not per client

**Remove:**
- the client-follow teardown, `mesh-flow.c:169` (`if(link.client && client!=link.client && link_down(&link)) continue;`)
- the client/configured pairing gate, `mesh-flow.c:170` (`if(!link.client && client && configured==client)`)
- `link_down` on client change (`mesh-flow.c:95-110`)
- client takeover, `mesh-dataflow.c:49-54`
- `mesh_retire`, both at takeover and in detach (`mesh-dataflow.c:21-32`, `:59-63`)

**Why:**
- D13: connection metadata is exchanged out of band once.
- Operator: nodes need no mechanism to be kept in step.

**Incident.** In run e2b-spmd-050b a killed process left `client == configured == dead pid`. The M5 bridge paired for
the dead pid, then tore down when the real participant took over the region. The M4 bridge's client never changed,
so it stayed paired with a peer that was gone. No path re-entered pairing. Each side's connection state followed its
own local client transitions, and that asymmetry is the stall.

**Ledger text to amend.** D14 ("A connection lives as long as the client program that uses it", in-process pairing
retry) and the second D16 ("when the attached owner has realized") require these gates.

### R2. Delete the setup readiness byte

**Remove** `mesh-verbs.h:216-218`: the setup receive post followed by the one-byte `ready` exchange before RTS.

**Why:**
- D2 (TN3205 credit flow): a SEND is not processed until a matching RECV exists.
- D12 cuts readiness frames and handshakes.

**Text to amend.** `streaming-algebra.md:139-141` ("completes a bilateral setup boundary before sending") mandates the
byte.

### R3. Delete output-claim refusal and the PRODUCING plane

**Remove:**
- the output claimability gate in `mesh_issue`, `mesh-dataflow.c:305-307` (`mesh_claimable` over outputs)
- `mesh_claimable`, `mesh-dataflow.c:272-285`
- the PRODUCING reset/set, `mesh-dataflow.c:307`
- the send reader plane that makes a producer's next issue wait for its previous SEND (`mesh-dataflow.c:213-218`,
  `:282-283`)

**Why:**
- D7: transport completion never gates a producer's next issue. These gates exist only to stop a reused page being
  overwritten.
- The operator's planned pages remove reuse, so each output instance is fresh by construction.

### R4. Post receives for every planned instance up front; delete send/receive postable gating

**Remove:**
- the reader-gated receive repost, `mesh-flow.c:45` (`if(!mesh_receive_postable(M,row)) break;`), and the D9 repost
  rule that motivates it
- the send-postable break, `mesh-flow.c:66` (`if(!entry.plane || !mesh_send_postable(...)) break;`), which holds a
  queue's later sends behind its first unproduced block

**Why:**
- D2 (RDMAmojo): always keep enough receive requests.
- D4 (Pathways §4.5): inputs are allocated and addressed before the producer completes.
- With planned distinct pages there is nothing to repost after readers, and nothing to hold behind an unproduced
  block. Each block's SEND is posted when it is published.

### R5. Delete slot-modulo reuse in the streaming algebra

**Remove** from `streaming-algebra.md:72-77` and the matching `mesh-algebra.m` slot logic:
- "Invocation i uses slot i modulo R."
- "The next invocation in that slot is produced after that slot's outputs are consumed"

**Why.** The larger-buffer approach the same section cites ("distinct live value instances receive distinct pages")
is applied fully. Instances are planned for the run's invocations with no modulo reuse and no produced-after-consumed
rule.

### R6. Setup failure ends the run; delete in-process re-pairing loops

**Remove the silent retry paths:**
- `mesh-flow.c:172-176`: re-pair after a `verbs_up` failure
- `mesh-flow.c:166`: listener retry
- `mesh-verbs.h:134`, `:143-149`: silent -1 when the port is inactive or the device is missing

**Replace with:** one setup per bridge process. A failure is recorded as D12 status and ends the process's setup.

**Why.** A retry loop that can diverge between the two sides is a stall source (R1). D12 makes errors status, out of
band.

### R7. Delete competing whole-value API surface

**Remove:**
- `mesh_algebra_transfer` (`mesh-algebra.h:43`, `.m:548-556`): no callers; superseded by `mesh_algebra_copy`
- whole-extent `mesh_algebra_return` and `mesh_algebra_export` (`.h:44-45`, `.m:610-622`), in favour of
  `mesh_algebra_return_part`
- whole-extent `mesh_tensor_issue`, `mesh_tensor_complete` and `mesh_tensor_publish` (`.h:36-38`); host producers
  become part-indexed
- `rdma/mesh_numpy.py`, `examples/streaming-expression.py` and its Makefile rule: a second symbolic front end
- the public `module Mesh` and the `setup.py` install of `mesh-dataflow.h`, so callers cannot bypass the algebra
  layer

**Why.**
- `backend-streaming.md`: "no second graph interpreter … no optional publication hook".
- `streaming-algebra.md`: publication cannot be omitted through the API.

### R8. Ledger consistency

- The ledger has two sections titled D16. Renumber one.
- Amend D14, D16 and `streaming-algebra.md:139-141` per R1 and R2.
- Amend D9 per R4.

### R9. Delete the host scan; fire functions from the events that change presence

Filed 2026-09-13. Line numbers are from the mesh `c6ad5cc` working tree.

> submissions shouldn't be synchronous, so you have identified yet another exrtemely basic problem, as both producers
> and consumers must always be waitless, guardless, syncless, and totally determined by local node state
> — operator, 2026-09-13

**Remove:**
- `mesh_algebra_scan` (`mesh-algebra.m:679-682`). A host thread sweeps every bound function and calls `mesh_issue` on
  each, so a function starts when some thread polls, not when its operands arrive.
- Synchronous submission. `emit_part` (`.m:675-677`) runs `f.execute` on the scanning thread, and so does the
  `mesh_algebra_function` binding (`.m:335`). Python's `Program._call` `submit` (`__init__.py:258-265`) runs the NumPy
  kernel inline on that thread, so one kernel's arithmetic stalls every other function's start.
- Caller scan loops: `examples/streaming-algebra.py:55-62`, `Program.scan`, and `while not result.ready: scan()`.

**Replace with the dataflow firing rule, applied where presence changes:**
- **Local completion.** After `mesh_complete` (`mesh-dataflow.c:336-342`) publishes its rows, the functions that read
  those rows and now have every operand present are submitted from that completion. Their reader lists are fixed at
  realize, as reader slots already are.
- **Receive completion.** `mesh_receive_complete` (`mesh.h:87-94`) sets presence in the bridge on a RECV completion
  (D8). That presence change is the event that submits the consuming functions. The path from the bridge's
  completion to the client's submission is the open part of this ticket, and it is not a sweep.
- **Every submission returns immediately.** Metal submissions already encode, commit and return, with completion in
  the command buffer's handler. CPU and NumPy kernels run on their own execution context and call completion when done.

**Why.**

> operators may fire as soon as tokens arrive at their inputs
> — [TTDA] (D8)

> When node A completes, its outputs are sent via the accelerator interconnect directly into node B's input buffers,
> and then host B starts node B.
> — [Pathways] §4.5 (D4)

> … but nothing can prevent a process from performing a send on a line.
> — [Kahn] (D7)

- The bridge's completion-queue poll (D3) is the transport's licensed progress loop. It does not license a numerical
  scan in the client.
- A sweep over all functions makes start latency a property of the sweeping thread. It serializes CPU kernels behind
  one another and ties progress to a host loop instead of to local node state.

## Caller side (metal-microbench, done by that session)

Deletes the second executor and the collective/ABI duplicates: `mesh_graph`, `mesh_shaders`, `mesh_context`,
`mesh_replay`, `mesh_declaration`, `mesh_metrics`, `mesh_configure`, `mesh_ffi`, `mesh_port_declaration`, `mesh_abi`,
`reduce_scatter`, the NFE/decode launchers and the research duplicates. The decode caller is rewritten onto
`mesh-algebra.h` only.
