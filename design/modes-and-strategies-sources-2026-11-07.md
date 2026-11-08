# The driver's two-sided sends, MPI's communication modes and completion, and the compile-time scheduling strategies: sources, quoted verbatim

Retrieved 2026-11-07. Each block below was copied by script from the cited lines of the file named; nothing in a block is retyped.

## 1. Apple, TN3205

- **Source:** Apple, "TN3205: Low-latency communication with RDMA over Thunderbolt" (TN3205), https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt.
- **Where read:** Apple's documentation JSON for that page, `https://developer.apple.com/tutorials/data/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt.json`, saved as `/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit9/tn3205.json`; its paragraphs, list items and notes written out in order by a script as `lit9/tn3205.txt` (a note is marked `[note]`).

`tn3205.txt` L187, feature list:

```text
- Send and receive operations only
```

`tn3205.txt` L240, registering memory:

```text
[note] RDMA over Thunderbolt only supports 2-sided operations and not hardware-initiated remote writes, thus applications should only register memory as `IBV_ACCESS_LOCAL_WRITE`.
```

`tn3205.txt` L340–341, L363, "Preparing to receive":

```text
## Preparing to receive
Once an application has connected a queue pair to its peer as described above, it can post a receive work request (WR) to indicate to the Thunderbolt hardware where incoming data should be written. When a sender sends a matching frame, the Thunderbolt controller directly copies the data into the address specified in the receive work request.
[note] Thunderbolt devices require a receiver and sender to post messages which are the same number of frames long. For example, if the sender sends a 16K message and the receiver posts a 4K or 32K receive buffer the receive operation will fail.
```

`tn3205.txt` L366, L392, sending:

```text
To send data to a peer, applications should call `ibv_post_send` to send data from a pre-registered buffer. The Thunderbolt controller hardware ensures this send work request isn’t processed until the receiving Mac has posted a matching receive work request using a credit-based flow control system.
[note] Completion of a send operation indicates that the peer has posted sufficient receive requests to receive all sent frames and that the Thunderbolt hardware has sent the buffers to the peer. Completion does not indicate the receiver has successfully received the sent data or that data was not corrupted in flight because Thunderbolt does not perform Acks in hardware.
```

`tn3205.txt` L166, a collective in the note's own words:

```text
For some workloads, such as collective communication operations, applications might allow intermediate nodes to perform portions of the computation rather than merely forwarding bytes. Consider an “All Sum” operation where each node adds its buffer to every other node’s buffer. Rather than forward all buffers to one node which performs the addition, intermediate nodes might receive from one neighbor, add their buffer to the running sum, then forward the result to the next neighbor.
```

## 2. MPI 4.1: communication modes, and completion

- **Source:** as design/completion-and-period-sources-2026-11-07.md §1 (MPI 4.1), text by `pdftotext -layout` of `lit3/mpi41/mpi41-report.pdf`, `lit3/mpi41/mpi41-report.txt`; bare numbers are the PDF's margin line numbers.

`mpi41-report.txt` L7240–7252, L7258–7274, §3.4 "Communication Modes", buffered and synchronous modes, pp. 50–51 (the bare numbers between lines are the PDF's margin numbers):

```text
          A buffered mode send operation can be started whether or not a matching receive
45
     has been started. It may complete before a matching receive is started. However, unlike the
46
     standard send, this operation is local, and its completion does not depend on the occurrence
47
     of a matching receive. Thus, if a send is executed and no matching receive is started, then
48
     MPI must buffer the outgoing message, so as to allow the send call to complete. An error will
51                                                                  3.4 Communication Modes

                                                                                                  1
occur if there is insufficient buffer space. The amount of available buffer space is controlled
     A send that uses the synchronous mode can be started whether or not a matching
                                                                                                  5
receive was started. However, the send will complete successfully only if a matching receive is
                                                                                                  6
started, and the receive operation has started to receive the message sent by the synchronous
                                                                                                  7
send. Thus, the completion of a synchronous send not only indicates that the send buffer
                                                                                                  8
can be reused, but it also indicates that the receiver has reached a certain point in its
                                                                                                  9
execution, namely that it has started executing the matching receive. If both sends and
                                                                                                  10
receives are blocking operations then the use of the synchronous mode provides synchronous
                                                                                                  11
communication semantics: a communication does not complete at either end before both
                                                                                                  12
MPI processes rendezvous at the communication. A send executed in this mode is nonlocal.
```

`mpi41-report.txt` L7276–7292, §3.4, ready mode, p. 51:

```text
     A send that uses the ready communication mode may be started only if the matching
                                                                                                  14
receive is already started. Otherwise, the operation is erroneous and its outcome is unde-
                                                                                                  15
fined. On some systems, this allows the removal of a hand-shake protocol that is otherwise
                                                                                                  16
required and results in improved performance. The completion of the send operation does
                                                                                                  17
not depend on the status of a matching receive, and merely indicates that the send buffer
                                                                                                  18
can be reused. A send operation that uses the ready mode has the same semantics as a
                                                                                                  19
standard send operation, or a synchronous send operation; it is merely that the sender
                                                                                                  20
provides additional information to the system (namely that a matching receive is already
                                                                                                  21
started ), that can save some overhead. In a correct program, therefore, a ready send could
```

`mpi41-report.txt` L9757, §3.7.3 "Communication Completion", MPI_WAIT:

```text
     A call to MPI_WAIT returns when the operation identified by request is complete. If the      35
```

## 3. Bhattacharyya, Sriram and Lee (1997): the scheduling strategies

- **Source:** as design/completion-and-period-sources-2026-11-07.md §4 (the authors' preprint, `lit3/bsl1997/synch.pdf`), its text `lit3/bsl1997/synch.txt`; the preprint's page 3. Its reference [17] is E. A. Lee and S. Ha, "Scheduling strategies for multiprocessor real-time DSP", Globecom 1989.

`synch.txt` L58–79, §1, p. 3:

```text
Multiprocessor implementation of an algorithm specified as a DFG involves scheduling the actors.
By “scheduling” we collectively refer to the tasks of assigning actors in the DFG to processors, ordering
execution of these actors on each processor, and determining when each actor fires (begins execution) such
that all data precedence constraints are met. In [17] the authors propose a scheduling taxonomy based on
which of these tasks are performed at compile time (static strategy) and which at run time (dynamic strategy); in this paper we will use the same terminology that was introduced there.
In the fully-static scheduling strategy of [17], all three scheduling tasks are performed at compile
time. This strategy involves the least possible runtime overhead. All processors run in lock step and no
explicit synchronization is required when they exchange data. However, this strategy assumes that exact
execution times of actors are known. Such an assumption is generally not practical. A more realistic
assumption for DSP algorithms is that good estimates for the execution times of actors can be obtained.
Under such an assumption on timing, it is best to discard the exact timing information from the
fully static schedule, but still retain the processor assignment and actor ordering specified by the fully
static schedule. This results in the self-timed scheduling strategy of [17]. Each processor executes the
actors assigned to it in the order specified at compile time. Before firing an actor, a processor waits for the
data needed by that actor to become available. Thus in self-timed scheduling, processors are required to
perform run-time synchronization when they communicate data. Such synchronization is not necessary in
the fully-static case because exact (or guaranteed worst case) times could be used to determine firing times
of actors such that processor synchronization is ensured. As a result, the self-timed strategy incurs greater
run-time cost than the fully-static case because of the synchronization overhead.
A straightforward implementation of a self-timed schedule would require that for each inter-processor communication (IPC) the sending processor ascertains that the buffer it is writing to is not full, and
the receiver ascertains that the buffer it is reading from is not empty. The processors suspend execution
4
```

## 4. Sriram (1995): the ordered transactions strategy

- **Source:** as design/completion-and-period-sources-2026-11-07.md §5a (S. Sriram, PhD thesis, UC Berkeley, ERL memorandum M95/90, `lit3/sriram-thesis/ERL-95-90.pdf`), its OCR text `lit3/sriram-thesis/thesis.txt`, the abstract; the OCR keeps its misreadings ("acompile-time", "rniriimizing").

`thesis.txt` L70–84, abstract:

```text

In such acompile-time (static) scheduling strategy, it is possible to predict
in

the run time inter-processor communication (IPC) pattern. We present two tech

niques that make use of this compile-time determined communication pattern, for
rniriimizing IPC and synchronization overhead in the parallel implementation. The
first technique is aimed at eliminating arbitration and synchronization costs when

using shared memory for IPC. We call this the Ordered Transactions strategy; the
idea is to determine the order in which processors require access to shared
resources and to enforce this order at run time. Enforcing such an order eliminates
contention for shared resources and the need for explicit synchronization. We

```
