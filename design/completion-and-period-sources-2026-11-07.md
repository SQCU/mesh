Sources for the revisions of design/torch-capture-change-request.md, retrieved 2026-11-06/07 by a retrieval agent and recorded here as it wrote them. The local paths it names are the session's download directories.

# Completion and period: sources, quoted

Retrieved 2026-11-06/07. Every download is in its own directory under
`/private/tmp/claude-501/-Users-mdot-metal-microbench/66e06aa7-8a9f-469f-a879-1939b883b550/scratchpad/lit3/`.

Conventions. Quotations are verbatim. `[…]` marks an omission, `[sic]` marks an error in the source, and `[p. N]` marks a page break inside a quotation. Line-break hyphens are removed. Display formulas are transcribed from the page images: `_` marks a subscript and `{ … / … }` marks a stacked fraction. Page numbers are the printed numbers. The PDF page index is given where it differs. For scanned sources (3, 5-thesis, 6, 7, 8), each quotation was checked against the rendered page image, not only the OCR layer.

---

## 1. MPI 4.1: MPI_REQUEST_FREE, persistent requests, freeing an active request

**Bibliographic:** Message Passing Interface Forum, *MPI: A Message-Passing Interface Standard, Version 4.1*, November 2, 2023. https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf

**Read:** yes. Full PDF at `lit3/mpi41/mpi41-report.pdf`. Printed page = PDF page − 42.

**§3.7.3 Communication Completion, MPI_REQUEST_FREE, p. 81 (PDF 123):**
> "MPI_REQUEST_FREE is a local procedure. Upon successful return, MPI_REQUEST_FREE sets request to MPI_REQUEST_NULL. For an inactive request representing any type of MPI operation, MPI_REQUEST_FREE shall do the freeing stage of the associated operation during its execution."

**§3.7.3, freeing an active request, p. 81:**
> "For a request representing a nonblocking point-to-point or a persistent point-to-point operation, it is permitted (although strongly discouraged) to call MPI_REQUEST_FREE when the request is active. In this special case, MPI_REQUEST_FREE will only mark the request for freeing and MPI will actually do the freeing stage of the operation associated with the request later."

> "Calling MPI_REQUEST_FREE with an active request representing any other type of MPI operation (e.g., any partitioned operation (see Chapter 4), any collective operation (see Chapter 6), any I/O operation (see Chapter 14), or any request-based RMA operation (see Chapter 12)) is erroneous."

**§3.7.3, Advice to users, p. 82 (PDF 124):**
> "Once a request is freed by a call to MPI_REQUEST_FREE, it is not possible to check for the successful completion of the associated communication with calls to MPI_WAIT or MPI_TEST. Also, if an error occurs subsequently during the communication, an error code cannot be returned to the user—such an error must be treated as fatal."

**§3.7.3, MPI_WAIT on a persistent request, p. 79 (PDF 121):**
> "If the request is an active persistent communication request, it is marked inactive. Any other type of request is deallocated and the request handle is set to MPI_REQUEST_NULL."

> "One is allowed to call MPI_WAIT with a null or inactive request argument. In this case the procedure returns immediately with empty status."

**§3.9 Persistent Communication Requests, p. 110 (PDF 152):**
> "The request becomes inactive after successful completion of such call. The request is not deallocated and it can be activated anew by an MPI_START or MPI_STARTALL call."

> "A persistent communication request is deallocated by a call to MPI_REQUEST_FREE (Section 3.7.3). The call to MPI_REQUEST_FREE can occur at any point in the program after the persistent request was created. However, the request will be deallocated only after it becomes inactive. […] Collective operation requests […] must not be freed while active."

> "It is preferable, in general, to free requests when they are inactive. If this rule is followed, then the functions described in this section will be invoked in a sequence of the form,
> **Create (Start Complete)∗ Free**
> where ∗ indicates zero or more repetitions."

> "Inactive persistent requests are not automatically freed when the associated communicator is disconnected (via MPI_COMM_DISCONNECT, see 11.10.4) or the associated World Model or Sessions Model is finalized (via MPI_FNIALIZE [sic], see 11.2.2, or MPI_SESSION_FINALIZE, see 11.3.1). In these situations, any further use of the request handle is erroneous."

**§2.4.1 MPI Operations, p. 12 (PDF 54):** additional passages on operation states and pending operations.
> "These four stages lead to the operation states initialized, started, complete, and freed. A started operation is also named active, and the states initialized and complete are also named inactive."

> "Active communication and I/O operations are also named pending operations. Note that a pending operation can be a nonblocking or persistent operation that is started and not yet complete (even if the request handle has been freed), […]"

**§3.8.4 Cancel, pp. 102–103 (PDF 144–145):** additional passage on ending an active persistent operation.
> "If a communication is marked for cancellation, then a MPI_WAIT call for that communication is guaranteed to return, irrespective of the activities of other MPI processes (i.e., MPI_WAIT behaves as a local function); […]" (p. 102)

> "Cancelling a persistent send request by calling MPI_CANCEL is deprecated. A successful cancellation cancels the active communication, but not the request itself. After the call to MPI_CANCEL and the subsequent call to MPI_WAIT or MPI_TEST, the request becomes inactive and can be activated for a new communication." (p. 103)

**§2.8 Error Handling, pp. 26–27 (PDF 68–69):** the base standard on failures, included for (b).
> "Similarly, MPI itself provides no mechanisms for handling MPI process failures, that is, when an MPI process unexpectedly and permanently stops communicating (e.g., a software or hardware crash results in an MPI process terminating unexpectedly)." (p. 26)

> "This document does not specify the state of a computation after an erroneous MPI call has occurred." (p. 27)

---

## 2. ULFM: MPI_COMM_REVOKE and completion of pending and future operations after a failure

### 2a. MPI Forum fault tolerance working group, ULFM chapter draft, May 9, 2022

**Bibliographic:** Message Passing Interface Forum, "DRAFT Document for a Standard Message-Passing Interface", May 9, 2022. A LaTeX run of the single chapter "Chapter 16 Process Fault Tolerance", marked "Unofficial Draft for Comment Only". It is listed as "Current (full) draft proposal (May 2022)" in the body of mpi-forum/mpi-issues#20, "User-Level Failure Mitigation": https://github.com/mpi-forum/mpi-issues/files/8655090/20220509-ulfm-master.pdf

**Read:** yes, `lit3/ulfm-spec2022/20220509-ulfm-master.pdf`. Printed page = PDF page − 2.
- The issue states: "the versions attached to the ticket are updated loosely and are not always current. Please access the repository to see the latest revision."
- That repository, `mpiwg-ft/mpi-standard`, is not publicly accessible: the GitHub API returns 404. **Not read.**

**§16.1 Introduction, p. 1:**
> "The expected behavior of MPI in the case of an MPI process failure is defined by the following statements: any MPI operation that involves a failed process must not block indefinitely but either succeed or raise an MPI error (see Section 16.2); an MPI operation that does not involve a failed process will complete normally, unless interrupted by the user through provided functionality."

**§16.2 Failure Notification, p. 2:**
> "An operation involving a failed MPI process must always complete in a finite amount of time (possibly by raising one of the process failure error classes listed in Section 16.4). Operations do not raise fault tolerance errors during their initialization, initiation, or starting stages. The corresponding completion stage raises a fault tolerance error when appropriate."

> "Advice to users. Persistent collective communication initiation procedures are themselves collective operations, and may therefore raise a process failure error of their own. (End of advice to users.)"

**§16.2.1 Error Reporting Range, value of info key "mpi_error_range", p. 3:**
> “"group" The failure of any MPI process in the group of the communicator causes the communicator to become revoked. When a communicator is revoked, communication operations on the communicator raise a fault tolerance error of class MPI_ERR_REVOKED, even in operations that involve only non-failed processes.”

**§16.2.2 Fault Tolerance Errors in Point-to-Point Communication, p. 3:**
> "In all other cases, the operation raises an error of class MPI_ERR_PROC_FAILED to indicate that the failure prevents the operation from following its failure-free specification. If there is a request identifying a point-to-point communication, it is completed. Communication involving the failed MPI process, started on this communicator after the error raised, must also raise an error of class MPI_ERR_PROC_FAILED."

**§16.3.1 Communicator Functions, MPI_COMM_REVOKE(comm), pp. 8–9 (semantics on p. 9):**
> "The revocation of a communicator by any MPI process completes non-local MPI operations on comm at all MPI processes by raising an error of class MPI_ERR_REVOKED (with the exception of MPI_COMM_SHRINK, MPI_COMM_AGREE, and MPI_COMM_IAGREE). This function is not collective and therefore does not have a matching call on remote MPI processes."

> "Once a communicator has been revoked at an MPI process, all subsequent non-local operations on that communicator (with the same exceptions as above), are considered local and must complete by raising an error of class MPI_ERR_REVOKED at that MPI process."

**§16.4, Table 16.1 "Fault tolerance error classes", p. 18:**
> "MPI_ERR_PROC_FAILED_PENDING — The operation was interrupted by an MPI process failure (a fail-stop failure). The request is still pending and the operation may be completed later."

### 2b. Same working group, earlier draft, February 21, 2017

**Bibliographic:** Message Passing Interface Forum, "DRAFT Document for a Standard Message-Passing Interface", February 21, 2017, "Chapter 15 Process Fault Tolerance". https://fault-tolerance.org/wp-content/uploads/2012/10/20170221-ft.pdf. Open MPI's ULFM documentation (docs.open-mpi.org v5.0.10, `features/ulfm.rst`) links this file in the sentence: "This implementation produces the three supplementary error codes and five supplementary interfaces defined in the communicator section of the ULFM chapter standard draft document."

**Read:** yes, `lit3/ulfm-spec2017/20170221-ft.pdf`.

**§15.3.1, p. 8:** the MPI_COMM_REVOKE paragraph is word-for-word identical to the 2022 §16.3.1 paragraph quoted in 2a. This was checked by comparing the extracted text (1034 characters, equal).

**§15.2, p. 2:**
> "An operation involving a failed MPI process must always complete in a finite amount of time (possibly by raising one of the process failure error classes listed in Section 15.4). If an operation does not involve a failed MPI process (such as a point-to-point message between two non-failed MPI processes), it must not raise a process failure error."

### 2c. Bland, Bouteiller, Herault, Bosilca, Dongarra (2013)

**Bibliographic:** W. Bland, A. Bouteiller, T. Herault, G. Bosilca, J. Dongarra, "Post-failure recovery of MPI communication capability: Design and rationale", *International Journal of High Performance Computing Applications* 27(3):244–254, 2013. The first page reads "© The Author(s) 2012". DOI 10.1177/1094342013488238.

**Read:** yes. The typeset article PDF is hosted on T. Herault's ICL page, https://icl.utk.edu/users/herault/papers/ ("018 - Post-failure recovery of MPI communication capability - Design and rationale - IJHPCA (2013).pdf"). Local copy: `lit3/ulfm-bland2013/bland2013.pdf`.

**§3 Design goals, p. 247:**
> "An error must be raised when a failure prevents a communication from completing. However, we propose that such a notice indicates only the local status of the operation, and does not permit inferring whether the associated failure has impacted MPI operations at other ranks."

**§4.1 Failure reporting, p. 247:**
> "However, unlike any other collective MPI constructs it does not require a symmetric call on all processes, a single processes [sic] in the communicator calling the revoke operation ensure the communicator will be eventually revoked. In other words it has a behavior similar to MPI_ABORT with the exception that it does not abort processes, instead it terminate [sic] all ongoing operations on the communicator and mark the communicator as improper for future communications."

**§4.1, pp. 247–248:**
> "When receiving this notification, any communication on the communicator (ongoing or future) is interrupted and a special error code returned. From this point, any [p. 248] operation (point-to-point or collective) on the communicator returns that same error code, and the communicator becomes effectively unusable for any purpose."

**§5.2 Communication objects status, p. 250:**
> "When a failure has happened, but the MPI operation can proceed without disruption, it completes normally. Obviously, when the failed process is supposed to participate to the result of the operation, it is impossible for the operation to succeed, and an appropriate error is returned. Posting more operations that involve the dead processes is allowed, but is expected to result in similar errors."

**§5.4 Restoring consistency and communication capabilities, p. 250:**
> "After a communication object has been revoked, it can never be repaired."

### 2d. Retrieval note

*MPI: A Message-Passing Interface Standard, Version 5.0* (June 5, 2025; https://www.mpi-forum.org/docs/mpi-5.0/mpi50-report.pdf; `lit3/mpi50/`) was searched. Its text contains no occurrence of "MPI_COMM_REVOKE", "MPI_ERR_REVOKED", or "MPI_ERR_PROC_FAILED".

---

## 3. Reed and Kanodia (1979): advance(E), read(E), await(E, v)

**Bibliographic:** David P. Reed and Rajendra K. Kanodia, "Synchronization with Eventcounts and Sequencers", *Communications of the ACM* 22(2):115–123, February 1979. DOI 10.1145/359060.359076.

**Read:** yes. This is ACM's scanned PDF. dl.acm.org returned a Cloudflare 403 to curl, so the copy was taken from the Internet Archive snapshot of `https://dl.acm.org/doi/pdf/10.1145/359060.359076` (2025-06-09). Local copy: `lit3/reed-kanodia/rk1979.pdf`.

**"Eventcounts", p. 116:**
> "An eventcount can be thought of as a nondecreasing integer variable."

> "The primitive advance(E) is used to signal the occurrence of an event in the class associated with the eventcount E. The effect of this operation is to increase the integer "value" of E by 1. The "value" of an eventcount equals the number of advance operations that have been performed on it (i.e. the initial value of an eventcount is zero)."

> "To be more precise, the value returned by read(E) counts all of the advance operations that precede the execution, and may or may not count those in progress during the read. Thus, although the value may be changing during the execution of the read primitive, the result of read(E) is a lower bound on the current value of E after the read, and an upper bound on the value of E before the read."

> "Consequently, we define a primitive await(E, v) that suspends the calling process until the "value" of the eventcount E is at least v.¹ Of course, as in all mechanisms designed for asynchronous processes, the await primitive may not return immediately once the vth advance on E is executed; the only guarantee is that *at least* v advances have been performed by the time await(E, v) returns."

**"Timing Relationships", p. 117:** the axiom for await, and footnote 4. Here ⇒ is defined on p. 117 as: "A ⇒ B (read "A precedes or is concurrent with B") iff not B → A".
> "D5. If W_E is the execution of an await operation of the form await(E, t), then there are at least t members of the set {A_E | A_E is the execution of advance(E) and A_E ⇒ W_E}.⁴"

> "⁴ In this paper, we ignore questions of liveness, e.g. we have no axiom that says when an await will eventually complete."

**"Single Producer, Single Consumer Example", p. 117.** The eventcounts IN and OUT are "(initially zero)". The loop in procedure producer( ):
```
for i := 1 to infinity do
begin
  await(OUT, i − N);
  buffer[i mod N] := produce( );
  advance(IN);
end
```
The loop in procedure consumer( ):
```
for i := 1 to infinity do
begin
  await(IN, i);
  consume(buffer[i mod N]);
  advance(OUT);
end
```

**"Implementation of Eventcounts and Sequencers", p. 121:**
> "If eventcounts are used to synchronize cyclic processes that never terminate, one may be concerned that the values of eventcounts cannot be stored in a finite amount of memory. In practice, however, one can always bound the values that will be encountered, since no real system will operate forever."

---

## 4. Bhattacharyya, Sriram and Lee (1997): iteration period as maximum cycle mean; self-timed buffer bound

**Bibliographic:** Shuvra S. Bhattacharyya, Sundararajan Sriram, Edward A. Lee, "Optimizing Synchronization in Multiprocessor DSP Systems", *IEEE Transactions on Signal Processing* 45(6):1605–1618, June 1997. DOI 10.1109/78.600002.

**Read:**
- Read: the authors' preprint, headed "— IEEE Transactions on Signal Processing, Vol. 45, No. 6, June 1997 —" and dated "February 11, 1997". Source: the Ptolemy project, https://ptolemy.berkeley.edu/papers/97/synchronization/synch.pdf. Local copy: `lit3/bsl1997/synch.pdf`. Page numbers below are the preprint's own; formulas were checked on the page images.
- **Not read:** the typeset IEEE version, so journal page numbers for these passages are not given.

**§1, p. 5:**
> "As a performance metric for evaluating DFG implementations we use the average iteration period T, (or equivalently the throughput T⁻¹) which is the average time that it takes for all the actors in the graph to be executed once."

**§3 Terminology, p. 7:**
> "Given a DFG G, and vertices x, y, we define ρ_G(x, y) to be ∞ if there is no path from x to y, and equal to the path delay of a minimum-delay path from x to y if there exists a path from x to y."

**§4.1 Inter-processor Communication Modelling Graph, p. 9:** the edges that model each processor's execution order. Earlier in the paragraph: "processor 1 executes A and then E repeatedly."
> "The delay-free edge from A to E represents the fact that the k th execution of A precedes the k th execution of E, and the edge from E to A with a delay represents the fact that the k th execution of A can occur only after the (k – 1) th execution of E has completed. Thus if actors v_1, v_2, …, v_n are assigned to the same processor in that order, then G_ipc would have a cycle ((v_1, v_2), (v_2, v_3), …, (v_{n – 1}, v_n), (v_n, v_1)), with delay((v_n, v_1)) = 1. If there are P processors in the schedule, then we have P such cycles corresponding to each processor."

**§4.1 IPC graph, p. 10:**
> "An IPC edge represents a buffer implemented in shared memory, and initial tokens on the IPC edge are used to initialize the shared buffer."

> "start(v_i, k) ≥ end(v_j, k – delay((v_j, v_i))), ∀(v_j, v_i) ∈ E_ipc, ∀k > delay(v_j, v_i). (1)"

**p. 11:**
> "Lemma 2: [3] The number of tokens in any cycle of the IPC graph is always conserved over all possible valid firings of actors in the graph, and is equal to the path delay of that cycle."

> "Lemma 3: The asymptotic iteration period for a strongly connected IPC graph G when actors execute as soon as data is available at all inputs is given by [24]:
> T = max_{cycle C in G} { Σ_{v is on C} t(v) / Delay(C) }. (4)"

**p. 12:**
> "In such a case, the iteration period of the overall graph (and hence the self-timed schedule) is the maximum over the maximum cycle means of all the SCCs of G_ipc, because the execution of the schedule is constrained by the slowest component in the system."

> "Definition 2: The maximum cycle mean of an IPC graph G_ipc, denoted by λ_max, is the maximal cycle mean over all SCCs of G_ipc: That is, […]"

> "Thus the throughput of the system of processors executing a particular self-timed schedule is equal to the corresponding 1/λ_max value."

**§4.3 Strongly Connected Components and Buffer Size Bounds, p. 13:**
> "However, from Lemma 2, the number of tokens on each feedback edge (an edge that belongs to an SCC, and hence to some cycle) during the execution of the IPC graph is bounded above by a constant. We will call this constant the self-timed buffer bound of that edge, and for a feedback edge e we will represent this bound by B_fb(e). Lemma 2 yields the following self-timed buffer bound:
> B_fb(e) = min({Delay(C) | C is a cycle that contains e}) (5)"

**§4.3, feedforward edges, pp. 13–14:**
> "Since the effect of placing such a restriction includes "artificially" constraining src(e) from getting more than B_ff(e) invocations ahead of snk(e), its effect on the estimated [p. 14] throughput can be modelled by adding the reverse edge d_m(snk(e), src(e)), where m = B_ff(e) – delay(e), to G_ipc (grey edge in Fig. 2(b))."

---

## 5. Sriram and Bhattacharyya, *Embedded Multiprocessors: Scheduling and Synchronization*

**Bibliographic:**
- 1st edition: Sundararajan Sriram and Shuvra S. Bhattacharyya, *Embedded Multiprocessors: Scheduling and Synchronization*, Marcel Dekker, 2000. ISBN 978-0-8247-9318-0.
- 2nd edition: CRC Press, 2009. ISBN 978-1-4200-4801-8, DOI 10.1201/9781420048025.

**The book: not read.** The book's own statements of the maximum cycle mean and of the IPC-edge buffer bound were not reached, for these reasons:
- Google Books lists 1st-edition id `m75xsY0a22UC` as "Preview this book". Its search-inside requests returned HTTP 429, and the Books API quota was exhausted.
- Taylor & Francis chapter PDFs are paywalled.
- The Internet Archive holds no copy.

**What was readable of the book:** the publisher's "first-paragraph" excerpt of 2nd-ed. chapter "Multiprocessor Scheduling Models" (pp. 107–126), from https://www.taylorfrancis.com/chapters/mono/10.1201/9781420048025-10/multiprocessor-scheduling-models-sundararajan-sriram-shuvra-bhattacharyya. The page's text omits the inline symbols; the omissions are marked:
> "The performance metric of interest for evaluating schedules is the average iteration period [symbol omitted on the page]: the average time it takes for all the actors in the graph to be executed once."

### 5a. Authors' own text: Sriram's PhD thesis, which covers the same IPC-graph material

**Bibliographic:** Sundararajan Sriram, *Minimizing Communication and Synchronization Overhead in Multiprocessors for Digital Signal Processing*, Ph.D. thesis, University of California, Berkeley. Memorandum UCB/ERL M95/90, November 7, 1995. https://www2.eecs.berkeley.edu/Pubs/TechRpts/1995/ERL-95-90.pdf

**Read:** yes. Scanned copy at `lit3/sriram-thesis/ERL-95-90.pdf`. Printed page = PDF page − 12.

**§4.1 Inter-processor Communication graph (G_ipc), p. 83 (PDF 95):** the edges that model each processor's execution order. Earlier in the paragraph: "processor 1 executes A and then E repeatedly."
> "The delay-free edge from A to E represents the fact that the k th execution of A precedes the k th execution of E, and the edge from E to A with a delay represents the fact that the k th execution of A can occur only after the (k – 1) th execution of E has completed. Thus if actors v_1, v_2, …, v_n are assigned to the same processor in that order, then G_ipc would have a cycle ((v_1, v_2), (v_2, v_3), …, (v_{n – 1}, v_n), (v_n, v_1)), with delay((v_n, v_1)) = 1 (because v_1 is executed first)."

**p. 85 (PDF 97):**
> "Lemma 4.1: The number of tokens in any cycle of the IPC graph is always conserved over all possible valid firings of actors in the graph, and is equal to the path delay of that cycle."

**p. 87 (PDF 99):**
> "Lemma 4.3: The asymptotic iteration period for a *strongly connected* IPC graph G when actors execute as soon as data is available at all inputs is given by:
> T = max_{cycle C in G} { Σ_{v is on C} t(v) / Delay(C) } (4-3)"

**§5.2.1 Estimated throughput, p. 114 (PDF 126):**
> "Recall from Eqn. 4-3 that the average iteration period corresponding to a self-timed schedule with an IPC graph G_ipc is given by the maximum cycle mean of the graph MCM(G_ipc)."

**§5.3, pp. 114–115:**
> "However, from Lemma 4.1, every **feedback edge** (an edge that belongs to a strongly connected component, and hence to some cycle) can only have a finite number of tokens at any time during the execution of the IPC graph. We will call this constant the **self-timed buffer bound** of that edge, […]
> [p. 115] B_fb(e) = min({Delay(C) | C is a cycle that contains e}) (5-1)"

**§5.4, p. 122 (PDF 134):**
> "Then, the iteration period of the resulting system is determined by the maximum cycle mean of G_s (MCM(G_s)). This is because the synchronization edges alone determine the interaction between processors; a communication edge without synchronization does not constrain the execution of the corresponding processors in any way."

**§5.8 Computing buffer bounds from G_s and G_ipc, p. 141 (PDF 153):**
> "Theorem 5.4: If G_s preserves G_ipc and the synchronization edges in G_s are implemented, then for each feedback communication edge e in G_ipc, the self-timed buffer bound of e (B_fb(e)) — an upper bound on the number of data tokens that can be present on e — is given by:
> B_fb(e) = ρ_{G_s}(snk(e), src(e)) + delay(e),"

**Proof of Theorem 5.4, p. 142 (PDF 154):**
> "Taking p to be an arbitrary minimum-delay path from snk(e) to src(e) in G_s, we get start(src(e), k) ≥ end(snk(e), k − ρ_{G_s}(snk(e), src(e))). That is, src(e) cannot be more that [sic] ρ_{G_s}(snk(e), src(e)) iterations "ahead" of snk(e)."

### 5b. Authors' own text: Bhattacharyya, Sriram and Lee, resynchronization memorandum (1996)

**Bibliographic:** S. S. Bhattacharyya, S. Sriram, E. A. Lee, "Resynchronization of Multiprocessor Schedules: Part 1 — Fundamental Concepts and Unbounded-Latency Analysis", Memorandum UCB/ERL M96/55, Electronics Research Laboratory, U. C. Berkeley, October 1996. https://ptolemy.berkeley.edu/publications/papers/96/resync1/resynch1.erl.pdf

**Read:** yes, `lit3/bsl-resynch/resynch1.erl.pdf`.

**§4.2, p. 8:**
> "Such a simplified protocol is possible because each feedback edge has a buffer requirement that is bounded by a constant, called the self-timed buffer bound of the edge, which can be computed efficiently from the synchronization graph topology [5]."

**§4.3 Estimated throughput, pp. 8–9:**
> "If the execution time of each actor v is a fixed constant t∗(v) for all invocations of v, and the time required for IPC is ignored (assumed to be zero), then as a consequence of Reiter's analysis in [32], the throughput (number of DFG iterations per unit time) of a synchronization graph G is given by 1/(λ_max(G)), where
> λ_max(G) ≡ max_{cycle C in G} { Σ_{v∈C} t∗(v) / Delay(C) }. (3)"

---

## 6. Reiter (1968): maximum cycle ratio and the bound on the computation rate

**Bibliographic:** Raymond Reiter, "Scheduling Parallel Computations", *Journal of the ACM* 15(4):590–599, October 1968. DOI 10.1145/321479.321485.

**Read:** yes. This is ACM's scanned PDF, taken from the Internet Archive snapshot of `https://dl.acm.org/doi/pdf/10.1145/321479.321485?download=true` (2020-06-04). Local copy: `lit3/reiter1968/reiter1968.pdf`.

**Abstract, p. 590:**
> "The computation rate of a given admissible schedule is defined and is shown to have a limiting value 1/π where π is a parameter dependent upon the cycles in the graph. Thus, the computation cannot proceed at a rate exceeding 1/π."

**§1, p. 591:** branch parameters.
> "A_ij = the initial number of data words on branch d_p;"

> "τ_ij = the time required by node n_i to place the result of its operation on branch (n_i, n_j)."

**§3.A Periodic admissible schedules, pp. 594–595:**
> "Let us define the *maximum cycle ratio* of G,
> π = max { Σ_{(n_i,n_j)∈C} τ_ij / Σ_{(n_i,n_j)∈C} A_ij },
> [p. 595] the max over all cycles C of G (including cycles of the form (n_i, n_i)) consisting of essential branches."

> "π thus emerges as the minimum possible period of any periodic admissible schedule." (p. 595)

**p. 596:**
> "*Thus, asymptotically, n_1 cannot achieve a computation rate greater than that attained under a periodic admissible schedule σ′ with period π. Moreover, this maximum computation rate is 1/π.*"

---

## 7. Lam (1988): prolog, steady state, epilog

**Bibliographic:** Monica Lam, "Software Pipelining: An Effective Scheduling Technique for VLIW Machines", in *Proceedings of the SIGPLAN '88 Conference on Programming Language Design and Implementation*, Atlanta, Georgia, June 22–24, 1988, pp. 318–328. DOI 10.1145/53990.54022.

**Read:** yes. This is ACM's scanned PDF, taken from the Internet Archive snapshot of `https://dl.acm.org/doi/pdf/10.1145/53990.54022` (2023-10-26). Local copy: `lit3/lam1988/lam1988.pdf`.

**Abstract and §1, p. 318:**
> "In software pipelining, iterations of a loop in the source program are continuously initiated at constant intervals, before the preceding iterations complete."

> "The steady state of this pipeline constitutes the loop body of the object code."

**§2 Simple loops, p. 320:**
> "Instructions 1 to 3 are called the *prolog*: a new iteration is initiated every instruction cycle and execute concurrently with all previously initiated iterations. The *steady state* is reached in cycle 4, and this state is repeated until all iterations have been initiated."

> "On leaving the steady state, the iterations currently in progress are completed in the *epilog*, instructions 5 through 7."

**§2.1 Problem statement, p. 320:**
> "Software pipelining is unique in that pipeline stages in the functional units of the data path are not emptied at iteration boundaries; the pipelines are filled and drained only on entering and exiting the loop."

**§2.3 Modulo variable expansion, p. 322:**
> "The lifetime of a register variable is defined as the duration between the first assignment into the variable and its last use. If the lifetime of a variable is l, and an iteration is initiated every s cycles, then at least ⌈l/s⌉ number of values must be kept alive concurrently, in that many locations."

**§2.4 Code size, p. 323:**
> "If the number of iterations is not known at compile time, then additional code must be generated to handle the cases when there are so few iterations in the loop that the steady state is never reached, and when there are no more iterations to initiate in the middle of an unrolled steady state."

> "Let k be the number of iterations started in the prolog of the pipelined loop, u be the degree of unrolling, and n be the number of iterations to be executed. Before the loop is executed, the values of n and k are compared. If n < k, then all n iterations are executed using the unpipelined code."

---

## 8. Lee and Messerschmitt (1987): delays (initial tokens) on arcs

**Bibliographic:** Edward Ashford Lee and David G. Messerschmitt, "Static Scheduling of Synchronous Data Flow Programs for Digital Signal Processing", *IEEE Transactions on Computers* C-36(1):24–35, January 1987. DOI 10.1109/TC.1987.5009446.

**Read:** yes. This is a scan of the typeset IEEE pages from the Ptolemy project, https://ptolemy.berkeley.edu/publications/papers/87/staticscheduling/staticscheduling.pdf. Local copy: `lit3/lee-messerschmitt1987/staticscheduling.pdf`.

**§IV, p. 26:**
> "The SDF graph is nonterminating (cf. [29], [30]) meaning that it can run forever without deadlock."

> "The desired schedule is *admissible*, meaning that blocks will be scheduled to run only when data are available, and that a finite amount of memory is required."

**§IV.A, p. 27:**
> "We define a *unit delay* on an arc from node A to node B to mean that the nth sample consumed by B will be the (n − 1)th sample produced by A. This implies that the first sample the destination block consumes is not produced by the source block at all, but is part of the initial state of the arc buffer. Indeed, a delay of d samples on an arc is implemented in our model simply by setting an initial condition for (3)."

> "Delays, therefore, affect the way the system starts up."

**p. 30:** this sentence is about directed loops, i.e. feedback arcs.
> "Networks with insufficient delays in directed loops are not computable."

Fig. 7 caption: "Two SDF graphs with consistent sample rates but no admissible schedule."

**§IV.B Constructing a PAPS, p. 31:**
> "That is, if a schedule on a given processor dictates that a node should be invoked, but there is no input data for that node, then the processor halts until these input data are available."

> "Further, given the delays on two of the arcs, we note that there are several possible minimum period PASS's, […]"

**p. 32:** precedence in the multiprocessor precedence graph. Here the arc is named *a*, from node η to node α.
> "Of these samples, b_a are provided as initial conditions. If b_a ≥ −jΓ_aα then there is no dependence of the jth run of α on η."

**§V.C Connections to the Outside World, p. 33:**
> "Arcs into a SDF graph from the outside world, like those shown in Figs. 1 and 2, are ignored by the scheduler."

> "As it stands now, the model cannot reflect this, so buffering of input data is likely to be required."

**Retrieval note:** no passage in this paper relates delays on feedback arcs to pipelining or to buffer sizes. Both the full OCR text and the page images were searched for "pipelin", "feedback", "loop", "delay", "buffer" and "initial". The occurrences of "pipelin" are in the introduction, about pipelined machines and chips, and in the reference list.
