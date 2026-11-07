# Grounding report: the recorder, the masked passes, the waiter, the views and the lifetimes, against the literature

Code read: metal-microbench 0771d07a (`metal_recording.{h,m}`, `tools/e2b_replay.m`), mesh 165f2c98 (`rdma/mesh-metal.m`,
`rdma/mesh-dataflow.c`, `rdma/mesh-call.h`, `rdma/mesh-link.c`, `rdma/mesh-flow.c`, `rdma/mesh.h`,
`rdma/mesh-collective.{h,c}`, `rdma/mesh-graph.{h,c}`, `rdma/mesh_compile.py`), mesh fcf5d18a (its diff), and
metal-microbench d0a72fc6 (its diff), all with `git show`. No tracked file in either repository was changed. (A helper's browser tool wrote two files into
the git-ignored `metal-microbench/.playwright-mcp/`, which it deleted at once, and appended to an existing ignored log
there.)

**What counts as a source.** Papers, books, standards, vendor API text (the macOS 26.4 SDK headers on this machine, the
Metal Shading Language standard library headers in `/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023.864/include/metal/`,
vendor documentation pages), and open-source reference programs with their licenses. Not sources: any `.md` file of
either repository (read only to find code and, for section A, the works the change requests cite), torch in any form
(including ExecuTorch, which the change requests cite; flagged where it matters), and `docs/facts.md` (not opened).

**How quotes were taken.** Every downloaded file was in its own directory of a session's temporary download folder,
named below as `lit/<dir>/`; those files are not kept in this repository. Quotes marked as read were checked against the downloaded text (pdftotext, an HTML-to-text pass
run with `python3 -I`, or the PDF page images); scanned pages are quoted as printed. Sources that could not be read are
marked **not read (from memory)**, and nothing from them appears in quotation marks. Sources gathered by helper agents
are marked with the agent's directory prefix (`a-`, `b-`, `c-`, `d-`, `e-`); my own downloads are `f-`, `lam1988/` and
`mpi41-node83/`.

**Verdict vocabulary.** *Matches*: the code does what the source's algorithm or data structure does. *Deviates, required*:
the difference is forced by quoted hardware or API text. *Deviates, not required*: no quoted text forces it. *No
counterpart*: no published algorithm or data structure found that does this.

## Summary

| # | Mechanism (code) | Standard name | Decades of sources read | Verdict |
|---|---|---|---|---|
| 1 | `capture:`/`replay:`, `MetalBlitPreparation`, `MetalCreatePipeline`, `MetalReplayEncode` | Stored command list replayed by a separate processor (channel program, display file/list, bundle, CUDA graph, indirect command buffer), captured by API interposition (apitrace, GFXReconstruct); inspector/executor | 1960s, 1970s, 1980s, 1990s, 2000s, 2010s, 2020s | Matches record-once/replay-many by interposition. Deviates, required: commands split at every wait and later passes outside the indirect command buffer (ICBs hold draws and dispatches only). Deviates, not required: copies refused (MTL4 compute encoders encode copies; CUDA graphs and IREE record them; only D3D12 bundles drop them); fills executed once at capture (equivalent only for one-time lazy clears such as Dawn's; wrong for per-execution fills; nothing distinguishes the two). The supportIndirectCommandBuffers raise is required only for the indirect path; direct dispatch is allowed. |
| 2 | `place:…`, `MetalBinding`, `MetalReplayPeriod` | Modulo variable expansion (buffers); rotating register file; multiple buffering | 1960s, 1980s, 1990s, 2010s, 2020s | Matches MVE (copy t mod copies, `ahead` renaming). Deviates, not required: unroll degree is the lcm of all periods plus a 4u phase-row period, where Lam recommends max_i q_i with each q_i rounded to a factor of it; the 2^22-command segment limit has no header text. |
| 3 | `MetalMaskReaders`, `MetalMaskedPass`, `MetalCompactionSource`, later pass | Presence/validity array apart from the data with a never-reset Gray-coded phase (phase tag, wrap counter, mbarrier parity, sense reversal); predication + stream compaction + indirect dispatch (topology-driven then data-driven); coarse queue wait | 1950s, 1960s, 1970s, 1980s, 1990s, 2000s, 2010s, 2020s | Components match. Whole-operand test (all P pages per threadgroup) deviates, not required — per-element sources test only what is read. Threadgroup granularity required (MSL barrier rule). Queue wait required once in-kernel waiting is excluded (no MSL wait primitive). Source rewriting has no counterpart. No single published counterpart for the composition. |
| 4 | mesh `waits_run`, `mesh_metal_waits_start` | Eventcount await/advance on a monotone completion counter; counting events with thresholds; a host progress thread bridging memory counters to GPU events (where CUDA has stream wait-on-value) | 1960s, 1970s, 1990s, 2000s, 2010s, 2020s | Matches eventcount await / counting-event threshold (Portals, OpenSHMEM, counting notifications) / proxy-thread designs (NCCL GIN proxy, NVSHMEM). The CPU agent is required by Metal (no queue wait on a memory word; CUDA has cuStreamWaitValue); its placement in the consuming process is not (a shared event can be recreated in another process). The failure path (final value, not a failure field or retry) deviates, not required; a Metal-side wait timeout ends the same way because the later pass re-tests. |
| 5 | `MetalRelease` (the `published:` hooks) | Release-then-flag publication (message-passing pattern); single-writer signal word; counting signal (signal-add); fail-stop | 1960s, 1970s, 1980s, 1990s, 2000s, 2010s, 2020s | Matches release-then-flag (C11 §7.17.4, PTX §8.8, NCCL) and fail-stop. seq_cst for release is required by the installed MSL headers only (MSL 4.1 adds release). The 32-bit add on a 64-bit word read as 64 bits is a mixed-size access outside every published model (Flur et al.; PTX §8.7.2), not required; the fence scope value 3 / "system" coherence has no public Metal text. |
| 6 | `MetalRecorderRun` (`lead`), `mesh_transfers_start`/`mesh_host_started`, `HOST_LAG` | Persistent requests started ahead (MPI_START), a synchronization edge with delay in a self-timed schedule; double buffering | 1960s, 1980s, 1990s, 2010s, 2020s | Mechanism matches (MPI_START; BSL delay edge). The constants lead = 1/0 and HOST_LAG = 2 are not derived by any published sizing rule (BSL's maximum cycle mean, Lam's ⌈l/s⌉, Rau's Kmin, Mowry's prefetch distance, Kung's credits); Apple's own guide says two buffers can stall and calls three ideal; no API requires either constant. The start runs in an event-listener callback, which the operator's turns call forbidden (context). |
| 7 | mesh `mesh_view_compose`/`mesh_view_values`, NoCopy wrap | Artificial contiguity by reserve-and-remap (user-space page-table composition), synonyms in one address space | 1960s, 1970s, 1980s, 1990s, 2000s, 2010s, 2020s | Matches (Randell–Kuehner/Denning; XNU's own pshm_mmap sequence; CUDA VMM, GMLake, vAttention). Placement-sparse exclusion required by the headers (heap memory is Metal's). Apple's bytesNoCopy precondition "within a single virtual memory (VM) region" is not met by the text for a multi-remap view: unresolved. Metal 3 has no alias flush in its headers. |
| 8 | `program()` lifetimes; mesh-graph `mesh_graph_fat_cover`, `mesh_graph_chains` | Liveness-based storage sharing (interference / interval-graph colouring); cyclic interval (circular-arc) colouring for modulo lifetimes; Dilworth chains | 1970s (Lamport), 1980s, 1990s, 2010s, 2020s; 1950s/1970s primaries (Dilworth, Hashimoto–Stevens, Tucker, Fabri) not read | Fat cover matches Mukerji's two-phase heuristic (chameleon intervals replaced by new colours, as §3.2.3's failure path allows); chains match Dilworth/Fulkerson. `program()`'s [0, levels − 1] for every captured EXECUTION value deviates, not required, from first/last-use liveness (safe, costs memory). |
| 9 | fcf5d18a | Mixed-mode execution (recorded/replayed where possible, interpreted elsewhere) | 1970s (not read), 1990s/2000s, 2010s, 2020s | No published counterpart for a per-node split between replay and per-operator dispatch in one distributed program; not required by Metal (a whole later pass needs no kernel source, d0a72fc6). |

## A. What the four sanctioned change requests already cite, mechanism by mechanism

Read at mesh 165f2c98 (unchanged at df3c634b): `design/disjoint-pages-change-request.md` (DP),
`design/plan-to-bridge-change-request.md` (PB), `design/aot-compilation-change-request.md` (AOT),
`design/problems-of-14b9a757-change-request.md` (P14). Their prose is not a source here; only the works they cite are,
and each work's text is quoted in sections 1–9 from the work itself where it could be read. The operator's turns these
documents quote verbatim are context (marked "operator, as quoted in …"), never a source.

| # | Method the documents name for it (where) | Code follows it? |
|---|---|---|
| 1 Capture/replay | No published method for the recorder's capture itself: AOT A1 says only "the recorder's capture of the delegate's commands (metal_recording.m), as now". For replay: CUDA Graphs, "a graph instantiated once and launched many times (NVIDIA, CUDA C++ Programming Guide)", and JAX's trace cache (Frostig, Johnson, Leary, SysML 2018), AOT appendix 6. The operator's "cudagraphs style 'shapes are compiled but indexes inside of shapes might not need to be'" (AOT A1). DP D.3.4: "No Metal 3 command buffers" (operator: "no metal 3 commandbuffers"). | Instantiate-once/launch-many: yes. CUDA Graphs' capture semantics: no — CUDA graphs hold memcpy and memset nodes, event-wait and external-semaphore-wait nodes (§1 below); the recorder refuses copies, runs fills once at capture, and splits command buffers at waits. Only the last is forced by Metal (an indirect command buffer holds draws and dispatches only). |
| 2 Rebinding t mod copies | Modulo variable expansion: DP C3.3, "The template is unrolled into u copies, and execution t runs copy t mod u" (Lam, PLDI 1988; Rau, MICRO 1994). P14 quotes Lam §2.3: "the minimum degree of unrolling, u, to implement the same schedule is simply max_i q_i." | Copy t mod copies: yes. The replay's period: the code takes the least common multiple (citing Lam's preceding sentence), not the max_i q_i form the documents quote, which Lam recommends because the lcm "can lead to an intolerable increase in code size" (§2 below). |
| 3 RAW on landed data | DP turns and C3.1/C3.7: read-after-write "by mask and a later pass"; predication (Allen, Kennedy, Porterfield, Warren, POPL 1983; ILLIAC IV 1968 mode bits; CRAY-1 1978 vector mask; Connection Machine 1985 context flags; NVIDIA Tesla 2008 SIMT active masks; AVX-512 opmasks; ARM SVE 2017; RISC-V V 1.0); masks and indices as dense arrays apart from the data (Kilburn 1962; Denning 1970; PagedAttention 2023; Apache Arrow validity bitmap; Saad 2003 CSR; Blelloch 1990 scan compaction); counted completion (Portals 4, SAND2012-10087; OpenSHMEM 1.5 put-with-signal; PTX mbarrier; Belli and Hoefler, IPDPS 2015); the phase field as "PTX mbarrier's phase bit, widened to two"; presence-per-element memory (I-structures, TOPLAS 1989; HEP full/empty bits; NCCL LL/LL128) named as the prior art of the rejected per-page word. | Follows C3.1/C3.7 as written: phase list apart from the pages, Gray-coded 2-bit phase, ballot/prefix-sum compaction, indirect later pass, queue wait. Against the cited works: the predicate is one per operand copy (all P pages), not per element read (I-structures, MPI partitions, tile signals); the "mask" is a threadgroup-uniform early return, not lane predication; the later pass is held by a wait (§3). |
| 4 Counters → events | C3.1: "The client alone signals the event: when the count reaches the step's target, or at the step's deadline. Its host thread waits on the count's address with a timeout (`os_sync_wait_on_address`, woken by the bridge's arrivals)"; counted completion: Portals 4 counting events, CUDA `cuStreamWaitValue32`, Metal `MTLSharedEvent`; P14 Terms: libfabric fi_cntr(3) "Counters record the number of requested operations that have completed". | Yes. The CPU thread stands where `cuStreamWaitValue` would put the GPU's front end; Metal's headers offer no queue wait on a memory word (§4), so some CPU agent is required. |
| 5 Sender's release | C3.7: "A supplied word has one writer, its range's producer, which stores t + 1 after its last write in execution t." PB R4: "Its store follows its pages in C3's writing order (Boehm and Adve, 2008)." OpenSHMEM `SHMEM_SIGNAL_ADD` named under counted completion. | In effect (one release per repetition raises the word to t + 1). Not named anywhere: the seq_cst fence standing for a release (MSL has no release order), the 32-bit add on the low half of a 64-bit word that the bridge reads with a 64-bit load, and the fence scope value 3 (§5). |
| 6 Start `lead` ahead; HOST_LAG | P14 Terms: "AOT compiled persistent requests ... MPI 4.1 §3.9"; MPI_START (code); NCCL stream semantics; Lam §1; AOT appendix 2: Bhattacharyya, Sriram and Lee's synchronization edges with delays, "start(vi, k) ≥ end(vj, k − delay((vj, vi)))", the self-timed buffer bound and DetermineDelays; DP D.3.3: host inputs get copies "as modulo variable expansion does for any value. With the host writing step t + 1 while the GPU runs step t, the lag is 2"; P14 Terms quotes Lam §2.3's "at least ⌈l/s⌉ number of values". | The start mechanism follows MPI_START and a BSL edge with delay lead + 1. The numbers do not follow the named methods: `lead` is a caller constant (e2b 1, mesh_compile 0) and HOST_LAG a `#define 2`; neither is computed by ⌈l/s⌉ or by BSL's maximum cycle mean / DetermineDelays (§6). (The section number §3.9 is the MPI 4.1 PDF's; the HTML page the code links, node83.htm, labels the same section "4.9".) |
| 7 Contiguous view | PB R2 corrected: the page table (Kilburn et al. 1962; Denning 1970), PagedAttention's block tables (SOSP 2023), Legion (SC 2012); "Such an operand's run is address translation, fixed when the plan is made"; the precedent cited is mesh's own `ncclMeshPageTableCompose`/`mach_vm_remap` of 2026-11-02 (repository history, not literature); Metal 4 placement sparse buffers named and then dropped ("With the run a mapping, none of these is needed"). | Yes, address translation fixed at plan time. No published work on reserving a virtual range and remapping physical pages under it is cited (§7 supplies Mach VM, CUDA VMM, vAttention, TPCircularBuffer). |
| 8 Lifetimes/sharing | DP A3, C3.3 and Appendix B: interference by happens-before (Lamport 1978), CUDA graph memory nodes, Dilworth (1950) by Fulkerson (1956); P14 Terms and item 14: Lam §2.3 lifetimes and ⌈l/s⌉, "colored as Mukerji 1994 §3.2 colors a cyclic interval graph", Hendren, Gao, Altman, Mukerji (CC 1992; ACAPS memo 33); P14 item 7: a value's lifetime is its first and last node (ExecuTorch `update_tensor_lifetime`), never-freed for data kept across executions; MPI §3.7 holds. | `mesh_graph_fat_cover` follows Mukerji §3.2; `mesh_graph_chains` follows Dilworth/Fulkerson. `program()` does not follow P14 item 7's first/last rule: every EXECUTION value of a captured function gets [0, levels − 1] (§8). |
| 9 Mixed execution | AOT A1 corrected: `torch.export`, "nothing runs eagerly between its compiled calls"; no method for running some nodes by replay and others by per-operator dispatch. | No cited method. fcf5d18a's split sits against the operator's requirement quoted in AOT A1 (below). |

**Operator turns these documents quote that bear on the mechanisms** (context only):

- On masking versus waiting and the list's placement (DP, the turns, 2026-11-03): "is this solved by MASKING or is it
  solved by WAITING? finally, is the LIST OF MASKED PAGE INDICES a contiguous array iN ITS OWN SECTION OF THE MEMORY
  ALLOCATION, or is it a struct which MUST BE CONTIGUOSU WITH THE PAGE IT IS ABOUT FOR SOME STUPID REASON?" The answer
  that follows it in DP ("lanes or threadgroups whose input pages aren't present are predicated off, and that work runs
  in a later pass when they are") is Claude's turn, not the operator's.
- On the first implementation (DP C1): "\"1. Read-after-write is a wait, not a mask. Your answer was masking: \" then the
  change request document's requirements simply have not been satisfied yet. revision is required."; and "so what i'm
  hearing is that there is a reference algorihtm which should have been cited and used ,but someone wrote an incomplete
  or incorrect variant algorithm instead."
- On the phase field (DP C1): "i would suggest the ptx mbarrier phase bit and similar; if you want to rotate through an
  int2 of phases instead of the reductive int1 (or bit) go for it, idc."
- On the allocator (DP Appendix B): "\"This is how CUDA graph memory nodes reuse memory.\" this + unrolling is the user's
  proposal for how all of this is implemented for the subsidiary component of this change request."
- On views (PB R2): "does this really require special kernels or is it an address translation problem that can be solved
  aot and be effectively zero marginal latency at runtime...?"
- On eager execution (AOT A1): "is it idiomatic claudelish trying to imply a just in time or online problem when the user
  sided documentation emphasizes mayn constraints including saying in as many semantic frames as possible 'the bridge and
  mesh collectives do not support dynamic or eager operations conmingled with compiled operations, and we never run
  non-compiled operations ever, damn the consequences'?"
- On callbacks (PB, the turns): "are they callback driven programming (forbidden) or inversion of control to avoid
  stating a problem plainly and assigning its solution to the simple and clear point in control flow and aot
  compilation vs runtime execution?" — `MetalRecorderRun` starts each execution's transfers from a
  `notifyListener:atValue:block:` block (§6), and AOT appendix 7 itself prescribes a publish "done in that event's
  notification"; the documents are in tension here.
- On rings (PB, the turns): "when is the last time anyone has given you a requirement saying 'you must find a waiting
  ring buffer algorithm and use it at all costs at all times'? citation?"

**Literature the documents leave missing** (no cited method, or a method named and not applied):
1. Capture by interposing the API; what a capture records, refuses, or executes (copies, fills, waits). None cited.
2. The replay period: the documents quote Lam's max_i q_i; the code uses the lcm; no source chooses the 4u phase-row
   period.
3. Testing every page of an operand copy in every threadgroup, and retrying failed threadgroups after a queue wait: no
   single cited source; the parts are cited.
4. The progress-thread structure (a CPU thread converting a memory counter into a GPU-visible event): not cited as such.
5. Mixed-size atomic access, a seq_cst fence for a release, and the fence's scope value: none.
6. How many copies (HOST_LAG) and how far ahead (`lead`): the methods are named (Lam's ⌈l/s⌉, BSL's delays) and not
   applied; "the lag is 2" has no source.
7. Reserving a virtual range and remapping scattered physical pages under it: none beyond repository history.
8. Giving every captured value the whole range of levels: contradicts the named first/last rule, and the named rule's
   source is in the torch family, which the operator has ruled out as a reference; a non-torch source for graph
   inputs/outputs and kept values is missing (§8 gives TFLite's arena planner and earlier liveness work).
9. Running some nodes by replay and others by per-operator dispatch in one program: none.

## B. The nine mechanisms against the literature

Code is read at metal-microbench 0771d07a and mesh 165f2c98 (`git show`); line numbers are of those revisions.

### 1. Capture and replay of a GPU command stream

**Code.** `MetalRecordingDevice`/`MetalRecordingDevices` (metal_recording.m 4097–4113) replace
`MTLCreateSystemDefaultDevice` and `MTLCopyAllDevices` through dyld's `__interpose` section (4115–4120). Every object the
delegate obtains is an `NSProxy` forwarder (`MetalForwarder`, 2286–2397). While `activeCapture` is set,
`MetalComputeRecorder` records each dispatch's pipeline, bindings, inline bytes and grid into `struct
MetalRecordedDispatch` and does not forward the dispatch (2656–2666); `MetalCommandRecorder blitCommandEncoder`
(3052–3056) returns `MetalBlitPreparation`, which forwards `fillBuffer:range:value:` to the real encoder — so the fill
runs once, in the delegate's capture-time command buffer — and refuses every other blit command with
`MetalRefuseInvocation` (3022–3037). `capture:` (3373–3421) then analyses the trace: renaming of whole-output buffers,
fusions, masked readers (`MetalMaskReaders`), the delegate's buffers as values (`MetalDelegateValues`, 2052–2105, where a
buffer whose first access in a repetition is a load becomes METAL_INVARIANT and is copied once at `replay:`), and
dependency levels (`MetalPrepareDataflow`, 2128–2280: a command's level is the maximum over its read-after-write,
write-after-read and write-after-write predecessors, and a level change sets the command's barrier). `replay:`
(3508–3762) builds indirect command buffers of `MTLIndirectCommandTypeConcurrentDispatch |
MTLIndirectCommandTypeConcurrentDispatchThreads`, `inheritBuffers = NO`, holding L repetitions of the capture's commands
(L from `MetalReplayPeriod`, 3249–3275), in segments of at most 2^22 commands (metal_recording.h 6–8; the header text
gives no maximum), and raises "Pipeline does not support indirect commands" (3713–3714) for a pipeline without
`supportIndirectCommandBuffers`; `MetalCreatePipeline` (430–465) sets that flag on every pipeline made through the proxy
device. `MetalReplayExecute`/`MetalReplayRun` (3804–3841) execute ranges with `executeCommandsInBuffer:withRange:`;
`MetalReplayEncode` (3948–3993) encodes repetition t: the commands through each first pass's compaction, a commit, the
queue's `waitForEvent:value:` at base + p + 1, then the later pass by `dispatchThreadgroupsWithIndirectBuffer:` through
an MTL4 argument table, then the rest. d0a72fc6 records copies and fills as commands (`MetalBlit`, kind
MetalCopy/MetalFill) and dispatches of non-indirect pipelines as kind MetalDirect, but no recorder enables it and
`replay:` still raises for a direct dispatch.

**(a) Standard names.** A *stored command list executed repeatedly by a separate processor* (channel program, display
file, display list, command buffer, bundle, CUDA graph, indirect command buffer), obtained by *trace capture through API
interposition* (proxy objects around the real interface, the device-creation entry point redirected: apitrace,
GFXReconstruct), recorded once and *replayed* many times — the inspector/executor split of "plan once, amortize over
iterations". The analysis between capture and replay (renaming, levels separated by barriers) is a dependence
levelization of the trace. Commands executed during recording instead of being stored are OpenGL's "executed
immediately" commands.

**(b) Sources** (agent a unless marked; quotes checked against the downloads).
- 1964 — IBM, *IBM System/360 Principles of Operation*, form A22-6821-0 (http://bitsavers.org/pdf/ibm/360/princOps/A22-6821-0_360PrincOps.pdf;
  lit/a-s360-pop/). p. 20: "One or more CCW's make up the channel program that directs channel operations. If more than
  one CCW is to be used, each CCW points to the next CCW to be fetched, except for the last CCW in the chain, which
  identifies itself as the last in the chain." p. 85: "Instructions are decoded and executed by the CPU and are part of
  the CPU program. Commands are decoded and executed by the channels, and initiate I/O operations, such as reading and
  writing." p. 96: "Fetching of the CCW's by the channel does not affect the contents of the location in main storage."
- 1968 — T. H. Myer and I. E. Sutherland, "On the Design of Display Processors", CACM 11(6):410–414, 1968
  (http://cva.stanford.edu/classes/cs99s/papers/myer-sutherland-design-of-display-processors.pdf; lit/a-myer-sutherland-1968/).
  p. 411: "The channel takes successive data words from a display file in core until the word count goes zero, at which
  point the central processor restarts the channel at the beginning of the display file. Now the processor is freed for
  other work and the display can operate as fast as its analog circuits permit." p. 411: "One is a jump command. This
  allows the channel to display a file repetitively—to refresh the display without intervention from the central
  processor." p. 412: "it would be nice if all parts of a display program could be affected by changing one key word."
- 1973 — Digital Equipment Corporation, *VT11 Graphic Display Processor*, DEC-11-HVTGA-A-D, Dec 1973 (bitsavers;
  lit/a-dec-gt40/). p. 2-2: "Capable of executing its own display program, the display processor, once started, can
  function independently."
- 1989 — K. Crowley, J. Saltz, R. Mirchandaney, H. Berryman, "Run-time Scheduling and Execution of Loops on Message
  Passing Machines", ICASE Report 89-7 (NASA CR-181785), 1989; journal version JPDC 8(4), 1990
  (https://ntrs.nasa.gov/api/citations/19890011267/downloads/19890011267.pdf; lit/a-saltz-1989/). p. 3: "Without them
  performance penalties would have to be paid during every loop iteration rather than being paid for once during an
  initial optimization." p. 11: "were the elements needed by an index of computation to change between iterations, the
  information computed by the preprocessor will be invalid after that iteration."
- 1989 — DEC, *DEC GKS User Manual*, AA-HW45C-TE, 1989 (implements ISO 7942-1985; lit/a-dec-gks-1989/). p. 1-4: "A
  segment is a group of output images that can be stored and manipulated as a group." p. 1-4: "Once you close a segment,
  you cannot add or delete primitives from the segment."
- 1990 — J. Saltz, R. Mirchandaney, K. Crowley, "Run-time Parallelization and Scheduling of Loops", ICASE Report 90-34
  (NASA CR-182039), 1990 (lit/a-saltz-1990/), abstract: "(1) inspector procedures that perform execution time
  preprocessing and (2) executors or transformed versions of source code loop structures. These transformed loop
  structures carry out the calculations planned in the inspector procedures." "the overheads associated with this type of
  reordering are amortized when the loop is executed several times with the same dependency structure."
- 1997 — M. Segal, K. Akeley, *The OpenGL Graphics System: A Specification (Version 1.1)*, SGI, 1997
  (https://registry.khronos.org/OpenGL/specs/gl/glspec11.pdf; lit/a-opengl-spec/). §5.4, p. 140: "A display list is
  simply a group of GL commands and arguments that has been stored for subsequent execution. The GL may be instructed to
  process a particular display list (possibly repeatedly) by providing a number that uniquely specifies it." p. 140: "As
  always, pointers which are passed as arguments to commands are dereferenced when the command is issued." p. 142:
  "Certain commands, when called while compiling a display list, are not compiled into the display list but are executed
  immediately."
- 1999/2000 — V. Bala, E. Duesterwald, S. Banerjia, "Transparent Dynamic Optimization: The Design and Implementation of
  Dynamo", HP Labs HPL-1999-78 (Wayback copy; lit/a-dynamo/), §3.2, p. 12: "When interpreting in this mode, the native
  instructions comprising the trace are emitted into a buffer." PLDI 2000 version, §2: "the performance benefits from
  repeated reuse of the optimized fragments can be sufficient to offset the overhead of Dynamo's operation." — recording
  what executes, then replaying the recording.
- 2009 — A. Gal et al., "Trace-based Just-in-Time Type Specialization for Dynamic Languages", PLDI 2009 (lit/a-tracemonkey/),
  p. 466: "Hence, recording and compiling a trace speculates that the path and typing will be exactly as they were during
  recording for subsequent iterations of the loop."
- 2018 — R. Frostig, M. J. Johnson, C. Leary, "Compiling machine learning programs via high-level tracing", SysML 2018
  (https://mlsys.org/Conferences/doc/2018/146.pdf; lit/f-frostig2018/, mine): "newly encountered array element types,
  array dimensions, or tuple members trigger a re-compilation. On a trace cache miss, JAX executes the corresponding
  Python function and traces its execution into a graph of primitive functions with static data dependencies."
- 2011– — apitrace (github.com/apitrace/apitrace, commit 3a0ee82a; MIT-style license; lit/a-apitrace/). README: "trace
  OpenGL, Direct3D, and DirectDraw APIs calls to a file;" "replay OpenGL and Direct3D calls from a file;". USAGE:
  "`apitrace trace` solves this issue by injecting a DLL `dxgitrace.dll` and patching all modules to hook only the APIs of
  interest." wrappers/trace.py builds "Wrap" + interface proxies that record each call, forward it, and wrap returned
  objects.
- 2018– — GFXReconstruct (github.com/LunarG/gfxreconstruct, dev at f9b0fb0d; MIT; lit/a-gfxreconstruct/). README: "The
  GFXReconstruct project provides tools for the capture and replay of graphics API calls, allowing the graphics commands
  executed by an application to be recorded to a file that may later be replayed to reconstruct the graphics-specific
  behavior of the captured application." layer/d3d12/d3d12.def: `D3D12CreateDevice=gfxrecon_D3D12CreateDevice @101`.
- 2015– — Microsoft, Direct3D 12 documentation (learn.microsoft.com; lit/a-d3d12/). D3D12_INDIRECT_ARGUMENT_TYPE lists
  draws, dispatches, views, constants, rays and mesh dispatch — no copy and no fill. "Creating and recording command
  lists and bundles": "Bundles are designed to be used and re-used any number of times." "The following command list API
  calls are not allowed on command lists created with type: D3D12_COMMAND_LIST_TYPE_BUNDLE: Any Clear method Any Copy
  method" and "If any of these APIs are called on a bundle, the runtime will drop the call." "Bundles inherit the command
  list's root signature bindings".
- 2016– — Khronos, Vulkan specification (lit/a-vulkan-spec/): "VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT specifies
  that each recording of the command buffer will only be submitted once, and the command buffer will be reset and
  recorded again between each submission."
- 2018– — Apple Metal documentation (developer.apple.com JSON; lit/a-apple-metal/): MTLIndirectCommandBuffer, "Use an
  indirect command buffer to encode commands once and reuse them, and to encode commands on multiple CPU or GPU threads."
  inheritBuffers: "If you set the value to true, don't set buffer arguments when you encode commands into the indirect
  command buffer. The commands use (inherit) the buffer arguments that you set on the parent encoder." Sample "Encoding
  indirect command buffers on the CPU": "you can't update an ICB's buffer set after you encode its commands".
  MTL4ComputeCommandEncoder: "Each Metal 4 compute encoder combines compute dispatch commands, blit commands, and
  acceleration structure commands into a single pass." Headers (SDK 26.4, mine): MTLIndirectCommandBuffer.h, "A bitfield
  of commands that may be performed indirectly" (draws and dispatches only); MTLComputePipeline.h,
  `supportIndirectCommandBuffers`, "This flag makes this pipeline usable with indirect command buffers.";
  MTL4ComputeCommandEncoder.h, "Encodes a command that copies data from a buffer instance into another." and "Encodes a
  command that fills a buffer with a constant value for each byte."
- 2020/2024 — Khronos, VK_NV_device_generated_commands (2020) and VK_EXT_device_generated_commands (2024)
  (lit/a-vulkan-spec/): "This extension allows the device to generate a number of commands for command buffers." Tokens
  hold state, draws, dispatches and ray tracing — no copy or fill. "VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT
  specifies that the pipeline can be used in a VkIndirectExecutionSetEXT."
  "VK_PIPELINE_CREATE_INDIRECT_BINDABLE_BIT_NV specifies that the pipeline can be used in combination with
  Device-Generated Commands."
- 2026 — NVIDIA, CUDA Programming Guide v13.5, §4.2 CUDA Graphs (https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html;
  lit/a-cuda-guide/, lit/f-cuda-graphs/): "A graph is defined separately from its execution. This allows a graph to be
  defined once and then launched repeatedly." "Work submission using graphs is separated into three distinct stages:
  definition, instantiation, and execution." §4.2.1.1 node types include "memory copy", "memset", "CPU function call",
  "waiting on a CUDA Event" and "waiting on an external semaphore". §4.2.2.1.2: "When a stream is being captured, work
  launched into the stream is not enqueued for execution. It is instead appended to an internal graph that is
  progressively being built up." §4.2.2.1.2.2: "It is therefore also invalid to call synchronous APIs in this case. One
  example of a synchronous APIs is cudaMemcpy() which enqueues work to the legacy stream and synchronizes on it before
  returning." §4.2.2.1.2.3: "When an invalid operation is attempted during stream capture, any associated capture graphs
  are invalidated."
- 2020s — IREE, runtime/src/iree/hal/command_buffer.h (iree-org/iree main, last change 17fb04de; Apache-2.0 WITH
  LLVM-exception; lit/a-iree/), lines 958 and 982: "Fills the target buffer with the given repeating value." "Copies a
  range of one buffer to another." Lines 779–788: "Indirect references are placeholders indicating that at the time the
  command buffer is submitted to a device queue a buffer will be provided allowing for the same command buffer to be
  reused with different buffers."
- 2026 — W3C, WebGPU Editor's Draft, 6 Nov 2026 (https://gpuweb.github.io/gpuweb/; lit/a-webgpu-spec/), §2.1.3:
  "WebGPU conceptually initializes all the resources to zero, although in practice an implementation may skip this step
  if it sees the developer initializing the contents manually." NOTE: "The initialization status of a resource used in a
  queue operation can only be known when the operation is enqueued (not when it is encoded into a command buffer, for
  example). Therefore, some implementations will require an unoptimized late-clear at enqueue time". §13.4 clearBuffer:
  "Encode a command into the GPUCommandEncoder that fills a sub-region of a GPUBuffer with zeros."
- Dawn (google/dawn main at 9165acf0; BSD-3-Clause; lit/a-dawn/): Toggles.cpp, LazyClearResourceOnFirstUse, "Clears
  resource to zero on first usage. This initializes the resource so that no dirty bits from recycled memory is present
  in the new resource."; metal/BufferMTL.mm 304–306, `[commandContext->EnsureBlit() fillBuffer:mMtlBuffer.Get()
  range:NSMakeRange(offset, size) value:clearValue];`.
- Not read (from memory): ISO 7942:1985 itself; Newman and Sproull, *Principles of Interactive Computer Graphics*
  (1973/1979), on display files; RenderDoc (MIT), frame capture by hooking API entry points.

**(c) Comparison.**
- *Matches.* Interposition through proxies and a redirected device constructor is apitrace's and GFXReconstruct's
  technique (they forward each call; the recorder forwards none of the dispatches during capture, which is CUDA stream
  capture's rule: "work launched into the stream is not enqueued for execution"). Record once, replay many with a
  validity condition (the same command structure each execution) is the display list's and Crowley/Saltz's
  preprocessor/executor; the requirement that a pipeline be flagged before an indirect list may hold it is Metal's own
  header and Vulkan's INDIRECT_BINDABLE bit.
- *Deviates, required.* Splitting each repetition into several command buffers at every wait, and running each later
  pass outside the indirect command buffer, is forced: `MTLIndirectCommandType` holds draws and dispatches only and
  `MTLIndirectComputeCommand` has no wait (headers). CUDA graphs and IREE can hold waits and copies; D3D12 indirect
  arguments and Vulkan device-generated commands cannot.
- *Deviates, not required — copies refused.* Metal 4's compute encoder encodes copies ("Encodes a command that copies
  data from a buffer instance into another"), so a captured copy could be replayed directly between ranges of the
  indirect command buffer (d0a72fc6 records them as `MetalCopy` but nothing replays them yet). The closest precedent for
  refusing is D3D12's bundles, which drop Copy and Clear calls; CUDA graphs (memcpy nodes), IREE (copy commands) and
  OpenGL display lists record copies.
- *Deviates, not required — a fill executed once at capture.* OpenGL's "executed immediately" list is the precedent for
  executing a command at record time instead of storing it; clears are not on that list. CUDA graphs (memset nodes),
  IREE (fill commands) and D3D12 bundles (which drop clears) all treat a fill as part of the replayed program. The
  equivalence holds only for a fill that is a one-time initialization of the buffer bound at capture — Dawn's lazy clear
  ("Clears resource to zero on first usage"), whose result reaches the placed pages through `MetalDelegateCopies` because
  the recorder classes a buffer read before it is written as an invariant. For an explicit per-execution fill (WebGPU
  `clearBuffer`), executing it once is wrong: from the second execution on, a kernel that accumulates into the buffer
  starts from the previous execution's sums (`MetalDelegateValues` classes a read-before-write buffer as invariant, i.e.
  carried across executions). The recorder cannot tell the two kinds of fill apart. No Metal text requires this; the MTL4
  compute encoder encodes fills ("Encodes a command that fills a buffer with a constant value for each byte").
- *Deviates, required for the indirect path only — the raise for a pipeline without `supportIndirectCommandBuffers`.*
  The header makes the flag a condition of indirect use, not of use: a pipeline without it can be dispatched directly in
  an MTL4 compute encoder. d0a72fc6 records such dispatches as `MetalDirect`, but `replay:` still raises.
- *Replay with different buffers* is IREE's binding table, CUDA's graph update and D3D12's inherited root bindings; the
  recorder instead unrolls L slots with fixed bindings (§2), because a CPU-encoded indirect command's buffer set is fixed
  ("you can't update an ICB's buffer set after you encode its commands"). The MSL standard library's
  `compute_command::set_kernel_buffer` (metal_command_buffer header, mine) lets a GPU kernel re-encode an indirect
  command's bindings, an API-available alternative the code does not use.
- *Unlike TraceMonkey and Dynamo*, a replay checks no guard: the recording is assumed valid for every execution. CUDA
  graphs make the same assumption and require re-instantiation when topology changes.

**(d)** Published counterparts exist for everything but the treatment of fills (record-time execution of a fill whose
kind — one-time or per-execution — is unknown). Verdict: **matches record/replay by interposition; deviates (required)
in splitting at waits; deviates (not required) in refusing copies and running fills once.**

### 2. Per-execution rebinding: execution t binds copy t mod copies

**Code.** `MetalBinding` (3130–3168): a placed operand in repetition t binds `offset + ((t + ahead) mod copies) ×
stride` where the command writes it and `offset + (t mod copies) × stride` where it reads; a placed value binds `offset +
(t mod copies) × stride`; a phase row binds `64 r + 16 ((t / u + 1) & 3)` of the phase table. `place:operand:…`
(3440–3454) refuses an `ahead` that is a multiple of `copies`. `MetalReplayPeriod` takes L as the least common multiple of
every bound operand's and value's copies, 4u for each received operand of a later pass, and the step word's copies.
`replay:` encodes slot t (t < L) with its own `setKernelBuffer:offset:atIndex:` calls, and the run's repetition t
executes slot t mod L (`MetalReplaySegments`, 3778–3788).

**(a) Standard name.** Modulo variable expansion (MVE) applied to buffers: the loop body (one execution's command list)
is unrolled and each unrolled instance names its own copy of every value whose lifetime spans iterations; a value
written for a later iteration lands `ahead` copies on. With hardware support the same renaming is a rotating register
file (or a "rotating memory buffer"); without unrolling, modulo addressing at run time, or binding buffers per
submission. The generic structure is the bounded buffer / multiple buffering.

**(b) Sources** (agent b unless marked; Lam is mine).
- 1965 — E. W. Dijkstra, "Cooperating Sequential Processes", EWD 123 (https://www.cs.utexas.edu/users/EWD/transcriptions/EWD01xx/EWD123.html;
  lit/b-dijkstra-ewd123/), §4.3 "The Bounded Buffer": "the producer can be arbitrarily far ahead of the consumer, on the
  other hand the consumer can never be ahead of the producer. The relation becomes symmetric, if the two are coupled via a
  buffer of finite size, say N portions."
- 1981 — Rau and Glaeser (MICRO-14) and Charlesworth (IEEE Computer 14(9)): **not read**; secondary, verified in Rau et
  al. below ("modulo scheduling as formulated by Rau and Glaeser [2]"; rotating registers "first appeared in the
  scratchpad register files of the FPS AP-120B and FPS-164").
- 1988 — M. Lam, "Software Pipelining: An Effective Scheduling Technique for VLIW Machines", PLDI 1988, pp. 318–328
  (https://suif.stanford.edu/papers/lam88.pdf; lit/lam1988/, page images read). p. 322: "We call this optimization of
  allocating multiple registers to a variable in the loop modulo variable expansion." "Modulo variable expansion ...
  reduces the number of locations allocated to a variable by reusing the same location in non-overlapping iterations."
  "If the lifetime of a variable is l, and an iteration is initiated every s cycles, then at least ⌈l/s⌉ number of
  values must be kept alive concurrently, in that many locations." "If each variable v_i is allocated its minimum number
  of locations, q_i, the degree of unrolling is given by the lowest common multiple of {q_i}. Even for small values of
  q_i, the least common multiple can be quite large and can lead to an" — p. 323 — "intolerable increase in code size.
  The code size can be reduced by trading off register space. We observe that the minimum degree of unrolling, u, to
  implement the same schedule is simply max_i q_i. This minimum degree of unrolling can be achieved by setting the number
  of registers allocated to variable v_i to be the smallest factor of u that is no smaller than q_i".
- 1989 — Dehnert, Hsu, Bratt, "Overlapped loop support in the Cydra 5", ASPLOS-III: **not read**; secondary in Rau et
  al. ("which were termed the MultiConnect in the Cydra 5").
- 1992 — B. R. Rau, M. Lee, P. P. Tirumalai, M. S. Schlansker, "Register Allocation for Modulo Scheduled Loops:
  Strategies, Algorithms and Heuristics", HP Labs HPL-92-48, 1992 (PLDI 1992 abridged)
  (http://shiftleft.com/mirrors/www.hpl.hp.com/techreports/92/HPL-92-48.pdf; lit/b-rau1992-hpl/), pp. 10–11: "The
  minimum degree of unroll, Kmin, is determined by the longest lifetime among the loop-variants. Because iterations are
  initiated every II cycles, Kmin can be calculated as: Kmin = MAX_i(⌈(end_i−start_i)/II⌉). By unrolling the kernel more
  than Kmin times, there is the possibility that fewer registers might be required in the modulo scheduled code, but at
  the cost of increasing the generated code size and the compile time of the allocation process." p. 28: "For over 80% of
  the loops, unrolling by more than Kmin saved at most one register." p. 31: "we now require a rotating memory buffer to
  hold the spilled values for the same reasons that motivate the use of rotating register files. Alternatively, we need
  to unroll the kernel in a manner similar to MVE." p. 32: "Our recommendation would be to employ the minimum degree of
  unroll."
- 1992 — B. R. Rau, M. S. Schlansker, P. P. Tirumalai, "Code Generation Schema for Modulo Scheduled Loops", MICRO-25,
  pp. 158–169 (lit/b-rau1992-micro25/, page images), §3, p. 162: "One way to accomplish this is to use different versions
  of the code for successive iterations, with each version making use of different registers as a result of modulo
  variable expansion. The alternative is to use a single version of the code and to provide a rotating register file
  that dynamically renames the instruction-specified sources and targets, thereby achieving the same objective."
- 1993 — Q. Ning and G. R. Gao, ACAPS Technical Memo 42, McGill, 1993 (POPL 1993 version not read; lit/b-ninggao-memo42/),
  §6.2: "we simply let the next iteration write to the next position in the corresponding FIFO buffer. Thus, data are kept
  stationary, while accesses to the registers of a FIFO buffer by the producer and successors are performed with the
  modulo addressing method."
- 1994 — B. R. Rau, "Iterative Modulo Scheduling", MICRO-27, pp. 63–74 (lit/b-rau1994-micro27/), p. 64: "If rotating
  registers [37, 5] are absent, the kernel (i.e., the new loop body after modulo scheduling has been performed) is
  unrolled to enable modulo variable expansion [21]."
- 2017/2018 — Apple, Metal Best Practices Guide, "Triple Buffering" (lit/b-apple-triple/), Listing 4-1:
  `_currentFrameIndex = (_currentFrameIndex + 1) % kMaxInflightBuffers;` … `setVertexBuffer:_dynamicDataBuffers[_currentFrameIndex]`
  — rebinding by modulo index, re-encoded each frame.
- 2020s — IREE (iree-org/iree @ 9b906db8; Apache-2.0 WITH LLVM-exception; lit/b-iree-hal/, lit/a-iree/),
  command_buffer.h 781–785: "Indirect references are placeholders indicating that at the time the command buffer is
  submitted to a device queue a buffer will be provided allowing for the same command buffer to be reused with different
  buffers. Indirect command buffers are not concurrently schedulable unless specified as many implementations need per
  submission shadow resources."
- 2026 — CUDA Programming Guide v13.5 §4.2.3 (lit/b-cuda-graphs/, lit/f-cuda-graphs/): "it is common for only node
  parameters (such as kernel parameters and memory addresses) to change while the graph topology remains the same. For
  this case, CUDA provides a lightweight “Graph Update” mechanism that allows certain node parameters to be modified
  in-place without rebuilding the entire graph".
- Metal (mine): MTLIndirectComputeCommand binds with `setKernelBuffer:offset:atIndex:` when encoded; Apple's ICB sample
  (lit/a-apple-metal/): "you can't update an ICB's buffer set after you encode its commands"; `inheritBuffers`: "The
  commands use (inherit) the buffer arguments that you set on the parent encoder."; the MSL standard library
  (GPUCompiler 32023.864, metal_command_buffer) declares `compute_command::set_kernel_buffer(device T *buffer, uint
  index)`, i.e. a GPU kernel may re-encode an indirect command's bindings.

**(c) Comparison.**
- *Matches* MVE: slot t of the unrolled indirect command buffer binds copy t mod copies, and a write for a later
  execution goes to copy (t + ahead) mod copies, which `place:operand:` refuses where it would hit the copy being read.
- *Deviates in the unroll degree, not required.* The code takes L = lcm of every binding's period (`MetalReplayPeriod`,
  which quotes Lam's lcm sentence). Lam, in the next sentences, and Rau et al. (who measured it) recommend the minimum
  degree, max_i q_i, with each q_i rounded up to a factor of it. With e2b's periods {u for routed operands, HOST_LAG = 2,
  4u for a received operand's phase rows}, L = 4u (4 for u = 1, 8 for u = 2). The 4u is the code's own: it comes from
  binding a different phase-table row per (t/u mod 4) instead of computing the expected phase in the kernel from the step
  word the later pass already reads. Without it the periods are {u, 2} and Lam's form gives L = max(u, 2) with host inputs
  rounded up (u = 3: L = 3 with three host-input copies; the lcm gives 6).
- *Unrolling at all* follows from fixed per-command bindings of a CPU-encoded indirect command buffer (Apple's sample).
  API-available alternatives the literature names: per-submission binding (IREE), graph update (CUDA), modulo addressing
  at run time (Ning–Gao), rotating buffers (Rau); in Metal, inherited buffers (one binding set per executed range) or GPU
  re-encoding through `compute_command::set_kernel_buffer`. IREE's caveat applies to every per-submission form:
  concurrent executions need per-submission shadow state, so some multiplicity (at least the executions in flight)
  remains.
- The 2^22-command segment limit (metal_recording.h) has no header text; `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:`
  documents no maximum. It is an empirical limit.

**(d)** Published counterpart exists. Verdict: **matches MVE; deviates (not required) in using the lcm — and a 4u period
of its own making — where Lam and Rau et al. prescribe the minimum degree max_i q_i.**

### 3. Read-after-write on data a remote writer lands: masked first pass, compaction, later pass

**Code.** `MetalMaskReaders` (1920–2018) takes each recorded command that loads a
received operand, rewrites its MSL source with `MetalMaskedPass` (1724–1870) into two kernels, and inserts a compaction
command after it (`MetalCompactionSource`, 1878–1897). First form: each threadgroup loads, for each received operand,
the whole phase group of copy t mod u (`ceil(P/64)` 16-byte loads, through `volatile coherent(system)` pointers), checks
every field k < P against the expected phase and the one-behind phase of row `(t/u + 1) & 3` (rows hold
`MetalPhase(q) = q ^ (q >> 1)`, i.e. 0, 1, 3, 2, written once by `replay:`, 3573–3586), reduces the per-lane votes with
`simd_ballot` and threadgroup memory, writes its flag (1 landed), and returns before any store if not landed; else a
seq_cst fence at scope 3, then the body. The compaction (one threadgroup) builds the list of flag-0 threadgroups with
`simd_prefix_exclusive_sum` per SIMD-group and an exclusive scan across SIMD-groups, and writes
`MTLDispatchThreadgroupsIndirectArguments {left, 1, 1}`. Later form: threadgroup g runs as `list[g]`, its grid builtins
recomputed; a threadgroup still not landed adds to the status's `behind` or `ahead`, raises `first` to `~t`, and
returns. Flags, list and arguments are values of the capture (separate allocations placed by the plan); the phase list
is in the plan's state section, apart from the data pages (mesh rdma/mesh.h 355–375). The bridge lands a page and then
flips one bit of its field (`atomic_fetch_xor_explicit(word, 1 << (field + (round & 1)), release)`, mesh-flow.c 1474),
so the field walks the 2-bit Gray sequence; then it adds one to the copy's count (1475).

**(a) Standard names.** Three published structures composed:
1. a *presence/validity flag per element kept as a dense array apart from the data* (full/empty bits, I-structure
   presence bits, a validity bitmap), widened to a 2-bit *phase* so that it is never reset between rounds — the
   sense-reversal / phase-tag / wrap-counter / mbarrier-parity idea — and Gray-coded so that each round flips one bit;
2. *predication by a mask, then stream compaction of the mask into an index list* (APL compression, Hillis–Steele
   enumerate, Blelloch's pack by prescan) driving an *indirect dispatch* of the residue: a topology-driven pass followed
   by a data-driven (worklist) pass;
3. a *coarse wait* (the queue's `waitForEvent:value:` on an event a host thread raises from completion counts, §4)
   before the residue runs.
In the CUDA guide's taxonomy (§4.6, Table 14) the readiness test is a *flag protocol* read without waiting, and the
retry is ordered at stream (queue) granularity. Validate-then-run with re-execution of the failed units is the shape of
optimistic concurrency control (check, then write only if the check passed).

**(b) Sources** (agent c unless marked "mine"; quotes checked by script against the downloads).
- 1953 — F. Gray, "Pulse Code Communication", US Patent 2,632,058, filed 1947, issued 17 Mar 1953
  (https://patents.google.com/patent/US2632058A/en; lit/c-gray/): "...the code in question, which has as yet no
  recognized name, is designated in this specification and in the claims as the reflected binary code." "The reflected
  binary code may take many forms, all of which have the property that no number differs from its neighbors in more than
  one digit." Two-digit table: "Zero 00 One 01 Two 11 Three 10".
- 1962 — K. E. Iverson, *A Programming Language*, Wiley, 1962, §1.9 "Compression" (https://www.jsoftware.com/papers/APL1.htm#1.9.1;
  lit/c-iverson/): "the vector c is obtained from a by suppressing from a each component aᵢ for which uᵢ = 0. The vector
  u is said to compress the vector a. Clearly ν(c) = +/u."
- 1967 — R. M. Tomasulo, "An Efficient Algorithm for Exploiting Multiple Arithmetic Units", IBM J. R&D 11(1):25–33,
  1967 (https://www.cs.virginia.edu/evans/greatworks/tomasulo.pdf; lit/c-tomasulo/), p. 31: "Each active reservation
  station (selected but awaiting a register operand) compares its sink and source tags to the CDB tag. If they match, the
  reservation station ingates the data from the CDB."
- 1977 — L. Lamport, "Concurrent Reading and Writing", CACM 20(11):806–811, 1977 (lit/c-lamport77/), p. 808: "The writer
  increments v1 before writing the data item and increments v2 after writing. The reader reads v2 before reading the data
  item and v1 after reading it. If it finds them equal, then it knows that it read a single version of the data."
- 1978 — R. M. Russell, "The CRAY-1 Computer System", CACM 21(1):63–72, 1978 (reprint scan, mine;
  lit/f-cray1-1978/): "the 64-bit vector mask (VM) register controls vector element designation in vector merge and test
  instructions. Each bit of the VM register corresponds to a vector register element."
- 1981 — H. T. Kung and J. T. Robinson, "On Optimistic Methods for Concurrency Control", ACM TODS 6(2):213–226, 1981
  (lit/c-kung81/), p. 215: "If validation does fail, the transaction will be backed up and start over again as a new
  transaction. Thus a transaction will have a write phase only if the preceding validation succeeds."
- 1986 — W. D. Hillis and G. L. Steele Jr., "Data Parallel Algorithms", CACM 29(12):1170–1183, 1986 (lit/c-hillis/),
  p. 1174: "two operations are frequently useful: determining how many processors are active, and assigning a distinct
  integer to each processor. We call the first operation count and the second enumerate"; p. 1175: "every processor
  receives a count of the number of active processors that precede it (including itself) in the ordering."
- 1989 — Arvind, R. S. Nikhil, K. K. Pingali, "I-Structures: Data Structures for Parallel Computing", TOPLAS
  11(4):598–632, 1989 (authors' manuscript; lit/c-istruct/), pp. 12–13: "a consumer that tries to read some A[i] while it
  is still empty is made to wait until the corresponding (f i) has completed. One way to achieve this synchronization is
  to use I-structure storage, where each location has "presence bits" to indicate whether the value is present or
  absent." p. 13: "the consumer of an array can begin work on parts of the array while the producer of the array is still
  working on other parts." p. 19: "If the location is still empty, the controller simply queues the tag at that
  location." p. 18: "These restrictions— write-once, deferred reads, and no test for emptiness— ensure that the language
  remains determinate".
- 1990 — R. Alverson et al., "The Tera Computer System", ICS 1990 (lit/c-tera/), §5: "the full/empty bit is used for
  lightweight synchronization." §4: "If a synchronization is not satisfied for a long time, then possibly a heavier weight
  mechanism that avoids busy waiting should be used to wait for the synchronization."
- 1990 — G. E. Blelloch, "Prefix Sums and Their Applications" (Ble93.pdf version of CMU-CS-90-190;
  lit/c-blelloch/), p. 38: "We call such an operation, a prescan." p. 46: "The basic idea is to determine a new index for
  each element and then permute the elements to these new indices using an exclusive write."
- 1991 — J. M. Mellor-Crummey and M. L. Scott, "Algorithms for Scalable Synchronization on Shared-Memory
  Multiprocessors", TOCS 9(1):21–65, 1991 (lit/c-mcs/), p. 34: "Consecutive barriers cannot interfere with each other
  because all operations on count occur before sense is toggled to release the waiting processors." p. 37: Hensgen,
  Finkel and Manber "use sense reversal to avoid resetting variables after every barrier."
- 2002 — C. E. Cummings, "Simulation and Synthesis Techniques for Asynchronous FIFO Design", SNUG 2002 (Wayback copy;
  lit/c-cummings-wb/), §2.3: "Gray codes only allow one bit to change for each clock transition, eliminating the problem
  associated with trying to synchronize multiple changing signals on the same clock edge."
- 2007 — M. Harris, S. Sengupta, J. D. Owens, "Parallel Prefix Sum (Scan) with CUDA", GPU Gems 3, ch. 39
  (lit/c-gpugems3/): "Informally, stream compaction is a filtering operation: from an input vector, it selects a subset of
  this vector and packs that subset into a dense output vector." "Stream compaction requires two steps, a scan and a
  scatter."
- 2009/2012 — Direct3D 11 `DispatchIndirect` and OpenGL 4.3 `glDispatchComputeIndirect` (lit/c-d3d11/, lit/c-gl43/):
  "glDispatchComputeIndirect launches one or more compute work groups using parameters stored in the buffer object
  currently bound to the GL_DISPATCH_INDIRECT_BUFFER target." "work groups are executed completely independently and in
  unspecified order."
- 2013 — R. Nasre, M. Burtscher, K. Pingali, "Data-driven versus Topology-driven Irregular Computations on GPUs",
  IPDPS 2013 (lit/c-nasre/), p. 1: "In topology-driven implementations, all nodes are assumed to be active so the
  operator is applied everywhere even if there is no work to do at some nodes"; p. 2: "data-driven implementations are
  organized around a worklist"; p. 3: "The threads in a thread block can compute the total amount of work to be added by
  all threads in the thread block using a prefix sum (in shared memory)".
- 2011/2019 — NVM Express 1.4 (2019; lit/c-nvme/) §1.4: "A Phase Tag (P) bit is defined in the completion queue entry to
  indicate whether an entry has been newly posted without consulting a register. ... each round through the Completion
  Queue entries, the controller inverts the Phase Tag bit." §4.6: "If a Completion Queue Entry is constructed via multiple
  writes, the Phase Tag bit shall be updated in the last write of that Completion Queue Entry."
- 2019 — OASIS, VIRTIO 1.1 cs01, §2.7.1 (lit/c-virtio/): "Each of the driver and the device are expected to maintain,
  internally, a single-bit ring wrap counter initialized to 1." "To mark a descriptor as used, the device sets the
  VIRTQ_DESC_F_USED bit in Flags to match the internal Device Ring Wrap Counter."
- 2014 — A. Dragojević et al., "FaRM: Fast Remote Memory", NSDI 2014 (lit/c-farm14/), p. 407: "We use 64-bit header
  versions to prevent wrapping around but cache line versions only keep the least significant l bits of the version to
  save space. We can do this because there is a lower bound on the time it takes to perform a write and we abort RDMA reads
  that take longer than an upper bound".
- rdma-core v65.0 providers/mlx5/cq.c (GPL-2.0/OpenIB BSD; lit/c-rdmacore/): the expected owner bit is the wrap parity of
  the consumer index (`!((cqe64->op_own & MLX5_CQE_OWNER_MASK) ^ !!(n & (cq->verbs_cq.cq.cqe + 1)))`), then "Make sure we
  read CQ entry contents after we've checked the ownership bit." `udma_from_device_barrier();`.
- NCCL v2.32.3-1 (Apache-2.0; lit/c-nccl/), src/include/device.h 77–87: `union ncclLLFifoLine` with the comment "Flags
  have to be *after* data, because otherwise, an incomplete receive from the network may receive the flag but not the
  data."; prims_ll.h 113–119: the reader spins `while ((flag1 != flag) || (flag2 != flag))`; prims_ll128.h 203:
  `} while (__any_sync(WARP_MASK, needReload));`.
- 2016 — D. Merrill and M. Garland, "Single-pass Parallel Prefix Scan with Decoupled Look-back", NVR-2016-002
  (lit/c-merrill/), §4.2: "Safety. The algorithm will run to completion if the system guarantees forward-progress for all
  processors." p. 7: "The fence can be eliminated if the status flag and the corresponding value being updated can be
  combined into a single architectural word."
- 2016 — T. Sorensen et al., "Portable Inter-workgroup Barrier Synchronisation for GPUs", OOPSLA 2016 (lit/c-sorensen/),
  §1: "the GPU execution model is licensed to postpone execution of workgroup B until execution of workgroup A has
  completed, which will lead the kernel to deadlock due to starvation."
- 2015 — R. Belli and T. Hoefler, "Notified Access", IPDPS 2015 (mine; lit/f-belli-hoefler2015/): "we introduce counting
  notifications that only notify after n matching accesses were performed."
- 2020 — PTX ISA (9.5 as fetched; mine and agent c; lit/f-ptx-isa/, lit/c-ptx/): "The .parity variant of the
  instructions test for the completion of the phase indicated by the integer parity of the operand phaseParity, which
  denotes either the current phase or the immediately preceding phase of the mbarrier object." "tx-count tracks the
  number of asynchronous transactions, in units specified by the asynchronous memory operation, that are outstanding and
  yet to be complete." "mbarrier.test_wait is a non-blocking instruction which tests for the completion of the phase."
  "Modifier .parity is introduced in PTX ISA version 7.1."
- 2020 — OpenSHMEM 1.5, §9.8.3 (lit/c-openshmem/, lit/f-openshmem15/): "The put-with-signal routines provide a method for
  copying data from a contiguous local data object to a data object on a specified PE and subsequently updating a remote
  flag to signal completion."
- 2021/2023 — MPI 4.1 (2 Nov 2023), ch. 4 "Partitioned Point-to-Point Communication" in the PDF (§4.2.1–4.2.2, pp.
  119–120; lit/d-mpi41/), numbered ch. 5 in the HTML rendering (node86–89; lit/c-mpi41/): "MPI_PREADY is a
  send-side call that indicates that a given partition is ready to be transferred." "The MPI_PARRIVED function can be
  used to determine if the message data for the indicated partition has been received into the receive buffer. Upon
  success, the receiver becomes free to access the indicated partition".
- 2022 — T. Dao et al., "FlashAttention", arXiv 2205.14135 (lit/c-flashattn/), §3.3: "Given a predefined block sparsity
  mask M ∈ {0,1}^{N/B_r×N/B_c} we can easily adapt Algorithm 1 to only compute the nonzero blocks of the attention matrix.
  The algorithm is identical to Algorithm 1, except we skip zero blocks."
- 2024 — L.-W. Chang et al., "FLUX", arXiv 2406.06858 (lit/c-flux/), p. 9: "WaitSignal is fused in the prologue. Its
  latency varies based on arrival time of corresponding data transfers. For the tile with data already arrived, the
  latency is close to zero." p. 12: "All signals are allocated contiguously for easy preset and reset ... On the host
  side, a signal is set through a cuStreamWriteValue API with a stream, while on the kernel, WaitSignal is implemented
  through spinning."
- 2025 — S. Zheng et al., "TileLink", arXiv 2503.20313 (lit/c-tilelink/), Table 3: "Consumer tile is blocked until all its
  dependent producer tiles done".
- 2024 — S. Pati et al., "T3", ASPLOS 2024 (arXiv 2401.16677; lit/c-t3/): "a lightweight Tracker at the memory
  controller, that tracks local and remote/DMA accesses to memory regions and triggers a DMA transfer once the required
  accesses are complete."
- Apache Arrow Columnar Format, "Validity bitmaps" (mine; lit/f-arrow-columnar/): "The associated validity bitmap is
  contiguously allocated (as described above) but does not need to be adjacent in memory to the values buffer."
- RISC-V V 1.0 (mine; lit/f-rvv/): "The mask value used to control execution of a masked vector instruction is always
  supplied by vector register `v0`."
- 2026 — Apple, Metal Shading Language Specification 4.1, 2026-06-04 (lit/c-msl/), §6.10.1: "All threads in a threadgroup
  executing the kernel, fragment, mesh, or object need to execute this function before any thread can continue execution
  beyond the threadgroup_barrier." p. 216: "If threadgroup_barrier (or simdgroup_barrier) is inside a conditional statement
  and if any thread enters the conditional statement and executes the barrier function, then all threads in the
  threadgroup (or SIMD-group) need to enter the conditional and execute the barrier function." §5.2.3.6: "SIMD-groups
  execute concurrently within a given threadgroup and make independent forward progress with respect to each other, in
  the absence of threadgroup barrier operations." (No statement on progress between threadgroups.) §6.10.2 `simd_ballot`:
  "The function sets the bits that correspond to inactive threads to 0."
- 2026 — NVIDIA, CUDA Programming Guide v13.5, §4.6 "Fine-Grained Synchronization" (mine; lit/f-cuda-fine-grained-synchronization/):
  "Fine-grained synchronization lets a consumer determine when each tile of its input is ready by inspecting the data
  itself, rather than by waiting on a flag or on the completion of the producer grid." §4.6.2: "A flag protocol resolves
  dependencies between individual thread blocks while both kernels run. The producer writes a tile, executes a
  device-scope release fence so that the tile is visible before the flag (see Section 3.2.3), and then writes the flag.
  The consumer polls the flag with acquire semantics and only then loads the tile." Table 14: the flag protocol's
  granularity "Tile", requirement "Producer able to make progress while the consumer waits; one flag per tile". §4.6.5:
  "The protocol is correct without overlap, because a consumer re-issues its copies until the data is present." Appendix
  5.9.2: "Threads in other thread-block clusters are not guaranteed to eventually make progress."
- Metal SDK 26.4 and MSL headers (mine): `MTLIndirectComputeCommand` has no wait; the MSL standard library (GPUCompiler
  32023.864) declares `threadgroup_barrier`, `simdgroup_barrier` and indirect-command encoding, and no wait, sleep or
  suspend on memory.
- Not read (from memory): Allen, Kennedy, Porterfield, Warren, "Conversion of control dependence to data dependence",
  POPL 1983 (ACM blocked; if-conversion turns branches into guards on statements); J. B. Dennis and D. P. Misunas,
  data-flow processor (1974/75; a cell fires when its operands are present); B. J. Smith, HEP (1978/81; full/empty bits,
  retry of not-ready instructions); J. E. Thornton, CDC 6600 scoreboard (1964).

**(c) Comparison.**
- *Matches.*
  - The page state is a dense array in its own section, not beside the pages: Arrow's validity bitmap, which "does not need
    to be adjacent in memory to the values buffer", and FLUX's signals "allocated contiguously". (Per-entry inline flags —
    NVMe's phase tag, VIRTIO's flags, NCCL's LL lines, FaRM's versions — are the other published placement; NCCL keeps
    them inline because an 8-byte store lands data and flag together, which a whole-page DMA landing does not offer, so
    the separate array follows the bridge's landing by completion and then flip.)
  - A phase that is never reset, compared with an expected value derived from the round: NVMe's phase tag, VIRTIO's wrap
    counter, mlx5's owner bit (wrap parity of the consumer index), PTX `mbarrier` parity, and sense reversal; widened to
    two bits as the operator allowed. PTX states that its parity tests "are valid only for : the current incomplete
    phase, for which waitComplete returns False. the immediately preceding phase, for which waitComplete returns True."
    (PTX ISA 9.5, mbarrier.test_wait); the second bit lets the reader tell one-behind (not yet) from two-behind/ahead
    (error), where FaRM's truncated versions instead rely on a time bound to stay unambiguous.
  - Gray coding: Gray's reflected binary code, so that consecutive phases differ in one bit and the bridge advances a
    page's field by one atomic XOR of one bit, which cannot carry into the neighbouring page's field. Cummings' reason
    (only one bit changes per transition, so a non-atomic sampler sees no multi-bit transient) is the classical one; here
    the word is updated atomically, so the operative benefit is the carry-free single-bit update.
  - Compaction: Iverson's compression, Hillis and Steele's enumerate, Blelloch's prescan, GPU Gems 3 stream compaction;
    the count is `+/u`, written as indirect-dispatch arguments (D3D11/GL 4.3/Metal).
  - Two passes: topology-driven first (every threadgroup tests), data-driven second (the compacted list), as Nasre et al.
    describe; "write only if validation succeeds, else re-run" is Kung and Robinson's optimistic shape.
  - Counting landings to decide when the residue may run: Belli and Hoefler's counting notifications, `mbarrier`
    tx-count, Portals counting events (§4).
- *Deviates, not required — the predicate's granularity.* Every threadgroup tests all P pages of the operand's copy, so
  a threadgroup runs in the first pass only if the whole copy has landed. Every per-element source tests only what the
  consumer reads: an I-structure read waits for "A[i]", MPI_PARRIVED for one partition, a CUDA §4.6 consumer for its
  tile, FLUX's prologue for "the tile with data already arrived". I-structures name the payoff the code forgoes: "the
  consumer of an array can begin work on parts of the array while the producer of the array is still working on other
  parts." No Metal text forces whole-operand tests; a footprint test needs each threadgroup's page set, i.e. analysis of
  the kernel's indexing, which the recorder does not do.
- *Deviates, required — threadgroup granularity.* MSL 4.1 requires that if any thread of a threadgroup executes a
  threadgroup_barrier, all do; the recorded kernels share work through threadgroup memory, so a threadgroup must pass or
  fail as a unit. The per-lane masks of CRAY-1 and RISC-V V would break those kernels. The vote's reduction
  (`simd_ballot`, then threadgroup memory and one barrier) is NCCL LL128's warp vote (`__any_sync`) carried to the
  threadgroup.
- *Deviates — the residue waits (a queue wait), it does not retry in the kernel.* CUDA §4.6 and NCCL retry or spin inside
  the kernel, which their own texts make conditional on the producer's progress ("Producer able to make progress while
  the consumer waits"; Merrill–Garland's "Safety" clause; Sorensen's starvation deadlock). Here the producer is an
  external agent, so that hazard is absent, but MSL has no wait or suspend primitive (headers) and the spec states no
  progress guarantee between threadgroups; spinning would occupy the GPU, and the operator's rule forbids it. With no
  in-kernel option, the queue's `waitForEvent:value:` is the only GPU-side wait in the API (MTL4CommandQueue.h), so the
  coarse wait is required once spinning is excluded. Tera's text names the same escalation: "a heavier weight mechanism
  that avoids busy waiting". Because the later pass tests every page again, a release of the wait without the landing —
  the waiter's final value, or Apple's documented case that "Wait commands that time out in an MTL4CommandQueue unblock
  subsequent work in the queue" (agent d, lit/d-apple-mtlsharedevent/) — ends as a recorded failure, never as a read of
  unlanded pages.
- *Deviates, not required — built by rewriting the reader's MSL source.* No published method adds a readiness prologue
  to a captured kernel by rewriting its source text. It makes the mechanism depend on source, which is fcf5d18a's reason
  for the mixed execution of §9; d0a72fc6's whole later pass (a reader held by the queue wait, no mask) needs none.
- *What is listed.* The compaction lists *threadgroup* indices of the first pass, not page indices; flags, list and
  arguments are values placed by the plan (separate allocations). There is no "list of masked page indices" in the code;
  the page-level state is the phase array.

**(d)** No single published counterpart for the composition "every threadgroup tests every page of a remotely written
operand at entry; the threadgroups that saw a missing page are compacted and run after a queue wait". Each component has
one. Answer to the operator's question in these terms: the first pass *masks* (by a threadgroup-uniform early return),
the residue *waits* (at the queue), and the page state is a dense array in its own section of the allocation.
Verdict: **components match (phase/flag array, Gray code, compaction, indirect dispatch, counted completion);
whole-operand testing deviates (not required); threadgroup granularity is required by MSL; the queue wait is required
once in-kernel waiting is excluded.**

### 4. A host thread that turns completion counters into GPU event values

**Code.** mesh rdma/mesh-metal.m `waits_run` (213–244), started by `mesh_metal_waits_start` (256–286): for t = 0
.. repetitions − 1 and each wait i of one repetition (nondecreasing pass), value = t × passes + pass + 1, target =
(t / copies + 1) × pages, word = count + 8 (t mod copies); before waiting for value v it sets the event to v − 1 if
higher than what it set; it loads the word with acquire; below target it waits with
`os_sync_wait_on_address_with_timeout` (OS_SYNC_WAIT_ON_ADDRESS_SHARED, 8 bytes) for min(10 ms, time left); the bound
(MESH_PLATFORM_NS, 5 s) restarts whenever the word read differs from the last read; at the bound it ends with
ETIMEDOUT; on a stop, ETIMEDOUT or EINVAL it sets the event to the final value repetitions × passes. The bridge's landing
wakes the word when its add completes a round (mesh-flow.c 1475–1476).

**(a) Standard names.** An *eventcount* (a non-decreasing counter; `await(E, v)` until E ≥ v) raised by remote
completions — a *counting event / counting notification* with one threshold per round — observed by a *progress
(proxy) thread* that converts it into a GPU-queue-visible *event value*, because the GPU's front end cannot wait on a
memory word (where CUDA has stream wait-on-value). The thread's primitive is a futex (compare-and-wait on an address);
its bound is a *timeout restarted on progress*, i.e. a failure detector by timeout.

**(b) Sources** (agent d unless marked; scans marked image-verified by the agent).
- 1965 — E. W. Dijkstra, EWD 123 (lit/d-ewd123/), §3.1: "is such, that "sleeping" is a less expensive activity than this
  busy way of waiting"; §4.1: "The V-operation finally confirms its presence, i.e. signals it to the consumer. Note, that it
  is absolutely essential, that the V-operation is preceded by the complete addition of the portion."
- 1979 — D. P. Reed and R. K. Kanodia, "Synchronization with Eventcounts and Sequencers", CACM 22(2):115–123, 1979
  (https://www.cs.uml.edu/~bill/cs515/Eventcounts_Sequencers_Reed_Kanodia_79.pdf; lit/d-reed-kanodia1979/,
  image-verified), p. 116: "An eventcount can be thought of as a nondecreasing integer variable." "we define a primitive
  await(E, v) that suspends the calling process until the "value" of the eventcount E is at least v." "there may be a
  communications delay between the time an advance operation is finished and the time that another processor can
  determine that an advance has occurred." p. 117: "stores the ith value in buffer[i mod N]"; consumer `await(IN, i);
  consume(buffer[i mod N]); advance(OUT);`. Footnote 4: "In this paper, we ignore questions of liveness, e.g. we have no
  axiom that says when an await will eventually complete." p. 121: "In any case, implementing await in terms of a loop
  using read is trivial."
- 1992 — T. von Eicken, D. E. Culler, S. C. Goldstein, K. E. Schauser, "Active Messages", ISCA 1992, pp. 256–266
  (lit/d-activemessages1992/), p. 260: "The most common versions of PUT and GET increment a separately specified flag on
  the processor that receives the data. This allows simple synchronization through checking the flag or busy-waiting."
- 1993/1995 — Split-C `store_sync`, read through J. Demmel's CS267 notes, Lecture 8, 9 Feb 1995 (lit/d-cs267-1995/): "The
  simplest way is to call store_sync(n), which waits until n bytes have been stored in the memory of the processor
  executing store_sync(n). This presumes the parallel algorithm is designed so that one knows how much data to expect."
  (The SC'93 paper itself: **not read**.)
- 1996 — T. D. Chandra and S. Toueg, "Unreliable Failure Detectors for Reliable Distributed Systems", JACM 43(2):225–267,
  1996 (lit/d-chandra-toueg1996/), p. 229: "most implementations of failure detectors are based on time-out mechanisms."
  p. 227: "it can suspect that a process p has crashed even though p is still running."
- 2002 — H. Franke, R. Russell, M. Kirkwood, "Fuss, Futexes and Furwocks: Fast Userlevel Locking in Linux", OLS 2002,
  pp. 479–495 (lit/d-futex2002/), p. 487: "5. If the value read at step 3 is not equal to the “val” argument provided to
  the system call, set the return to EWOULDBLOCK." "6. Otherwise, sleep for the time indicated by the “reltime” argument,
  or indefinitely if that is NULL." "(a) If we timed out, set the return value to ETIMEDOUT."
- 2008 — T. Hoefler and A. Lumsdaine, "Message Progression in Parallel Computing – To Thread or not to Thread?", IEEE
  Cluster 2008 (preprint; lit/d-hoefler2008/), p. 3: "A progress thread also enables fully asynchronous progression, i.e.,
  without any user interaction." p. 4: "the progression thread goes to sleep (enters the wait queue) when no work is to be
  done and is woken up (enters the run queue) when specific network events (e.g., a packet is received) occur."
- 2011 — RFC 6298, "Computing TCP's Retransmission Timer", §5 (lit/d-rfc6298/): "(5.3) When an ACK is received that
  acknowledges new data, restart the retransmission timer so that it will expire after RTO seconds (for the current value
  of RTO)."
- 2015 — R. Belli and T. Hoefler, "Notified Access", IPDPS 2015 (mine; lit/f-belli-hoefler2015/): "we introduce counting
  notifications that only notify after n matching accesses were performed."
- 2017 — B. W. Barrett et al., "The Portals 4.1 Network Programming Interface", SAND2017-3825 (lit/d-portals41/), §3.14.6,
  p. 87: "The PtlCTWait() function provides blocking semantics to wait for a counting event to reach a given value.
  PtlCTWait() returns when either the success field of a counting event is greater than or equal to the test value or
  when the failure field is non-zero." §3.14.7: "PtlCTPoll() provides a timeout to allow applications to poll, block for
  a fixed period, or block indefinitely." §3.16, p. 101: "When the success field of the count ... reaches or exceeds the
  threshold (equal to or greater), the operation proceeds at the initiator of the operation."
- 2016–2020 — libgdsync (GPUDirect Async; github.com/gpudirect/libgdsync @ f9e63388; BSD-3-style; lit/d-libgdsync/,
  lit/d-libgdsync-src/): README, "GPUDirect Async is all about moving control logic from third-party devices to the GPU.";
  gdsync.cpp 601/606: `param.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;` `param.waitValue.flags =
  CU_STREAM_WAIT_VALUE_GEQ;` (the GPU stream waits on the NIC's completion word). (Agostini et al., JPDC 2018: **not
  read**.)
- 2026 — NVIDIA, CUDA Driver API 13.5, §6.38 Stream Memory Operations (lit/d-cuda-memop/): cuStreamWaitValue64, "Enqueues a
  synchronization of the stream on the given memory location. Work ordered after the operation will block until the given
  condition on the memory is satisfied. By default, the condition is to wait for (int64_t)(*addr - value) >= 0, a cyclic
  greater-or-equal." §6.5, CU_STREAM_WAIT_VALUE_FLUSH: "Follow the wait operation with a flush of outstanding remote
  writes."
- libfabric fi_cntr(3) (lit/d-fi-cntr/): "Counters record the number of requested operations that have completed."
  fi_cntr_wait: "This call may be used to wait until the counter reaches the specified threshold, or until an error or
  timeout occurs." "If the call returns due to timeout, -FI_ETIMEDOUT will be returned."
- 2020 — OpenSHMEM 1.5, §9.10.15 (lit/f-openshmem15/, mine): "shmem_signal_wait_until operation blocks until the value
  contained in the signal data object, sig_addr, at the calling PE satisfies the wait condition."
- 2025 — K. Hamidouche et al., "GPU-Initiated Networking for NCCL", arXiv 2511.15076 (lit/d-nccl-gin2025/), p. 7: "A
  dedicated CPU proxy thread per communicator ... continuously polls these queues." "the proxy thread polls on
  completions using the network plugin’s test interface, matches completed operations to their associated GIN counters,
  and updates completion state in GPU-visible memory". NVSHMEM 3.8 docs (lit/d-nvshmem-latest/): "NVSHMEM_IBGDA_NIC_HANDLER
  ... cpu: Use CPU proxy thread."
- Apple Metal documentation (DocC JSON; lit/d-apple-mtlsharedevent/, lit/d-apple-mtl4-waitforevent/): MTLEvent, "Each
  event represents an unsigned 64-bit integer that starts with a value of 0, which can only increase over time."
  "Multiple producing workloads can’t combine their signals with one event to unblock any dependent workloads. This is
  because an event’s signal method can only increase its value to a specific number, unlike a semaphore that can
  increment or decrement its current value by one." "Wait commands that time out in an MTL4CommandQueue unblock subsequent
  work in the queue, and wait commands that time out in an MTLCommandBuffer terminate the command buffer with the
  MTLCommandBufferError.Code.timeout error code." signaledValue: "When you set the value of a shared event, its value is
  changed only if you provide a larger value than the value currently stored in the event."
- Apple os documentation and SDK header (lit/d-apple-os-sync-wait/; usr/include/os/os_sync_wait_on_address.h, mine):
  "Low-level Darwin's futex style APIs." os_sync_wait_on_address_with_timeout: "This function reads a value from addr,
  compares it to the expected value, and blocks the current thread for up to timeout_ns nanoseconds if the two are equal."
  OS_SYNC_WAIT_ON_ADDRESS_SHARED: "Flag to indicate an address is in a shared memory region, allowing for a futex wake from
  another process."
- Metal SDK 26.4 headers (mine): MTL4CommandQueue.h 269–275, `waitForEvent:value:`, "Schedules an operation to wait for a
  GPU event of a specific value before continuing to execute any future GPU work."; MTLEvent.h, MTLSharedEventHandle
  "may be passed between processes via XPC connections and then used to recreate a MTLSharedEvent via an existing
  MTLDevice." The waits in MTL4CommandQueue.h, MTL4CommandBuffer.h, MTLCommandBuffer.h, MTL4CommandEncoder.h and
  MTL4ComputeCommandEncoder.h are event, fence, stage-barrier, CPU-side and drawable waits; none is on a memory location.

**(c) Comparison.**
- *Matches.* The count word is Reed and Kanodia's eventcount (never reset, one advance per landed page; copy t mod u is
  their `buffer[i mod N]`); the waiter's test, count ≥ (t/u + 1) × P, is `await(E, v)`, Portals' `PtlCTWait` (≥ test) and
  OpenSHMEM's `signal_wait_until`; the per-round threshold is Belli and Hoefler's counting notification, Split-C's
  `store_sync(n)`. A host thread folding completions into GPU-visible state is the CPU proxy thread of NCCL's GIN proxy
  backend and NVSHMEM; it sleeps on a futex as Hoefler and Lumsdaine's progress thread does. The bound restarted at each
  rise of the count is RFC 6298's rule 5.3; ending with ETIMEDOUT is a timeout failure detector (Chandra and Toueg), and
  a suspicion, which may be wrong for a slow peer.
- *Deviates, required.* That a CPU thread does this at all: CUDA lets the GPU's stream wait on the memory word itself
  (cuStreamWaitValue64, used by libgdsync for NIC completions), and Metal's headers offer only event waits on the queue.
  One monotone value per (repetition, pass) is how Apple's documentation says to signal through an event ("Multiple
  producing workloads can’t combine their signals with one event"), so folding P landings into one value needs an agent
  that counts. What is *not* required is that the agent be a thread of the consuming program: MTLEvent.h lets a shared
  event be recreated in another process (the bridge, which already counts and wakes), at the cost of the bridge knowing
  the GPU's events.
- *Deviates from the sources' failure handling, not required.* Portals and libfabric carry failure in a separate field
  that releases the waiter; Chandra and Toueg lengthen the timeout after a wrong suspicion; TCP backs off and retries.
  The waiter instead sets the event to its final value, so every held pass runs and finds its pages behind. Apple's
  documentation adds a case the code's notes do not mention: an MTL4CommandQueue wait that times out "unblock[s]
  subsequent work in the queue" by itself. The later pass re-tests every page and records a failure in the status, so a
  Metal-side timeout yields an error, not a read of unlanded data; the outcome is the same as the waiter's final value.
- *Small differences.* The futex test is equality, so the loop re-reads and re-waits on each new value (as the header
  instructs); a count raised without a wake is seen at the next 10 ms slice, comparable to the OS time slice Hoefler and
  Lumsdaine cite. CUDA's comparison is cyclic; the code's 64-bit count never wraps in practice.

**(d)** Published counterparts exist (eventcounts, counting events, proxy threads). Verdict: **matches eventcount await /
counting-event threshold / proxy-thread designs; the CPU agent is required by Metal (no queue wait on memory), its
placement in the consumer process is not; the failure path deviates from the sources (not required) and is consistent
with Metal's own wait timeout.**


### 5. The sender's side: a fence, then an atomic add to the supplied word

**Code.** metal_recording.m `MetalRelease` (346–351), used by `mesh_publish` (3325–3332), the
release folded into a one-thread producer's end or a reader's entry (`MetalEntryRelease`, 361–363), and a direct
publication's last arrival (`MetalCoherentStores`, 505–542): `atomic_thread_fence(mem_device, memory_order_seq_cst,
static_cast<thread_scope>(3))`; if the node's status words `behind | ahead` read 0, `atomic_fetch_add_explicit(word, 1,
memory_order_relaxed)` on a 32-bit `atomic_uint` at the 64-bit word's low half; the fence again. `published:operand:supplied:`
(3492–3494) records the word's offset. The bridge releases execution t's sends once a 64-bit acquire load of the word
reads at least t + 1 (mesh-flow.c 1650–1651).

**(a) Standard names.** Release-then-flag publication (the message-passing pattern): the payload's stores, a release
fence, then a store or atomic increment of a flag the consumer reads with acquire; the flag here is a single-writer,
monotone counter (an eventcount `advance`, OpenSHMEM's signal-add, NCCL's step counter); skipping it when the node's
status shows a failure is fail-stop.

**(b) Sources** (agent d unless marked).
- 1965 — Dijkstra, EWD 123, §4.1 (above): the signal must follow "the complete addition of the portion".
- 1979 — L. Lamport, "How to Make a Multiprocessor Computer That Correctly Executes Multiprocess Programs", IEEE Trans.
  Computers C-28(9):690–691, 1979 (lit/d-lamport1979/, image-verified), p. 690: "the result of any execution is the same as
  if the operations of all the processors were executed in some sequential order, and the operations of each individual
  processor appear in this sequence in the order specified by its program."
- 1983 — R. D. Schlichting and F. B. Schneider, "Fail-Stop Processors", ACM TOCS 1(3):222–238, 1983
  (lit/d-schlichting1983/, image-verified), p. 222: "Such a processor automatically halts in response to any internal
  failure and does so before the effects of that failure become visible."
- 1995/1996 — S. V. Adve and K. Gharachorloo, "Shared Memory Consistency Models: A Tutorial", DEC WRL 95/7 (IEEE Computer
  29(12), 1996) (lit/d-adve-wrl95-7/), §5.1.2: "it is possible for another processor to observe the new value of Head and
  yet obtain the old value of Data." §6.4.2: "A release is a write operation that is performed to grant permission for
  accessing a set of shared locations (e.g., an unlock operation or setting of a flag)."
- 2008 — H.-J. Boehm and S. V. Adve, PLDI 2008 (mine; lit/f-boehm-adve2008/): data-race-free models "guarantee sequential
  consistency to such data-race-free programs, and do not provide any guarantees whatsoever to others."
- 2011 — ISO/IEC 9899:201x (C11), N1570 (lit/d-n1570/), §7.17.4 ¶3, p. 278: "A release fence A synchronizes with an
  atomic operation B that performs an acquire operation on an atomic object M if there exists an atomic operation X such
  that A is sequenced before X, X modifies M, and B reads the value written by X or a value written by any side effect in
  the hypothetical release sequence X would head if it were a release operation." §7.17.4.1: a fence "is a sequentially
  consistent acquire and release fence, if order == memory_order_seq_cst."
- 2014 — J. Alglave, L. Maranget, M. Tautschnig, "Herding Cats", TOPLAS 36(2), 2014 (arXiv 1308.6810; lit/d-herdingcats/),
  §4.5.1: "the write to x needs to be propagated to the reading thread before the write to y." "TSO guarantees it, but
  Power or ARM need at least a lightweight fence (e.g. lwsync on Power) between the writes."
- 2017 — S. Flur et al., "Mixed-Size Concurrency: ARM, POWER, C/C++11, and SC", POPL 2017 (lit/d-flur-mixed/), p. 2: "the
  effective type rules of the ISO C standard prohibit mixed-size use of atomics." §8: "supporting mixed-size atomics and
  mixtures of atomic and non-atomic accesses represents another open problem for the design of C/C++11 models."
- PTX ISA 9.5, §8 memory consistency model (lit/d-ptx-isa/): §8.8, a release pattern is "a release or acquire-release
  memory fence followed by a strong write on M in program order — E.g.: fence.release; st.relaxed [M]; or fence.release;
  atom.relaxed [M];" §8.7.2: "The axioms in the memory consistency model do not apply if a PTX program contains one or
  more mixed-size data-races."
- NCCL v2.32.3-1 (Apache-2.0; lit/d-nccl-v2.32.3-1/), prims_simple.h 170–173: `fence_acq_rel_sys();` then
  `st_relaxed_sys_global(connStepPtr, step);`; op128.h 380/395: `st.relaxed.sys.global.u64` and `fence.acq_rel.sys;`.
- 2020 — OpenSHMEM 1.5, §9.8.1–9.8.2 (mine): "All signaling operations ... are performed on a signal data object, a
  remotely accessible symmetric object of type uint64_t." "SHMEM_SIGNAL_ADD An update to signal data object is an atomic
  add operation. It adds an unsigned 64-bit value as a signal into the signal data object on a remote PE as an atomic
  operation."
- 2026 — Apple, Metal Shading Language Specification 4.1, 2026-06-04 (lit/d-msl/, lit/c-msl/): §6.16.1: "All OS: Metal 4.1
  and later support the additional memory order values, memory_order_acquire, memory_order_release, and
  memory_order_acq_rel." "For atomic operations other than atomic_thread_fence, memory_order_relaxed is the only
  enumeration value." §6.16.2: `enum thread_scope { thread_scope_thread, thread_scope_simdgroup, thread_scope_threadgroup,
  thread_scope_device }`. §6.16.4.5: fetch-and-modify functions "are applicable to any atomic object of type atomic_int,
  and atomic_uint." §6.16.4.6: for `atomic_ulong`, max and min only. Installed headers (GPUCompiler 32023.864 and Metal
  toolchain 32023.883, mine): `enum memory_order { memory_order_relaxed, memory_order_seq_cst }`, the same four scopes.
- 2026 — CUDA Programming Guide v13.5 (mine): appendix 5.8.1.1, "Each thread in the system is related to each other thread
  in the system by the system thread scope: cuda::thread_scope_system."; §4.6.2, "The producer writes a tile, executes a
  device-scope release fence so that the tile is visible before the flag (see Section 3.2.3), and then writes the flag."

**(c) Comparison.**
- *Matches* the message-passing publication: payload stores, then a fence, then a relaxed atomic modification of the flag,
  which C11 §7.17.4 and PTX §8.8 make a release (a seq_cst fence "is a sequentially consistent acquire and release
  fence"); the bridge's 64-bit acquire load is the acquire. The flag being a single-writer monotone counter is NCCL's step
  counter and OpenSHMEM's signal word; one release per repetition after the last writer (d0a72fc6 voids an earlier
  writer's releases) is Dijkstra's "preceded by the complete addition of the portion". Skipping the add when the status
  shows failure is fail-stop: halt "before the effects of that failure become visible", the peers learning of it by
  timeout (§4).
- *Deviates — seq_cst where release suffices.* Message passing needs only a release fence (Herding Cats; NCCL's
  `fence.acq_rel.sys`). The installed MSL headers offer seq_cst and relaxed only, so on this toolchain seq_cst is
  required; MSL 4.1 adds `memory_order_release` for fences, so with a 4.1 toolchain it is not.
- *Deviates, not required — a 32-bit add on the low half of a 64-bit word that the bridge reads with a 64-bit load.* MSL
  4.1 has no 64-bit fetch-and-add, so a GPU-side add must be 32-bit; but nothing requires the word to be 64 bits: a 32-bit
  word read with a 32-bit acquire load (as `os_sync_wait_on_address` itself allows 4-byte values) avoids the mixed-size
  access. As written, the access is outside C11 ("the effective type rules of the ISO C standard prohibit mixed-size use
  of atomics", Flur et al.) and, in PTX's model, a mixed-size race that voids the axioms. OpenSHMEM's signal object is a
  64-bit atomic updated by a 64-bit add; NCCL's counter is written by one 64-bit store. No published model covers the
  code's form; it relies on single-copy atomicity of aligned accesses on this hardware.
- *Deviates, not required by any public text — the fence's scope.* The code fences at `static_cast<thread_scope>(3)`
  (with `#pragma METAL internals`) and declares `coherent(system)` pointers; MSL 4.1's widest scope is
  `thread_scope_device` and the specification defines no system or host scope. CUDA defines `thread_scope_system` and the
  conditions under which system-scope atomics are atomic; Metal's public text has no counterpart, so visibility of the
  GPU's stores to the bridge's CPU loads rests on unpublished behaviour.

**(d)** Published counterparts exist. Verdict: **matches release-then-flag (and fail-stop); seq_cst is required only by
the installed toolchain; the mixed-size 32/64-bit access and the undocumented "system" scope are not required and have
no published model.**

### 6. Starting each execution's persistent communication `lead` executions ahead; HOST_LAG = 2

**Code.** metal_recording.m `MetalRecorderRun` (4055–4094): wait until the end event reads at least
t − lag + 1, reset allocator t mod lag, then for each call `start(context, plan, t + 1)` (mesh `mesh_transfers_start`,
mesh-dataflow.c 710–741: waits for the plan's synchronization edges into this call's place, then appends a submission),
then `[ended notifyListener:… atValue:(t ≥ lead ? t − lead : 0) block:^{ started(context, plan, 0, t); }]` (mesh
`mesh_host_started`, 751–781: raises the plan's `executions` to t + 1, the bridge serving executions below it), then
`MetalReplayEncode` of repetition t, then `signalEvent:ended value:t + 1`. The listener at value t − lead runs once
execution t − lead − 1 has ended. tools/e2b_replay.m: `#define HOST_LAG 2u` (77) sizes the input copies, the step word's
copies and the command allocators ("With the host writing step t + 1 while the GPU runs step t, the lag is 2", 74–76);
`ir.function.lead=1` (1411); its own loop uses the same listener at origin + t − lead (2058–2067). rdma/mesh_compile.py
`_replayed` passes lag HOST_LAG = 2 and lead 0 (3312–3318).

**(a) Standard names.** Persistent requests started per iteration (MPI_START), with receives enabled ahead of their
sends (prepost; credit return); in dataflow scheduling, a synchronization edge with a delay in a self-timed schedule,
whose delays bound the buffers; in software pipelining, the prefetch distance and the number of rotating buffers.
HOST_LAG = 2 is double buffering.

**(b) Sources** (agent b unless marked; MPI 4.1, BSL, CUDA §4.6, NCCL, Metal are mine).
- 1961 — J. D. C. Little, "A Proof for the Queuing Formula: L = λW", Operations Research 9(3):383–387, 1961
  (lit/b-little1961/, page image): "L be the mean number of units in the system, and W be the mean time spent by a unit
  in the system ... then L=λW".
- 1965 — Dijkstra, EWD 123 (above), §4.1: "Such irregularities in production speed, however, can be smoothed out by a
  buffer of sufficient size and that is, why such a queuing device is called "a buffer"."
- 1988 — Lam, PLDI 1988, §1, p. 318: "In software pipelining, iterations of a loop in a source program are continuously
  initiated at constant intervals without having to wait for preceding iterations to complete. That is, multiple
  iterations, in different stages of their computations, are in progress simultaneously." §2.3: "at least ⌈l/s⌉ number
  of values must be kept alive concurrently, in that many locations."
- 1992 — T. C. Mowry, M. S. Lam, A. Gupta, "Design and Evaluation of a Compiler Algorithm for Prefetching", ASPLOS-V,
  pp. 62–73 (lit/b-mowry1992/), §2.3, p. 66: "Prefetches must be issued early enough to hide memory latency. They must
  not be issued too early lest the data fetched be flushed out of the cache before they are used. ... The number of
  iterations to prefetch ahead is ⌈l/s⌉ where l is the prefetch latency and s is the length of the shortest path through
  the loop body."
- 1994 — H. T. Kung, T. Blackwell, A. Chapman, "Credit-Based Flow Control for ATM Networks", SIGCOMM 1994
  (lit/b-kung1994/, page images), p. 104: "to prevent data and credit underflow, it suffices to choose N3 to be: N3 = B_VC
  · RTT". 1995 — H. T. Kung and R. Morris, IEEE Network 9(2):40–48 (lit/b-kungmorris1995/), p. 6: "if traffic is 100
  percent steady state, M need only be RTT for sustaining the peak bandwidth of the output link. However, for bursty
  traffic, we need M to be larger than RTT".
- 1995 — MPI Forum, MPI-1.1, §3.9 "Persistent communication requests" (https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node51.html;
  lit/b-mpi11/): "binding the list of communication arguments to a persistent communication request once and, then,
  repeatedly using the request to initiate and complete messages." MPI_START: "The request becomes active once the call
  is made." "The request is not deallocated and it can be activated anew by an MPI_START or MPI_STARTALL call."
- 1997 — S. S. Bhattacharyya, S. Sriram, E. A. Lee, "Optimizing Synchronization in Multiprocessor DSP Systems", IEEE
  Trans. Signal Processing 45(6), 1997 (preprint 11 Feb 1997, https://ptolemy.berkeley.edu/papers/97/synchronization/synch.pdf;
  lit/f-bsl1997/): the self-timed strategy of Lee and Ha: "retain the processor assignment and actor ordering specified by
  the fully static schedule. This results in the self-timed scheduling strategy of [17]. Each processor executes the
  actors assigned to it in the order specified at compile time. Before firing an actor, a processor waits for the data
  needed by that actor to become available." Eq. (1): "start(v_i, k) ≥ end(v_j, k – delay((v_j, v_i)))". Definition 2:
  λ_max is the maximum over cycles C of Σ_{v on C} t(v) / Delay(C), and "the throughput of the system of processors
  executing a particular self-timed schedule is equal to the corresponding 1/λ_max value." Eq. (5): "Bfb(e) = min({Delay(C)
  | C is a cycle that contains e})".
- 2017/2018 — Apple, Metal Best Practices Guide, "Triple Buffering" (lit/b-apple-triple/): "For the processors to work in
  parallel, the CPU should be working at least one frame ahead of the GPU. This solution requires multiple instances of
  dynamic buffer data, so the CPU can write the data for frame n+1 while the GPU reads the data for frame n." "This
  sequence can be parallelized with two dynamic data buffers, but the command buffer transactions may cause the CPU to
  stall or the GPU to starve if either processor is waiting on a busy dynamic data buffer." "Adding a third dynamic data
  buffer is the ideal solution when considering processor idle time, memory overhead, and frame latency."
- 2010s — A. Overvoorde, Vulkan Tutorial, "Frames in flight" (CC BY-SA 4.0; lit/b-vulkan-tutorial/): "We choose the
  number 2 because we don't want the CPU to get too far ahead of the GPU. ... With 3 or more frames in flight, the CPU
  could get ahead of the GPU, adding frames of latency."
- 2023 — MPI Forum, MPI: A Message-Passing Interface Standard, Version 4.1, 2 Nov 2023, §3.9 "Persistent Communication
  Requests" (PDF https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf, lit/d-mpi41/; the HTML rendering the code
  links, node83.htm, lit/mpi41-node83/, labels the same section "4.9"): MPI_START, "If the request is for a ready mode
  send operation, then a matching receive operation should be started before the call is made. The communication buffer
  should not be modified after the call, and until the operation completes."
- 2026 — CUDA Programming Guide v13.5 §4.6.4.5 (lit/f-cuda-fine-grained-synchronization/): "Rotating between two or three
  buffers keeps the clear off the critical path, because one buffer can be cleared while the others are written and
  read." NCCL 2.32.3, "CUDA Stream Semantics" (lit/f-nccl-streams/): "NCCL calls are associated to a stream which is passed
  as the last argument of the collective communication function. The NCCL call returns when the operation has been
  effectively enqueued to the given stream, or returns an error."
- Metal SDK 26.4 (mine): MTLEvent.h `notifyListener:atValue:block:`, "When the event's signaled value reaches value or
  higher, invoke the block on the dispatch queue owned by the listener."; MTL4CommandAllocator.h, "You are responsible to
  ensure that all command buffers with memory originating from this allocator instance are complete before calling
  resetting it."

**(c) Comparison.**
- *The mechanism matches* MPI_START on a persistent request (`mesh_transfers_start` submits; `mesh_host_started` makes
  execution t's receives active) and is a BSL synchronization edge from "execution t − lead − 1 ended" to "execution t's
  transfers start", delay lead + 1; starting receives before the matching sends is what MPI's ready mode requires, and the
  credit view (Kung) reads the same edge as credit returned when a copy's last reader ends. Allocator reuse after t − lag
  ended is required by MTL4CommandAllocator.h; one allocator per execution in flight is BSL's bounded-buffer
  synchronization with Bfb = lag.
- *The numbers are not derived.* Every published rule sizes them from times: Little's L = λW, Lam's ⌈l/s⌉, Rau's Kmin,
  Mowry's ⌈l/s⌉ prefetch distance, Kung's N3 = B_VC · RTT, BSL's λ_max ≤ the target period (on the cycle host write → GPU
  step → step end → host write, the delay is HOST_LAG, so HOST_LAG = 2 keeps the host off the critical cycle exactly when
  the host's write-and-encode time plus the cycle's latencies is at most one GPU step). The code fixes HOST_LAG = 2 with
  "the host writes step t + 1 while the GPU runs step t" — Apple's own sentence for the double-buffered case, after which
  Apple warns that two buffers "may cause the CPU to stall or the GPU to starve" and recommends three; the only source
  choosing 2 (the Vulkan Tutorial) chooses it to bound latency. `lead` is a caller constant (e2b 1, mesh_compile 0)
  cited to Lam §1, which gives no number. Kung and Morris add that bursty arrivals need more than the steady-state count.
  Nothing in Metal requires 2 or 1.
- *The start runs in a `notifyListener` block.* The raise of `executions` is a store to the mapped region that a GPU
  command could make (the supplied word is raised that way, §5); the CPU is needed for the rest of `mesh_host_started`:
  the futex wake of the bridge and the view check with its cancellation. CUDA's ordering for the same purpose is the
  stream itself (NCCL's stream semantics). The operator's "callback driven programming (forbidden)" (PB, the turns) is
  context.

**(d)** Published counterparts exist for the mechanism and for choosing its numbers. Verdict: **mechanism matches
(MPI_START; BSL delay edge; credits); the constants `lead` and HOST_LAG deviate from every published sizing rule — no API
requires them, and the rules need measured times the code never reads.**

### 7. A virtually contiguous view of scattered pages

**Code.** mesh rdma/mesh-link.c creates the region: `shm_open(MESH_NAME, O_CREAT|O_EXCL|O_RDWR, …)`,
`ftruncate`, `mmap(MAP_SHARED)` (296–299). A program attaches with `shm_open(name, O_RDWR)` and `mmap(…, MAP_SHARED, …)`
(mesh-dataflow.c 145–165). `mesh_view_values` (260–284) lists every `run` value's rows in order and calls
`mesh_view_compose` (208–237): `mmap(NULL, count × block, PROT_NONE, MAP_PRIVATE|MAP_ANON)`, then, for each run of rows
whose blocks are consecutive in the region, `mach_vm_remap(self, &at, run × block, 0, VM_FLAGS_FIXED|VM_FLAGS_OVERWRITE,
self, region + row address, FALSE, &cur, &max, VM_INHERIT_NONE)`. tools/e2b_replay.m (1506) and rdma/mesh_compile.py
`_placed_buffer` (2539–2545) wrap the whole view in one `newBufferWithBytesNoCopy:length:options:deallocator:` buffer,
shared storage, no deallocator.

**(a) Standard names.** *Artificial contiguity* (Randell and Kuehner's term, repeated by Denning): an address map that
presents separate blocks as one run of names; here built in user space by *reserving a virtual range and remapping
shared-memory blocks under it* (the reserve-then-overwrite pattern), with deliberate *synonyms* (two virtual addresses
of one physical page) inside one address space. Its GPU-side relatives are sparse/tiled resources and CUDA's virtual
memory management (reserve, create, map), "virtual memory stitching" (GMLake) and vAttention; the software alternative
is a block table read by the kernel (PagedAttention).

**(b) Sources** (agent e unless marked; every quote checked against the download).
- 1962 — T. Kilburn, D. B. G. Edwards, M. J. Lanigan, F. H. Sumner, "One-level storage system", IRE Trans. Electronic
  Computers EC-11(2):223–235, 1962; read in the reprint in Bell and Newell, *Computer Structures* (1971), ch. 23
  (gordonbell.azurewebsites.net; lit/e-kilburn1962/), p. 279: "Associated with each of these core store page positions
  is a "page address register" (P.A.R.) which contains the address of the block of information at present occupying that
  page position." p. 280: "any block of information can be stored in any absolute position in either core or drum store."
- 1968 — B. Randell and C. J. Kuehner, "Dynamic storage allocation systems", CACM 11(5):297–306, 1968
  (https://eprints.ncl.ac.uk/158029; lit/e-randell-kuehner1968/), "Artificial Contiguity", pp. 299–300: "However in several
  recent computer systems, mechanisms (usually part-hardware, part-software) have been provided to give name contiguity
  without the necessity for a complete address contiguity. ... A set of separate blocks of locations, whose absolute
  addresses are contiguous, can then be made to correspond to a single set of contiguous names".
- 1968 — R. C. Daley and J. B. Dennis, "Virtual memory, processes, and sharing in MULTICS", CACM 11(5):306–312, 1968
  (https://multicians.org/daley-dennis.html; lit/e-daley-dennis1968/): "An immediate consequence is that the same segment
  will, in general, be identified by different segment numbers in different processes."
- 1970 — P. J. Denning, "Virtual memory", Computing Surveys 2(3):153–189, 1970 (lit/e-denning1970/), p. 158: "by virtue
  of the mapping f, the programmer may be given the illusion that items consecutive in N are stored consecutively in M,
  even though the items may in fact be stored in arbitrary order. This property of address maps is known as "artificial
  contiguity" [R3]."
- 1987 — R. Rashid et al., "Machine-independent virtual memory management for paged uniprocessor and multiprocessor
  architectures", ASPLOS II, 1987 (https://www.cs.columbia.edu/~nieh/teaching/e6118_s00/papers/rashid_mivmm87.pdf;
  lit/e-rashid1987/), p. 3: "Both copy-on-write and read/write sharing of memory are permitted between Mach tasks."
  p. 5: "An address map is a doubly linked list of address map entries each of which maps a contiguous range of virtual
  addresses onto a contiguous area of a memory object." p. 10, on the IBM RT: "it allows only one valid mapping for each
  physical page, making it impossible to share pages without triggering faults."
- 1983/1993/2024 — 4.4BSD System Manual (PSD:5), §2.2.2 (lit/e-44bsd-sysman/): "A successful mmap will delete any previous
  mapping in the allocated address range." POSIX (IEEE Std 1003.1-2024; lit/e-posix-shm_open/, lit/e-posix-mmap/),
  shm_open: "If name begins with the <slash> character, then processes calling shm_open() with the same value of name
  refer to the same shared memory object"; mmap: "If MAP_SHARED is specified, write references shall change the
  underlying object." "shall not permit any access where PROT_NONE alone has been set."
- 1997 — M. Cekleov and M. Dubois, "Virtual-address caches, Part 1: problems and solutions in uniprocessors", IEEE
  Micro 17(5):64–71, 1997 (lit/e-cekleov-dubois1997/), p. 66: "When different virtual addresses map to the same physical
  address, we say that they are synonyms. A programmer may also define synonyms in the same context, for convenience. As
  shown later, synonyms are at the root of many problems in virtual-address caches." p. 69: "the software invalidates
  all or part of the cache when needed so that synonyms are never present in the cache at the same time."
- XNU (apple-oss-distributions/xnu at f6217f89, xnu-12377.1.9; APSL 2.0; lit/e-xnu-vm_remap/, lit/e-xnu-posix_shm/):
  osfmk/man/vm_remap.html: "Copy indicator. If true, the kernel copies the region for the memory to the specified task's
  address space. If false, the region is mapped read-write." osfmk/vm/vm_map.c 18303–18307: "If we're sharing this memory
  (copy=false), we'll share this VM object." bsd/kern/posix_shm.c (the kernel's own mmap of a POSIX shm object):
  "/* reserve the entire space first... */" with `VM_PROT_NONE, VM_PROT_NONE, VM_INHERIT_NONE`, then "/* Now overwrite
  with the real mappings. */" with `VM_MAP_KERNEL_FLAGS_FIXED(.vmf_overwrite = true)`.
- 2002 — Linux `remap_file_pages(2)` (man-pages 6.19; lit/e-linux-remap_file_pages/): "used to create a nonlinear
  mapping, that is, a mapping in which the pages of the file are mapped into a nonsequential order in memory. The
  advantage of using remap_file_pages() over using repeated calls to mmap(2) is that the former approach does not require
  the kernel to create additional VMA (Virtual Memory Area) data structures." "It is possible to map the same page of a
  file into multiple locations within the mapped region."
- 2011– — TPCircularBuffer (github.com/michaeltyson/TPCircularBuffer at cc520397; zlib-style license in README;
  lit/e-tpcircularbuffer/), TPCircularBuffer.h: "This implementation makes use of a virtual memory mapping technique that
  inserts a virtual copy of the buffer memory directly after the buffer's end, negating the need for any buffer
  wrap-around logic."; .c: `vm_remap(mach_task_self(), &virtualAddress, ... mach_task_self(), bufferAddress, 0, // MAP
  READ-WRITE, NOT COPY`.
- 2013 — Microsoft, Direct3D 11.2 tiled resources (lit/e-d3d11-tiled/): "A tile pool is a pool of memory (backed by one
  or more allocations behind the scenes - unseen by the application)." UpdateTileMappings: "Updates mappings of tile
  locations in tiled resources to memory locations in a tile pool."
- 2016– — Vulkan spec, sparse resources (Vulkan-Docs fa30d4e3; CC-BY-4.0; lit/e-vulkan-sparse/): "Sparse resources can
  be bound non-contiguously to one or more VkDeviceMemory allocations." "Memory dependencies must be used to separate
  writes to one alias from reads or writes to another alias." memory.adoc: "Importing memory from a host pointer shares
  ownership of the memory between the host and the Vulkan implementation."
- 2019 — B. Powers, D. Tench, E. D. Berger, A. McGregor, "Mesh: Compacting Memory Management for C/C++ Applications",
  arXiv 1902.04738 (PLDI 2019; lit/e-mesh2019/), §4.5.1: "We exploit the fact that mmap lets the same offset in a file
  (corresponding to a physical span) be mapped to multiple addresses."
- 2019/2020 — NVIDIA, CUDA Driver API §6.48 Virtual Memory Management (lit/e-cuda-vmm-driverapi/): cuMemAddressReserve
  "Reserves a virtual address range based on the given parameters"; cuMemCreate "The created allocation will not have any
  device or host mappings."; cuMemMap "This range must be an address reservation previously reserved with
  cuMemAddressReserve". C. Perry and N. Sakharnykh, "Introducing Low-Level GPU Virtual Memory Management", NVIDIA blog,
  15 Apr 2020 (lit/e-nvidia-blog-vmm-2020/): "What you really want is to grow the allocation as you need more memory, yet
  maintain the contiguous address range that you always had." "This is similar to how VirtualAlloc or mmap work."
- 2023 — W. Kwon et al., PagedAttention, SOSP 2023 (arXiv 2309.06180; lit/e-pagedattention/), p. 6: "block tables—the
  mapping between logical and physical KV blocks of each request." p. 12: the kernel's "extra overheads of accessing the
  block table, executing extra branches, and handling variable sequence lengths ... this leads to 20–26% higher attention
  kernel latency".
- 2024 — C. Guo et al., GMLake, ASPLOS 2024 (arXiv 2401.08156; lit/e-gmlake2024/), p. 1: "GMLake employs a novel virtual
  memory stitching (VMS) mechanism, which can fuse or combine non-contiguous memory blocks with a virtual memory address
  mapping."
- 2024/2025 — R. Prabhu et al., vAttention, ASPLOS 2025 (arXiv 2405.04437; CC-BY 4.0; lit/e-vattention/), p. 1: "We
  present vAttention — an approach that mitigates fragmentation in physical memory while retaining the virtual memory
  contiguity of the KV cache. We achieve this by decoupling the allocation of virtual and physical memory using CUDA
  virtual memory management APIs."
- Apple documentation (developer.apple.com JSON, fetched 2026-11-06; lit/e-apple-bytesnocopy/,
  lit/e-apple-placement-sparse/), makeBuffer(bytesNoCopy:length:options:deallocator:): "Creates a buffer that wraps an
  existing contiguous memory allocation." pointer: "A page-aligned pointer to the starting memory address." Important:
  "The existing memory allocation needs to exist within a single virtual memory (VM) region." MTLHeap: "to create a
  heap, configure an MTLHeapDescriptor instance and call the makeHeap(descriptor:) method of an MTLDevice instance."
- macOS 26.4 SDK headers (agent e and mine): mach/mach_vm.h 264–283 (`mach_vm_remap(…, boolean_t copy, …,
  vm_inherit_t inheritance)`); mach/vm_statistics.h 263–265, VM_FLAGS_FIXED "Allocate new VM region at the specified
  virtual address, if possible." and 282–284, VM_FLAGS_OVERWRITE "The new VM region can replace existing VM regions if
  necessary (to be used in combination with VM_FLAGS_FIXED)."; MTLDevice.h 690–694, `newBufferWithBytesNoCopy`, "Create a
  buffer by wrapping an existing part of the address space."; MTLDevice.h 676, `newHeapWithDescriptor:`, the only method
  in the Metal headers returning an MTLHeap; MTLHeapDescriptor's properties (size, storageMode, cpuCacheMode,
  sparsePageSize, hazardTrackingMode, resourceOptions, type, maxCompatiblePlacementSparsePageSize) take no pointer;
  MTL4CommandQueue.h 394–409, `updateBufferMappings:heap:operations:count:`, heap "An ``MTLHeap`` you allocate with type
  ``MTLHeapType/MTLHeapTypePlacement``"; MTL4CommandEncoder.h 28–32, MTL4VisibilityOptionResourceAlias, "Flushes caches to
  ensure that aliased virtual addresses are memory consistent." The word "alias" appears in no Metal 3 barrier or
  visibility API (grep over all Metal headers, mine).
- CUDA Programming Guide v13.5 §4.2.5.3 (lit/f-cuda-graphs/, mine): "Physical memory reuse between graphs is done with
  virtual aliasing: different graphs can map the same physical memory to their unique virtual addresses."

**(c) Comparison.**
- *Matches* artificial contiguity exactly (Randell and Kuehner; Denning), built with the same reserve-then-overwrite
  sequence the XNU kernel uses for its own POSIX shared-memory mmap, and with `copy=FALSE` meaning shared, not
  copy-on-write (vm_remap.html; vm_map.c). The creator/mapper split of a named shared-memory object is POSIX shm_open's.
  The nearest user-space precedents are TPCircularBuffer (in-task `vm_remap`, copy 0) and Mesh (several virtual spans of
  one file offset); the nearest GPU precedents are CUDA VMM, GMLake's stitching and vAttention, all of which keep
  kernels unaware of the scatter, against PagedAttention's block table read in the kernel (20–26% kernel latency, by its
  own measurement).
- *Deviates from the GPU precedents* in who owns the physical memory: CUDA's `cuMemCreate` memory, D3D tile pools and
  Metal heaps are driver-allocated; here the pages are a POSIX shm object created by another process. The exclusion of
  Metal 4 placement sparse buffers is *required by the headers* under that ownership: tiles come only from a placement
  heap, a heap comes only from `newHeapWithDescriptor:`, and its descriptor takes no memory. (Vulkan, by contrast, can
  import a host pointer, VK_EXT_external_memory_host.)
- *Unresolved against Apple's text.* Apple's documentation requires the memory wrapped by bytesNoCopy to "exist within a
  single virtual memory (VM) region", while each `VM_FLAGS_FIXED|VM_FLAGS_OVERWRITE` remap is, in the SDK's words, a "new
  VM region", and runs of the view whose blocks are not consecutive in the region cannot be one Mach map entry (Rashid:
  an entry maps "a contiguous range of virtual addresses onto a contiguous area of a memory object"). The code merges
  consecutive rows into one remap but does not make the view one region. No source says Metal accepts such a buffer;
  PB R2 reports that the composed form ran on these machines on 2026-11-02 (a report, not a source). This is a deviation from Apple's stated precondition that no quoted text permits.
- *Synonyms.* Two values' rows naming one block are two virtual addresses of one page — Cekleov and Dubois' "synonyms in
  the same context". Their remedy (software-directed invalidation) is what MTL4VisibilityOptionResourceAlias provides, and
  the replay puts it on handover barriers. The Metal 3 interface (the delegate's and torch's) has no alias option in the
  headers, so a Metal 3 consumer of the view has no documented way to flush synonyms; mesh_compile.py's note relies on
  Metal 3 hazard tracking of one buffer, which orders commands but is not documented as making aliased virtual
  addresses coherent.

**(d)** Published counterparts exist. Verdict: **matches artificial contiguity / reserve-and-remap; the placement-sparse
exclusion is required by the Metal headers; the single-VM-region precondition of bytesNoCopy is not met by the text and
is unresolved.**

### 8. Lifetimes and storage sharing for captured buffers

**Code.** `MetalPrepareDataflow` gives each capture value [begin, end] = its first and last level; a value no
command binds gets [0, levels − 1]. rdma/mesh_compile.py `program()` (3758–3940) sets, for every EXECUTION value of
lag 0 of a function whose call was captured, `value.begin, value.end = 0, max(levels, 1) - 1` (3809–3811), discarding
the IR's per-value positions; tools/e2b_replay.m does the same for operands (`.end=level?level-1:0`, begin 0,
1295–1302). mesh rdma/mesh-collective.c `plan_interval` (951–972) turns a LOCAL item of copy c into [c n + begin,
c n + end] on a ring of u n times (n the function's levels), and a ROUTED item into an interval that starts `lead`
executions back and may wrap; `plan_place` (973–1040) colours each function's intervals with `mesh_graph_fat_cover`
(mesh-graph.c 979–1143, the two-phase fat-cover heuristic for cyclic interval graphs: cyclic intervals first, by fat
covers at the fat spots, then a left-to-right sweep with a free pool) and every other item with `mesh_graph_chains`
(Dilworth's fewest chains by matching).

**(a) Standard names.** Liveness-based storage sharing: values whose lifetimes do not overlap (do not interfere) share
storage — register allocation by graph colouring applied to buffers, interval-graph colouring by the left-edge
algorithm, offline memory planning by tensor usage intervals. Lifetimes that wrap around a loop's period are circular
arcs (cyclic intervals); colouring them is NP-complete, and the fat-cover heuristic of Hendren, Gao, Altman and Mukerji
is the published method the code implements. Interference over a partial order of events (happens-before) and fewest
chains (Dilworth) is the code's other allocator. Giving a value the whole range of levels is the treatment liveness
planners reserve for graph inputs, outputs and persistent state.

**(b) Sources** (agent b unless marked).
- 1971/1975/1979 — Hashimoto and Stevens (left edge, DAC 1971), Tucker ("Coloring a family of circular arcs", SIAM J.
  Appl. Math. 29(3), 1975), Fabri ("Automatic storage optimization", 1979): **not read**; secondary, verified: Mukerji's
  proof cites the "left edge algorithm [HS71]"; Garey et al. p. 217 cite Tucker [12] on the register-allocation use of
  circular arcs; Hendren et al. p. 31: "Interval graphs have also been used to overlay arrays and thereby minimize program
  memory requirements [25]" ([25] = Fabri's 1982 thesis).
- 1978 — L. Lamport, "Time, Clocks, and the Ordering of Events in a Distributed System", CACM 21(7):558–565, 1978
  (https://lamport.azurewebsites.net/pubs/time-clocks.pdf; lit/f-lamport1978/, mine), p. 559 (arrows normalized from the
  scan): "(1) If a and b are events in the same process, and a comes before b, then a → b. (2) If a is the sending of a
  message by one process and b is the receipt of the same message by another process, then a → b. (3) If a → b and b → c
  then a → c." "This implies that → is an irreflexive partial ordering on the set of all events in the system."
- 1980 — M. R. Garey, D. S. Johnson, G. L. Miller, C. H. Papadimitriou, "The Complexity of Coloring Circular Arcs and
  Chords", SIAM J. Algebraic Discrete Methods 1(2):216–227, 1980 (lit/b-gjmp1980/, page images), p. 216: "the circular arc
  graph coloring problem, and the circle graph coloring problem, as well as several related problems, are proved to be
  NP-complete." p. 217: "For each assignment of a value to a variable within the loop, the lifetime of that assignment
  consists of the portion of the loop that begins where the assignment is made and that ends where that value is used for
  the last time. Each such lifetime thus corresponds to an arc of the circle."
- 1982 — G. J. Chaitin, "Register Allocation & Spilling via Graph Coloring", SIGPLAN '82, pp. 98–105 (lit/b-chaitin1982/),
  §2, p. 99: "two symbolic registers conflict or interfere, if they are ever live simultaneously, more precisely, if one
  of them is live at a definition point of the other."
- 1988 — Lam, PLDI 1988 §2.3 (mine): "The lifetime of a register variable is defined as the duration between the first
  assignment into the variable and its last use."
- 1992 — Rau et al., HPL-92-48 (lit/b-rau1992-hpl/), p. 16: "This is done by wrapping the infinite plane into a cylinder
  of minimum circumference, i.e., minimum number of registers"; p. 20: "(taking into account the wrap-around of the
  lifetime at K*II)".
- 1993 — L. J. Hendren, G. R. Gao, E. R. Altman, C. Mukerji, "A Register Allocation Framework Based on Hierarchical Cyclic
  Interval Graphs", ACAPS Technical Memo 33 (Revised), McGill, 26 Feb 1993 (CC 1992 version not read;
  https://www.sable.mcgill.ca/~hendren/ftp/acaps/memo33.ps.gz; lit/b-hendren-memo33/), p. 4: "We call such a "wrapped"
  interval — a cyclic interval." §2.2, p. 5: "cyclic intervals can also be used to represent live ranges of variables in
  loops which are first unrolled. Of course, each variable in the original loop will be split into several variables
  depending the unrolling factor." p. 10: "If the first phase fails to find a fat cover at some stage, the second phase
  simply colors the remaining cyclic intervals with new colors ... In this case, the resulting coloring may or may not be
  optimal." p. 15 (1000 random graphs): "the hybrid fat cover algorithm is the winner."
- 1994 — C. Mukerji, "Register Allocation Using Cyclic Interval Graphs", M.Sc. thesis, McGill, July 1994
  (https://mcgill.scholaris.ca/server/api/core/bitstreams/5031fdf9-9e78-49b5-97e9-f74540dde00d/content;
  lit/b-mukerji1994/, page images), §2.1, pp. 21–22: "In this paper, we define the lifetime interval of x to be [t, t′)."
  Thm 2.3.2, p. 24: "Let G be an interval graph containing cyclic intervals. Then G is optimally colorable with Wmax(G) ≤ k
  ≤ Wmax(G) + Wmin(G) colors." Def 3.2.2, p. 32: "A fat cover of a cyclic interval graph, G, relative to interval I is a
  subgraph F (I ∈ F) of G that obeys the following two properties: (1) all intervals in F are non-overlapping, and (2)
  ∀t_i ∈ fatspots(G), there exists an interval in F that covers t_i." §3.2.3, pp. 32–33, the two phases (as quoted in the
  agent's list). p. 85: "It deviates from the optimal number of registers in only two cases, and on the average requires
  10% more registers." p. 89: "For the unrolled tomcat loop, the interval graph had a minimal coloring of 34, and our method
  introduced 4 chameleon intervals to make it 32 colorable."
- 2016 — T. Chen, B. Xu, C. Zhang, C. Guestrin, "Training Deep Nets with Sublinear Memory Cost", arXiv 1604.06174
  (lit/b-chen2016/), §3: "We can only share memory between the nodes whose lifetime do not overlap." "We use this as a
  static memory allocation algorithm, to allocate the memory to each node before the execution starts".
- 2018 — T. Chen et al., TVM, OSDI 2018 (arXiv 1802.04799; lit/b-tvm2018/), §3: "a static memory planning pass, which
  pre-allocates memory to hold each intermediate tensor".
- 2020 — Y. Pisarchyk and J. Lee, "Efficient Memory Management for Deep Neural Net Inference", arXiv 2001.03288
  (lit/b-pisarchyk2020/), §3: "Tensor Usage Interval of an intermediate tensor t is defined as the pair {first_op_t,
  last_op_t} ... Note that no two tensors with intersecting usage intervals can share memory."
- TensorFlow Lite arena planner (tensorflow/tensorflow @ 3a54360f; Apache-2.0; lit/b-tflite-arena/, lit/b-tflite-options/):
  arena_planner.cc 239–241: "We must make sure the output tensors are never overwritten. We do that by artificially adding
  one to their ref-counts so they are never selected for deallocation."; 248–249: "Variable tensors also should be ensured
  to be never overwritten and need to be alive all the time."; arena_planner.h 51–53: "The inputs to the graph will not
  share memory with any other tensor, effectively preserving them until the end of inference."; simple_memory_arena.cc
  393–395: "Usage interval of alloc doesn't intersect with current tensor's usage interval, so we skip it.";
  interpreter_options.h 28 (the option that keeps every tensor): "Preserving all intermediates tensors for debugging."
- 2026 — CUDA Programming Guide v13.5 §4.2.5 "Graph Memory Nodes" (lit/f-cuda-graphs/, mine): "Graph allocations have
  fixed addresses over the life of a graph including repeated instantiations and launches. ... Within a graph, allocations
  whose graph ordered lifetimes do not overlap may use the same underlying physical memory." §4.2.5.3.1: "CUDA may reuse
  memory within a graph by assigning the same virtual address ranges to different allocations whose lifetimes do not
  overlap." Fig. text: "The new alloc node is not dependent on the free node (2) so cannot reuse the address from the
  associated alloc node (2)."
- Not read (from memory): R. P. Dilworth, "A decomposition theorem for partially ordered sets", Annals of Math. 51(1),
  1950; D. R. Fulkerson, "Note on Dilworth's decomposition theorem for partially ordered sets", Proc. AMS 7(4), 1956 (AMS
  returned 403): a finite poset's minimum chain cover equals its largest antichain; found by bipartite matching.

**(c) Comparison.**
- *Matches.* `mesh_graph_fat_cover` follows Mukerji's §3.2.3/§3.2.6 structure: fat spots at maximum width, a fat cover per
  cyclic interval chosen by the stated heuristic (earliest start, then longest), new colours for the cover, then a
  left-to-right sweep with a pool of freed colours for the rest; a cyclic interval with no cover gets new colours, which
  is §3.2.3's failure path ("simply colors the remaining cyclic intervals with new colors"). Mukerji's chameleon intervals
  (her chapter 4), which recovered Wmax on the tomcat loop, are not implemented. The per-copy intervals on a ring of u × n
  levels are Hendren et al.'s unrolled variables "split into several variables depending the unrolling factor", each a
  cyclic interval. Reuse only where the dependence order puts one value's last access before the other's first write is
  CUDA graph memory nodes' rule (DP Appendix B's sanctioned method) and Lamport's happens-before; the chains are
  Dilworth's minimum chain cover. Closed [begin, end] intervals at level granularity are stricter than Chaitin's and
  Mukerji's half-open [t, t′), which is safe where commands of one level run concurrently.
- *Known limits of the method, carried into the code.* Fat cover is a heuristic for an NP-complete problem (Garey et al.):
  up to Wmax + Wmin colours (Mukerji Thm 2.3.2), 10% more than optimal on average, and Hendren et al.'s own experiments
  prefer their hybrid (reduction first, then fat cover). The code uses plain fat cover.
- *Deviates, not required — `program()`'s whole range.* For every EXECUTION value of lag 0 of a captured function,
  `program()` sets [begin, end] = [0, levels − 1], discarding the IR's per-value positions. In every liveness source a
  value lives from first to last use (Lam, Chaitin, Pisarchyk's usage interval, TFLite's first/last node); whole-run
  lifetimes are given only to graph inputs, outputs and variables (TFLite), and "preserve every tensor" exists in TFLite
  only "for debugging". The rule P14 item 7 names (first and last node) is the liveness rule; the code applies the
  inputs/outputs rule to every value. It is safe — values that never share cannot race — and costs memory: such values of
  one copy share no pages with each other within an execution. The cause is missing data, not an API: the recorder knows
  each operand's last level (`ends`) but exports no first level ("the operand-level query", listed as not written in
  d0a72fc6). (P14's named source for the first/last rule is ExecuTorch, torch-family; TFLite and Pisarchyk–Lee state the
  same rule without torch.)

**(d)** Published counterparts exist. Verdict: **fat cover and chains match their sources (fat cover without chameleon
intervals, as its failure path allows); the whole-range lifetime for captured values deviates, not required, from
first/last-use liveness.**

### 9. Mixed execution (mesh fcf5d18a): routed nodes re-run their operators from Python, the rest replay

**Code.** mesh fcf5d18a, rdma/mesh_compile.py `program()`: `replayed = here == {'mps'} and not any(d.value
!= mesh.ABSENT …)`. A node whose values a declared transfer names runs `_execute` each execution (3124–3133):
`mesh_transfers_start`, `started(t)`, then copy t mod period's precomputed list of Python closures (`_steps`, 2975–2980):
torch operator calls into placed pages, host waits on landings (`_arrived`, spinning on `mesh_host_arrived` with a
progress-based bound), and publishes through MPSEvent notifications. Other nodes replay their capture through
`MetalRecorderRun`. The commit message gives the reason: the recorder masks a reader by rewriting its source, and torch's
kernels come from a metallib with no source.

**(a) Standard names.** Mixed-mode execution: compiled (here, replayed) code where it can be recorded, an interpreter
elsewhere — the split of trace-based JITs and dynamic optimizers, with side exits or bail-outs to the slow path. The
routed nodes' per-execution path is a precomputed straight-line list of closures run in order (closure compilation /
threaded code). The published alternative is to capture the communication into the replayed graph (NCCL in CUDA graphs)
or to hold the reader behind a recorded wait.

**(b) Sources.**
- 1973 — J. R. Bell, "Threaded Code", CACM 16(6):370–372, 1973, DOI 10.1145/362248.362270. **Not read (from memory):** a
  program as a list of addresses of routines run in sequence, an interpretive form needing no conventional interpreter
  loop — the form of `_steps`' lists of closures.
- 1999/2000 — V. Bala, E. Duesterwald, S. Banerjia, "Transparent Dynamic Optimization: The Design and Implementation of
  Dynamo", HP Labs HPL-1999-78 (Wayback copy; lit/a-dynamo/), §3.4, p. 14: "Once Dynamo bails out, the program continues
  to execute directly on the processor, and Dynamo never regains control." PLDI 2000 version, §2: "When control eventually
  exits the fragment cache, Dynamo resumes interpreting the instruction stream, and the process repeats itself." (agent a)
- 2009 — A. Gal et al., "Trace-based Just-in-Time Type Specialization for Dynamic Languages", PLDI 2009, pp. 465–472
  (https://www.cs.williams.edu/~freund/cs434/gal-trace.pdf; lit/a-tracemonkey/), p. 465: "Our system uses a mixed-mode
  execution approach: the system starts running JavaScript in a fast-starting bytecode interpreter. As the program runs,
  the system identifies hot (frequently executed) bytecode sequences, records them, and compiles them to fast native
  code." p. 466: "Every compiled trace contains all the guards (checks) required to validate the speculation. If one of
  the guards fails (if control flow is different, or a value of a different type is generated), the trace exits." §3.1:
  "When any situation occurs that prevents the implementation from continuing trace recording, the implementation
  aborts trace recording and returns to the trace monitor." (agent a)
- 2021– — NVIDIA, NCCL 2.32.3 documentation, "Using NCCL with CUDA Graphs"
  (https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html; lit/f-nccl-cudagraph/, mine): "Starting
  with NCCL 2.9, NCCL operations can be captured by CUDA Graphs." "Whether an operation launch is graph-captured is
  considered a collective property of that operation and therefore must be uniform over all ranks participating in the
  launch (for collectives this is all ranks in the communicator, for peer-to-peer this is both the sender and
  receiver)." "Mixing graph-captured and non-graph-captured NCCL operations is supported by NCCL."
- 2026 — CUDA Programming Guide v13.5 §4.2.2.1.2.2 (lit/a-cuda-guide/): "As a general rule, when a dependency relation
  would connect something that is captured with something that was not captured and instead enqueued for execution,
  CUDA prefers to return an error rather than ignore the dependency." §4.2.1.1: graph nodes include "CPU function call",
  "waiting on a CUDA Event" and "waiting on an external semaphore" — a replayed graph can hold the host's work and the
  wait for a remote writer.
- 2018 — Frostig, Johnson, Leary, SysML 2018 (above): the traced graph is reused until a "trace cache miss".

**(c) Comparison.**
- *Matches* the mixed-mode split in form: TraceMonkey and Dynamo run recorded code where recording succeeded and an
  interpreter elsewhere. The recorder's refusal is an abort of recording ("returns to the trace monitor"); fcf5d18a makes
  the choice per node before capture instead of discovering it.
- *Deviates from NCCL's rule only nominally:* the routed nodes are exactly the endpoints of the declared transfers, so
  both ends of each transfer run the same (non-replayed) form, as NCCL requires capture to be uniform over the ranks of an
  operation.
- *Not required by Metal.* The cause given in the commit (a reader is masked by rewriting its MSL source, and a library
  without source cannot be rewritten) applies only to the masked first pass. A reader can be held, whole, behind the
  queue's `waitForEvent:value:` with no source at all; d0a72fc6 records exactly that ("a whole later pass with no masked
  first pass") but `replay:` and `MetalReplayEncode` do not yet handle it. CUDA graphs and NCCL show the other published
  route: the wait and the communication inside the replayed program.
- Against the change requests' requirement (AOT A1, context): "nothing runs eagerly between its compiled calls"; the
  routed nodes' operators run as torch calls dispatched from Python each execution, between the plan's compiled
  transfers. Whether a straight-line list of exported-graph operators counts as "compiled" is the operator's call; the
  literature names it interpretation, not replay.

**(d)** No published counterpart for choosing, per node of one distributed program, between replaying captured GPU
commands and dispatching the exported graph's operators one by one from Python. Verdict: **no counterpart for the
per-node split; the mixed-mode form matches trace JITs; the split is not required by Metal (a whole later pass needs no
source).**

## C. Where the code's own citations misstate their sources

- **MPI 4.1 section numbers (not an error of the code).** The code cites "MPI 4.1 §3.9" and "§3.7"; the official PDF
  (2 Nov 2023) numbers "3.7 Nonblocking Communication" and "3.9 Persistent Communication Requests" exactly so, while the
  HTML pages the code links label the same sections "4.7" (node71.htm) and "4.9" (node83.htm), and partitioned
  communication is chapter 4 in the PDF, chapter 5 in the HTML. Cite the PDF's numbers with a PDF page, or the HTML's
  with its node; the quoted sentences are correct in both.
- **Lam 1988 §2.3.** `MetalReplayPeriod` quotes "the degree of unrolling is given by the lowest common multiple of {q_i}"
  as its authority for L = lcm. Lam's next sentences call that lcm a possible "intolerable increase in code size" and give
  the minimum degree, max_i q_i, as the remedy; Rau et al. (HPL-92-48) measured and recommend the minimum.
- **Lam 1988 §1 for `lead`.** mesh-collective.h and tools/e2b_replay.m cite §1's description of software pipelining for
  `lead`; the sentence states that iterations overlap and gives no value or rule for the lead. The rule for a lead is
  ⌈latency / iteration time⌉ (Mowry, Lam, Gupta 1992, §2.3).
- **MSL memory orders.** metal_recording.m: "MSL's memory_order holds relaxed and seq_cst alone (metal_types, enum
  memory_order)". True of the installed headers (GPUCompiler 32023.864 and Metal toolchain 32023.883: `enum memory_order {
  memory_order_relaxed, memory_order_seq_cst }`), not of the current specification: MSL 4.1 (2026-06-04) §6.16.1, "All OS:
  Metal 4.1 and later support the additional memory order values, memory_order_acquire, memory_order_release, and
  memory_order_acq_rel." (agent c).
- **"System scope".** The code calls `static_cast<metal::thread_scope>(3)` a system-scope fence and uses
  `coherent(system)`. The installed `enum thread_scope` names thread, simdgroup, threadgroup and device only, and MSL 4.1
  §6.16.2 lists the same four (agent c); no public text defines scope value 3 or a system scope. The code reaches it
  through `#pragma METAL internals : enable`.
- **The 2^22-command ceiling** (metal_recording.h, "a buffer allocated past it faults on the first write beyond") has no
  header text; it is an observation, stated as a fact.
- **Hazard tracking as alias coherence.** rdma/mesh_compile.py's note on `_placed_buffer` relies on "Metal orders by its
  hazard tracking (MTLResource.h hazardTrackingMode)" for a write and a read of one physical page through two virtual
  addresses in one buffer. Hazard tracking orders commands; making synonyms coherent is what MTL4CommandEncoder.h's
  ResourceAlias option is documented to do ("Flushes caches to ensure that aliased virtual addresses are memory
  consistent"), and the Metal 3 interface has no such option.
