# AGENTS.md: the operator's words for this repo

`THREAT-MODEL.md` holds the operator's words on access, verbatim (2026-08-27). Code is ranked by `design/rubrics.md`.
The operator, 2026-10-22: "rubric supersedes rules. rubric lines addressed this topic. attempts to write rules must now
be compeltely removed. no more rules anywhere in the documentation, memoryfiles, anywhere. anything not expressible in
the rubric's constrained dialect is not evidence, is not a rule, is not justification, and does not reflect a sanctioned
action supported by the user or by literature."

## What we have (2026-10-22)

1. Each node runs one launchd job, `mesh-heartbeat`, which opens the bridge (`mesh-flow`), opens it again at most once a
   minute when it ends unasked, and closes it on SIGHUP (restart, region kept) or SIGUSR1 (release, region freed).
2. The bridge maps one shared memory region for the node, registers it once with every Thunderbolt RDMA device, and
   runs one child process per port, which alone holds that port's queue pairs.
3. A child pairs its link once with the peer's child (hello, make, connect), keeps up to four streams on it, posts
   receives and SENDs as clients' plans present them, raises landed words, writes ~0 into them when the link is lost,
   and never touches a payload byte; its exit is the only release of its queue pairs.
4. A lost link (link off, a termination, a failed completion, the control socket's end) ends that port's child after a
   5 s quiet window, and a new child pairs again when the peer answers; other ports, clients and leases are untouched.
5. A child absorbs the fault of a post racing a vanished cable, so that its exit does not fall inside the device's
   termination: the kernel path that hung the Mini on 2026-09-05 and panicked the M5 on 2026-10-18.
6. libnccl-mesh is each process's client: it takes rows of the region (`ncclMemAlloc`), holds a stream toward each
   peer node, compiles each call's plan once and starts it again (MPI-4 persistent collectives), allocates every
   start's send and landing pages as rows released by use count, and returns failures as values (`ncclRemoteError`).
7. Kernels cross in-stream through the device-function header (`mesh_enter`, `mesh_leave`, landing, own and send
   views), and host-path calls run the library's own combine kernels.
8. torch-mesh is the `torch.distributed` backend over libnccl-mesh (`init_process_group(backend="mesh")`), and
   `rdma/mesh_mpi.py` the NumPy binding.
9. Data code (examples/, metal-microbench's engine, torch programs) passes tensors and views and reaches nothing of the
   bridge or the region.
10. In progress: routing through a third node while a link is down, a communicator re-peering itself after a loss, and
    the heartbeat watching its bridge's progress.

## Recovery

The operator, 2026-10-06, of reconnection and resumption: "lets split out reconnection and resumption logic into a
completely different module which has never been fully defined in terms of the rubric of all requirements and all
mandates. there are many different kinds of network problems which can happen in differnt link topologies, some which
are not testable in the live sense until certain other real life conditions come to pass; we should treat this as a
critical level priority which is unfinished, unsettled, improperly reviewed, but nevertheless is allowed to operate
unchanged in the interim"; 2026-10-14: "connection recovery on spotty hardware (rdma over thunderbolt is NOT
trustworthy at the physical substrate level) is required, rather than optional". The transport, how it fails and how it
recovers: `design/transport.md`.

The operator, 2026-10-22, the three kinds of failure the mesh handles, and no others ("we cannot and will not 'handle'
wacky scary problems"):

> one is that someone might stamp on a cable and break a port entirely by any sort of means. maybe all sends through
> that cable fail. maybe that port stops working. maybe the connection is severed. i don't care what the entailment is;
> i want for collectives trying to use the damaged connection to fail immediately rather than break phase or timing
> with steady state operation, and for a resolve of the collective to forward around degraded or missing connections
> at a new steady state speed instead of hanging forever, faulting, or trying to interact iwth something degraded.
>
> another problem is that a connection might be made during runtime, or a cable might be retethered during runtime.
> this too must result in a fast failure and then a renegotiation of all connections. if a cable is unplugged from slot
> 1 and moved to slot 3, the collective operations just need to deal with ti during the time untethered without any
> consideration for when or hwo an alternate edge came back.
>
> a final kind of error is that a node is unresponsive. unresponsive nodes should become responsive in finite time, and
> while they're not back, calculations where possible should ocntinue with fewer ranks at a different steady state.

## The hardware

The operator, 2026-10-22; the measurements, literature and mechanisms behind each are in
`design/hardware-and-discipline.md`:
1. "the hardware is weird and not like arm64 cpus or older apple silicon."
2. "dma is fundamental and requires exceptional programming practices to make useful collectives from weak silicon."
3. "weak silicon is programmed at a similar level of abstraction to streaming multiprocessors, not abstractions
   attempting to wrap entire gpus or nodes."
4. "we economize on different tradeoffs from nccl, and use a more constrained (but still universally expressive and easy
   to use) memory model in exchange for strictly fewer waits, copies, and protocol dependency relation depths. this is
   not 'nice-to-have' but is needed for baseline performance."
5. "bridges and collectives are expressive enough to allow data corruption wrt a data producing or consuming program. we
   never do anything to prevent what might be a data bug for some semantics of data. data processing lcients might be a
   guitar pedal with a realtime guarantee or they might be a deferred data processing cron job. programmers have to
   write their own syncs and guards or they get the fastest resutls possible."

## Contiguous and non-contiguous tensors (final)

<table><tr><td>

> fi only one backend requires contiguity then the existence of other backends requires both contiguous and noncontiguous tensors to be supported. this is final.
>
> — the operator, 2026-10-22

</td></tr></table>

Every layer between a producer and its consumer (the collectives, the bridge, torch-mesh) takes contiguous and
non-contiguous tensors alike: a tensor that crosses is a list of page runs on each side, scattered or gathered as its
layout is. Contiguity is made only at the one consumer that requires it, in the page table, when it asks (design/
SPECIFICATION.md, the operator 2026-09-14: "copy pages into a literal contiguous span and expose views to that
contiguous view if an allocator or user says 'this one has to be contiguous'").

## Access

The operator (THREAT-MODEL.md, 2026-08-27):

> rejection filtering and defensive programming are broadly forbidden by our security model. accessibility is required;
> inaccessibility is *contractually forbidden*. no configuration or operation of any software in our mesh repo can
> demote the reachability, accessibility, and workload of a node or edge between nodes

> the most important class of compromise is the false negative: the computer should be on and receiving work and doing
> work, but isn't, for any reason, no matter how sensible the 'reason' might sound

> are you inventing fictional reasons for machines to become solipsistic and inaccessible? this is an attack surface and
> must be removed categorically as it violates our security model.

Rejection filters in collective and data code, design/rubrics.md C6: rejection filters ≺ none (2026-10-20: "count every
rejection filter everywhere in client and data processing code. then delete absolutely all of them."). The operator,
2026-10-21: "rejection filter removal. categorical. all of them without sanction in rubric".

facb7cf9 (2026-08-27) deleted the `appliance` and `portable` node profiles, which withheld screen sharing, the firewall
posture and the power policy from a class of machine; no requirement stood behind them.

`enable-autologin.sh` checks an auto-login password with `dscl -authonly` before it writes `/etc/kcpassword` (facb7cf9).
A wrong password written there leaves a node that boots to a login it cannot pass. No operator quote stands behind the
check.

## Version control

The operator (THREAT-MODEL.md, 2026-08-27):

> index only these two choices for selection of version controlled repos in code or configuration: 1:`main` 2:`{any
> named branch}` [...] you can *never* index by commit *ever* unless you register that commit as a branch. once you do,
> code which indexes against version control must *always* use the most up to date commit on whatever branch is chosen.

> you may *never* induce a paradox of the unpatchables *anywhere* in *any* code

`bootstrap.sh` fetches `$MESH_REPO` at `$MESH_BRANCH` (`main` where unset). A node pinned at a commit would fetch its
bootstrap at that commit too, so no rerun of `install.sh` would bring it the change that unpins it.

## Node identity and names

The operator, 2026-10-21: "from this point onwards, we never rely on something like a localhostname as a name or
identifier for something which we are programming. localhostnames are apparently a legacy thing, can be named
automatically, collide, etc. undignifed, and we almost certainly do not need them for anything algorithmic", and "we
should be able to use any kind of virtualization of 'names' such that we never rely on anything like, again, that
localhostname thing, which makes no promises of selectivity or unique identifiability or anything of the sort".

A node's identity is its host UUID (`IOPlatformUUID`, `gethostuuid()`). The bridge's node id, the fabric ULA
(`mesh-router-init.sh`) and node-info's `node_id` are all drawn from it; the hardware serial number is the same
machine's other fixed identifier. The name table is `etc/mesh-nodes.json`, entries keyed by host UUID, read through
`rdma/peers.py` (`host`, `table`, `label`): a node it does not list is `node <number>`. The beacon announces the host
UUID; `mesh-peers`, node-info, `mesh-top`, the observer and telemetry key on the node number and take names from the
table. The bridge's record carries no name (`rdma/mesh.h` `struct mesh_record`: its node and boot), and `mesh-stat`
prints none. The bootstrap still reads `gethostname()` where no host is configured (`rdma/nccl-mesh.c`).

macOS assigns `LocalHostName`, `ComputerName` and `HostName` in Setup Assistant and renames a running machine on its own
when another claims the same name. On 2026-10-21 the second M5 Ultra Studio came up as `mdots-Mac-Studio`, macOS
renamed the first, an hour into its service, to `mdots-Mac-Studio-2`, and `mdots-Mac-Studio.local`, its beacon and
`mesh-peers` all pointed at the other machine.

## The provisioning scripts on a failure

- Not root: `bootstrap.sh` and `install.sh` re-exec themselves under `sudo`.
- No named user: `install.sh` takes the first set of `MESH_USER`, the recorded admin user and `SUDO_USER`, the console
  owner where none is set or the one taken does not exist, and the first admin where that is root or does not exist;
  `enable-autologin.sh` takes `SUDO_USER`, then the console owner.
- `bootstrap.sh` enables and starts sshd before it fetches anything, so a node whose later steps fail is still reachable.

## Outside a script's reach

- `rdma_ctl enable` runs only in Recovery OS (exit 77 under a booted system): one physical 1TR visit per machine.
- `systemsetup -setremotelogin` needs Full Disk Access, which cannot be granted headlessly; `bootstrap.sh` and
  `install.sh` start sshd with `launchctl bootstrap`, which needs only root.

## Restarting an idle node

Operator instruction, 2026-09-05:

> a priori you're allowed to restart any mesh machine without a userspace or active service running without any consideration

## Review and the linter

The operator, 2026-10-20 (design/rubrics.md, "How the operator reviews"): "the thing called a rubric is not
metaphorical. all code needs to be reviewed in terms of the rubric before being built and compiled from this point
onwards"; "use a subagent to review code which has been newly written within a session wtih the rubric file".

`make -C rdma lint` runs `bin/mesh-lint`, which prints each form in the bridge, the collective and their clients that
design/rubrics.md ranks below another, as the rubric's ranked pair with the operator's words; `--semantic` gives a
language model the rubric's text and one function at a time. The operator, 2026-10-22: "the result of this 'audit'
should be a linter/typechecker/parser which evaluates source and writes logged warnings of deductively determinable
errors related to waits, syncs, guards, etc. i'm tired of this runaround where one agent writes extremely incorrect
collective code then another agent 'notices' what was extremely incorrect, 'fixes' the code, then stops existing and
leaves no guidance or corrective or example in the repo"; and "a parser can use language model calls on compact units
of text or source if it reduces implementation complexity and delivery speed by an order of magnitude".

## Collectives and tensor parallelism

The algorithm on two peers and on infinitely many Minis, and why waiting on anything but stamped pages fails on both:
`design/pages-and-functions.md` (its analysis: metal-microbench `docs/mesh_distributed_reduction_analysis.md`). The
operator's statements of scope and the literature: `../../metal-microbench/docs/principles.md` (from this repo's root;
on the Mini, `~/metal-microbench/docs/principles.md`); how methods are compared: its `docs/measurement.md`. A
collective is a compiled program made ahead of time along trees of an explicit link map (`rdma/mesh-plan.h`,
`rdma/mesh-collective.c`). Using the library: `rdma/NCCL.md`; PyTorch programs: `rdma/torch-mesh/README.md`; NumPy:
`rdma/mesh_mpi.py`; where nodes or links differ: `design/heterogeneity.md`; the transport: `design/transport.md`; the
rubrics: `design/rubrics.md` (M for the bridge, C for the collectives, D for programs that use them); adding a node:
metal-microbench `docs/nodes.md`.

## Tests and specification

The operator instructed on September 6, 2026: "delete all tests. tests aren't
specification". The complete instruction is preserved in
[design/SPECIFICATION.md §24](design/SPECIFICATION.md#24-whole-program-review-and-deletion-of-tests-september-6-2026).
Repository-owned tests and verification harnesses have been removed.

On 2026-10-14 the operator stopped all runtime tests of code (T9). The stop holds until each item of metal-microbench `docs/acceptance-20261014.md` is met in source. Its `docs/principles.md` section 5 has the quote. The checklist records all 20 items attained on 2026-10-14.

## Source workflow

Operator instruction, September 9, 2026: use commits and checkouts, and use
`main` when it suffices. Preserve overwritten source only in commit history;
no resets, cleans, stashes, backup copies, or uncommitted source transfers.
This governs development synchronization; the bootstrap continues to follow the latest named
branch.
