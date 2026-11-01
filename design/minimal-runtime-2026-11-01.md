# What a prepared program must be, and what may run at its start (2026-11-01)

The operator, 2026-10-31 and 11-01: "review the requirements of the data flow and find the absolutely lowest overhead
approach for the runtime given total access to all types and all properties of all programs ahead of time"; "reviewing
what the program must be then making it exactly that and nothing less and nothing more"; nothing run "should take more
than around 8000ms in any situation"; and any compilation step slower than "approximately two shm <-> register
transaction[s] of latency" is a program for the library's own SIMD and SIMT lowering, not host bookkeeping.

This note derives the program from its requirements, then lists each place where today's start or hour does more, with
its measured cost, what replaces it, and why these machines allow it. Costs are measured on the five nodes unless
marked as an estimate.

## The machines (design/scalarized-sets-2026-10-31.md, sysctl)

- Studio (m5ult1-4): M5 Ultra, 10 + 20 cores, 96 GiB (82 GiB region), 64 GPU cores, macOS 27.0. Mini: M4 Pro, 8 + 4
  cores, 24 GiB (16 GiB region), 16 GPU cores, macOS 26.5.1.
- A core: 31 x 64-bit general and 32 x 128-bit vector registers, SME2; 128 KiB L1 data, 16 MiB L2 a cluster; 128-byte
  lines. A GPU core's SIMD group is 32 lanes.
- A cable: about 8.6-8.9 GB/s a direction. A small message between two nodes' regions: 17-21 us (bandwidth.py's 64 KiB
  broadcast); a megabyte about 0.12 ms on one cable.

A problem of a few KiB of state fits a core's registers and L1. Megabytes of state move to any of five machines and
back in about a millisecond. A second of host time is three to six orders of magnitude above what the work needs.

## What the program is

A program's types, shapes, layout and membership are its configuration operands, all known before it runs. They fix:

1. **Its GPU commands:** each dispatch's pipeline, its argument-table rows (region offsets affine in the invocation
   index: `offset + invocation x stride`), its threadgroup grid. The ordering between dispatches comes from their
   declared page reads and writes: a barrier where a later command reads or overwrites what an earlier one writes, an
   event where the two lie on different queues.
2. **Its transport rows:** each channel's SEND and landing lists (unit ranges, the word and base they wait on, their
   `after` requirement lists: 76de94ce), bound to region offsets.
3. **Its pipelines:** Metal functions specialized by those shapes.

None of these depends on a value the program computes. Data-dependent work, such as an expert router's choice, is a
masked loop over a bounded domain (design/compiler-runtime-inventory-2026-10-31.md section 5B), still fixed in shape.

So per invocation, a node needs only:
- the GPU running its recorded commands;
- the bridge posting each row as its words say its data is there.

The host does nothing per call but one commit, and none where the GPU's own resident stream repeats.

At a program's start a node needs only:
- its pipelines loaded from a binary archive (milliseconds);
- its argument tables and transport rows written into region pages it already holds (a memcpy of kilobytes to megabytes);
- one exchange with its peers, agreeing that each holds the same program (one small message each way, tens of
  microseconds).

At a node's start the bridge needs only:
- its devices opened;
- the one region object registered (378dbd5f);
- each link's queue pairs paired with the peer's.

None of these needs doing again until the region's layout or a cable changes.

## The interventions

Each with its cost now, what replaces it, and why we can.

1. **Recording, re-run at every program start.**
   - *Now:* the chain service's 12,337 dispatches took 2.07 s. With the whole-lease page tables gone (3597fe0d) they
     take 0.38-0.42 s, about 0.2 s of it host work on the recording thread (sampled). What remains: an Objective-C
     dictionary of bucketed spans and an NSIndexSet a dispatch (metal-recording.m MetalViewAccesses, MetalViewPrior,
     MetalOrderViews, MetalEncoderReaches), NSData and block allocations an access.
   - *Replace with:* the frontier as flat arrays of fixed-width rows (key, lo, hi, node), sorted once, swept once. The
     dependencies are a merge of sorted interval lists: SIMD compares four or eight bounds at once. The recorded
     schedule, a pure function of the configuration, is serialized at preparation (design/replay-templates-2026-10-29.md)
     and loaded at start.
   - *Why:* 12,337 rows of about six intervals of 32 bytes are about 2.4 MB, inside one cluster's 16 MiB L2. A sweep at
     a few nanoseconds a compare is about a millisecond on one core (estimate), less over a node's 30 cores, or on its
     64 GPU cores as a library kernel.
2. **Pipelines compiled from source in each process** (torch_mesh _stream.mm newLibraryWithSource).
   - *Now:* the library's own kernels are AIR committed in vendor/ plus a binary archive each node makes at install,
     loaded with MTLPipelineOptionFailOnBinaryArchiveMiss. torch_mesh's specializations (chain.metal and the rest) are
     compiled from source by the OS compiler in every process. They are served from its cache only where the text is
     unchanged; on 11-01 the chain's first run after mesh.metal changed wrote nothing for over 8 s after placement.
   - *Replace with:* a program's specializations, named by its configuration, compiled into a per-node binary archive by
     the node's preparation, as the library's are, and loaded at start.
   - *Why:* the shapes are configuration operands, so the set of specializations is known before the run.
3. **The observation declaration, a 4.4 MB line a run.**
   - *Now:* 0.74 s of the main thread in a pipe write (sampled 10-31), moved after the start by b4ab4394.
   - *Replace with:* the declaration named by its configuration key, as completion lines already are; its body written
     once where the configuration is.
4. **A deploy with nothing to build.**
   - *Now:* about 1.7 s a node where nothing changed (mesh_nodes node prepare: mesh-user.sh 0.4 s, mesh_c_build.py
     0.7-0.9 s, setup.py 0.5-0.8 s, both run every time), 3.5-5.8 s where the libraries rebuilt (10-31 and 11-01).
   - *Done for the modules (the commit adding `make modules`):* _mesh_c and the torch extension are make targets with
     their real inputs; an up-to-date one is an mtime read. *Remaining:* mesh-user.sh's 0.4 s (uv's requirements check,
     three patch dry runs over torch, the launchd definitions rewritten and compared each time) as make targets too.
   - *Why:* a no-op make over a few dozen targets reads their mtimes, milliseconds.
5. **A bridge restart.**
   - *Now:* after bc17811b, a 5 s quiet window (B9), about 2 s of the kernel releasing the children's registrations,
     0.85 s exit to reap, and 0.9 s pairing. Every node pays this at every deploy that changes a bridge image.
   - *Background:* the quiet window is a recorded hardware constraint. A release inside a Thunderbolt peer interface's
     termination panicked the M5 (design/RDMA-KERNEL-RECOVERY.md "October 18"), so the hold stays wherever a link is lost.
   - *Replace with:* no link lost where only the parent's image changed. The children own every queue pair and
     registration; a parent that re-executes itself keeps its children (the same process across execve) and their
     control sockets (inherited descriptors), and reads each port's state back from its child (its pairing, lanes,
     peer). No loss means no quiet window, no release and no pairing. Only a change to the child's own code restarts a
     child, one port at a time, with its windows.
   - *Why:* nothing the kernel must release lives in the parent's image.
6. **The hourly, repeating all of the above every hour.**
   - *Now:* a preparation (item 4), a pairing wait of up to 300 s, then each program's whole start (items 1-3).
     Since ff2aee1d the hourly moves no checkout.
   - *Replace with:* nothing but the programs once items 1-5 hold: the preparation is a no-op make, nothing restarts, so
     nothing pairs, and each program loads its artifacts.
7. **A host wait with no end.**
   - *Was:* MetalScheduleEnd waited on its completion word with no bound. On 10-31 four ranks waited 42 minutes for a
     peer whose kernel had failed to build, while every GPU wait is bounded at about 1.1 s (NCCL_MESH_POLLS).
   - *On main since 8632a4ad:* completion is observed, never awaited (MetalScheduleStatus, mesh_statistics'
     `_emit`), and a teardown's waits (ncclCommDestroy) run under mesh_deadline, MESH_PLATFORM_NS, after which the
     program ends naming the wait. Resolved; kept here as the bound every later host wait meets.
8. **Compile steps as the library's own programs.**
   - Any preparation step over tuples that takes longer than a few register and shared-memory round trips is a SIMD or
     SIMT program: the frontier sweep (item 1), the zelloth run tables (ec69f367), the plan's step lists. The library
     runs such kernels already; these steps are declared and run through it, over a node's 64 GPU cores or its 30 CPU
     cores, and across nodes where their state is megabytes.
9. **Reaching the nodes.**
   - Tooling reaches a node by its tailnet identity through one module (mesh_nodes) that checks the path first: an idle
     tailnet path left connects timing out for hours on 11-01 until `tailscale ping` made it direct again. No script
     derives routes of its own.

## Order

Item 2 first: it ends a failure that costs seconds to minutes a start. Then 1 and 4, both contained in one file each.
Then 5, which is the bridge's lifecycle and is written in the bridge by hand (no subagent), one port at a time, with
the B9 hold kept wherever a link is lost. Item 6 follows from the others; 3 and 8 alongside.
