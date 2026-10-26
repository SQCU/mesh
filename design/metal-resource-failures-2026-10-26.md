# Metal resource failures and mixed-host configuration

Recorded 2026-10-26, 18:06 UTC. Numerical execution and the canonical hourly remain paused at the operator's request.
This record distinguishes observed failures from source defects found during the investigation.

## Hosts observed

| Node | Device | macOS | Paired ingress ports |
|---|---|---|---:|
| 521750041 | Studio, Mac17,15 | 27.0 / 26A425 | 6 |
| 455770905 | Studio, Mac17,15 | 27.0 / 26A425 | 6 |
| 1974797816 | Mini, Mac16,11 | 26.5.1 / 25F80 | 3 |
| 1980260065 | Laptop, Mac17,6 | 26.3.2 / 25D2150 | 1, reported by its Studio peer |

The map contains eight cables. The laptop's running bridge dates from October 22. Its region reports layout 138;
current clients require 154. Its appearance in the map does not mean current clients can attach to it. `mesh-stat`
now reports the actual and expected layouts on failure. The hourly bandwidth configuration explicitly requires all
four nodes, so the existing installation check cannot silently measure a smaller collective.

## Mini evidence and source boundaries

The Mini rebooted at 10:38 local time on October 26. Read-only kernel logs before that reboot show:

* 07:59:58: command-queue creation attempts numbered into the thousands; the driver reports a limit of 512.
* 08:04:40: `kIOGPUKernelCommandSignalSharedEvent: bad event port name`.
* Repeated GPU timeouts and later submissions ignored after two GPU restarts, continuing through 10:32.

`d4916d1e` introduced prepared schedule and `TorchSchedule` interfaces. The latter could create a queue for each
invocation, plus two events. `5019e93d`, used by the 14:43 UTC hourly, still had that constructor.
`70884019` moved its queue to the shared stream context. Later hourlies still recorded GPU timeouts: that fix did
not resolve the whole failure sequence. `7f5886e0` removed `TorchSchedule` and its per-crossing framework handshake.
The crash log alone does not identify which signal generated the invalid event-port message. In `5019e93d`,
`MetalScheduleEnqueue` accessed `s->signal` after `commit`; the completion callback could already have released
that schedule and event. This supplies a source-level stale-event mechanism. The later removal of the queue signal
API and the post-commit access removes it. Attribution of that particular kernel log line remains unconfirmed.

The hourly driver continued launching workloads after a failed one. The 15:19, 16:28 and 17:28 UTC records show a
bandwidth GPU timeout followed by further workloads and submissions-ignored errors. It now ends the run at the first
failed whole-function workload. This removes a source of repeated launches after GPU failure; it does not fix the
first GPU timeout or cancel commands already submitted by that function.

The remaining `MetalScheduleCreate(device, form)` API is now removed. `MeshRankPrepare` must consume an existing queue,
as the FX path already does. The unused `MetalScheduleSignal` and `MetalScheduleHost` interfaces are removed too.
Only declared host producers create host-result events. Those events are retained by their consumers and schedules.

A separate source race existed in `MetalScheduleEnqueue`: after committing a command it read the schedule's successor.
Completion can resume the owner on another thread and release the schedule before that read. The successor is now
read before commit. This adds no wait or allocation. It is a source-level lifetime correction, not a demonstrated
explanation of the historical kernel message.

## Compiler responsibilities and corrections

The build floor and shader language now come from `etc/compute/metal.mk`; the provider's queue-pair reservation comes
from `etc/bridge/thunderbolt.mk`. Live device queries supply differing port totals. No OS-27-specific branch selects
an algorithm. Metal 4's queue API is available from macOS 26.0 in the installed SDK.

Separately prepared invocations can reuse output pages and transport counters. The bandwidth example enqueues its
prepared cases without a host join. The former access-order compiler tracked hazards only inside one command encoder,
so later cases could overtake accesses to the same bridge lists and publication counts. Metal 4's encoder barrier does
not order earlier encoders or command buffers. Apple's
[Metal 4 API description](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)
distinguishes explicit resource synchronization from submission order.

The common compiler now keeps declared access ranges across preparation on the supplied queue. Numeric commands
use their existing binding/reach metadata. Descriptor publication declares the list slots and counter words it
accesses. The first overlapping access involving a write gets a queue-stage dependency; disjoint accesses do not.
This is prepared GPU ordering, with no host wait or runtime access lookup. Weak allocation keys let completed
functions release their pages. Preparation and enqueue order must agree, as the bridge's compiled counts require.

Another source defect affected functions split around CPU/ANE shares: the last command buffer's callback reported
completion even if an independent earlier buffer was still running. Completion now counts all prepared parts. The
callback counter observes completion and does not gate execution. Buffers and events remain owned through that point.

Cooperative chain grids also used a GPU core count as a residency assumption. Their per-pipeline thread and memory
checks did not establish simultaneous residency of every group. The chain now compiles its existing input/join and
two product stages separately. Their X/Y access declarations order the stages. The numerical bodies and direct
publication remain; the global spin barrier and its stamp allocation are removed. This permits groups to run in waves.
The wider geometry still uses the existing linear product kernels. This defect cannot explain bandwidth-only failures.

The bridge, collective library, Torch extension and native Engine module compile locally. Rank and Goal A pass Swift
type checking. Mach-O metadata for the collective library and Torch extension reports minimum macOS 26.0, SDK 26.4.
The revised chain phases also compile offline to Metal AIR. These checks establish build compatibility, not runtime
performance. No numerical workload has been launched after the pause and no performance recovery is claimed here.
The source mechanisms above have been addressed; committed deployment and the laptop's bridge transition remain
before the next canonical run.
