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
The crash log alone does not identify which signal generated the invalid event-port message.

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

## Compiler responsibilities still under investigation

The build floor and shader language now come from `etc/compute/metal.mk`; the provider's queue-pair reservation comes
from `etc/bridge/thunderbolt.mk`. Live device queries supply differing port totals. No OS-27-specific branch selects
an algorithm. Metal 4's queue API is available from macOS 26.0 in the installed SDK.

Separately prepared invocations can reuse output pages and transport counters. The current access-order compiler
tracks hazards inside one command encoder. Metal 4's encoder barrier does not order earlier encoders or command
buffers. The compiler still needs to carry those inter-invocation data relations into its prepared commands. Adding a
blanket host drain would not fix that omission appropriately. Apple's
[Metal 4 API description](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)
distinguishes explicit resource synchronization from submission order.

Cooperative chain grids also use a GPU core count as a residency assumption. Their per-pipeline thread and memory
checks do not establish simultaneous residency of every group. This needs a supported execution constraint or a
compiled decomposition using the existing product kernels. It cannot explain bandwidth-only failures by itself.

The bridge, collective library, Torch extension and native Engine module compile locally. Rank and Goal A pass Swift
type checking. Mach-O metadata for the collective library and Torch extension reports minimum macOS 26.0, SDK 26.4.
These checks do not establish that the historical GPU restart causes have all been removed. No numerical workload
has been launched after the pause, and no performance recovery is claimed here. These changes have not been deployed
to the cluster; the laptop's bridge transition also remains pending.
