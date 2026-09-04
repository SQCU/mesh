# Bridge and shared-memory control flow

Each live region has one `mesh-flow` process. That process owns the verbs context,
protection domain, queue pair, completion queue, registered shared-memory mapping, and
TCP pairing listener. The process attached to that region owns application submission
and completion consumption. Workload code remains outside the bridge.

This boundary prevents every strategy, OCR, or rendering process from becoming a new
verbs and host-firewall identity. It also keeps all tensor meaning above the transport:
the bridge sees pages, source and destination node identities, hop count, ownership,
and byte mass only.

## Realized region

The bridge derives a page count from the selected fraction of physical memory. It creates
a POSIX shared-memory object, sizes it for the control header plus pages, maps it shared,
and registers the page range in one-GiB memory regions. The page relation is divided into:

- a receive/forward pool containing one quarter of the pages;
- an application arena containing the remaining three quarters;
- four 65,536-entry descriptor rings for submission, completion, release, and send
  acknowledgement.

The one-GiB registration extent, 4,095-entry verbs working set, and ring depths are
resident scheduling quantities. They do not bound a datum. The application queue grows
with submitted row mass, and arbitrary-extent streams repeatedly consume page credits.

The command surface is exactly:

```text
mesh-flow [-I node] [-M percent] [-s region] [peer]
```

`-I` supplies the node identity carried on wire pages. `-M` supplies the physical-memory
share realized as the region. `-s` supplies the POSIX region name. `peer` supplies a
pairing address; without it the same bridge listens and accepts. Device selection finds
the active verbs port. There is no duration, page-size, queue-depth, tensor-size, team,
player, cart, expert, or workload switch.

## Page ownership and flow

The page ownership states are FREE, RECV, SEND, APP, and NOWN. Their transition is:

```text
FREE -> RECV -> APP -> FREE
FREE -> RECV -> SEND -> FREE
APP arena slot -> SEND -> application acknowledgement
```

The bridge continually posts FREE pool pages as receives until the queue-pair working set
is full. A received page addressed to this node enters APP and is published through CMP.
A received page addressed to a later node is sent onward and its hop count advances. An
application returns a consumed receive page through REL. Completion of an application
arena send publishes ACK so the library can reuse that arena slot.

The `wire` header contains only 16-bit source, destination, and hop values. Payload bytes
remain unchanged. The transport does not pack target identities, reduce feature rows,
normalize tensors, choose experts, or interpret policy data.

## Pair continuity

The pairing listener is raised before verbs setup. Pairing failure backs off and retries.
Loss of the active port or absence of completions while sends are outstanding tears down
the pair resources and returns to the same pairing loop. The shared-memory region and its
application attachment remain alive through that loop.

SIGINT and SIGTERM set the bridge stop state. Orderly exit destroys the queue pair and
completion queue, deregisters every memory region, deallocates the protection domain,
closes the verbs device, and unlinks the region. The repository kill guard protects this
teardown path from uncatchable process death as specified by [`../RDMA-RULES.md`](../RDMA-RULES.md).

If the attached application process disappears, the bridge releases its APP-owned receive
pages, advances abandoned ring tails, and leaves the transport available for the next
application. A new application records its process identity when it attaches.

## Measurement surface

The shared header publishes cumulative sent, received, and malformed-page counts, bridge
uptime, current application identity, and per-second mean and sample deviation of every
page ownership population. `mesh-stat` and the node telemetry service read these fields.
Whole-fabric aggregation and leases are specified in
[`mesh-coprocessor-demo.md`](mesh-coprocessor-demo.md); the application ABI is specified in
[`abi-streams.md`](abi-streams.md).

The release claim is not merely that bytes crossed one cable. It is that every reachable
leased node contributes its literal capacity, fabric counters, workload measures, and
performance bounds while the bridge remains ignorant of the workload that produced them.
