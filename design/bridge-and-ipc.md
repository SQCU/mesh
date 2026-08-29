# One bridge, many applications

A ruling and the measurements that support it. Both matter, because the previous plan —
collapse the transport and the workload into one process — was wrong, and the hardware says
so rather than merely taste.

## The ruling

One **bridge** process per node owns every RDMA link, every queue pair, every memory region,
and the single out-of-band bootstrap. It is the only binary on the node with a network
identity. It drives k links with k threads, not k processes.

Applications are **separate processes** and are **subordinate to the mesh**: they operate on
buffers the mesh moves, and the mesh does not operate inside them. There is no inversion of
control in which the bridge hosts an application in a callback. Cross-process IPC is therefore
a mandatory feature, not an optimisation.

The mesh is **not** a single global virtual address space across k nodes. Data is better
processed as it moves. The dataflow-compilation questions that follow from that are deferred,
not answered here.

## What the hardware says

Measured on `rdma_en2`, Thunderbolt, `PORT_ACTIVE`.

| resource | scope | budget |
|---|---|---|
| queue pairs | **per device** | 10; a second protection domain gets **zero** |
| memory regions | **per protection domain** | 100 per PD; the advertised 16.384 MB size limit is not the real one |
| registered bytes | bounded by wired memory, not by the quota | at a 1 GiB chunk, one PD registers 100 GiB — more than any node here has |

`max_qp: 11` is advertised; 10 are obtainable. The second PD returning zero QPs is the whole
argument: queue pairs are a device-wide resource of about ten. A design in which each
application opens its own queue pairs exhausts the device at ten applications, and every one
of them is a separate binary with its own out-of-band socket and its own host-firewall
identity. One bridge is what the device permits.

Memory registration behaves oppositely. The advertised `max_mr: 100` is a per-PD quota, so the
bridge allocates protection domains to buy registration capacity. A substantial fraction of
system memory as addressable mesh memory is therefore available, bounded by wired memory
rather than by the quota.

`max_mr_size` advertises 16.384 MB. That number is wrong in the dangerous direction: it is
not enforced, and it is far too small to use, since honouring it would need 1416 regions for
a 23 GB pool against a quota of 100. Regions of 0.258, 2.749 and 3.092 GB were verified
carrying real traffic with `corrupt=0`; 5.154 and 8.246 GB corrupt every page. The cliff is
2^32. The bridge chunks at a fixed 1 GiB and the size is not selectable, because an operator
who picks it wrongly gets a run that looks healthy and moves corrupt data.

Chunking is arithmetic performed once at setup, not a per-datum protocol.

## The IPC mechanism

POSIX shared memory. `shm_open` + `ftruncate` + `mmap(MAP_SHARED)`, and the result **registers
successfully with `ibv_reg_mr`** — verified, 100 MRs over one shm object. The NIC therefore
DMAs directly into memory that an unrelated process has mapped. No file on disk, no page-cache
writeback, no rendezvous path pretending to be storage.

For array workloads the region is handed across two standards, because one does not suffice:

- **into a framework**: `mx.from_dlpack(numpy_view_of_region)` — zero copy, and writes made by
  the NIC are visible live.
- **out of a framework**: `np.asarray(memoryview(arr))` — a writable aliased view; 21.2 us for
  16 MB against roughly 2500 us for the copy. MLX's DLPack *export* fails with "Unsupported
  device in DLTensor", so the buffer protocol carries this direction.

## What this reverses

The previous entry proposed deleting the submission and completion rings along with the free
list. That was reasoned from a single-process design and is withdrawn. Cross-process
submission needs exactly such a channel, and the rings are the right shape for it; they live
in the shared region.

What was actually wrong with the earlier ABI was narrower and stands corrected: it was
file-backed for want of a name, and the workload was statically linked into the transport
through `mesh_f`. The first is replaced by POSIX shared memory. The second is replaced by the
application being its own process.

## Residency is the page table's job, and it is free

An earlier draft of this file framed a dilemma: either register the whole 80% of a node at
once, needing thousands of memory regions across dozens of protection domains, or land data
in a small registered ring and copy it out. Both branches are wrong, and the page table the
data plane already has is the reason.

Memory regions are frames, not a static allocation of the address space. The node addresses
its full share; the 100 MRs per protection domain are the **resident set**; the page table
maps a mesh page to a frame or to nothing. Making a page resident is `ibv_reg_mr` over a
different range, so the frame is rebound and **the data never moves**. That is address
translation, not copying, and it is why the copies-per-datum rule is not threatened by a
resident set smaller than the address space.

Measured on `rdma_en2`, 16.384 MB frames, pages pre-touched (the rate is per byte, so it
holds at the 1 GiB chunk the bridge actually uses):

| operation | per frame | aggregate |
|---|---|---|
| `ibv_reg_mr` | 0.105 ms | **155 GB/s** of residency turnover |
| `ibv_dereg_mr` | 0.002 ms | effectively free |

The link moves 51.85 Gbit/s, or 6.5 GB/s. Residency rebinds **24x faster than the wire
delivers**, so residency management cannot be the bottleneck and does not need to be
designed around.

What actually bounds the resident set is wired memory. Registration wires pages roughly 1:1
(32 GB registered raised wired memory by 32 GB), and `vm.global_user_wire_limit` defaults to
76-85% of RAM depending on the machine. That limit, not any RDMA quota, is what decides how
much of a node the mesh can hold, and it is the one number an operator may need to change.

The MR quota therefore bounds nothing that matters: at a 1 GiB chunk a single protection
domain covers 100 GiB. Addressable memory is bounded by RAM and by the wire limit. On a
24 GiB mini a 23.193 GB region was registered fully resident in 22 regions, giving an arena
of 21.989 GB, or 85.83% of the machine.


## The application surface

Three functions, in `mesh.h`. Nothing below a page appears in it, and no input to either the
library or the bridge can select a configuration that moves corrupt data.

```c
void  *mesh_open(size_t bytes, size_t *stride, size_t *usable);
size_t mesh_write(const void *p, size_t nbytes, int node);
size_t mesh_read(void **p, int *from);
```

`mesh_open` maps a share of the node and returns memory. `mesh_write` hands a range to
another node and returns how much it took, so a caller loops on the remainder and there is no
size at which it fails. `mesh_read` returns one arrived slot and recycles the previous one,
which is why there is no release call.

Memory arrives in slots rather than one flat span because the device reports `max_sge: 1` and
has no immediate data, so every page must carry its own header contiguously. The gap is 24
bytes in 4096. That is page granularity, which is the floor this API was asked to expose.

The bridge accepts only what it cannot infer: `-I` for the node index, `-M` for the share of
the machine, `-s` for the region name, `-T` for a duration, and a peer address. There is no
switch for the page size, the registration chunk, the send window, the span size or the
device. Each of those can select a run that looks healthy and delivers wrong data, and an
operator should not be able to reach that. Unknown options are refused rather than ignored,
and the device is chosen by finding the port that is up.

## Throughput does not fall with the sender's working set

An earlier revision recorded a large fall: 57-62 Gbit/s from a 1-2 GB working set, 2.6 from
9 GB, 2.15 from 21.9 GB, same bridge and same code path, with three explanations tested and
none surviving.

It does not reproduce. Measured on the current transport, 0.5 GB gives 20.88 Gbit/s and 3.0 GB
gives 21.92 — no fall, and the larger set is marginally faster. Three gigabytes is the arena
size at this configuration, so the original observation at 9 and 21.9 GB has not been
re-tested at those sizes.

The likely cause is the machinery that was deleted between the two measurements. Those
readings were taken while the software loss-repair path was active, and that path invented
phantom sequence gaps and drove retransmit traffic from them, at a rate that grew with how far
the sender's sequence numbers had advanced. That is a mechanism whose cost scales with the
working set, and it no longer exists.
