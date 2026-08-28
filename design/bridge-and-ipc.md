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
| memory regions | **per protection domain** | 100 per PD, 16.384 MB each |
| registered bytes | scales with PDs | 10 PDs → 1000 MRs → **16.384 GB**, linear, no ceiling hit |

`max_qp: 11` is advertised; 10 are obtainable. The second PD returning zero QPs is the whole
argument: queue pairs are a device-wide resource of about ten. A design in which each
application opens its own queue pairs exhausts the device at ten applications, and every one
of them is a separate binary with its own out-of-band socket and its own host-firewall
identity. One bridge is what the device permits.

Memory registration behaves oppositely. The advertised `max_mr: 100` is a per-PD quota, so the
bridge allocates protection domains to buy registration capacity. A substantial fraction of
system memory as addressable mesh memory is therefore available; 16.384 GB registered with no
sign of a wall and 87% of system memory still free.

`max_mr_size` of 16.384 MB forces chunking. That is arithmetic performed once at setup, not a
per-datum protocol.

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

Measured on `rdma_en2`, 16.384 MB frames, pages pre-touched:

| operation | per frame | aggregate |
|---|---|---|
| `ibv_reg_mr` | 0.105 ms | **155 GB/s** of residency turnover |
| `ibv_dereg_mr` | 0.002 ms | effectively free |

The link moves 51.85 Gbit/s, or 6.5 GB/s. Residency rebinds **24x faster than the wire
delivers**, so residency management cannot be the bottleneck and does not need to be
designed around. A resident set of 1.638 GB in front of a 6.5 GB/s link is a quarter second
of traffic, turned over at 24x the rate it drains.

The MR quota therefore bounds the working set, not the addressable region. Addressable
memory is bounded by RAM and by `vm.global_user_wire_limit` (116.8 GB here, 85% of RAM),
both of which sit above the 80% target.
