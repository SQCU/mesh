# Xonotic server ↔ external solver bridge

Implements option 1 from the bridge survey: a POSIX shared-memory region with a
seqlocked SPSC ring, exposed to QuakeC as ten new darkplaces builtins (#644–#653).
Nothing on the hot path makes a syscall, takes a lock, or waits on an external
condition, so the server frame cannot be extended by a slow or dead solver.

No RDMA/verbs device is opened anywhere in this tree. The solver process is the only
thing that would ever touch the fabric; the engine only ever touches mapped pages.

## Layout

```
engine/mesh_shm.h                 the ABI. shared verbatim by engine and solver
engine/mesh_ipc.c                 the ten builtins
engine/0001-mesh-shm-builtins.patch   66 lines against darkplaces, 4 files
solver/mesh_attach.h              create/attach for the solver side
solver/fakesolver.c               reference solver, computes out[i] = in[i]*2+1
qc/mesh_ipc.qh                    QuakeC declarations for the ten builtins
qc/meshtest.qc                    in-engine test, drives the bridge from StartFrame
test/meshtest.c                   standalone transport test + fake solver, no engine
test/run.sh                       size/rate sweep, C only
test/engine.sh                    patch, build darkplaces, build progs.dat, run
apply.sh                          applies the engine patch to a darkplaces tree
mkpatch.sh                        regenerates the .patch from a patched tree
```

## The QuakeC interface

```
float(string name, float nreq, float nresp)                    mesh_open    = #644;
void (float h)                                                 mesh_close   = #645;
void (float h, float i, float v)                               mesh_set     = #646;
float(float h, float i)                                        mesh_get     = #647;
void (float h, float dst, .float fld, float first, float n)    mesh_gather  = #648;
void (float h, float src, .float fld, float first, float n)    mesh_scatter = #649;
float(float h)                                                 mesh_publish = #650;
float(float h)                                                 mesh_poll    = #651;
float(float h, float seq, float usec)                          mesh_wait    = #652;
float(float h, float sel)                                      mesh_stat    = #653;
```

The intended per-tick shape is four calls, none of which block:

```
mesh_gather(h, 0, myfield, firstbot, nbots);
seq = mesh_publish(h);
if (mesh_poll(h) > lastseen) { lastseen = mesh_poll(h); mesh_scatter(h, 0, outfield, firstbot, nbots); }
```

`mesh_gather`/`mesh_scatter` are the load-bearing primitives. A `.float` parameter
arrives as a raw field offset and `prog->edictsfields` is a flat array of stride
`prog->entityfields`, so one builtin call strided-copies one component across every
bot. 64 components is 64 calls, not 64 × nbots interpreted QC operations.

`mesh_wait(h, seq, usec)` is the only call that can spin, and it is bounded by a
deadline computed from `Sys_DirtyTime()` before the loop starts. It exists because
the survey measured that an in-frame round trip is affordable; the demo does not
have to use it, and `qc/meshtest.qc` does not.

`mesh_stat` selectors: 0 published, 1 delivered, 2 in flight, 3 nreq, 4 nresp,
5 solver-side served counter, 6 seqlock retries, 7 ring depth.

## Why there is no string on the payload path

Every engine→QC string goes through `PRVM_SetTempString`, whose first act is
`strlen(s) + 1`; `VM_fputs` and `VM_bufstr_add` `strlen` in the other direction.
The PRVM has no length-carrying string. An fp32 array is full of zero bytes —
`0.0f` is four of them — so every existing builtin path silently truncates.

The standalone test prints the concrete instance of this before it runs anything:

```
nul-hazard: first zero byte in a zeroed fp32 request at offset 0 of 16384
            -> any strlen-based builtin would move 0 of 16384 bytes
```

The bridge does not use strings for payload at all. The only string that crosses is
the region name in `mesh_open`, which is a real C string. That is the whole of the
mitigation: the hazard is designed out rather than escaped around, which also avoids
the 0.15–0.79 ms of QuakeC that ASCII encoding costs per 4096 floats.

## Transport

```
mesh_hdr_t          magic, version, nreq, nresp, depth,
                    req_seq, resp_seq, solver_alive, epoch   (each on its own 64B line)
MESH_DEPTH request slots     each: s0, s1, nreq floats
MESH_DEPTH response slots    each: s0, s1, nresp floats
```

Sequence numbers are monotonic and never wrap in practice (uint64). Slot index is
`seq % MESH_DEPTH`, depth 4.

Each slot is a **seqlock**: the writer stores `s0 = seq`, releases, copies the
payload, then release-stores `s1 = seq`. A reader acquire-loads `s1`, copies, then
checks `s0`. A read that raced a slot being recycled fails and is retried up to
`MESH_DEPTH` times; a permanently-losing reader returns the previous sequence rather
than spinning. This is what makes it safe for the server to publish faster than the
solver consumes: the solver never observes a half-written request, it just misses one.

The engine keeps a private `req[]`/`resp[]` scratch pair per handle. `mesh_set` and
`mesh_gather` write scratch; `mesh_publish` copies scratch into the slot. So QuakeC
never reads or writes shared memory directly and can never observe a torn slot.
The cost is one 16 KiB memcpy per publish, measured below at 0.15–0.36 µs.

**Epoch.** The header carries a generation counter bumped by `mesh_reset` on every
`mesh_open`. Without it, a solver started against a region left behind by a previous
server run latches onto the stale `req_seq`, and then ignores every request the new
server publishes until the sequence climbs back past it. This is not hypothetical —
it is what the first engine integration run did, silently, for 30 seconds
(`served=1`, `delivered=0`, no error anywhere). The solver now resets its cursor when
the epoch changes.

## Measured — standalone transport (`test/run.sh`)

Host side is the same `mesh_put_req`/`mesh_get_resp` code the engine builtins call.
"in-frame bridge cost" is wall time from the start of the request fill to the end of
the poll, i.e. what a server frame actually pays.

M5 Max, 4096 floats (16 KiB) each way, 600 ticks at 60 Hz:

| mode | bridge cost med | p90 | p99 | max | publish→deliver med |
|---|---|---|---|---|---|
| pipelined (never blocks) | 3.00 µs | 4.67 | 8.37 | 51.3 | next tick (16.67 ms) |
| in-frame blocking (`mesh_wait`) | 6.04 µs | 8.96 | 24.08 | 371 | 6.04 µs |

M4 Pro mini, same:

| mode | bridge cost med | p90 | p99 | max |
|---|---|---|---|---|
| pipelined | 2.50 µs | 2.75 | 3.25 | 6.79 |
| in-frame blocking | 4.58 µs | 5.21 | 5.75 | 11.54 |

Size sweep, in-frame blocking round trip (median, µs):

| floats each way | bytes | M5 Max | M4 Pro |
|---|---|---|---|
| 512 | 2 KiB | 2.37 | 0.92 |
| 4096 | 16 KiB | 6.04 | 4.58 |
| 16384 | 64 KiB | 19.62 | 18.50 |
| 65536 | 256 KiB | 71.92 | 67.00 |

Saturation (no tick pacing), aggregate payload moved both directions:

| case | M5 Max | M4 Pro |
|---|---|---|
| 16 KiB, 20000 round trips | 3858 MB/s, 4.83 µs med RT | 4252 MB/s, 4.33 µs med RT |
| 256 KiB, 5000 round trips | 3919 MB/s, 74.4 µs med RT | 4260 MB/s, 70.7 µs med RT |
| 16 KiB pipelined, 200000 ticks | 5335 MB/s | 5593 MB/s |

The M5 Max carried a load average of ~14 from unrelated jobs for this entire
session. Its dispatch-bound numbers are therefore pessimistic and noisier than the
mini's, which is why the smaller machine appears faster at small sizes. The
throughput rows, which are memcpy-bound, are not affected in the same way.

### Non-blocking, demonstrated

Solver deliberately made slower than the tick, 4096 floats, 60 Hz, pipelined:

| solver work per request | frame bridge cost med | responses in 600 ticks |
|---|---|---|
| 0 µs | 3.00 µs | 599 |
| 5000 µs | 3.04 µs | 599 |
| 30000 µs (1.8× the tick) | 2.88 µs | 332 |

The frame cost does not move. A solver that falls behind costs responses, never
frame time. At 500 Hz publish against a 30 ms solver (15 publishes per response) the
frame cost was still 2.13 µs median.

Solver process exits after 3 s of a 10 s run, 600 ticks at 60 Hz:

| mode | frame bridge cost med | max | ticks completed | responses |
|---|---|---|---|---|
| pipelined | 2.29 µs | 15.3 µs | 600/600 | 180 |
| `mesh_wait`, 4000 µs cap | 4000.00 µs | 4005.13 µs | 600/600 | 180 |

The pipelined server does not notice. The blocking server pays exactly its declared
deadline per frame and no more — 4005 µs observed against a 4000 µs cap — and still
completes every tick. That is the bound in `RDMA-RULES.md` holding under the failure
it exists for. The solver was stopped by letting its own deadline expire; nothing in
this tree was sent `SIGKILL`.

### Seqlock, demonstrated

Unpaced pipelined run, 200 000 publishes, host and solver both free-running so the
4-deep ring recycles constantly:

```
M5 Max:  200001 published, 189218 responses verified, 0 mismatches,
         47 solver-side seqlock retries, 2 host-side retries
M4 Pro:  200001 published, 199524 responses verified, 0 mismatches
```

Every delivered response is checked element-by-element against `in[i]*2+1` for the
request it answers, including the deliberately planted `0.0f`, `-0.0f` and
`65536.0f` entries. Zero mismatches across every run in this document.

## Measured — inside a real darkplaces dedicated server (`test/engine.sh`)

`darkplaces-dedicated` built from the shipped Xonotic source with the patch applied,
arm64, `make sv-release`, zero warnings from the new code under the project's
`-Wall -Wold-style-definition -Wstrict-prototypes -Wsign-compare
-Wdeclaration-after-statement -Wmissing-prototypes`. (The two warnings the build does
emit for `prvm_cmds.c` are pre-existing, in `VM_bufstr_find`.) `progs.dat` built with
`gmqcc`. 512 spawned edicts carrying a `.float slot`; the fake solver runs as a
separate process.

Builtin cost measured inside the QuakeC VM, 65536 repetitions each:

| | M5 Max | M4 Pro |
|---|---|---|
| `mesh_gather` 512 edict fields | 0.179 µs | 0.298 µs |
| `mesh_scatter` 512 edict fields | 0.179 µs | 0.268 µs |
| `mesh_publish` 16 KiB | 0.179 µs | 0.149 µs |
| `mesh_poll` | 7.8 ns | 9.8 ns |
| **full cycle** gather+set+publish+poll+scatter | **0.775 µs** | **0.596 µs** |

600 server frames at `sys_ticrate 0.0166`, gathering 512 bot fields, publishing,
polling and scattering the answer back into a second field every frame:

```
M5 Max:  frames 600  responses 600  mismatches 0  delivery lag 1 seq (worst 1)
         seqlock retries 6 host / 220 solver   solver served 21595
M4 Pro:  frames 600  responses 600  mismatches 0  delivery lag 1 seq (worst 1)
         seqlock retries 1 host / 0 solver     solver served 20851
```

The answer to a request published in frame N is scattered into edict fields in frame
N+1, every frame, with no dropped frames and no mismatched values. The bridge costs
0.6–0.8 µs of a 16.6 ms frame, which is **0.005%**.

For contrast, the survey measured the existing stringbuffer path at 0.95 ms of
QuakeC per round trip on the mini. The full cycle here is **~1600× cheaper** and,
unlike that path, it is not truncating the data.

## Things that are true and worth knowing

- The builtins are registered in **both** `vm_sv_builtins` and `vm_cl_builtins`, at
  the same numbers, so CSQC can open a region too. The handle table is one static
  array in `prvm_cmds.c`, shared by both VMs in a listen server. Two VMs opening
  regions get distinct handles and cannot collide by accident, but a QuakeC bug that
  passes SVQC's handle number from CSQC will reach SVQC's region. Documented, not
  guarded.
- `MESH_MAX_REGIONS` is 8. The region is two flat float arrays; the engine has no
  opinion about what the floats mean. Request/response layout is entirely QuakeC's.
- `mesh_open` on macOS calls `ftruncate` on a shm object that may already exist, where
  it fails with EINVAL. That is harmless — the object is already the right size — and
  the return is deliberately not checked, because refusing to open would withhold the
  capability for no gain.
- The engine writes `epoch` and `magic` on open; the solver attaches by polling for
  `magic`, so start order does not matter in either direction.

## What I could not do

- **No RDMA, no verbs, no `mesh-flow`, no `ibv_*`, nothing linked against `-lrdma`.**
  Every number here is same-machine shared memory on each host separately. The
  Thunderbolt leg is untested by me. The design puts that leg entirely inside the
  solver process: the same mapped pages the solver reads are the pages the NIC would
  write, so the engine side does not change when the solver becomes remote — but that
  claim is a design property, not a measurement.
- **The bridge was never run under Xonotic's own `progs.dat`.** The engine test uses a
  minimal id1-style progs on a stub map. Nothing in Xonotic's QuakeC calls these
  builtins yet; wiring them into the payload gamemode is separate work.
- **No client/CSQC run.** The CSQC table entries compile and link, and the code is the
  same, but I only exercised the server VM. A listen-server test with both VMs open
  on one region has not been done.
- **The per-frame `gettime(2)` accumulation in `qc/meshtest.qc` is unreliable** and I
  am not reporting it. It read 1.8–10.4 µs across runs while the 65536-iteration loop
  measurement of the identical work was steady at 0.6–0.8 µs. `gettime(GETTIME_HIRES)`
  returns a `float` and is documented to reset between QC invocations; the loop
  measurement is the one to trust.
- **The M5 Max was under load average ~14 throughout** from unrelated jobs on the
  machine. I did not stop them. Dispatch-bound rows on that machine are pessimistic.
- I tested solver death only by clean exit, not by `SIGKILL`. Per the operating rules
  nothing here was `kill -9`'d. A hard kill leaves the same shared memory in the same
  state as a clean exit — the solver holds no kernel object the server observes — but
  I did not run it.
