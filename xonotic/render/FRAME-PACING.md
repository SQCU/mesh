# Live client pacing review, September 4, 2026

The compiled SDL client is connected to the persistent mixed-policy curriculum as
`mesh-off-policy`, at 1280×720 with vsync requested. A server-observed human row
confirms participation; this is not inferred from process existence.

## Measurement

`cl_frame_report 1` enables passive, bounded batches in the ordinary console/log:

- Intervals between returns from `VID_Finish`, including intervening simulation,
  scheduling, rendering and presentation waits.
- Screen submission duration before `VID_Finish`, and time spent inside that call.
- Rendered particle/draw counts, window activity/visibility and adaptive quality.
- Engine time, batch clock, resolution, requested vsync, field labels and schema.

The hook does not call `glFinish`, does not add GPU synchronization and does not
write a file per frame. It flushes at one second or the bounded console buffer limit.
Renderer counters are copied before `R_TimeReport_EndFrame` clears them. Schema 1
identifies this ordering. The initial unversioned diagnostic generations read those
counters too late; `frame-report.py` marks their counters unavailable rather than
interpreting them as zero-particle evidence.

Summarize actual logs from the repository root:

```
bin/mesh-python xonotic/render/frame-report.py xonotic/solver/strat/runs/joint-live-20260904/00039-joint-00007-00000/client-0.log
```

The report partitions by focus, resolution, vsync and particle count; gives p50/p95/p99,
maximum interval, submission/swap times, draw counts and strict 30/60 Hz budget
exceedance fractions. The 0, 1–63 and 64+ particle bins are descriptive groupings, not
causal interventions. Initial loading and focus-transition tails remain in the data.
Regular visible renders are covered; hidden/loading-only render paths are not a
physical-display presentation trace. Vsync requested is not proof of a display rate.

## Before the quality-feedback correction

The complete schema-1 launch `00039-joint-00007-00000` retained 4,920 frames with
zero malformed batches:

| Scope | Frames | Interval p50 / p95 / p99 | Particle maximum |
| --- | ---: | --- | ---: |
| Active, particles present | 111 | 8.435 / 12.968 / 15.916 ms | 23 |
| Active, no particles | 1,077 | 8.370 / 12.122 / 16.704 ms | 0 |
| Background, particles present | 676 | 50.005 / 53.767 / 57.960 ms | 48 |
| Background, no particles | 3,056 | 50.001 / 53.056 / 55.881 ms | 0 |

The active particle sample had a maximum interval of 18.847 ms and no interval over
33.333 ms. Its screen-submission p95 was 2.045 ms and swap p95 10.194 ms. The
non-particle active scope includes a 2.495 s startup/loading gap; it must not be
presented as a particle hitch. This is limited particle exposure, not a heavy-burst
benchmark and not proof of stable frame pacing for all gameplay.

A preceding five-second native sample put about 76% of main-thread samples in
swap/presentation paths. That identifies where sampled wall time was spent, not a
particle CPU bottleneck or the number of missed display deadlines. The older
`timerefresh` throughput captures likewise do not establish live frame pacing.

## Concrete controller error and correction

`cl_maxidlefps` defaults to 20 while `cl_minfps` targets 40. The old quality feedback
used start-to-start elapsed time, including the scheduler's deliberate wait. In the
background sample above, quality fell to 0.25 even though screen submission took
roughly 2 ms. Reducing draw distance cannot make an intentional 20 FPS cap satisfy
a 40 FPS target.

The controller now measures the preceding screen/presentation duration, excluding
the between-frame scheduler wait, and clips its target to the applicable explicit
frame cap. Rendering over budget can still reduce quality. This changes neither
particle count nor rendering features, introduces no forced synchronization, and
leaves frame-cap preferences untouched. Timedemo and video-capture timing retain
their separate scheduling contract.

New control flow only controls optional measurement publication and the arithmetic
interpretation of an existing scheduling cap. It does not skip rendering, simulation,
network work or accessibility. The ordinary client build succeeds; the live supervisor
picks up each executable on its normal next launch without changing either RDMA bridge.

## Live verification of the correction

A 10,278-frame sample from launch `00041-joint-00008-00000` had zero malformed
batches. Its 549 background frames retained quality 1.0 throughout while still
honoring the existing approximately 50 ms frame interval. This verifies the
quality/pacing feedback correction; it does not change the 20 FPS preference.

The 540 active frames with at least 64 rendered particles reached 655 particles;
interval p95 was 11.702 ms, p99 15.211 ms and maximum 16.374 ms. This scene did not
reproduce the suspected particle-heavy slowdown. A separate low-particle active
frame did take 127.705 ms, so occasional hitches remain and the client is not being
declared hitch-free. That frame had three particles, 1.171 ms of screen submission,
13.410 ms in `VID_Finish`, and 113.124 ms outside those regions. The residual includes
simulation, input/network handling, scheduler waits and other between-frame work;
it does not identify which one caused the hitch. These are successive live matches, not identical-scene A/B
measurements; their differences cannot be credited wholesale to this patch.
Client timing evidence is from this MacBook/display; a second rendering host/display
has not been measured. The Mini's live server and both bridges were checked separately.

The next particle-specific check should use the same recorded gameplay/camera and
render settings for baseline and changed builds, separate loading/focus transitions,
retain heavy-burst frames and compare deadline tails. Particle simulation traces,
transparent batching, dynamic lights and presentation waits are separate candidate
costs. No particle-quality reduction or claim of a particle bottleneck follows from
the present sample.
