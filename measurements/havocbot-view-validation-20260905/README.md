# Havocbot view validation — September 5, 2026

The [implementation contract](../../design/POLICY-STATE-STEERING.md) describes the
read-only residual and fixed-width page interface. [Summary JSON](summary.json)
records scope, limitations and staging recovery. Persistent game and learner
services were not restarted; finite native engine fixtures exited themselves.

| Check | M5 Max laptop | M4 Pro Mini |
| --- | ---: | ---: |
| Python numeric/replay/reporting checks | 47 passed | 33 passed |
| Native checks, including actual wire responses | 28 passed | 28 passed |
| Native finite-difference derivative | 0.3296000 | 0.3296000 |
| Analytic derivative | 0.3296809 | 0.3296809 |
| Missing-bridge open/pump/read probe | 9.0 µs | 16.0 µs |
| Median stock / viewed whole think | 4.32 / 14.80 µs | 20.02 / 67.38 µs |
| Median paired additional cost | 10.47 µs | 44.65 µs |

The timing uses 20 calls per sample, discards two warmup samples and records ten
paired samples of an already-running bot at fixed engine time. It does not establish
worst-case frame latency. Different initial bot states and host scheduling also make
these two runs unsuitable as a pure hardware speed comparison. The extra cost for 16
such thinks is approximately 0.17 ms locally and 0.71 ms on the Mini.
The relative interpreter overhead is substantial even though these absolute times
are small. No whole-world or touched-page snapshot is copied on each think: words
are captured as read, and relaxation coefficients are shared by the scope.

The native fixture compiles the actual Havocbot aiming and keyboard code. It checks
independent bot views, repeated and nested reads, early returns, global/integer/
function references, actual entity-slot reuse, scalar/native movement parity and
unforced decay. A target-view residual points the bot about 84 degrees upward;
a compensating rate returns its aim, with the target's real position unchanged.

A separate POSIX shared-memory fixture crosses the real engine transport and native
consumer with an MLX readout. Its finite difference varies a readout bias before
packing the emitted rate. Five invalid response cases (NaN, session mismatch,
owner-generation mismatch, duplicate page, invalid extent) preserve the preceding
view atomically. It opens no verbs device. Integration and full-model parameter
and input gradients also have independent Python tests.

The game-law fixture passes 39 score/projection cases, six ownership transitions,
six player-driven motion cases and 3,000 unattended ticks. Two existing bridge tests
pass, including bounded progress and teardown fault injection.

- [Laptop native results and engine log](laptop-native.json), [Mini native results](mini-native.json).
- [Laptop whole-bot samples](laptop-bot.json), [Mini whole-bot samples](mini-bot.json).
- [Laptop test log](laptop-tests.txt), [Mini core test log](mini-tests.txt), [game checks](game-tests.txt), [bridge checks](rdma-tests.txt).
- [Build manifest](BUILD_MANIFEST), [current numeric inventory](inventory/summary.json), [runtime entity words](inventory/runtime-entity-coordinates.csv).

The rebuilt schema has 6,115 words per runtime entity record
(5,140 floats and 975 integers) and 53,212 runtime
global words. A bot's transitive view contains however many entity/global pages it
reads; one entity record is not the whole bot state vector.

PRVM64 precision, unrestricted random-reference execution and live learning quality
are not established by these checks. The Mini's initial SSH outage recovered; its
missing game assets were staged under `/tmp/mesh-native-view-20260905`, alongside the
test engine. Only the copied engine's JPEG load path and ad-hoc signature differ.
