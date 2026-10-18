# Policy implementation ledger — September 7, 2026

The controlling requirements remain the user quotations in `SPECIFICATION.md`.
This ledger describes implementation and evidence, not a substitute specification.

## 1. Persistent complete tensor program

The model, sampling/integration, loss, VJP, ordered contribution fold, clipping and
AdamW now use one symbolic graph and persistent native Metal execution. Source
extents remain symbolic through lowering; capacity realization changes metadata,
buffer placement and dispatch dimensions without recompiling shaders. Buffer
lifetime planning removes unused storage and aliases reshape/detached views.
Routed expert forward/input-gradient/weight-gradient products use the same tiled
SIMD matrix implementation. Sparse neighborhood input rows and full-axis Gram/MoE
reductions retain their specified meanings.

## 2. Distributed numerical ownership

The runtime owns the server's single mesh context; the game is its local client.
Graph regions, their VJPs and their parameter updates execute on the declared peer.
Native scoped calls carry numerical spans. Metal consumes complete receive-page
bundles directly and gathers larger bundles into persistent storage. It publishes
results through separate staging storage after the entire remote invocation
completes. Parameter generations, GPU borrow retirement, rebinding and local
recovery are connected to this path. The former tensor-packet and independent
MLX-worker implementations are deleted.

The complete current flow, lifetime rules and reporting fields are documented in
[POLICY-PROGRAM.md](POLICY-PROGRAM.md#persistent-execution-and-transport) and
[POLICY-TRANSPORT.md](POLICY-TRANSPORT.md).

## Compilation and numerical evidence

The native library, generated learner shaders, remote region manifest, linear/FFN
comparison arms, stock arm and native transfer/commit shaders compile on this host.
Bounded local inference, loss/VJP accumulation and AdamW calculations complete.
At a nonzero point containing a read-only skill tag, policy means agreed with the
MLX expression within 3.92e-8 and values within 2.39e-6. A separate seeded actor/value
calculation had finite derivatives throughout and relative gradient L2 difference
1.83e-4; its native and MLX losses were -7.641301 and -7.637756. The Gaussian
likelihood ratio can amplify small forward/reduction differences; that explanation
has not been isolated from other numerical differences at this point.
Those observations do not establish bitwise equality or training quality.

The initial small learner allocation fell from 194,679,552 bytes to 40,491,264
bytes after lifetime planning. These are allocation observations at one physical
capacity, not an upper bound for real maps. Compiler review also exposed unstable
Metal `asinh` results at large negative projected values; the lowering now uses a
stable signed logarithmic expression. Model inputs remain unchanged.

Doubling all ten physical capacities retained parameters, moments and gradient
accumulators exactly, kept 270 shader variants, and left the occupied policy/value
outputs unchanged at the observed point. A nonzero value contribution advanced
AdamW to step 1 with finite gradients. Routed products spanning 193 and 257 rows,
97 input columns and 77 output columns reused six shader variants; forward/input/
weight derivatives differed from NumPy float32 products by at most 1.50e-7.
The numerical observations are retained in
[the measurement record](../measurements/persistent-policy-20260907.json).

No repository test suite or verification harness was added. No game service,
bridge or remote workload was launched or restarted. Live RDMA execution and
match-level throughput are not claimed by this local evidence.

## 3. Reporting continuity

The viewer owns a durable SQLite report store under `.joracle`, keyed by its
stable source path, or an explicit `--state-directory`. It retains the last
measured J/model/counterfactual reports, optimizer updates, plotted series,
outcomes, run history and episode contexts across viewer restarts. Source changes
retain prior records, and retained reports carry their original source directory.
The raw replicated telemetry and J artifacts remain the source evidence.

Journal cursors use byte positions and prefix/tail anchors so an append-only
journal survives rsync inode replacement. Complete records have persistent
identities; a rewritten prefix is replayed without duplicating observed frames
or matches. Incomplete tails remain pending. Malformed complete records are
reported with path and offset, and later complete records remain readable.

Report bodies are serialized when content changes. Browser polling reads the
small status endpoint first and fetches/renders report bodies only after a content
revision changes. Replication keeps its existing finite SSH/rsync budgets and
retains the previous complete replica through failures. Worker joins on viewer
shutdown are bounded.

After its first measured frame, a producer atomically publishes
`~/.local/share/mesh/xonotic-active.json`. The replica source `mesh:xonotic`
reads that registry locally and through discovered mesh nodes. It retains a
currently advancing producer, then selects an advancing replacement when that
source becomes stale. The source file accepts the same selector. Discovery errors
and candidates appear in replication status; no candidate leaves the last complete
replica intact. `--replicate-from host:path` continues to select an explicit source.

## 4. Configuration and exposure

The engine exports actual bot-definition traits, including random fallback
realizations, and the actual global skill setting. The values enter the common
observation projection unchanged. A configuration-schema field distinguishes
available configurations from missing data. Edict, spawn time, native episode,
producer UUID, request sequence, seed metadata, acknowledged controller lineage
and exact update version accompany the observation.

Historical paired-study Elo is a separate report.

## Skill parameters are percepts, not actions

The latest user instruction explicitly excludes randomized Havocbot skill
parameters from the control surface. Native type tags carry a read-only flag;
input featurization retains the values and output sampling/likelihood/integration
mask those coordinates. The adapter masks received and retained commands again.
Native view stores cannot change the protected skill entity fields. Copied skill
values and the stock named aliases carry read-only metadata through VM frames.

The observation layout is now 84 words and the native type-tag semantics change.
The engine, QC payload and Python application therefore need coordinated source
realization before a new match. Policy versions are matrix 21 / baseline 19,
architecture 12. The local numerical observations above include a protected skill tag. They do
not establish compatibility with a previously deployed engine or a live match.
