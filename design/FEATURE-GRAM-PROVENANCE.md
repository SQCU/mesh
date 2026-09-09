# Feature integration and the extra feature Gram

September 6, 2026. This audit distinguishes original user requirements from
assistant implementation decisions. The selected original messages, authorship,
timestamps, source paths and line numbers are preserved in
[`provenance.json`](../measurements/policy-feature-integration-20260906/provenance.json).
Repository documentation describing an operation is not evidence that the user
specified it.

## What the deleted operation computed

“Scale-feature Gram” was an implementation label, not a distinct mathematical
construction. For widened routed activations `S[N,D]`, `SᵀS` is a feature Gram:
its entries are inner products between feature columns, summing over rows.
`SSᵀ` is a row Gram: its entries are inner products between rows, summing over
features. Both are genuine Gram matrices; they retain different indexed objects.

The extra scale branch formed `G = SᵀS / D`, applied `asinh(G)`, multiplied it
by one learned vector `p[D]`, and broadcast the resulting `g[D]` back to all
rows through `S * g[None,:]` before the output projection. Earlier revisions
used row-count division and `tanh`. This branch compressed a feature-by-feature
matrix into one shared feature gate. Surrounding residuals preserved local
rows, but did not make this gate a player-specific or spatial interaction.

The exact identity `(RRᵀ)V = R(RᵀV)` does justify computing a row-Gram action
without materializing a quadratic row matrix. It does not justify replacing
`RᵀV` by a nonlinear feature Gram and a single learned probe. Those are
different functions with different derivatives and information paths.

## Where it came from

| Original record | Authorship and evidence |
|---|---|
| Claude session, August 28, line 2285 | User invites a substantial matrix problem whose outputs affect physics/planning/motion over delta time. This authorizes meaningful computation, not a specific feature gate. |
| August 29, line 9625 | User says benchmark code is not the algorithm, requires per-player integration of a shared observation map, and stale observations contracting toward an uninformative prior. |
| August 29, line 9654 | User requires contiguous navigable v-cell neighborhoods, parallel distance masking, 5–15% spatial support and velocity-defined strategy. |
| August 29, line 9776 | User proposes a learned output operator, RMS normalization and SwiGLU after a Gram intermediate instead of handwritten behavior semantics. |
| August 30, line 10583 | User identifies the missing learned cross-team all-to-all residual as an implementation defect. |
| August 31, lines 10822 and 10858 | User specifies Gram plus SwiGLU and full relevant game-state features; the tractable cart-game reward description is not a replacement input representation. |
| Codex session, September 1, 06:31:43 UTC, line 9604 | Assistant identifies a mismatch between a rank-128 player Gram and a benchmark's width-2048 residual Gram, and announces restoration of the benchmark-shaped path. |
| September 1, 06:33:25 UTC, line 9637 | Assistant says it is adding a large operator before every head so the operator cannot be “algebraically removable.” |
| September 1, 06:36:48 UTC, line 9717 | Assistant announces a nonlinear 2048-by-2048 feature Gram and learned probe in the live policy. |
| September 1, 22:20:01 UTC, line 27608 | User explicitly rejects preselected workload numbers followed by algorithm tampering to make hardware saturation coincide with those numbers. |
| September 2, lines 52068 and 52079 | User reiterates the Gram's cross-team information-mixing purpose and requires accurate names. |

The September 1 assistant messages establish the immediate provenance of the
extra feature Gram and probe. The selected original requirements support
learned feature projections, spatial integration, row mixing and routed FFNs;
they do not specify this probe-and-broadcast construction. This is evidence of
an unsupported implementation decision, not evidence about an author's intent.
The September 4 single-policy manifest later preserved that mathematics while
correcting parameter/optimizer ownership. Preserving an existing implementation
did not supply the missing algorithmic justification.

## Implemented correction

The feature-Gram/probe/broadcast branch is deleted. The scale module is now an
ordinary row-local routed SwiGLU residual with sigmoid top-k gates, the usual
frequency-times-probability balancing loss and no shared expert. Formula and
primary-paper references are in [ROUTED-MOE.md](ROUTED-MOE.md).

Global and same-team row-Gram actions remain. The default remote operation
computes their actual global cross factor `RᵀV[r,d_ir]` and returns every
coordinate plus both operand derivatives. That matrix is called a cross factor;
the composed operator `RRᵀ` is the Gram. This removes a model component without
inventing a replacement workload to preserve its FLOP count.

Literal native observations, state pages, carts, teams, events and serialized
navigation rows reach their first learned projections intact. Structural
presence and ownership remain explicit axes/masks. Events and navigation rows
enter the integrated local neighborhood before its outputs enter global row
mixing. Timestamp decay is applied after projection. An append-only raw event
history survives responder continuation; realized-event reporting consumes each
new event once. Neighborhood RMS normalization that canceled age decay was
removed from the caller. Detailed shapes and remaining reductions are in
[POLICY-AXIS-AUDIT.md](POLICY-AXIS-AUDIT.md), and spatial support is specified in
[VCELL-POLICY-INTEGRATION.md](VCELL-POLICY-INTEGRATION.md).

Work estimates now count these actual row families and three routed SwiGLU
matrices. Events are projected once rather than multiplied by player count;
they do not expand the global MoE row count. Estimates are logical contraction
counts, not measured hardware FLOPs. Parameter-count tests compare the estimator
against the actual parameter tree.

The geometric artifact currently carries navigation-graph length measure, not
walkable floor area. Per-cell unattainable 5–15% bounds are reported explicitly.
Compiled regions reuse fixed prepared extents, but capacity growth still causes
preparation and buffer allocation. Neither this audit nor the existing execution
code establishes a fully persistent allocation-free arena for unbounded history.
Those are concrete remaining differences from the stronger project requirements.
