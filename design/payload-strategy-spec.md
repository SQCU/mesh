# The payload game, as a strategy object

> **Correction (banking model).** An earlier version of this document taught
> "un-banking" — that a cart retreating past a banked control point *destroys* that
> bank. **That model is retracted.** Cart POSITION reverses; banked SCORE is monotone
> and never un-banks. What makes this a game rather than a race is not a reversible
> banked asset but the **relative objective**: take the path-to-victory away from
> whoever holds it and acquire it. The authoritative statements are
> `rl-training-spec.md` (§2 asymmetric W/L objectives, §3 policy gradient) and
> `payload-spec.md` (§0 monotone-score layer); `strategy-layers-and-modality.md` §1
> records the same retraction. The passages below (§1, §2, §6) have been rewritten to
> that corrected model.

This recovers what the k-cart payload mode *is* — the objective, the reversible
push/pull on each cart, the all-to-all computation that scales with teams x players
x carts, and how that maps onto classical game theory and non-neural RL. It is the
reference the mechanics and the solver both answer to. Where the code and this
document disagree, this document states the intent and the code is the bug.

## 1. The objective

`k` carts. `j` teams. Each cart rides one path: a chain of control-point
waypoints with an **origin** at arclength `s = 0` and an end at `s = L`. A cart's
entire state is the scalar `s(t)` plus which team, if any, **controls** it.

A team **controls** a cart when it holds a strict plurality of presence inside the
cart's activation cylinder (radius x height around the cart, the same cylinder the
occupancy law uses). Control is a function of bodies in a place, recomputed every
tick — it is not owned, it is contested continuously.

A team **banks** a control point by being in control as the cart crosses that point
*moving forward*. Banking scores, and **banked score is monotone — it never un-banks**
(retracting the earlier "reversible banks" model; see the correction note at the head
of this doc and `strategy-layers-and-modality.md` §1). A round is won by delivering a
cart to its end (`s -> L` under control) or, at timeout, by the relative outcome the
role objective encodes (`rl-training-spec.md` §2) — not by a raw count of banked points.

Two things make this a game rather than a race, and both were missing from the
drag-only law that this document corrects:

- **Cart POSITION is reversible.** A cart under opposing plurality moves **backward**,
  `s` decreasing toward its origin. What reverses is the cart's place on its path, not
  any score already banked.
- **The objective is RELATIVE, not an accumulable asset.** The strategic quantity is
  not "how many points have I banked" — that is the monotone-progress reward
  `rl-training-spec.md` §5 rejects as a *race*. It is possession of the
  **path-to-victory**: deny it to whoever currently holds it and acquire it. A lead is
  contestable because position reverses and control is recomputed every tick, not
  because banked score can be destroyed.

Origins and iterative waypoints exist *because* position reverses. Without reversal,
an origin is just a start line and waypoints are just a progress bar. With reversal,
the origin is the sink a losing cart falls back to, the waypoints are the ladder a
push climbs and a counterattack drives back down, and `s` is the tug-of-war rope. The
denial a counterattack achieves is over *position and control of the path-to-victory*,
not over already-banked score.

## 2. The push/pull law — two regimes, distinguished by presence

Let `w_j` be team `j`'s presence weight in the cart's cylinder (teammates add with
the existing falloff cap: linearly helpful, saturating). Let `A` be the cart's
**color** — the team it is currently banking for, a sticky property that changes
only at origin. The mechanic is two *separate* parametric rules, and which applies
turns on whether the color team is present at all, not on who is winning:

**Regime A — contested (`w_A > 0`, the color team is here).** This is active
tug-of-war combat, and it must stay spatially in the locale of the cart. The velocity
is bounded and damped by opposition:
```
v = clamp( speed * (w_A - w_opp) / (1 + w_opp^2),  -contest_speed,  +max_speed )
```
where `w_opp = sum_{j != A} w_j`. The color team advancing unopposed moves forward at
full push; a balanced brawl sits near zero; even a losing-but-present defense caps
backward motion at a small `contest_speed`, so a cart being *fought over* barely
drifts from where the fight is. A cart is never driven home while its defenders are
still swinging.

**Regime B — abandoned (`w_A = 0`, the color team has left).** Now, and only now, an
opposing team `B` present at the cart drives it back toward origin, **linear in team
B's player count** (no quadratic — this is capture, not contest):
```
v = -reverse_speed * w_B         // B = strongest present opposing team
```
The cart retreats as `s` crosses control points downward, until `s = 0`. At origin the
cart **recolors** to `B` (a POSITION/control property, sticky until origin); banked
score is untouched — retreat reverses the cart's place on the path, not any score
already accrued. From there B advances it under Regime A. You cannot steal a cart
mid-track and turn it your way — you must first push its POSITION all the way home,
which is what makes a deep push a real *positional* asset (it holds the
path-to-victory) and abandonment a real risk (it cedes that path), independent of the
monotone score line.

This separation is the point the drag-only law missed: reversal is triggered by the
color team's **absence**, not by its **disadvantage**. Presence keeps the fight local
(A); absence surrenders the cart to a linear walk home (B).

`s(t+dt) = clamp(s + v*dt, 0, L)`. Crossing a control point upward while controlling
banks its score (monotone; a downward crossing does NOT un-bank it — the earlier
un-banking clause is retracted). Delivery is `s = L` under control; origin collapse is
`s = 0`, which recolors the cart and resets its POSITION, not the banked score.

This one law produces the whole strategic surface: a lead is defensible only by
sustained presence, a counterattack is a real instrument (it reverses the leader's
POSITION and denies the path-to-victory, not just tempo), and a team must choose per
tick whether to push its cart, suppress an opponent's, or hold the line.

## 3. Why the computation is all-to-all, unavoidable, and scales

The per-tick decision each team faces is an **allocation of its players over
instruments** — {push cart i, suppress cart i, escort cart i} for each of `k` carts,
plus {contest item post p} for scarce timed pickups. The payoff of any one
allocation depends on *every other team's* allocation (control is a plurality, so my
push succeeds only relative to your suppress) and on *every player's* position and
resources relative to *every* cart and post. That is an all-to-all coupling of size

```
teams x players x (carts + posts)
```

and it is not reducible to per-cart independent problems, because the teams' choices
are coupled across carts by the pigeonhole constraint (`j` teams cannot each fully
cover `k < j` carts) and by the shared player pool (a body pushing cart 1 is not
suppressing cart 2). This is the load the RDMA coprocessor exists to carry:

- **Routed experts (MoE)** over player feature rows: a sparse, routed transform of
  each player's state — position, resources (the dominance tuple), team, nearest-cart
  economics — into a per-player contribution, routed so different player situations
  are scored by different expert weights. This is the sparse/routed matmul.
- **The Gram over the rolling context window** is the genuinely quadratic object:
  `Z^T Z` over the recent history of all players' feature residuals is the all-pairs
  second moment — *who has been co-present with whom, contesting what*. Its
  off-diagonal blocks are the empirical contest-intensity between team pairs over
  each cart and post. This is the O(n^2) term that makes the workload worth a
  teraflop-class device and that scales with players and history depth.
- **Fictitious play** (see 4) reads the window's *first* moments — each team's
  empirical mixture over instruments — as the belief each team best-responds to.

The solve is ~10^2 GFLOP/tick at the current context size and grows with players,
carts, posts, and window. It is the coprocessor's reason to exist: a small map with
one cart needs no matrix; a fused multi-map world with many carts, many posts, and
`j > k` teams makes the coupling dense and the solve load-bearing.

## 4. The classical structure it instantiates

The point of the formalism is that the game is a *recognizable* object in classical
theory, computed exactly rather than approximated by a black box.

- **Stackelberg, nested.** Between teams: the score/tempo-dominant team is the leader;
  it commits its allocation first (its commitment is legible from the history), and
  trailing teams best-respond. Within a team: the resource-dominant player leads
  execution and teammates best-respond to its committed target. Backward induction,
  two tiers, closed form.
- **Fictitious play (Brown).** No team is told the others' current move; each
  best-responds to the *empirical mixture* of what the others have recently done —
  the first moments of the context window. This is the classical belief dynamic, and
  its fixed points are the correlated/Nash-like equilibria the match settles toward.
- **A positional/differential game on `s`.** The reversible push/pull is a tug-of-war
  over each cart's scalar `s` (position, not banked score) — a positional game whose
  value is contested at every control point, with pursuit-evasion flavor (defend the
  control line vs drive it back).
- **Coalition formation.** At `j >= 3` the trailing teams' best response to a leader
  is to gang up on the leader's cart — a coalition that the Gram's off-diagonal
  contest blocks make visible and that the allocation head acts on. The pigeonhole
  fork (`j` teams, `k < j` carts) forces the coalition to *choose which* leader-cart
  to collapse, which is a real strategic decision, not a dominated one.

- **RL over things that are not deep nets.** The learned surface is deliberately
  legible: a handful of named strategy scalars (temperatures, appetites, commitment,
  hysteresis) plus a **regularized logit field** produced by the heavy solve — a
  linear map from the Gram-conditioned features to per-instrument logits, sampled,
  moved by **REINFORCE** with a baseline, and pulled toward zero by an L1/L2 penalty
  so that *without* training the policy is a broad weighted sampling over effective
  strategies and *with* training it peaks without collapsing to a single action. The
  features are the routed-expert and Gram outputs; the policy is a sampled
  softmax over an interpretable field, not an opaque network. This is policy-gradient
  RL and fictitious play over a classical feature basis — the "RL over
  things-that-aren't-DNNs" the design commits to.

## 5. What the prototyped tooling lets us capture

Each tool we built exists to make one of the above real rather than aspirational.

- **The sealed RDMA transport + one-bridge mesh** makes the all-to-all solve
  *offloadable* to a second machine at line rate with zero-copy pages — the coupling
  is computed where the FLOPs are, streamed back as picks, with the game none the
  wiser about queue pairs. This is what turns "a teraflop of coupling per tick" from a
  cost into an architecture.
- **The navmesh k-center / network-span / anti-aligned-flow placer** makes the carts
  a genuine *portfolio*: equidistant origins in walking distance, distinct low-overlap
  tracks flowing against each other, so "win one position, win them all" is impossible
  and the pigeonhole fork has real geometry to bite on.
- **The activation-corridor constraint** guarantees every point of every cart's path
  is within pushing range of standable space — so control is always physically
  contestable, and "occupied but frozen" (which would break the push/pull law) is
  unrepresentable.
- **The item-post wire** puts the timed scarce resources of the map into the
  allocation as first-class instruments, so playerstock can be spent on map control,
  not only cart pushing — and the dominance loop (hold a post -> fatter resource tuple
  -> higher measured dominance -> leadership) closes.
- **The map-fusion / q3map2 pipeline** makes the *space* combinatorially rich: gluing
  stock maps and authored arenas raises the branching factor and floor area so the
  coupling is dense and the strategy space is large, with connector edges (corridor /
  jump-pad / teleporter) that are themselves contested chokepoints.
- **The diegetic layer** (path ribbons, cart-state dashboard, waypoint rules) makes the
  contested scalar `s`, the control color, and the push/pull legible to a spectator —
  so the computed strategy is *watchable*, which is the demo's whole claim.

## 6. The immediate correction

The current velocity law floors `v` at zero (`bound(0, ...)`), so opposition stalls
but the cart's POSITION cannot reverse. Per section 2 the law must be signed
(`bound(-max, ..., +max)`) so position is a reversible tug-of-war. Banked score stays
monotone — a downward crossing reverses position, it does NOT un-bank score (the
earlier "banking must un-bank" instruction is retracted; see the head-of-doc
correction and `strategy-layers-and-modality.md` §1). That signed-velocity change is
the mechanic the team- and player-level policies in sections 3-4 are computed to
steer. It is implemented as the companion change to this spec.
