# MULTISCALE.md — two nested subgames, one closed-form price, a statewise policy

This document fixes the single most load-bearing idea in the project and the one agents
keep dissolving into an ML default: **the problem is multiscale.** There are two
subgames at two scopes; only one of them is closed-form-interpretable; and we use that
one to *price every state* of the other. Nothing here is sequential, stateful, or
recurrent. We are not learning to *play through time* — we are learning to **price all
states along a game history** and act statewise.

## 1. The two subgames (two scopes)

**Subgame B — Xonotic point-and-shoot (the real game).** Basically-continuous, rapidly
updating. Its state is the full resource configuration: every playerbot's position,
velocity, health, armor, weapons owned, ammo, powerup timers — plus the map. This is
where all action happens (move, shoot, grab health/ammo/guns, knock enemies around).
**It is intractable.** There is no closed-form scheme that tells you what a Xonotic
state *means* in the semantics of winning or losing. You cannot analytically price it.

**Subgame A — the cartgame (the tractable shadow).** Discrete, and *laggy* — its state
(cart depths + control) changes slowly and in steps, a coarse projection of B. Each cart
on its path is a heap; the board is a multi-heap nim-like counting game. **It is
closed-form-interpretable:** `PW(s)` (projected winner, nim-sum), `SUCC(s)` (the
backward-induction succession), and the per-team hierarchy `N_i(s)` are exact
combinatorics over cartstate. Subgame A is where "who is winning / losing" has a
*computable meaning*.

**Containment.** B totally contains A — the carts live inside the Xonotic map; the
cartgame state is a coarse-graining of the full Xonotic state. A is B's tractable
shadow: the only place the game exposes closed-form winning/losing semantics.

```
   Subgame B : Xonotic point-and-shoot   (continuous, rapid, INTRACTABLE, no closed-form meaning)
   ┌───────────────────────────────────────────────────────────────────────┐
   │  all players: pos, vel, health, armor, weapons, ammo, powerups, map    │
   │                                                                        │
   │        coarse-grain ▼ (a laggy discrete projection)                    │
   │   Subgame A : cartgame  (discrete, laggy, TRACTABLE, closed-form)      │
   │   ┌───────────────────────────────────────────────┐                    │
   │   │  cart depths + control  =  multi-heap nim      │                    │
   │   │  PW / SUCC / N_i  = who is winning, computable │                    │
   │   └───────────────────────────────────────────────┘                    │
   └───────────────────────────────────────────────────────────────────────┘
```

## 2. Why `reward = score` and whole-game RLVR are both degenerate

We do **not** price states with the score, and we do **not** train on the terminal win
(RLVR). Both are informationally degenerate for *this* game.

- **`reward = score` is degenerate** because scores integrate monotonically toward
  victory. A monotone integrator carries no per-state gradient about your *strategic
  situation*: you can lead on score and be one coordinated push from losing the winning
  region. Pricing by the level of a monotone quantity cannot distinguish "safely ahead"
  from "ahead and about to be ganged."

- **Whole-game RLVR (`reward = win`) is degenerate** because the terminal-win scalar has
  *marginalized away the multipolar structure*. Optimizing `E[win]` gives a gradient
  that cannot tell "teams B and C are coordinating against me" from "I am simply
  losing," because both collapse into the same expectation. A policy trained on it is
  **formally unable to notice it is being ganged up on** — the identity of who is
  ganging whom is a function of *all teams'* cartstates (the hierarchy/succession), and
  the win-scalar does not carry it. RLVR also replaces an available analytic quantity
  with a high-variance sparse-reward estimator: we can compute the advantage in closed
  form, so we do not need to bootstrap it from a terminal outcome.

The multiscale structure is exactly what lets us avoid both: **price subgame A in
closed form, and use that price on subgame B.**

## 3. The closed-form price (the tractability trick)

We cannot price B. We *can* price A. So the pipeline computes a per-state price from A
and hands it to the policy acting in B.

1. **Semantics** — from cartstate, closed-form: `PW`, `SUCC`, `N_i`. Who is the
   projected winner and what is the succession if the leader is denied.
   > "Nimber-Preserving Reduction: Game Secrets and Homomorphic Sprague-Grundy Theorem"
   > (Burke, Ferland, Teng, FUN 2022) — the position value is the PSPACE-hard *game
   > secret*; where the reduction is impartial it is an exact nim-sum, and partizan/
   > cyclic positions are kept explicitly unresolved. The cart path is the Generalized
   > Geography carrier of each heap's magnitude.

2. **Price** — a per-team **advantage** from convex/control-theoretic tools, asymmetric
   by role: `W` for a team that holds the path (preserve the winning region, restore its
   margin under perturbation); `L` for a team that does not (climb the hierarchy, deny
   the leader, acquire). This is a control signal, not a bootstrapped reward.
   > "No-regret learning dynamics ensure that a learner asymptotically achieves an
   > average reward no worse than that of any fixed strategy. **This no-regret guarantee
   > does not determine the value of the asymptotic average reward.**" — Abdelraouf &
   > Shamma, *Can a Learner Regret Using a No-Regret Algorithm?* (arXiv:2603.03173).
   >
   > "We cast the global comparison … as an optimal-control problem and show that **the
   > minimal cumulative reward gap is zero**, thereby establishing global uniform
   > dominance of anticipatory replicator dynamics … for arbitrary payoff trajectories."
   > — ibid., §VIII-C.

   The `W` objective *is* that paper's control picture: hold a target region and keep
   reachability margin so adversarial disturbances (the ganging) can be corrected. The
   "who is the key player to deny" readout is the network-games intercentrality of
   Ballester–Calvó-Armengol–Zenou — a closed form over the same coupling.

3. **Act** — the price is the learning signal for a policy that acts through the **full
   B abstraction**: which resources to dominate (health/ammo/guns), which rival to
   focus, which cart to push — to *prevent perturbation of the winning state*. The
   policy is a **learned filter on full resource state**: e.g. whether a bot holding a
   rocket launcher should run at the cart (spend the resource lead) or at more health/
   ammo (extend it), *and* which cart — a decision the closed-form price makes
   learnable without ever pricing B directly.

## 4. Statewise, not sequential — no recurrence

There is **no sequential or stateful learned model, and no recurrence.** The hard
sequential reasoning (succession, backward induction) is *closed-form per state* in
subgame A, so the learned part never has to unroll time.

- A "game history" is a **bag of priced states**, not a sequence to recur over. Each
  observed state is priced by the closed-form A-solver; the policy learns from the
  set.
- The policy is a **statewise map** `state → per-player action distribution`, trained to
  act well given the per-state price. We interpret **entire multiplayer games
  state-by-state** and use closed-form solvers to get strong strategies — precisely
  because the closed form removes the need to learn temporal credit assignment.

This is why the learned operator is **shallow-and-wide (a big Gram + SwiGLU), not a deep
multi-head-attention-resnet and not an RNN.** Depth/recurrence would only be forced if
we made the network *learn* the `H`-hop succession; we compute it. What remains is a
statewise pricing readout — one big all-to-all matmul over the player rows, the compute
shape that saturates a bandwidth-bound mesh of M-series SoCs, rather than a latency-bound
serial stack.

## 5. Ball-and-stick data flow

```mermaid
flowchart TD
  subgraphB["Subgame B — Xonotic (continuous, rapid, intractable)"]
  X["full state s_B: all players pos/health/armor/weapons/ammo/powerups + map"]
  subgraphB --- X

  X -- "coarse-grain (laggy, discrete)" --> C["cartstate s_A: cart depths + control (multi-heap nim)"]
  X -- "featurize (per-player rows: resources, belief)" --> F["feature rows R0 = W_q[x ; belief]"]

  C -- "closed-form nim / Sprague-Grundy" --> PW["PW, SUCC, N_i  (who is winning; succession)"]
  PW -- "convex / control-theoretic (W preserve+margin, L climb+deny)" --> ADV["per-team role price: advantage A_W / A_L"]

  F -- "big all-to-all Gram over ALL player rows (O(n^2), mesh-saturating)" --> G["coupled rows R'"]
  PW -. "closed-form succession as a feature, not learned" .-> G
  G -- "SwiGLU + value heads" --> POL["statewise policy: per-player instrument weights"]

  ADV == "per-STATE price = learning signal (NOT reward=score, NOT RLVR=win)" ==> POL

  POL -- "target weights -> stock navmesh + point-and-shoot" --> ACT["actions in s_B: push cart / grab health-ammo-gun / focus key rival"]
  ACT -- "changes s_B (fast), and occasionally flips s_A (slow)" --> X
```

Text form of the same graph:

```
 full Xonotic state s_B ──coarse-grain──► cartstate s_A ──nim/SG──► PW,SUCC,N_i
        │                                                              │
        │                                              convex/control  ▼
        │                                            role price: A_W / A_L (per state)
        │                                                              │  (learning signal)
        └──featurize──► player rows ──BIG GRAM (all-to-all, O(n^2))──► R' ─SwiGLU/value─► statewise policy
                                            ▲ (SUCC as feature)                 │
                                            └── closed-form, not learned        ▼
                                              per-player instrument weights ──stock navmesh + point&shoot──► actions
                                                                                          │
                                              actions change s_B fast, flip s_A slow ◄────┘
```

## 6. What this forbids

- No pricing of subgame B directly (intractable) — price A, apply on B.
- No `reward = score`, no whole-game RLVR — both degenerate (§2).
- No recurrence, no sequential learned state machine — statewise pricing only (§4).
- No deep attention/resnet in the learned operator — the depth is closed-form in A; the
  learned part is one big Gram + SwiGLU (§4, and the expressivity note in the strategy
  spec).
- No fake re-simulation of the game to train against — B is already the simulation;
  train on it or prove the price. Synthetic re-simulation is not Game-2 evidence.
