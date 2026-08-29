# Superagent review checklist — qualities STATED by the owner, never CLOSED by the owner

Scope of authority: the **user-authored turns only** (transcript
`d3ad4328-…jsonl`, 258 user turns). Assistant turns, compaction summaries, and the
design docs are **not** authority for what is required. A quality is "closed" ONLY
if a user turn says it is covered, done, or judiciously omitted. The docs are read
solely to mark whether the quality currently *appears* (yes / partial / no) — never
to decide whether it is required.

The owner's last two turns are the charter:
- Turn 256 (verbatim, emphatic, six-fold REINFORCE refrain):
  `REINFORCE. nimbers. REINFORCE. explicit multicart. REINFORCE. monotonic score.
  REINFORCE. 'golden path'. REINFORCE. backwards induction (explicit, not implicit or
  described allusively then forgotten as if it was optional). REINFORCE.`
- Turn 257: names *"Nimber-Preserving Reduction: Game Secrets And Homomorphic
  Sprague-Grundy Theorem"* and asserts there are **"several explicitly missing
  qualities which have no negative answer in the user turns describing them as covered
  or judiciously omitted."**

None of the items below were ever closed by a user turn. Turn 256 re-opens them by
listing them as required; turn 249's "all of the linear algebra decisions are done now"
closes the **linear-algebra mixing machinery** (DPP / learned operator / velocity-on-
weight / gram-as-coupling) and **nothing in the game-theory column**.

---

## A. Nimber-paper-derived items (LEAD — the centerpiece of turn 257)

The paper (Burke–Ferland–Teng, FUN 2022, LIPIcs vol 226 art. 10) fixes what a
*principled* nimber treatment must contain, versus "Nim as a loose analogy." The
current spec (`payload-spec.md §2.6`, commit `6d22699`) upgraded nimbers/backward
induction from `[OPEN, analogy only]` to `[FIRM]`, but it states them as a scalar
"nimber names the swing" formula and still omits every structural requirement the
paper makes load-bearing. `strategy-layers-and-modality.md` still contradicts it,
carrying `[OPEN, analogy only]` and the line *"not a claim that … a Grundy value
exists in closed form."* That live contradiction is itself a review item.

| # | Quality | Owner's words (turn) | In docs? | Closed by user? |
|---|---------|----------------------|----------|-----------------|
| A1 | **Impartial-vs-partizan honesty.** The Sprague-Grundy theorem and nimbers are defined for **impartial** games (both players share options at every position); a colored/teamed cart game is **partizan**, where a single Grundy value does *not* exist — the paper flags partizan as an *open* extension (§5.3). A principled treatment must either reduce to an impartial per-cart abstraction (a heap whose "who controls" is factored out of the move-set) or explicitly state the game is partizan and that the nim-value is an approximation, not an identity. | Turn 243 "we don't actually want for this game to be really, really, really different algorithmically from nim; in fact the game being more similar to nim is more good" + the paper link. Turn 256 "nimbers". | **No** (no doc mentions impartial/partizan) | **No** |
| A2 | **Disjunctive sum with XOR/nim-sum composition of the multicart position.** Paper Eq. 2: `nimber(G+H)=nimber(G) ⊕ nimber(H)`; a move is made in **exactly one** component, the others unchanged. The match nimber must be the **nim-sum of per-cart nimbers**, each cart a separable component game — this XOR is precisely what makes "where to throw your team" have *multiple* answers (attack the component that flips overall parity). Current spec says "the combined position has a value … which cart to attack to flip" but never states ⊕, component independence, or the move-in-one-component rule. | Turn 243 "'where to throw your team' must ultimately have multiple answers (woah…)"; "put a ton of your team onto a neglected cart to reverse the cart of a team on the path to victory (woah… nim mentioned…)". Turn 256 "explicit multicart". | **Partial** (multi-cart mechanics present; disjunctive-sum/⊕ structure absent) | **No** |
| A3 | **mex rule / Grundy recursion / terminal=0 / normal-play losing condition.** Paper Eq. 1: `nimber(G)=mex{nimber(options)}`, terminal position = 0, normal-play = player with no move loses. A principled per-cart nimber needs a defined game tree, terminal (origin? delivery?) mapped to 0, and the mex recursion. Current docs invoke a "key-player solve over kappa / `(I-a·kappa)^-1` intercentrality" as the "computable realization" — that is a **centrality heuristic, not a Grundy/mex computation**; the identification is asserted, not derived. | Turn 243 nim framing; Turn 256 "nimbers", "backwards induction (explicit…)". | **No** (mex, terminal=0, Grundy recursion absent; "Grundy" appears only as the *denied* claim) | **No** |
| A4 | **Nimber as the PSPACE-hard hidden "game secret," estimated under incomplete information.** The paper's thesis: winnability (nimber≠0) is cheap, but the **nimber itself is a hard secret of deep alternation** even for tractable games. This is exactly the owner's "it's ambiguous how to split resources when you've disrupted another team's strategic hold enough that it's not self-evident what their nim-analogue-number is." The hidden-nimber must be tied to the belief/observation (incomplete-information) machinery: each team *estimates* opponents' nim-values, it is not read from omniscient state. | Turn 243 "not self-evident what their nim-analogue-number is"; Turn 245 "inferences under uncertainty s.t. matrix solutions give good team strategy … under deliberately incomplete information". | **Partial** (payload-spec: value "cannot be read off cheaply"; not linked to belief/incomplete-info as *the* secret) | **No** |
| A5 | **Monotone heap correspondence.** A Nim heap is a monotone counter drawn toward 0; the owner requires **cart score monotonically increasing**, `d(cartscore)/dt = depth-of-control * cart-lanes`, and explicitly killed the un-banking that broke monotonicity ("Un-banks A's points on the way down — that's too much"). The heap↔score mapping (cart depth-under-control as heap magnitude) should be stated as the thing that makes the Nim structure legitimate rather than decorative. | Turn 243 "the game structure has to be integrated over time s.t. team cartscore monotonically increases, with d_cartscore/d_t determined by the depth of control * cart lanes". Turn 256 "monotonic score". | **Yes** (payload-spec §2.1 monotone score; `s` reverses, banked score never does) | **No** (stated as requirement; user never signed it off — re-listed in 256) |
| A6 | **Explicit backward induction = the Sprague-Grundy/mex recursion from terminal leaves, NOT a relabeled Stackelberg overlay.** Owner demands it "explicit, not implicit or described allusively then forgotten as if it was optional." The recursion that computes nimbers *is* backward induction over the game tree. Current spec makes it a `[FIRM]` "induct" step but realizes it as "nimber-leading team commits; trailing best-respond" — that is Stackelberg leader/follower, i.e. **winnability**, not the value recursion. The two must be reconciled: backward induction produces the per-cart values that the commit/best-respond then uses. | Turn 256 "backwards induction (explicit, not implicit or described allusively then forgotten as if it was optional)". Turn 246 "there *is* room for the backward induction stuff over something here." | **Partial** (payload-spec §2.6 [FIRM]; strategy-layers-and-modality.md still [OPEN] — contradiction) | **No** |
| A7 | **'golden path' as the geography/path spine.** The SG-complete ruleset the paper builds on is Generalized **Geography** — a token traversing a directed path, deleting vertices. The cart's origin→end waypoint path (arclength `s∈[0,L]`) is that path; "depth of control" is position along it. The golden path should be named as the geography-path carrier of each heap's magnitude. | Turn 256 "'golden path'". | **Yes** (payload-spec: "golden path" = arclength `s`, origin→end) | **No** (named as requirement; not signed off) |

**What the paper says our nimber treatment must add that the current spec lacks:**
(1) declare impartial-vs-partizan and stop implying a closed-form Grundy value for a
teamed game [A1]; (2) the **⊕ nim-sum over independent per-cart components with the
move-in-exactly-one-component rule** — the missing core [A2]; (3) an actual
**mex/terminal=0 Grundy recursion**, not a centrality inverse relabeled as "the
nimber" [A3]; (4) nimber-as-hard-secret bound to the incomplete-information belief
buffer [A4]. Items A5–A7 exist in `payload-spec.md` but were never user-closed and
still sit beside a contradicting `[OPEN, analogy only]` doc.

---

## B. REINFORCE specifics (the six-fold refrain — highest owner salience)

| # | Quality | Owner's words (turn) | In docs? | Closed by user? |
|---|---------|----------------------|----------|-----------------|
| B1 | **A well-defined REINFORCE at the per-second (intra-match) scale, not only over match outcomes.** The owner's turn 255 is a direct challenge to the doc's "REINFORCE over match outcomes"; turn 250 fixes two integration cadences (1–10 Hz) decoupled from the engine tick, which implies a within-second reward/return. This must be answered, not left as match-terminal credit. | Turn 255 "is there a well defined REINFORCE within any given second of a match?" Turn 256 REINFORCE ×6. | **Partial / wrong direction** (payload-spec: "outer objective is REINFORCE over match outcomes" — the framing 255 questions) | **No** |
| B2 | **REINFORCE as the calibration signal for the learned parameters, filling responses WITHIN the backward-induction skeleton (not replacing it), with L2-toward-zero so untrained = broad weighted sampling.** Owner wants minimal feature engineering, maximal commitment to learned params fed by "tons of winning and losing playerbots in tons of matches." | Turn 246 "commit as much as possible to the learned parameters and the expectation that … tons of winning and losing playerbots in tons of matches gives us tons of valid REINFORCE food"; Turn 228 "l1 or l2 regularization towards a logit of 0 … without any reinforcement from REINFORCE we see a weighted sampling of effective strategies". | **Yes** (payload-spec §2.3–2.5, logit field + L2-toward-zero) | **Partial** (turn 249 closed the *linal*; the REINFORCE *scale* question B1 stayed open) |

---

## C. Game-mechanics coverage the owner requires the strategy to exercise (turn 243)

Turn 243 requires that the strategy "somehow involve **all** of the game mechanics in
xonotic" — each is a stated quality with no negative answer.

| # | Quality | Owner's words (turn 243 unless noted) | In docs? | Closed by user? |
|---|---------|----------------------|----------|-----------------|
| C1 | **Resource collection** as a strategic lever. | "resource collection" | Partial (resource/economy mentioned) | No |
| C2 | **Exploration** to collect powerups on the map's resource schedule. | "exploration (e.g. to collect powerups on the resource schedule of maps or map sections)" | Partial (exploration in payload-spec) | No |
| C3 | **Exploiting resource leads** — beat enemies to the combo before they arm. | "exploiting resource leads over other players (you're supposed to beat enemy players before they get the rocket launcher + beam gun combo!)"; Turn 244 weapons "have splatter damage and knockback … perturbs and soft-threatens" and learned params "should end up encoding this". | Partial (leader/economy; "splatter" absent) | No |
| C4 | **Playing-to-win-matches** by committing bot spawn/travel times to carts. | "playing-to-win-matches (… committing playerbot spawntimes and travel times to the carts)" | Partial (playerbot-interface) | No |
| C5 | **Stealth / information mechanic** — an observation exists only if a bot actually looked toward it (frustum, ≤2 v-cell raycast); enemy positions featurized position/egocentric-weighted. | Turn 250 "if the playerbot actually looked towards the rocket launcher's placement … (woah, stealth mechanics)"; "a 2 v-cell maximum raycast". | Yes (payload-spec: stealth, frustum) | No (raised late, never signed off) |

---

## D. Belief / observation / spatial-mixing machinery (stated; user considers the
*design* given, but not signed off as covered)

| # | Quality | Owner's words (turn) | In docs? | Closed by user? |
|---|---------|----------------------|----------|-----------------|
| D1 | **No team belief — only per-player egocentric belief**, an integration over the team observation map, contracting to an uninformative prior when stale/far. | Turn 244 "we therefore cannot have something like 'team belief' … only players get 'beliefs' … egocentric … more impact if close and fresh". | Yes (belief in payload-spec / strategy-layers) | No |
| D2 | **v-cell minimap featurization with a contraction/context-mask horizon** (parallel, not recurrent), bounded to **no less than 5% / no more than 15% of the map**; complexity scales with contraction parameter + mapsize, NOT playercount. | Turn 243 "v-celling and subfeaturization of cells and a reduction into a low-rank integration … maximum entanglement horizon of cells (aka a context mask)"; Turn 245 "no less than 5% of the map ever, no more than 15%". | Partial (v-cell, context mask present; 5–15% bound?) | Partial (turn 250: "already told you a complete solution … for the minimap horizon" — owner treats the *horizon design* as given) |
| D3 | **Two-input query per bot**: engine game vector ⊕ map-position-conditional belief integration (shared with teammates in the same cell); a **spatial mixing operator (attention)** to spatialize enemy positions. | Turn 244 "the query for each bot comes from the projection of two different inputs, their xonotic normal game engine vector and their map-position-conditional belief integration"; Turn 250 "since we never mentioned using attention or another spatial mixing operator, there was no way for that to actually get spatialized". | Partial (attention in payload-spec/strategy-qkv) | No |
| D4 | **Every buffer online, rebuilt each strategy-estimation step** (this is *why* the mesh is non-vacuously used). | Turn 245 "every single one of these buffers is online and constructed every single strategy estimation step". | Partial | No |

---

## E. Named game-theory constructs the owner raised (status)

| # | Construct | Owner's words (turn) | In docs? | Closed by user? |
|---|-----------|----------------------|----------|-----------------|
| E1 | **Stackelberg — as a LEARNED/derived leader-*team* readout, closed-form, NOT a hand-declared nested game tree.** "leader" is a team property derived from dynamics, never a primitive "leader character." | Turn 245 rebukes the "nested stackelberg" as illocutionary, not spec; Turn 247 "why do i keep seeing mentions of leader character instead of leader team?"; Turn 248 "multiscale stackelberg leader overlay ingame through a closed form solution (probably actually)". | Yes (payload-spec: leader = derived argmax intercentrality); strategy-layers still [OPEN] | No (turn 248 leaves it as "probably"; contradiction with strategy-layers) |
| E2 | **DPP kernel (determinant → diversity semantics)** and **gram/coupling as a *learned* mixing operator (rmsnorm+swiglu), not a fake gram**; **emit a velocity on an integrated weight state**, not an instantaneous decision. | Turn 246 "as a DPP kernel (→ determinant, diversity) — this is probably the only answer"; "don't emit an instantaneous decision, emit a velocity on an integrated weight state"; Turn 248 learned operator. | Yes (dpp-mixing docs, cart-force-field) | **Yes** — turn 249 "okay i think all of the linear algebra decisions are done now" |
| E3 | **Quantal-response / fictitious-play (Brown).** Appear in `payload-strategy-spec.md`. These were introduced by the assistant/compaction summary, **not** by a user turn — not owner-stated requirements. Flag as unrequested methodology to confirm or excise. | (not in any user turn) | Yes (payload-strategy-spec) | n/a — not user-stated; verify against "don't import unrequested methodology" |

---

## F. Geometry / map requirements (stated; mostly realized in `xonotic/…` tools, not
in `design/`)

| # | Quality | Owner's words (turn) | In design docs? | Closed by user? |
|---|---------|----------------------|-----------------|-----------------|
| F1 | ≥3 cart origins **approximately equidistant in navmesh-walking distance**, for any map/cart/team count. | Turn 208 | No (tool-side) | No |
| F2 | Cart paths **within activation distance of standable negative space** (not an "unstick" hack); terrain-following via tangent-energy curvature minimization. | Turn 219, 210, 215 | Partial (cart-force-field) | No |
| F3 | **Track placement anti-aligned** (not all carts same direction — "winning one cart means winning every cart"), branching factor ≥1.5. | Turn 225 | No | No |
| F4 | **Procedural map fusion** (glue k maps via bridge/portal/jumppad/teleporter connectors; verticality), with playerbot navmesh navigability across joins; objective entries easy to notice/fight over. | Turn 223, 240 | No (tool-side) | No |
| F5 | ≥2 carts, ≥3 teams (target k≥5 teams, 3+ carts) forcing computable strategic forks; a cart pushed backward by any non-controlling team regresses toward its own origin. | Turn 187, 222, 93 | Yes (payload-spec multi-cart) | No |
| F6 | **Fully continuous & differentiable cart-velocity near-field** (soften the integer "players contesting in near-field" into a smooth field); its own writeup. | Turn 242 (final of that session) | **Yes** — `cart-force-field.md` written for exactly this | **Yes** (the doc answers the request) |

---

## Bottom line for the superagent

**Fix first (paper-derived, entirely missing):** A1 impartial/partizan honesty · A2
disjunctive-sum ⊕ over independent per-cart components (the missing core) · A3 a real
mex/terminal-0 Grundy recursion instead of a centrality inverse relabeled "nimber" ·
A4 nimber-as-hard-secret tied to incomplete-information belief.

**Reconcile (contradiction):** `strategy-layers-and-modality.md` still says Nim is
"[OPEN, analogy only]" and "not a claim that a Grundy value exists," directly against
`payload-spec.md §2.6 [FIRM]`. One of them is wrong per turn 256.

**Answer, don't assume:** B1 — is there a well-defined REINFORCE *within a second*
(turn 255), versus the doc's match-outcome framing.

**Confirm required, currently only partial:** C1–C4 all-xonotic-mechanics strategic
levers; D3 two-input query + attention spatialization.

**Already user-closed (do not reopen):** the linear-algebra mixing stack — DPP,
learned gram operator (rmsnorm+swiglu), velocity-on-integrated-weight (E2, turn 249);
the differentiable cart-velocity field (F6, turn 242). E3 (quantal/fictitious play) is
assistant-introduced, not owner-stated — verify or excise.
