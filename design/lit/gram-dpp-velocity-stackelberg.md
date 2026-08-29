# Literature review: DPP-diversity coupling, velocity/replicator strategy flow, and learned Stackelberg leadership

Scope: a focused review for the payload-game strategy design (`design/payload-strategy-spec.md`,
`design/strategy-qkv.md`, `design/cart-force-field.md`). Two decisions are fixed and are the
frame, not the subject, of this review: (1) the coupling operator over agents' candidate
instruments is a **DPP / Gram-determinant** object chosen for anti-redundancy; (2) strategy is
emitted as a **velocity** `dw/dt` on an integrated weight state, decoupling strategy cadence
(1-10 Hz) from the 72 Hz tick.

Modality convention used throughout: **[SETTLED]** = the cited literature establishes it as a
theorem or a reproduced empirical result; **[INFERENCE]** = my own reasoning connecting the
literature to this design, not something a paper states about this design; **[OPEN]** = a
modeling choice the literature does not decide for us. The design owner is strict about this
line, so I keep it explicit.

---

## A. DPPs for diverse multi-agent action/target selection

**What a DPP is and why the determinant induces diversity.** A determinantal point process over a
ground set of `N` items is a distribution over subsets `Y` with `P(Y) proportional to det(L_Y)`,
where `L` is an `N x N` positive-semidefinite **L-ensemble** kernel and `L_Y` is its principal
submatrix on `Y` (Kulesza & Taskar, *Determinantal Point Processes for Machine Learning*, FnT ML
2012, arXiv:1207.6083). The standard **quality-diversity decomposition** writes
`L_ij = q_i * phi_i^T phi_j * q_j`: `q_i >= 0` is a per-item *quality* score and `phi_i` (unit
norm) is a *feature/diversity embedding*, so `det(L_Y) = (prod q_i^2) * Vol^2({phi_i})`. The
determinant is the squared volume spanned by the selected items' feature vectors — collinear
(redundant) items span near-zero volume and are exponentially suppressed, which is exactly the
"don't pile every bot onto collinear instruments" property the design wants. [SETTLED]

**k-DPPs and conditioning.** A **k-DPP** (Kulesza & Taskar, ICML 2011) conditions a DPP to fixed
cardinality `k`, giving direct control over how many instruments get committed and decoupling
"how diverse" (kernel) from "how many" (budget). DPPs are closed under **conditioning on
inclusion and exclusion**: fixing that some items are in (or out) of `Y` yields another DPP whose
kernel is a Schur-complement update of `L` (Kulesza & Taskar 2012, Sec. 5.2). This is the exact
mechanism behind Part D below: "select diversely *given* the leader's already-occupied set" is a
conditional-DPP marginal, computable in closed form. [SETTLED]

**Sampling vs MAP.** Exact sampling is an `O(N^3)` eigendecomposition plus an `O(Nk^2)` elementary
step (the spectral algorithm, Hough et al. 2006 as presented in Kulesza & Taskar 2012). **MAP**
(the single most-diverse-and-high-quality set) is **NP-hard** but the log-det objective is
**submodular**, so greedy selection has a `1 - 1/e` guarantee; Chen, Zhang & Zhou (NeurIPS 2018,
"Fast Greedy MAP Inference for DPP") give an `O(N k^2)` incremental-Cholesky greedy that is the
practical default. For our cadence (a solve every 100 ms-1 s over tens of instruments), greedy MAP
is cheap; sampling is the choice only if we want stochastic exploration in the outer REINFORCE
loop. [SETTLED] / [OPEN — sample vs MAP is ours to pick, see D]

**Training a kernel from LEARNED embeddings, and REINFORCE-friendliness.** This is the load-bearing
question. Established results:
- **Maximum-likelihood learning of DPP kernels is tractable and gradient-friendly.** Gillenwater,
  Kulesza, Fox & Taskar (NeurIPS 2014, "Expectation-Maximization for Learning DPPs") and Mariet &
  Sra (ICML 2015) give EM / fixed-point learners; the log-likelihood gradient reduces to
  **item marginals and subset expectations that are efficiently computable** because of DPP
  tractability. The catch the literature is explicit about: the log-likelihood is
  **non-convex in the kernel**, and for k-DPPs the normalizer `e_k(L)` (an elementary symmetric
  polynomial in eigenvalues) makes the gradient carry a `sum over C(N,k)` term handled via
  recursive `e_k` evaluation, not brute force (Affandi, Fox, Adams & Taskar, ICML 2014, "Learning
  the Parameters of DPP Kernels", arXiv:1402.4862). [SETTLED]
- **Deep / end-to-end kernels work.** Gartrell et al. (AAAI 2019, "Deep DPPs", arXiv:1811.07245)
  learn `phi_i = f_theta(item)` end-to-end through the DPP log-likelihood and beat low-rank DPPs
  built on separately-learned embeddings. Direct support for the design's "features feed a learned
  kernel" bet. [SETTLED]
- **The policy-gradient outer loop is the well-trodden path in MARL.** **Q-DPP / Multi-Agent
  Determinantal Q-Learning** (Yang, Wen, Wang et al., ICML 2020, arXiv:2006.01482; code
  github.com/QDPP-GitHub/QDPP) is the closest prior art to this whole design. Its structure:
  the joint action set is a DPP with a **partition-matroid constraint** — the ground set is
  partitioned into per-agent action blocks and exactly one item is drawn per block, i.e. **each
  bot commits to exactly one instrument** — and the kernel again factorizes into per-agent
  **quality** (reward-seeking) and **diversity** (feature) parts. The `det` couples the agents:
  the joint value factorizes *without* hand-imposed VDN/QMIX/QTRAN mixing structure, and Q-DPP
  provably generalizes all three. They train with a **linear-time sample-by-projection sampler**
  (partition-matroid exact sampling is `O(mp)`, `m`=ground-set, `p`=partitions) and note the DPP
  sampling itself doubles as a **coordinated-exploration** mechanism (agents cover orthogonal
  state-space directions). [SETTLED]

**Bottom line for A.** The determinant-as-volume diversity, k-DPP budgeting, closed-form
conditioning, submodular greedy MAP, and tractable *likelihood* gradients are all settled. The
specific pairing "DPP selection head + REINFORCE outer loop over learned quality/diversity
features, one-instrument-per-agent via a partition matroid" is not a novel gamble — it is
essentially the Q-DPP recipe, which is the single sharpest paper to read next for A.

Caveat the literature forces us to hold: DPP *likelihood* training (max the log-prob of observed
diverse sets) and *policy-gradient* training (max reward of sampled sets) are **different outer
losses**. Q-DPP uses the DPP as the Q-function factorization and learns by Q-learning; our design
uses the DPP as an *action-selection coupling* scored by REINFORCE. Both are in the literature,
but they are not the same objective, and which one we run is a real choice (see D).

---

## B. Velocity / flow forms of strategy over a weight simplex

The design emits `dw/dt` and integrates `w(t+Delta) = w(t) + (dw/dt) Delta`. The literature that
tells us when such an emitted velocity has an integral that *tracks an equilibrium*:

**Replicator dynamics is the canonical "strategy as velocity on weights".** The replicator equation
`dw_i/dt = w_i * (f_i(w) - <f>(w))` — each weight's growth rate is its fitness minus the population
mean fitness — is exactly "velocity on the simplex" and keeps `w` on the simplex by construction
(Taylor & Jonker 1978; Hofbauer & Sigmund, *Evolutionary Games and Population Dynamics*, CUP 1998;
Sandholm, *Population Games and Evolutionary Dynamics*, MIT 2010). Key facts that bear on cadence
and tracking:
- **Folk theorem of evolutionary game theory** [SETTLED]: Nash equilibria are the rest points of
  the replicator flow; strict Nash equilibria are asymptotically stable; any asymptotically stable
  rest point is a Nash equilibrium. So an emitted replicator-style velocity has fixed points that
  *are* the equilibria we want.
- **But convergence of the trajectory is NOT guaranteed.** In zero-sum / cyclic games the flow
  produces **closed orbits (neutral stability)** and does not converge pointwise — it cycles around
  the equilibrium. What *is* guaranteed is that the **time-average of the trajectory converges to
  Nash** (this is the same time-average result that underlies no-regret learning; see below).
  [SETTLED] This is a direct warning for the design: integrating an emitted `dw/dt` can orbit the
  equilibrium rather than settle, and the usable quantity may be the time-averaged weight, not the
  instantaneous one.

**The no-regret / online-learning bridge.** Replicator dynamics is the continuous-time limit of
**multiplicative-weights / Hedge**, and the general result is: if every agent runs a
**no-regret** learning rule, the **empirical (time-averaged) play converges to the set of coarse
correlated equilibria** (Hart & Mas-Colell 2000; Cesa-Bianchi & Lugosi, *Prediction, Learning, and
Games*, 2006; Blum & Mansour). Follow-the-Regularized-Leader / mirror-descent dynamics are the
umbrella (the replicator is FTRL with the negative-entropy regularizer, i.e. mirror descent in the
KL geometry). [SETTLED] The design's "REINFORCE with L1/L2 regularization toward a broad sampling
distribution" is recognizably in this family: a regularized-logit velocity is a mirror-descent
step, and the regularizer is what keeps the flow from collapsing to a vertex.

**Step-size / cadence vs tracking error.** This is where the design's 1-10 Hz emit, 72 Hz integrate
split meets theory:
- **Discretization matters.** The **discrete-time** replicator can be **chaotic** where the
  continuous flow is benign; larger step size (here: lower emit cadence, larger `Delta` per emitted
  velocity) pushes toward instability and chaos (Pangallo, Sanders, Galla, Farmer and successors —
  e.g. arXiv:2402.09824, "On the discrete-time origins of the replicator dynamics: from
  convergence to instability and chaos"). [SETTLED] The design's `w += (dw/dt)*Delta` is precisely
  a forward-Euler discretization, so `Delta` (set by emit cadence) is a stability parameter, not a
  free performance knob.
- **Tracking a moving/time-varying equilibrium.** In **strongly monotone** games (Rosen's regime,
  Part C-adjacent) online gradient / mirror-descent tracks a drifting equilibrium with **tracking
  error that scales with the equilibrium's path length (drift rate) times the step size**; faster
  emit cadence shrinks the tracking-error floor (Yan et al., ICML 2023, "Fast Rates in Time-Varying
  Strongly Monotone Games"; Duvocelle et al. on online learning in time-varying games). [SETTLED
  for strongly-monotone games] For general games no such clean bound exists. [OPEN — our game is
  not known to be monotone]

**Bottom line for B.** "Emit a fitness-minus-average velocity on the weights and integrate it" is
the replicator equation, a canonical and well-understood object; its fixed points are Nash and its
time-average tracks (coarse-)correlated equilibria under a regularizer. **What is settled** is the
fixed-point / time-average behavior and that emit-cadence = Euler step-size = a *stability*
parameter with a chaos threshold. **What is open** is whether our specific game is monotone enough
for pointwise (not just time-averaged) convergence, which is what would license using the
instantaneous `w` rather than a running average.

---

## C. Stackelberg / leader-follower with LEARNED leadership

**Backward-induction / bilevel structure.** A Stackelberg game is the bilevel problem: leader
commits `x`, follower best-responds `y*(x) = argmax`, leader optimizes anticipating `y*(x)`. In
MARL this is realized by **alternating** follower best-response learning with leader policy
improvement, and can be learned **without knowing the follower's reward** (Zhong et al.,
JMLR 2023, "Can RL Find Stackelberg-Nash Equilibria?"; contextual/bilevel RL, arXiv:2406.01575).
Differentiable-game versions use the **total/Stackelberg gradient** `d/dx [ f(x, y*(x)) ]` with the
implicit-function term through `y*(x)` (Fiez, Chasnov & Ratliff, "Implicit Learning Dynamics in
Stackelberg Games", ICML 2020). [SETTLED]

**Emergent / learned leadership rather than hand-assigned roles.** The design wants leadership to be
a *readout of state* (a team's resource concentration near a lane), not a declared role. Prior art:
- **ST-MADDPG / Stackelberg autocurricula** (Huang et al., arXiv:2305.03735, "Stackelberg Games for
  Learning Emergent Behaviors During Competitive Autocurricula") impose a leader-follower gradient
  bias and get **richer emergent behavior and symmetry-breaking** than symmetric MADDPG — evidence
  that the Stackelberg structure *causes* qualitatively better play, but the leader is still
  *assigned*, not read from state. [SETTLED that assigned leadership helps]
- **Markov Stackelberg games with role-switching**: the leader/follower roles can be **interchanged
  per state**, giving emergent leadership patterns (Learning in Stackelberg Markov Games,
  arXiv:2509.16296; Mediator-induced fair leaders, arXiv:2508.02421). This is the closest published
  support for "who leads is a function of the current state". [SETTLED that state-dependent role
  assignment is a studied, workable construct] — but note none of these compute leadership from a
  *physical resource-concentration readout*; that specific readout is our design contribution, not
  something the literature validates. [OPEN]

**Non-dominated follower responses.** The concern "followers must not emit obviously dominated
responses to a committed leader" is exactly the follower-best-response step: by construction the
follower plays `argmax` against the committed leader action, which is non-dominated *by
definition*. The subtlety the literature flags: with **multiple follower equilibria**, the leader's
guaranteed payoff depends on **optimistic vs pessimistic** tie-breaking (strong vs weak Stackelberg
equilibrium), and pessimistic Stackelberg values can be discontinuous in the leader's action
(Fiez et al.; classic bilevel-optimization literature). [SETTLED — this is a known hazard, not a
blocker]

**Coalition formation, >2 players, multi-objective, and the "key player" as the swing-lane
computation.** The design already names coalition formation and pigeonhole (spec Sec. 3-4). The
sharpest literature match for "which lane is the swing" is the **network-games key-player** result:
- **Ballester, Calvo-Armengol & Zenou, "Who's Who in Networks. Wanted: The Key Player",
  Econometrica 2006.** In a linear-quadratic game with local complementarities encoded by a
  coupling matrix `G`, the **Nash action of each player is proportional to their Bonacich
  centrality** `b(G, a) = (I - aG)^{-1} * 1` (with `a` the complementarity strength). The **key
  player** — the one whose removal maximally reduces aggregate activity — is *not* the most central
  by Bonacich; it is the argmax of an **intercentrality** measure that also accounts for the
  player's contribution to *others'* centralities, and it is computed from the *same inverse
  coupling matrix* `(I - aG)^{-1}`. [SETTLED]
- **Direct relevance.** The design's solver already forms a coupling/Gram object over
  (team, instrument) groups (`kappa = V Gm V^T`, `strategy-qkv.md` Sec. 0-1) and the spec calls for
  an inverse-coupling read. **[INFERENCE]** The "which cart is the swing lane / which resource
  concentration is the key player" question is *formally* the intercentrality argmax over that
  coupling matrix: equilibrium instrument-intensity `proportional to (I - a*kappa)^{-1} * 1`, and
  the swing lane is the column whose removal most reduces the aggregate. This gives the design a
  *named, closed-form* candidate for the leadership/swing readout instead of a heuristic — and it
  is computed from an object the solver already materializes. This is an inference I am drawing, not
  a claim any paper makes about a payload game.

**Bottom line for C.** Bilevel/backward-induction learning, learned/state-dependent leadership, and
Stackelberg-gradient follower responses are settled constructs. The key-player / Bonacich result
gives a principled swing-lane computation from an inverse coupling matrix the solver already builds.
What is open is our specific bet that a *physical resource-concentration tuple* is a good leadership
readout — the literature supports "leadership as a learned readout of state" in general but does not
bless this particular feature.

---

## D. The synthesis: do DPP-diversity + replicator velocity + learned leadership cohere?

**The three pieces are compatible, and two of them are literally the same object viewed twice.**
[INFERENCE, grounded in the cited results]
1. The DPP coupling (A) is a *selection* operator over instruments; the replicator velocity (B) is
   a *flow* over the weights that parameterize that selection. Q-DPP already runs exactly this
   stack: a DPP over per-agent action blocks whose quality/diversity features are moved by an outer
   RL loop. Swapping Q-learning for REINFORCE is a change of outer loss, not of the coupling. So
   "DPP coupling + policy-gradient velocity on its parameters" is a realized architecture, not a
   speculative combination. [SETTLED that the pairing runs; INFERENCE that our REINFORCE variant
   inherits its coordination properties]
2. The DPP L-ensemble Gram and the key-player coupling matrix (C) are the *same* second-moment
   object read two ways — `det`-diversity for anti-redundant selection, `(I - a*kappa)^{-1}` for
   swing-lane/leadership. The design's `strategy-qkv.md` already flags that the Gram is currently
   "folded twice and thrown away to a scalar"; the literature says both reads are principled and
   both want the *un-folded* matrix. [INFERENCE]

**The specific hypothesis under test** — *diversity-seeking (high-`det` DPP) against a committed
leader's occupied action-set is mechanically the non-dominated response: it selects the lanes the
leader vacated (pigeonhole), while "pile onto the leader's key lane" (coalition) is the
low-diversity / high-shared-appetite regime, and REINFORCE prices which regime applies.*

Assessment: **the literature SUPPORTS the mechanism and REFINES the framing; it does not refute
it.**
- **Support for the vacated-lanes half.** Conditioning a DPP on the leader's occupied set being
  *included* (or equivalently selecting the *complementary* diverse set) is the closed-form
  **conditional DPP** (Kulesza & Taskar 2012, Sec 5.2): after conditioning, an item collinear with
  the leader's occupied features has its inclusion marginal driven down by the Schur-complement
  update — the *pigeonhole* effect is literally the determinant volume collapsing on the leader's
  spanned subspace. So "diversity against the leader's set picks what the leader vacated" is a
  theorem about conditional DPPs, not just an intuition. [SETTLED as a DPP fact; INFERENCE that it
  equals the follower best-response]
- **Refinement the literature forces.** "Diverse response is the non-dominated response" is true
  *when payoffs are (sub)modular / anti-complementary* — when instruments are substitutes so
  spreading out is optimal. But the key-player / linear-quadratic model (C) is built on **local
  complementarities**: when actions are *complements* (a=+, piling on a lane is synergistic), the
  Nash response *concentrates* (Bonacich), i.e. the **coalition / low-diversity regime**. So the
  two regimes in the hypothesis map cleanly onto the **sign of the coupling / substitutes-vs-
  complements** in exactly the two canonical models: DPP (substitutes -> diversify -> take vacated
  lanes) vs linear-quadratic network game (complements -> concentrate -> pile on key lane). The
  hypothesis is not just supported, it is the substitutes/complements dichotomy stated in game
  terms. [SETTLED that these are the two regimes; INFERENCE that our game switches between them]
- **What REINFORCE actually prices.** The claim that "REINFORCE prices which regime applies" is
  supported in the weak sense that the outer loss will push the *quality* weights `q_i` and the
  *diversity* embeddings `phi_i` (and, if we expose it, the complementarity sign/strength `a`)
  toward whichever regime earns reward — this is standard policy-gradient credit assignment over a
  DPP kernel (Gartrell deep-DPP + Q-DPP). But **no paper shows a single learned kernel smoothly
  interpolating substitutes <-> complements as a function of state**; DPP kernels are PSD and model
  *negative* correlation (repulsion) natively, whereas complementarity is *positive* correlation
  (attraction), which a standard (symmetric PSD) DPP **cannot represent**. This is the sharpest
  refinement/caution: a vanilla DPP is structurally a *diversity-only* operator. [SETTLED —
  DPPs model repulsion only]

  Two literature-backed escape hatches: (i) **Nonsymmetric DPPs (NDPPs)** (Gartrell et al.,
  NeurIPS 2019 / ICLR 2021, "Scalable Learning and MAP Inference for Nonsymmetric DPPs",
  arXiv:2006.09862) drop the symmetry constraint and **can model positive correlations /
  attraction** as well as repulsion — i.e. an NDPP kernel *can* represent both the pile-on and the
  spread-out regime in one learned object. (ii) Keep the DPP diversity-only and put the
  concentration regime in a *separate* additive quality term (the `q_i` mass on the key lane),
  letting the two heads compete. The design must pick one. [OPEN]

**On the "represent, don't evaluate" bet** (hand-featurize only raw engine tuples so leadership /
dominance / swing-value are *representable*, push all *evaluation* into learned parameters):

The literature **supports this bet, with one precise qualification.**
- **Support.** This is the **reservoir-computing / random-features** thesis: fixed (even random)
  feature maps with a *trained linear readout* are universal approximators for a large function
  class — you do not need to learn the representation to evaluate arbitrary functionals of it, you
  need only that the representation *spans* them (reservoir-computing universality results, e.g.
  Grigoryeva & Ortega; random-features, Rahimi & Recht NeurIPS 2007). The design's own
  `strategy-qkv.md` correctly identifies its current solver as exactly this regime ("an output head
  learned on top of frozen random features... the reservoir / random-feature regime"). The DPP
  quality/diversity decomposition is *built* for this split: `q_i` and `phi_i` are learned
  *evaluations* on top of whatever features you hand it (Kulesza & Taskar 2012; Gartrell deep-DPP).
  So "make leadership/swing representable in raw tuples, learn their evaluation" is the standard and
  validated division of labor. [SETTLED that trained-readout-on-fixed-features is expressive]
- **The qualification.** Representability is a *spanning* condition, and random/fixed features pay
  for it in **sample efficiency and dimension**: universal approximation with fixed features can
  need a very wide feature layer, and the readout only learns functions the feature map already
  linearly exposes. The design's move to make the *query/value geometry learnable* (`strategy-qkv.md`
  Part 4: the learnable diagonal/low-rank `M`, `Wq/Wk/Wv`) is precisely the literature's
  recommended fix when a pure fixed-feature readout underfits — end-to-end deep-DPP beats
  fixed-embedding low-rank DPP (Gartrell 2019) for the same reason. So "represent-don't-evaluate"
  is supported as a *starting point / floor*, and the literature predicts that **partially learning
  the feature geometry** (not just the readout) is what buys the last increment — which is already
  the direction `strategy-qkv.md` proposes. [SETTLED that some feature learning helps; OPEN how much
  to freeze vs learn is our tuning axis]

---

## Per-question: SETTLED / OPEN / sharpest next paper

**A — DPPs for diverse multi-agent selection.**
- SETTLED: determinant = squared feature-volume = anti-redundancy; quality/diversity kernel
  factorization; k-DPP budgeting; closed-form conditioning; submodular greedy MAP with `1-1/e`;
  tractable, gradient-friendly *likelihood* learning; end-to-end deep kernels; and the full
  pairing "partition-matroid DPP (one instrument per bot) + RL outer loop" as a working MARL method.
- OPEN: sample vs greedy-MAP for our selection; and whether our outer loss is DPP-likelihood or
  reward policy-gradient (they are different objectives).
- Sharpest next: **Yang, Wen, et al., "Multi-Agent Determinantal Q-Learning", ICML 2020**
  (arXiv:2006.01482) — it is this design's coupling + a MARL loop, end to end.

**B — Velocity/flow strategy over the weight simplex.**
- SETTLED: `dw/dt = fitness - mean` is the replicator equation; rest points = Nash; strict Nash
  asymptotically stable; in cyclic/zero-sum games trajectories orbit but the *time-average*
  converges (the no-regret / mirror-descent bridge); emit-cadence is a forward-Euler step size with
  a discrete-time chaos threshold.
- OPEN: whether our game is monotone enough for *pointwise* (not just time-averaged) tracking, which
  is what would let us trust instantaneous `w`; the concrete emit-Hz vs tracking-error/stability
  operating point for our dynamics.
- Sharpest next: **Hofbauer & Sigmund, *Evolutionary Games and Population Dynamics* (CUP 1998)** for
  the folk theorem and the time-average result; pair with **Mertikopoulos & Sandholm, "Learning in
  Games via Reinforcement and Regularization" (Math. of OR 2016)** for the FTRL/mirror-descent =
  regularized-replicator link that our regularized-logit velocity actually is.

**C — Learned Stackelberg leadership.**
- SETTLED: bilevel/backward-induction RL with follower best-response + leader improvement, learnable
  without the follower's reward; Stackelberg gradient; state-dependent / interchangeable leadership
  is a studied, workable construct; Bonacich-proportional equilibrium and the key-player
  intercentrality argmax from the inverse coupling matrix.
- OPEN: whether a *physical resource-concentration tuple* is a good leadership readout (literature
  blesses "learned readout of state" generically, not this feature); optimistic vs pessimistic
  follower tie-breaking for our multi-equilibrium lanes.
- Sharpest next: **Ballester, Calvo-Armengol & Zenou, "Who's Who in Networks. Wanted: The Key
  Player", Econometrica 74(5), 2006** — gives the swing-lane computation as a closed form over the
  coupling matrix the solver already builds.

**D — The synthesis.**
- SETTLED: the three pieces coexist (Q-DPP already runs DPP-coupling + RL-velocity); the
  vacated-lanes-via-diversity effect is a conditional-DPP theorem; the diversify-vs-pile-on
  dichotomy is exactly substitutes-vs-complements (DPP repulsion vs linear-quadratic
  complementarity); trained-readout-on-fixed-features is expressive (reservoir/random-features), so
  "represent, don't evaluate" is a supported floor.
- OPEN / the one caution the literature forces: a **symmetric PSD DPP models repulsion ONLY** — it
  cannot represent the positive-correlation "pile-on/coalition" regime in the *same* kernel. To let
  REINFORCE actually *switch* regimes in one object we need either **nonsymmetric DPPs** or a
  separate additive concentration head. Also open: how much feature geometry to freeze vs learn.
- Sharpest next: **Gartrell, Han, Dohmatob, Gillenwater & Brunel, "Scalable Learning and MAP
  Inference for Nonsymmetric DPPs", ICLR 2021** (arXiv:2006.09862) — it is the one result that
  decides whether the diversify/pile-on switch can live in a single learned kernel.

---

### One-paragraph verdict for the design owner

The design is not composing three unrelated ideas: the DPP coupling + policy-gradient velocity is
substantially the **Q-DPP** architecture, the velocity flow is the **replicator/mirror-descent**
object with known fixed-point and time-average guarantees, and the swing-lane/leadership readout has
a closed form in the **key-player intercentrality** over the very Gram the solver already computes.
The hypothesis that diversity-against-the-leader is the non-dominated (vacated-lane) response is a
conditional-DPP theorem, and it is the *substitutes* half of the substitutes-vs-complements
dichotomy; the coalition/pile-on half is the *complements* half. The single sharpest technical risk
the literature surfaces is that a **standard symmetric DPP can only model the diversity/repulsion
regime**, so pricing "which regime applies" inside one learned kernel requires a **nonsymmetric
DPP** (or an explicit second concentration head) — that is the concrete modeling fork to resolve
next.
