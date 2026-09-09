# Specification — the quote index (payload strategy)

This is the central specification for the RDMA-mesh Xonotic payload-strategy
project. Its content rule follows the epistemic-provenance law (see the final
section): **every normative sentence here is a verbatim user-authored transcript
block quote**, cited by session prefix + timestamp. Everything that is not a block
quote is provenance, a section title, or a pointer. If a `design/` doc disagrees
with a quote here, the quote governs.

Source: raw transcript `~/.claude-personal/projects/-Users-mdot/d3ad4328-…jsonl`
(session prefix `d3ad4328`). Quotes copied verbatim (user typos preserved). This
quote index preserves the requested behavior. `rl-training-spec.md` maps it to
current executable code and checks; transcripts, memory, and generated run files are
caches rather than specifications.

## 1. The task — a combinatorial-game-theory solver, not a bot router

`d3ad4328`, 2026-08-30T23:42:59Z:

> that name is cursed ... is repeated over and over agian by agents who thought
> they were being told to wirte something very different from a combinatorial game
> theory solver which lets a policy recognize whether it is winning or losing in a
> nim-like counting-game-over-payload-carts then commit all resources towards
> winning as fast as possible (insofar as the solver knows how to spend resources
> towards winning)

## 2. Multiscale — two nested subgames, statewise, no recurrence

`d3ad4328`, 2026-08-31T00:19:47Z:

> a laggy and discrete state machine for one subgame (the cartgamestate), and a
> 'basically continuous' rapidly updating state machine for another subgame (the
> xonotic part), with one subgame totally containing the other but being
> intractably complicated and with no obvious links to closed form schemes telling
> us what game states 'mean' in the semantics of winning or losing

`d3ad4328`, 2026-08-31T00:19:47Z:

> no part of this is sequential or stateful and that we aren't interested in
> recurrence (lol), only in learning to price all states along a game history and
> using wacky closed form solvers to get extremely good strategies despite
> interpreting entire mp games statewise

## 3. The policy is a learned filter over full game state

`d3ad4328`, 2026-08-31T00:19:47Z:

> the POLICY is integrating FULL RELEVANT GAME STATE FEATURES like THE HEALTH OF
> ALL PLAYERBOTS AND THEIR AMMO COUNTS AND GUNS therefore the POLICY is implicitly
> a LEARNED FILTER on whether a GUY WITH A ROCKET LAUNHER IN A TEAM SHOULD RUN
> TOWARDS THE CART OR TWOARDS MORE HEALTH AND AMMO, alongside WHICH cart to run
> towards

## 4. Reward is NOT score and NOT whole-game RLVR

`d3ad4328`, 2026-08-31T00:19:47Z:

> computing reward signals tractably without using whole-game rlvr or 'reward=score'
> (which, btw, is totally degenerate and cannot interpret a game where scores
> integrate towards victory. yeah that's right rlvr is for stupid chuds who can't
> compute a derivative and *formally guarantees* policies which cannot notice
> they're being ganged up on by every other team)

`d3ad4328`, 2026-08-30T00:04:31Z:

> now we recognize the reward is sparse again and it's tangible only at the
> projected winner state transition ... (dense reward based on current winner
> clearly doesn't converge on inducing strategic behavior in weight shared
> playerbot policies.)

## 5. W and L are reward definitions; value estimators are linear probes on the IR

`d3ad4328`, 2026-08-31T00:38:10Z:

> W and L are *only* definitions of rewards, which exist to provide a training loss
> for value estimators trained as linear probes upon the final IR from which policy
> actions/vectors are projected for the interface to playerbot control code

`d3ad4328`, 2026-08-31T00:38:10Z:

> the POLICY has its PARAMETERS changed by OPTIMIZATION to increase ADVANTAGE

## 6. The objective of the CGT scaffold: guarantee a big embedded vector

`d3ad4328`, 2026-08-31T00:38:10Z:

> all of this combinatorial game thoery stuff only exists to guarantee that there's
> a big embedded vector which allows learned linear projections to map something
> which can in theory be used to compute value to policy acitons over seemingly
> irrelevant errata to the combinatorial game theory topic, like 'who is holding
> rocket launcher' and 'who should stand on top of the cart to push it faster' or
> even 'wheher or not interacting with carts is causally related to the cartgame
> state vectors'

`d3ad4328`, 2026-08-31T00:38:10Z:

> RL is here to ground the semantic meanings of input vectors just as much as it is
> to induce value learning over a mixing IR output from using gram matrix stuff

## 7. Don't feature-engineer behavior; saturate the SoCs; the Gram matrix lands in the IR

`d3ad4328`, 2026-08-31T00:38:10Z:

> you aren't suppsoed to try to feature engineer behavior into the solver. the
> solver is a big expensive operation which allows playerbots to 'look context
> conditioned' when they're in motion inside of a match, in ways that are impossible
> if we're not saturating multiple m-series chips with tensor ops. if we're spending
> flops on a gram matrix that gram matrix better fucking end up in the IR consumed
> by subsequent probes, and the value gradient BETTER put trivially sematnically
> measurable features into the learned projections which give a gram matrix
> something with semantic values at all

## 8. The operator is Gram-matrix fusion + SwiGLU (not softmax attention), into a wide IR

`d3ad4328`, 2026-08-31T00:06:00Z:

> where idd a softmax come from? why are you talking about attention? im pretty sure
> a gram matrix and a swiglu were described earlier. what are you talking about lol

`d3ad4328`, 2026-08-31T00:38:10Z:

> how wide did you think the hidden states were supposed to be for this? under 128d?
> maybe you were slippin.

## 9. DPP kernel + velocity-on-integrated-weight (the coupling and the update)

`d3ad4328`, 2026-08-29T21:37:38.472Z, raw transcript line 9654:

> finally, moving on: how do we implement strategy and tactics as a state space which implements d_stratweight/d_t and d_tacticsweight/d_t (think team,player here) changes to weighting over change in time for each playerbot, rather than dictating an instantaneous tactic or strategy? the velocity-definition-of-strategy here is going to be necessary if we want to have a strategy computation cadence which is different from a game tick cadence, e.g. 1 strategy integration per second, or 10 strategy integrations per second.

`d3ad4328`, 2026-08-29T21:51:04Z:

> "as a DPP kernel (→ determinant, diversity semantics)," this is probably the only
> answer.

`d3ad4328`, 2026-08-29T21:51:04Z:

> "don't emit an instantaneous decision, emit a velocity on an integrated weight state."
> for similar reasons, probably the only answer.

`d3ad4328`, 2026-08-29T22:32:55.436Z, raw transcript line 9848:

> "Engine-side per-tick integration" two scale integration both decoupled from engine tick for trivial reasons (woha, offloading, the rdma thesis, plugging in 17 more mac minis does not result in an allreduce inside of the engine core loop, only how often the buffer the engine reads from is updated)

Original messages and authorship: [`../measurements/policy-vocabulary-provenance-20260905.json`](../measurements/policy-vocabulary-provenance-20260905.json).

## 10. Sampling + regularization toward logit 0

`d3ad4328`, 2026-08-29T08:57:54Z:

> lets upgrade to something which does logit sampling and has a definition of task
> which results in distributions of strategies which can peak but have some kind of
> basic l1 or l2 regularization towards a logit of 0 at output (meaning without any
> reinforcement from REINFORCE) we see a weighted sampling of effective strategies
> instead of only some actions happening

## 11. The relative objective

`d3ad4328`, 2026-08-30T00:34:06Z:

> the only important thing for an agent to do is to take the path to victory away
> from any other team which isn't winning, and jointly to put their own team in the
> position of path to victory

## 12. matmul = WHAT, stock navmesh = HOW (the QC boundary)

`d3ad4328`, 2026-08-30T23:54:51Z:

> 'what navigation targets should the bot be going towards' is something that has to
> be encoded in the big matmul code partition. 'how do i move around the map to get
> to a target?' is something that is in normal playerbot code

`d3ad4328`, 2026-08-30T23:54:51Z:

> the playerbot code (lol) is just there so that different gameplay objectives can be
> given to playerbots so tehy're doing something instead of nothing

## 13. No fake re-simulation of the game

`d3ad4328`, 2026-08-31T (interrupt during the 00:06 turn):

> if we wrote our linear algebra correctly we should not need to 'test' our code on
> fake resimulations of a videogame that is itself a literal simulation. like, ever.

## 14. W and L stay separate through reward, value, and policy advantage

User, 2026-08-30, this session:

> there are two value estimators: W-heads and L-heads. W-heads have one sparse reward
> they learn to estimate. L-heads have another sparse reward they learn to estimate.
> winning teams are always policy optimized from the W-value. non-winning teams are
> always policy optimized from the L-value.

## 15. L reward is the positive loser-rank-flip event

User, 2026-08-30, this session:

> the L-sparse-reward comesf rom first knowing what the loser hierarchy is, and
> produces sparse reward events for those teams who increase their rank-among-losers,
> null otherwise (losing rank is null, gaining rank is positive, holding rank is null).
> this, if strictly implemented and actually done right instead of wrong, would train a
> value head to estimate states which are closer and closer to a rank flip, and assign
> higher relative advantage to the teams that are closer to flipping cartstate ranks
> relative to other teams, meaning that losers about to lose even harder get even more
> relative negative advantage compared to losers who are increasing rank...

---

## 16. Literal checkpoint game and frozen-state winner

`d3ad4328`, 2026-08-29T20:48:41.744Z:

> "Un-banks A's points on the way down," that's too much; the game structure has to be integrated over time s.t. team cartscore monotonically increases, with d_cartscore/d_t determined by the depth of control * cart lanes.

`d3ad4328`, 2026-08-29T23:26:27.914Z:

> i think yo'ure still ignoring something really obvious, which is that the cart depths form a vector of numbers which produce an instantaneous description of who will win the game if no other state changes, and that this gives you more than enough type information to do translation into terms commensurate to the simplest abstract game

User, September 4, 2026, current session:

> "Delivery" is not aprt of the game mode as defined; carts were to be specified wrt plural checkpoints per path, w/ victory characterized by 'capturing' chains of checkpoints by having a cart moved past them along its path, and total matchscore monotonically increasing by something like sum of checkpoints held.

Executable algebra and current verification: [`CART-GAME-CONTRACT.md`](CART-GAME-CONTRACT.md).

## 17. State vectors and deletion of the fixed action vocabulary

User, September 5, 2026, current session:

> "behavioral coordinate" this sounds like a euphemism. we are talking about state vectors and policy vectors. i have never seen a construction of a 'behavioral coordinate' in these terms, which STILL SOUNDS like some attempt to override full functions of linear transformations of polciy state, havocbot state -> havocbot state.

User, September 5, 2026, current session:

> okay so why was someone trying to slip in a bullshit fixed vocabulary trucating autoencoder into any part of this design? what precedent, what specification, what transcript, what intuitions, whatever make it 'sound like it's part of the problem'?  we need to find out the primary components of 'tedious hand-written wrapper' related ideas and what we need to categorically delete (not 'revise and amend', delete) in the current code and documentation to eliminate this defect

Provenance and deletion inventory: [`POLICY-VOCABULARY-RCA.md`](POLICY-VOCABULARY-RCA.md).

## 18. Exact exponential relaxation; no adapter state exclusions

User, September 5, 2026, current session:

> exact exponential relaxation sounds fine, since the residual against exact exponential relaxation will be very, very, very easy for the policy to learn to emit from trivially early gradient feedback. from here we have a specification telling us how playerbots are steered by the policy, how we integrate the returned value from the policy, and also what inductive biases we want for which playerbot states the adapter inhibits (none of them, other than this exponential relaxation). how straightforward!

State mapping and integration algebra: [`POLICY-STATE-STEERING.md`](POLICY-STATE-STEERING.md).

## 19. Havocbot views, September 5, 2026

Current session, operator, verbatim:

> sure do the related validation. 'should modifications mutate shared game state' nah, only ever a residual upon the views of gamestate teh havocbot has. if the havocbot uses direct unmediated zero copy references to engine state, figure out how to refactor that one in a hurry lol (this should add no substantial latency or arithmetic intensity to the playerbot ai runtime btw)

Residual application is a read operation in the bot computation. Authoritative
entity storage is not a residual destination. Havocbot's ordinary writes and game
commands retain their ordinary effects; view isolation does not undo the causal
consequences of acting on a modified view. Native access paths and their runtime
cost must be validated along with the policy and integration numerics.

## 20. No premature state reduction, September 6, 2026

Current session, operator, verbatim:

> why is anything being averaged or pre-reduced by literally anything at any part of any data flow? discuss, name critical errors, and remediate them

## 21. Fixed execution shapes, September 6, 2026

Current session, operator, verbatim:

> fixed kernel shapes mean that kernel shapes do not have data dependent control flow or the equivalent of recompiling a kernel or doing allocation of buffers inside of a hot loop, like solving a specific policy or value step...
> this is the equivalent of removing graph breaks from pytorch kenrels, as everything taht would be a graph break in pytorch is bad programming for array programming and tensor programming anyways...

Implementation and evidence: [`POLICY-STATE-STEERING.md`](POLICY-STATE-STEERING.md),
[`STATE-REDUCTION-RCA.md`](STATE-REDUCTION-RCA.md).

## 22. Earlier auxiliary query-value task

`d3ad4328`, 2026-08-30T00:26:56.564Z, raw transcript line 10174:

> +1 linear projection on the final intermediate value used to project to strategy velocities, +1 linear projection on the query vectors (formed by learned projection of [gamestate,belief]). final intermediate projection trained on reward signal, early query projection trained on imitation of outputs of other value estimator.

`d3ad4328`, 2026-08-31T07:06:11.841Z, raw transcript line 13154:

> vera is VERA_WINNIE and also VERA_LOU, as there are two values to estimate anywhere value is estimated.

Original messages and authorship: [`policy-head-audit-20260906/provenance.json`](../measurements/policy-head-audit-20260906/provenance.json).

## 23. Common aggregate IR for policy and value heads, September 6, 2026

Current session, operator, verbatim:

> policy and value heads should be both 'reading' from the exact same embedding/IR...? how is there a 'thing that the policy reads' which is described in terms of the policy head's output target instead of the aggregate network?

Current implementation: [`POLICY-PROGRAM.md`](POLICY-PROGRAM.md).

## 24. Whole-program review and deletion of tests, September 6, 2026

Current session, operator, verbatim:

> "  1. Critical — consolidate the network before splitting into heads.
>      The actor has an extra FFN and separate state/residual embeddings
>      unavailable to the main critics." why are these separate at all? to me this sounds like an implementation drift problem of 'treating every statement about one specificagtion of the problem as a totally different tensor program which is unrelated to the last tensor program'. this cannot be fixed by piecewise 'fixing' or 'revising' one step at a time instead of finishing the problem of faked, duplicated, unspecified or spec noncompliant code. we should have found this exact problem ebfore through several previous rounds of code review, yet it remained. perhaps the issue of code duplication, code inlining, and unspecifeid code was *not fully appreciated* as a defect or a scope for code review in previous turns.
> "  4. High — replace inadequate architectural acceptance checks.
>      Existing tests" delete all tests. tests aren't specification and you have inductively proven that tests create deviant pseudospec which is treated as countermanding orders to do random things that bloat project linecount and create (sometimes days) of pointless revision and reconciliation work, like this work now.
> review the data and execution flow more thoroughly and review any outputs which do not have a corresponding input featurization and any inputs which do not have a corresponding output featurization.

Program and input/output correspondence: [`POLICY-PROGRAM.md`](POLICY-PROGRAM.md).
Deletion and source-review evidence: [whole-program review](../measurements/policy-whole-program-20260906/README.md).

## Provenance law (carried from the agentfile / vine-polycompiler stratagem)

> here is a heirarchy of epistemic certainty:
> 1: block quotes from transcript. this is epistemologically from the user.
> 2: "the user said..." this is not epistemologically from the user, and can't be
>    spec if it doesn't have a block quote.
> 3: "the repo says/does/is..." by principle of charity in communication, this can
>    only be said when there is a message. the message cannot be 'the repo
>    says/does/is xyz' unless the claim is block quoting code, or is an algebra, or a
>    proof written in text. therefore anythign annotated iwth 'the repo says/does/is'
>    is not only epistemologically certain to not be from the user, it is also
>    epistemologically certain to be a lie.

Consequence: everything normative in this project's docs must reduce to a §1–§24
quote above, or to code/algebra/proof. Papers (Abdelraouf–Shamma, Burke–Ferland–
Teng, Ballester) are **level-3 support**, admissible only as verbatim quotes of
their text, and are **never** the spec — a doc that cites a paper as the spec (e.g.
`MULTISCALE.md` §3) violates this law and must be re-grounded to the quotes here.


## Havocbot skill parameters remain percepts

Current mesh implementation conversation, 2026-09-07 (user-authored message):

> the one taboo about havocbots we're not allowed to break is allowing the policy to 'strategize' by learning a transform that sets all of the havocbot skill levels to maximum performance in the velocity field interface somehow. they should be available as percepts but be masked out in the output somehow (for some really basic reasons, equivalent to domain randomization in robotics ml)
