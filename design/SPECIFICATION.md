# Specification — the quote index (payload strategy)

This is the central specification for the RDMA-mesh Xonotic payload-strategy
project. Its content rule follows the epistemic-provenance law (see the final
section): **every normative sentence here is a verbatim user-authored transcript
block quote**, cited by session prefix + timestamp. Everything that is not a block
quote is provenance, a section title, or a pointer. If a `design/` doc disagrees
with a quote here, the quote governs.

Source: raw transcript `~/.claude-personal/projects/-Users-mdot/d3ad4328-…jsonl`
(session prefix `d3ad4328`). Quotes copied verbatim (user typos preserved). This
file and `AGENDA.md` are the recoverable project state; transcripts/memory/config
are caches.

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

## 7. Don't feature-engineer behavior; saturate the SoCs; the Gram lands in the IR

`d3ad4328`, 2026-08-31T00:38:10Z:

> you aren't suppsoed to try to feature engineer behavior into the solver. the
> solver is a big expensive operation which allows playerbots to 'look context
> conditioned' when they're in motion inside of a match, in ways that are impossible
> if we're not saturating multiple m-series chips with tensor ops. if we're spending
> flops on a gram matrix that gram matrix better fucking end up in the IR consumed
> by subsequent probes, and the value gradient BETTER put trivially sematnically
> measurable features into the learned projections which give a gram matrix
> something with semantic values at all

## 8. The operator is a Gram + SwiGLU (not softmax attention), into a wide IR

`d3ad4328`, 2026-08-31T00:06:00Z:

> where idd a softmax come from? why are you talking about attention? im pretty sure
> a gram matrix and a swiglu were described earlier. what are you talking about lol

`d3ad4328`, 2026-08-31T00:38:10Z:

> how wide did you think the hidden states were supposed to be for this? under 128d?
> maybe you were slippin.

## 9. DPP kernel + velocity-on-integrated-weight (the coupling and the update)

`d3ad4328`, 2026-08-29T21:51:04Z:

> "as a DPP kernel (→ determinant, diversity semantics)," this is probably the only
> answer.

`d3ad4328`, 2026-08-29T21:51:04Z:

> "don't emit an instantaneous decision, emit a velocity on an integrated weight
> state." ... probably the only answer.

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

---

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

Consequence: everything normative in this project's docs must reduce to a §1–§13
quote above, or to code/algebra/proof. Papers (Abdelraouf–Shamma, Burke–Ferland–
Teng, Ballester) are **level-3 support**, admissible only as verbatim quotes of
their text, and are **never** the spec — a doc that cites a paper as the spec (e.g.
`MULTISCALE.md` §3) violates this law and must be re-grounded to the quotes here.
