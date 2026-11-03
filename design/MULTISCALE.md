# Two nested state spaces, separate W/L estimators

The literal mechanics and algebra are in [CART-GAME-CONTRACT.md](CART-GAME-CONTRACT.md),
grounded in the user quotes in [SPECIFICATION.md](SPECIFICATION.md). A first-delivery
game, reversible accumulated score, and portfolio-XOR winner ranking are not alternative
realizations of that contract.

The full Xonotic state contains the checkpoint game, plus participants, health, armor,
weapons, ammo, locations, pickups and observed resources. Cart motion changes checkpoint
control; held checkpoints determine the nonnegative derivative of accumulated team score.
The smaller state gives a directly computable frozen-control winner and a counterfactual
denial succession. It does not give the unknown effects of FPS actions.

The shared feed-forward policy consumes resource observations, cart instruments and the
literal score/control hierarchy. Gram fusion and SwiGLU produce its representation.
Separate linear W and L heads estimate sparse role-event returns over that representation.

- W rows receive minus one when their previously projected winner loses that role,
  otherwise zero.
- L rows receive plus one when their team rises in rank among nonwinners, otherwise zero.
- Changing role terminates the old role's bootstrap. Actual match termination ends all
  bootstrap; a cart retreat is not match termination.
- Role-selected, stopped-gradient advantages update the policy. Historical observed
  returns train value estimation through the backbone without treating historical actions
  as supervised demonstrations.
- The requested terminal-win PPO ablation fits and optimizes from actual score-threshold
  outcomes, without changing the game or replacing stratCGT's reward definition.

The policy acts statewise; learning observes real transitions and replay. There is no
recurrent learned policy, guessed simulator, terminal-delivery proxy, or deterministic
combat script standing in for the strategy computation.

Frozen-control first-passage and explicit checkpoint-denial calculations are exact for
the algebra they evaluate. They do not establish a Sprague–Grundy equivalence for the
multiplayer FPS dynamics, nor a theorem that one training objective must beat another.
The live mixed-team comparison measures that behavioral question.
