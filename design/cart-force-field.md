# The cart force field, made continuous and differentiable

The cart velocity law of `payload-strategy-spec.md` §2 reads, in the game code, as a
stack of hard decisions: a player is *in* the cylinder or out, the count of players
is an *integer*, control is an *argmax*, and the contested/abandoned regimes are a
*branch*. Each of those is a discontinuity. This page removes all of them, so that

```
ds/dt  =  f( positions, healths, teams )
```

is `C^infinity` in every player's position and health. The payoff of that is not
tidiness: once `f` is differentiable, `d(cart progress)/d(player position)` exists,
and the solver can price the marginal value of nudging a body by a gradient instead
of by re-simulating every allocation. The whole match becomes a smooth vector field
over player state, which is the object classical differential game theory actually
wants.

## The hard version (what the code does today)

Per player `i`, membership is two boxcar tests, and team presence is their integer
sum:
```
in_i = 1[ r_i < R ] * 1[ |z_i| < h ]                 (r_i horizontal, z_i vertical)
w_j  = sum_{i in team j, alive_i} in_i                (an integer)
c    = argmax_j w_j    (strict; ties -> none)
v    = A-regime if w_c present else B-regime          (a branch)
```
Every one of `1[.]`, `sum of integers`, `argmax`, and the regime branch is
non-differentiable, and several are discontinuous — a player one unit outside the
radius contributes exactly nothing, a tie flips control discontinuously, and the
abandoned regime switches on at `w_A = 0` exactly.

## The smooth version

**Soft membership.** Replace each boxcar with a logistic on its signed margin, so a
player fades in over a shell of width `eps` instead of popping:
```
m_i = sigma( (R - r_i)/eps_r ) * sigma( (h - |z_i|)/eps_h ) ,   sigma(x)=1/(1+e^-x)
```
`m_i -> 1` well inside, `-> 0` well outside, smooth and monotone in `r_i, z_i`. As
`eps -> 0` it recovers the boxcar, so the hard game is the zero-temperature limit.

**Soft aliveness — the integer count dissolves.** The "number of living players
contesting" is the piece that *sounds* integer and is the most interesting to
smooth. A player contributes a continuous *vitality* `a_i in [0,1]`, and team
presence is the sum of membership-weighted vitality:
```
a_i = sigma( health_i / eps_h0 )          (or (health+armor) for a fuller tuple)
w_j = sum_{i in team j} a_i * m_i          (now a real number, not a count)
```
A full-health player deep in the cylinder contributes ~1; a body at the rim, or one
being killed, fades continuously toward 0. "Five players contesting" is no longer a
step function of five spawn/death events but a smooth surface that rises as bodies
arrive and healthy-up and falls as they die or leave — and it is differentiable in
`health_i`, so the marginal value of a point of damage on a contesting enemy is a
real gradient the solver can read.

**Soft control.** Replace `argmax` with a temperature softmax over presences, giving
each team a continuous *control share*:
```
pi_j = softmax_j( w_j / tau )
```
At `tau -> 0` this is the hard plurality (the winner takes `pi = 1`); at finite `tau`
control is *shared* in proportion to presence, which is itself a more honest model of
a contested cart than a knife-edge plurality. The "controlling team" is now the
distribution `pi`, differentiable in every `w_j` hence in every position and health.

**Soft regime gate.** The contested/abandoned branch (§2, Regimes A and B) becomes a
sigmoid blend on the color team's own presence `w_A`:
```
g = sigma( w_A / eps_g )                          (1 = present/contest, 0 = absent/capture)
v_A = clamp( speed*(w_A - w_opp)/(1 + w_opp^2), -contest_speed, +max_speed )
v_B = -reverse_speed * w_B                        (B = the softmax-leading opponent)
v   = g * v_A + (1 - g) * v_B
```
`w_opp = sum_{j != A} w_j` and `w_B` are smooth sums; `g` slides continuously from the
capture walk to the local tug as defenders arrive. There is no instant at which the
law switches — a cart whose defenders are trickling back eases out of its retreat.

Every term is now a composition of sigmoids, softmaxes, sums, and rational functions
of continuous inputs. `f` is `C^infinity`.

## Why the differentiability is load-bearing, not decorative

- **Marginal value of a body, by gradient.** `d(ds/dt)/d(x_i)` and
  `d(ds/dt)/d(health_i)` are closed forms. The solver's allocation head can therefore
  score "move player i one meter toward cart k" or "spend a point of damage on the
  enemy contesting cart k" without re-simulating — the same gradient that a
  differentiable-game solver would use, computed on the coprocessor alongside the Gram.
- **A smooth potential the policy can ascend.** With `f` differentiable, each team's
  contribution to every cart's velocity is a smooth field; the portfolio allocation is
  choosing directions in that field, and REINFORCE on the logit head is doing
  stochastic gradient ascent on the induced return surface. The hard game's
  discontinuities were noise in that gradient; the smooth game removes them.
- **The zero-temperature limit is the playable game.** Shrinking `eps_r, eps_h,
  eps_h0, tau, eps_g` recovers the boxcar/argmax/branch law exactly, so the smooth
  field and the crisp game are one family. The game can be *played* near the hard
  limit (crisp, readable) while the solver *reasons* about the smooth interior where
  the gradients live.

## What runs where

The game code (`sv_payload.qc`) can evaluate the smooth law directly — sigmoids and a
softmax over `j <= 5` teams are cheap per cart per tick — giving live play the
continuous field and, incidentally, a much less twitchy cart than the boxcar count
(no single spawn/death snapping the velocity). The solver's internal model of cart
dynamics uses the same closed form, so the coprocessor's marginal-value gradients and
the server's actual physics are the *same function*, not an approximation of one by
the other. That identity — server dynamics and solver model being one differentiable
`f` — is what lets the RDMA-computed policy steer the exact game being played.
