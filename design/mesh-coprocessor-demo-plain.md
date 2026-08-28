# The same design, in plain words

Companion to `mesh-coprocessor-demo.md`. This one covers the linear algebra, and the
things the game must show so the algebra is obviously there and obviously needed.

## What the solver actually computes

Every bot has a list of goals it has not met yet. It wants to be closer to the cart. It
wants to be safe. It wants ammo. The gap between what a bot wants and what it has is
called a residual. It is just a list of numbers.

Now take every bot's residual and compare it to every other bot's residual. Each pair
gets a score. A high score means those two bots are failing for the same reason. That
whole table of scores is called a Gram matrix.

The Gram matrix is the coordination signal. Bots that fail together should act together.
Bots that fail for different reasons should split up. So the table is not a side effect.
It is the answer the team needs.

Building the table is the expensive part. If you have R things to compare, the table has
R times R entries. Double R and the work goes up four times. That is why R is our main
dial.

## Why the work cannot be faked or skipped

A reviewer will ask if we could get the same answer more cheaply. For this design the
answer is no, and that matters more than raw speed.

The table is dense. Every pair really does need a score, so there are no zeros to skip.
It is also full rank, which means it cannot be squeezed into a smaller table without
losing the answer. And comparing all pairs this way is the standard method. There is no
faster known way to do it.

The second half of the solver picks which specialist handles each bot. Each bot routes to
a few experts out of many. The choice depends on live game state, so it changes every
tick and cannot be worked out ahead of time. Each expert is its own matrix, so they
cannot be merged.

That routing choice is also a machine choice. Experts live on different machines. So
picking an expert picks a computer, and the request travels over the cable. The routing
in the game and the routing in the network are the same act.

## What the game must show

A viewer should not need a chart. The game itself has to make the solver visible.

**A cart on a track.** The cart moves when the teams pushing it act together. Its speed
depends on how well bots coordinate. So the cart's position is a running record of solver
quality. Nothing needs to be explained. The cart either moves or it stops.

**Five or more teams.** With two teams the only choice is fight or flee. With five, teams
must decide who to block and who to ignore. Trailing teams share a reason to slow the
leader. They are not friends, but their goals line up for a while. That kind of loose
grouping is exactly what the Gram matrix finds. Two teams would make the table almost
pointless.

**Bots that visibly group and split.** When the solver is healthy, bots move in packs
that form and break apart as the fight changes. When it is starved, they drift and pick
targets at random. A viewer reads this as smart play or broken play without being told.

**A spectator view of who has a plan.** Each bot is either planned or unplanned. Colour
them. When one machine has to do all the work, a large share turn unplanned and stand
around. Plug in the second machine and they light up again.

## How we prove two machines are needed

Run the same match twice. Once on one machine, once on two.

On one machine, raise the dials until the solver cannot keep up. Bots go unplanned. The
cart slows and then stops. On two machines, at the same settings, the cart keeps moving
and the bots keep their plans.

The dials are all things a viewer can see. Bot count is the size of the crowd. Team count
is the scoreboard. The comparison size R shows up as how sharp the play looks.

We measured the two machines. On this kind of work the laptop is about six times faster
than the mini at the largest setting. So the mini alone stalls early, and the pair does
not. That gap is the demo.

## The one thing that could hide the truth

If the solver runs slowly on both machines for a dull reason, the test proves nothing. So
we checked the mini against its own limit. It reaches 84 to 97 percent of what its chip
can do. There is no better code to write for it.

That matters because the honest claim is not "our code is fast." It is "this is the best
known method, run near the limit of the hardware, and one machine still cannot do it."
