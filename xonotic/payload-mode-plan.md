# Xonotic checkpoint-control implementation map

The earlier delivery-game plan was not the specified game and has been removed.
The authoritative user quotes are in [SPECIFICATION.md §16](../design/SPECIFICATION.md),
and the implemented algebra is in [CART-GAME-CONTRACT.md](../design/CART-GAME-CONTRACT.md).

There are independently configured teams and cart lanes. Each lane carries plural
checkpoints. Held checkpoints determine the derivative of monotonically increasing
team score. The score threshold determines victory; path endpoints are not deliveries.
Projected winner and denial succession use that same score ledger and checkpoint state.
W/L remain sparse role-loss/rank-improvement objectives; terminal-win PPO is the
separately requested ablation on the same game.

| Responsibility | Canonical implementation |
| --- | --- |
| Cart physics, checkpoint control, integration, score outcome | `qcsrc/common/gamemodes/gamemode/payload/sv_payload.qc` |
| Mode defaults and score configuration | `payload/cfg/gamemodes-payload.cfg` |
| Independent team/lane/checkpoint realization | `payload/tools/mkentfile.py` |
| Literal participant, cart, team and event wire rows | [Strategy I/O](payload/STRATEGY-IO.md) |
| Frozen-control first passage and denial succession | `solver/strat/game_value.py` |
| W/L event rewards and hierarchy observations | `solver/strat/runtime.py` |
| Shared policy/value optimizer and historical value mixture | `solver/strat/online.py` |
| Persistent server/client, outcome-driven match transition | `solver/strat/curriculum.py` |

The [payload README](payload/README.md) documents the existing direct-velocity mover,
contested/reversal law, entity format and presentation. Stock navigation realizes
selected policy targets; it does not define a second game or reward function.

Build with `payload/build.sh <output-directory>`. Verify the actual generated map,
server observations, score outcome and policy provenance. `measure.py cgt` checks
recorded real-game state against the shared algebra; it is not a fake game simulator
or evidence that a policy has learned strategic superiority.
