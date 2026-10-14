# Xonotic on mesh

The game's numerical caller is `planner/plan.py`, the routed-expert bot planner. Each
rank holds a block of every expert's columns (Megatron-LM tensor parallelism), and one
all-reduce per tick completes the expert output: `mesh_mpi`'s `Comm().Allreduce` on
libnccl-mesh (`rdma/NCCL.md`), over the map nccl.h `ncclCommInitRank`'s bootstrap joins
from the ranks' bridges. Each node's bridge is its resident service. The planner is a
`mesh_mpi` world, one rank a node: on every node, from the repository root, with that
node's rank and rank 0's address,

    RANK=r WORLD_SIZE=n MASTER_ADDR=host0 MASTER_PORT=29500 \
        PYTHONPATH=xonotic:rdma bin/mesh-python -B -m planner.plan [ticks] [bots]

This directory also holds the game engine, payload and rendering source. Build the
payload from it with `payload/build.sh`. The engine's side of the fabric is
`darkplaces-work/mesh_ipc.c`, the game's
`qcsrc/common/gamemodes/gamemode/payload/sv_payload_strategy_io.{qc,qh}`.

## The 256-player / 256-team payload contract

The engine and scoreboard admit 256 clients, and shared score, spawn, sound,
weapon-effect, hook, vehicle, damage-text, and team entity streams carry literal short
player or team identities. There is no byte-zero reconstruction of player 256 or team
256. Payload cart state and ribbon endpoints (cart controller/leader/runner-up/home and
link endpoints) are reliable entity messages with full team indices. The team dialog
exposes a numeric selector alongside the stock color shortcuts; numeric selection and
best-team balance compare the direct team arrays at every index up to 256.

The external strategy tick consumes the native source families and returns one typed
response row per addressed state page. The [policy program](../design/POLICY-PROGRAM.md)
defines the global interaction, common output projection and private-view integration.

The bot-controller frame has a separate native `VCellPlan`. `StartFrame` opens one
controller batch and advances `bot_think` exactly once for every bot before DarkPlaces
begins its per-client physics callbacks. World-aware decisions retain ordinary ordered
QuakeC semantics; due keyboard controllers deposit rows instead of applying their pure
transform inline. The native plan derives a three-dimensional
cell extent and causal-cell radius from live player hulls, speed, and frame time, colors
spatially separated cells, and writes the active wave onto every scheduled bot. The pure
Havocbot keyboard transform consumes each entire wave as one structure-of-arrays vector
loop; there is no fixed 512-row buffer or synthetic 32-lane warp. Entity/global delta
transactions are outside this kernel's contract. Fourteen row inputs scatter into the
kernel and nine controller outputs stable-gather at the wave barrier. Vector results commit
after all vector work completes, then the existing engine callbacks consume
the updated controller state exactly once. `bot_nextthink` prevents those callbacks from
recomputing a controller row in the same frame. `[BOTWARP]`, `[VCELL]`, and `[BOTQC]`
report plan shape, native consumption, producer staging, scalar work, and fallback work;
`g_payload_vcell_log 1` enables these without dumping the full strategy transport. A
missing native field leaves staged rows for the scalar controller, so it cannot suppress
a bot update. Full world physics remains ordered by DarkPlaces' per-client callback;
parallel world-delta execution requires a reentrant trace/collision command-buffer
boundary and is not claimed here.

Static geometry measurements exist for all 30 selectable maps. The fused map's
8-team/4-cart artifact measures path lengths of 1,105 / 1,246 / 1,246 / 752 units,
four origins separated by 4,371 units, zero head-on flow, and 240 spawn-to-track
distances of at least 560 units.
