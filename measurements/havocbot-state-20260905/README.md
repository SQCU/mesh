# Havocbot numeric state inventory

This snapshot predates the state-view replacement and decoder deletion. Current
compiled offsets are regenerated in the [view validation inventory](../havocbot-view-validation-20260905/inventory/summary.json).

This is an enumeration of existing storage, not a proposed action menu or feature
selection. It was extracted from the deployed `progs.dat` and the current source.
The laptop's staged game binary and the Mini's runtime binary have identical
SHA-256 fingerprints and produce identical counts. The game workload remained
stopped throughout the audit.

## Counts and exact enumeration

| Storage scope | Floating-point coordinates | Integer coordinates | Total |
| --- | ---: | ---: | ---: |
| Fields declared under the bot subsystem | 178 | 111 | 289 |
| Globals declared under the bot subsystem | 167 | 39 | 206 |
| Complete compiled entity field record | 4,982 | 934 | 5,916 |
| Engine-added entity field record | 186 | 43 | 229 |
| Complete runtime entity field record | 5,168 | 977 | 6,145 |

These rows overlap: the bot-declared fields are part of the complete entity schema,
and its globals are part of the server's global storage. They must not be summed as
independent blocks. Bot declarations also supply fields used on waypoint, weapon and
scripting entities; 289 is not an independently allocated per-bot struct.

- [Every runtime entity coordinate](runtime-entity-coordinates.csv): offsets
  0–6144, field/coordinate names, numeric type and original QC type.
- [Every bot-declared field](bot-declared-state.md): readable table, grouped by
  declaring module, with compiled offsets and vector widths.
- [Bot-declared scalar coordinates](bot-declared-coordinates.csv): the same
  subset expanded to exactly one row per scalar.
- [Bot-declared globals](bot-declared-globals.csv): all 182 declarations,
  expanding to 206 scalars, including aim/nav scratch variables, configuration,
  scheduler state and scripting tables.
- [Compiled functions and local-storage extents](bot-functions.csv): 192
  source-matched bot functions, parameter widths and local-storage ranges.
- [All 53,188 compiled global coordinates](program-global-coordinates.csv):
  named variables, aliases and unnamed storage, including compiler temporaries.
- [Engine-added globals](engine-added-global-coordinates.csv): 75 additional
  coordinates, giving 53,263 runtime global words.
- [Summary and limitations](summary.json).

The full entity table deliberately includes the general player/physics/weapon
fields that Havocbot reads and writes, and fields used only by other entity classes
and game modes. It is an exact storage layout, not a minimal dependence slice.
The source lexical field-reference list is explicitly a lower bound; macros,
indirect field selectors and external callees add accesses.

## Bot subsystem breakdown

| Declaration group, with duplicates removed | Floats | Integers | Scalars |
| --- | ---: | ---: | ---: |
| General bot and API | 28 | 7 | 35 |
| Havocbot behavior and roles | 31 | 6 | 37 |
| Aiming/filter history | 30 | 0 | 30 |
| Navigation and route state | 21 | 39 | 60 |
| Waypoint state | 42 | 46 | 88 |
| Scripting and queues | 26 | 13 | 39 |
| Total | 178 | 111 | 289 |

The field table includes the 32 goal-stack references, 32 waypoint references,
32 waypoint costs, eight hardwired waypoint references, five three-component aim
filter states, aim errors, previous desired angles, goal/enemy references, role
function references, timing state, per-bot skill/configuration values, keyboard
state, scripting command state and fixed scripting arrays.

## Integer and float encoding

A `vector` occupies three floating-point coordinates. Entity, string, function and
field values are integer handles/offsets. The complete compiled entity record's
934 integer coordinates consist of 581 entity handles, 195 string handles,
152 function handles and six field offsets.

This source build defines `int` and `bool` as `float` in `lib/_all.inc`; their VM
numeric type is therefore float, even when their values represent integral flags,
counters or categories. The tables report storage types rather than pretending
those declarations compile to native integer fields. Packed 32-bit program words
are promoted according to the engine's configured VM scalar width when loaded.
Scalar-coordinate counts do not depend on that promotion.

String handles do not contain the strings themselves. Likewise, a target entity
handle does not contain the target's state. The referenced entities, strings and
queues are separate storage. Preserving those references is enough to identify a
graph; flattening its contents requires including its referenced records as well.

## Inputs, intermediate values and outputs

Havocbot mutates state in place. For example, `v_angle`, movement, goal stacks,
enemy selection, aim filters, timeouts and button fields can each be both an input
and an output on successive calls. There are no three disjoint allocation blocks
whose lengths can be added to obtain the bot's state dimension.

Its concrete state locations include:

| State | Existing numeric locations |
| --- | --- |
| Player/world inputs | `origin`, `velocity`, bounds, health/armor/ammo/weapon stat fields, team, alive/dead state, water/ground flags, referenced enemy/item/cart/waypoint entities |
| Configuration inputs | Per-bot skill fields and global cvars, including aim/think/dodge/weapon/movement settings, time and frame duration |
| Navigation intermediates | `goalcurrent`, `goalstack01`–`goalstack31`, `goalentity`, previous/ignored goals, waypoint links and costs, navigation scratch globals |
| Aiming intermediates | `bot_mouseaim`, `bot_badaimoffset`, five aim filters, `bot_olddesiredang`, fire/aim clocks, shared trajectory and trace results |
| Behavioral intermediates | `aistatus`, role function handles, enemy selection, commitment/role timeouts, keyboard/random-direction state, command queue state |
| Movement/aim outputs | `CS(bot).movement` (three floats), `bot.v_angle` (three floats) |
| Button outputs | Client-state button fields, including attack, secondary attack, jump, crouch, zoom, hook, use and chat; each is a numeric field |
| Weapon outputs/state | Two weapon-slot entity references and their `m_switchweapon`, `m_weapon`, switching, firing, charge and timing fields |
| Spawn/game-mode state | Respawn clocks and the `plc_*` fields already present in the complete entity table |

`CS(bot)` points to a separate ClientState entity. PlayerState and weapon-slot
entities are also separate records. The two weapon slots are declared in
`common/weapons/weapon.qh`. All these records use the same compiled field layout;
their applicable coordinates differ by role.

The compiler enables overlapping locals. The 192 source-matched bot functions use
an overlapping 37-word declared-local region at global offsets 52075–52111; this
includes function parameters. This is not the total intermediate state: unnamed
compiler temporaries elsewhere in the global arena, external callee locals and
saved caller frames also exist. Adding each function's local count would duplicate
the same reused storage. The engine saves overwritten local words on its VM local
stack and keeps call frames separately.

The native keyboard batch has its own explicit 18-float row:

```
movement.xyz, origin.xyz, destination.xyz,
keyboard_time, move_skill, keyboard_skill, duck_time, random,
keyboard.xyz, crouch
```

Fourteen columns supply row inputs. Its writes overlap those columns and also clear
the actor's pending flag. The five arithmetic parameters are time, global skill,
maximum speed, trigger and distance; the wrapper additionally uses frame duration
for scheduling. Its row-packing indices, pointers, coordinates and wave identifiers
are integers. This is one execution kernel inside Havocbot, not the whole bot.

## What length can be stated honestly

For the fully enumerated runtime entity/global storage, the length is exactly

```
N_entity_and_global(E) = 6,145 * E + 53,263
```

where E counts the entity records included. This includes the compiled global
temporary arena. Engine allocation has two extra global guard words, which are
storage padding rather than named state coordinates.

A complete closed snapshot additionally includes saved VM locals/call frames,
dynamic strings and command-buffer contents, native engine/client/physics state,
and the map/world data supplied to engine builtins. Their populations depend on
the map, entity graph, active calls and allocated buffers. The stopped workload
did not retain such a complete memory snapshot, so this audit does not invent one
constant total for it. The 6,145-coordinate entity record is exact; calling that
the entire transitive state of Havocbot would be false.

The eight-coordinate mesh adapter is a chosen projection into this much larger
numeric state surface. It is not evidence that those eight values exhaust the
playerbot's inputs, outputs or controllable internal state.

## Reproduce

```
python3 xonotic/payload/tools/havocstate.py \
  .build/mini-game-20260905/mini-learner-20260905/userdir/data/progs.dat \
  --source xonotic/qcsrc --engine xonotic/darkplaces-work \
  --out measurements/havocbot-state-20260905
```

The generator checks that the scalar entity coordinates exactly cover every
compiled offset once, without counting vector-component aliases twice. The same
audit ran on the Mini against its deployed program and source. Engine-added offsets
are derived from `prvm_offsets.h` and the append logic in `PRVM_Prog_Load`; no game
server was launched to obtain a new live memory dump.
