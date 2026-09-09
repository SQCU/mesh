# Input/output correspondence, 2026-09-06

Follow-up: [static flow refactor](../policy-static-flow-20260906/README.md) consolidates ingress, emission and checkpoint interpretation, shares packed inputs and persists partial transport state. This record describes the preceding review.

This inventory follows production producers, tensor assembly, heads, packet
packing and native application. It does not use tests as a specification. The
native state is a growing set of witnessed VM reads, not a simultaneous dump of
every engine variable. World context, controllable views and predicted values
have different roles; they do not require identical coordinate counts.

## Source families and destinations

Let O be the number of native client slots, I the number of state owners, P the
prepared page capacity per owner, J carts, K teams, E retained events, and V/A/C
navigation nodes/edges/cells. W=256. Each physical family has its own prepared
capacity and presence mask. Actual game players are the present subset of O;
vacant native slots and extra prepared slots are both structural absence.

At their physical extents, the first learned projections receive
`1556 I P + 85 O + 18 J + 7 K + 18 E + 4 V + 3 A + 2 C` scalar positions,
including prepared capacity when present. Unpresent words/rows
are structural holes. This count deliberately distinguishes raw metadata from
the `256 I P` possible native word actuators; it does not invent commands for
cart scores, map edges or prediction targets.

| Native or realized source | Literal model input | Role and output correspondence |
|---|---|---|
| OBSERVATION[O,83] | All 83 columns, plus requested duration and relaxation time, first project as [O,85]. | Scene and participant context. Health/ammo/position/velocity/weapon words, spawn metadata, counters and response diagnostics are observations, not an invented action vocabulary. A state owner has a structural OBS-row reference; its OBS hidden row contributes to all that owner's page outputs. Human/world observations can provide context without owning a rate output. |
| CART[J,18] | All 18 columns project separately. | Objective scene state, including native path position/length/control and checkpoints. There is no policy command to directly write a cart's position or ownership. Cart motion follows game rules and player actions. |
| TEAM[K,7] | All seven columns project separately. | Team identity/score/checkpoints/rate/limit/episode/finished context and structural membership. No direct policy output writes team score. |
| STATE native pages | Six W-word arrays plus 15 headers; the learned page row is 6W+20=1,556 columns. | Every present captured word gets corresponding Gaussian mean/log-scale coordinates and a sampled residual forcing rate. Address/type/presence/time data identify the same read-view coordinate on return. |
| EVENT[E,18] | All 18 fields project before spatial or age contraction. | Per-episode observation memory and live unowned topology. Team zero is public/unowned context; positive IDs share within the owning team. Spatial support still applies. Events are not a list of policy action candidates. |
| Navigation nodes[V,4], edges[A,3], cells[C,2] | Literal XYZ+node measure, endpoint indices+length, and site index+cell measure each project separately. | Offline map context, mixed locally into observation rows; no policy output edits the navigation graph. |
| OUTCOME[*,5] | Not a policy source family. | Durable session/episode/winner/time records close training targets and reporting. A future outcome is a target, not a missing policy input. Current episode/finished/score already arrive through TEAM. |
| Structural arrays | Owner row indices, team labels, presence masks, neighborhood indices/distances/radii. | Preserve row identity and delimit physical/local support. They need no corresponding actuator. The former serialized `relations` array was dead after neighborhood preparation and is deleted. |

Producers are [strategy I/O QC](../../xonotic/qcsrc/common/gamemodes/gamemode/payload/sv_payload_strategy_io.qc),
[native page publication](../../xonotic/darkplaces-work/mesh_ipc.c),
[wire column schema](../../xonotic/payload/tools/strategy_io_schema.py), and
[navigation realization](../../xonotic/payload/tools/navmesh.py).
The complete assembly is [inputs.py](../../xonotic/solver/strat/inputs.py),
[policy_inputs](../../xonotic/solver/strat/state_steering.py), and
[embedded_rows](../../xonotic/solver/strat/strategy.py).

## Exact page and actuator contract

STATE is now 1,551 float32 columns: 15 metadata columns and six 256-word arrays.
The metadata order is owner, schema, export time, applied sequence, entity arena,
page offset, entity generation, owner generation, session high/low, command
source time, command duration, command relaxation time, received sequence and
page forcing flag. The six arrays are last-read source, current integrated
residual, packed type/bit metadata, last-read time, held command velocity and
command-anchor residual.

Forcing has a real native anchor. At time t, let a=max(t−command_source_time,0),
h=min(a,command_duration). Native application is

`r(t) = [exp(−h/tau) r_anchor + tau (1−exp(−h/tau)) velocity] exp(−(a−h)/tau)`.

Both the raw command state and the current residual now enter the first learned
projection. They are deliberately distinct: current residual describes the view
now, while the held command and its timing determine how the view continues
before another response arrives. Requested duration/tau describe the new command,
not the preceding accepted command. Received and applied sequence also remain
distinct.

STRATEGY remains 524 columns: 12 metadata columns, the source snapshot's current
residual and the selected sampled velocity. Metadata identifies owner/schema,
source export time, new duration/tau, entity/address/generations/session and source
applied sequence. The engine validates the complete response before updating
pages. It integrates from the source snapshot's residual using actual engine
age; the model's nominal end-of-interval residual is not sent back as though it
were an instantaneous engine observation.

Native reads add the residual to a temporary view. Floating words add directly;
reference/integer words add the rounded integer residual with native word
arithmetic. VM writes remain ordinary consequences of bot execution. No policy
response directly overwrites authoritative game state. Entity/global source
values are captured immediately before the read residual is applied. Packed
nonfloating words retain low bits in the source array and high bits in the tag
array; the model receives both, rather than a lossy float conversion of an entire
integer word. This packing describes the current 32-bit VM build; the optional
PRVM_64 build has a separate wider-state coverage question.

Source: [PRVM view structures](../../xonotic/darkplaces-work/prvm_view.h),
[PRVM read/application implementation](../../xonotic/darkplaces-work/prvm_view.c),
[response validation/publication](../../xonotic/darkplaces-work/mesh_ipc.c), and
[StatePages unpack/response](../../xonotic/solver/strat/state_steering.py).

## Common hidden state and heads

The main model embeds each family, applies the local event/navigation operator,
row-local input FFN, learned global/team row Gram, row-local routed MoE, and final
FFN. The resulting H is the representation used by the policy and both value
outputs. `read_heads` applies one learned projection of H to 2W+2 coordinates.
The first 2W coordinates become means/log-scales for each owner page, with that
owner's OBS-row contribution; the last two coordinates sum across that owner's
OBS/page rows to produce winner-role and loser-role values. The rate
distribution then applies only numeric scale parameterization and structural
output presence. The former second state/residual projections in StateReadout
and earlier-layer auxiliary value heads are gone.

Coordinate coverage does not prove full instantaneous controllability. With the
current width128 H and fixed learned head weights, each page's 512 mean/log-scale
coordinates vary through a locally rank-at-most128 representation. The owner
and page contributions use the same projection, so their addition does not
double this bound. Head parameters can themselves learn different output
directions, and bot dynamics integrate effects over time; neither observation
is a proof of arbitrary Havocbot behavior. This is an explicit learned-rank
choice, distinct from a hand-written action vocabulary.

The learned rate outputs all have native input coordinates. The value scalars
are predictions of returns and correctly have no actuator or previous-prediction
input requirement. Coupling factors, local neighborhood messages, routed load and
residual diagnostics are internal/reporting outputs, not commands. The MoE
balance scalar contributes to optimization; its reporting reductions do not
replace raw policy inputs. [strategy.py](../../xonotic/solver/strat/strategy.py),
[cast_header.py](../../xonotic/solver/strat/cast_header.py),
[state_steering.py](../../xonotic/solver/strat/state_steering.py), and
[online.py](../../xonotic/solver/strat/online.py) define these roles.

Every instantiated policy is evaluated for all owners. The responder selects
the assigned policy's rate for each owner's team, records that behavior
distribution's likelihood and sends the selected velocity. Other policies'
outputs are counterfactual predictions. Critics can learn from observed returns
on those rows; actor updates use the source policy's first observed execution
assignment. Nothing here chooses a target from an adapter-generated discrete
list. See [responder](../../xonotic/solver/strat/strat_responder.py) and
[ActionHistory](../../xonotic/solver/strat/action_history.py).

## Defects corrected in this change

- Last-read source words previously all inherited export time without individual
  observation timestamps. Capture now records each word's actual last-read time
  on all three capture paths. Values retain their witnessed-read semantics.
- Current residual alone omitted held forcing, command anchor and actual timing.
  These literal native fields now cross the same packet/embedding boundary.
- Native OBS publishes maxclients rows, but only actual players populate scene
  fields. Formerly every row was considered present. Native PRESENT at column29
  now supplies authoritative occupancy for global mixing and local destinations.
- `relations` survived in transport/replay despite not entering the learned
  program; the neighborhood relation is now the single structural source.
- The sole native caller now supplies all source arrays, page metadata, owner
  row mapping, occupancy and observation relations explicitly. The packer no
  longer fabricates layout/tags, headers, cart/team rows or owner identities when
  omitted, and it no longer recomputes the responder's visibility relation.
  Structural all-present masks and the explicit no-navigation event neighborhood
  remain available.
- Public live CELL_LINK events had team0 and were masked out of every real
  positive-team observation. Their explicit unowned/public interpretation now
  gives them spatially supported access without changing their raw columns.
- Rate and value heads previously bypassed different parts of the representation.
  All heads now use the common final H, with one output projection.
- EVENT previously lacked episode ownership. The producer now appends the literal
  deposit-time episode. Transport retains batches by actual session/tick/request;
  ObservationMemory partitions them by session plus deposited episode, retains
  earlier/future histories, and supplies only the selected episode to policy
  context. Journal replay consumes exact recorded batch keys. Legacy 17-column
  rows remain explicitly unattributed rather than receiving a fabricated episode.

## Remaining boundaries and ambiguities

1. **Captured source is incomplete engine state.** Unread VM words have no source
   value and no rate coordinate yet. Write-only words are not discovered by the
   read capture. A global temporary reused several times retains its last source
   value/type/time, not its complete execution trace. QC opcode operands and
   typed entity loads are mediated, as are the explicit bot-batch source reads;
   this does not make all C-side engine/private buffers part of the bot view.
   String/function/entity handles carry their numeric words, not an embedding of
   every object they might refer to. See [view interpreter](../../xonotic/darkplaces-work/prvm_execprogram.h),
   [bot batch](../../xonotic/darkplaces-work/bot_batch.c), and
   [VM field accessors](../../xonotic/darkplaces-work/progsvm.h).
2. **Application acknowledgement is an attempt, not an atomic success proof.**
   A view fault unwinds and retries stock bot execution, then ViewEnd acknowledges
   the policy sequence. Prior side effects are not rolled back. OBS supplies
   fault count/sequence, but ActionHistory's actor eligibility uses control and
   acknowledgement. Automatically excluding the attempt would also erase real
   causal effects; successful execution, partial execution and stock retry need
   explicit accounting rather than an inferred field restriction.
3. **Legacy EVENT history cannot retroactively acquire omitted identity.** The
   new EVENT18 and keyed batch path resolve ownership for new observations;
   retained EVENT17 rows have no literal episode column and stay explicitly
   unattributed. Source preservation does not imply reconstructed historical
   completeness. Sources:
   [RuntimeFrames](../../xonotic/solver/xonwire.py),
   [ObservationMemory](../../xonotic/solver/strat/buffers.py), and
   [payload reset](../../xonotic/qcsrc/common/gamemodes/gamemode/payload/sv_payload.qc).
4. **Team-local EVENT visibility is not whole-policy information isolation.**
   All present opponents' OBS rows and captured page rows enter the global Gram.
   Thus the model has access to more than team-owned local observations. This is
   the implemented source contract, not proof that a partially observed game
   specification is satisfied.
5. **Live and offline graph sources differ.** Offline navigation derives an
   undirected metric graph from waypoint-cache realization; CELL_LINK traverses
   live waypoints, including possible relinking/hardwired links, and emits coarse
   2-D cell endpoints plus Euclidean length. Repeated static facts increase event
   mass and can change learned evidence magnitude. They are not proven duplicate
   tensors, and no input rows were silently deleted to make them look equivalent.
6. **Metadata duplication remains explicit.** Page entity/offset/generation
   appear in layout and headers; IDs are both literal learned columns and
   structural joins; command timing is shared per owner but repeated per page;
   requested timing also accompanies OBS rows. This is redundant representation,
   not hidden reduction. Literal numeric IDs do not guarantee invariance under
   consistent renaming. Reserved OBS30–38/43–46 and CART rollback fields remain
   supplied zeros because their producers currently write zeros.

Local production compilation succeeded for the dedicated C engine and all three
QC programs. Logs are [native-build.log](build/native-build.log) and
[qc-build.log](build/qc-build.log). Changed Python source parsed successfully.
No tests, replacement test harness, runtime start, deployment, remote workload or
bridge operation was used for this inventory/change. Compilation establishes
source compatibility, not runtime learning quality or full engine mediation.
