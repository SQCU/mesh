# NUM_TEAMS = 5 for Xonotic (team 5 = Green)

Deliverables:

- `/Users/mdot/dox/mesh/xonotic/teams-k5.patch` — unified diff, `-p1` against a copy of
  `/Applications/Xonotic/source/qcsrc`. 27 files, 688 lines, +136 / -66.
- `/Users/mdot/dox/mesh/xonotic/teams-k5-assets/effectinfo.txt` — the full stock
  `effectinfo.txt` with three real `spawn_point_green` / `spawn_event_green` particle
  blocks appended (recoloured from the pink ones to the palette green `0x0FFF0F`). Drop
  into `<userdir>/data/`; a loose file overrides the copy inside the pk3.
- `/Users/mdot/dox/mesh/xonotic/teamsk5/` — working tree: `qcsrc-pristine/`, `qcsrc/`
  (patched), `base/` (clean-room baseline build), `verify/` (clean-room patch-apply
  build), `run/` (dedicated-server sandbox). `/Applications/Xonotic` was not modified.

## Measured results

Compiler: `gmqcc` 0.3.6 at `/Users/mdot/dox/mesh/xonotic/gmqcc-work/gmqcc`, with the
Makefile's shipped flags: `-std=gmqcc -Ooverlap-locals -O3 -Werror -Wall
-Wno-field-redeclared -flno -futf8 -fno-bail-on-werror -frelaxed-switch
-freturn-assignments`.

Both builds below are clean-room: a fresh `cp -R` of the pristine tree into its own
directory, same watermark string (`teams-k5-verify`, which affects size by a few bytes),
own `WORKDIR`.

| | baseline `qcsrc` | + teams-k5.patch | delta |
|---|---|---|---|
| `progs.dat`   | 6,663,685 | 6,728,086 | +64,401 |
| `csprogs.dat` | 4,021,230 | 4,073,796 | +52,566 |
| `menu.dat`    | 1,756,665 | 1,757,221 | +556 |

**Zero errors and zero warnings** on all three programs with `-Werror -Wall` in force.
`patch -p1` applied to the fresh pristine copy with no fuzz and no rejects. That
apply-then-build is the reproducible evidence, not the in-place tree.

(The `crc:` gmqcc prints is the Quake progs *header* crc over the system field
definitions — it is 0x1727 / 0xCBE3 / 0x2724 before and after, by construction. It is not
a content hash and proves nothing here.)

## Runtime evidence

`darkplaces-dedicated` (arm64, built from the shipped engine source by an earlier agent,
at `/Users/mdot/dox/mesh/xonotic/darkplaces-work/`) was run against the patched `.dat`s
placed in a private `-userdir`, bounded by `perl -e 'alarm N; exec @ARGV'` (macOS has no
`timeout`). Nothing was `kill -9`'d; the server exited on `SIGALRM` or on `quit` fed to
its stdin.

```
darkplaces-dedicated -xonotic -basedir /Applications/Xonotic \
  -userdir .../teamsk5/run +developer 1 +sv_public 0 +port 26005 +g_tdm 1 +map boil
```

Observed in the log:

- `server detected csqc progs file "csprogs.dat" with size 4073788` — the patched
  csprogs is what loaded, not the pk3's.
- `PRVM_LoadProgs: no cvar for autocvar global autocvar_g_forced_team_green in server,
  creating...`, likewise `autocvar_sv_defaultplayermodel_green` and
  `autocvar_sv_defaultplayerskin_green` — the new autocvars are live.
- `:gamestart:tdm_boil:` → `tdm_DelayedInit ... No "tdm_team" entities found on this map,
  creating them anyway.` with `g_tdm_teams_override 5` → `Server spawned.` →
  `========Initialized=========`, with no `LOG_FATAL` anywhere.

That clears the two runtime traps the survey flagged, neither of which the compiler can
see: `Team_GetTeamFromIndex(5)` does not hit the `LOG_FATALF` at `server/teamplay.qc:59`,
and the notification registry initialises without tripping `NOTIF_CHOICE_MAX = 20`
(13 single + 5 multiteam choice groups = 18, unchanged, because of the
`nent_choice_count` fix below).

**No bots ever joined**, so no player was actually placed on team 5. I checked whether
that was my doing: with the loose `.dat`s moved aside so the stock pk3 `progs.dat` runs,
the same command line and the same `bot_number 10` also reports `Finished listing 0
client(s) out of 16 slots`. Bot spawning is broken in this headless sandbox independently
of the patch. I did not chase it further.

## The invariant the patch is built on

Team identity is a Quake colormap **pants nibble**, not a QC abstraction. `setcolor`
(`server/teamplay.qc:158`) does `this.team = (clr & 15) + 1`; CSQC's `entcs_GetTeam()`
returns `clientcolors & 15`. So `NUM_TEAM_n(SVQC) == NUM_TEAM_n(CSQC) + 1` must hold
exactly for every n, and the server id must stay inside `[1,15]`
(`server/scores.qc:117`: `if(t <= 0 || t >= 16) return;`).

Team 5 = **Green**, palette pants index 3, which is `15,255,15` in Xonotic's own
`gfx/colormap_palette.lmp`. Hence **CSQC `NUM_TEAM_5 = 3`, SVQC `NUM_TEAM_5 = 4`.**
Colour code `^2`; `Team_ColorRGB` returns `'0.0625 1 0.0625'` (0x0FFF0F), formed the same
way as the four existing entries; `STATIC_NAME_TEAM_5 = "Green"`; translatable
`TEAM^Green`, `KEY^Green`, `FLAG^Green`, `GENERATOR^Green`. Real palette values, not
placeholders.

`TEAMNUMBERS_THAT_ARENT_STUPID` is left alone and still off — enabling it destroys the
+1 offset above.

## What the patch changes

**Team core**

- `common/teams.qh` — `NUM_TEAMS = 5`; `NUM_TEAM_5` for both VMs (and in the disabled
  `TEAMNUMBERS_THAT_ARENT_STUPID` branch, where `NUM_SPECTATOR` moves 5 → 6);
  `COL_TEAM_5`, `NAME_TEAM_5`, `KEY_TEAM_5`, `FLAG_TEAM_5`, `GENERATOR_TEAM_5`,
  `STATIC_NAME_TEAM_5`; all nine switches extended (`Team_ColorCode`, `Team_ColorRGB`,
  `Team_ColorName`, `Static_Team_ColorName`, `Team_ColorToTeam` plus `case "green"`,
  `Team_IsValidTeam`, `Team_IsValidIndex`, `Team_IndexToTeam`, `Team_TeamToIndex`).
- `server/scores_rules.qc` — `NumTeams()` was four hardcoded `BIT(n)` tests. Rewritten as
  a loop to `NUM_TEAMS`, so it cannot go off by one again. Getting this wrong is silent:
  it feeds `AVAILABLE_TEAMS`, bot fill and the "all teams eliminated" checks.
- `server/scores.qc` — `TeamScore_Spawn(NUM_TEAM_5, "Green")` under `BIT(4)`.
- `server/spawnpoints.qc` — `spawnfunc(info_player_team5)`.
- `server/world.qc` — the four `teamN_score` locals and the 4-deep if/else that picked the
  single scoring team are replaced by one loop over `Team_GetTeamFromIndex(1..NUM_TEAMS)`.
- `client/main.qc` — `SetTeam()` accepts `NUM_TEAM_5`; the spawn-effect switch extended.

**Notifications** (the largest hardcoded-4 surface)

- `common/notifications/all.inc` — a `_GREEN` line added to each of `MULTITEAM_ANNCE`,
  `MULTITEAM_INFO`, `MULTITEAM_CENTER`, `MULTITEAM_MULTI`, `MULTITEAM_CHOICE`.
- `common/notifications/all.qh` —
  - `APP_TEAM_NUM` gained an explicit `NUM_TEAM_4` test so `_GREEN`, not `_PINK`, is the
    fallthrough. Without this every team-5 notification is announced as Pink.
  - `notif_arg_missing_teams` gained `BIT(4)`, and the `BIT(3)` arm gained the `", "`
    separator it did not need when it was last.
  - `MSG_CHOICE_NOTIF_`'s `if (!teamnum || teamnum == NUM_TEAM_4)` became `== NUM_TEAM_5`.
    This one is a trap: it counts a multiteam choice group once by testing "am I the last
    team". Extending the macro without fixing it takes the count 18 → 23 and trips
    `NOTIF_CHOICE_MAX = 20` as a startup `LOG_FATAL` — not a compile error.

**Admin and presentation**

- `server/teamplay.qc`, `server/player.qh`, `server/client.qc` — `g_forced_team_green`,
  the `g_forced_team_otherwise "green"` case, `sv_defaultplayermodel_green` and
  `sv_defaultplayerskin_green` (both switches).
- `server/command/cmd.qc` — `selectteam` usage text only; the command already routes
  through `Team_ColorToTeam`, so `cmd selectteam green` works from the teams.qh change.
- `menu/xonotic/dialog_teamselect.{qh,qc}` — a fifth `green` button, `columns` 4 → 5, the
  auto-select and spectate spans widened to 5, `teams & 16` wired to it. `_teams_available`
  already carried `Team_IndexToBit()`, so bit 4 was already arriving; the menu just
  ignored it.
- `common/effects/all.inc` — `EFFECT_SPAWNPOINT_GREEN` and `EFFECT_SPAWN_GREEN`, backed by
  the real particle definitions in the shipped `effectinfo.txt`.
- `common/weapons/weapon/hook.qc` — the green hook uses the **existing**
  `particles/hook_white` texture tinted `'0.3 1 0.3'`. There is no `hook_green` in the
  data pk3, and naming one would draw the engine's missing-texture checkerboard.
- `common/mutators/mutator/nades/nades.qc`, `.../spawn_near_teammate/`,
  `.../buffs/buffs.qh` (which adds the `*_team5` buff spawnfuncs).
- `common/stats.qh` — `GREENALIVE`, `DOM_PPS_GREEN`.

**Gamemodes**

- `bound(2, x, 4)` → `bound(2, x, NUM_TEAMS)` in tdm, tmayhem, freezetag, clanarena, race.
- tdm and tmayhem gained their `BIT(4)` → `SpawnTeam("Green", NUM_TEAM_5)` arm.
- freezetag and clanarena publish `GREENALIVE`; `cl_clanarena.qc` draws slot 4.
- `TeamBalance_*`, the scoreboard, the score panel, `Team_TeamToBit`,
  `have_team_spawns_forteams` and `teamscorekeepers[16]` already generalise over
  `NUM_TEAMS` and are untouched. `.entity m_team_balance_team[NUM_TEAMS]` and
  `entity g_team_entities[NUM_TEAMS]` resize themselves.

## What I could not resolve

- **CTF, keyhunt, domination, tka and nexball are still capped at 4 teams.** Not
  oversights — the two structurally hard cases and their neighbours:
  - `common/gamemodes/gamemode/ctf/ctf.qh:42-59` packs 2 bits per team into
    `STAT(OBJECTIVE_STATUS)` as literals 1/2/3, 4/8/12, 16/32/48, 64/128/192 — and
    **neutral sits at 256**, exactly where team 5's pair would land. A fifth team needs the
    whole field re-laid out, not a line added. The stat is `int`, so there is width; it is
    the constants that collide.
  - keyhunt has `vector KH_SLOTS[4]`, a `for(i = 0; i < 4; ++i)`, four literal slot
    positions and four unrolled draw loops with literal texture names in `cl_keyhunt.qc`.
  - domination needs a `WP_DomGreen` waypoint sprite and
    `models/domination/dom_green.md3` to spawn a fifth control-point team honestly. I added
    the `DOM_PPS_GREEN` stat but deliberately did **not** raise its `bound(2, x, 4)` —
    doing so would spawn a team whose control points have no model. The payload mode
    sketched in `xonotic/payload-mode-plan.md` writes its own entities and does not
    inherit this.
- **Two HUD assets are missing** and will draw as the engine's missing-texture pattern if
  those panels are used at k = 5: `gfx/hud/{default,luma,luminos,old}/player_green.jpg`
  (plus `_alpha`) for the clan-arena / freezetag alive-count modicon, and
  `dom_icon_green.jpg` if domination is ever raised. I could not author them — no PIL and
  no ImageMagick on this machine, and the source art is JPEG inside the pk3. I chose to
  reference `player_green` and report the gap rather than silently label a pink icon
  "green".
- **No `.po` catalogue entries** for the four new `_()` strings. They fall back to msgid,
  which is correct English and untranslated everywhere else.
- **No `data/*.cfg` overlay.** `g_forced_team_matchsetup` in `xonotic-server.cfg:584` takes
  exactly four team arguments, and the `g_*_teams` cvar descriptions in
  `gamemodes-server.cfg` still say "max 4". The code accepts 5; the shipped configs still
  describe 4.
- **No bot-filled match, no second client, no visual check.** The colours, the fifth menu
  button and the notification text are unverified by eye. Everything asserted above is
  either compiler-verified or engine-load-verified; nothing has been *played*.
- **`tools/compilationunits.sh` does not run here** — it needs a git repo, and it also
  deletes `PROGS_OUT/*.dat` as a side effect. The Makefile's `git describe` and
  `git hash-object` need one too, so pass `QCCFLAGS_WATERMARK=<anything>` explicitly.
  Note also that `PROGS_OUT` and `WORKDIR` default to `qcsrc/..` and `qcsrc/../.tmp`, so
  two trees sharing a parent share a build cache and silently do not rebuild — the
  `base/` and `verify/` directories exist to avoid exactly that.
- **No RDMA of any kind.** No verbs device opened, no `mesh-flow` or `ibv_*` run, nothing
  linked against `-lrdma`. Nothing was `kill -9`'d.

## Reproducing

    cp -R /Applications/Xonotic/source/qcsrc /tmp/k5/qcsrc
    cd /tmp/k5 && patch -p1 -i /Users/mdot/dox/mesh/xonotic/teams-k5.patch
    cd /tmp/k5/qcsrc && make QCC=/Users/mdot/dox/mesh/xonotic/gmqcc-work/gmqcc \
        QCCFLAGS_WATERMARK=teams-k5 qc
