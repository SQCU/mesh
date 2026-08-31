# The j-oracle viewer — a continuous read of the live policy

`design/jspace-probe.md` is a one-shot measurement: 228 archived lines, run once,
verdict written down. This document specifies the **continuous** version of that
measurement, wired to a real Xonotic match while it is being played, next to the
behavior the same policy is producing.

Everything here lives in `xonotic/solver/strat/joracle/` (the tap and the server)
and `xonotic/solver/strat/web/` (the page). Nothing in those directories writes
to the game, the responder, the mesh, or a checkpoint. The tap is a `tail -F` on
the responder's own telemetry JSONL and nothing else.

The design constraint that shapes the whole page comes from AGENDA R19: a run can
look completely healthy from the outside — bots moving, carts advancing, losses
descending — while its model input is rank 4, its per-player state is identically
zero, and its trained encoder is indistinguishable from a random-initialised one.
None of that is visible in behavior. So the page shows the internals **and their
controls**, and it shows absent fields as absent.

---

## 1. Bring-up — one command

```
xonotic/solver/strat/joracle/demo.sh up
```

That brings up all three pieces:

| piece | where | what |
|---|---|---|
| Xonotic dedicated cartserver | this Mac, `udp/26042` | `darkplaces-dedicated`, payload gamemode, 12 playerbots, map `fused`, strategy I/O over the mesh |
| mlx strategy responder | `mesh-mini`, `~/.venv-mesh` | `solver.strat.strat_responder --train`, one strategy tick per server request |
| j-oracle viewer | this Mac, `http://127.0.0.1:8795` | this package |

and prints:

```
  j-oracle viewer     http://127.0.0.1:8795
  xonotic client      launch Xonotic, press ~ for the console, then:
                          connect 127.0.0.1:26042
                      (from another machine on the LAN: connect <en0 addr>:26042)
```

**Connecting the on-device client.** Start the normal Xonotic app, open the
console with `~`, and type `connect 127.0.0.1:26042`. The server runs
`sv_public 0`, so it does not appear in the public browser; a direct `connect` is
the way in. The human player is featurized exactly like the bots — the responder
tags the row `controller: "human"` and still computes an assignment for it, which
the viewer shows in the assignments table as `HUMAN`. Your own row is therefore
visible in the same table as the bots you are playing against.

Ports: `26042` for the game, `8795` for the viewer, both overridable
(`JORACLE_PORT`, `JORACLE_VIEWER_PORT`). `26012` is refused by the script
outright, and any port already bound is refused rather than fought over.

### Subcommands

```
demo.sh up       # server + responder + viewer
demo.sh viewer   # viewer only — reattaches to a run that is already going
demo.sh status   # bridge client, server/viewer pids, responder, telemetry length
demo.sh down     # stops only the server and viewer this script started
```

`demo.sh viewer` with `JORACLE_TELEMETRY=host:/path/live.jsonl` points the viewer
at any responder's telemetry, including one somebody else launched.

### Interruptible and resumable

The run directory is a fixed path (`/tmp/mesh-joracle`), not a `mktemp -d`. That
is the whole resumability story:

* the **viewer** follows its source through a supervised `tail -F` subprocess. If
  the file is truncated, deleted and recreated, or the whole match is restarted
  underneath it, the follower's subprocess dies and is respawned; the page keeps
  rendering the frames it already has, marked stale, and picks up again by
  itself. Restarts are counted (`epochs`, `resp_id_resets`) and drawn on the
  depth chart as dashed red verticals so a restart is never mistaken for a jump
  in the game;
* the **responder** is launched with `--append-telemetry`, so it restores its RNG
  state, `resp_id` and telemetry cursor from `live.npz.runstate.json` and appends
  to the same JSONL rather than truncating it;
* the **server** can be killed and re-`up`'d against the same run directory.

### Preflight refusals

`demo.sh up` stops with `BLOCKED:` and a non-zero exit rather than doing damage
when: the port is 26012 or in use; the engine, basedir or payload `.dat`s are
missing; the local mesh bridge is down; `mesh-mini` is unreachable; or — the
important one — **another live process already holds the mesh bridge's single
client slot**. `rdma/mesh-flow.c` keeps one `M->client`; a second attaching game
server would take the mesh away from a run in progress. The script reports the
holding pid and its command line and exits instead.

---

## 2. What the tap publishes

`joracle/follow.py` parses each telemetry line; `joracle/server.py` serves it.

| endpoint | content |
|---|---|
| `/api/live` | behavior series (last 600 ticks), latest policy internals, field audit, follower status |
| `/api/joracle` | the rolling probe report |
| `/api/frame` | the newest raw telemetry line, verbatim, for anything the page does not render |
| `/api/health` | liveness of the tap itself |

The responder samples the large arrays every `--model-sample-every` ticks. The
demo sets that to 1 so the probe sees every tick; when it is larger, the viewer
reads the newest frame that actually carries the arrays and labels it "N ticks
back". It never carries the previous tick's numbers forward as if they were this
tick's.

---

## 3. The page

`xonotic/solver/strat/web/index.html` — plain files, no build step, no CDN, no
dependencies at all. Open it through the viewer's own port; it is not a static
page (it polls `/api/*`: 1 Hz for behavior, 3 s for the probe).

**Behavior.** Cart depth per cart over time, with a control strip under the axis
coloured by the controlling team; the PW timeline as a coloured band with white
flip markers and a flip count; per-team hierarchy (alive, health, armor, ammo,
weapon slots, carts controlled, SUCC denial budget, PW); the cross-team focus
matrix — row *i* column *j* is how many of team *i*'s players were assigned an
instrument aimed at team *j* this tick, i.e. who is hunting or suppressing whom;
and the per-player assignment table (edict, team, bot/human, policy/uniform,
instrument kind, subject, packed target, gain, lane, commit, spawn, log π).

**Internals.** The final IR as a heatmap (rows = players, columns = IR dims), its
singular spectrum, the Gram matrix, the W and L value-head outputs with W−L, the
advantage, `diag(K)` per assignment, and the online-training losses.

**The j-oracle.** See §4.

**What is missing.** A table over every field the viewer wants, each marked
`present` / `all_zero` / `shape_only` / `absent`, with the producer that would
have to emit it. Next to it, the engine-input block audit: the 14 named column
blocks of `x` from `estimator.state_from_runtime`, each showing how many of its
columns carry any signal. A zeroed `health` / `ammo` / `weapon bitset` block is
the AGENDA E9 condition and is drawn in red.

---

## 4. The measurement, and why the controls are the point

Methodology is deliberately identical to `runs/jspace_probe.py` so the live
numbers and the archived ones are comparable:

* ridge least squares, λ = 1e−3, features standardised on the train split with a
  bias column appended;
* one-vs-rest ridge for categorical targets; accuracy is reported against the
  test-split majority baseline, never alone;
* **split by tick**, 60/40, fixed seed — no test row shares a server tick with a
  training row;
* rolling window of the last 4000 player-rows, recomputed every 4 s;
* the window must hold at least **16 player-rows per IR dimension** (2048 rows at
  the 128-wide IR) before any score is published. Below that a 128-feature ridge
  interpolates the train split and explodes on the test split — which shows up as
  a hugely negative shuffled-label R², i.e. as the honesty control failing. The
  page says `accumulating: N/2048 player-rows` until the window is wide enough,
  rather than publishing a degenerate fit. (`jspace_probe.py` had 3150 rows
  against 16–40 features; this is the same discipline at the live width.)

Four columns are shown for every target, and only the relationship between them
means anything:

| column | what it is | what it tells you |
|---|---|---|
| **IR** | ridge on the live final IR | the claim |
| **ctrl: rand-proj** | ridge on a fixed random Gaussian projection of the raw input `x`, at the same width as the IR | if the IR does not beat this, the IR added nothing to what the input already handed it |
| **ctrl: shuffled** | ridge on the IR against permuted labels | if this is not ≈0 (or ≈majority), the probe is fitting noise and **every** other number on the page is void |
| **raw x** | ridge on `x` itself | marks tautological targets — ones that *are* input columns |

**The control is applied per target, not once for the page.** Tick-level targets
(`total_cart_depth`, `pw_team`) take the same value for every player-row in a
tick, so their effective sample size is the tick count and they go degenerate
long before the per-player targets do. Each row therefore carries its own
`control_ok`: a target whose shuffled-label control did not land at chance is
struck through and excluded from the verdict, and the alarm names exactly which
targets failed. That is a stronger statement than one global honesty flag.

Targets that are literally input columns (health, armor, ammo, own nimber, …) are
tagged `taut` and greyed. They are kept because `jspace-probe.md` reports them and
because their *absence* is informative — a health probe that fails means health is
not in `x` at all — but they can never be evidence of a j-space. The verdict line
counts only non-tautological targets, with a Δ > 0.05 threshold over the
random-projection control.

**The one control this cannot run.** `jspace-probe.md`'s decisive comparison was
trained IR vs a *random-initialised encoder's* IR, which agreed to three decimals.
That needs a second forward pass through the model. This process never touches
the model, so it is stated as not-measured on the page rather than quietly
omitted. The random-projection control is the one that runs live, and it is the
one R19 reports the trained IR failing.

### The pathology panel

Top of the page, before anything else, because R19's failure mode was invisible
from behavior:

| alarm | fires when | the R19 fact behind it |
|---|---|---|
| `rank-collapsed-input` | rank of `x` over the window ≤ 5 | *"the rank of the real input matrix `[x;beta]` over all 3150 rows = **4**"* |
| `per-player-state-zeroed` | ≤ 8 of `x`'s columns carry any signal | *"health, armor, ammo, weapon bitmask … were all zero in the model's own input on this run"* |
| `ir-too-narrow` | IR width < 128 | SPEC §8, *"under 128d? maybe you were slippin"* |
| `ir-rank-collapse` | IR rank < 10% of IR width | *"a 128-dimensional embedding of a 4-dimensional signal"* |
| `probe-dishonest` | more than 30% of targets failed their own shuffled-label control | those columns are not admissible; the alarm names them |
| `underdetermined-probe` | fewer than 4 player-rows per IR dimension | the ridge fit has not enough data yet |
| `no-jspace` | no non-tautological target beats rand-proj | *"nothing beats the random-projection control on any non-tautological target"* |

The geometry panel next to it always shows IR width, IR rank and effective rank,
input width, input rank, nonzero input columns, and belief width/rank — so the
numbers are there even when no alarm fires.

---

## 5. Hooks needed from files this package does not own

Consumed defensively — the page shows each as `absent` with the owning producer
named — but each is one line in a sibling-owned file.

1. **`diag(K)`, appetite and `dw/dt` in the telemetry.**
   `strat_responder.py` builds `model_arrays` from the `ForwardResult`; it takes
   `ir`, `gram`, `score`, `winner_value`, `loser_value` but not
   `result.diag_k`, `result.appetite` or `result.dw_dt`, all of which already
   exist on the result. Add to the `model_arrays` dict:

   ```python
   "diag_k": array(result.diag_k, np.float32),
   "appetite": array(result.appetite, np.float32),
   "dw_dt": array(result.dw_dt, np.float32),
   ```

   and add `"diag_k", "appetite", "dw_dt"` to the `for name in (...)` tuple in the
   `if sampled:` branch. Until then the DPP marginal-inclusion signal — AGENDA
   C4, *"as a DPP kernel (→ determinant, diversity semantics)"* — is computed
   every tick and never observable, and C5's velocity-on-integrated-weight is
   visible only through its integral.

2. **Advantage in the online metrics.**
   `online.OnlineLearner.update` returns eight named losses; the advantage — the
   quantity SPEC §5 says optimization exists to increase — is computed inside and
   discarded. Add its mean to the returned `metrics` dict as `"advantage"`.

3. **The CGT game value on the live line.**
   AGENDA B11 is open (`228/228` `unresolved`). Whatever `game_value.py` returns
   for the tick's cartstate, emitted as `game_value` on the telemetry line, makes
   B11 continuously observable instead of only auditable after the fact. The
   viewer already reads `frame["game_value"]` and reports `absent — B11
   unresolved` when it is not there.

4. *(not a hook, a note)* `--model-sample-every 1` is an existing responder flag
   and the demo passes it. No change required.

---

## 6. Rules this package keeps

* No unit tests, and no simulator: `joracle/` imports `numpy` and the standard
  library, and nothing else from the project. `cartsim` is not imported anywhere
  in it, so a sibling deleting it (AGENDA D7) cannot break the viewer.
* Never `pkill`; `demo.sh down` sends `TERM` only to the pids it wrote itself,
  and leaves the responder to its own `--secs` bound.
* The viewer is a committed repo page, opened locally over the loopback port.
* Absent is rendered as absent. There is no default-to-zero anywhere on the read
  path.
