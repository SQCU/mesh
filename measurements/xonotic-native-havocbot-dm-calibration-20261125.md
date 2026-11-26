# Batch-Isolated Per-Bot Native HavocBot Skill Patch & 4-Studio 1v1 DM Calibration (2026-11-25)

## 1. Engine & QuakeC Patch (Zero Distortion of Native HavocBot Equations)

Opened and patched Xonotic's native HavocBot skill pathway across both **QuakeC** and **DarkPlaces C** (`cc79a1bb3`) so every individual bot carries its own isolated native Xonotic `skill` value (`0..17`) inside the same server and even on the same team, without mutating global `skill`:

- **QuakeC (`bot.qh`, `bot.qc`, `havocbot.qc`, `roles.qc`, `aim.qc`, `sv_payload_strategy_io.qc`):**
  - Added `.bool bot_has_base_skill; .float bot_base_skill;` and `#define BOT_SKILL(e) ((e).bot_has_base_skill ? (e).bot_base_skill : skill)`.
  - `bot_setnameandstuff` (`bot.qc`) reads an optional 19th TAB column (`argv(18)`) from `bots.txt` into `this.bot_base_skill` (falling back identically to server `skill` on stock 18-column `bots.txt`).
  - Replaced every per-bot `skill` reference in `bot.qc`, `havocbot.qc`, `roles.qc`, and `aim.qc` with `BOT_SKILL(this)` (eliminating `aim.qc`'s global `skill_save` mutation completely!).
  - Exported `BOT_SKILL(it)` into `OBS["BOT_GLOBAL_SKILL"]` (`sv_payload_strategy_io.qc:531`) — masked out of the actor by `prenormalize_obs`, exact ground truth for offline calibration and privileged training targets.
- **DarkPlaces C Batched & View-Isolated Execution (`bot_batch.c`, `prvm_view.c`):**
  - `bot_batch.c`: packs each bot's own `bot_base_skill` into its per-row `BOT_BATCH_SKILL` slot (`data[BOT_BATCH_SKILL * padded + active]`) so the batched controller warp is 100% per-bot isolated.
  - `prvm_view.c`: added `bot_base_skill`, `bot_has_base_skill`, `bsk`, and `bsk_eff` to `PRVM_ViewSkill` so engine-level read-only view isolation protects per-bot base skill identically to the 12 native skill offsets.

### Single-Match Isolation Verification
Spawned a single live `boil` payload server (`+skill 7`) with 9 distinct native HavocBot base skills (`[1, 2, 3]` on Team 1, `[5, 6, 8]` on Team 2, `[10, 12, 15]` on Team 3):
```
PER-BOT OBSERVED (team, BOT_GLOBAL_SKILL) INSIDE SINGLE MATCH:
  [(1, 1.0), (1, 2.0), (1, 3.0), (2, 5.0), (2, 6.0), (2, 8.0), (3, 10.0), (3, 12.0), (3, 15.0)]
PER_BOT_SKILL_ISOLATION_VERIFIED=TRUE
```

---

## 2. 4-Studio Offline Reference-Game Calibration (`96` Native 1v1 DM Duels in `91.0 s`)

Ran [`xonotic/solver/strat/mech_calibration.py`](../xonotic/solver/strat/mech_calibration.py) across all **4 Apple M5 Ultra Mac Studios** (`96 / 96` 1v1 DM duels on `boil`, `slowmo 3.0`, `22 s` wall / `66 s` game time per duel), crossing Xonotic's **18 real native `bots.txt` personalities** (`Hellfire`, `Toxic`, `Discovery`, `Pegasus`, `Eureka`, `Airhead`, `Gator`, `Delirium`, `Death`, `Scorcher`, `Necrotic`, `Dominator`, `Thunderstorm`, `Mystery`, `Lion`, `Sensible`, `Shadow`, `Resurrection`) with native `bot_base_skill ∈ {0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15}` (anchor `s05_Neutral = 1000.0`):

| Native `bot_base_skill` | Mean Empirical 1v1 DM Elo | Mean Frags / Duel | Sample Native Competitors (Elo) |
|---:|---:|---:|---|
| `0` | `941.5` | `0.55` | `s00_Hellfire` (`935.4`), `s00_Toxic` (`947.6`) |
| `1` | `910.0` | `0.33` | `s01_Discovery` (`900.4`), `s01_Pegasus` (`919.7`) |
| `2` | `964.0` | `0.88` | `s02_Eureka` (`1006.3`), `s02_Airhead` (`921.6`) |
| `3` | `872.6` | `0.22` | `s03_Gator` (`922.5`), `s03_Delirium` (`822.8`) |
| `4` | `954.9` | `0.86` | `s04_Death` (`949.7`), `s04_Scorcher` (`960.0`) |
| `5` | **`1012.1`** | `1.24` | `s05_Neutral` (`1000.0`), `s05_Necrotic` (`996.7`), `s05_Dominator` (`1039.7`) |
| `6` | `994.3` | `1.00` | `s06_Thunderstorm` (`959.1`), `s06_Mystery` (`1029.5`) |
| `8` | `1023.4` | `1.60` | `s08_Lion` (`1034.7`), `s08_Sensible` (`1012.0`) |
| `10` | **`1074.4`** | `2.14` | `s10_Shadow` (`1044.0`), `s10_Resurrection` (`1104.8`) |
| `12` | **`1076.2`** | `1.62` | `s12_Hellfire` (`1134.4`), `s12_Toxic` (`1018.1`) |
| `15` | `1046.4` | **`2.18`** | `s15_Discovery` (`1060.5`), `s15_Pegasus` (`1032.3`) |

- **Empirical native spread inside the same engine:** **`311.6` Elo** (`822.8` → `1134.4`) and **`6.6×` frag output** (`0.33 → 2.18` frags/duel) purely from native HavocBot parameters, with zero synthetic handicaps.

---

## 3. Native Parameter → Empirical Elo Regression & Inverse Roster Sampling

Fitting $\hat{E}(\text{base\_skill}, \theta_{12\text{D}}) = w_0 + w_s\,\text{base\_skill} + w_{\log}\log(1 + \text{base\_skill}) + \mathbf{w}_{12\text{D}}^\top \theta_{12\text{D}}$:

| Model / Regularization | In-Sample $R^2$ | Leave-One-Out $R^2$ | In-Sample RMSE (Elo) | Leave-One-Out RMSE (Elo) |
|---|---:|---:|---:|---:|
| `base_skill + log1p` only (`3` params) | `0.5452` | `+0.4154` | `46.5` | `52.8` |
| Full `(base_skill, 12D personality)`, `ridge = 15.0` | `0.7207` | `+0.4066` | `36.5` | `53.2` |
| **Full `(base_skill, 12D personality)`, `ridge = 40.0`** | **`0.6640`** | **`+0.4401`** | **`40.1`** | **`51.7`** |
| Full `(base_skill, 12D personality)`, `ridge = 100.0` | `0.6013` | **`+0.4434`** | `43.6` | **`51.5`** |

- Both `base_skill` (`+10.44` Elo/level + `+6.06 * log1p(s)`) and the native 12D `bots.txt` personality offsets contribute real held-out predictive power (`LOO R² = +0.443`, `LOO RMSE = 51.5` Elo), enabling `sample_roster_from_target_elos` to invert the map and sample native HavocBot rosters matching arbitrary target Elo distributions (e.g., human-server Gaussian or skewed lobbies) directly inside any single match.
