# placement-24: paired endpoint selection (SAME979; schedule.py only; uncommitted; HOLD)

- **RECEIVE with a landing:** `endpoint()` reads `made.landings.get(channel)` and resolves `binding_of` on the landing destination's `.root`.
  - Destination bound to a shared run 1 root (`direct`): `root` is the destination Root; `view` is its typed view moved by `delta`; `backing` is that storage's full nbytes; `staged` False; row root `storage + 1`.
  - Otherwise: `root` is the actual receive-wire Root (`received[channel]`); `view` and `backing` are that wire's typed view and root bytes; `staged` True; row root 0. `storage`/`delta` keep the destination's binding.
- **SEND and ordinary RECEIVE:** unchanged.
- **Retained on every endpoint:** occurrence, vertex, logical identity and construction. The copy operation IDs stay in `construction.landings`.
- **Local rows:** every one now carries `endpoint.root.root`; the `MESH_ABSENT` staged placeholder is gone.
- **Removed:** `Prepared.received` and `bind(received=...)`. `bind(wired, copied=())` is as before. `remap` is still local-only on the selected Root.
- **`typed_view`:** its comment now cites `metal.py _Builder.value` (linear layout).
- **Checks:** `ast.parse` passes; pyflakes exits 0; the diff is schedule.py only, with no new line over 120. sha256 `7903375770df2948dcca5021975ba3520a2b718034e99b18af0d89aad45c7078`. No import, build or run.

Supervisor removed an unused optional-CallOperands fallback (analyse is the sole constructor), corrected old Prepared prose, and repeated AST/pyflakes/diff checks: PASS.
