# Cost filters in `shares.py` (SAME593 `59368744-…`, claude-opus-5-5): HOLD, unstaged, no import or run
- **Superseded:** my two added `ValueError`s (the shard-without-part refusal and the `KeyError` wrapper) are removed, as root reviewed.
- **The silent omission, fixed** (`supplied`, +15 −9 against HEAD): `segment(node) is None` used to drop a node's supplied time.
- **Algebra:** placement's shard `units` are (x local, n full) (`placement.py:647, 800`), and `speed(times, x) = x / times[x]`. So:
  - a shard of an actual part (`parallel.py` `segment` not None, one of `solved`'s parts) is coefficient `n / speed(x)` of its variable;
  - every other node, a shard in no part included, is fixed current work `x / speed(x)`, exactly `times[x]`.
  The branch sets only `fixed` and `coefficients`; one `Cost(...)` call then shares the identity fields.
- **Kept:** `SEND`/`RECV` stay `solved`'s collective time. A missing supplied time raises at the direct `times[…]` lookup.
  The comment says work without a variable stays fixed. `measured` is unchanged (`parallel.py:125–128` raises on `costed=False`).
- **Checks:** `ast.parse` passes, pyflakes reports nothing, `git diff --check` is clean, and no line is over 120 characters.
