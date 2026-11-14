# Rejection-filter correction, orders 9 and 10 (SAME593 `59368744-…`, claude-opus-5-5): HOLD, unstaged, no client run
- `check_shards.py` (+37 −15 against HEAD): fewer than whole + one rank path raises `ValueError` before any read. Name
  pairing keeps its lookup (`constants()` records an ambiguous name's (shape, type)). Absent names, and names with more
  than one value in the whole model or a rank, are listed as INCOMPLETE name-based comparisons, worded as not proof the
  ranks are wrong nor a requirement to keep every constant. Numeric checks stay MISMATCHES. After the full report, any
  of either raises one `RuntimeError` listing both; success returns 0.
- `run_pair.py` (+40 −12): no `except Exception: pass` (`emission_error`, `log_error` recorded); `_median` keeps `None`.
  `main` writes `result.json` and every run record, then fails on any run's error, failed ranks, rank errors or absent
  measurement, and on any requested set (the solo, each split) without exactly --passes x --repeats records (zero included):
  `status='failed'` with `failures`, then `RuntimeError`. Otherwise `complete`, 0.
- The rejection-filter comments are kept. Checks: AST passes on both files, `git diff --check` is clean; pyflakes reports
  only `run_pair.py:28` `re` unused (predates these orders).
