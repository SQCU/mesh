# Selected global state observation: status

Worker: Claude Opus 5.5 (`claude-opus-5-5`), session `047e12be-cd27-4637-a3bb-7d46f6e8216b`. Unit:
[eigh-global-state.md](eigh-global-state.md), at mesh HEAD `548edd7e`. **TERMINAL: HOLD.** One run, no production
edit, no further probe.

## Run

- **Process:** PID 89751, **exit 0** (`eigh-global-state.exit`), with an empty stderr.
- **What ran:** `compiler/eigh-native-command.md`'s client exactly (`compile_module`, then `Program(scheduled)`, then one
  ordinary `program(a)`), through `/tmp/archive-worker/diag/eigh_global_state.py`.
- **Files:** [eigh-global-state.json](eigh-global-state.json) and `.stdout`. There were no capture or read errors.
- **Hashes, re-verified now:** every sha256 equals the one captured at run time: the script, `native.py`,
  `operations.py`, `metal.py`, `schedule.py`, `lower_metal.py`, `libmetal_ir.dylib`, `libmetal_recording.dylib` and
  `libmesh.dylib`.
- **Client result, unchanged:** eigenvalue error 0.3628804865, residual 0.6454972244, orthogonality 0.0, values [0, 0,
  1, 4] against reference [0.2547188, 1.8227171, 3.1772829, 4.7452812]; finite.
  - The vectors are the permutation [[0,0,0,1], [0,0,1,0], [1,0,0,0], [0,1,0,0]].
  - **The original wrong outputs persist under this diagnostic.**

## Diagnostic ownership

- **Passive capture:** `sys.setprofile` return events only.
  - `operations._qdwh_member` returned once (one member), and its named Tensor locals were retained.
  - `_Builder.value` returns were recorded only for those exact objects (by identity), all from the row 1 body's
    Builder (`builder.p` = that body's program). No extra `value()` call or record was made.
- **The pin:** at `native.Target.functions`' return, inside `compile_program`, before the one `solve_program`. The hook
  was then removed, so `Program` and the invocation ran unhooked.
- **The read:** after `program(a)` returned, through `program.function` (the Bound; its target is the compiled
  Scheduled's) and its own `Arena` (`owner.solution.function_value_first[1]`, `owner.span`). That is not compile's
  arena.
- **The row:** row 1 is call 1, source F 1, instance 1, with `plan_calls` [0..8] and planned call (function 1, ordinal
  0, every 1). It has no composite map (`view_of`/`root_of` are the identity, and agree with `ValueInfo.storage`).
  Execution 0, copy 0 of 1, for each pinned root.

## Mapping, pinning and lifetime changes

- **Labels (view → root):** offset 23→22, size 25→24, below 27→26, h 29→28, squares 31→30, off_diagonal 33→32,
  e 35→34, d 37→36, u 1147→1146, at_median[0..1] →1148/1150, failed/retried →1152/1154, at_mean[0..1] →2249/2251,
  minus/plus/rank/rest →2253/2255/2257/2259, rotated/error/again/previous →2524/2526/2528/2530, unaccepted →2750.
- **Pinned:** 19 roots in `Function[1]`, with **only `kind` changed, EXECUTION (1) to INVARIANT (0)**. Each had bytes > 0,
  run 1, no buffer, and was not explicit or joint-bound. Bytes, begin/end, run, count, shares, lag and state are
  unchanged (before/after fields in the JSON).
- **Not pinned, unavailable:** size, d, off_diagonal, at_mean[0] and at_mean[1]. They are internal: Storage bytes 0,
  run 0, not realized.
- **The layout changed.** Against the unpinned capture (`eigh-source-capture.json`):
  - offset moved 6440→6632, below 6448→6640, squares 6456→6648 and e 6464→6656, while h stayed at 512;
  - each pinned root now has its own non-reused bytes (`plan_intersect` for INVARIANT).
  - Other values' placement may have moved too.
  - This is not the original layout's runtime state. The wrong client result persists with these extended lifetimes; the original-layout intermediate state and algorithm correctness remain unproved.

## Final state (pinned roots, after the one invocation; final values only)

| Value | Final | Value | Final |
|---|---|---|---|
| offset | 0 | minus, plus, rank, rest | 2, 2, 2, 2 |
| below | 0 | failed, retried | 0, 0 |
| e | 4 | at_median[0], at_median[1] | 4.0, 0.0 |
| squares | 36.0 | error | 3.1903733e-12 |
| h | exactly the input a | previous | 3.1903733e-12 |
| | | again, unaccepted | 0, 0 |

- **u:** finite (rounded display; full values in the JSON), `[0.82263, 0.39842, −0.33751, −0.22501; 0.39842, 0.08671, 0.84843, 0.33751; −0.33751,
  0.84843, −0.08671, 0.39842; −0.22501, 0.33751, 0.39842, −0.82263]`.
- **rotated (rounded display):** `[1.61240, 0.53436, 6.5e−8, 6.8e−9; 0.53436, 0.46504, 7.4e−7, 1.6e−6; 1.2e−7, 7.9e−7, 3.26647, −0.36317;
  −3.4e−8, 1.6e−6, −0.36317, 4.65609]`.
- **Gating:** e = 4 > 0, so the e-gated regions were active. No loop-intermediate history is inferred from the final
  state. The values match the supervisor's findings.
- Final `again = 0` and `unaccepted = 0` do not establish zero loop passes. Entry conditions and iteration history
  were not observed. The full recorded `error` and `previous` are both `3.1903733127558453e-12`.

## Local raw-log identity

- `eigh-global-state.stdout`: 101636 bytes, SHA-256 `33ec3be1896ded1bda1f7adf05df6ebf841a9ea0de3d2878fb04ec17bbbd9723`.
- `eigh-global-state.stderr`: 0 bytes, SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
