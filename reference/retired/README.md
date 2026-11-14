# Retired reference source

Style: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights).

This directory holds dead source. It isn't a build input, an import target or a runnable alternative. Each file keeps
the malformed function header that mesh `b25ec14b` made, so it fails to parse by design. The bodies were moved
verbatim from source checkpoint `1ac90a36`.

The operator's instructions (`design/compiler-uncovered-work-2026-11-14.md`):

> then lets handle the concrete uncovered work next. if there are 'legacy' paths delete arbitrary substrings from their function headers by line so that they are unrunnable (producing python or c or whatever compile/interpreter errors) so there is no confusion about what their role in the codebase is: dead code. you do understand that we are not here to 'support code' within mesh or metal microbench, but to 'implement the basic algorithm specified by the change request documents and not something else'? there is no room for multiple methods here, inlined code, complexity, excess, or distractions.

> the requirement to make the old code uncompileable while leaving its structure behind for reference is literal and didactic; it is easily done and strictly communicates something important.

## Files

| File | Moved from (at `1ac90a36`) | Malformed header |
|---|---|---|
| `mesh_program.py` | `rdma/mesh_compile.py` lines 7611–7686, 8182–8279, 8348–8512, 8734–8764 and 8767–8821 | `_replays(`, `_bound(`, `program(`, `load(`, `_runner(` |
| `lower_coreml.py` | `compiler/mesh_compiler/lower_coreml.py`, the whole file | `lower(scheduled):` |
| `tp_cost.py` | `examples/tp.py` lines 102–116 (`Timed`) and 119–126 | `taken(fixed, inputs):` |
