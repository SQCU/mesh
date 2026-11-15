# Scalar conditional builder checkpoint

The scalar region API now opens an actual Tint `If`, emits each arm in its own block, and joins typed results with `ExitIf`. Existing scalar and logical load/store builders append to the current lexical block. Construction requires closed arms before the function yield. Python `_Builder.replay` preserves the same record/result correspondence; an unpredicated scatter uses the existing UINT32_MAX predicate contract.

This is the native/Python API only. The rejected and incomplete `lower_inductor.py` producer is excluded. No public driver connection, `_instruction` displacement or numerical execution is claimed by this checkpoint.

Supervisor validation: actual Makefile Tint flags with `-fsyntax-only -Wall -Wextra -Werror` passed for `metal_ir.mm`, empty stderr and unchanged source hashes recorded in `numerical60-native-check.json`. Current `metal.py` AST, pyflakes and diff checks passed. Native shared-library build/ABI and workload execution remain executor-coordinated.

The actual masking reference is installed Torch `codegen/mps.py:MetalOverrides.masked`; `OpsHandler.masked` defines guarded loads/stores and the selected result. This does not assert that upstream Halide's predicated pure arithmetic is invalid.
