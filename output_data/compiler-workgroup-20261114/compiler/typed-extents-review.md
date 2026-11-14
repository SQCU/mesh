# Typed extent handoff checkpoint

The existing structured operation now retains ordered `(tensor, bound)` symbols and each loop's optional symbol index. The live Metal encoder passes the actual tensor ValueID to `metal_ir_symbol`, retains its returned dimension, and supplies it to `metal_ir_loop_extent`. Affine rows include loops followed by symbols. Static operations keep their prior dimensions. No scalar is read into Python, and no numerical backend is added.

Supervisor source review checked the existing native consumer: symbol values contribute read effects, including a temporary written by an earlier operation. The existing lowerer clips iteration extents and uses the supplied affine symbol dimensions.

SAME numerical session 64ddaf6e-3a70-4b6a-9d0b-50691a84c6cd observed the actual structured-record/Builder path in `/tmp/numerical-decompose-check/symbol.py`: producer scalar ValueID 5, temporary storage 4; consumer symbol dimension 1 and loop 0 clipped by symbol 1. Prior installed `metal_ir_lower` succeeded. Fused construction had one dispatch; a CPU producer with a GPU consumer retained the canonical RAW witness on storage 4 (source view 1, target view 4). The tool output is retained below, extracted from that session's output.

This is a construction/dependency observation, not numerical execution or the current full compiler path. The diagnostic used committed mesh bindings from `/tmp/numerical-mesh-head` to match the installed library and an in-process repository path override because the current observer ABI was not built. The ongoing explicit construct/lower phase change is excluded from this checkpoint.

QDWH-eigh/SVD/pinv/lstsq and non-Hermitian eig remain unfinished. The extent interface supplies one required operand; no outlined spectral step or whole-program spectral execution is claimed. The earlier matrix-exp connection is separately documented in matrix-exp-review.md.
