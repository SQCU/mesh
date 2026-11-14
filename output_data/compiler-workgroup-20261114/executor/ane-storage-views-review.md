# ANE storage views — reviewed source checkpoint

Metal `7f7ed436` follows dynamic-shape `38f67112` and coupled-reduction `ff3609f7`.
The source consumer is complete for this bounded handoff; hardware behavior remains unmeasured.

- One `Lowering::Storage` owns the current physical SSA tensor for each original `Value.storage` class.
  Source views keep their original IDs, offset and layout. Mutable operand, logical-load and symbol reads
  compose those exact layouts; each Store updates the shared current tensor. Partial writes accumulate.
- Written aliases are merged once per class by the existing zero-domain conditional; its false branch
  returns the incoming class tensor. Lexical branch/loop construction restores the storage environment.
- Fixed physical spans come from original `Bytes` reach, including root capacity and actual live views.
  An appended flat typed alias carries that same reach into bindings and later `Realize`; no early realized
  allocation lookup, alias solver or new runtime descriptor/schema.
- The existing maximal ANE run's scope projection includes outside inputs, outputs, gathers and symbols.
  Only confined classes without declared initial data or zeroed metadata receive local zero-fill undef
  refinement. Every view including the flat alias receives the same internal mark in one Values pass.
- Any mutable initializer remains under original native preparation, including alias-attached payloads
  and their original order. The rejected draft copied only root data into a MIL constant and left flat
  metadata noninternal; both source-owner mismatches are removed. Immutable constant/quantized decoding
  is unchanged. Confinement does not supply a written-before-read proof.
- Root declaration scalar affects capacity only; live read/write interpretations choose tensor type.
  Mixed live element interpretations fail explicitly rather than applying a numeric reinterpret cast.

Validation: SAME e48 session, exact model `claude-opus-5-5`; supervisor strict Objective-C++ syntax with
actual Makefile TINT_CFLAGS, `-Wall -Wextra -Werror`, PASS; diff check PASS. Only metal_neural.mm changed.
Original malformed files remain unchanged. Exact source and session-log hashes are recorded alongside.

No native build, MLModel compile/predict, requested-ANE execution or scaling measurement was run for this
unit. Compiler owns the matching build and sole public 2×2 real Hermitian requested-ANE client with
MLComputePlan placement counts. Dynamic shape/tree eligibility, CoreML buffer overlap behavior and the
pre-existing nonempty unsigned-extrema gap remain explicit; source success does not prove ANE execution.
