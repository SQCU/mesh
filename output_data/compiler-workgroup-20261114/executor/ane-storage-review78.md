# SAME e48 storage SSA review correction 78

Accepted draft77c is held/private strict pass; exact corrections only, metal_neural.mm.

1. Classes appends a flat Value but does not set internal. Realize includes it, so confined local SSA classes retain native allocation accidentally. Apply the SAME accepted whole-class decision to every original view of that root AND the new flat view. Keep original aliases/views; no new proof. The output-only final internal loop must not override this decision independently.
2. Current(root) copies Contents(root) into a new MIL constant, dropping alias-attached initial data and changing initializer ownership/lifetime. Compiler accepts removing this branch entirely. Scan ORIGINAL Value initial metadata by storage root once: any .data (including alias-attached) or .zeroed means mutable class remains native-backed. Existing MetalInitialContents/native state initialization consumes the original payloads in original order. No new copy/repack/clamp/shift. Immutable CONSTANT/quantized decoding unchanged. Contents(value) payload origin is physical root, not packed view; do not reconstruct initialization via scatter. Only confined class WITHOUT declared initializer can use local zero-fill undef refinement.
3. Source docs/status must accurately state these final rules; do not assert written-before-read proof from mere confinement. Undefined source contents may be refined, but no new read-history proof is supplied.

Keep remaining reviewed actual view/read/write/branch/fixedIO handoff. Existing IDs and source maps retained. No extra backend/algorithm/cache/fallback, no other source, no ABI/build/import/target/evaluator/agents/commit. Strict actual TINT_CFLAGS syntax, update status, HOLD. Compiler co-reviewed these corrections.
