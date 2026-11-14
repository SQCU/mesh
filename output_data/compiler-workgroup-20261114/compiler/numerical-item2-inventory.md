# Item 2: remaining numerical producer/consumer work

Scope: design/torch-source-lowering-change-request.md §6. The requested compositions cover every processor named by placement and required numerical precision. The request supplies QDWH/eigh/SVD and scaling-and-squaring methods; it explicitly leaves the nonsymmetric method to implementation. Source formulas and observed cases are different completion claims.

## Connected and observed

Public compile_module → native construction/whole archive reload → invocation has numerical output evidence for real4×4 eigh, complex2×2 eigh, indefinite rank-deficient3×3 pinv, full rank-deficient3×2 SVD, wide2×3 gelsd, batched2×2 matrix_exp with different squaring counts, and dense3×3 complex nonsymmetric eig. Latest complex eig residual is1.80e−7 (a9618bd3); no broad dtype/domain result is inferred from these cases. Existing CPU operator paths remain Accelerate LAPACK.

## Implementation gaps

1. Real non-CPU eig/eigvals still reaches _eigen → _Lapack(GEEV). The placed complex path has ZGEBAL → unblocked ZGEHD2 → ZLAHQR → ZTREVC → ZGEBAK with source INFO checking. Next unit is the real-arithmetic source path, preserving real Schur1×1/2×2 blocks and complex conjugate outputs; no target cap or real-to-complex workaround is assumed.
2. Method/performance difference, not an added completion requirement: current complex driver uses unblocked routines and ZLAHQR for every shape, without ZHSEQR’s large-matrix multishift/AED/rescue dispatch. Existing numerical failure checking remains active. §6 leaves method selection open; this does not impose a shape restriction or turn unmeasured large shapes into unsupported ones.
3. Requested ANE extent/reduction/storage-alias lowering is implemented/strict/built, but the last actual2×2 eigh case stopped before model construction at bounded-sort scalar work preparation. The exact finite-threshold replacement is committed9fa86439 and in the matching local build; its ANE retry is held under the new priority. Integer bit semantics, U32 comparison semantics and mixed-scalar alias reinterprets remain exact target-consumer concerns, not hardware declarations.

## Required evidence and contract review still pending

- Real nonsymmetric dense input with real and complex-conjugate eigenvalues through the actual public native caller, followed by values-only/source composition.
- Remaining supported dtype/precision paths, batching, values-only/no-vector forms, empty/size-one and relevant shape/layout/composition cases. Source exists for many; lack of a run is not a rejection policy.
- Tall gelsd/gelss: current _lstsq_metadata rejects m>n>0 before learning whether rank-dependent residuals are selected. Solution-only and rank-only functions have fixed result sizes. Inspect actual export symbolic-result/normalization support; residual variability does not justify rejecting every tall program.
- Root593 owns the concrete matrix_exp error-result ordering correction: NaN output must precede status CHECK through an actual retained dependency, not merely Python statement order. Current pre-result _assert_async does not establish that.
- Actual runtime error/domain results: matrix_exp squaring-limit error+NaNs, spectral nonconvergence/invalid input and source-defined tolerance/rank behavior. Do not manufacture nonconvergence by changing algorithms or weakening checks.
- Shared generated RNG inter-call ordering is assigned to frontend item3; numerical source-site/retry operands must pair with that existing owner. Repeat multiplicity and unresolved Work cost fields are separate unfinished cost consumers, not reasons to reject numerical construction.

DLANV2 and DLAHQR now have source checkpoints and F32/F64 construction evidence; neither is yet a complete public real eig path. Next source unit: real quasi-triangular eigenvector recovery (DLALN2/DTREVC) and the shared GEEV driver/result consumer. Canonical Reference-LAPACK sources and hashes are in real-eigen-references.json. Shared balancing/Hessenberg/eigenvector helpers retain their common owner where their algorithms coincide; this inventory does not authorize a second backend or evaluator. No numerical target runs while source is transitional.
