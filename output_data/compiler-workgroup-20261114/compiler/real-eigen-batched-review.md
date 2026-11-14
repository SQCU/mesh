# Batched values-only native result

The actual public client in real-eigen-batched-command.md ended with exit 0 and empty stderr. It compiled a float64 input of shape (2,3,3), stride (9,1,3), placed on MPS, and evaluated `eigvals(a) * 0.5`. Output shape was (2,3), dtype complex128; every value was finite. Permutation-invariant normalized eigenvalue errors were 3.675308451366845e-17 and 0. The result contains both real and conjugate eigenvalues for each member.

Construction retained 1,213 values, 4,728 bytes, 92 dispatches, 65 commands and 27 native entries. Program preparation restored five positions from its whole archive without a relation solve. Source hashes were identical before/after import, and all recorded source and native library hashes remained unchanged through invocation. The terminal tool session was 50536.

This observes real binary64, batch enumeration, values-only lowering, noncontiguous input geometry and a downstream numerical consumer. It does not establish all dtype/domain or ANE behavior. The CPU eigvals call supplies an output metric; it is not the implementation under test.
