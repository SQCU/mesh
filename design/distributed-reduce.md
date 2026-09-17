# Distributed reduction

The [asynchronous collective contract](async-collectives.md) is the complete scope.
The caller configures placement and the numerical reduction. Mesh binds the
corresponding values and executable functions before invocation.

For a contraction, a partial over domain Q is

    P[I,J,Q] = sum(k in Q, A[I,k] * B[k,J]).

Separate Q domains may compute and publish independently. The configured reduction
combines these contributions into Y[I,J]. A nonlinear consumer of Y requires that
sum; unrelated output coordinates and upstream partials remain independent.
Floating-point association and accumulator type belong to the supplied arithmetic.

Rabenseifner, *Optimization of Collective Reduction Operations* (ICCS 2004), and
Patarasuk and Yuan, *Bandwidth optimal all-reduce algorithms for clusters of
workstations* (JPDC 2009), provide reduce-scatter/all-gather decompositions.
Use canonical implementations of the configured collective rather than inventing
a new scheduler, handshake, mutable accumulation protocol or transport mode.

Publication exposes completed contributions without waiting for transfer or a
consumer. Actual operand presence enables consumers. Distinct live contributions
retain distinct canonical backing until their numerical and transport reads end.

## L5 same function, different operand scopes

The user's construction fixes the numerical function and changes its operands.
For coordinate projections with `sum_i Q_i = I`, let `P_i = Q_i X`. Linearity gives

    T(X) = T(sum_i P_i) = sum_i T(P_i).

Every occurrence is the same T. A contribution may feed another ordinary call
before recombination. For another linear U, the caller can equivalently arrange
`sum_i U(T(P_i)) = U(T(X))`. Mesh has no numerical rule to infer or enforce here;
it binds the input and output scopes that the caller actually declared.

The supplied example has

    T(x,y,z) = (x+y, y+z, z+x)
    T(P_1) = (x,0,x), T(P_2) = (y,y,0), T(P_3) = (0,z,z).

Their sum is T(X). Given only `t = T(X)`, invertibility permits the split
`S(t) = (T Q_1 T^-1 t, T Q_2 T^-1 t, T Q_3 T^-1 t)` and the sum R satisfies
`R S = I`. This algebra does not require Mesh to compute T, its inverse,
projections, or a numerical reduction. Those are supplied operations. Packed
contiguous subviews retain the caller's coordinate/index meaning; a slice of
T(X) is not silently substituted for T applied to a projected input.

`TensorPart` describes operand storage and placement. It carries no numerical
permission or pending-reduction type. `call`, `map`, and reduction's supplied
combine all bind through the same public `call`. Producers publish each declared
output scope at numerical completion; sends move that scope; consumers receive
their declared scopes without a library-imposed whole-tensor reduction barrier.
If a particular operation requires a recombined input, its caller names that
input in the dataflow. The library does not inspect or classify the operation.

The former L5 was a specification error, removed on 2026-09-16. It treated a
reduction input as forbidden to ordinary calls, even though the same value could
legitimately feed T or U. Removing it deletes `TensorPart.partial`,
`withPartial`, `partialContributions`, `MeshError.partialOperand`, the validation
closure per call, the private function-binding bypass, and the fresh identity
created only to certify a reduction result. A one-contribution reduction is the
existing send of that value; it introduces no new identity or storage.

These deletions remove setup records and work. They do not by themselves shorten
the C runtime's dependent-load chain or establish zero-copy contiguous receive
placement. Those remaining defects retain their status in H1–H8.

The Mesh module, native assembly, existing Core ML/Gram/indexed-gather/explicit-sync
callers and the engine Mesh library build with the deletion. No runtime workload
or new evaluator was used. Swift source decreases by 24 lines; documentation
correction is separate from implementation reduction.
