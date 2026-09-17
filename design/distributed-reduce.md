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

Every occurrence is the same T. For another linear U,
`sum_i U(T(P_i)) = U(T(X))`. This identity does not extend to an arbitrary
nonlinear f: `f(sum_i T(P_i))` generally differs from `sum_i f(T(P_i))`.
Applying the same function to each operand does not make those two programs
equivalent. The required recombination remains a data dependency. It may be
restricted to the coordinates consumed by a call; this does not require a
whole-tensor or whole-mesh barrier.

The supplied example has

    T(x,y,z) = (x+y, y+z, z+x)
    T(P_1) = (x,0,x), T(P_2) = (y,y,0), T(P_3) = (0,z,z).

Their sum is T(X). Given only `t = T(X)`, invertibility permits the split
`S(t) = (T Q_1 T^-1 t, T Q_2 T^-1 t, T Q_3 T^-1 t)` and the sum R satisfies
`R S = I`. This algebra does not require Mesh to compute T, its inverse,
projections, or a numerical reduction. Those are supplied operations. Packed
contiguous subviews retain the caller's coordinate/index meaning; a slice of
T(X) is not silently substituted for T applied to a projected input.

The [Definition and L5/F1/F2 requirements](deliverables.md#definition--partial-tensor)
are the contract. Commit `3bd1eaa` deleted `TensorPart.partial`,
`withPartial`, `partialContributions`, `MeshError.partialOperand`, the validation
closure per call, the private function-binding bypass, and the reduction-result
identity. It also rewrote the requirements to license that deletion. Commit
`9114872` restored the requirement text; it did not restore the implementation.
The deletion is an open implementation gap, not a correction to the definition.

The current source binds `call`, `map`, and reduction's supplied combine through
the same public `call`. That describes the implementation and does not establish
that L5, F1 or F2 is satisfied. Mesh must preserve the declared tensor algebra
while keeping numerical functions supplied by callers.

These deletions remove setup records and work. They do not by themselves shorten
the C runtime's dependent-load chain or establish zero-copy contiguous receive
placement. Those remaining defects retain their status in H1–H8.

The build checks recorded at `3bd1eaa` establish compilation only, not compliance
with the restored requirements. Its 24-line Swift reduction is not completion
of L5.
