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
