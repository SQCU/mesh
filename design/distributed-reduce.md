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

## L5 contribution typing

`TensorPart.partial` describes a value view pending reduction, following the
[canonical DTensor reference](algorithm-sources.md#tensorpartpartial).
`reduce` marks its contribution views and nonfinal intermediate views partial.
Its supplied combine is bound directly through the ordinary preparation path.
Public `call` instead records this validation closure:

```swift
preparations.append { [unowned self] in
    if let part = inputs.first(where: { $0.partial || partialContributions.contains($0.identity) }) {
        throw MeshError.partialOperand(part.withPartial(true))
    }
}
```

That is the only throw in `call`. It executes inside `start()`, after declaration
has identified every reduction contribution. Declaring a consumer before its
reduction therefore does not bypass the check. `map` uses public `call` and has
the same rule. The check does not inspect the supplied numerical function.

There are two identities to distinguish: a logical value and the registered
storage it names. `send` preserves the former while binding destination storage.
The completed result of `reduce` has a fresh logical identity over the final
result's storage. This applies uniformly to every contribution count. For a
single contribution, the reduction is the identity function and adds no combine
call, allocation or copy just to distinguish the completed view.

The source consequences are:

| Declaration | Setup consequence |
|---|---|
| A call reads a contribution, before or after `reduce` is declared | Its logical identity occurs in `partialContributions`; `start()` throws |
| A value is sent, then either source or received view is declared a contribution | Both views have the same logical identity; neither bypasses validation |
| An already-partial view passes through a cached delivery | `send` preserves the view's partial bit |
| A public call reads the completed result of `reduce([x])` | The completed identity differs from x; the call is valid |
| A public call reads a result of a reduction with several contributions | The completed view is nonpartial and has a distinct identity; the call is valid |
| The same logical value is explicitly sent from different ranks | Source rank distinguishes the delivery keys; an earlier route does not replace the requested edge |

Earlier value-type snapshots are not retroactively mutated; the setup contribution
set is what recognizes them. The thrown operand explicitly has `partial == true`.
`preparations`, `deliveries` and `partialContributions` are discarded before
`mesh_calls_start`. The C descriptors, native submission closures and numerical
completion code are unchanged. There is no partial-type query or identity lookup
on invocation, publication, TX, or RX.

The source check covers the declaration rules above; the module and existing
Core ML, Gram-chain and explicit-sync callers build. No runtime measurement is
claimed. L5 does not resolve the separately recorded W1–W6 runtime defects.
