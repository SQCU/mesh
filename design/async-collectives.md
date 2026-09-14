# Asynchronous collective interface

The sole purpose of this interface is asynchronous publication and asynchronous
consumption of tensor regions, without introducing synchronization around them.
Pallas supplies the API structure and calling syntax for expressing these qualities.
It does not enlarge the feature set.

The caller supplies the mesh, tensor placement, numerical functions, shapes,
layouts and simultaneous value instances. The implementation realizes that
configuration. It does not search for, score, tune or choose a useful placement.

## Calling contract

A call describes numerical functions over values, indices and masks, their input
and output regions, and their configured placement. Configuration binds the actual
storage and executable functions before numerical invocation.

A producer publishes a finished region while independent production continues.
Publication does not wait for delivery, acknowledgement, consumption, or completion
of the enclosing tensor operation. A consumer executes the partial numerical work
whose operands exist while independent inputs remain absent. Missing operands
constrain only the arithmetic that reads them.

The interface exposes no page-table manipulation, synchronization protocol,
completion-token scheduling, or manual partial-launch orchestration to numerical
callers. Mesh owns storage binding, transport, presence and lifetime handling.

## Strict dependencies

- Exact value, source and destination indices, shapes, strides and numerical masks.
- Realization of caller-specified placement and executable bindings before invocation.
- Actual registered backing for shared operands, with distinct storage for
  simultaneously live values. No copied transport operand store.
- Publication after the region's writes are visible, without waiting for unrelated
  writes, transport completion or a consumer.
- Canonical presence information sufficient to issue precisely the available work.
- Source storage retained through actual numerical and device reads; lifetime
  handling must not hold independent production or consumption.
- Correct arithmetic on the available region. Reduction contributions remain
  contributions until the configured reduction has all its required terms;
  nonlinear consumers read the resulting values, not unfinished sums.
- Transport bindings to the configured endpoints using the substrate's existing
  SEND/RECV and completion mechanisms.

Implement these dependencies using established library functions and canonical
mesh mechanisms. Refactor incompatible code directly. Do not insert wrappers,
parallel representations or invented algorithms around it.

## Scope boundary

Automatic placement, cost models, candidate search, benchmark targets, compiler
feature expansion, application migration, and additional compatibility surfaces
are not collective requirements and must not be added to this interface. There
is no separate performance, model, test-count, or source-line deliverable.
Runtime testing is not a requirement and does not determine implementation work.
Completion includes implementation and integration of the stated interface and
its strict dependencies, demonstrated from their source data and execution flow.
There are no code or runtime acceptance/rejection criteria. The
[end-to-end source derivation](streaming-algebra.md#end-to-end-source-derivation)
traces the implemented algorithms and their nonblocking streaming properties.

## Mechanism sources

The JAX authors' [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html)
provides the kernel/ref/grid/index-map calling structure.
Their [collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)
uses an existing local numerical implementation and distinct receive destinations
with asynchronous forwarding. Its GPU-specific synchronization is not caller API
for this mesh implementation.
Their [software pipelining](https://docs.jax.dev/en/latest/pallas/pipelining.html)
explains independent transfer and compute and the lifetimes of simultaneous values.
Papadopoulos and Culler, *Monsoon: an Explicit Token-Store Architecture* (ISCA 1990),
provides operand-associated presence and dataflow firing prior art.
Apple [TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)
defines the actual Thunderbolt SEND/RECV substrate and device ownership obligations.
These sources supply mechanisms, not additional project objectives.
