# Existing graph ISL objects

`mesh_graph_context_isl`, `mesh_graph_domain_isl`, and
`mesh_graph_index_value` / `mesh_graph_index_predicate` return the exact existing
context, domain box, value and predicate pointers. `mesh_graph_index_domain`
returns the index's existing domain. These are borrowed views: they allocate
nothing, increment no reference count and transfer no ownership. The domain's
dimension storage remains available through `mesh_graph_domain_dims`.

Retain the complete Mesh owners for every use of these views. Each domain
retains its context, and each index retains its domain. An index contains a
value or a predicate; the other accessor returns NULL. Do not pass a borrowed
ISL pointer to an operation that consumes ownership. Obtain an ISL reference
under the context lock when its C++ wrapper or operation requires one.

`mesh_graph_context_lock` and `mesh_graph_context_unlock` operate on the same
normal pthread mutex used by existing graph operations and return its status.
Hold that lock for all ISL operations on the context, including reference-count
increments, destruction, error handling and Polly affinator/cache operations.
The mutex is not recursive. Do not invoke internally locking Mesh operations,
including domain/index/access-map destruction, while holding it. Retain Mesh
owners before the locked interval and release them after unlocking. Consumers
must preserve this ordering on exception paths as well as normal returns.

`mesh_graph_index_take_locked(domain, value, predicate)` consumes owned ISL
references already produced in that same context and domain space. It calls the
original `index_made` function under the caller's existing lock. That function
retains the domain and owns the references in the canonical index; on allocation
failure it frees them, and with neither value nor predicate it returns NULL.
There is no extra lock, reconstruction or relation calculation. After unlocking,
the returned index uses the ordinary `mesh_graph_index_free` lifecycle. Passing
an extra ISL reference lets another complete result owner retain its own view.

The intended original Polly consumer receives `isl::set` and `PWACtx` views
from these objects, retaining the input domain, complete analysis and fixed
binding owners. Its wrapping restrictions and recorded assumptions remain its
own result; this API does not discard or discharge them.

Compile against Mesh's existing ISL 0.27 headers and link to its existing ISL
implementation, built with `--with-int=imath-32`. `rdma/Makefile` already links
that archive into `libmesh.dylib`. The consumer must use that same implementation
for all borrowed objects; linking a second ISL implementation or system 0.28 is
not this interface. The public header only forward-declares the original ISL
types and introduces no replacement context or object representation.
