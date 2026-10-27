# Whole-function assignment

The solver consumes work extents, physical paths, and records of completed calls. It returns numbers for the
compiler: work shares and path shares. It receives no callable, stream, timer, tensor, or communicator. It cannot
run work, wait for peers, allocate a runtime buffer, or install a plan.

## Numerical operands

`assignment.assign(decisions, observations, step=proposal)` uses bounded linear shares. The compiler rounds compute shares to
whole tiles and path shares to Q40 units. Projection preserves each declared total. The numerical fits have positive
regularization; they use neither logarithms of shares nor division by a share that can vanish.

A compute share starts from declared capacity, with equal capacity where none is known. Torch configuration reads
each node's GPU core count as a prior, not as a performance measurement. Every solve fits fixed and work-dependent
costs from complete-call observations. A fixed cost per code version separates changes in the implementation from
changes in its operands. The duration of each complete call fits the stage envelope directly; the model does not
require an optional list of per-operation costs. The affine envelope estimates whole-call cost by work share. Its
coefficient variances retain uncertainty where the completed calls cannot distinguish two resource costs.

The historical equal-finish solver rounds the fitted allocation. Whole-call finite differences move all compute
and route coordinates together. Compiler parameters use the same vector. An explicit proposal ordinal advances
the paired perturbations; retaining only a bounded history cannot freeze their direction or radius. Each compute
or route axis carries its realization grain, so a perturbation can still cross a compiler rounding boundary. No result is accepted or
rejected as a candidate. There is no saved controller or hidden solver lifecycle to restore. In-memory observations name what
ran, its actual operands, and its measured completion interval.

This is an empirical local solver. A small residual, a compiled plan, or a capacity prior does not establish a global
optimum. The required evidence remains lower whole-function elapsed time and higher useful output per second.

## Compilation

`configuration.placement` reads observations before tensors are built. `configuration.bind` declares the complete
function and installs the resulting page placements and routes. The numerical function is absent from execution.
More routes change the compiled tables; they do not add a controller around each transfer.

`rates.step`, the engine example, and the Xonotic planner and solver consume the same assignment operands. They keep
observations as data. They do not reconstruct a controller from a JSON object. A failed invocation is a record, not
a fabricated duration or an instruction to roll back its allocation.

## Observations

Programs execute their own function calls. There is no measurement runner taking a callable. Whole-function
observations use a host clock and ordinary command-buffer completion state. No GPU timing event, timestamp shader,
completion callback, or timing-only host drain is added. The stream's existing publication operation submits output
writes without waiting for their completion.

A separate statistics thread polls that state. It records the interval between its last pending observation and its
first completed observation. Reported elapsed time uses the upper end, so useful bandwidth uses the conservative
lower end. Completion at a shared command-buffer boundary is coarse. Pending records and several late peer records
can coexist; observing them does not halt a subsequent function. Shutdown drains the records after submission ends.

Records contain whole-call time and useful bytes together. Relays do not increase the useful byte count. The hourly
runner uses committed main and the same library functions as the examples. Its scripts supply configuration operands.
