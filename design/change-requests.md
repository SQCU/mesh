# Change requests

A change request describes what ought to be written. It uses only non-idiomatic descriptions of work: each requirement
cites the reference authors, theory, papers, books or source code that say exactly what to write, and the code is
written as that source states it. Code already in these repositories and documents Claude wrote are not sources, since a
change request exists to replace idiomatic or incomplete code with the cited method. Its body is the turns that led to
it, quoted verbatim, and where no source says what to write it says so and invents nothing.

For a request composing other requests in externally cited roles, see the
[example and reusable form](compilation-composition-change-request.md#reusable-form).
For compilation work, that form identifies the properties preserved, representations consumed or eliminated,
and actual residual runtime operands. Its [reference systems](compilation-composition-change-request.md#s8-analysis-results-become-lower-ir-abstract-operations-disappear)
show how solved abstract problems become optimized programs; retaining the original abstraction at runtime is
not the preservation criterion. State performance obligations separately from semantic preservation.
