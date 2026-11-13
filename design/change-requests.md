# Change requests

A change request describes what ought to be written. It uses only non-idiomatic descriptions of work: each requirement
cites the reference authors, theory, papers, books or source code that say exactly what to write, and the code is
written as that source states it. Its body is the turns that led to
it, quoted verbatim, and where no source says what to write it says so and invents nothing.

The operator, 2026-11-13, on the meaning of "idiomatic":

> "“Idiomatic” here concerns agents substituting familiar implementation habits or local terminology for a named algorithm, " it actually involves limiting hte propagation of any data structure or any algorithm which is using *even claude-written claude-preferred words* instead of the literal words used in literal existing papers. this is a critical strategy to prevent defamiliarization of what algorithms or data structures are being used in which context, so that it is easier to maintain a type-discipline about what types irs are, serialized data is, deserialized data is, serialized schedules are, deserialized schedules are, etcetera

Use the cited source's literal terminology for its algorithms and data structures. Preserve that identification
through IRs, serialization and deserialization, so the representation and its role remain identifiable at each
producer and consumer.

For a request composing other requests in externally cited roles, see the
[example and reusable form](compilation-composition-change-request.md#reusable-form).
For compilation work, that form identifies the properties preserved, representations consumed or eliminated,
and actual residual runtime operands. Its [reference systems](compilation-composition-change-request.md#s8-analysis-results-become-lower-ir-abstract-operations-disappear)
show how solved abstract problems become optimized programs; retaining the original abstraction at runtime is
not the preservation criterion. State performance obligations separately from semantic preservation.
