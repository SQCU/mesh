# Change requests

A change request describes what ought to be written. It uses only non-idiomatic descriptions of work: each requirement
cites the reference authors, theory, papers, books or source code that say exactly what to write, and the code is
written as that source states it. Its body is the turns that led to
it, quoted verbatim, and where no source says what to write it says so and invents nothing.

The operator, 2026-11-13, on the meaning of "idiomatic":

> "“Idiomatic” here concerns agents substituting familiar implementation habits or local terminology for a named algorithm, " it actually involves limiting hte propagation of any data structure or any algorithm which is using *even claude-written claude-preferred words* instead of the literal words used in literal existing papers. this is a critical strategy to prevent defamiliarization of what algorithms or data structures are being used in which context, so that it is easier to maintain a type-discipline about what types irs are, serialized data is, deserialized data is, serialized schedules are, deserialized schedules are, etcetera

The operator, continuing the same instruction:

> we also aren't requiring type systems or pydantic or other attempts to encode logical or formal typings in code by any means; the specific kind of type discipline the user emphatically and selectively asks for continues to be using variable names and function names (and so on) which are lifted exactly from specific sources or specific literature; why this choice was made does not need to be justifeid to any implementing agent as it is never their position to supervene the user on simple constraints like these

The operator, clarifying grammatical variation and how to revise existing names:

> "Implementers must follow it without substituting preferred terminology," somewhat clearly we need to prevent this from becoming a nonsense excuse for agents; conjugating, pluralizing, etc. the literal dictionary words literally used in cited implementations, cited papers, etc. is clearly comaptible with the agenda here; inventing alternate metaphorical language is not compatible with the agenda here. idiomaticity requirements are not a funny strategy agents may use to supervene work orders; if they're told to work with something that has a wrong-name but right-function, a `#revise this; idiomatic vocabulary` note in one commit and a second renaming commit is the appropriate response, not an 'agent-level rejection filter wrt work tickets'

Use the words from the specified sources in variable names, function names and other algorithm and data-structure
names. Ordinary grammatical variation, including conjugation and pluralization, is compatible with this
requirement; invented alternate metaphors are not. This naming requirement does not request a type system,
Pydantic models or other formal typing machinery. Implementing agents follow the constraint without requiring a
justification of the operator's choice.

When assigned code has the right function under a wrong name, continue the assigned work, record
`#revise this; idiomatic vocabulary` in one commit, and follow with a renaming commit. A naming defect is work to
revise, never a reason to reject the work order.

For a request composing other requests in externally cited roles, see the
[example and reusable form](compilation-composition-change-request.md#reusable-form).
For compilation work, that form identifies the existing result consumed, derived result produced, decisions
specialized into executable operations, and remaining runtime operands. Its
[reference systems](compilation-composition-change-request.md#s8-analysis-results-become-lower-ir-abstract-operations-disappear)
show how solved abstract problems become optimized programs. The operator's
[14 November clarification](compilation-composition-change-request.md#clarification-2026-11-14-one-compilation-over-retained-results)
requires compact retained results and references while removing repeated interpretation from execution. Use the
shared algorithms across their consumers; leave simple representation choices as implementation choices rather
than expanding the requirements. State performance obligations separately from semantic preservation.
