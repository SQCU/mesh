# The commentary review

The hooks are described in [review-warnings.md](review-warnings.md), rules 14 to 16.

## Why

The earlier change-request reviewer (bin/mesh-cr-review.py, deleted by mesh 2d2eefef) returned status 2. The commit had
already happened, but agents treated the error feedback as a veto. The operator, 2026-11-07 22:07:

> throw out the reviewer if they've been somehow implemented in a manner where they rejection-filter requirements that have been literally passed from the user to an implementer

And at 22:10:

> have a subagent write a post-commit reviewer which isn't given any ability to interrupt integratino work or otherwise act beyond writing commentary wrt committed documents s.t. there are no furhter inversions of authority wrt the results of the reviewer as a check (rather than you know a line item veto) upon out of order behavior by subagents

The operator, 2026-11-08, on the reviewer's treatment of its input documents:

> okay so write the reviewer to be more harness-of-llm-like and less gofai-string-parsing-encoding-as-many-assumptions-as-possible-into-a-broken-ah-ah-parser-forcing-bad-outputs-from-bad-inputs.

On the relation between compiler review and the synchronization mechanism:

> that's a problem for focusd review and revision on compiler lowering strategy; removing duplicate lowerings from the codebase is one of the first basic prerequisites for compiler optimizations to be applied and not regressed later. (the current codebase's design standards cannot protect any optimization from being 'fallbacked' or 'partially called and then never finished in implementatino coverage' later by deviant subagents. hence then eed for the sensitizing mechanism, allowing immediate review and feedback of suspiciously regressive performance without needing to revert days or weeks of all progress.)
