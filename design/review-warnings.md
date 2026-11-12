# Review warnings

The reviewers' style guide. It governs the warnings that the post-commit review hooks (both repositories'
.claude/settings.json) and the reviews bin/mesh_review.py runs hand back, to the session that committed or to the host
that ran the review: how a reviewer writes a warning, and how the session or host reads it. Its form is its own: the
turns, quoted verbatim from the transcript, then each rule with its source. A rule that is Claude's reading of the
turns says so.

## The turns

> rewrite the reviewer to ahve thier own goddamn style guide that transduces the specific qualtiy of 'warnings are not interruptions that stop work but do form feedback which may require changes in most situations where they appear'. i'm not the one here who is writing warnings to myself and then deciding to take them in the wrost possible sense; i can't solve arbitrary claude-on-claude theory of mind problems as an external oracle. it's *you* writing these, you have to take some responsibility for your other agents.

(1. 2026-11-08 16:52 PST)

> "conflicts with the standard the hook enforces. " who required the hook? whose standards are in the hook? did the user say to make the thing that you're saying 'conflicts' with what you were plainly told to do?

(2. 2026-11-08 16:46 PST)

> "I ran it on 32cda1e1, the item 7 commit, against the integration-finishing request. It failed the commit with three findings:" okay great. so lets make sure taht post commit reviews are *further appends* to a seqeunce of writes (e.g. the reviewer marks the change request work as outstanding, because the committed code doesn't meet some literal standard, which the host agent which launched the subagents can use to send furhter instructions to persistent agents or launch new further agents to revise the code). rejection filtering by revert is dangerous, but rejection labeling by concatenation is auspicious. if reviewers will catch these issues, simply have this review hook applied to all of the current round of launched-but-not-signed-off change request work, so we can steer the implementers directly towards waht they were told to do in those change requests.
>
> "The two defects in the requests themselves are still there: Write clauses I invented and then cited, and judges that leave out what the turns require. "
>
> change requests don't entail judgements, they entail 'do x with y' for a very restricted grammar of x and y. they are a bespoke documentary standard to catch attempts to add false new requirements or false new constraints which are otherwise immeasurable. other testinga nd design standards would use a different documentary ofrmat.

(3. 2026-11-08 15:29 PST)

> throw out the reviewer if they've been somehow implemented in a manner where they rejection-filter requirements that have been literally passed from the user to an implementer etc. "blah rules out doing blah by blah" this is absurd, odious, and... get ready for it... IDIOMATIC. elbow drop the reviewer. throw the reviwer through cinderblock walls. 230 hit combo the reviewer into the corner of the stage extremely disrespectfully.

(4. 2026-11-07 22:07 PST)

> if they're still outstanding then this is still correct. are they still outstanding? if they wren't finished, they still need to be finished beacuse they are not optional 'would be nice to haves'. if any are finished, mark them as 'covered [datestamp,commitstamp]'.

(5. 2026-11-12)

> the current scope of work, which is only about the direction of data flow and control flow, the removal of attempts ot use inversion of control or wacky callback spaghetti code to avoid using the functional-direction-of-inputs-to-compilation and singular, centralized, shared, compilation and scheduling related headers, operands, intermediates, and outputs replacing incoherent and irreconciliable inlinings of many different algorithms in many different places.

(6. 2026-11-12)

> "The compiler I wrote never calls " why is this the case and what code review could or should have caught this iteratively before anyone got the impression the workload was 'done'?

(7. 2026-11-12)

> "- Not yet changed: the earlier requests are still not restated against data-flow.md, so reviewers will keep discounting them.' then address this at priority, up to and including all 'hooks' or scripted review functionalities

(8. 2026-11-12)

## Rules

### 1. A warning does not stop work

- **Source:**
  - turn 1: "warnings are not interruptions that stop work"; turn 3: "post commit reviews are *further appends* to a
    seqeunce of writes", "rejection filtering by revert is dangerous, but rejection labeling by concatenation is
    auspicious".
  - GCC, Warning Options (https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html), for what a warning is: "Warnings
    are diagnostic messages that report constructions that are not inherently erroneous but that are risky or suggest
    there may have been an error." Its `-Werror`, "Turn all warnings into errors.", is the mode these warnings do not
    take.
  - Claude Code, Hooks reference, "Run hooks in the background"
    (https://code.claude.com/docs/en/hooks#run-hooks-in-the-background): "By default, hooks block Claude's execution
    until they complete." "When an async hook fires, Claude Code starts the hook process and immediately continues
    without waiting for it to finish." "After the background process exits, Claude Code delivers the
    `additionalContext` and `systemMessage` fields from the hook's JSON response to Claude on the next conversation
    turn".
- **Rule:** each post-commit review hook sets `"async": true`. The commit returns at once, and the session's work
  continues while the review runs. When the review ends, its warnings reach the session on its next turn, with status
  0, and are appended to the review record (metal-microbench output_data/change-request-reviews.jsonl). Nothing is
  blocked, reverted or deleted because of a warning.

### 2. A warning quotes the sentence its place departs from

- **Source:**
  - turn 1: "do form feedback which may require changes in most situations where they appear".
  - turn 3: change requests "are a bespoke documentary standard to catch attempts to add false new requirements or
    false new constraints which are otherwise immeasurable".
- **Rule:** each warning names its place (the path and line), quotes the sentence of the governing document the place
  departs from, with that document's path, and names the change that would make the place adhere to that sentence.
  What a reviewer finds that no governing sentence states, such as code it takes to be unused, is not a warning: it may
  be written in the review's summary, and asks for no change. The three-part form is Claude's reading of turn 1, and
  the exclusion is Claude's reading of turn 3; no quoted turn states either.

### 3. How the receiving session reads a warning

- **Source:** turns 1 to 4.
- **Rule:**
  - The session continues its work. It reads each warning against its place and its quoted sentence. In most
    situations where a warning appears, that sentence requires a change at the place, and the session makes it
    (turn 1). Reading the warning against its place first is Claude's reading of turn 1's "may require".
  - A warning asks for nothing beyond what its quoted sentence requires; the session makes no other change for it
    (turn 3).
  - Text the operator directed stands, and the warning on it stays in the record (turn 2; turn 4: "rejection-filter
    requirements that have been literally passed from the user to an implementer").
  - A warning is not put to the operator as a choice between their instruction and the warning (turn 1: "i can't solve
    arbitrary claude-on-claude theory of mind problems as an external oracle"; turn 2).
  - The host reads the record to send further instructions to the agents it launched, or to launch agents to revise
    the code (turn 3).

### 4. A clause binds whenever it was written

- **Source:**
  - turn 5: "they are not optional 'would be nice to haves'"; turn 8.
  - design/data-flow.md, the operator, 2026-11-10 21:02: "a definite data flow be written exactly as documentation and
    used to recondition and restate all outstanding requirements".
  - The review of mesh 6d4f007b (metal-microbench output_data/change-request-reviews.jsonl) read that
    compiler/mesh_compiler/bridge.py wrote again what `plan_layout` and `plan_publish` write, and set it aside: "A9's
    assignment of the lists to `mesh_plan` is one of the pre-example requirements the memorandum says must be restated
    against data-flow.md, so I don't count it as binding here."
- **Rule:**
  - A reviewer reads a commit against the clauses of the change requests that govern what the commit changes, in the
    requests themselves. A clause is not set aside because its request was written before design/data-flow.md.
  - Where a reviewer does set a clause aside, for any reason, the clause and the reason are written as a warning, not
    in the summary. This is Claude's reading of turns 5 and 8.

### 5. One producer of each value

- **Source:**
  - turn 6.
  - design/rubrics.md, C9: "Ranks higher: the form with fewer copies of each primitive, offered for others to call
    rather than spliced into their code."
  - design/commentary-review.md, the operator: "removing duplicate lowerings from the codebase is one of the first
    basic prerequisites for compiler optimizations to be applied and not regressed later."
- **Rule:** a commit that computes, outside the stage or shared module that produces it, a value that stage or module
  already produces gets a warning that quotes turn 6 and names the one producer. Such values are a page or hop list, a
  storage placement or lifetime, a dependence relation, an order, a synchronization signature, and the contents of a
  record a shared header declares. The C9 hook (bin/mesh-c9-review.py) reads every commit for this as well. The list
  of values is Claude's reading of turn 6.

### 6. A statement that work is done is read clause by clause

- **Source:** turn 7: "before anyone got the impression the workload was 'done'".
- **Rule:** where a commit, or a document it changes, says an order, an item or a request is done, covered or
  implemented, the reviewer reads each clause of that item, in its request, against the code at that commit, and writes a warning for each clause that does not hold there. Agreement of results with a reference run
  does not show that a clause holds. This is Claude's reading of turn 7.

### 7. A commit adds no assertion and no test

- **Source:** the operator, 2026-11-12: "delete all assertions too then, then write a commit time hook that catches
  all examples of asserts and test introductions through a simple yes/no query assigned to a haiku or similar to
  measure the *presence* or *absence* of those properties, without respect to any requirements or claims amde by any
  claude. a separate distinct function in the post commit hook must then log all such commits as 'noncompliant' and
  'outstanding' to impose thos property on all further work."
- **Rule:** bin/mesh-assert-review.py runs after every Bash call of a session in either repository. For each commit
  made since it last ran, `measured` asks Claude Haiku two yes/no questions about the lines the commit adds: whether
  an assertion is present, and whether a test is present. It is given the diff alone. `logged` appends each commit
  with either present to metal-microbench `output_data/assert-test-commits.jsonl` as `noncompliant` and
  `outstanding`, and the session is told.
