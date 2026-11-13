# Review warnings

The turns, quoted from the transcript, then each rule with its source.

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
- **Rule:** nothing is blocked, reverted or deleted because of a warning.

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
  - For vocabulary findings, follow the operator's clarification in [the change-request standard](change-requests.md):
    ordinary grammatical variation is allowed; a wrong name with the right function is noted and renamed in
    successive commits while the assigned work continues. It is not grounds to reject a work order.

### 7. A commit adds no assertion and no test

- **Source:** the operator, 2026-11-12: "delete all assertions too then, then write a commit time hook that catches
  all examples of asserts and test introductions through a simple yes/no query assigned to a haiku or similar to
  measure the *presence* or *absence* of those properties, without respect to any requirements or claims amde by any
  claude. a separate distinct function in the post commit hook must then log all such commits as 'noncompliant' and
  'outstanding' to impose thos property on all further work."

### 8. A document's claims are measured

- **Source:** the operator, 2026-11-12:

> write another commit hook to, yet again, produce yet another append-only log for whether a document is inlining claims of fact or claims of requirement & whether there is any citation to a block quoted verbatim user turn. these are three different bits, and also are haiku-judgeable from a commit diff alone. a separate, distinct, script, as with the previous stage, must report an appended warning for every commit which is taking such an action, as factual claims without source, like unit tests or asserts, are strongly confirmatory proof that agents are going 'feral' and 'totally ignoring the scope of proper sanctioned work in the repository as usual'.

### 9. A quoted turn is found where it was typed

- **Source:** the operator, 2026-11-12:

> now was this something the user said to do or is htis yet another thing that requires post commit hooks to act as syntax checkers or parity checkers to ensure that change requests aren't forged by subagents or briefly unattended agents...?

### 10. A brief is a direction to read a file

- **Source:** the operator, 2026-11-12:

> is there any mechanism e.g. a system prompt to induce subagnets to immediately fail and log to a file as a warning if they are given 'agent briefs' or any inlined instructions or contents *at all* besides directions to read files using file read tools

The operator, 2026-11-13:

> review the claude canon to see whether this is a superfluous requirement or an extrapolated and incorrect property that does not itself come from a cited source, e.g. a user transcript turn, a textbook, a blogpost, whatever. if this is an attempt by claudes to add overconstriants, it has to be removed without replacement, no matter what 'breaks' from the absence of the excess constraint.

Provenance review, 2026-11-13: `e646c57470edad1817862e7a81743d3a18c1a024` introduced the gate with the
2026-11-12 file-reading instruction above. `14c4736a660953a494b2f3b4b1601aa513d1690b` added index membership
to the gate and rule 16, citing that rule's instruction to finish omitted work. Neither quoted instruction requires
index membership. The transcript records that instruction in user turn `118b042c-4931-4fd3-894f-0d148126a795`
and Claude's announcement of the additional restriction in assistant turn `5ab8d99a-4980-493a-9933-8df0767fe26d`, in session
`55974018-f56d-4c8b-8973-6fc61b49140c`. This audit classified the restriction as an extrapolation and removed it
without replacement, following the 2026-11-13 instruction.

The earlier gate description also quoted the operator's turn of 2026-11-08:

> why are you 'descirbing' anything in 'briefs' instead of literally saying 'there is a change request document. read it and act on the topic.'

### 11. The change request measurement

- **Source:** the operator, 2026-11-12:

> rewrite the review hook to have those seven properties or whatever idk

### 12. The measurement of a change to a design document

- **Source:** the operator, 2026-11-12:

> apply the same approach to each or whatever

### 13. The measurement of a new file's style guide

- **Source:** the same turn, and the operator, 2026-11-09:

> add a new post-commit hook for catching claudes adding non-executable files to the repo which have no style guide or cite a style guide but do not adhere to it, which must produce a post-commit warning that claudes are doing decadent and deceptive things as usual. existing nonexecutable files are grandfathered for now because i do'nt want to have a cutesy dialogue about this.

### 14. One record for each commit

- **Source:** the operator, 2026-11-12:

> why is a post commit hook not always writing an output wrt a specific commit? who put an iterator into any of these?

> post commit hooks souldn't be string typed like this. preventing claims and pseudodocumentation and tests from appearing is unconditional.

### 15. Who was given what, and what an item has had

- **Source:** the operator, 2026-11-12:

> how in particular does this model keep subagents responsible for naming the scope of work they're attacking and citing it? what mechanisms are missing or overloaded?

> "I have not built these." what are you being told to write?

> why do these still exist in this form then?

### 16. What an agent leaves and says

- **Source:** the operator, 2026-11-12:

> it is in the spirit of this exercise to identify what you didn't do, then fix this immediately, wrt this module, instead of going into ever finer grained detail of 'why why why why'.

## Dated audits

The [2026-11-13 meta-review](review-hooks-audit-2026-11-13.md) records the inspected hook revisions,
transcript provenance, corrections and validation limits. It is a historical review, not authority for requirements
or evidence of later execution. The implementation descriptions formerly below rules 10 and 14 to 16 were removed
in that review; their quoted operator instructions remain above.
