# Review warnings

The reviewers' style guide. It governs the warnings that the reviewers bin/mesh_review.py runs hand back to the
session that committed: how a reviewer writes a warning, and how that session reads it. Its form is its own: the turns,
quoted verbatim from the transcript, then each rule with its source. A rule that is Claude's reading of the turns says
so.

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
- **Rule:** a reviewer's warning is handed back to the session that committed with status 0, and appended to the
  review record (metal-microbench output_data/change-request-reviews.jsonl). Nothing is blocked, reverted or deleted
  because of it, and the session's work continues.

### 2. A warning is feedback that may require a change

- **Source:** turn 1: "do form feedback which may require changes in most situations where they appear".
- **Rule:** each warning names its place (the path and line), the sentence of the governing document the place departs
  from, and the change that would make the place adhere. This three-part form is Claude's reading of turn 1; no quoted
  turn states it.

### 3. How the receiving session reads a warning

- **Source:** turns 1 to 4.
- **Rule:**
  - The session continues its work. In most situations where a warning appears, it makes the change the warning names
    (turn 1).
  - Text the operator directed stands, and the warning on it stays in the record (turn 2; turn 4: "rejection-filter
    requirements that have been literally passed from the user to an implementer").
  - A warning is not put to the operator as a choice between their instruction and the warning (turn 1: "i can't solve
    arbitrary claude-on-claude theory of mind problems as an external oracle"; turn 2).
  - The host reads the record to send further instructions to the agents it launched, or to launch agents to revise
    the code (turn 3).
