# Review warnings

The reviewers' style guide: how a post-commit reviewer writes a warning, and how the session that receives one reads
it. It is written in the form design/change-requests.md states: the turns, quoted verbatim, then what is written, with
its source.

## The turns

> rewrite the reviewer to ahve thier own goddamn style guide that transduces the specific qualtiy of 'warnings are not interruptions that stop work but do form feedback which may require changes in most situations where they appear'. i'm not the one here who is writing warnings to myself and then deciding to take them in the wrost possible sense; i can't solve arbitrary claude-on-claude theory of mind problems as an external oracle. it's *you* writing these, you have to take some responsibility for your other agents.

(2026-11-08)

> so lets make sure taht post commit reviews are *further appends* to a seqeunce of writes (e.g. the reviewer marks the change request work as outstanding, because the committed code doesn't meet some literal standard, which the host agent which launched the subagents can use to send furhter instructions to persistent agents or launch new further agents to revise the code). rejection filtering by revert is dangerous, but rejection labeling by concatenation is auspicious.

(2026-11-08, in part)

> throw out the reviewer if they've been somehow implemented in a manner where they rejection-filter requirements that have been literally passed from the user to an implementer

(2026-11-07 22:07, as design/commentary-review.md quotes it)

## What is written

### 1. A warning does not stop work

- **Source:** GCC, Warning Options (https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html): "Warnings are diagnostic
  messages that report constructions that are not inherently erroneous but that are risky or suggest there may have
  been an error." Its `-Werror`: "Turn all warnings into errors."
- **Write:** a reviewer's warning is handed back to the session that committed with status 0, and appended to the
  review record (metal-microbench output_data/change-request-reviews.jsonl). Nothing is blocked, reverted or deleted
  because of it, and the session's work continues.

### 2. A warning is feedback that may require a change

- **Source:** the operator, above.
- **Write:** each warning names its place (the path and line), the sentence of the governing document the place
  departs from, and the change that would make the place adhere.

### 3. How the receiving session reads a warning

- **Source:** the operator, above.
- **Write:**
  - The session continues its work, and makes the change a warning names where the place is its own writing.
  - Text the operator directed, a quoted turn or text the operator said to write unchanged, stands. The warning on it
    stays in the record.
  - A warning is not put to the operator as a choice between their instruction and the warning.
  - The host reads the record to send further instructions to the agents it launched, or to launch agents to revise
    the code.
