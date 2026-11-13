# Memorandum: review-hook provenance and evidence, 2026-11-13

Style guide: [Google developer documentation style guide, highlights](https://developers.google.com/style/highlights),
for clear language, descriptive links and code formatting.

## Commission and method

The operator, 2026-11-13:

> might as well do a meta review of all of the other review hooks and review scripts the claudes were applying then... just in case...

In the preceding turn:

> review the claude canon to see whether this is a superfluous requirement or an extrapolated and incorrect property that does not itself come from a cited source, e.g. a user transcript turn, a textbook, a blogpost, whatever. if this is an attempt by claudes to add overconstriants, it has to be removed without replacement, no matter what 'breaks' from the absence of the excess constraint.

> a claim that's true about code can only remain true about code by being the result of an end to end run of that code through its sanctioned calling paths and sanctioned input domain, (which tragically makes unit tests just as dangerous and untrustworthy as self-referential inner citations).

Codex inspected mesh `4845b55854d43d3b4765cf435a2bcffa29d9e0c4`, metal-microbench
`928a9cd185ddf501ea5b84982f24c0a5a66f39f2`, their Git hooks and Claude settings, the referenced scripts,
their introduction commits, and the originating transcript. The existing edit to `placement.py` was excluded.
This memorandum records that inspection and the accompanying revisions. It is not a standing specification,
an implementation-completion ledger, or evidence that later versions behave the same way.

The transcript cited below is session `55974018-f56d-4c8b-8973-6fc61b49140c`, stored under
`~/.claude-personal/projects/-Users-mdot-metal-microbench/`. Turn identifiers identify its messages rather than a
later prose paraphrase. The source findings below are not claims that every path was executed in a live agent session.

## What was connected at the inspected revisions

Both repositories' settings named the same mesh scripts: `mesh-brief-gate.py` before assignments,
`mesh-agent-mark.py` before Bash, `mesh-told.py` asynchronously after Bash, and `mesh-stop.py` at Stop and SubagentStop.
Both Git post-commit paths resolved to mesh `bin/post-commit`, which launched `mesh-post-commit.py`.
`mesh-item-tally.py` was a separate command, not an event hook.

The older assertion, claims, quotation, commentary, style, change-request, graph and comment reviewers had been
deleted by `b727c7cc` and `a98a4596`; their replacement questions were in `mesh-post-commit.py`. No old reviewer
was named by the two inspected project settings. Their behavior was examined through their replacement and history,
not assumed to remain active from documentary references.

Two files named `pre-commit.countermanded-20260922` remained under `.git/hooks`. Their source imposed document
whitelists and documentation-growth/source-deletion ratios. Their names were not Git's `pre-commit` entry point;
this review did not reactivate or apply them. The separate `~/.claude/settings.json` named `no-pytest.sh`, whose Bash
text matcher returned status 2. The inspected `~/.claude-personal/settings.json` had no such hook. This inspection
did not establish that the separate global-profile hook was used in the personal sessions, and did not modify it.

## The requested measurements and the additions

The operator's turn `ff5f94d0-1383-4baa-a99a-bd8b9f2f56d7`, 2026-11-12:

> " Everything else about it is the opposite of the Haiku graders:" then what properties must it have to be correctly scoped and applied only to commits which relate to a change request document (not all actions in the repo must use the change request document model, which is being formalized purely to remove several previous attempts to create ungoverned side chanenls which a: avoid oversight b: are consistently used for bad behavior and wasteful indiscretions, not for good behavior and handling surprises with laudable discretion)

Claude's next proposal, turn `f8e91fc6-2e7e-4197-8901-5a1911821055`, specified a commit naming its document and item,
the named document and diff as inputs, fixed questions, quoted clauses, fresh context, append-only records without
blocking, and no authority conferred by the result. The operator's turn `bb406ba5-cfb5-49f4-8aac-8b878c9a6629`:

> rewrite the review hook to have those seven properties or whatever idk

That approval supplies provenance for those seven properties. It does not supply index membership, a minimum
clause length, a punctuation grammar, or a cumulative completion score. Those appeared in the implementation.

| Finding in the inspected source | Origin and distinction | Revision made in this audit |
| --- | --- | --- |
| The item reviewer skipped documents absent from `change-requests-index.md`. | `9188bac5` added the filter after the seven-property proposal; `b727c7cc` retained it. The approved proposal required an explicitly named document, not membership in an index. | Removed the filter, without an alternative registry or eligibility test. |
| Briefs required a tracked, unmodified file and an agent description equal to its basename; Workflows had additional naming restrictions. | `b727c7cc` added these while citing the request for per-commit records. The file-reading instruction in turn `588cf13d-e088-463a-b8fa-30214246ab5e` did not impose those properties. | Removed those conditions. Retained the file-direction gate; paths containing spaces are no longer excluded by its path pattern. |
| A web style-guide URL was searched as a Git path, then reported as a nonexistent repository file. | The style instruction cited by `e732f74a` required a cited guide and adherence; it did not require a local guide. `b727c7cc` implemented only Git-file retrieval. | Added retrieval of cited HTTP(S) guides, with the retrieved source and content digest recorded. Retrieval failure is reported as an unmeasured comparison, not noncompliance. |
| Item text was split on punctuation, fragments of 20 characters or fewer were dropped, and the fragments received sentence numbers. | `b727c7cc` introduced this to stabilize counts; the approved proposal instead requested clauses quoted from the item. | Removed the splitter and length threshold. The reader receives the complete named document and returns quoted clauses. The record identifies the document revision and blob. |
| The tally unioned historical sentence numbers across commits and compared that union with the largest sentence count. | `a98a4596` added the tally. A sentence number had no identity across document revisions; a later reversal did not subtract earlier coverage. | Removed the coverage aggregation. The command prints individual historical records with their commits. |
| The tally treated names in backticks as symbols and searched basenames across non-Markdown files. | `14c4736a` generalized a search described for one assignment into a parser for every item. A lexical occurrence or absence did not establish a sanctioned execution path. | Removed the generalized symbol search without replacement. |
| Stop-report claims were counted only when they omitted a commit name. | `14c4736a` introduced this proxy. A named commit did not make a behavioral claim verified. | Count claim presence independently of whether a commit is named. The output does not certify truth. |
| A process-wide file held whichever agent had most recently made a Bash call; commits read it as their authoring agent. | `14c4736a` added `mesh-agent-mark.py`. Concurrent calls could replace that file before the commit read it. | Removed the marker, the two settings entries and the inferred agent attribution. Retained process and launch records without pretending they identify the authoring agent. |
| A commit naming an item not found in the process's exact-item launch strings generated a warning. | `a98a4596` added it. A direction to read a complete document did not match a later item reference, and process history was not individual-agent assignment history. | Removed the inferred mismatch warning. The actual launch records remain available for inspection. |

The retained brief syntax is an implementation form for the file-reading instruction, not evidence that the contents
of a named file are correct. Neither a clean Git state nor inclusion in an index could establish that correctness.

## Evidence and delivery defects

The operator, turn `4473614a-f2e9-4fb6-aacf-2c0ea5cab0d4`, requested separate presence measurements:

> write another commit hook to, yet again, produce yet another append-only log for whether a document is inlining claims of fact or claims of requirement & whether there is any citation to a block quoted verbatim user turn. these are three different bits, and also are haiku-judgeable from a commit diff alone. a separate, distinct, script, as with the previous stage, must report an appended warning for every commit which is taking such an action, as factual claims without source, like unit tests or asserts, are strongly confirmatory proof that agents are going 'feral' and 'totally ignoring the scope of proper sanctioned work in the repository as usual'.

The inspected implementation had further limitations and errors:

- **Quotation lookup overstated its result.** It split quotations at ellipses, ignored fragments shorter than 12
  characters, and searched a cache containing arbitrary nested transcript strings. The reporter said no typed turn
  contained missing text. The revision searches the quoted passage within individual human-message contents,
  removes the arbitrary threshold and fragment splicing, and records the searched transcript patterns. An unlocated
  passage means only that this search did not locate it. A match may still be the user quoting someone else; it does
  not establish authorship, endorsement or an instruction's scope.
- **Prompt text was exempt from claims measurement.** The inspected questions exempted program data including text
  asked of a model. The revision includes governing claims and requirements in reviewer/implementer prompts while
  distinguishing ordinary runtime messages. The requested presence measurement does not validate a claim merely
  because its file also contains a quotation.
- **Diff judgments were rendered as facts.** The source reader saw a diff, not every producer and consumer in the
  repositories. Its flags are now labelled as that reader's findings. Item feedback does not present counts as
  current implementation coverage or authenticate the governing prose. Truncated inputs are reported in feedback.
  Formal proofs/algebra and machine directives are excluded from the comment flag, as allowed by the supplied
  metal-microbench source standard.
- **Line removal was not semantic withdrawal.** The old removal flags compared added and removed wording. Rewrapping
  a quotation or restating a requirement could set them. Their feedback now identifies a textual change and does
  not instruct agents to restore it or claim that the underlying requirement was withdrawn.
- **A failed commit reader suppressed the item measurements.** Item references were populated only inside the
  successful model call. The revision reads explicit trailers before that call, so failure of one measurement does
  not silently discard the other requested measurements.
- **Later measurements could go unreported.** The warning reporter skipped a commit forever once any warning note
  existed, although separate measurement records were appended later. It also examined only the latest 40 commits
  and assumed every historical record had the newest fields. The revision reads measurement-note entries, records
  the digest and length of what it reported, and appends feedback for newly appended records. Older records are read without
  inventing missing fields or translating them into current coverage.
- **Concurrent note writes were unchecked.** The scripts ignored Git write failures while appending to shared note
  refs. Their writes now share one small append operation that serializes writes per repository/ref and checks Git's
  result. This addresses concurrent writers using these scripts, not arbitrary external modifications of Git notes.
- **Stop measurements waited synchronously.** Both settings used synchronous command hooks with a 150-second
  timeout. They now request asynchronous execution. The [Claude Code hook reference](https://code.claude.com/docs/en/hooks#run-hooks-in-the-background)
  describes asynchronous command hooks as unable to veto the completed action. It also documents that unfinished
  async hooks can be cancelled at non-interactive session teardown; this audit does not claim guaranteed completion
  in that case.
- **The Git launcher was not detached into its own session.** It used a background shell group and `nohup`. The
  revised launcher starts a separate session with closed standard streams. The earlier memorandum commit had no
  measurement note when this audit inspected it; that observation alone did not establish the cause of the absence.

## Scope of the corrections

The presence checks requested for assertions, tests, factual claims, requirements, quotations and hook changes remain
measurements. Naming no request item remains recorded; the instruction quoted above explicitly allows work outside
the change-request scheme, so absence of a trailer alone is not a violation. No new requirement registry, whitelist,
completion gate or rule against removing requirements was introduced by these corrections.

The implementation summaries formerly embedded below rules 10 and 14 to 16 of `review-warnings.md` were removed.
Their quoted user instructions remain. This memorandum supplies a dated account of the changes, not a substitute
set of standing orders for future implementing agents.

## Execution record

Python syntax compilation, settings JSON parsing and `git diff --check` completed before the implementation commits:
mesh `8b38d404` and metal-microbench `1db3c620`. Their actual Git post-commit invocations produced measurement notes:
four for mesh's commit and newly added files, and two for metal-microbench's settings commit. Invoking
`mesh-told.py` on those real records returned feedback and appended warning notes, including records using older
schemas. The model invocations in those first runs failed, and the records said they were not measured.

The initial diagnostic discarded stdout when the model command exited unsuccessfully. Capturing that diagnostic on
an explicit second invocation against `8b38d404` identified an expired OAuth session in the default Claude profile.
The operator had identified the personal launcher in an earlier turn:

> i forget can you launch claude agents with the currnet claude code harness and the current codex harness? teh shell script for claudes for this context is `claude-personal` for reference. if you can, try feeding two change request documents to claudes and monitor their behavior

The inspected shell alias resolved to `claude-profile personal`. The wrapper selected `~/.claude-personal` and also
appended standing prompt files. The Git launcher was revised to inherit an explicit `CLAUDE_CONFIG_DIR`, or use the
personal profile when none was supplied. It does not invoke the wrapper's additional prompts: the reviewer retains
the fixed criterion requested in the seven-property assignment. Failed reader diagnostics now include stdout when
stderr is empty.

An explicit invocation against `8b38d404` using the personal profile completed all four requested measurements at
21:29 UTC. The commit reader flagged claims and requirements in the changed prompts. The source reader flagged
the repeated extraction of request trailers; the follow-up revision passes the already extracted value to the
commit reader. The guide comparison retrieved the cited Google page, recorded its content digest, and returned
`ADHERES: NO`. That fixed yes/no instrument supplied no explanation; this record does not claim a passing style
judgment. Running the feedback command again exposed those later records despite its earlier warning note.

No live Agent/Task/Workflow launch or Stop/SubagentStop session was commissioned for this audit. No compiler,
distributed runtime or ML performance claim was evaluated.
