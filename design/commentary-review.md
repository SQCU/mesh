# The commentary review

The change-request, document-commentary and new-file style reviews use bin/mesh_review.py. It runs a tool-using
Claude Code session, supplies the commit's identity and context locations, and records the investigation and its
result. The model chooses what evidence to read and how to interpret it. The harness transports that evidence and
records feedback; it does not decide documentary meaning by parsing sentences, headings or vocabulary overlap.

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

## Investigative review

The review adapter supplies the assignment in the system prompt, separately from the evidence. That assignment
defines scope; the presence of a commit manifest does not assign a broader task. The harness supplies the target
repository, full commit identity, parents, commit message and changed paths. It also
names both repositories and their heads when the review starts, the registered context, the invoking session's
transcript when supplied by the hook, and the two projects' transcript directories with their top-level file lists.
The existing reviewer style guide (design/review-warnings.md), change-request index, change-request standard, rubrics,
principles and measurement document are discovery starting points. The style guide is read in its own scope; its
interpretation is not encoded in a sentence parser or a fixed warning schema.

The reviewer has Read, Grep, Glob, WebFetch and WebSearch, and the read tools of the upstream
[MCP Git server](https://github.com/modelcontextprotocol/servers/tree/main/src/git). It can inspect a commit or a file
at a revision with git_show, compare revisions, follow history, read original turns, examine callers and callees,
and retrieve cited sources. It reads the governing documents as evidence whose meaning and provenance can themselves
need review. It can consider supporting evidence, counterevidence, inherited defects and companion changes.

The session runs through the existing [Claude Code CLI](https://code.claude.com/docs/en/cli-reference), in restricted
mode with explicit read tools, no shell or editing tools, no skills, no inherited hooks, and only the two configured
Git servers. The servers' mutation tools are denied. The Git server is the published mcp-server-git 2026.11.4 package,
launched through uvx; uv caches it. Claude authentication follows the invoking environment, including
CLAUDE_CONFIG_DIR and authentication environment variables. The harness removes parent-session identity and cmux
integration variables. It runs in a temporary directory without repository instruction auto-discovery.

Built-in file reads can see current working files and existing measurement artifacts. The prompt distinguishes those
from the reviewed revision and asks for committed evidence when making historical claims. Git tools can retrieve
historical content without checking out, copying or mutating source. Missing evidence is reported as a limitation.
The prompt directs compiler review toward actual lowering use, duplicate paths, fallback paths and implementation
coverage, with synchronization and performance observations informing that review. It starts no benchmarks.

The output envelope holds a prose summary, prose findings, prose limitations and the model's assessment of whether
corrective work is outstanding. There is no prescribed sentence grammar, vocabulary test or PASS token. The program
does not derive the assessment from a word, a count of observations or the existence of a limitation. No registration does not itself establish a
defect. The shared registration file remains available as context, without claiming to be a per-agent assignment.
A registered revision is a pointer the reviewer can investigate, not the exclusive or automatically valid standard.

A review has up to 880 seconds, retaining the existing hook time budget. A reader failure or invalid response is
recorded as incomplete, with the partial trace retained. The model can conclude with specific limitations when
evidence is unavailable. Recoverable tool errors remain in the trace; they do not themselves decide the review.

## Scopes and records

- bin/mesh-change-request-review.py reviews the commit's code and documentary changes together. Its register and clear
  commands maintain context hints. `review-commits <repo>:<revision>[=<document>,...]` runs a historical review; the
  model reads the commit message itself rather than a parser deciding which documents it is allowed to use.
- bin/mesh-commentary-review.py reviews changed documents under design/, including deletions and renames. The hook
  still starts it in the background and returns immediately. `--commit <mesh|metal-microbench|path> <revision>` runs it
  explicitly. The former --strings mode and its separate interpretation machinery are removed.
- bin/mesh-style-guide-review.py retains its scope of newly added non-executable files, excluding implementation
  source and makefiles. Existing files remain grandfathered. The model finds and reads the applicable guides; it is
  not restricted to Markdown links extracted from the new file. The requested warning accompanies actual findings;
  an incomplete read is identified as incomplete.

Each run appends its outcome to metal-microbench/output_data/change-request-reviews.jsonl. The record includes the
review kind, commit, context hints, summary, findings, limitations and trace path. The model sets `outstanding`: true
for identified corrective work, false when the assigned review finishes without it, and null when it cannot reach a
conclusion. A transport or reader failure is recorded as null. This is a record of the review, not a proof of implementation compliance. Historical records remain.
The outstanding command keeps review kinds and historical per-file style records separate.

The complete request manifest, review prompt, model tool calls, tool results and final response are retained in a
unique `.git/review-commentary/<commit>.<kind>.<run>.jsonl` for each run. Document commentary also updates the existing
`<commit>.md` view and appends a line to INDEX.md there. Repeated runs retain their individual traces and journal
records. Background invocation errors go to that directory's log.

## Post-commit findings

The C9, comment, change-request and style hooks run in the background (`"async": true`,
[Run hooks in the background](https://code.claude.com/docs/en/hooks#run-hooks-in-the-background)): the commit returns
at once, and the session continues while they read. They deliver feedback with status 0 through
`hookSpecificOutput.additionalContext` and `systemMessage`, using the
[PostToolUse interface](https://code.claude.com/docs/en/hooks#posttooluse-decision-control), which Claude Code hands
to the session on its next turn. Findings are material for
review and revision. A reviewer finding alone does not authorize rollback, deletion of required behavior or a
replacement specification. An incomplete review stays visible. The hook's successful exit means feedback delivery,
not implementation approval. Explicit comment-review commands retain their diagnostic exit statuses.

Both repositories' existing .claude/settings.json entries continue to invoke these scripts. The document-commentary
hook remains detached; the C9, comment, change-request and style hooks return feedback to the invoking session on its
next turn. C9 and source
comment review retain their focused implementations.
