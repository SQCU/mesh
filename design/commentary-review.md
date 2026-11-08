# The commentary review

After each commit to mesh or metal-microbench, bin/mesh-commentary-review.py writes commentary on the documents under
design/ that the commit adds, changes or renames. Writing that commentary is all it can do. It never blocks, delays,
vetoes or redirects the work that made the commit, and nothing reads what it writes as a gate.

## Why

The change-request review it follows (bin/mesh-cr-review.py, deleted by mesh 2d2eefef) exited with status 2. Claude Code
hands that status back to the committing session as a blocking error, so the review's findings acted as vetoes. The
operator, 2026-11-07 22:07: "throw out the reviewer if they've been somehow implemented in a manner where they
rejection-filter requirements that have been literally passed from the user to an implementer". And at 22:10: "have a
subagent write a post-commit reviewer which isn't given any ability to interrupt integratino work or otherwise act
beyond writing commentary wrt committed documents s.t. there are no furhter inversions of authority wrt the results of
the reviewer as a check (rather than you know a line item veto) upon out of order behavior by subagents".

The deleted review's string checks are kept here as observations: quoted turns looked for in the session transcripts,
and quoted words looked for in the retrieval records. Its authority is not kept.

## The hook

Both repositories' .claude/settings.json run it after every Bash call that matches `git *commit*`, beside the C9 and
comment reviews:

```json
{
  "type": "command",
  "if": "Bash(git *commit*)",
  "timeout": 10,
  "statusMessage": "Commentary review of the commit, started in the background",
  "command": "/opt/homebrew/bin/python3 /Users/mdot/dox/mesh/bin/mesh-commentary-review.py >/dev/null 2>&1; exit 0"
}
```

The hook copies its payload to an unlinked temporary file and starts the review from it as a new session (setsid), its
output to /dev/null. Then it exits 0. It reads nothing else and prints nothing, so it returns in well under a second.
The redirection and `exit 0` keep the status 0 and the output empty even if Python itself fails. The background
process lowers itself to the Darwin background band, then finds the repository and commit the way the C9 review does
(`made` in bin/mesh-c9-review.py).

## What it reads and writes

It reads only what a commit holds, through git cat-file, ls-tree, show and log, with GIT_OPTIONAL_LOCKS=0. It also reads
the operator's turns and Claude's text in the top-level transcripts of the two projects,
~/.claude-personal/projects/-Users-mdot-metal-microbench and -Users-mdot-dox-mesh. It never reads the working tree, so
untracked files and build products are never read. Nor does it read tailscale.key, NODE-STANDUPS.md, keys/, vendor/
or docs/facts.md. It runs no git command that writes, and it has no network access except the optional `claude -p`
read below.

It writes only to its store, review-commentary/ in each repository's git directory:

- `<full sha>.md`: the commentary on one commit;
- `INDEX.md`: one line for each commit read, newest last;
- `log`: what failed, with its traceback where there is one. A failure never changes the exit status.

The store is files under .git/ rather than git notes, because a notes write changes the repository the session is
committing to. It takes the lock of refs/notes/review, writes objects and commits, and can start `git gc --auto`.
Files under .git/ are outside the working tree, so `git status` never shows them, and no git state changes when they
are written.

## The observations

Each observation points at a line of the document as the commit has it. Lines the commit changed are marked
"(this commit)".

1. Quotations. Each quoted span of 20 or more characters, cut at marks of omission, is looked for verbatim. The places
   are: an operator turn in the two projects' transcripts, the document's own quoted turns, a retrieval record
   (design/*-sources-*.md, replay-grounding-*.md, packing-and-coflows-*.md), design/change-requests.md, or a tracked
   source file of either repository. A change request's turns, the block quotes under "## The turns", are looked for
   among operator turns only. When the time after a turn differs by more than a minute from that turn's time in the
   transcript, the commentary says so. When a span is found only in Claude's own text in the transcripts, or only in
   another document under design/, it says where. Quotations in a retrieval record are not looked for, since the
   record is where they were read.
2. Statements and the quotations of their section. For a change request, it lists, for each sentence of an item's
   heading, Write and Written parts and unlabelled paragraphs, the words that no quotation in the same item contains.
   Common words are left out, and word endings are dropped before comparing. The Replaces, Keeps, Source and Stop
   parts are not examined. For other documents, one line for each section gives the lines whose sentences have such
   words. Sections that quote nothing are listed together.
3. Code locations and commits. It checks file paths, `path:line`, `(:line)` after a backticked name, and commits named
   with their repository. A path is read at the commit its line names, by repository name or by a hash only one of
   the two repositories holds ("before" names that commit's parent). Otherwise it is read at the commit the document
   declares ("Code is cited at mesh … and metal-microbench …"). Otherwise it is read at the reviewed commit, with the
   other repository at its last commit before that one. The commentary notes a path in
   neither repository, with the commit that deleted it, and a line past a file's end. It also notes a backticked name
   that is not at the cited line, with where it is, and a named commit that does not exist.
4. Requirements and the operator's turns, for change requests. It notes an item of "## Requirements" that quotes
   operator words "## The turns" does not hold. It also notes an item that quotes no operator turn, with the turn
   whose words its heading and Write share most.
5. Model commentary, for change requests up to 200,000 characters. Claude Sonnet reads the standard, the document and
   the observations above, with a limit of 600 seconds. The call is `claude -p --model sonnet --tools ''
   --setting-sources user --no-session-persistence --strict-mcp-config`, run in the background process with no Claude
   Code session variables in its environment. Its text goes into the commentary verbatim, under its own heading. A
   model's reading differs from run to run.

## Reading it

`.git/review-commentary/INDEX.md` lists the commits read. Each line names the file holding that commit's commentary.
To write the commentary on any commit by hand: `python3 bin/mesh-commentary-review.py --commit mesh <rev>`. Use
`metal-microbench` for that repository, and add `--strings` to leave out the model read.
