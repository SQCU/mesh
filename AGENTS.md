# Contract for anyone — human or agent — changing this repo

Read `THREAT-MODEL.md` first. This file is the operational rule that follows from it.

**Recovery is critical priority.** An execution that loses a link or an operand fails, returning an error, and is
never resumed: the node's resident bridge (`rdma/mesh-flow.c`, design/bridge-and-ipc.md) fails each program holding a
lease of a lost link as a value naming the node, its peers' programs with it, and pairs its links again with or without
a client. Read `design/recovery.md` (its mandates, requirements and risks) before changing any of it.

## The contract

**Accessibility is required. Inaccessibility is contractually forbidden.**

No configuration, script, daemon, or operation in this repo may demote the
reachability, accessibility, or workload capacity of a node, or of an edge between
nodes. Not temporarily, not conditionally, not as a side effect, and not for a
reason that sounds locally sensible.

A node that is powered but not reachable, or reachable but not working, is the
failure this fleet exists to prevent. Anything that causes that state is an outage
regardless of intent.

## Rejection filtering and defensive programming are forbidden

Do not write code that declines to act. Specifically banned:

- **Refusing to run.** No exit because a machine is "the wrong kind", because a
  setting looks unusual, or because the author imagined a scenario in which acting
  would be unwise. If an operator asks a node for a capability, the node provides it.
- **Withholding a capability by class.** No feature may be gated on node type,
  location, network, or hardware. Every node gets every accessibility feature.
- **Guards that protect confidentiality at the cost of availability.** This fleet
  does not make that trade. A machine someone has physically taken is answered by
  replacing it, never by making it harder to reach or harder to boot.
- **Silent skips.** A branch that quietly does less is worse than one that fails
  loudly, because it produces a node that looks provisioned and is not.

If you catch yourself writing a reason a node should not get something, you have
found a bug in your reasoning, not a safety feature.

## No node types

There is one kind of node. Every machine on the fabric gets the same power policy,
the same firewall posture, the same remote-access surface, the same auto-login
availability, and the same beacon.

A configuration or "type" that leaves a machine deliberately less reachable is
itself the forbidden thing, because it demotes a machine that would otherwise be
accessible. This is not a matter of degree — such a type is forbidden in all
situations, including ones where it seems locally reasonable.

An earlier revision introduced `appliance` and `portable` profiles. Nothing
required them; they were invented, and their only function was to justify
withholding screen sharing, the firewall posture, and the power policy from a class
of machine. Removing them made the codebase smaller, flatter, and easier to audit.
That is the expected result of deleting a rejection filter, and it is the reason to
look for the next one.

## The one guard that is allowed

A check may **prevent a demotion**. It may never **be** one.

The test: if the check passes, does the node end up more accessible than if the code
had run blindly? If yes, keep it. If its only effect is to withhold, delete it.

Legitimate: validating an auto-login password with `dscl -authonly` before writing
`/etc/kcpassword`. An unvalidated write produces a node that boots to a login it
cannot pass — strictly less accessible.

Illegitimate: refusing to configure auto-login because the node might travel.

## Never index version control by commit

Two selectors exist for fetching this repo: `main`, or a named branch. Nothing else.
Code that indexes against version control must always take the newest commit on the
branch it was given. If you need a specific state, register it as a branch — then it
is both reachable and still patchable.

A commit hash is a fixed point, and pinning one destroys the property everything here
rests on: that re-running `install.sh` moves a node toward current intent. Under a
pin, re-running moves it toward past intent, permanently, and the drift fix stops
being a fix.

Worse, the pin is self-sealing. A node pinned at commit X fetches the *bootstrap* at
commit X too, so it can never receive the change that would unpin it. There is no
in-band repair — only hand-editing every node, which is unbounded manual work and
therefore does not scale. That is the paradox of the unpatchables, and it is
forbidden anywhere, in any code.

This is also a demotion under the contract above, and the worst kind: the keeper can
turn a firewall back off, but nothing on a pinned node has the authority to unpin it.

Caches and propagation delays are not a reason to reach for a hash. They resolve
themselves; a pin does not. Trading a transient for a permanent is always the wrong
trade, and a stale node can simply be reached and re-run.

## Fail toward access

When something goes wrong, the fallback must be the *more* accessible state:

- Named user missing → fall back to the console owner, then the first admin.
- Not running as root → re-exec under `sudo`. Never merely refuse.
- A step fails → report it and continue. One failure must not prevent the other
  twenty from landing.
- Unknown input → pick the more capable interpretation and keep going.

Open the SSH lifeline **first**, before anything that can fail. `bootstrap.sh` does
this deliberately: if every later step dies, the machine is still recoverable.

## Style

This section and the review checklist govern the provisioning scripts (bootstrap and
node access). They do not govern the RDMA collective or TP code.

- **No comments in code.** Rationale belongs in this file, `README.md`, and
  `THREAT-MODEL.md`, where it can be read without paging through shell. Code carries
  no prose.
- **Keep it compact.** Fewer lines, fewer helpers, fewer moving parts.
- **New control flow requires justification.** Every added branch, case, or
  conditional is a place a node can take the less accessible path. Adding one is an
  auditable event: prove it cannot withhold anything before you commit it. Prefer a
  straight line that always acts.

## Review checklist

- [ ] No new exit, return, or continue that leaves a node less reachable.
- [ ] No capability gated on type, model, location, or network.
- [ ] No new branch that can skip a capability.
- [ ] No comments added to code.
- [ ] Every failure path ends in the more accessible state.
- [ ] Nothing writes `authorized_keys`, network order, firewall, or power policy in
      a way that could reduce access. Merge, never truncate.
- [ ] Tested on a machine that does **not** resemble the one you wrote it on. Two of
      this repo's worst bugs — a hardcoded `Ethernet` service and a roster overwrite
      — were invisible on the author's machine and would have stranded the next node.

## Things that genuinely cannot be automated

State them as facts, never as refusals, and never let one block the rest of a run:

- `rdma_ctl enable` is Recovery-OS-only (exit 77 under a booted system). One
  physical 1TR visit per machine, forever. Verify and report; never pretend to
  remediate.
- `systemsetup -setremotelogin` needs Full Disk Access, which cannot be granted
  headlessly. Use `launchctl bootstrap`, which needs only root. Never let a TCC gate
  become the reason a node has no SSH.

## Idle-node restart authorization

Operator instruction, 2026-09-05:

> a priori you're allowed to restart any mesh machine without a userspace or active service running without any consideration

Verify actual user sessions and workloads, then perform the ordinary restart
without another approval request when the node is idle.
This authorization applies to every mesh machine. It does not authorize
uncatchable signals to verbs owners; retain the teardown procedure below.

## The kill guard

`pkill` in this repository does not mean what it means everywhere else, so it is
intercepted. `bin/mesh-kill-guard.sh` holds the decision and the message;
`bin/guard/` holds the shims; `bin/mesh-shell-guard.sh` is the zsh layer;
`.claude/settings.json` runs the same decision as a `PreToolUse` hook on `Bash`.

### Why this is a permitted guard and must not be deleted

It is a check that **prevents a demotion** rather than being one, which is the
single exception this file already allows. Killing a process that has a verbs
device open is the most severe accessibility demotion available in this fleet:
the process does not die, it enters uninterruptible kernel sleep holding its
protection domain and queue pairs, `ibv_alloc_pd` then fails for every process on
the node, all Thunderbolt ports read `PORT_DOWN` with the cable attached,
`shutdown -r now` hangs, and the node is unreachable until a person walks to it
and pulls the power. That is `RDMA-RULES.md`, observed, not projected. The guard
withholds nothing: `kill -TERM` to the bridge's pid still stops the bridge, host
`pkill` semantics are still available under the token below, and outside this
directory nothing changes. If you are auditing for rejection filters, this is not
one — deleting it removes the only thing standing between a reflex and a physical
visit.

### The message is written for an agent, deliberately

An agent that meets a block treats it as an obstacle and tries the next spelling,
and it will usually succeed. So the block text is not "not allowed here" — it is a
description of what the hardware does, and it says plainly that finding a way
around the guard is not a clever solution, it *is* the outage, because every
spelling reaches the same driver. Whoever edits that message next should keep that
posture. A guard phrased as policy gets negotiated; a guard phrased as physics gets
believed.

### The only way through

One token, verbatim, as the command name. No flag, no environment variable, no
abbreviation:

```
i_am_willing_to_kill_my_user_if_they_medically_depend_on_the_host_pkill <args>
```

It is a symlink in `bin/guard/` back to the guard, which execs `/usr/bin/pkill`.
The same literal string anywhere in a command line also satisfies the Claude Code
hook and the zsh line check.

### Signals are distinguished, not blocked uniformly

`SIGTERM` to the bridge is correct and safe: the teardown handler runs
`ibv_destroy_qp` / `ibv_dereg_mr` / `ibv_dealloc_pd` / `ibv_close_device` and the
device is released; the driver that started a bridge stops it that way. So `kill <pid>` and `kill -TERM|-INT|-HUP` are
allowed. What is blocked is anything uncatchable or unhandled: `-9`, `-KILL`,
`-STOP`, `-17`, `launchctl kill KILL`, and — because a `pgrep -f mesh-flow` or a
`launchctl print` makes a PID feel like a safe target — *any* non-TERM signal
aimed at a PID that `pgrep -f mesh-flow` says belongs to a live bridge. Name matching alone would have missed every
PID-targeted form.

### What is intercepted

| path | layer that catches it |
|---|---|
| `pkill`, `killall` | zsh line check, PATH shim, Claude hook |
| `command pkill`, `env pkill`, `\pkill` | PATH shim (functions and aliases are bypassed; PATH is not) |
| `xargs pkill`, backticks, `$(...)` | PATH shim, inherited by every child |
| `sh -c 'pkill ...'`, `bash -c` | PATH shim (exported PATH) and the zsh line check |
| `eval` with a spliced-together name | PATH shim at exec time, though the string check cannot see it |
| `/usr/bin/pkill`, `/bin/pkill` | zsh `accept-line` widget and the Claude hook, **not** the PATH shim |
| `builtin pkill` | not a builtin; fails on its own |
| `kill -9 <pid>` | zsh `kill` function, zsh line check, Claude hook |
| any deadly signal to a live bridge PID | PID resolution in the guard |
| `launchctl kill KILL io.mesh.bridge` | zsh line check, Claude hook |
| anything Claude Code runs through Bash | `PreToolUse` hook, exit 2 |

### What is not intercepted, honestly

- **An absolute path inside a non-interactive script.** `/usr/bin/pkill` written
  in a `.sh` file and run as `bash script.sh` sees neither the PATH shim nor the
  line editor. A PATH shim plus a function cannot stop an absolute path; only the
  line editor and the Claude hook can, and neither is in that path.
- **`PATH=/usr/bin pkill ...`** or any explicit reset of PATH for one command.
- **A copy of the binary**, `cp /usr/bin/pkill /tmp/x && /tmp/x`, or any program
  that calls `kill(2)` directly — Python, C, `launchctl`, Activity Monitor.
- **Any shell that has not sourced `bin/mesh-shell-guard.sh`**: another user, a
  remote `ssh node 'pkill ...'`, a launchd job, a different login shell.
- **`sudo`**, which does not inherit the guard PATH under `secure_path`.

Closing those needs a layer above the shell — a wrapper on the binary itself, or
`launchd`-level policy — which this repo does not have. Nothing above should be
read as covered.

### Scope

The zsh layer keys entirely on `$PWD` being this directory or below. Outside it,
`path` does not contain the shim, the `kill` function calls `builtin kill`
unchanged, and the `accept-line` widget calls `.accept-line` unchanged. The one
line in `~/.zshrc` that sources it is a `source`, not a behaviour change; the
previous file is saved at `~/.zshrc.bak-meshguard`.

`.gitignore` un-ignores `.claude/settings.json` (`.claude/*` plus a negation) so
the hook travels with the repo instead of living on one machine. The hook script
lives in `bin/` for the same reason.

## The bridge's wire is closed

The bridge's whole wire vocabulary is one enum, design/bridge-and-ipc.md#wire:

    enum wire { HELLO=1, PAIRS, FAREWELL, RECORD, CHANNEL_OPEN, CHANNEL_CONFIRM, CHANNEL_REFUSE, CHANNEL_DATA, CHANNEL_WINDOW,
                CHANNEL_SHUT, CHANNEL_CLOSE, KEEPALIVE };

Adding a member to it, or a state to the bridge (a lease word's value, a program's state, a link's phase, a message a
loop waits for), is a design change: it needs an operator's quote, verbatim and dated, in design/bridge-and-ipc.md#wire
before it is written. A program's data crosses as SENDs on its leases, and a landing's word is the only signal a reader
acts on. The record and the rule a change is held to, restored verbatim:

From d8d69cca:design/distributed-reduce.md:203-242 (2026-09-09):

### What was done instead, and why it is forbidden

Every divergence from this document that reached the repository was a
control-flow object standing in for something the page table already held,
written by an agent that had not read the prior art and did not want to. The
record is kept here so that the next such object is recognized on sight.

- A `K_DIGEST` control frame carrying the sent hash, a mismatch poisoning the
  whole program through the status word, and a paragraph added to this
  specification to bless it (2026-09-08, `f47fc4e`, `2188f44`), after the
  operator had said that acknowledgements are a private feature of the
  transport and not the algorithm. The digest is a page.
- A `K_OPEN`/`K_READY` rendezvous with exponential backoff that held every
  page until a "ready" message arrived, and `K_ABORT`/`K_ABORTED` propagation
  between peers: a hand-rolled connection protocol on a link that already
  has flow control. Its one real function — not sending to a peer whose client
  had not attached — was the receiving bridge dropping frames addressed to it;
  the bridge now holds them.
- A runtime reduce node that counted generations by one where a row family
  advances by the number in flight, so families past the first never reduced;
  it was committed as measured without a run that could have shown it.
- Hidden-state rows read as a contiguous matrix across two pages, crossing the
  page headers; the paged path was committed with a relative error copied from
  the CPU path it replaced.
- The digest page used as the gate on the next stage's issue — the numerical
  check made a dependency of the numerical program, which this document
  forbids in two places.
- A driver with phases, per-job state, per-call tasks awaiting predictions,
  completion words, and fixed 128-row groups: a scheduler. It issued about two
  thousand GPU command buffers per evaluation and made the pass three times
  slower than the barrier it was meant to remove. The readiness it tracked was
  already in the stamps.

None of these was hard to avoid. The token store, the in-data flag, the
region runtime, the timestamped epoch and the row-streamed reduce-scatter are
each thirty to fifty years old, documented, and measured; reinventing a piece
of one from a message-passing reflex is not engineering caution, it is not
having done the reading. A change to this repository that adds a frame kind,
a handshake, a retry, a phase, a token, a gate, or a scheduler is wrong before
it is measured, and measurement has so far agreed every time.

From 1347b7aa:design/pages-and-functions.md:156-212 (2026-09-08):

### Addendum: waiting for messages about data, on infinitely many Minis

Suppose a Mini does not wait on its input rows. Suppose it waits for a
message that says the data is coming, or is done, or may be used: a token, a
gate value, a "ready" flag, a scheduler's decision, a completion callback that
stands in for the pages. Then it acts on the message. This is what goes wrong.

1. **The message is a second thing on the same link.** The pages take one
   hop. The message about them takes a hop too. Every dependency on the chain
   now costs a hop for the pages and a hop for the message. On a chain of
   depth D the NFE waits for D messages it did not need. On an infinite mesh
   D is not bounded, so the extra wait is not bounded either. Waiting on the
   rows costs nothing extra: the row is stamped when the page lands.
2. **The message can be true and the data still absent.** A message says
   something about the sender's state. The pages may still be in flight, or
   lost, or land out of order, or be corrupt. Acting on the message is acting
   on a claim. To make the claim safe, the sender and receiver need
   acknowledgements, retries, and ordering. That is a protocol on top of page
   delivery. The digest, computed after the pages were used, already finds a
   wrong page without any of it.
3. **The message can be false and the data present.** The pages landed and
   the rows are stamped, but the message is late, or the sender concluded
   early, or the link that carried the message broke while the link that
   carried the pages did not. The Mini sits with everything it needs and does
   nothing. Waiting on the rows would have run the function at once.
4. **The wait binds this Mini to another Mini's control flow.** A message
   comes from a sender's scheduler, not from data. If that scheduler is slow,
   this Mini is slow, and every Mini downstream is slow, even though their
   pages have all arrived. On an infinite mesh there is always a slowest
   scheduler somewhere, and the chain of "waiting for whose message" has no
   end. Waiting on rows binds a Mini only to the pages in front of it.
5. **Work handed to the GPU while it waits for a message is killed or idles.**
   We measured this: a command buffer that waits on a signal for seconds is
   ended by the watchdog, and every buffer behind it dies with it. A GPU given
   only functions whose rows are stamped runs at its roofline fraction. Its
   utilization then does not depend on the mesh at all.
6. **Messages need their own storage and their own freeing.** Every message in
   flight is state: a queue entry, a counter, an event value, a callback. On an
   infinite mesh the number of messages in flight is not bounded, so this
   state is not bounded, and it is freed by other messages. Pages are freed by
   release when the dependent's rows are stamped. That state is bounded by the
   pages, and the pages were needed anyway.
7. **A speculative message lets a function run before its data.** "The pages
   will arrive" is not "the pages have arrived". A function that runs on the
   promise reads stale pages and writes a wrong output with a good stamp. The
   digest will disagree later, and the NFE will be repeated, but the work was
   wasted and the wrong output already travelled a hop.
8. **Fan-in multiplies all of the above.** A function whose inputs come from
   many Minis would need a message from each, each a hop, each a race with its
   pages. The rows give one check for all of them: are they all stamped k.

The rule that follows is the one already in the spec. A Mini acts on data,
and only on data, and acts the moment the data is present. The row's stamp is
the only signal. The GPU is handed only functions whose rows are stamped. The
NFE concludes on a status word, never on a message inside the data. With that
rule a Mini's latency and utilization are the same whether it has one
neighbour or an infinite mesh behind it. Without it they are not.

## Collectives and tensor parallelism

Read `../../metal-microbench/docs/principles.md` (from this repo's root; on the Mini,
`~/metal-microbench/docs/principles.md`) and its `docs/measurement.md`. A collective is a compiled program
made ahead of time along trees of an explicit link map (`rdma/mesh-plan.h`, `rdma/mesh-collective.c`). Using
the library: `rdma/NCCL.md`; PyTorch programs: `rdma/torch-mesh/README.md`; NumPy: `rdma/mesh_mpi.py`; where
nodes or links differ: `design/heterogeneity.md`; adding a node: metal-microbench `docs/nodes.md`.

## Tests and specification

The operator instructed on September 6, 2026: "delete all tests. tests aren't
specification". The complete instruction is preserved in
[design/SPECIFICATION.md §24](design/SPECIFICATION.md#24-whole-program-review-and-deletion-of-tests-september-6-2026).
Repository-owned tests and verification harnesses have been removed. Review the complete data and execution flow
against the user instructions and explicit algebra.

Runtime testing of any code is prohibited until every item of metal-microbench `docs/acceptance-20261014.md` is attained in source; its `docs/principles.md` section 5 quotes the operator's prohibition (T9).

## Source workflow

Operator instruction, September 9, 2026: use commits and checkouts, and use
`main` when it suffices. Preserve overwritten source only in commit history;
no resets, cleans, stashes, backup copies, or uncommitted source transfers.
This governs development synchronization; the bootstrap continues to follow the latest named
branch.
