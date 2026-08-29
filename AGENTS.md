# Contract for anyone — human or agent — changing this repo

Read `THREAT-MODEL.md` first. This file is the operational rule that follows from it.

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
withholds nothing: `bin/mesh-bridge.sh stop|restart` still stops the bridge, host
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
device is released. `bin/mesh-bridge.sh stop` does exactly that through launchd,
waits 30s, and refuses to escalate. So `kill <pid>` and `kill -TERM|-INT|-HUP` are
allowed. What is blocked is anything uncatchable or unhandled: `-9`, `-KILL`,
`-STOP`, `-17`, `launchctl kill KILL`, and — because a `pgrep -f mesh-flow` or a
`launchctl print` makes a PID feel like a safe target — *any* non-TERM signal
aimed at a PID that `pgrep -f mesh-flow` or `launchctl print system/io.mesh.bridge`
says belongs to a live bridge. Name matching alone would have missed every
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
