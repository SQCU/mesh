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
