# Contract for anyone — human or agent — changing this repo

Read `THREAT-MODEL.md` first. This file is the operational rule that follows from it.

**Recovery is critical priority.** The transport, how it fails and how it recovers: `design/transport.md`.

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

## A node is its hardware; names are virtual

The operator, 2026-10-21: "from this point onwards, we never rely on something like a
localhostname as a name or identifier for something which we are programming.
localhostnames are apparently a legacy thing, can be named automatically, collide,
etc. undignifed, and we almost certainly do not need them for anything algorithmic",
and "we should be able to use any kind of virtualization of 'names' such that we never
rely on anything like, again, that localhostname thing, which makes no promises of
selectivity or unique identifiability or anything of the sort".

A node's identity is its host UUID (`IOPlatformUUID`, `gethostuuid()`). The bridge's
node id, the fabric ULA (`mesh-router-init.sh`) and node-info's `node_id` are all drawn
from it; the hardware serial number is the same machine's other fixed identifier. Code
that selects, matches, deduplicates, addresses or labels a node keys on that identity or
on something derived from it. A name a person uses is a virtual layer over the identity:
a table keyed by identity, resolved by the code, never the other way round.

Never read `LocalHostName`, `ComputerName`, `HostName`, `hostname`, `gethostname()`,
`socket.gethostname()` or `platform.node()` to identify, select, match or label a node,
and never write one into anything a program reads back (a beacon's instance name, a
map's name field, a tailnet label). macOS assigns them in Setup Assistant and renames a
running machine on its own when another claims the same name. On 2026-10-21 the second
M5 Ultra Studio came up as `mdots-Mac-Studio`, macOS renamed the first, an hour into its
service, to `mdots-Mac-Studio-2`, and `mdots-Mac-Studio.local`, its beacon and
`mesh-peers` all pointed at the other machine.

The name table is `etc/mesh-nodes.json`, entries keyed by host UUID, read through
`rdma/peers.py` (`host`, `table`, `label`): a node it does not list is `node <number>`.
The beacon announces the host UUID; `mesh-peers`, node-info, `mesh-top`, the observer
and telemetry key on the node number and take names from the table. Reliance that
predates this rule and remains is debt to remove, not precedent: the bridge map's
`name` (`rdma/mesh-link.c`, which also reaches metal-microbench's `tools/mesh/grid.py`
name selection), and the bootstrap's `gethostname()` (`rdma/nccl-mesh.c`).

## No tuples

The operator, 2026-10-21: "anything which claims to be fitted on a specific pair of
devices has to go forever. thjere will be no tuples in this codebase. there can be no
tuples in this codebase. the instant it muts be said that 'x was fitted on node1, node2,
not node3 or node4', we cede that it is not a model of anything that matters, it is not
durable, it is not predictive, and it cannot be used in code for programs that are
uniform to all meshes we might ever run hereon". And: "there are several points in this
codebase where using an online optimizer to solve a problem is more responsible than
attempting to a priori 'model' and 'persist' data; they are chiefly related to the
incredible mysteries of tensor parallelism ... some things in this list of 'models'
should probably become 'configurations', emphasizing semantically their unjustified
nature".

Every value the code acts on is one of three kinds, and says which:

- **Fact:** read from the running mesh, per node, keyed by its identity (a link's
  bandwidth as its bridge reports it, a device's queue pairs, the memory a plan counts).
- **Online optimizer:** decides from observations of the members it is serving, and
  persists nothing but those observations. Tensor-parallel placement and routing belong
  here; no closed form chooses them ahead of observation.
- **Configuration:** a stated choice that claims no justification, named a
  configuration, the same for every mesh (a resolution, a window, a starting split of
  equal parts).

Nothing may be fitted on, derived from, sized for or keyed to particular machines,
pairs or sets of machines: no constant from "one pair ledger pass", no table of two
chips from which others are extrapolated, no code path defined for two members, no
configuration that names the machines it was written for. A measurement of particular
machines may be kept as a dated record; no code reads it.

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
- [ ] No host name (`LocalHostName`, `ComputerName`, `HostName`, `gethostname()`) read
      to identify, select, match or label a node; identity is the host UUID.
- [ ] No tuples: nothing fitted on, sized for or naming particular machines; every value
      a fact, an online optimizer's observation, or a configuration named as one.
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
This authorization applies to every mesh machine.

## Collectives and tensor parallelism

Read `design/pages-and-functions.md` first: the algorithm on two peers and on infinitely many Minis, and why
waiting on anything but stamped pages fails on both (its analysis: metal-microbench
`docs/mesh_distributed_reduction_analysis.md`). Then `../../metal-microbench/docs/principles.md` (from this repo's root; on the Mini,
`~/metal-microbench/docs/principles.md`) and its `docs/measurement.md`. A collective is a compiled program
made ahead of time along trees of an explicit link map (`rdma/mesh-plan.h`, `rdma/mesh-collective.c`). Using
the library: `rdma/NCCL.md`; PyTorch programs: `rdma/torch-mesh/README.md`; NumPy: `rdma/mesh_mpi.py`; where
nodes or links differ: `design/heterogeneity.md`; the transport: `design/transport.md`; review: `design/rubrics.md`
(rubric M for the bridge, C for the collectives, D for programs that use them); adding a node: metal-microbench
`docs/nodes.md`.

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
