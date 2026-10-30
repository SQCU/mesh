# The workstation and the collective, 2026-10-29

The development laptop (Ms-MacBook-Pro, node 1980260065) was reset by a watchdog on 10-28 and restarted on 10-29, both
times by our code. This records why, who put the collective's tooling on that machine, and why the repositories
disagree about whether it is where the collective's code runs or only where it is written. Times are Pacific unless
marked Z.

## The requirement

No program of ours may crash, hang, reset or restart any machine, whether it is a member of the collective or not. The
laptop is where a failure is most visible: the operator's session, display and open work are on it, and a boot clears
its /private/tmp. The first version of this document (55f87055) ended by asking the operator whether the laptop is a
node. That was wrong. Membership does not change the requirement, and the question made a defect in our code into a
configuration choice for the operator.

## What our code did to machines on 10-28 and 10-29

- **10-28 19:46, the laptop.** It was linked that evening (the 19:28 record lists all six nodes), so the 19:28 hourly
  placed ranks on it for the first time (bandwidth, tp, ep, cp, fabric-dnn).
  - 36a321be, landed at 15:27 that day, had removed the GPU waits' fixed bound: `mesh_step_wait` looped while
    anything on its communicator moved.
  - A spinning wait held the GPU that also drives the display. WindowServer's main thread stuck in the GPU driver,
    with every other drawing process blocked behind it, from 19:43:22. The userspace watchdog panicked at 19:46:41
    (no check-in for 120 s).
  - Fixed in a2ff516b: every wait is again bounded when it is made.
  - The other session established all of this at 10-29 02:56Z. The first version of this document, and metal-microbench
    docs/facts.md, said the cause was "not established". I wrote that without reading the record.
- **10-29, every machine restarted.** Node prepare's standup rule, acting on two false positives (see below; fixed in
  bd4d90b6), restarted:
  - the Mini at 12:31, and m5ult1 and m5ult2 at 12:51;
  - all six machines at 13:31, with shutdown stalls from 13:31:32 to 13:31:46: the laptop, the Mini, m5ult1, m5ult2,
    m5ult3 and m5ult4. The 13:28 record names only the laptop and m5ult1, because its driver died with the laptop.
  - The laptop's restart was the launcher restarting its own host mid-run. The boot cleared /private/tmp, which held
    another session's working clone.
- **10-29 15:32-17:55, m5ult3 and m5ult4 hung.**
  - It began during the 15:28 hourly. chain-w16384's linear form had stored its outputs tens of MB past Y (a stride
    defect from 61d37e10 on 10-26, fixed in 60b84953). The root's own list entries were overwritten, and the links to
    m5ult3 and m5ult4 went ENETDOWN at 22:32:11Z. Both then stopped answering during lstm-h16384.
  - For more than two hours they answered neither ping nor ssh, and wrote no panic report: the kernel hung and no
    watchdog ended it.
  - Both booted at 17:55:13, the same second, with the power controller recording the supply lost
    (`por,por_vddrtc`): their power was cut.
  - What hung them is not yet known.
- **Before these.** docs/facts.md records the Mini's watchdog panic on 10-20 (one process wired all its memory) and
  the Studios' watchdog panics and resets on 10-22, 10-23 and 10-24 (cp three times in ten minutes). Every one came
  from our processes or our installs.

## What restarted the laptop at 13:31

The evidence, in order:
- The laptop booted at 13:32:00 (`kern.boottime`). The last report before it is a shutdown stall at 13:31:41, and there
  is no panic report, so this was an orderly shutdown that some process asked for.
- At the stall, the processes still alive were cron's `bash`, then `hourly.py`, then mesh's driver, with six `ssh`
  sessions open: the 13:28 hourly preparing its six configured nodes, this laptop among them.
- The 13:28 run's record (output_data/hourly/20261029T202801Z) names the reason for node `1980260065@localhost`, and the
  same reason for m5ult1 (521750041): "the bridge did not restart within 180 s of its signal; no kernel fault this boot,
  so the machine is restarting". That is `rdma/mesh_nodes.py` `restart_unless_faulted`, which runs `sudo -n shutdown -r
  now`. The driver itself ended by SIGTERM, since it runs on the machine it restarted.
- The 12:28 hourly had restarted the Mini the same way at 12:31 (804f2743's message).
- Two of today's commits combined to trigger it:
  - 804f2743 (12:50) added the standup rule: restart a machine whose bridge does not answer its restart signal, or
    whose heartbeat is not the installed one.
  - d517ffe4 (13:10) changed the bridge, so the 13:28 install replaced it and asked it to restart.

Two defects made the rule restart machines whose bridges had restarted correctly. Both are fixed in bd4d90b6:
- **Timing.** `bin/mesh-user.sh` sends the bridge's SIGHUP in the first of node prepare's three build steps, but the
  180 s clock started after the third (torch-mesh's `build_ext`, about a minute). A bridge that restarted promptly
  began before the clock started, so it could never answer. Every node whose bridge images changed was restarted
  180 s later.
- **Re-exec.** The heartbeat runs its new image again on that same SIGHUP (mesh-heartbeat.c, `execv`). `execv` keeps
  the pid and the process's start time, so a start-time comparison with the new image's install read every
  re-executed heartbeat as stale. Its file is the test: the inode it executes against the installed image's.
  804f2743's evidence that m5ult1 and m5ult2 "ran heartbeats started on 10-26" is what a re-executed heartbeat looks
  like, so that finding needs a second look.

c8b9422b (13:28, not yet prepared at the time) edits `rdma/Makefile`, on which mesh-flow and mesh-heartbeat depend.
Re-signed with a certificate, which carries a signing time, they differ from the installed ones, so without the fix
the 15:28 hourly would have restarted every node again. It did not: the 15:28 run prepared all six nodes in 12.5-49.8 s
and restarted none.

The operator's standup rule ("restart machines that don't have kernel faults if they don't respond to bridge shutdown
signals", 10-29) is a recovery for remote nodes. Here it acted on our false positives. A bridge that really does not
answer its signal is itself a defect in our code. The machine running the launcher is not a remote node.

## Who put the collective's tooling on the laptop

No requirement put it there. Each piece was placed by an agent working on the laptop, in the place its own shell ran:

| On the laptop | Placed by | Since |
|---|---|---|
| The hourly's launcher | A Claude session's own scheduler (CronCreate in session 15c469fa, 10-23 06:55Z), which fires only inside that session on this machine; then metal-microbench 0feb7db3's crontab file (10-24 23:42, no agent attribution), installed in this user's crontab, with this laptop's paths | 10-23 |
| The node role | The laptop was in the first name table (fd580de0, 10-21) when it was cabled to the Mini. ee60e226 (10-28, a Claude session) gave it the role `launcher` in etc/mesh-nodes.json, later `workstation`. That made the launcher a node in configuration | 10-28 |
| A node checkout, preparation and standup, and the bridge they reinstall | Node tooling that walks etc/mesh-nodes.json, so it reaches every configured machine, the launcher's own host included (`@localhost`). The bridge itself dates from when the laptop was a member | 10-26 |
| One interpreter for ranks and development | ~/.venv-mesh-uv (10-26), whose mesh.pth was pointed at the node checkout on 10-28 | 10-26 |

What the operator said and did points the other way:
- **The laptop was a member once.** On 10-01 the operator spoke of "the machines connected to this dev machine", and
  on 10-23 of the LSTM target "on a macbook and a mac mini". That was the two-node fabric before the Studios.
- **Then it was unplugged.**
  - Its Thunderbolt ports were down from 10-26 (docs/facts.md).
  - On 10-28 the operator rejected a driver that refused to run with the laptop's ports down: "why does the
    collective try to use a specific list of computers instead of what's in the collective?"
  - After the 19:46 reset it was out at the cable level again by 20:07.
  - Every hourly since records it "outside the linked fabric".
- **Where the hourly runs.** The hourly's mandate (metal-microbench 0bdb5a9f, 10-22) is short because its premise
  already answers this. The collective is several nodes with a lot of aggregate bandwidth and high reachability, and
  the hourly runs on it, its results "logged by whatever computers survive those runs". A single launcher on the
  laptop contradicts that premise. It writes the only record to its own disk (output_data is not versioned), and its
  own run can restart it.
- **A bridge on a development machine.** The operator's one statement of this, on 10-21, is a tool for upgrades:
  "build the same version of the bridge on your dev machine and use that to connect and upgrade". That makes the
  machine neither a member, nor the launcher, nor a restart target.

## Why the confusion spreads

Every agent session runs on the laptop. Whatever an agent schedules, configures or tests lands where its shell is,
because nothing in either repository says where things run. The next agent then reads that placement as a
requirement:
- a role in the node table;
- a crontab line in the repository;
- a "workstation" with its own memory accounting.

This review did the same thing. Its first version listed the laptop's three roles as facts, and asked the operator to
choose among them, though two of the three had never been asked for.

The tooling also keeps two definitions of the collective:
- the README's ("how to reach a node, not who is a member (the live fabric is)");
- node prepare and standup's, the configured table.

So the hourly prepares and can restart a machine that its placement, a few seconds later, records as absent.

The same placement made the laptop look like a place to collect evidence:
- today, single-rank collectives were run on its bridge and called "local end-to-end";
- the M5 Max timings in design/compile-once-2026-10-29.md;
- the facts recorded about "the M5 Max laptop".

None of that is the collective. Every run uses the entire collective, the hourly's largest linked one.

## What a run on the collective needs

- **Records.** The run needs none on the operator's machine.
  - Each member writes its own rank's record, named by the run, in its own output directory. A reader (the laptop,
    anyone) fetches them from whichever members survive, when it wants to.
  - Today:
    - ranks' outputs return over the launcher's ssh sessions;
    - the record is written only on the launcher's disk;
    - each rank's stdin is held open by the launcher, and its guard ends the rank when that stdin closes
      (`mesh_nodes.ranks`). So the laptop sleeping, rebooting or losing its network ends every rank on the collective.
- **Membership and capacity.** The bridges already hold both.
  - A member's own bridge (mesh-stat, on the node) lists its linked part: nodes, links, each port's bandwidth, and each
    node's model, memory, region and available bytes. etc/mesh-capacity.json gives each model's GPU and fabric
    figures.
  - The beacons (bin/mesh-beacon.sh, `_meshnode._tcp`) announce every bridge, its node number and model, on every
    network a node is on.
  - The fabric is routed (babeld on every fabric port, bin/mesh-router-init.sh), and ranks already meet at rank 0's
    fabric address.
  - etc/mesh-nodes.json is needed only to reach a node from outside the fabric (its management address and key) and
    for names.
  - Today discover() starts from that table, and node preparation runs on every reachable configured machine (`found`
    in `_execute_locked`), not on the fabric's members. So the unlinked laptop was prepared, and could be restarted, in
    every hourly.
  - "Largest" is by node count today (`max(parts(...), key=len)` in discover and run_workload). By capacity it is a sum
    over the same records.
- **Starting a run.** A program on each member, given the run (its scheduled minute and the main commits as of that
  minute):
  - prepares its own checkout (node prepare already runs on the node);
  - reads its linked part from its own bridge, and from that its rank (sorted node numbers, as today) and rank 0's
    fabric address;
  - runs its own rank under its own guard, with a local stdin;
  - writes its own record.

  The members agree without talking: the run's name comes from its minute and commits, and its rank port from its
  name, as today (crc32 of the run).

  Each member keeps its own schedule, a launchd calendar job installed with the node. mesh has none today: "No job
  runs on a schedule". An unscheduled run is the same program started on every member, with nothing held open by
  whoever started it.

  A node outside the largest part does not run its own part. Seeing the other parts is not a limit of the node. Only
  its bridge's map is bounded by its links, and the node is also on the LAN and the tailnet with every other node:
  - a 3 s browse of `_meshnode._tcp` from m5ult1 (10-29 18:3x) listed all six beacons, the unplugged laptop's
    included, each with its node number and model;
  - the launcher's discover() compares every part each hour by reading every map over that same network.

  The part each node is linked into is published nowhere. That is an omission, not a limit. Choosing the part that
  runs is a top-1 (the operator, 10-29):
  - **Who speaks for a part.** Each part's representative is its smallest node number, a choice every member reads
    from the same bridge map, so the members exchange nothing to make it.
  - **What it shares.** The representative announces its part's capacity, the sum over its members' records, on
    whatever non-RDMA network reaches other nodes. Today that is its beacon, beside the node's model.
  - **Who runs.** Every node takes the top-1 of the announcements it heard: the largest capacity, ties going to the
    smaller representative. The winning part's members run the hourly. Every other node writes one record naming
    the part that won and its capacity.
  - **A node that hears no other node by any means.** It closes the hourly early, with a record whose warning says it
    does not know what is going on.
- **What was never written as code.** The run's start and its record, on the collective. Everything else a run needs
  is already on the node: membership, capacity, routing, rendezvous, preparation and the guard. Only three things live
  outside:
  - the start (the laptop's crontab, then ssh into each machine);
  - the rank's lifetime (a stdin held by the launcher);
  - the record (the launcher's disk).

  These three tie the collective to the laptop, and put the collective's tooling on the laptop.
- Every crash above is a source defect, fixed in code: unbounded waits, writes past an operand, a restart rule acting
  on false positives, a bridge that cannot answer its signal.
- Development does not run on, attach to or import from the collective's tooling on the laptop. A check written there
  is a development step, not evidence.
