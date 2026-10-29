# The workstation and the collective, 2026-10-29

The development laptop (Ms-MacBook-Pro, node 1980260065) restarted itself at 13:31. This records why, and why the
repositories disagree about whether that machine is where the collective's code runs or only where it is written.

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
the 15:28 hourly would have restarted every node again.

The 10-28 19:46 reset is a different mechanism. WindowServer's main thread stopped answering for 40 s (userspace
watchdog, 19:44:02), and the hardware watchdog then reset the machine (boot faults `wdog, reset_in_1 timeout`). That
fits work holding the GPU the display uses. What held it is not established.

## Why the repositories disagree about where code runs

The laptop is three things at once, and each part of the tooling sees only one of them:

1. **A configured node.** `etc/mesh-nodes.json` lists it beside the Mini and the four Studios, with the role
   `workstation`, which today only changes its memory accounting (docs/live-observations.md) and its guard headroom
   (5.5 GiB).
   - It runs a bridge (io.mesh.bridge, a per-user launch agent, so it starts only at login: 14:47:50 after the 13:32
     boot).
   - It has a node checkout (`~/.cache/mesh-checkouts/mesh/current`), which every hourly moves to `main` and builds.
   - So every hourly prepares it, and standup can restart it. The rule's memory ("remote restarts are standup")
     assumes every node is a remote machine.
2. **Not a member.** The README says the configuration is "how to reach a node, not who is a member (the live fabric
   is)". Every hourly today recorded it absent: "outside the linked fabric … its map holds ['1980260065']". Its
   cables pair with no node, so it runs no workload. Preparation and standup use the configured table, while
   placement uses the live fabric: two definitions of the collective in one run.
3. **The development machine and the launcher.**
   - The working clones (~/dox/mesh, ~/metal-microbench) live here and are shared by three agent sessions.
   - The hourly's crontab runs here, so the launcher prepares its own host (`@localhost`) and can restart itself
     mid-run.
   - `~/.venv-mesh-uv` is both the rank interpreter (`python` in the configuration) and the development
     interpreter. Its `mesh.pth` points at the node checkout's rdma, and its site-packages holds a stale torch_mesh.
     So development Python imports the node checkout's library, not the clone being edited.

Because of (1) and (3), agents treat the laptop's bridge as a place to run checks, and its numbers enter the record as
if it were a member:
- today, single-rank collectives run on its bridge and called "local end-to-end";
- the M5 Max timings in design/compile-once-2026-10-29.md;
- the facts recorded about "the M5 Max laptop".

Because of (2), none of that is the collective, and the operator's rule is that every run uses the entire collective
(the hourly's largest linked one). Because of (1), the hourly treats the development machine as a node to rebuild and
restart.

## The decision this needs

Whether the workstation is a node of the collective is a configuration decision, the operator's. Either:
- **Develop and launch only.** It is not prepared, signalled or restarted by node tooling, runs no bridge, and code
  is run only on the linked nodes, through the hourly or the launcher.
- **A node like the others.** Then its cables are paired into the fabric, so it is a member when it is prepared, and
  the launcher runs elsewhere, so a restart of it is not a restart of the run.

Either way, a check written on the workstation is a development step, not evidence. The evidence is the hourly's, on
the entire collective.
