# Mini restart over SSH, 2026-09-05

The operator requested a normal remote reboot to test whether the Mini could
recover without physical intervention. It succeeded: SSH returned after 253
seconds, with a new boot UUID and no remaining uninterruptible processes. Both
RDMA bridges paired automatically at full configured capacity. Later analysis
established that shutdown stalled and the watchdog reset the machine; this was
not a clean kernel shutdown.

Before the request, SSH worked. Boot UUID was
`753BD0B6-A834-4289-BEB3-5C584843F2F8`, with boot time
2026-09-05 04:56:02 UTC. The RDMA bridge was in `Us`, with 478 additional device
probes in `U`. The probe accumulation had already been stopped.

The curriculum received SIGTERM and completed its normal shutdown at
09:34:14.999 UTC. Both learners saved their checkpoints; the expert exited and
the game server and client quit. Each checkpoint retained 162 optimizer updates,
30 parameter arrays and 62 optimizer arrays, all finite. No new learning had
occurred while transport was unavailable.

At 09:34:53 UTC the command sent through SSH was:

```sh
sudo -n /sbin/shutdown -r now
```

The Mini replied `Shutdown NOW!` and `System shutdown time has arrived` before
the SSH command exited with status 255. This establishes that the command reached
the host; connection closure alone does not establish that reboot completed.

SSH probes request the boot UUID, boot time and uptime. They use both a connection
timeout and SSH keepalive settings, with fifteen seconds between attempts. A
successful connection must report a changed boot UUID to count as a verified
restart. The planned observation window is ten minutes from the command.

| Time after command | Observation |
| --- | --- |
| 46–93 seconds | SSH connection refused. |
| 118 seconds onward | SSH timed out or reported the host down. |
| About two minutes | Direct-IP SSH, screen sharing and telemetry probes timed out; ping received no replies. |
| 253 seconds | SSH returned with boot UUID `9707F1C5-92FC-495E-A4C9-40F04BEF20C3`, boot time 09:38:44 UTC, and uptime 21 seconds. |
| Follow-up | Zero processes in `U`; Mini bridge PID 362 and unchanged laptop bridge PID 65397 both reported `pair up`. |

No forced reboot, bridge SIGKILL, second shutdown request or physical power cycle
has been performed as part of this experiment. Network probes alone did not
establish the host's internal state after SSH closed. The recovered shutdown-stall
report identifies a kernel deadlock during PD deallocation. The panic records
218 seconds without watchdogd check-ins during shutdown; ResetCounter records
`wdog,reset_in_1`. The watchdog supplied the reset that completed recovery.
The completed boot and recovered bridge are established by the subsequent SSH
and bridge observations. The earlier assertion that this incident necessarily
required a physical power cycle was incorrect.

Reboot removed the runtime directories under `/tmp`. They were restaged, and the
server, client and saved learners resumed at 09:43 UTC. Observations and optimizer
updates advanced again. See [the kernel RCA](RDMA-KERNEL-RECOVERY.md) for subsequent
provider failures and reliability changes.

The exact command result and timestamped SSH observations are retained in
`measurements/ssh-reboot-20260905.json`. Additional process, network and checkpoint
evidence is in `/tmp/mesh-ssh-reboot-20260905`. That directory also contains the
prepared resume command, using the checkpoints saved immediately before reboot.
