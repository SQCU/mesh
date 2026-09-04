# Accessibility implementation claims

Controlling specifications: [`../../THREAT-MODEL.md`](../../THREAT-MODEL.md),
[`../../AGENTS.md`](../../AGENTS.md), and
[`../../RDMA-RULES.md`](../../RDMA-RULES.md), with network realization controlled by
[`../ACCESS-TOPOLOGY-SPEC.md`](../ACCESS-TOPOLOGY-SPEC.md).

Implementation surfaces:

- `bootstrap.sh`, `install.sh`, `install-user.sh`, and `enable-autologin.sh` realize the
  same reachability posture on every node.
- `bin/mesh-bridge.sh` performs catchable bridge teardown and restart.
- `bin/mesh-kill-guard.sh`, `bin/guard/`, and `bin/mesh-shell-guard.sh` prevent the
  observed uninterruptible RDMA-driver demotion while retaining orderly teardown.
- `networks.conf`, `install.sh`, and `bin/mesh-networks.sh` carry one noninteractive
  network inventory into every node's credential store and preferred-network set.
- `rdma/mesh-flow.c` retains the configured registration and page capacity across every
  pairing attempt. Pair failure changes retry time, never shared-memory extent.
- `bin/mesh-status.sh` invokes the colocated `bin/mesh-peers.sh` generation, so source
  and installed status observations use the same bounded, parallel peer transducer.
- `xonotic/darkplaces-work/netconn.c` treats `sv_public` only as master-server
  advertisement state. Direct challenge, status, discovery, and compatible connection
  handling remain active for every value, including during server redirection.
- `serve.sh` replaces the single 8787 page-table service.

Every step proceeds independently after reporting failure. Repository fetches select a
named branch head, authorized access is merged rather than truncated, and no machine
class, location, model, or network withholds a capability.

The repository contains one power, firewall, remote-access, auto-login, and beacon
policy. No post-install epilogue can turn a machine into a less reachable node class.

An idle bridge yields its CPU only after an iteration observes no ring transfer, no
completion, and no send in flight. This preserves active data-plane service while returning
idle host capacity to the renderer, audio deadline, telemetry, and policy processes.

Network credentials are ordinary availability configuration. They are stored in the
gitignored inventory and installed at mode `0600`; no provisioning control flow retrieves
them through a dialogue. Existing connections remain active while every inventory row is
realized, so adding another edge never removes the edge carrying the run.
