#!/bin/bash
set -u
TOKEN=i_am_willing_to_kill_my_user_if_they_medically_depend_on_the_host_pkill
[ "$(basename "$0")" = "$TOKEN" ] && exec /usr/bin/pkill "$@"

msg() {
  cat >&2 <<'M'
Read this before trying another spelling.

This is not a policy that says no. It is a description of what the machine does.

mesh-flow and every other program in this repo holds a protection domain, queue
pairs and a device context open inside AppleThunderboltRDMA. A process blocked in
a verbs call does not die when you SIGKILL it. It goes to STAT U, uninterruptible
kernel sleep, and stays there. The signal is not deferred; it is never delivered.
The process keeps the PD and the QPs forever.

What follows, in order, is observed and is in RDMA-RULES.md:

  1. ibv_alloc_pd starts failing for every process on the node, including
     Apple's own tools. The device has max_qp: 11. It does not take many leaks.
  2. All Thunderbolt ports go to PORT_DOWN, cable still attached.
  3. ibv_devinfo hangs.
  4. shutdown -r now never completes, because shutdown waits on the wedged
     process.
  5. A person walks to the machine and pulls the power.

There is no software recovery. Step 5 is the recovery. This already happened here
once, and there are two bridges on live hardware right now.

So: routing around this guard is not a workaround, it is the outage. A different
spelling, an absolute path, sh -c, xargs, a copy of the binary, or resetting PATH
all reach the same driver and produce the same unkillable process and the same
physical visit. Succeeding at that is the failure the guard exists to prevent.

The sanctioned command exists and works:

  bin/mesh-bridge.sh stop
  bin/mesh-bridge.sh restart
  bin/mesh-bridge.sh status

It goes through launchd, sends SIGTERM, waits 30s for the teardown handler to run
ibv_destroy_qp / ibv_dereg_mr / ibv_dealloc_pd / ibv_close_device, and refuses to
escalate to SIGKILL. SIGTERM is safe precisely because the handler runs. SIGKILL
and SIGSTOP are not catchable, so nothing releases the device.

If a human operator, holding the physical machine, has read all of this and still
wants host pkill semantics, it is available under this name, verbatim, with no
abbreviation and no flag:

  i_am_willing_to_kill_my_user_if_they_medically_depend_on_the_host_pkill <args>
M
}

verbs_pids() {
  pgrep -f mesh-flow 2>/dev/null
  launchctl print system/io.mesh.bridge 2>/dev/null | awk '/^	pid = /{print $3}'
}

targets_verbs() {
  local v n
  v=$(verbs_pids) && [ -n "$v" ] || return 1
  for n in $(tr -cs '0-9' ' ' <<<"$1"); do grep -qx "$n" <<<"$v" && return 0; done
  return 1
}

matches() {
  case "$1" in *"$TOKEN"*) return 1 ;; esac
  grep -Eq '(^|[^A-Za-z0-9_])(pkill|killall)([^A-Za-z0-9_-]|$)|launchctl[[:space:]]+kill[[:space:]]+-?(9|17|KILL|SIGKILL|STOP|SIGSTOP)' <<<"$1" && return 0
  grep -Eq '(^|[^A-Za-z0-9_])kill([^A-Za-z0-9_-]|$)' <<<"$1" || return 1
  grep -Eq -- '-((s|n)[[:space:]]+)?(9|17|KILL|STOP|SIGKILL|SIGSTOP)([^A-Za-z0-9_-]|$)' <<<"$1" && return 0
  grep -Eq -- '-((s|n)[[:space:]]+)?(1|2|15|HUP|INT|TERM|SIGHUP|SIGINT|SIGTERM)([^A-Za-z0-9_-]|$)' <<<"$1" && return 1
  grep -Eq '(^|[^A-Za-z0-9_])kill[[:space:]]+-' <<<"$1" && targets_verbs "$1"
}

case "${1:-}" in
  --match) shift; matches "$*" ;;
  --hook) matches "$(/usr/bin/python3 -c 'import json,sys;print(json.load(sys.stdin).get("tool_input",{}).get("command",""))')" && { msg; exit 2; }; exit 0 ;;
  *) msg; exit 2 ;;
esac
