#!/bin/bash
# The node's bridge, a launchd job the node keeps (design/page-queue-change-request.md, "Replacing the --link form": the
# run's driver starts no bridge; "The bridge's process lifecycle": the failport's).  The job's form is tag
# pre-return-20261103 bin/mesh-user.sh lines 36-62, for this repo's paths:
# - io.mesh.bridge, the user's job in user/<uid>, its Background session (LimitLoadToSessionType; tag mesh-user.sh:49,
#   56, 61, bin/mesh-launchd-agent.sh:6): no GUI login needed, as a gui/<uid> domain is;
# - it runs the heartbeat (rdma/mesh-heartbeat.c) with the bridge image and the configuration's path, its
#   ProgramArguments alone; the heartbeat opens the resident bridge (mesh-flow CONFIGURATION) in a process group of its
#   own and closes it when asked;
# - RunAtLoad; KeepAlive {SuccessfulExit: false}: a heartbeat that ended 0 (launchd's stop of the job, a release) is not
#   started again, one that failed is, at most once a ThrottleInterval of 60 s;
# - ExitTimeOut 60: launchd's SIGKILL past it reaches the heartbeat alone, which holds no device (AbandonProcessGroup;
#   ExitTimeOut 0 was SIGKILL at once on the Mini, 65157c0e);
# - StandardErrorPath: the bridge log, which the heartbeat, the bridge and its children append to (mesh_say);
# - the configuration, the canonical property list the bridge reads (the tag's mesh-link.c configuration_read): the tag's
#   rdma/bridge_configuration.py keys `name` (the document's stem: here bridge.conf's) and `region_bytes` (the arena's
#   bytes: mesh_arena_pages pages, else mesh_pct of the node's memory), and main's geometry as integers, each where
#   bridge.conf states it, else mesh_layout's default: `block_pages`, `rows`, `window_pages`, `hops` (the page lists'
#   room) from mesh_block_pages, mesh_rows, mesh_window_pages, mesh_hops, each in 1..INT32_MAX, and `window_pages`
#   within a 4 GiB bank, written as a bank's pages where bridge.conf states none and the region is larger;
# - the configuration written beside it and renamed into place only where its bytes differ; a loaded heartbeat then sent
#   SIGHUP, which restarts the bridge on the configuration and on the images installed (the heartbeat's own included),
#   the region kept.
# bridge.conf's node= and links= were the --link form's and are not read: the bridge finds its ports and its node itself.
# Neither are region= (the node has one region, mesh.h MESH_NAME, and one bridge: no label of a second region's),
# mesh_qps (the queue pairs are the bridge's to negotiate at each pairing) nor scope= (the job is the user's, as above).
# A node runs one bridge: main's earlier job (the --link form's mesh-flow), in gui/<uid> where bridge.conf said
# scope=gui, is booted out before this one loads; in system (scope=system, /Library/LaunchDaemons) it is root's, and
# start refuses while it is in place.
#   mesh-bridge.sh start    the configuration and the job written; the job loaded where it is not, a loaded job of
#                           another program (the --link form's mesh-flow) stopped first; a loaded heartbeat sent SIGHUP
#                           where the configuration changed; the bridge ready within 8 s, or an error naming what is not
#   mesh-bridge.sh restart  as start, a loaded heartbeat sent SIGHUP whatever changed: the bridge restarted on the images
#                           make install-bridge installed, its region kept
#   mesh-bridge.sh stop     the job booted out (launchd's SIGTERM to the heartbeat: the bridge closed by SIGTERM, its
#                           region released, then the heartbeat ends, at most 5 MESH_PLATFORM_NS on), awaited 30 s and
#                           never escalated; main's earlier gui/<uid> job where it is the one loaded
#   mesh-bridge.sh status | ready
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
CONF="${MESH_CONF:-/usr/local/mesh/bridge.conf}"
[ -f "$CONF" ] || CONF="$HOME/.mesh-bridge.conf"
[ -f "$CONF" ] || CONF="$ROOT/etc/bridge.conf"
LABEL=io.mesh.bridge
BIN="${MESH_BIN:-/usr/local/mesh/bin/mesh-flow}"; [ -x "$BIN" ] || BIN="$ROOT/rdma/mesh-flow"
# the heartbeat beside the bridge image, installed with it (rdma/Makefile install-bridge)
HEARTBEAT="$(dirname "$BIN")/mesh-heartbeat"
STAT="$(dirname "$BIN")/mesh-stat"; [ -x "$STAT" ] || STAT="$ROOT/rdma/mesh-stat"
case "$BIN" in *-wt/*|*/.build/*|/tmp/*|/private/tmp/*) echo "mesh-bridge: refusing to launch the bridge from a worktree, build or scratch path ($BIN); run make install-bridge" >&2; exit 65 ;; esac

mesh_pct=
mesh_arena_pages=; mesh_block_pages=; mesh_rows=; mesh_window_pages=; mesh_hops=

if [ -f "$CONF" ]; then . "$CONF"; fi
mesh_arena_pages="${MESH_ARENA_PAGES:-$mesh_arena_pages}"
mesh_block_pages="${MESH_BLOCK_PAGES:-$mesh_block_pages}"
region=/mesh0

# the tag's domain (bin/mesh-launchd-agent.sh:6) and its paths: the job's definition, its configuration (tag
# mesh-user.sh:22, 36) and this repo's log
DOM="user/$(id -u)"; PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
CONFIGURATION="$HOME/Library/Application Support/$LABEL/configuration.plist"
LOGDIR="${MESH_LOG_DIR:-$HOME/.mesh-logs}"
# main's earlier job, the --link form's, where bridge.conf said scope=gui; scope=system's was a LaunchDaemon
FORMER="gui/$(id -u)"; DAEMON=/Library/LaunchDaemons/$LABEL.plist

# the configuration from bridge.conf, written to $CONFIGURATION.new; renew=1 where its bytes differ from the one in place
configuration_write() {
  page=$(getconf PAGESIZE) && memory=$(sysctl -n hw.memsize) || return 1
  # each count the bridge reads is in 1..INT32_MAX (mesh-link.c configuration_read)
  for key in mesh_arena_pages mesh_block_pages mesh_rows mesh_window_pages mesh_hops; do
    value=${!key}
    case "$value" in '') continue ;; 0*|*[!0-9]*) echo "mesh-bridge: $key=$value in $CONF is not a positive count" >&2; return 64 ;; esac
    [ "${#value}" -le 10 ] && [ "$value" -le 2147483647 ] || { echo "mesh-bridge: $key=$value in $CONF is past 2147483647" >&2; return 64; }
  done
  if [ -n "$mesh_arena_pages" ]; then region_bytes=$((mesh_arena_pages*page))
  elif [ -z "$mesh_pct" ]; then echo "mesh-bridge: $CONF states no region (mesh_arena_pages or mesh_pct); the bridge has no default" >&2; return 64
  fi
  if [ -n "$mesh_pct" ]; then
    cap=$(awk -v m="$memory" -v p="$mesh_pct" 'BEGIN{ if(!(p+0>0 && p+0<=100)) exit 1; printf "%d", m*p/100 }') \
      || { echo "mesh-bridge: mesh_pct=$mesh_pct in $CONF is not a percent of the node's memory" >&2; return 64; }
    [ -n "$mesh_arena_pages" ] || region_bytes=$cap
    [ "$region_bytes" -le "$cap" ] || { echo "mesh-bridge: a region of $region_bytes bytes exceeds mesh_pct=$mesh_pct of the node's $memory bytes" >&2; return 64; }
  fi
  [ "$region_bytes" -ge "$page" ] && [ "$region_bytes" -le "$memory" ] \
    || { echo "mesh-bridge: a region of $region_bytes bytes is not within a page and the node's $memory bytes of memory" >&2; return 64; }
  # the registered window within one 4 GiB bank (mesh-verbs.h MESH_BANK; wire_map refuses more, and an absent
  # window_pages is the whole arena): bridge.conf's window where it states one, else, past a bank, a bank's pages
  window_pages=$mesh_window_pages; bank=4294967296
  if [ -n "$window_pages" ]; then
    [ $((window_pages*page)) -le "$bank" ] || { echo "mesh-bridge: mesh_window_pages=$window_pages in $CONF is past a 4 GiB bank of $page-byte pages" >&2; return 64; }
  elif [ "$region_bytes" -gt "$bank" ]; then window_pages=$((bank/page))
  fi
  name=$(basename "$CONF"); name=${name%.*}
  new="$CONFIGURATION.new"
  mkdir -p "$(dirname "$CONFIGURATION")" && rm -f "$new" || return 1
  plutil -create xml1 "$new" && plutil -insert name -string "$name" "$new" && plutil -insert region_bytes -integer "$region_bytes" "$new" \
    || { rm -f "$new"; return 1; }
  for pair in "block_pages:$mesh_block_pages" "rows:$mesh_rows" "window_pages:$window_pages" "hops:$mesh_hops"; do
    [ -z "${pair#*:}" ] || plutil -insert "${pair%%:*}" -integer "${pair#*:}" "$new" || { rm -f "$new"; return 1; }
  done
  renew=; cmp -s "$new" "$CONFIGURATION" || renew=1
}
# the configuration renamed into place, or the one written dropped where it is the same
configuration_install() { if [ -n "$renew" ]; then mv -f "$new" "$CONFIGURATION"; else rm -f "$new"; fi; }

# the bytes the bridge registers are its window, at most the region's arena (mesh-verbs.h wire_map: wire_pages of the
# arena): read from the configuration, where the --link form asked its image (mesh-flow --layout), which the resident
# bridge does not take
wire_check() {
  want=$region_bytes
  [ -z "$window_pages" ] || [ $((window_pages*page)) -ge "$want" ] || want=$((window_pages*page))
  if [ "$want" -gt "$(sysctl -n vm.global_user_wire_limit)" ]; then
    if [ "$(id -u)" = 0 ]; then sysctl -w vm.global_user_wire_limit="$want"
    else sudo -n sysctl -w vm.global_user_wire_limit="$want"; fi
  fi
}

# the job's definition put in place (tag bin/mesh-launchd-agent.sh: parsed before it replaces the one in place;
# job_changed=1 where its bytes differ, a loaded job running on until its next load)
write_plist() {
  mkdir -p "$(dirname "$PLIST")" "$LOGDIR" || return 1
  cat > "$PLIST.new" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!-- the node's bridge (bin/mesh-bridge.sh): its heartbeat (rdma/mesh-heartbeat), which opens and closes the bridge; a new binary or configuration is installed by rename and the heartbeat's SIGHUP restarts both on them, the region kept; SIGUSR1 releases the bridge and its region, SIGTERM releases them and ends the heartbeat -->
<plist version="1.0"><dict><key>Label</key><string>$LABEL</string>
<key>ProgramArguments</key><array><string>$HEARTBEAT</string><string>$BIN</string><string>$CONFIGURATION</string></array>
<key>LimitLoadToSessionType</key><string>Background</string><key>AbandonProcessGroup</key><true/>
<key>RunAtLoad</key><true/><key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict><key>ThrottleInterval</key><integer>60</integer><key>ExitTimeOut</key><integer>60</integer>
<key>StandardErrorPath</key><string>$LOGDIR/$LABEL.log</string></dict></plist>
PL
  plutil -lint -s "$PLIST.new" >/dev/null 2>&1 || { rm -f "$PLIST.new"; echo "$LABEL: its new definition does not parse; the one in place kept" >&2; return 1; }
  job_changed=; cmp -s "$PLIST.new" "$PLIST" || job_changed=1
  mv -f "$PLIST.new" "$PLIST"
}

pid_of() { launchctl print "${1:-$DOM}/$LABEL" 2>/dev/null | awk '/^\tpid = /{print $3}'; }
# the loaded job's program (tag bin/mesh-user.sh line 56), by its physical path
program_of() {
  program=$(launchctl print "$DOM/$LABEL" 2>/dev/null | awk '/^[[:space:]]*program = /{sub(/^[[:space:]]*program = /,""); print; exit}')
  [ -z "$program" ] || echo "$(cd "$(dirname "$program")" 2>/dev/null && pwd -P)/$(basename "$program")"
}
# the bridge the heartbeat opened, by the pid its region records (the job's own pid is the heartbeat's)
bridge_of() { "$STAT" "$region" 2>/dev/null | sed -n 's/.*"bridge_pid":\([1-9][0-9]*\).*/\1/p'; }
client_attached() { "$STAT" "$region" 2>/dev/null | grep -qE '"client":[1-9]'; }

# the job in DOMAIN stopped (default: this job's, else main's earlier gui/<uid> job where that is the one loaded)
do_stop() {
  d=${1:-$DOM}
  [ -n "${1:-}" ] || launchctl print "$d/$LABEL" >/dev/null 2>&1 || ! launchctl print "$FORMER/$LABEL" >/dev/null 2>&1 || d=$FORMER
  p=$(pid_of "$d"); b=$(bridge_of)
  launchctl bootout "$d/$LABEL" >/dev/null 2>&1 & request=$!
  for _ in $(seq 1 300); do
    if ! kill -0 "$request" 2>/dev/null && { [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; } && { [ -z "$b" ] || ! kill -0 "$b" 2>/dev/null; }; then break; fi
    sleep 0.1
  done
  if kill -0 "$request" 2>/dev/null || { [ -n "$p" ] && kill -0 "$p" 2>/dev/null; } || { [ -n "$b" ] && kill -0 "$b" 2>/dev/null; }; then
    echo "mesh-bridge: stop pending after 30s (heartbeat=${p:-none}, bridge=${b:-none}, request=$request); not escalating or replacing its region" >&2
    return 1
  fi
  wait "$request" 2>/dev/null
  echo "mesh-bridge: stopped"
}

do_start() {
  force=${1:-}
  [ -x "$BIN" ] && [ -x "$HEARTBEAT" ] || { echo "mesh-bridge: no bridge image and heartbeat at $(dirname "$BIN"); run make install-bridge" >&2; return 66; }
  configuration_write || return $?
  if [ -e "$DAEMON" ] || launchctl print "system/$LABEL" >/dev/null 2>&1; then
    rm -f "$new"
    echo "mesh-bridge: main's earlier system job ($DAEMON, system/$LABEL) is in place, root's, and a node runs one bridge; boot it out and remove its definition as root first" >&2
    return 1
  fi
  if launchctl print "$FORMER/$LABEL" >/dev/null 2>&1; then
    if [ -z "$force" ] && [ -n "$(pid_of "$FORMER")" ] && client_attached; then
      rm -f "$new"
      echo "mesh-bridge: main's earlier job ($FORMER/$LABEL) runs with a client attached; not replacing it" >&2; return 1
    fi
    echo "mesh-bridge: main's earlier job ($FORMER/$LABEL) is loaded; stopping it for the heartbeat's job in $DOM"
    do_stop "$FORMER" || { rm -f "$new"; return 1; }
  fi
  if launchctl print "$DOM/$LABEL" >/dev/null 2>&1; then
    running=$(program_of); p=$(pid_of)
    if [ -n "$p" ] && [ "$running" = "$(cd "$(dirname "$HEARTBEAT")" && pwd -P)/$(basename "$HEARTBEAT")" ]; then
      if [ -z "$renew$force" ]; then
        configuration_install; write_plist || return $?
        [ -z "$job_changed" ] || echo "$LABEL: its new definition is in place and takes effect at its next load (login or boot); the loaded job runs on"
        echo "mesh-bridge: already running as $(pid_of)"; return 0
      fi
      if [ -z "$force" ] && client_attached; then
        rm -f "$new"
        echo "mesh-bridge: its configuration changed and a client is attached; not restarting (the configuration in place kept)" >&2; return 1
      fi
      configuration_install; wire_check || return $?; write_plist || return $?
      [ -z "$job_changed" ] || echo "$LABEL: its new definition is in place and takes effect at its next load (login or boot); the loaded job runs on"
      launchctl kill SIGHUP "$DOM/$LABEL" || return $?
      echo "$LABEL: its configuration or images are installed; its heartbeat restarts the bridge on them (SIGHUP), the region kept"
      return 0
    fi
    # a loaded job that is not this heartbeat running (the --link form's mesh-flow, or a heartbeat that ended 0, which
    # KeepAlive does not start again) is booted out and loaded again, as before the heartbeat's job
    if [ -z "$p" ]; then echo "mesh-bridge: $LABEL is loaded and not running; loading it again"
    else
      if [ -z "$force" ] && client_attached; then
        rm -f "$new"
        echo "mesh-bridge: running another program (${running:-unknown}) with a client attached; not replacing it" >&2; return 1
      fi
      echo "mesh-bridge: running another program (${running:-unknown}); replacing it with the heartbeat's job"
    fi
    do_stop "$DOM" || { rm -f "$new"; return 1; }
  fi
  configuration_install; wire_check || return $?; write_plist || return $?
  launchctl bootstrap "$DOM" "$PLIST" || return $?
  # the job's heartbeat and its bridge awaited 8 s at most from here (the project's bound on a wait), the job left loaded
  deadline=$((SECONDS+8)); p=
  while [ "$SECONDS" -lt "$deadline" ]; do
    p=$(pid_of)
    if [ -n "$p" ] && "$STAT" --ready "$region" >/dev/null 2>&1; then
      echo "mesh-bridge: running as $p (its heartbeat), bridge $(bridge_of), region of $region_bytes bytes"
      return 0
    fi
    sleep 0.1
  done
  if [ -z "$p" ]; then
    echo "mesh-bridge: $DOM/$LABEL is loaded but its heartbeat had not started 8 s on; see $LOGDIR/$LABEL.log" >&2
  else
    echo "mesh-bridge: its heartbeat $p runs, but 8 s on no bridge was ready (mesh-stat --ready: no region $region of this version with its bridge alive); the job stays loaded; see $LOGDIR/$LABEL.log" >&2
  fi
  return 1
}

do_status() {
  p=$(pid_of)
  [ -n "$p" ] && echo "$LABEL pid $p (its heartbeat)" || echo "$LABEL not running"
  [ -x "$STAT" ] && "$STAT" "$region"
}

case "${1:-status}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_start restart ;;
  status)  do_status ;;
  ready)   "$STAT" --ready "$region" ;;
  *) echo "usage: $0 {start|stop|restart|status|ready}" >&2; exit 64 ;;
esac
