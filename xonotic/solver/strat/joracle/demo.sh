#!/bin/sh
# j-oracle demonstration: real Xonotic cartserver + mlx responder on the mini +
# the continuously-available j-oracle viewer, all three at once.
#
#   xonotic/solver/strat/joracle/demo.sh up      # the whole demonstration
#   xonotic/solver/strat/joracle/demo.sh attach  # responder + viewer onto a running cartserver
#   xonotic/solver/strat/joracle/demo.sh viewer  # viewer only, reattaches to a live run
#   xonotic/solver/strat/joracle/demo.sh status
#   xonotic/solver/strat/joracle/demo.sh down    # stops ONLY what this script started
#
# The run directory is a FIXED path, not a mktemp: that is what makes the demo
# resumable.  Kill any of the three pieces and bring it back with the same
# command; the viewer reattaches to the same telemetry file by itself and the
# responder resumes its RNG/cursor from its runstate.
#
# Rules honoured here, in code:
#   * port 26012 is refused outright, and any port already in use is refused;
#   * nothing is ever pkill'd; the server is asked to quit and the responder is
#     left to its own lifecycle;
#   * if another process already holds the mesh bridge's single client slot the
#     script STOPS and reports it instead of stealing the mesh.

set -eu

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
STRAT=$(cd -- "$HERE/.." && pwd)
XONOTIC=$(cd -- "$STRAT/../.." && pwd)
REPO=$(cd -- "$XONOTIC/.." && pwd)

PORT=${JORACLE_PORT:-26042}
VIEWER_PORT=${JORACLE_VIEWER_PORT:-8795}
RUNDIR=${JORACLE_RUNDIR:-/tmp/mesh-joracle}
MINI=${JORACLE_MINI:-mesh-mini}
MINI_RUN=${JORACLE_MINI_RUNDIR:-/tmp/mesh-joracle}
MINI_PY=${JORACLE_MINI_PYTHON:-\$HOME/.venv-mesh/bin/python}
ENGINE=${JORACLE_ENGINE:-$HOME/dox/xonotic/build-engine/darkplaces-dedicated}
BASEDIR=${JORACLE_BASEDIR:-$HOME/dox/xonotic/Xonotic}
MAP=${JORACLE_MAP:-fused}
BOTS=${JORACLE_BOTS:-12}
TEAMS=${JORACLE_TEAMS:-0}
SKILL=${JORACLE_SKILL:-4}
PEER_NODE=${JORACLE_PEER_NODE:-0}
OFF_POLICY=${JORACLE_OFF_POLICY:-3}
TELEMETRY="$MINI_RUN/output/live.jsonl"

say() { printf '[demo] %s\n' "$*" >&2; }
die() { printf '[demo] BLOCKED: %s\n' "$*" >&2; exit 3; }

port_free() {
  if command -v lsof >/dev/null 2>&1; then
    ! lsof -nP -iUDP:"$1" -iTCP:"$1" >/dev/null 2>&1
  else
    ! netstat -an 2>/dev/null | grep -q "\.$1 "
  fi
}

bridge_client() {
  "$REPO/bin/mesh-bridge.sh" status 2>/dev/null | tail -1 |
    sed -n 's/.*"client":\([0-9]*\).*/\1/p'
}

preflight() {
  [ "$PORT" = 26012 ] && die "port 26012 is off limits"
  [ -x "$ENGINE" ] || die "engine not found or not executable: $ENGINE"
  [ -d "$BASEDIR/data" ] || die "basedir has no data/: $BASEDIR"
  [ -f "$XONOTIC/payload-build/progs.dat" ] || die "no payload progs.dat; run xonotic/payload/build.sh"
  [ -f "$XONOTIC/payload-build/csprogs.dat" ] || die "no payload csprogs.dat; run xonotic/payload/build.sh"
  port_free "$PORT" || die "port $PORT is already in use; set JORACLE_PORT"
  port_free "$VIEWER_PORT" || die "viewer port $VIEWER_PORT is in use; set JORACLE_VIEWER_PORT"
  "$REPO/bin/mesh-bridge.sh" status >/dev/null 2>&1 || die "local mesh bridge is down; bin/mesh-bridge.sh start"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$MINI" true 2>/dev/null || die "cannot ssh $MINI"
  client=$(bridge_client)
  if [ -n "${client:-}" ] && [ "$client" != 0 ]; then
    if kill -0 "$client" 2>/dev/null; then
      die "mesh bridge client slot is held by pid $client:
       $(ps -o command= -p "$client" 2>/dev/null | cut -c1-140)
       the bridge serves ONE client; refusing to steal it. Wait for that run to
       finish, or point this demo at its telemetry with:
         $0 viewer   (JORACLE_TELEMETRY=host:/path/live.jsonl)"
    fi
  fi
}

stage() {
  mkdir -p "$RUNDIR/userdir/data/data" "$RUNDIR/logs"
  cp "$XONOTIC/payload-build/progs.dat"   "$RUNDIR/userdir/data/progs.dat"
  cp "$XONOTIC/payload-build/csprogs.dat" "$RUNDIR/userdir/data/csprogs.dat"
  cp "$XONOTIC/payload/cfg/gamemodes-payload.cfg" "$RUNDIR/userdir/data/gamemodes-payload.cfg" 2>/dev/null || true
  cat > "$RUNDIR/userdir/data/autoexec.cfg" <<EOF
exec gamemodes-payload.cfg
g_payload 1
g_payload_teams_override $TEAMS
sv_public 0
g_payload_warmup 2
g_payload_round_timelimit 180
g_payload_idle_time 3
minplayers 0
skill $SKILL
sv_autopause 0
sv_status_privacy 0
EOF
  say "staged $RUNDIR/userdir"
}

push_runtime() {
  ssh "$MINI" "mkdir -p $MINI_RUN/runtime $MINI_RUN/output"
  rsync -a --delete \
    --exclude '__pycache__' --exclude 'runs/curriculum' --exclude '*.npz' \
    "$REPO/rdma/" "$MINI:$MINI_RUN/runtime/rdma/"
  rsync -a --delete \
    --exclude '__pycache__' --exclude 'runs/curriculum' \
    "$XONOTIC/solver/" "$MINI:$MINI_RUN/runtime/xonotic/solver/"
  rsync -a --delete --exclude '__pycache__' \
    "$XONOTIC/payload/tools/" "$MINI:$MINI_RUN/runtime/xonotic/payload/tools/"
  say "runtime pushed to $MINI:$MINI_RUN/runtime"
}

start_server() {
  nohup "$ENGINE" -xonotic -basedir "$BASEDIR" -userdir "$RUNDIR/userdir" \
      +developer 0 +sv_public 0 +port "$PORT" +sv_autopause 0 \
      +g_payload 1 +g_payload_teams_override "$TEAMS" \
      +g_payload_round_timelimit 180 +timelimit 20 +maxplayers 24 \
      +bot_join_empty 1 +bot_number "$BOTS" +skill "$SKILL" +g_warmup 0 \
      +map "$MAP" >"$RUNDIR/logs/server.log" 2>&1 </dev/null &
  echo $! > "$RUNDIR/server.pid"
  say "cartserver pid $(cat "$RUNDIR/server.pid") on udp/$PORT map=$MAP bots=$BOTS -> $RUNDIR/logs/server.log"
}

start_responder() {
  ssh -o BatchMode=yes -f "$MINI" \
    "cd $MINI_RUN/runtime/xonotic && \
     PYTHONPATH=$MINI_RUN/runtime/xonotic:$MINI_RUN/runtime/xonotic/payload/tools \
     nohup $MINI_PY -m solver.strat.strat_responder \
       --train --peer-node $PEER_NODE \
       --off-policy-players $OFF_POLICY \
       --online-checkpoint $MINI_RUN/output/live.npz \
       --telemetry $TELEMETRY --append-telemetry \
       --environment joracle_demo --save-every 10 --save-secs 15 \
       --model-sample-every ${JORACLE_MODEL_SAMPLE_EVERY:-1} \
       > $MINI_RUN/output/responder.log 2>&1 &"
  say "responder launched on $MINI -> $MINI_RUN/output/responder.log"
}

start_viewer() {
  source_arg=${JORACLE_TELEMETRY:-$MINI:$TELEMETRY}
  # `exec` so the recorded pid IS the python process; a wrapping subshell would
  # make server.pid/viewer.pid point at a shell whose death leaves python behind.
  nohup /bin/sh -c "cd '$XONOTIC' && exec python3 -m solver.strat.joracle.server \
      --telemetry '$source_arg' --port '$VIEWER_PORT' \
      --server-address '127.0.0.1:$PORT' --map '$MAP' --note 'joracle demo'" \
    >"$RUNDIR/logs/viewer.log" 2>&1 </dev/null &
  echo $! > "$RUNDIR/viewer.pid"
  VIEWER_SOURCE=$source_arg
  say "viewer pid $(cat "$RUNDIR/viewer.pid") -> http://127.0.0.1:$VIEWER_PORT"
}

banner() {
  lan=$(ipconfig getifaddr en0 2>/dev/null || echo 127.0.0.1)
  cat >&2 <<EOF

  ------------------------------------------------------------------
  j-oracle viewer     http://127.0.0.1:$VIEWER_PORT
  xonotic client      launch Xonotic, press ~ for the console, then:
                          connect 127.0.0.1:$PORT
                      (from another machine on the LAN: connect $lan:$PORT)
  telemetry           ${VIEWER_SOURCE:-$MINI:$TELEMETRY}
  logs                $RUNDIR/logs/server.log
                      $MINI:$MINI_RUN/output/responder.log
  stop                $0 down
  ------------------------------------------------------------------

EOF
}

case "${1:-up}" in
  up)
    preflight
    stage
    push_runtime
    start_server
    sleep 3
    if [ "${JORACLE_SKIP_RESPONDER:-0}" = 1 ]; then
      # A responder is already serving the mesh from the mini (someone else's
      # long-running service).  The mini's bridge also has a single client slot,
      # so starting a second responder would take the mesh from theirs.  Skip it
      # and let the running one answer this server.
      say "JORACLE_SKIP_RESPONDER=1: leaving the responder on $MINI alone"
    else
      start_responder
    fi
    start_viewer
    banner
    ;;
  attach)
    # Responder + viewer only, against a cartserver that is ALREADY running and
    # already holds the mesh bridge's client slot.  This is the cooperative path
    # when someone else's match is in progress: it never touches their server and
    # never contends for node 0's single client slot.
    [ -x "$ENGINE" ] || die "engine missing (needed only to confirm the tree): $ENGINE"
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$MINI" true 2>/dev/null || die "cannot ssh $MINI"
    holder=$(bridge_client)
    [ -n "${holder:-}" ] && [ "$holder" != 0 ] \
      && say "attaching to the cartserver already holding the bridge: pid $holder" \
      || say "warning: no process currently holds the bridge client slot; the server may be between mesh_open retries"
    mkdir -p "$RUNDIR/logs"
    push_runtime
    start_responder
    start_viewer
    banner
    ;;
  viewer)
    port_free "$VIEWER_PORT" || die "viewer port $VIEWER_PORT is in use"
    mkdir -p "$RUNDIR/logs"
    start_viewer
    banner
    ;;
  status)
    say "bridge client: $(bridge_client)"
    for name in server viewer; do
      if [ -f "$RUNDIR/$name.pid" ] && kill -0 "$(cat "$RUNDIR/$name.pid")" 2>/dev/null; then
        say "$name: running pid $(cat "$RUNDIR/$name.pid")"
      else
        say "$name: not running"
      fi
    done
    ssh -o BatchMode=yes "$MINI" "pgrep -fl strat_responder || echo 'responder: not running'" 2>/dev/null || true
    ssh -o BatchMode=yes "$MINI" "wc -l $TELEMETRY 2>/dev/null || echo 'telemetry: none'" 2>/dev/null || true
    ;;
  down)
    # The server is asked to quit through its own console, never signalled hard.
    if [ -f "$RUNDIR/server.pid" ]; then
      pid=$(cat "$RUNDIR/server.pid")
      kill -TERM "$pid" 2>/dev/null && say "asked cartserver $pid to quit" || say "cartserver already gone"
      rm -f "$RUNDIR/server.pid"
    fi
    if [ -f "$RUNDIR/viewer.pid" ]; then
      pid=$(cat "$RUNDIR/viewer.pid")
      kill -TERM "$pid" 2>/dev/null && say "stopped viewer $pid" || say "viewer already gone"
      rm -f "$RUNDIR/viewer.pid"
    fi
    say "the responder on $MINI is left running; stop it there if you want it stopped"
    ;;
  *)
    echo "usage: $0 {up|attach|viewer|status|down}" >&2
    exit 2
    ;;
esac
