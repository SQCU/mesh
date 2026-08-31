#!/bin/sh
set -eu

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
STRAT=$(cd -- "$HERE/.." && pwd)
XONOTIC=$(cd -- "$STRAT/../.." && pwd)
REPO=$(cd -- "$XONOTIC/.." && pwd)

PORT=${JORACLE_PORT:-26042}
VIEWER_PORT=${JORACLE_VIEWER_PORT:-8795}
RUNDIR=${JORACLE_RUNDIR:-/tmp/mesh-joracle}
# Resolve the peer by IDENTITY at use time, not by a string baked into
# ~/.ssh/config. That config said 192.168.1.183 -- a subnet this machine has not
# been on for some time -- so every run reported "mesh-mini unreachable" while
# the node was up on 10.0.0.165 and on the fabric at 169.254.225.22. One stale
# string took out the responder, the telemetry viewer, and the conclusion that
# any of it worked. rdma/peers.py asks the fabric instead.
MINI=${JORACLE_MINI:-$(python3 "$REPO/rdma/peers.py" mesh-mini 2>/dev/null)}
[ -n "$MINI" ] || MINI=mesh-mini   # no live edge: fall through and fail loudly
MINI_RUN=${JORACLE_MINI_RUNDIR:-/tmp/mesh-joracle}
MINI_PY=${JORACLE_MINI_PYTHON:-\$HOME/.venv-mesh-uv/bin/python}
ENGINE=${JORACLE_ENGINE:-$XONOTIC/darkplaces-work/darkplaces-dedicated}
ASSETROOT=${JORACLE_ASSETROOT:-$HOME/dox/xonotic/Xonotic}
BASEDIR=$RUNDIR/basedir
TRAINING_ASSETS=${JORACLE_TRAINING_ASSETS:-}
BOTS=${JORACLE_BOTS:-30}
TEAMS=${JORACLE_TEAMS:-5}
CARTS=${JORACLE_CARTS:-4}
SKILL=${JORACLE_SKILL:-4}
MAXPLAYERS=${JORACLE_MAXPLAYERS:-64}
PEER_NODE=${JORACLE_PEER_NODE:-0}
OFF_POLICY=${JORACLE_OFF_POLICY:-3}
TELEMETRY="$MINI_RUN/output/live.jsonl"
MANIFEST="$RUNDIR/dev.manifest"
SERVER_SESSION=${JORACLE_SERVER_SESSION:-joracle-server}
CLIENT_SESSION=${JORACLE_CLIENT_SESSION:-joracle-client}
VIEWER_SESSION=${JORACLE_VIEWER_SESSION:-joracle-viewer}

say() { printf '[demo] %s\n' "$*" >&2; }
die() { printf '[demo] BLOCKED: %s\n' "$*" >&2; exit 3; }

port_free() {
  if command -v lsof >/dev/null 2>&1; then
    ! lsof -nP -iUDP:"$1" -iTCP:"$1" >/dev/null 2>&1
  else
    ! netstat -an 2>/dev/null | grep -q "\.$1 "
  fi
}

select_port() {
  requested=$1
  while ! port_free "$requested"; do requested=$((requested + 1)); done
  printf '%s\n' "$requested"
}

file_id() {
  cksum "$1" | awk '{print $1 ":" $2}'
}

# Identity of the SOURCE that produced the build. The manifest already proved
# deployed == payload-build; it could not say WHICH PROGRAM that was, so a
# running server could be reasoned about as if it were current source while
# being hours behind it. That ambiguity is what produced five progs.dat and a
# session spent debugging code that was not running.
#
# A build output has no diff to merge: progs.dat is a pure function of (source,
# compiler, flags). Two differing .dat are one program built twice, not two
# programs -- so the fix is never to reconcile them, it is to name the source
# and rebuild.
qcsrc_id() {
  find "$XONOTIC/qcsrc" \( -name '*.qc' -o -name '*.qh' -o -name '*.inc' \) \
    | sort | xargs cksum 2>/dev/null | cksum | awk '{print $1}'
}

write_manifest() {
  cat > "$MANIFEST" <<EOF
pid=$(cat "$RUNDIR/server.pid")
port=$PORT
engine=$ENGINE
engine_id=$(file_id "$ENGINE")
basedir=$BASEDIR
userdir=$RUNDIR/userdir
progs_id=$(file_id "$RUNDIR/userdir/data/progs.dat")
csprogs_id=$(file_id "$RUNDIR/userdir/data/csprogs.dat")
commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)
dirty=$(git -C "$REPO" status --porcelain -- xonotic/qcsrc | wc -l | tr -d ' ')
qcsrc_id=$(qcsrc_id)
set_id=$(sed -n 's/^set_id=//p' "$XONOTIC/payload-build/BUILD_MANIFEST" 2>/dev/null)
build_commit=$(sed -n 's/^commit=//p' "$XONOTIC/payload-build/BUILD_MANIFEST" 2>/dev/null)
build_dirty=$(sed -n 's/^dirty=//p' "$XONOTIC/payload-build/BUILD_MANIFEST" 2>/dev/null)
EOF
}

server_identity() {
  [ -f "$MANIFEST" ] && [ -f "$RUNDIR/server.pid" ] || return 1
  pid=$(cat "$RUNDIR/server.pid")
  kill -0 "$pid" 2>/dev/null || return 1
  manifest_engine=$(sed -n 's/^engine=//p' "$MANIFEST")
  manifest_port=$(sed -n 's/^port=//p' "$MANIFEST")
  manifest_userdir=$(sed -n 's/^userdir=//p' "$MANIFEST")
  manifest_engine_id=$(sed -n 's/^engine_id=//p' "$MANIFEST")
  manifest_progs_id=$(sed -n 's/^progs_id=//p' "$MANIFEST")
  manifest_csprogs_id=$(sed -n 's/^csprogs_id=//p' "$MANIFEST")
  [ "$manifest_engine_id" = "$(file_id "$manifest_engine")" ] || return 1
  [ "$manifest_progs_id" = "$(file_id "$manifest_userdir/data/progs.dat")" ] || return 1
  [ "$manifest_csprogs_id" = "$(file_id "$manifest_userdir/data/csprogs.dat")" ] || return 1
  [ "$manifest_progs_id" = "$(file_id "$XONOTIC/payload-build/progs.dat")" ] || return 1
  [ "$manifest_csprogs_id" = "$(file_id "$XONOTIC/payload-build/csprogs.dat")" ] || return 1
  # The QC source must not have moved under the running build. Without this the
  # server stays "identical" through any number of source edits, and every
  # observation of it is attributed to code that is not in it.
  manifest_qcsrc_id=$(sed -n 's/^qcsrc_id=//p' "$MANIFEST")
  [ -n "$manifest_qcsrc_id" ] || return 1
  [ "$manifest_qcsrc_id" = "$(qcsrc_id)" ] || return 1
  # The build SET must match too. progs/csprogs/menu are compiled separately but
  # share STAT indices, field offsets and string tables, so a mixed set is two
  # programs disagreeing about the wire format -- silently, since neither side
  # errors when a stat resolves to the wrong slot. build.sh records the set.
  bm=$XONOTIC/payload-build/BUILD_MANIFEST
  [ -f "$bm" ] || return 1
  manifest_set=$(sed -n 's/^set_id=//p' "$MANIFEST")
  [ -n "$manifest_set" ] && [ "$manifest_set" = "$(sed -n 's/^set_id=//p' "$bm")" ] || return 1
  cmd=$(ps -p "$pid" -o command= 2>/dev/null)
  case "$cmd" in
    "$manifest_engine"*"-userdir $manifest_userdir"*"+port $manifest_port"*) return 0 ;;
    *server-keep.sh*"$manifest_engine"*"-userdir $manifest_userdir"*"+port $manifest_port"*) return 0 ;;
  esac
  return 1
}

bridge_client() {
  "$REPO/bin/mesh-bridge.sh" status 2>/dev/null | tail -1 |
    sed -n 's/.*"client":\([0-9]*\).*/\1/p'
}

preflight() {
  [ "$PORT" = 26012 ] && PORT=26013
  [ -x "$ENGINE" ] || die "engine not found or not executable: $ENGINE"
  [ -d "$ASSETROOT/data" ] || die "asset root has no data/: $ASSETROOT"
  selected=$(select_port "$PORT")
  [ "$selected" = "$PORT" ] || say "udp/$PORT belongs to another process; dev server will use udp/$selected"
  PORT=$selected
  VIEWER_PORT=$(select_port "$VIEWER_PORT")
  "$REPO/bin/mesh-bridge.sh" status >/dev/null 2>&1 || "$REPO/bin/mesh-bridge.sh" start || true
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$MINI" true 2>/dev/null || say "mesh responder host $MINI is currently unreachable; local server and client still start"
  client=$(bridge_client)
  if [ -n "${client:-}" ] && [ "$client" != 0 ]; then
    if kill -0 "$client" 2>/dev/null; then
      SKIP_RESPONDER=1
      say "mesh bridge is already serving pid $client; local dev server and client start while that responder remains attached"
    fi
  fi
}

stage() {
  make -C "$XONOTIC/darkplaces-work" -j"${JOBS:-8}" sv-release
  "$XONOTIC/render/build-client.sh" sdl-release
  "$XONOTIC/payload/build.sh"
  mkdir -p "$RUNDIR/userdir/data/data" "$RUNDIR/logs" "$BASEDIR/data"
  find "$BASEDIR/data" -type l -delete
  for asset in "$ASSETROOT"/data/xonotic-*.pk3 "$ASSETROOT"/data/font-*.pk3; do
    [ -f "$asset" ] && ln -s "$asset" "$BASEDIR/data/$(basename "$asset")"
  done
  if [ -n "$TRAINING_ASSETS" ]; then
    for asset in "$TRAINING_ASSETS"/*.pk3; do
      [ -f "$asset" ] && ln -s "$asset" "$BASEDIR/data/$(basename "$asset")"
    done
  fi
  mkdir -p "$RUNDIR/userdir/data/maps"
  maps=$(python3 "$HERE/training_maps.py" "$BASEDIR/data" "$RUNDIR/userdir/data/maps")
  cp "$XONOTIC/payload-build/progs.dat"   "$RUNDIR/userdir/data/progs.dat"
  cp "$XONOTIC/payload-build/csprogs.dat" "$RUNDIR/userdir/data/csprogs.dat"
  cp "$XONOTIC/payload/cfg/gamemodes-payload.cfg" "$RUNDIR/userdir/data/gamemodes-payload.cfg" 2>/dev/null || true
  : > "$RUNDIR/userdir/data/autoexec.cfg"
  cat > "$RUNDIR/userdir/data/server.cfg" <<EOF
exec gamemodes-payload.cfg
g_payload 1
g_payload_teams_override $TEAMS
g_payload_cart_count $CARTS
sv_public 0
g_payload_warmup 2
g_payload_round_timelimit 180
g_payload_idle_time 3
minplayers 0
skill $SKILL
sv_autopause 0
sv_status_privacy 0
maxplayers $MAXPLAYERS
bot_join_empty 1
bot_number $BOTS
g_maplist "$maps"
g_maplist_shuffle 1
g_maplist_selectrandom 1
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
  tmux new-session -d -s "$SERVER_SESSION" /bin/sh "$HERE/server-keep.sh" \
    "$RUNDIR/logs/server.log" "$ENGINE" -xonotic -basedir "$BASEDIR" \
    -userdir "$RUNDIR/userdir" +developer 0 +sv_public 0 +port "$PORT" \
    +sv_autopause 0
  tmux display-message -p -t "$SERVER_SESSION" '#{pane_pid}' > "$RUNDIR/server.pid"
  write_manifest
  say "cartserver pid $(cat "$RUNDIR/server.pid") on udp/$PORT server-owned map rotation -> $RUNDIR/logs/server.log"
}

start_client() {
  tmux new-session -d -s "$CLIENT_SESSION" /bin/sh "$HERE/run-logged.sh" \
    "$RUNDIR/logs/client-supervisor.log" env \
    JORACLE_CLIENT_BIN="$XONOTIC/darkplaces-work/darkplaces-sdl" \
    JORACLE_BASEDIR="$BASEDIR" \
    JORACLE_CLIENT_USERDIR="$RUNDIR/client-userdir" \
    JORACLE_CLIENT_LOG="$RUNDIR/logs/client.log" \
    JORACLE_CLIENT_STATE="$RUNDIR/client-engine.pid" \
    "$HERE/client-keep.sh" "127.0.0.1:$PORT"
  tmux display-message -p -t "$CLIENT_SESSION" '#{pane_pid}' > "$RUNDIR/client.pid"
  say "client supervisor pid $(cat "$RUNDIR/client.pid") auto-connecting to 127.0.0.1:$PORT"
}

start_responder() {
  ssh -o BatchMode=yes -f "$MINI" \
    "cd $MINI_RUN/runtime/xonotic && \
     PYTHONPATH=$MINI_RUN/runtime/xonotic:$MINI_RUN/runtime/xonotic/payload/tools \
     nohup $MINI_PY -m solver.strat.strat_responder \
       --train --peer-node $PEER_NODE \
       --off-policy-players $OFF_POLICY \
       --online-checkpoint $MINI_RUN/output/live.npz \
       --allow-arch-mismatch \
       --telemetry $TELEMETRY --append-telemetry \
       --environment joracle_demo --save-every 10 --save-secs 15 \
       --model-sample-every ${JORACLE_MODEL_SAMPLE_EVERY:-1} \
       > $MINI_RUN/output/responder.log 2>&1 &"
  say "responder launched on $MINI -> $MINI_RUN/output/responder.log"
}

start_viewer() {
  source_arg=${JORACLE_TELEMETRY:-$MINI:$TELEMETRY}
  tmux new-session -d -s "$VIEWER_SESSION" /bin/sh "$HERE/run-logged.sh" \
    "$RUNDIR/logs/viewer.log" env PYTHONPATH="$XONOTIC" python3 -m solver.strat.joracle.server \
    --telemetry "$source_arg" --port "$VIEWER_PORT" \
    --server-address "127.0.0.1:$PORT" --map "server rotation" --note "joracle demo"
  tmux display-message -p -t "$VIEWER_SESSION" '#{pane_pid}' > "$RUNDIR/viewer.pid"
  VIEWER_SOURCE=$source_arg
  say "viewer pid $(cat "$RUNDIR/viewer.pid") -> http://127.0.0.1:$VIEWER_PORT"
}

banner() {
  lan=$(ipconfig getifaddr en0 2>/dev/null || echo 127.0.0.1)
  cat >&2 <<EOF

  ------------------------------------------------------------------
  j-oracle viewer     http://127.0.0.1:$VIEWER_PORT
  xonotic client      automatically connected to 127.0.0.1:$PORT
  LAN players         connect $lan:$PORT
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
    start_client
    sleep 3
    if [ "${JORACLE_SKIP_RESPONDER:-${SKIP_RESPONDER:-0}}" = 1 ]; then
      say "JORACLE_SKIP_RESPONDER=1: leaving the responder on $MINI alone"
    else
      start_responder
    fi
    start_viewer
    banner
    ;;
  attach)
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
    VIEWER_PORT=$(select_port "$VIEWER_PORT")
    mkdir -p "$RUNDIR/logs"
    start_viewer
    banner
    ;;
  responder)
    push_runtime
    start_responder
    ;;
  status)
    say "bridge client: $(bridge_client)"
    if server_identity; then
      say "server: verified dev server pid $(cat "$RUNDIR/server.pid") manifest=$MANIFEST"
    elif [ -f "$RUNDIR/server.pid" ]; then
      say "server: pidfile is stale or belongs to a different executable/configuration"
    else
      say "server: not running"
    fi
    for name in viewer client; do
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
    if [ -f "$RUNDIR/client.pid" ]; then
      pid=$(cat "$RUNDIR/client.pid")
      kill -TERM "$pid" 2>/dev/null && say "stopped client supervisor $pid" || say "client supervisor already gone"
      rm -f "$RUNDIR/client.pid"
    fi
    say "the responder on $MINI is left running; stop it there if you want it stopped"
    ;;
  *)
    echo "usage: $0 {up|attach|viewer|responder|status|down}" >&2
    exit 2
    ;;
esac
