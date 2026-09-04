#!/bin/sh
set -eu

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
STRAT=$(cd -- "$HERE/.." && pwd)
XONOTIC=$(cd -- "$STRAT/../.." && pwd)
REPO=$(cd -- "$XONOTIC/.." && pwd)
MESH_PY="$REPO/bin/mesh-python"

PORT=${JORACLE_PORT:-26042}
VIEWER_PORT=${JORACLE_VIEWER_PORT:-8787}
RUNDIR=${JORACLE_RUNDIR:-/tmp/mesh-joracle}
MINI=${JORACLE_MINI:-mesh-mini}
MINI_RUN=${JORACLE_MINI_RUNDIR:-/tmp/mesh-joracle}
MINI_PY=/usr/local/mesh/bin/mesh-python
ENGINE=${JORACLE_ENGINE:-$XONOTIC/darkplaces-work/darkplaces-dedicated}
ASSETROOT=${JORACLE_ASSETROOT:-$HOME/dox/xonotic/Xonotic}
BASEDIR=$RUNDIR/basedir
TRAINING_ASSETS=${JORACLE_TRAINING_ASSETS:-}
PBRPK3=${JORACLE_PBR_PK3:-$XONOTIC/render-build/zzzzz-mesh-pbr.pk3}
RUNTIMEPK3=${JORACLE_RUNTIME_PK3:-$XONOTIC/payload-build/zzzzzz-mesh-runtime.pk3}
BOTS=${JORACLE_BOTS:-255}
TEAMS=${JORACLE_TEAMS:-256}
CARTS=${JORACLE_CARTS:-32}
SKILL=${JORACLE_SKILL:-4}
MAXPLAYERS=${JORACLE_MAXPLAYERS:-256}
MAPLIST=${JORACLE_MAPLIST:-}
PEER_NODE=${JORACLE_PEER_NODE:-0}
OFF_POLICY=${JORACLE_OFF_POLICY:-3}
TELEMETRY="$MINI_RUN/output/live.jsonl"
MANIFEST="$RUNDIR/dev.manifest"
SERVER_SESSION=${JORACLE_SERVER_SESSION:-joracle-server}
CLIENT_SESSION=${JORACLE_CLIENT_SESSION:-joracle-client}
MINI_REACHABLE=0

say() { printf '[demo] %s\n' "$*" >&2; }
ssh_node() { ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$MINI" "$@"; }
ssh_node_bg() { ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -f "$MINI" "$@"; }
SSH_TRANSPORT="ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

file_id() {
  cksum "$1" | awk '{print $1 ":" $2}'
}

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
client_engine=$XONOTIC/darkplaces-work/darkplaces-sdl
client_engine_id=$(file_id "$XONOTIC/darkplaces-work/darkplaces-sdl")
basedir=$BASEDIR
userdir=$RUNDIR/userdir
assets_id=$(file_id "$RUNDIR/assets.manifest")
pbr_id=$(file_id "$PBRPK3")
runtime_id=$(file_id "$RUNTIMEPK3")
progs_id=$(file_id "$RUNDIR/userdir/data/progs.dat")
csprogs_id=$(file_id "$RUNDIR/userdir/data/csprogs.dat")
menu_id=$(file_id "$RUNDIR/userdir/data/menu.dat")
effectinfo_id=$(file_id "$RUNDIR/userdir/data/effectinfo.txt")
runtime=$($MESH_PY "$REPO/bin/mesh-runtime-id.py")
branch=$(git -C "$REPO" symbolic-ref --short HEAD 2>/dev/null || printf main)
dirty=$(git -C "$REPO" status --porcelain -- xonotic/qcsrc | wc -l | tr -d ' ')
qcsrc_id=$(qcsrc_id)
set_id=$(sed -n 's/^set_id=//p' "$XONOTIC/payload-build/BUILD_MANIFEST" 2>/dev/null)
build_branch=$(sed -n 's/^branch=//p' "$XONOTIC/payload-build/BUILD_MANIFEST" 2>/dev/null)
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
  manifest_client_engine=$(sed -n 's/^client_engine=//p' "$MANIFEST")
  manifest_client_engine_id=$(sed -n 's/^client_engine_id=//p' "$MANIFEST")
  manifest_assets_id=$(sed -n 's/^assets_id=//p' "$MANIFEST")
  manifest_pbr_id=$(sed -n 's/^pbr_id=//p' "$MANIFEST")
  manifest_runtime_id=$(sed -n 's/^runtime_id=//p' "$MANIFEST")
  manifest_progs_id=$(sed -n 's/^progs_id=//p' "$MANIFEST")
  manifest_csprogs_id=$(sed -n 's/^csprogs_id=//p' "$MANIFEST")
  manifest_menu_id=$(sed -n 's/^menu_id=//p' "$MANIFEST")
  manifest_effectinfo_id=$(sed -n 's/^effectinfo_id=//p' "$MANIFEST")
  [ "$manifest_engine_id" = "$(file_id "$manifest_engine")" ] || return 1
  [ "$manifest_client_engine_id" = "$(file_id "$manifest_client_engine")" ] || return 1
  [ "$manifest_assets_id" = "$(file_id "$RUNDIR/assets.manifest")" ] || return 1
  [ "$manifest_pbr_id" = "$(file_id "$PBRPK3")" ] || return 1
  [ "$manifest_runtime_id" = "$(file_id "$RUNTIMEPK3")" ] || return 1
  [ "$manifest_progs_id" = "$(file_id "$manifest_userdir/data/progs.dat")" ] || return 1
  [ "$manifest_csprogs_id" = "$(file_id "$manifest_userdir/data/csprogs.dat")" ] || return 1
  [ "$manifest_menu_id" = "$(file_id "$manifest_userdir/data/menu.dat")" ] || return 1
  [ "$manifest_effectinfo_id" = "$(file_id "$manifest_userdir/data/effectinfo.txt")" ] || return 1
  [ "$manifest_progs_id" = "$(file_id "$XONOTIC/payload-build/progs.dat")" ] || return 1
  [ "$manifest_csprogs_id" = "$(file_id "$XONOTIC/payload-build/csprogs.dat")" ] || return 1
  [ "$manifest_menu_id" = "$(file_id "$XONOTIC/payload-build/menu.dat")" ] || return 1
  [ "$manifest_effectinfo_id" = "$(file_id "$XONOTIC/payload-build/effectinfo.txt")" ] || return 1
  manifest_qcsrc_id=$(sed -n 's/^qcsrc_id=//p' "$MANIFEST")
  [ -n "$manifest_qcsrc_id" ] || return 1
  [ "$manifest_qcsrc_id" = "$(qcsrc_id)" ] || return 1
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
  [ -x "$ENGINE" ] || say "engine not found or not executable: $ENGINE"
  [ -d "$ASSETROOT/data" ] || say "asset root has no data/: $ASSETROOT"
  "$REPO/bin/mesh-bridge.sh" status >/dev/null 2>&1 || "$REPO/bin/mesh-bridge.sh" start || true
  if ssh_node true 2>/dev/null; then
    MINI_REACHABLE=1
  else
    MINI_REACHABLE=0
    say "mesh responder host $MINI is currently unreachable; local server and client still start"
  fi
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
  for asset in "$ASSETROOT"/data/*.pk3; do
    if [ -f "$asset" ]; then ln -sfn "$asset" "$BASEDIR/data/$(basename "$asset")"; else say "asset unavailable: $asset"; fi
  done
  if [ -n "$TRAINING_ASSETS" ]; then
    for asset in "$TRAINING_ASSETS"/*.pk3; do
      if [ -f "$asset" ]; then ln -sfn "$asset" "$BASEDIR/data/$(basename "$asset")"; else say "training asset unavailable: $asset"; fi
    done
  fi
  for asset in "$XONOTIC"/mapgen/build/*.pk3; do
    if [ -f "$asset" ]; then ln -sfn "$asset" "$BASEDIR/data/$(basename "$asset")"; else say "generated map asset unavailable: $asset"; fi
  done
  "$MESH_PY" "$XONOTIC/render/pbr-materials.py" "$BASEDIR" "$PBRPK3"
  ln -sfn "$PBRPK3" "$BASEDIR/data/$(basename "$PBRPK3")"
  ln -sfn "$RUNTIMEPK3" "$BASEDIR/data/$(basename "$RUNTIMEPK3")"
  : > "$RUNDIR/assets.manifest"
  for asset in "$BASEDIR"/data/*.pk3; do
    target=$(readlink "$asset" 2>/dev/null || printf '%s' "$asset")
    printf '%s\t%s\t%s\n' "$(basename "$asset")" "$(file_id "$target")" "$target" >> "$RUNDIR/assets.manifest"
  done
  mkdir -p "$RUNDIR/userdir/data/maps"
  maps=$("$MESH_PY" "$HERE/training_maps.py" "$BASEDIR/data" "$RUNDIR/userdir/data/maps" "$TEAMS" "$CARTS" $MAPLIST)
  printf '%s\n' "$maps" > "$RUNDIR/maps.list"
  cp "$XONOTIC/payload-build/progs.dat"   "$RUNDIR/userdir/data/progs.dat"
  cp "$XONOTIC/payload-build/csprogs.dat" "$RUNDIR/userdir/data/csprogs.dat"
  cp "$XONOTIC/payload-build/menu.dat" "$RUNDIR/userdir/data/menu.dat"
  cp "$XONOTIC/payload-build/effectinfo.txt" "$RUNDIR/userdir/data/effectinfo.txt"
  cp "$XONOTIC/payload/cfg/gamemodes-payload.cfg" "$RUNDIR/userdir/data/gamemodes-payload.cfg" 2>/dev/null || true
  : > "$RUNDIR/userdir/data/autoexec.cfg"
  cat > "$RUNDIR/userdir/data/server.cfg" <<EOF
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
  [ "$MINI_REACHABLE" = 1 ] || return 1
  ssh_node "mkdir -p $MINI_RUN/runtime $MINI_RUN/output" || return 1
  rsync -e "$SSH_TRANSPORT" -a --delete \
    --exclude '__pycache__' --exclude 'runs/curriculum' --exclude '*.npz' \
    "$REPO/rdma/" "$MINI:$MINI_RUN/runtime/rdma/" || return 1
  rsync -e "$SSH_TRANSPORT" -a --delete \
    --exclude '__pycache__' --exclude 'runs/curriculum' \
    "$XONOTIC/solver/" "$MINI:$MINI_RUN/runtime/xonotic/solver/" || return 1
  rsync -e "$SSH_TRANSPORT" -a --delete --exclude '__pycache__' \
    "$XONOTIC/payload/tools/" "$MINI:$MINI_RUN/runtime/xonotic/payload/tools/" || return 1
  say "runtime pushed to $MINI:$MINI_RUN/runtime"
}

start_server() {
  maps=$(cat "$RUNDIR/maps.list")
  first_map=${maps%% *}
  autoscreenshot_mass=$(awk '/"classname"[[:space:]]+"info_autoscreenshot"/ { n++ } END { print n + 0 }' "$RUNDIR"/userdir/data/maps/*.ent)
  tmux new-session -d -s "$SERVER_SESSION" /bin/sh "$HERE/server-keep.sh" \
    "$RUNDIR/logs/server.log" "$ENGINE" -norunaway -xonotic -basedir "$BASEDIR" \
    -userdir "$RUNDIR/userdir" +developer 0 +sv_public 0 +port "$PORT" \
    +maxplayers "$MAXPLAYERS" +exec gamemodes-payload.cfg +g_payload 1 \
    +g_max_info_autoscreenshot "$autoscreenshot_mass" \
    +g_payload_teams_override "$TEAMS" +g_payload_warmup 2 \
    +g_payload_round_timelimit 180 +g_payload_idle_time 3 +minplayers 0 \
    +skill "$SKILL" +sv_autopause 0 +sv_status_privacy 0 +bot_join_empty 1 \
    +bot_number "$BOTS" +g_maplist "$maps" +g_maplist_shuffle 1 \
    +g_maplist_selectrandom 1 +map "$first_map"
  tmux display-message -p -t "$SERVER_SESSION" '#{pane_pid}' > "$RUNDIR/server.pid"
  write_manifest
  say "cartserver pid $(cat "$RUNDIR/server.pid") on udp/$PORT server-owned map rotation -> $RUNDIR/logs/server.log"
}

start_client() {
  if tmux has-session -t "$CLIENT_SESSION" 2>/dev/null; then
    tmux display-message -p -t "$CLIENT_SESSION" '#{pane_pid}' > "$RUNDIR/client.pid"
    say "client already supervised by pid $(cat "$RUNDIR/client.pid")"
    return
  fi
  tmux new-session -d -s "$CLIENT_SESSION" /bin/sh "$HERE/run-logged.sh" \
    "$RUNDIR/logs/client-supervisor.log" env \
    JORACLE_CLIENT_BIN="$XONOTIC/darkplaces-work/darkplaces-sdl" \
    JORACLE_BASEDIR="$BASEDIR" \
    JORACLE_CLIENT_USERDIR="$RUNDIR/client-userdir" \
    JORACLE_CLIENT_LOG="$RUNDIR/logs/client.log" \
    JORACLE_CLIENT_EVENTS="$RUNDIR/logs/client.events" \
    JORACLE_CLIENT_STATE="$RUNDIR/client-engine.pid" \
    JORACLE_CLIENT_SESSIONID="$CLIENT_SESSION" \
    JORACLE_ASSET_MANIFEST="$RUNDIR/assets.manifest" \
    "$HERE/client-keep.sh" "127.0.0.1:$PORT"
  tmux display-message -p -t "$CLIENT_SESSION" '#{pane_pid}' > "$RUNDIR/client.pid"
  say "client supervisor pid $(cat "$RUNDIR/client.pid") auto-connecting to 127.0.0.1:$PORT"
}

start_responder() {
  ssh_node_bg \
    "cd $MINI_RUN/runtime/xonotic && \
     PYTHONPATH=$MINI_RUN/runtime/xonotic:$MINI_RUN/runtime/xonotic/payload/tools \
     nohup $MINI_PY -m solver.strat.strat_responder \
       --train --peer-node $PEER_NODE \
       --off-policy-players $OFF_POLICY \
       --online-checkpoint $MINI_RUN/output/live.npz \
       --telemetry $TELEMETRY --append-telemetry \
       --environment joracle_demo --save-every 10 --save-secs 15 \
       --model-sample-every ${JORACLE_MODEL_SAMPLE_EVERY:-50} \
       > $MINI_RUN/output/responder.log 2>&1 &"
  say "responder launched on $MINI -> $MINI_RUN/output/responder.log"
}

start_viewer() {
  source_arg=${JORACLE_TELEMETRY:-$MINI:$TELEMETRY}
  VIEWER_SOURCE=$source_arg
  if ! curl -fsS "http://127.0.0.1:$VIEWER_PORT/latest.json" >/dev/null 2>&1; then
    launchctl kickstart -k "gui/$UID/io.mesh.observer" >/dev/null 2>&1 || true
    n=0
    while ! curl -fsS "http://127.0.0.1:$VIEWER_PORT/latest.json" >/dev/null 2>&1 && [ "$n" -lt 10 ]; do
      sleep 1
      n=$((n + 1))
    done
  fi
  curl -fsS "http://127.0.0.1:$VIEWER_PORT/latest.json" >/dev/null 2>&1 \
    && say "whole-mesh reporter serving http://127.0.0.1:$VIEWER_PORT" \
    || say "whole-mesh reporter is not answering on http://127.0.0.1:$VIEWER_PORT"
}

banner() {
  lan=$(ipconfig getifaddr en0 2>/dev/null || echo 127.0.0.1)
  cat >&2 <<EOF

  ------------------------------------------------------------------
  mesh phase viewer   http://127.0.0.1:$VIEWER_PORT
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
    stage || say "staging did not complete; continuing with every independently available component"
    push_runtime || say "runtime push did not complete; continuing locally"
    start_server || say "server did not start; continuing with client, responder, and viewer"
    start_client || say "client did not start; continuing with responder and viewer"
    sleep 3
    if [ "${JORACLE_SKIP_RESPONDER:-${SKIP_RESPONDER:-0}}" = 1 ]; then
      say "JORACLE_SKIP_RESPONDER=1: leaving the responder on $MINI alone"
    else
      start_responder || say "responder did not start; viewer remains available"
    fi
    start_viewer || say "viewer did not start"
    banner
    ;;
  attach)
    ssh_node true 2>/dev/null || say "cannot currently ssh $MINI; local viewer still starts"
    holder=$(bridge_client)
    [ -n "${holder:-}" ] && [ "$holder" != 0 ] \
      && say "attaching to the cartserver already holding the bridge: pid $holder" \
      || say "warning: no process currently holds the bridge client slot; the server may be between mesh_open retries"
    mkdir -p "$RUNDIR/logs"
    push_runtime || say "runtime push did not complete"
    start_responder || say "responder did not start"
    start_viewer || say "viewer did not start"
    banner
    ;;
  viewer)
    mkdir -p "$RUNDIR/logs"
    start_viewer
    banner
    ;;
  responder)
    push_runtime || say "runtime push did not complete"
    start_responder || say "responder did not start"
    ;;
  client)
    mkdir -p "$RUNDIR/logs"
    start_client
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
    for name in client; do
      if [ -f "$RUNDIR/$name.pid" ] && kill -0 "$(cat "$RUNDIR/$name.pid")" 2>/dev/null; then
        say "$name: running pid $(cat "$RUNDIR/$name.pid")"
      else
        say "$name: not running"
      fi
    done
    curl -fsS "http://127.0.0.1:$VIEWER_PORT/latest.json" >/dev/null 2>&1 \
      && say "viewer: whole-mesh reporter answering on $VIEWER_PORT" \
      || say "viewer: whole-mesh reporter unavailable on $VIEWER_PORT"
    if [ -f "$RUNDIR/client-engine.health" ]; then
      sed 's/^/[demo] client engine: /' "$RUNDIR/client-engine.health" >&2
    else
      say "client engine: no health record"
    fi
    ssh_node "pgrep -fl strat_responder || echo 'responder: not running'" 2>/dev/null || true
    ssh_node "wc -l $TELEMETRY 2>/dev/null || echo 'telemetry: none'" 2>/dev/null || true
    ;;
  down)
    if [ -f "$RUNDIR/server.pid" ]; then
      pid=$(cat "$RUNDIR/server.pid")
      kill -TERM "$pid" 2>/dev/null && say "asked cartserver $pid to quit" || say "cartserver already gone"
      rm -f "$RUNDIR/server.pid"
    fi
    if [ -f "$RUNDIR/client.pid" ]; then
      pid=$(cat "$RUNDIR/client.pid")
      kill -TERM "$pid" 2>/dev/null && say "stopped client supervisor $pid" || say "client supervisor already gone"
      while tmux has-session -t "$CLIENT_SESSION" 2>/dev/null; do sleep 1; done
      rm -f "$RUNDIR/client.pid"
    fi
    say "the responder on $MINI is left running; stop it there if you want it stopped"
    ;;
  *)
    echo "usage: $0 {up|attach|viewer|responder|client|status|down}" >&2
    exit 2
    ;;
esac
