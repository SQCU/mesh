#!/bin/bash
# A node's user layers move to their upstreams between calls, each to images built and checked before its turn
# (README.md, Convergence; design/recovery.md).  With no argument, the mesh's own pass, which the admin user's agent
# io.mesh.converge runs (bin/mesh-user.sh writes it): the mesh checkout this script is in, built at its upstream in a
# stage, its images and bindings installed by rename in a turn once the node is quiet, the turn lasting until the bridge
# has stopped itself on its new binary (a new libmesh is installed with the binary), within wait_s.  Its subcommands
# are the node's convergence for any consumer's layer (metal-microbench tools/mesh/node-setup.sh converges its rank by
# them):
#   held CHECKOUT    prints why and succeeds where CHECKOUT has work its upstream lacks (nothing is fetched)
#   idle             succeeds where the node carries no call now, else prints why
#   quiet            succeeds once the node has been idle quiet_s on end, within wait_s; else prints why
#   begin LABEL PID  the node's one converging turn, LABEL's while PID runs: announced (node-info `converging`) for
#                    notice_s, then the node idle again; else prints why
#   end LABEL        LABEL's turn ended
#   stamp REVISION   what the mesh's layer converges to at REVISION: the revision and its recorded documents' digests
#   off              the node out of convergence, the operator's choice: the recorded convergence document and every
#                    layer's agent (<label>.converge) removed once no pass runs; mesh-user.sh's CONVERGENCE puts it back
# The node's state is ~/.cache/mesh: hold/<name> (a hold: its first word the unix second it ends, the rest why; any
# process that uses the node writes one, a driver for its calls, a measurement for its run), idle.d/<label> (a
# consumer's probe: an executable that succeeds printing the pids of its resident processes where they serve no call,
# else fails printing why), converging/ (the turn), converge/<label> (what LABEL's install last converged to, .failed
# the inputs whose staged build failed, .state its last outcome, .build.log its last staged build), facts/<label> (what
# the layer runs, which node-info publishes), stage/<repository> (the staged worktrees).  The configuration is the
# operand document bin/mesh-user.sh canonicalizes (etc/converge/README.md): period_s, quiet_s, notice_s, wait_s and
# gpu_percent, with no default.
set -u
export PATH=$PATH:/usr/local/mesh/bin:/usr/bin:/bin:/usr/sbin:/sbin
S=$HOME/.cache/mesh; LABEL=io.mesh.bridge
SUPPORT="$HOME/Library/Application Support"; CONF="$SUPPORT/io.mesh.converge/configuration.plist"
M=${MESH_CHECKOUT:-$(cd "$(dirname "$0")/.." && pwd -P)}
why=; note=; configured=1; period_s=; quiet_s=; notice_s=; wait_s=; gpu_percent=
for key in period_s quiet_s notice_s wait_s gpu_percent; do
  value=$(plutil -extract "$key" raw -o - "$CONF" 2>/dev/null) && [ -n "$value" ] || { configured=; break; }
  printf -v "$key" '%s' "$value"
done
configured() { [ -n "$configured" ] || why="no convergence configuration ($CONF: bin/mesh-user.sh CONFIGURATION CHECKOUT CONVERGENCE)"; [ -n "$configured" ]; }
short() { git -C "$M" rev-parse --short "$1" 2>/dev/null || printf '%s' "$1"; }

held() {
  local c=$1
  git -C "$c" rev-parse -q --verify '@{u}' >/dev/null 2>&1 || { why="$c follows no upstream"; return 0; }
  git -C "$c" diff --quiet HEAD -- 2>/dev/null || { why="$c has uncommitted changes"; return 0; }
  [ "$(git -C "$c" rev-list --count '@{u}..HEAD')" = 0 ] || { why="$c has commits its upstream lacks"; return 0; }
  return 1
}

idle() {
  local now f e r g s since vouched out others
  configured || return 1
  now=$(date +%s)
  for f in "$S"/hold/*; do
    [ -f "$f" ] || continue
    read -r e r < "$f" 2>/dev/null
    case ${e:-} in ''|*[!0-9]*) why="hold ${f##*/} names no end (its first word, a unix second)"; return 1 ;; esac
    [ "$e" -le "$now" ] || { why="held by ${f##*/} until $(date -r "$e" '+%F %T')${r:+: $r}"; return 1; }
  done
  for f in "$S"/converging/*; do
    [ -f "$f" ] && [ "${f##*/}" != "${SELF:-}" ] || continue
    read -r e r < "$f" 2>/dev/null
    case ${e:-} in ''|*[!0-9]*) continue ;; esac
    [ "$e" -le "$now" ] || { why="${f##*/} converges"; return 1; }
  done
  s=$("$M/bin/mesh-stat" 2>/dev/null)
  case $s in *'"idle":false'*) why="the bridge carries a call"; return 1 ;; esac
  since=$(printf '%s' "$s" | sed -n 's/.*"idle_since":\([0-9]*\).*/\1/p')
  [ -z "$since" ] || [ $((now - since)) -ge "$quiet_s" ] || { why="the bridge carried a call $((now - since)) s ago"; return 1; }
  g=$(ioreg -r -d 1 -c IOAccelerator 2>/dev/null | sed -n 's/.*"Device Utilization %"=\([0-9]*\).*/\1/p' | sort -n | tail -1)
  [ "${g:-0}" -le "$gpu_percent" ] || { why="the GPU is $g % busy"; return 1; }
  vouched=" $(pgrep -x -U "$(id -u)" mesh-flow | paste -sd' ' -) "
  for f in "$S"/idle.d/*; do
    [ -x "$f" ] || continue
    out=$("$f" 2>&1) || { why="${f##*/}: ${out:-it serves a call}"; return 1; }
    vouched="$vouched $(printf '%s' "$out" | tr -s '[:space:]' ' ') "
  done
  others=$(lsof -nP -u "$(id -un)" -d txt -Fpn 2>/dev/null \
    | awk -v v="$vouched" '/^p/{p=substr($0,2)} /^n.*\/libmesh\.dylib$/ && index(v, " " p " ") == 0 {print p}' | sort -un | paste -sd, -)
  [ -z "$others" ] || { why="mesh clients run on this node (pids $others)"; return 1; }
  return 0
}

quiet() {
  local start since='' last=''
  configured || return 1
  start=$(date +%s)
  while :; do
    if idle; then
      since=${since:-$(date +%s)}
      [ $(($(date +%s) - since)) -lt "$quiet_s" ] || return 0
    else since=; last=$why; fi
    [ $(($(date +%s) - start)) -lt "$wait_s" ] || { why=${last:-"not idle $quiet_s s on end within $wait_s s"}; return 1; }
    sleep 2
  done
}

end() {
  rm -f "$S/converging/$1"
  [ "$(cat "$S/converging.lock/label" 2>/dev/null)" != "$1" ] || rm -rf "$S/converging.lock"
  return 0
}

begin() {
  local label=$1 pid=$2 lock=$S/converging.lock owner
  configured || return 1
  mkdir -p "$S/converging"
  if ! mkdir "$lock" 2>/dev/null; then
    owner=$(cat "$lock/pid" 2>/dev/null)
    if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then why="$(cat "$lock/label" 2>/dev/null) converges"; return 1; fi
    rm -rf "$lock"; mkdir "$lock" 2>/dev/null || { why="another layer took the turn"; return 1; }
  fi
  printf '%s\n' "$pid" > "$lock/pid"; printf '%s\n' "$label" > "$lock/label"
  printf '%s %s\n' $(($(date +%s) + notice_s + period_s)) "$label" > "$S/converging/$label"
  sleep "$notice_s"
  SELF=$label
  idle || { end "$label"; return 1; }
}

stamp() {
  local doc conv
  doc=$(cat "$SUPPORT/io.mesh.bridge/document" 2>/dev/null); conv=$(cat "$SUPPORT/io.mesh.converge/document" 2>/dev/null)
  printf '%s %s %s\n' "$(git -C "$M" rev-parse "$1")" "$(shasum -a 256 < "$doc" 2>/dev/null | cut -c1-64)" "$(shasum -a 256 < "$conv" 2>/dev/null | cut -c1-64)"
}

report() {
  local line="$1${2:+: $2}"
  mkdir -p "$S/converge"
  [ "$line" != "$(cat "$S/converge/$LABEL.state" 2>/dev/null)" ] || return 0
  printf '[%s] %s\n' "$(date '+%F %T')" "$line"
  printf '%s\n' "$line" > "$S/converge/$LABEL.state"
}

signed() { codesign -dvv "$1" 2>&1 | grep -q '^Authority='; }

stage() {
  local st=$S/stage/mesh log=$S/converge/$LABEL.build.log venv=$HOME/.venv-mesh-uv sum
  mkdir -p "$S/stage"
  [ -e "$st/.git" ] && git -C "$st" checkout -q -f --detach "$1" 2>"$log" || {
    rm -rf "$st"; git -C "$M" worktree prune
    git -C "$M" worktree add -q --detach "$st" "$1" 2>"$log" || { why="git worktree add failed ($log)"; return 1; }; }
  st=$(cd "$st" && pwd -P)
  taskpolicy -b make -C "$st/rdma" all > "$log" 2>&1 || { why="make -C rdma all failed ($log)"; return 1; }
  taskpolicy -b "$venv/bin/python" "$st/rdma/mesh_c_build.py" >> "$log" 2>&1 || { why="the bindings do not build ($log)"; return 1; }
  sum=$(cat "$st"/rdma/libnccl-mesh.dylib "$st"/rdma/torch-mesh/setup.py "$st"/rdma/torch-mesh/torch_mesh/* 2>/dev/null | shasum)
  if [ "$sum" != "$(cat "$venv/torch-mesh.sum" 2>/dev/null)" ]; then
    (cd "$st/rdma/torch-mesh" && MESH_RDMA="$st/rdma" taskpolicy -b "$venv/bin/python" setup.py -q build_ext \
      -b "$S/stage/torch-mesh.build" -t "$S/stage/torch-mesh.objects") >> "$log" 2>&1 \
      || { why="torch-mesh does not build ($log)"; return 1; }
  fi
  if signed "$M/rdma/mesh-flow" && ! signed "$st/rdma/mesh-flow"; then
    why="the staged bridge is signed ad hoc where the running one carries an identity, whose Local Network grant it would not keep (docs: the identity's step)"; return 1
  fi
}

outputs() {
  git -C "$1" ls-files -o -i --exclude-standard -- rdma 2>/dev/null \
    | awk '/(^|\/)(__pycache__|torch-mesh)\/|\/_mesh_c[.]/{next} {r = /\/\.build\// ? 0 : /\/mesh-flow$/ ? 3 : /\/mesh-stat$/ ? 2 : 1; print r, $0}' \
    | sort -k1,1n -k2 | cut -d' ' -f2-
}

swap() {
  local st f doc conv bridge binary moved=
  st=$(cd "$S/stage/mesh" && pwd -P)
  doc=$(cat "$SUPPORT/io.mesh.bridge/document"); conv=$(cat "$SUPPORT/io.mesh.converge/document")
  bridge=$(pgrep -nx -U "$(id -u)" mesh-flow); binary=$(stat -f %i "$M/rdma/mesh-flow" 2>/dev/null)
  git -C "$M" merge -q --ff-only "$1" || { why="$M does not fast-forward to $(short "$1")"; return 1; }
  for f in $(outputs "$st"); do
    if cmp -s "$st/$f" "$M/$f"; then [ "$f" = rdma/mesh-flow ] && [ -n "$moved" ] || continue
    else [ "$f" != rdma/libmesh.dylib ] || moved=1; fi
    mkdir -p "$(dirname "$M/$f")"
    cp "$st/$f" "$M/$f.converge" && mv -f "$M/$f.converge" "$M/$f" || { why="$f not installed"; return 1; }
  done
  MESH_USER_BUILT=1 "$M/bin/mesh-user.sh" "$doc" "$M" "$conv" || { why="bin/mesh-user.sh failed on the installed images"; return 1; }
  restarted "$bridge" "$binary"
}

restarted() {
  local start
  [ -n "$1" ] && [ "$(stat -f %i "$M/rdma/mesh-flow" 2>/dev/null)" != "$2" ] || return 0
  start=$(date +%s)
  while kill -0 "$1" 2>/dev/null; do
    [ $(($(date +%s) - start)) -lt "$wait_s" ] || { note="the bridge still carries a call: it stops itself on its new binary once it carries none"; return 0; }
    sleep 1
  done
  until "$M/bin/mesh-stat" 2>/dev/null | grep -q '^{"up":true'; do
    [ $(($(date +%s) - start)) -lt "$wait_s" ] || { why="the bridge did not come up on its new binary within $wait_s s (~/Library/Logs/io.mesh.bridge.log)"; return 2; }
    sleep 1
  done
}

off() {
  local p l d
  rm -f "$CONF" "$SUPPORT/io.mesh.converge/document"
  for p in "$HOME"/Library/LaunchAgents/*.converge.plist "$HOME"/Library/LaunchAgents/*.converge.plist.new; do
    [ -f "$p" ] || continue
    l=${p##*/}; l=${l%.new}; l=${l%.plist}
    for d in "gui/$(id -u)" "user/$(id -u)"; do
      ! launchctl print "$d/$l" 2>/dev/null | grep -q '^[[:space:]]*pid = ' || printf '%s: its pass runs; removed once it ends\n' "$l"
      while launchctl print "$d/$l" 2>/dev/null | grep -q '^[[:space:]]*pid = '; do sleep 2; done
      launchctl bootout "$d/$l" 2>/dev/null
    done
    rm -f "$p"; printf '%s removed\n' "${p##*/}"
  done
  for p in "$S"/converge/*.state; do [ ! -f "$p" ] || printf 'off\n' > "$p"; done
  return 0
}

pass() {
  local target inputs swapped
  configured || { report failed "$why"; return 0; }
  [ -f "$(cat "$SUPPORT/io.mesh.bridge/document" 2>/dev/null)" ] || { report failed "the bridge's recorded document is gone ($SUPPORT/io.mesh.bridge/document)"; return 0; }
  if held "$M"; then report held "$why"; return 0; fi
  git -C "$M" fetch -q 2>/dev/null || { report held "the fetch of $M failed"; return 0; }
  target=$(git -C "$M" rev-parse '@{u}')
  if [ -z "${MESH_CONVERGE_UPSTREAM:-}" ] && ! git -C "$M" show "$target:bin/mesh-converge.sh" 2>/dev/null | cmp -s - "$0"; then
    mkdir -p "$S/converge"; git -C "$M" show "$target:bin/mesh-converge.sh" > "$S/converge/mesh-converge.sh"
    MESH_CONVERGE_UPSTREAM=$target MESH_CHECKOUT=$M exec /bin/bash "$S/converge/mesh-converge.sh"
  fi
  inputs=$(stamp "$target")
  [ "$inputs" != "$(cat "$S/converge/$LABEL" 2>/dev/null)" ] || { report converged "mesh $(short "$target")"; return 0; }
  [ "$inputs" != "$(cat "$S/converge/$LABEL.failed" 2>/dev/null)" ] || return 0
  quiet || { report busy "$why"; return 0; }
  report staging "mesh $(short "$target")"
  stage "$target" || { printf '%s\n' "$inputs" > "$S/converge/$LABEL.failed"; report failed "mesh $(short "$target"): $why"; return 0; }
  begin "$LABEL" $$ || { report busy "$why; mesh $(short "$target") staged"; return 0; }
  trap 'end "$LABEL"' EXIT
  report converging "mesh $(short "$target")"
  swap "$target"; swapped=$?
  end "$LABEL"; trap - EXIT
  [ "$swapped" != 2 ] || { rm -f "$S/converge/$LABEL"; printf '%s\n' "$inputs" > "$S/converge/$LABEL.failed"; }
  [ "$swapped" = 0 ] || { report failed "mesh $(short "$target"): $why"; return 0; }
  report converged "mesh $(short "$target")${note:+; $note}"
}

case ${1:-} in
  held) [ $# = 2 ] || exit 64; held "$2" && { printf '%s\n' "$why"; exit 0; }; exit 1 ;;
  idle) idle && exit 0; printf '%s\n' "$why"; exit 1 ;;
  quiet) quiet && exit 0; printf '%s\n' "$why"; exit 1 ;;
  begin) [ $# = 3 ] || exit 64; begin "$2" "$3" && exit 0; printf '%s\n' "$why"; exit 1 ;;
  end) [ $# = 2 ] || exit 64; end "$2" ;;
  stamp) [ $# = 2 ] || exit 64; stamp "$2" ;;
  off) [ $# = 1 ] || exit 64; off ;;
  '') pass ;;
  *) sed -n '2,17p' "$0" >&2; exit 64 ;;
esac
