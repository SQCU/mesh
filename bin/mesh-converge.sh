#!/bin/bash
# A node's user layers follow their upstreams' branch heads, each built at its upstream in a stage and installed by rename
# (README.md, Convergence; design/transport.md).  With no argument, one pass of the mesh's own layer: the mesh checkout
# this script is in, built at its upstream in a stage, fast-forwarded, its images installed by rename (the libraries
# first, the bridge's binary last, and the binary anew wherever libmesh changed) and bin/mesh-user.sh run on them,
# which sends the bridge's heartbeat SIGHUP, so it restarts the bridge on the new binary.  Its subcommands:
#   follow           a pass every period_s, read again before each, for as long as the node has a convergence
#                    configuration (the admin user's agent io.mesh.converge runs it, KeepAlive while it fails)
#   held CHECKOUT    prints why and succeeds where CHECKOUT has work its upstream lacks (nothing is fetched); a
#                    consumer's layer converges by it (metal-microbench tools/mesh/node-setup.sh)
#   stamp REVISION   what the mesh's layer converges to at REVISION: the revision and its recorded documents' digests
#   off              the node out of convergence, the operator's choice: the recorded convergence document and every
#                    layer's agent (<label>.converge) removed, and each layer's follow ending at its next pass;
#                    mesh-user.sh's CONVERGENCE puts it back
# The node's state is ~/.cache/mesh: converge/<label> (what LABEL's install last converged to, .state its last outcome,
# .build.log its last staged build), facts/<label> (what the layer runs, which node-info publishes), stage/<repository>
# (the staged worktrees).  The configuration is the operand document bin/mesh-user.sh canonicalizes
# (etc/converge/README.md): period_s, with no default.
set -u
export PATH=$PATH:/usr/local/mesh/bin:/usr/bin:/bin:/usr/sbin:/sbin
S=$HOME/.cache/mesh; LABEL=io.mesh.bridge
SUPPORT="$HOME/Library/Application Support"; CONF="$SUPPORT/io.mesh.converge/configuration.plist"
M=${MESH_CHECKOUT:-$(cd "$(dirname "$0")/.." && pwd -P)}
why=; period_s=; MODE=${1:-}
configured() { period_s=$(plutil -extract period_s raw -o - "$CONF" 2>/dev/null) && [ -n "$period_s" ] && return 0
  why="no convergence configuration ($CONF: bin/mesh-user.sh CONFIGURATION CHECKOUT CONVERGENCE)"; return 1; }
short() { git -C "$M" rev-parse --short "$1" 2>/dev/null || printf '%s' "$1"; }

held() {
  local c=$1
  git -C "$c" rev-parse -q --verify '@{u}' >/dev/null 2>&1 || { why="$c follows no upstream"; return 0; }
  git -C "$c" diff --quiet HEAD -- 2>/dev/null || { why="$c has uncommitted changes"; return 0; }
  [ "$(git -C "$c" rev-list --count '@{u}..HEAD')" = 0 ] || { why="$c has commits its upstream lacks"; return 0; }
  return 1
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
  taskpolicy -b make -C "$st/rdma" all install > "$log" 2>&1 || { why="make -C rdma all install failed ($log)"; return 1; }
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
  local st f doc conv moved=
  st=$(cd "$S/stage/mesh" && pwd -P)
  doc=$(cat "$SUPPORT/io.mesh.bridge/document"); conv=$(cat "$SUPPORT/io.mesh.converge/document")
  git -C "$M" merge -q --ff-only "$1" || { why="$M does not fast-forward to $(short "$1")"; return 1; }
  for f in $(outputs "$st"); do
    if cmp -s "$st/$f" "$M/$f"; then [ "$f" = rdma/mesh-flow ] && [ -n "$moved" ] || continue
    else [ "$f" != rdma/libmesh.dylib ] || moved=1; fi
    mkdir -p "$(dirname "$M/$f")"
    cp "$st/$f" "$M/$f.converge" && mv -f "$M/$f.converge" "$M/$f" || { why="$f not installed"; return 1; }
  done
  MESH_USER_BUILT=1 "$M/bin/mesh-user.sh" "$doc" "$M" "$conv" || { why="bin/mesh-user.sh failed on the installed images"; return 1; }
}

off() {
  local p l
  rm -f "$CONF" "$SUPPORT/io.mesh.converge/document"
  for p in "$HOME"/Library/LaunchAgents/*.converge.plist "$HOME"/Library/LaunchAgents/*.converge.plist.new; do
    [ -f "$p" ] || continue
    l=${p##*/}; l=${l%.new}; l=${l%.plist}
    rm -f "$p"; printf '%s removed; its follow ends at its next pass\n' "${p##*/}"
  done
  for p in "$S"/converge/*.state; do [ ! -f "$p" ] || printf 'off\n' > "$p"; done
  return 0
}

pass() {
  local target inputs f o
  configured || { report off "$why"; return 1; }
  [ -f "$(cat "$SUPPORT/io.mesh.bridge/document" 2>/dev/null)" ] || { report failed "the bridge's recorded document is gone ($SUPPORT/io.mesh.bridge/document)"; return 0; }
  if held "$M"; then report held "$why"; return 0; fi
  git -C "$M" fetch -q 2>/dev/null || { report held "the fetch of $M failed"; return 0; }
  target=$(git -C "$M" rev-parse '@{u}')
  if ! git -C "$M" show "$target:bin/mesh-converge.sh" 2>/dev/null | cmp -s - "$0"; then
    f=$S/converge/mesh-converge.$(git -C "$M" rev-parse --short=12 "$target").sh; mkdir -p "$S/converge"
    if git -C "$M" show "$target:bin/mesh-converge.sh" > "$f" 2>/dev/null && [ -s "$f" ]; then
      for o in "$S"/converge/mesh-converge.*.sh; do [ "$o" = "$f" ] || [ "$o" -ef "$0" ] || rm -f "$o"; done
      MESH_CHECKOUT=$M exec /bin/bash "$f" ${MODE:+"$MODE"}
    fi
  fi
  inputs=$(stamp "$target")
  [ "$inputs" != "$(cat "$S/converge/$LABEL" 2>/dev/null)" ] || { report converged "mesh $(short "$target")"; return 0; }
  report staging "mesh $(short "$target")"
  stage "$target" || { report failed "mesh $(short "$target"): $why"; return 0; }
  report converging "mesh $(short "$target")"
  swap "$target" || { report failed "mesh $(short "$target"): $why"; return 0; }
  report converged "mesh $(short "$target")"
}

case $MODE in
  follow) [ $# = 1 ] || exit 64; while pass; do sleep "$period_s"; done; exit 0 ;;
  held) [ $# = 2 ] || exit 64; held "$2" && { printf '%s\n' "$why"; exit 0; }; exit 1 ;;
  stamp) [ $# = 2 ] || exit 64; stamp "$2" ;;
  off) [ $# = 1 ] || exit 64; off ;;
  '') pass; exit 0 ;;
  *) sed -n '2,14p' "$0" >&2; exit 64 ;;
esac
