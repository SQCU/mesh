#!/bin/bash
set -u
set -o pipefail
mesh_target=${1:-/usr/local/mesh}
mesh_source=${2:-$(cd "$(dirname "$0")/.." && pwd)}
mesh_uv_version=${MESH_UV_VERSION:-0.12.6}
mesh_python_version=$(cat "$mesh_source/.python-version")
umask 022
mkdir -p "$mesh_target/bin" "$mesh_target/runtimes"
# rdma/mesh-disk.h's floor: a new runtime (about 265 MB) is made only while the disk keeps it free after it
mesh_floor=$(awk '$1=="#define" && $2=="MESH_DISK_FLOOR_GIB"{print $3}' "$mesh_source/rdma/mesh-disk.h" 2>/dev/null)
mesh_room(){
  [ -n "$mesh_floor" ] || return 0
  mesh_free=$(df -k "$1" 2>/dev/null | awk 'NR==2{print $4}')
  [ -n "$mesh_free" ] || return 0
  [ "$mesh_free" -ge $(( (mesh_floor << 20) + $2 / 1024 )) ] && return 0
  echo "mesh-disk: $3 not written ($2 bytes): $1 has $((mesh_free * 1024)) bytes free, under the floor of $mesh_floor GiB (mesh-disk.h)"
  return 1
}
mesh_current="$mesh_target/runtime-current"
[ -x "$mesh_current/.venv/bin/python" ] || mesh_current="$mesh_target"
# the runtime's identity: the lock files the source has, equal in the runtime; a file the source lacks
# (pyproject.toml, deleted 2026-09-14), the runtime lacks too.  Comparing a file the source lacks made a new
# runtime at every update (1276 of them, 292 GB, on the M4 Pro by 2026-09-30).
mesh_same=1
for mesh_file in uv.lock pyproject.toml .python-version; do
  if [ -f "$mesh_source/$mesh_file" ]; then cmp -s "$mesh_source/$mesh_file" "$mesh_current/$mesh_file" || mesh_same=0
  elif [ -e "$mesh_current/$mesh_file" ]; then mesh_same=0; fi
done
if [ "$mesh_same" = 1 ] \
    && UV_PROJECT_ENVIRONMENT="$mesh_current/.venv" "$mesh_target/bin/uv" sync --project "$mesh_current" --frozen --no-dev --check \
    && "$mesh_current/.venv/bin/python" -c 'import mlx.core,numpy'; then
  echo "mesh runtime retained: $mesh_current"
elif ! mesh_room "$mesh_target/runtimes" $((512 << 20)) "a new mesh runtime"; then
  echo "previous runtime retained: $mesh_current"
else
  mesh_stage=$(mktemp -d "$mesh_target/runtimes/runtime.XXXXXX")
  for mesh_file in pyproject.toml uv.lock .python-version; do
    [ -f "$mesh_source/$mesh_file" ] && install -m 644 "$mesh_source/$mesh_file" "$mesh_stage/"
  done
  # UV_NO_MODIFY_PATH: the installer appends no line to the user's shell profiles (it had, one a runtime)
  /usr/bin/curl -LsSf "https://astral.sh/uv/$mesh_uv_version/install.sh" | env UV_INSTALL_DIR="$mesh_stage/bin" UV_NO_MODIFY_PATH=1 /bin/sh >/dev/null 2>&1 || echo "UV download failed; trying installed UV"
  mesh_uv="$mesh_target/bin/uv"
  for mesh_uv_candidate in "$mesh_stage/bin/uv" "$mesh_target/bin/uv" /usr/local/mesh/bin/uv /opt/homebrew/bin/uv /usr/local/bin/uv "${HOME:-}/.local/bin/uv"; do
    [ -x "$mesh_uv_candidate" ] || continue
    mesh_uv="$mesh_uv_candidate"
    "$mesh_uv" --version | grep -q "$mesh_uv_version" && break
  done
  if UV_PROJECT_ENVIRONMENT="$mesh_stage/.venv" "$mesh_uv" sync --project "$mesh_stage" --frozen --no-dev --managed-python --python "$mesh_python_version" \
      && "$mesh_stage/.venv/bin/python" -c 'import mlx.core,numpy; print("mesh runtime ready")'; then
    ln -s "$mesh_stage" "$mesh_stage/current"
    mv -fh "$mesh_stage/current" "$mesh_target/runtime-current" || exit 1
    [ "$mesh_uv" = "$mesh_target/bin/uv" ] || install -m 755 "$mesh_uv" "$mesh_target/bin/uv"
  else
    echo "runtime realization failed; previous runtime retained; its stage deleted: $mesh_stage"
    rm -rf "$mesh_stage"
    exit 1
  fi
fi
install -m 755 "$mesh_source/bin/mesh-python" "$mesh_target/bin/mesh-python"
install -m 755 "$mesh_source/bin/mesh-runtime-id.py" "$mesh_target/bin/mesh-runtime-id.py"
# Every runtime but the current one deleted, except one a running process uses (its path in the process's
# command line) or one a shell profile of this user sources (a line the uv installer appended before
# UV_NO_MODIFY_PATH: deleting its runtime would break that shell's start; the line is the user's to remove).
mesh_keep=$(basename "$(readlink "$mesh_target/runtime-current" 2>/dev/null)" 2>/dev/null)
case "$mesh_keep" in runtime.?*) ;; *) echo "mesh runtimes: no current runtime named; none deleted"; exit 0 ;; esac
mesh_used=$( { ps -axww -o command= 2>/dev/null; cat "${HOME:-/nonexistent}"/.profile "${HOME:-/nonexistent}"/.bashrc \
  "${HOME:-/nonexistent}"/.bash_profile "${HOME:-/nonexistent}"/.zshrc "${HOME:-/nonexistent}"/.zshenv "${HOME:-/nonexistent}"/.zprofile \
  "${HOME:-/nonexistent}"/.config/fish/conf.d/uv.env.fish 2>/dev/null; } | grep -o 'runtimes/runtime\.[A-Za-z0-9]*' | sed 's|.*/||' | sort -u)
mesh_pruned=0; mesh_held=0
for mesh_runtime in "$mesh_target"/runtimes/runtime.*; do
  [ -d "$mesh_runtime" ] || continue
  mesh_name=$(basename "$mesh_runtime")
  [ "$mesh_name" = "$mesh_keep" ] && continue
  if printf '%s\n' "$mesh_used" | grep -qxF "$mesh_name"; then mesh_held=$((mesh_held + 1)); continue; fi
  rm -rf "$mesh_runtime" && mesh_pruned=$((mesh_pruned + 1))
done
echo "mesh runtimes: $mesh_pruned old ones deleted, $mesh_held kept (a running process's or a shell profile's), current $mesh_keep"
