#!/bin/bash
set -u
set -o pipefail
mesh_target=${1:-/usr/local/mesh}
mesh_source=${2:-$(cd "$(dirname "$0")/.." && pwd)}
mesh_uv_version=${MESH_UV_VERSION:-0.12.6}
mesh_python_version=$(cat "$mesh_source/.python-version")
umask 022
mkdir -p "$mesh_target/bin" "$mesh_target/runtimes"
mesh_current="$mesh_target/runtime-current"
[ -x "$mesh_current/.venv/bin/python" ] || mesh_current="$mesh_target"
if cmp -s "$mesh_source/uv.lock" "$mesh_current/uv.lock" \
    && cmp -s "$mesh_source/pyproject.toml" "$mesh_current/pyproject.toml" \
    && cmp -s "$mesh_source/.python-version" "$mesh_current/.python-version" \
    && "$mesh_current/.venv/bin/python" -c 'import mlx.core,numpy'; then
  echo "mesh runtime retained: $mesh_current"
else
  mesh_stage=$(mktemp -d "$mesh_target/runtimes/runtime.XXXXXX")
  install -m 644 "$mesh_source/pyproject.toml" "$mesh_source/uv.lock" "$mesh_source/.python-version" "$mesh_stage/"
  /usr/bin/curl -LsSf "https://astral.sh/uv/$mesh_uv_version/install.sh" | env UV_INSTALL_DIR="$mesh_stage/bin" /bin/sh >/dev/null 2>&1 || echo "UV download failed; trying installed UV"
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
    echo "runtime realization failed; previous runtime retained; diagnostic generation: $mesh_stage"
    exit 1
  fi
fi
install -m 755 "$mesh_source/bin/mesh-python" "$mesh_target/bin/mesh-python"
install -m 755 "$mesh_source/bin/mesh-runtime-id.py" "$mesh_target/bin/mesh-runtime-id.py"
