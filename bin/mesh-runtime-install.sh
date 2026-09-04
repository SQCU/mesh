#!/bin/bash
set -u
mesh_target=${1:-/usr/local/mesh}
mesh_source=${2:-$(cd "$(dirname "$0")/.." && pwd)}
mesh_uv_version=${MESH_UV_VERSION:-0.12.6}
mesh_python_version=$(cat "$mesh_source/.python-version")
mesh_stage=$(mktemp -d)
trap 'rm -rf "$mesh_stage"' EXIT
umask 022
mkdir -p "$mesh_target/bin"
install -m 644 "$mesh_source/pyproject.toml" "$mesh_target/pyproject.toml"
install -m 644 "$mesh_source/uv.lock" "$mesh_target/uv.lock"
install -m 644 "$mesh_source/.python-version" "$mesh_target/.python-version"
install -m 755 "$mesh_source/bin/mesh-python" "$mesh_target/bin/mesh-python"
install -m 755 "$mesh_source/bin/mesh-runtime-id.py" "$mesh_target/bin/mesh-runtime-id.py"
/usr/bin/curl -LsSf "https://astral.sh/uv/$mesh_uv_version/install.sh" | env UV_INSTALL_DIR="$mesh_stage" /bin/sh >/dev/null 2>&1 || true
for mesh_uv_candidate in "$mesh_stage/uv" "$mesh_stage/bin/uv" /usr/local/mesh/bin/uv /opt/homebrew/bin/uv /usr/local/bin/uv "${HOME:-}/.local/bin/uv"; do
  [ -x "$mesh_uv_candidate" ] || continue
  install -m 755 "$mesh_uv_candidate" "$mesh_target/bin/uv"
  "$mesh_target/bin/uv" --version | grep -q "$mesh_uv_version" && break
done
"$mesh_target/bin/uv" --version
UV_PROJECT_ENVIRONMENT="$mesh_target/.venv" "$mesh_target/bin/uv" sync --project "$mesh_target" --frozen --no-dev --managed-python --python "$mesh_python_version"
"$mesh_target/.venv/bin/python" -c 'import mlx.core,numpy; print("mesh runtime ready")'
