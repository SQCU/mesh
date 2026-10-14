#!/bin/bash
set -u
source=$1
venv=$HOME/.venv-mesh-uv
uv=/usr/local/mesh/bin/uv
[ -x "$venv/bin/python" ] || "$uv" venv -q --python "$(cat "$source/.python-version")" "$venv" || exit 1
"$uv" pip install -q --python "$venv/bin/python" -r "$source/rdma/requirements.txt" || exit 1
"$venv/bin/python" -c 'import sys, sysconfig; open(sysconfig.get_paths()["purelib"] + "/mesh.pth", "w").write(sys.argv[1] + "\n")' /usr/local/mesh/rdma || exit 1
sum=$(cat /usr/local/mesh/rdma/libnccl-mesh.dylib "$source"/rdma/torch-mesh/setup.py "$source"/rdma/torch-mesh/torch_mesh/* 2>/dev/null | shasum)
[ "$(cat "$venv/torch-mesh.sum" 2>/dev/null)" = "$sum" ] && exit 0
build=$(mktemp -d) || exit 1
cp -R "$source/rdma/torch-mesh/." "$build/"
MESH_RDMA=/usr/local/mesh/rdma "$uv" pip install -q --python "$venv/bin/python" --no-build-isolation --reinstall-package torch-mesh "$build"
status=$?
rm -rf "$build"
[ "$status" = 0 ] && printf '%s\n' "$sum" > "$venv/torch-mesh.sum"
exit "$status"
