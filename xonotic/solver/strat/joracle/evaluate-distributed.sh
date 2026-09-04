#!/bin/sh
set -u

HERE=$(cd -- "$(dirname -- "$0")" && pwd)
STRAT=$(cd -- "$HERE/.." && pwd)
XONOTIC=$(cd -- "$STRAT/../.." && pwd)
REPO=$(cd -- "$XONOTIC/.." && pwd)
MESH_PY=$REPO/bin/mesh-python
MINI=${JORACLE_MINI:-mesh-mini}
IDENTITY=${JORACLE_SSH_IDENTITY:-$HOME/.ssh/primeintellect_ed25519}
SSH_COMMAND=${JORACLE_SSH_COMMAND:-}
[ -n "$SSH_COMMAND" ] || SSH_COMMAND="ssh -i $IDENTITY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8 -o HostKeyAlias=mesh-mini -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
RUNDIR=${JORACLE_RUNDIR:-$STRAT/runs/release}
REMOTE_ROOT=${JORACLE_MINI_RUNDIR:-/tmp/mesh-xonotic-curriculum}
ASSETROOT=${JORACLE_ASSETROOT:-$HOME/dox/xonotic/Xonotic}
TRAINING_ASSETS=${JORACLE_TRAINING_ASSETS:-}
BASEDIR=${JORACLE_BASEDIR:-$RUNDIR/basedir}
PBRPK3=${JORACLE_PBR_PK3:-$XONOTIC/render-build/zzzzz-mesh-pbr.pk3}

mkdir -p "$BASEDIR/data"
for asset in "$ASSETROOT"/data/*.pk3 "$XONOTIC"/mapgen/build/*.pk3; do
  if [ -f "$asset" ]; then
    ln -sfn "$asset" "$BASEDIR/data/$(basename "$asset")"
  else
    printf 'release asset unavailable: %s\n' "$asset" >&2
  fi
done
if [ -n "$TRAINING_ASSETS" ]; then
  for asset in "$TRAINING_ASSETS"/*.pk3; do
    if [ -f "$asset" ]; then
      ln -sfn "$asset" "$BASEDIR/data/$(basename "$asset")"
    else
      printf 'release asset unavailable: %s\n' "$asset" >&2
    fi
  done
fi
"$MESH_PY" "$XONOTIC/render/pbr-materials.py" "$BASEDIR" "$PBRPK3"
for asset in "$XONOTIC"/render-build/*.pk3 "$XONOTIC"/payload-build/*.pk3; do
  if [ -f "$asset" ]; then
    ln -sfn "$asset" "$BASEDIR/data/$(basename "$asset")"
  else
    printf 'release asset unavailable: %s\n' "$asset" >&2
  fi
done

cd "$XONOTIC"
exec "$MESH_PY" -m solver.strat.curriculum \
  --run-dir "$RUNDIR" \
  --basedir "$BASEDIR" \
  --server-host "$MINI" \
  --ssh-command "$SSH_COMMAND" \
  --remote-run-root "$REMOTE_ROOT" \
  --remote-python /usr/local/mesh/bin/mesh-python \
  --maps "${JORACLE_MAPS:-auto}" \
  --study-repetitions "${JORACLE_STUDY_REPETITIONS:-3}" \
  --generate "${JORACLE_TRAIN_MATCHES:-1}" \
  --duration "${JORACLE_MATCH_SECONDS:-600}" \
  --distributed-scale \
  "$@"
