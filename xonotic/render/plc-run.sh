#!/usr/bin/env bash


set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
DP=${DP:-$root/darkplaces-work/darkplaces-sdl}
BASE=${XONBASE:-$HOME/dox/xonotic/Xonotic}
HOME_DIR=$here/plc-home
OUT=$here/plc-out
W=${W:-1280}; H=${H:-720}
label=$1; steps=$2
mkdir -p "$OUT" "$HOME_DIR/data"
cp "$steps" "$HOME_DIR/data/plcsteps.cfg"
"$DP" -xonotic -basedir "$BASE" -userdir "$HOME_DIR" \
      -window -width "$W" -height "$H" -nosound -condebug \
      +"exec plcsteps.cfg" > "$OUT/$label.log" 2>&1
echo "--- $label done"
