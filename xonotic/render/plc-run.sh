#!/usr/bin/env bash
# Drive a real payload match and capture frames / frame times.
#   plc-run.sh <label> <seconds> <extra cfg lines file>
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
DP=${DP:-$root/darkplaces-work/darkplaces-sdl}
BASE=${XONBASE:-$HOME/dox/xonotic/Xonotic}
HOME_DIR=$here/plc-home
OUT=$here/plc-out
W=${W:-1280}; H=${H:-720}
label=$1; secs=$2; steps=$3
mkdir -p "$OUT" "$HOME_DIR/data"
cp "$steps" "$HOME_DIR/data/plcsteps.cfg"
"$DP" -xonotic -basedir "$BASE" -userdir "$HOME_DIR" \
      -window -width "$W" -height "$H" -nosound -condebug \
      +"exec plcsteps.cfg" > "$OUT/$label.log" 2>&1 &
pid=$!
i=0
while kill -0 $pid 2>/dev/null; do
  i=$((i+1)); [ $i -gt "$secs" ] && { kill -9 $pid 2>/dev/null; break; }
  /bin/sleep 1
done
wait $pid 2>/dev/null || true
echo "--- $label done"
