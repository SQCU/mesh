#!/bin/sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
BRIDGE="$(dirname "$HERE")"
SRC="${XONOTIC_DP:-/Applications/Xonotic/source/darkplaces}"
GMQCC="${GMQCC:-/Users/mdot/dox/mesh/xonotic/gmqcc-work/gmqcc}"
WORK="${1:-/tmp/meshbridge-engine}"

rm -rf "$WORK"
mkdir -p "$WORK/run/id1/maps"
cp -R "$SRC" "$WORK/dp"
"$BRIDGE/apply.sh" "$WORK/dp"
(cd "$WORK/dp" && make sv-release -j8 >"$WORK/build.log" 2>&1)
cp "$WORK/dp/darkplaces-dedicated" "$WORK/run/"

cc -O2 -std=c11 -Wall -Wextra -o "$WORK/run/fakesolver" "$BRIDGE/solver/fakesolver.c"
"$GMQCC" -std=qcc -o "$WORK/run/id1/progs.dat" "$BRIDGE/qc/progsdefs.qc" "$BRIDGE/qc/meshtest.qc" >/dev/null 2>&1
cp "$HERE/_init.bsp" "$WORK/run/id1/maps/_init.bsp"

cd "$WORK/run"
./fakesolver /mesh_qc 40 0 >solver.log 2>&1 &
sleep 0.5
perl -e 'alarm 40; exec @ARGV' ./darkplaces-dedicated -basedir . -game id1 -nohome +developer 1 +sys_ticrate 0.0166 +map _init >dp.log 2>&1 || true
grep -a MESH dp.log
cat solver.log
