#!/usr/bin/env bash
# Frame-time harness for the DarkPlaces client.
#
#   ./bench.sh record <map>            record a fixed flythrough demo for <map>
#   ./bench.sh run    <map> <label> [cvar=value ...]
#                                      timedemo that demo 3x and print fps
#
# timedemo replays the identical input stream every run, so two labels differ
# only by the cvars under test.  Everything lands in render/bench-out/.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
DP=${DP:-$root/darkplaces-work/darkplaces-sdl}
BASE=${XONBASE:-$HOME/dox/xonotic/Xonotic}
HOME_DIR=${BENCHHOME:-$here/bench-home}
OUT=$here/bench-out
W=${W:-1280}; H=${H:-720}
mkdir -p "$HOME_DIR/data" "$OUT"

run_dp() { # $1 = cfg name already written into $HOME_DIR/data, $2 = seconds budget
	"$DP" -xonotic -basedir "$BASE" -userdir "$HOME_DIR" \
	      -window -width "$W" -height "$H" -nosound +exec "$1" \
	      > "$OUT/$1.log" 2>&1 &
	local pid=$!
	local i=0
	while kill -0 $pid 2>/dev/null; do
		i=$((i+1)); [ $i -gt "$2" ] && { kill -9 $pid 2>/dev/null; break; }
		/bin/sleep 1
	done
	wait $pid 2>/dev/null || true
}

cmd=$1; shift
case "$cmd" in
record)
	map=$1
	cat > "$HOME_DIR/data/rec.cfg" <<CFG
cl_curl_enabled 0
sv_autoscreenshot 0
g_maplist_shufflenow 0
bot_number 4
defer 1 "record bench_$map $map"
defer 9 "+forward"
defer 12 "+right"
defer 14 "-right"
defer 16 "+moveleft"
defer 18 "-moveleft; +left"
defer 21 "-left"
defer 24 "+right; +jump"
defer 25 "-jump"
defer 27 "-right"
defer 30 "-forward; +back; +left"
defer 34 "-back; -left"
defer 36 "stop"
defer 38 "quit"
CFG
	run_dp rec.cfg 70
	ls -la "$HOME_DIR/data/bench_$map.dem"
	;;
run)
	map=$1; label=$2; shift 2
	{
		echo "cl_curl_enabled 0"
		echo "cl_capturevideo 0"
		echo "vid_vsync 0"
		echo "showfps 0"
		for kv in "$@"; do echo "${kv%%=*} ${kv#*=}"; done
		echo "defer 1 \"timedemo bench_$map\""
		echo "defer 3 \"cl_timedemo_benchmark_runs 3\""
	} > "$HOME_DIR/data/run_$label.cfg"
	# timedemo chains its own runs; three sequential timedemos, then quit
	# five back-to-back runs; the first is warm-up (shader compile, texture upload)
	{
		echo "cl_curl_enabled 0"
		echo "vid_vsync 0"
		for kv in "$@"; do echo "${kv%%=*} ${kv#*=}"; done
		for i in 1 2 3 4 5; do echo "defer $((i*8)) \"timedemo bench_$map\""; done
		echo "defer 48 quit"
	} > "$HOME_DIR/data/run_$label.cfg"
	rm -f "$HOME_DIR/data/benchmark.log"
	run_dp "run_$label.cfg" 240
	echo "=== $label ==="
	grep -E "^[0-9]+ frames" "$OUT/run_$label.cfg.log" || tail -5 "$OUT/run_$label.cfg.log"
	;;
*) echo "usage: bench.sh record <map> | run <map> <label> [cvar=value ...]"; exit 1;;
esac
