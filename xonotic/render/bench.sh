#!/usr/bin/env bash
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
DP=${DP:-$root/darkplaces-work/darkplaces-sdl}
BASE=${XONBASE:-$HOME/dox/xonotic/Xonotic}
HOME_DIR=${BENCHHOME:-$here/bench-home}
OUT=$here/bench-out
W=${W:-1280}; H=${H:-720}
mkdir -p "$HOME_DIR/data" "$OUT"

run_dp() {
	local log=$1; shift
	"$DP" -xonotic -basedir "$BASE" -userdir "$HOME_DIR" \
	      -window -width "$W" -height "$H" -nosound "$@" \
	      > "$OUT/$log.log" 2>&1
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
alias cl_hook_gamestart_all "exec walk.cfg"
record bench_$map $map
CFG
	cat > "$HOME_DIR/data/walk.cfg" <<CFG
alias cl_hook_gamestart_all
+forward
defer 3 "+right"
defer 5 "-right"
defer 7 "+moveleft"
defer 9 "-moveleft; +left"
defer 12 "-left"
defer 15 "+right; +jump"
defer 16 "-jump"
defer 18 "-right"
defer 21 "-forward; +back; +left"
defer 25 "-back; -left"
defer 27 "stop; quit"
CFG
	run_dp rec.cfg +exec rec.cfg
	ls -la "$HOME_DIR/data/bench_$map.dem"
	;;
run)
	map=$1; label=$2; runs=$3; shift 3
	{
		echo "cl_curl_enabled 0"
		echo "vid_vsync 0"
		for kv in "$@"; do echo "${kv%%=*} ${kv#*=}"; done
	} > "$HOME_DIR/data/run_$label.cfg"
	rm -f "$HOME_DIR/data/benchmark.log"
	run_dp "run_$label.cfg" +exec "run_$label.cfg" -benchmark "bench_$map" -benchmarkruns "$runs"
	echo "=== $label ==="
	grep -E "^[0-9]+ frames|^(MIN|MED|MAX): " "$OUT/run_$label.cfg.log" || tail -5 "$OUT/run_$label.cfg.log"
	;;
*) echo "usage: bench.sh record <map> | run <map> <label> <runs> [cvar=value ...]"; exit 1;;
esac
