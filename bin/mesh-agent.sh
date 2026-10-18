#!/bin/bash
# A launchd job's definition NEW made the loaded one in DOMAIN (default user/<uid>) under PLIST (default NEW without .new):
# README.md, mesh-agent.sh.   mesh-agent.sh NEW [DOMAIN [PLIST]]
set -u
new=$1; domain=${2:-user/$(id -u)}; plist=${3:-${new%.new}}; label=$(basename "$plist" .plist); here=${0%/*}
inspect(){ launchctl print "$domain/$label" 2>/dev/null && return 0
  case $domain in system) return 1 ;; esac
  if [ "$(id -u)" = "${domain#*/}" ]; then launchctl list "$label" 2>/dev/null; else launchctl asuser "${domain#*/}" launchctl list "$label" 2>/dev/null; fi; }
defined(){ /usr/libexec/PlistBuddy -c "Print :$2" "$1" 2>/dev/null | sed '1d;$d;s/^[[:space:]]*//'; }
state=$(inspect)
loaded(){ printf '%s\n' "$state" | awk '/^\targuments = \{/{f=1;next} f&&/^\t\}/{exit} /"ProgramArguments" = \(/{g=1;next} g&&/^[[:space:]]*\);/{exit}
  f||g{sub(/^[[:space:]]+/,""); if(g){sub(/^"/,""); sub(/";$/,"")} print}'; }
pid=$(printf '%s\n' "$state" | awk '/^[[:space:]]*pid = /{print $3; exit} /"PID" = /{gsub(/[";]/,"",$3); print $3; exit}')
ours(){ p=$$; while [ -n "$pid" ] && [ "${p:-0}" -gt 1 ]; do [ "$p" = "$pid" ] && return 0; p=$(ps -o ppid= -p "$p" | tr -d ' '); done; return 1; }
exited(){ for py in "$here/mesh-python" /usr/local/mesh/bin/mesh-python "$HOME/.local/mesh/bin/mesh-python"; do
    [ -x "$py" ] || continue
    "$py" -c 'import select, sys
try: select.kqueue().control([select.kevent(int(sys.argv[1]), select.KQ_FILTER_PROC, select.KQ_EV_ADD | select.KQ_EV_ONESHOT, select.KQ_NOTE_EXIT)], 1)
except ProcessLookupError: pass' "$1" && return 0
  done; return 1; }
grep -q '<!--' "$new" || { sum=$(defined "$new" ProgramArguments | while IFS= read -r a; do [ -f "$a" ] && cat "$a"; done | shasum -a 256 | cut -c1-64)
  awk -v c="<!-- the files it runs: $sum -->" 'NR == 2 {print c} 1' "$new" > "$new.stamp" && mv "$new.stamp" "$new"; }
plutil -lint -s "$new" >/dev/null 2>&1 || { rm -f "$new"; echo "$label: its new definition does not parse; the loaded one kept"; exit 1; }
if cmp -s "$new" "$plist" && [ -n "$state" ] && [ "$(loaded)" = "$(defined "$new" ProgramArguments)" ]; then rm "$new"; exit 0; fi
! ours || { echo "$label runs this install; its new definition is staged at $new for the keeper, between its runs"; exit 0; }
if [ -n "$pid" ]; then
  printf '%s\n' "$state" | grep -q -e '^[[:space:]]*properties = .*keepalive' -e '"OnDemand" = false;' \
    || { echo "$label is running; its new definition is staged at $new for the keeper, between its runs"; exit 0; }
  ! ioreg -l -w0 -r -c AppleThunderboltRDMAInterface 2>/dev/null | grep -q "\"IOUserClientCreator\" = \"pid $pid," \
    || { mv "$new" "$plist"; echo "$label holds an RDMA device (pid $pid) and launchd stops a job by a signal it may follow with SIGKILL: its new definition is in place for its next load"; exit 0; }
fi
mv "$new" "$plist"
if launchctl bootout "$domain/$label" 2>/dev/null && [ -n "$pid" ]; then exited "$pid" || echo "$label: no mesh-python here to wait for pid $pid's exit by"; fi
launchctl enable "$domain/$label" 2>/dev/null
said=$(launchctl bootstrap "$domain" "$plist" 2>&1) && { echo "$label loaded in $domain"; exit 0; }
echo "$label did not load in $domain (launchctl bootstrap: $said; $(inspect >/dev/null && echo 'launchd still holds its old instance' || echo 'launchd holds none')); its definition is in place at $plist"; exit 1
