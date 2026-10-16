#!/bin/bash
# A launchd job's definition NEW made the loaded one in DOMAIN (default user/<uid>) under PLIST (default NEW without .new):
# README.md, mesh-agent.sh.   mesh-agent.sh NEW [DOMAIN [PLIST]]
set -u
new=$1; domain=${2:-user/$(id -u)}; plist=${3:-${new%.new}}; label=$(basename "$plist" .plist)
defined(){ /usr/libexec/PlistBuddy -c "Print :$2" "$1" 2>/dev/null | sed '1d;$d;s/^[[:space:]]*//'; }
loaded(){ launchctl print "$domain/$label" 2>/dev/null | awk '/^\targuments = \{/{f=1;next} f&&/^\t\}/{exit} f{sub(/^[[:space:]]+/,"");print}'; }
running(){ launchctl print "$domain/$label" 2>/dev/null | grep -q '^[[:space:]]*pid = '; }
grep -q '<!--' "$new" || { sum=$(defined "$new" ProgramArguments | while IFS= read -r a; do [ -f "$a" ] && cat "$a"; done | shasum -a 256 | cut -c1-64)
  awk -v c="<!-- the files it runs: $sum -->" 'NR == 2 {print c} 1' "$new" > "$new.stamp" && mv "$new.stamp" "$new"; }
plutil -lint -s "$new" >/dev/null 2>&1 || { rm -f "$new"; echo "$label: its new definition does not parse; the loaded one kept"; exit 1; }
if cmp -s "$new" "$plist" && launchctl print "$domain/$label" >/dev/null 2>&1 && [ "$(loaded)" = "$(defined "$new" ProgramArguments)" ]; then rm "$new"; exit 0; fi
launchctl print "$domain/$label" 2>/dev/null | grep -q '^[[:space:]]*properties = .*keepalive' || {
  for _ in $(seq 1 120); do running || break; sleep 1; done
  ! running || { echo "$label is running; its new definition is staged at $new for a pass after its run"; exit 0; }; }
mv "$new" "$plist"; launchctl bootout "$domain/$label" 2>/dev/null; launchctl enable "$domain/$label" 2>/dev/null
# the old instance is gone before the new one is asked for: launchctl print still finds a service it is letting go, so
# reading it as loaded right after the bootout took the old instance for the new one (2026-10-15: node-setup reported
# the Mini's rank loaded, and the call after it found no rank until the keeper loaded it a minute later)
for _ in $(seq 1 30); do launchctl print "$domain/$label" >/dev/null 2>&1 || break; sleep 1; done
for _ in $(seq 1 30); do
  launchctl bootstrap "$domain" "$plist" 2>/dev/null
  launchctl print "$domain/$label" >/dev/null 2>&1 && { echo "$label loaded in $domain"; exit 0; }
  sleep 1
done
echo "$label did not load in $domain"; exit 1
