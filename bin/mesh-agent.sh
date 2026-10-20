#!/bin/bash
# A launchd job's definition NEW put in place as PLIST (default NEW without .new) and the job loaded in DOMAIN (default
# user/<uid>) where it is not loaded; a loaded job is left running, its new definition taking effect at its next load
# (README.md, launchd).   mesh-agent.sh NEW [DOMAIN [PLIST]]
set -u
new=$1; domain=${2:-user/$(id -u)}; plist=${3:-${new%.new}}; label=$(basename "$plist" .plist)
loaded(){ launchctl print "$domain/$label" >/dev/null 2>&1 && return 0
  case $domain in system) return 1 ;; esac
  if [ "$(id -u)" = "${domain#*/}" ]; then launchctl list "$label" >/dev/null 2>&1; else launchctl asuser "${domain#*/}" launchctl list "$label" >/dev/null 2>&1; fi; }
plutil -lint -s "$new" >/dev/null 2>&1 || { [ "$new" = "$plist" ] || rm -f "$new"; echo "$label: its new definition does not parse; the one in place kept"; exit 1; }
changed=; [ "$new" = "$plist" ] || { cmp -s "$new" "$plist" || changed=1; mv "$new" "$plist"; }
if loaded; then [ -z "$changed" ] || echo "$label: its new definition is in place and takes effect at its next load (login or boot); the loaded job runs on"; exit 0; fi
# an operator's off (launchctl disable) stands: a disabled job is neither enabled nor loaded here (design/rubrics.md M3, 10-20)
launchctl print-disabled "$domain" 2>/dev/null | grep -q "\"$label\" => \(disabled\|true\)" && exit 0
said=$(launchctl bootstrap "$domain" "$plist" 2>&1) && { echo "$label loaded in $domain"; exit 0; }
echo "$label did not load in $domain (launchctl bootstrap: $said); its definition is in place at $plist"; exit 1
