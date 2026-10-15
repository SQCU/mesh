#!/bin/bash
# A user's agent, its definition written to PLIST.new: loaded in the console session's domain where there is one, else the
# user's own (a headless node, where the keeper loads the user's agents again at boot), replaced where it changed, its load
# retried while launchd lets the old instance go.   mesh-agent.sh ~/Library/LaunchAgents/LABEL.plist.new
set -u
new=$1; plist=${new%.new}; label=$(basename "$plist" .plist)
domain="gui/$(id -u)"; launchctl print "$domain" >/dev/null 2>&1 || domain="user/$(id -u)"
if cmp -s "$new" "$plist" && launchctl print "$domain/$label" >/dev/null 2>&1; then rm "$new"; exit 0; fi
mv "$new" "$plist"; launchctl bootout "$domain/$label" 2>/dev/null
for _ in $(seq 1 30); do
  launchctl bootstrap "$domain" "$plist" 2>/dev/null
  launchctl print "$domain/$label" >/dev/null 2>&1 && exit 0
  sleep 1
done
echo "$label did not load in $domain"; exit 1
