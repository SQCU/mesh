#!/bin/bash
# A user's agent, its definition written to PLIST.new: loaded in the user's background domain (user/<uid>) on every node,
# never a console session's, whose processes macOS's Local Network privacy holds until someone answers a prompt (a bridge
# there never hears its peers); the keeper loads the user's agents there again at boot. Replaced where it changed, its label
# enabled (a disabled label is refused), its load retried while launchd lets the old instance go.   mesh-agent.sh ~/Library/LaunchAgents/LABEL.plist.new
set -u
new=$1; plist=${new%.new}; label=$(basename "$plist" .plist)
domain="user/$(id -u)"
if cmp -s "$new" "$plist" && launchctl print "$domain/$label" >/dev/null 2>&1; then rm "$new"; exit 0; fi
mv "$new" "$plist"; launchctl bootout "$domain/$label" 2>/dev/null; launchctl enable "$domain/$label" 2>/dev/null
# the old instance is gone before the new one is asked for: launchctl print still finds a service it is letting go, so
# reading it as loaded right after the bootout took the old instance for the new one (2026-10-15: node-setup reported
# the Mini's rank loaded, and the call after it found no rank until the keeper loaded it a minute later)
for _ in $(seq 1 30); do launchctl print "$domain/$label" >/dev/null 2>&1 || break; sleep 1; done
for _ in $(seq 1 30); do
  launchctl bootstrap "$domain" "$plist" 2>/dev/null
  launchctl print "$domain/$label" >/dev/null 2>&1 && exit 0
  sleep 1
done
echo "$label did not load in $domain"; exit 1
