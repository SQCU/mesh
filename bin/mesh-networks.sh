#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
CONF=/usr/local/mesh/networks.conf
LOG=/usr/local/mesh/log/networks.log
[ -d /usr/local/mesh/log ] && exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
[ -f "$CONF" ] || exit 0
W=$(networksetup -listallhardwareports | awk '/Hardware Port: Wi-Fi/{getline; print $2}')
[ -n "$W" ] || exit 0
networksetup -setairportpower "$W" on >/dev/null 2>&1
have=$(networksetup -listpreferredwirelessnetworks "$W" 2>/dev/null | sed '1d;s/^[[:space:]]*//')
i=0
while IFS=$'\t' read -r ssid sec pass; do
  case "$ssid" in ''|\#*) continue ;; esac
  if ! printf '%s\n' "$have" | grep -qxF "$ssid"; then
    echo "[$(ts)] adding preferred network $ssid"
    networksetup -addpreferredwirelessnetworkatindex "$W" "$ssid" "$i" "${sec:-WPA2}" "$pass" >/dev/null 2>&1
  fi
  i=$((i+1))
done < "$CONF"
exit 0
