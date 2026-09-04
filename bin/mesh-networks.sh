#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
CONF="${MESH_NETWORKS_CONF:-/usr/local/mesh/networks.conf}"
[ -r "$CONF" ] || CONF="$HOME/.config/mesh/networks.conf"
LOG=/usr/local/mesh/log/networks.log
[ -d /usr/local/mesh/log ] && exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
if [ -r "$CONF" ] && command -v networksetup >/dev/null 2>&1; then
  W=$(networksetup -listallhardwareports | awk '/Hardware Port: Wi-Fi/{getline; print $2}')
  networksetup -setairportpower "$W" on >/dev/null 2>&1 \
    && echo "[$(ts)] Wi-Fi power realized on $W" \
    || echo "[$(ts)] Wi-Fi power realization failed on $W" >&2
  have=$(networksetup -listpreferredwirelessnetworks "$W" 2>/dev/null | sed '1d;s/^[[:space:]]*//')
  i=0
  while IFS=$'\t' read -r ssid sec pass; do
    [ -n "$ssid" ] || continue
    security add-generic-password -U -D 'AirPort network password' -a "$ssid" -s AirPort -w "$pass" /Library/Keychains/System.keychain >/dev/null 2>&1 \
      && echo "[$(ts)] credential realized for $ssid" \
      || echo "[$(ts)] credential realization failed for $ssid" >&2
    if printf '%s\n' "$have" | grep -qxF "$ssid"; then
      echo "[$(ts)] preferred network present for $ssid on $W"
    else
      networksetup -addpreferredwirelessnetworkatindex "$W" "$ssid" "$i" "${sec:-WPA2}" "$pass" >/dev/null 2>&1 \
        && echo "[$(ts)] preferred network realized for $ssid on $W" \
        || echo "[$(ts)] preferred network realization failed for $ssid on $W" >&2
    fi
    i=$((i+1))
  done < "$CONF"
elif [ -r "$CONF" ] && command -v nmcli >/dev/null 2>&1; then
  while IFS=$'\t' read -r ssid sec pass; do
    [ -n "$ssid" ] || continue
    key=wpa-psk; pmf=default
    [ "$sec" = WPA3 ] && { key=sae; pmf=required; }
    if nmcli -t -f NAME connection show | grep -qxF "$ssid"; then
      nmcli connection modify "$ssid" 802-11-wireless.ssid "$ssid" 802-11-wireless-security.key-mgmt "$key" 802-11-wireless-security.pmf "$pmf" 802-11-wireless-security.psk "$pass" connection.autoconnect yes >/dev/null \
        && echo "[$(ts)] connection profile realized for $ssid" \
        || echo "[$(ts)] connection profile realization failed for $ssid" >&2
    else
      nmcli connection add type wifi con-name "$ssid" ssid "$ssid" 802-11-wireless-security.key-mgmt "$key" 802-11-wireless-security.pmf "$pmf" 802-11-wireless-security.psk "$pass" connection.autoconnect yes >/dev/null \
        && echo "[$(ts)] connection profile realized for $ssid" \
        || echo "[$(ts)] connection profile realization failed for $ssid" >&2
    fi
  done < "$CONF"
else
  echo "[$(ts)] no readable network inventory or network configuration API" >&2
fi
exit 0
