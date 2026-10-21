#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
M=/usr/local/mesh
TS_VERSION=${MESH_TAILSCALE_VERSION:-1.102.3}
GO_VERSION=${MESH_GO_VERSION:-1.27.3}
[ -d "$M/log" ] && exec >>"$M/log/tailscale.log" 2>&1
ts(){ date '+%F %T'; }
[ -d /Applications/Tailscale.app ] && exit 0
mkdir -p "$M/tailscale"
if [ ! -x "$M/bin/tailscaled" ] || [ "$(cat "$M/tailscale/version" 2>/dev/null)" != "$TS_VERSION" ]; then
  floor=$(awk '$1=="#define" && $2=="MESH_DISK_FLOOR_GIB"{print $3}' "$M/bin/mesh-disk.h" 2>/dev/null)
  free=$(df -k "$M" | awk 'NR==2{print $4}')
  if [ -n "$floor" ] && [ "${free:-0}" -lt $(( (floor + 4) << 20 )) ]; then
    echo "[$(ts)] tailscale $TS_VERSION not built: $M has $((free >> 20)) GiB free, under the floor of $floor GiB plus the build's 4"
  else
    W=$(mktemp -d "$M/tailscale/build.XXXXXX")
    echo "[$(ts)] building tailscale $TS_VERSION with go $GO_VERSION in $W"
    if curl -fsSL "https://go.dev/dl/go$GO_VERSION.darwin-arm64.tar.gz" | tar xzf - -C "$W" \
      && env HOME="$W" GOPATH="$W/gopath" GOCACHE="$W/cache" GOMODCACHE="$W/mod" GOBIN="$W/bin" \
           "$W/go/bin/go" install "tailscale.com/cmd/tailscale@v$TS_VERSION" "tailscale.com/cmd/tailscaled@v$TS_VERSION" \
      && install -m 755 -o root -g wheel "$W/bin/tailscale" "$W/bin/tailscaled" "$M/bin/"; then
      printf '%s\n' "$TS_VERSION" > "$M/tailscale/version"
      echo "[$(ts)] tailscale $TS_VERSION installed"
      launchctl kickstart -k system/io.mesh.tailscale >/dev/null 2>&1
    else
      echo "[$(ts)] tailscale $TS_VERSION build failed; the installed tailscaled, if any, kept"
    fi
    rm -rf "$W"
  fi
fi
T="$M/bin/tailscale --socket=/var/run/tailscaled.socket"
state=$($T status --json 2>/dev/null | plutil -extract BackendState raw -o - - 2>/dev/null)
case "$state" in
  Running|Starting) ;;
  "") echo "[$(ts)] tailscaled not answering on /var/run/tailscaled.socket" ;;
  *)
    if [ -s "$M/tailscale.key" ]; then
      $T up --auth-key="file:$M/tailscale.key" --hostname="$(scutil --get LocalHostName)" --timeout=60s \
        && echo "[$(ts)] tailnet joined from $state" || echo "[$(ts)] tailnet join failed from $state"
    else
      $T up --hostname="$(scutil --get LocalHostName)" --timeout=10s
      echo "[$(ts)] tailnet $state and no $M/tailscale.key: log in at the URL above"
    fi ;;
esac
exit 0
