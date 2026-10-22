#!/bin/bash
REPO="$(cd "$(dirname "$0")" && pwd)"
MESH_ROOT=/usr/local/mesh
TB_SERVICE="Thunderbolt Bridge"
KS=/System/Library/CoreServices/RemoteManagement/ARDAgent.app/Contents/Resources/kickstart
FW=/usr/libexec/ApplicationFirewall/socketfilterfw

[ "$(id -u)" -eq 0 ] || exec sudo "$0" "$@"
export HOME=~root
launchctl enable system/com.openssh.sshd 2>/dev/null
launchctl bootstrap system /System/Library/LaunchDaemons/ssh.plist 2>/dev/null
RECORDED=$(cat "$MESH_ROOT/admin-user" 2>/dev/null)
id "$RECORDED" >/dev/null 2>&1 || RECORDED=
ADMIN_USER="${MESH_USER:-${RECORDED:-${SUDO_USER:-$(stat -f%Su /dev/console)}}}"
id "$ADMIN_USER" >/dev/null 2>&1 || ADMIN_USER=$(stat -f%Su /dev/console)
{ [ "$ADMIN_USER" = root ] || ! id "$ADMIN_USER" >/dev/null 2>&1; } && ADMIN_USER=$(dscl . -read /Groups/admin GroupMembership \
  | tr ' ' '\n' | grep -v -E 'GroupMembership:|^root$|^_|^$' | head -1)

sec(){ printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
ok(){  printf '  \033[32mOK  \033[0m %s\n' "$*"; }
failures=0; retained=0; replaced=0
bad(){ failures=$((failures+1)); printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
try(){ d="$1"; shift; "$@" >/dev/null 2>&1 && ok "$d" || bad "$d"; }
keys(){ grep -cve '^[[:space:]]*#' -e '^[[:space:]]*$' "$1"; }

echo "provisioning $ADMIN_USER   ($REPO)"
[ -n "$RECORDED" ] && [ "$RECORDED" != "$ADMIN_USER" ] && echo "admin user: $RECORDED -> $ADMIN_USER (MESH_USER)"

sec "1. Layout"
mkdir -p "$MESH_ROOT"/{bin,etc,jobs,log,templates,viz} /usr/local/bin
kick=
launchctl print system/io.mesh.beacon >/dev/null 2>&1 && ! cmp -s "$REPO/bin/mesh-beacon.sh" "$MESH_ROOT/bin/mesh-beacon.sh" && kick=io.mesh.beacon
launchctl print system/io.mesh.router >/dev/null 2>&1 && ! { cmp -s "$REPO/bin/mesh-router-init.sh" "$MESH_ROOT/bin/mesh-router-init.sh" && cmp -s "$REPO/vendor/babeld-arm64" "$MESH_ROOT/bin/babeld"; } && kick="$kick io.mesh.router"
launchctl print system/io.mesh.telemetry >/dev/null 2>&1 && ! { cmp -s "$REPO/user/mesh-telemetry.py" "$MESH_ROOT/bin/mesh-telemetry.py" && cmp -s "$REPO/etc/mesh-capacity.json" "$MESH_ROOT/etc/mesh-capacity.json"; } && kick="$kick io.mesh.telemetry"
launchctl print system/io.mesh.observer >/dev/null 2>&1 && ! { cmp -s "$REPO/viz/serve.py" "$MESH_ROOT/viz/serve.py" && cmp -s "$REPO/etc/mesh-capacity.json" "$MESH_ROOT/etc/mesh-capacity.json" && cmp -s "$REPO/etc/mesh-nodes.json" "$MESH_ROOT/etc/mesh-nodes.json"; } && kick="$kick io.mesh.observer"
install -C -m 755 -o root -g wheel "$REPO"/bin/*.sh "$MESH_ROOT/bin/"
try "UV runtime realized" "$MESH_ROOT/bin/mesh-runtime-install.sh" "$MESH_ROOT" "$REPO"
install -m 755 -o root -g wheel "$REPO"/bin/mesh-observe.py "$MESH_ROOT/bin/mesh-observe.py"
install -m 755 -o root -g wheel "$REPO"/bin/mesh-top.py "$MESH_ROOT/bin/mesh-top.py"
install -C -m 644 -o root -g wheel "$REPO"/rdma/mesh_disk.py "$REPO"/rdma/mesh-disk.h "$REPO"/rdma/peers.py "$MESH_ROOT/bin/"
install -S -m 755 -o root -g wheel "$REPO"/bin/mesh-stat "$MESH_ROOT/bin/mesh-stat" && rm -f "$MESH_ROOT/bin/libmesh.dylib" || bad "mesh-stat installed"
install -d -m 755 -o "${ADMIN_USER:-root}" "$MESH_ROOT/log/user" && chown -R "${ADMIN_USER:-root}" "$MESH_ROOT/log/user"
# the services' logs rotated by the system's newsyslog past 4 MB (etc/newsyslog-mesh.conf)
sed "s|__MESH_OWNER__|$(id -u "${ADMIN_USER:-root}"):$(id -g "${ADMIN_USER:-root}")|g" "$REPO"/etc/newsyslog-mesh.conf > /etc/newsyslog.d/io.mesh.conf
chmod 644 /etc/newsyslog.d/io.mesh.conf
install -C -m 755 -o root -g wheel "$REPO"/user/mesh-telemetry.py "$MESH_ROOT/bin/mesh-telemetry.py"
install -C -m 644 -o root -g wheel "$REPO"/etc/mesh-capacity.json "$MESH_ROOT/etc/mesh-capacity.json"
install -C -m 644 -o root -g wheel "$REPO"/etc/mesh-nodes.json "$REPO"/etc/mesh-store.json "$MESH_ROOT/etc/"
[ -f "$MESH_ROOT/etc/observer.json" ] || install -m 644 -o root -g wheel "$REPO"/etc/observer.json "$MESH_ROOT/etc/observer.json"
install -m 644 -o root -g wheel "$REPO"/etc/io.mesh.telemetry.plist "$MESH_ROOT/jobs/io.mesh.telemetry.plist.new"
install -m 644 -o root -g wheel "$REPO"/keys/authorized_keys "$MESH_ROOT/etc/authorized_keys"
install -C -m 755 -o root -g wheel "$REPO"/viz/serve.py "$MESH_ROOT/viz/serve.py"
install -C -m 644 -o root -g wheel "$REPO"/viz/index.html "$MESH_ROOT/viz/index.html"
rm -f /Library/LaunchDaemons/io.mesh.{bridge,update,keeper,cartlane.learner,cartlane.viewer}.plist "$MESH_ROOT"/bin/{mesh-flow,mesh-update.sh,mesh-keeper.sh,mesh_restart.py} "$MESH_ROOT"/policy{,.default}; rm -rf "$MESH_ROOT/rdma"
for L in io.mesh.keeper io.mesh.cartlane.learner io.mesh.cartlane.viewer; do launchctl bootout "system/$L" >/dev/null 2>&1; done
try "memory bandwidth sampler built" xcrun clang -fobjc-arc -framework Foundation -lIOReport "$REPO/user/mesh-bandwidth.m" -o "$REPO/user/mesh-bandwidth"
try "memory bandwidth sampler installed" install -m 755 -o root -g wheel "$REPO/user/mesh-bandwidth" "$MESH_ROOT/bin/mesh-bandwidth"
install -C -m 755 -o root -g wheel "$REPO"/vendor/babeld-arm64 "$MESH_ROOT/bin/babeld"
install -m 644 -o root -g wheel "$REPO"/templates/* "$MESH_ROOT/templates/" 2>/dev/null
ln -sf "$MESH_ROOT/bin/mesh-status.sh" /usr/local/bin/mesh-status
ln -sf "$MESH_ROOT/bin/mesh-peers.sh"  /usr/local/bin/mesh-peers
ln -sf "$MESH_ROOT/bin/mesh-observe.py" /usr/local/bin/mesh-observe
ln -sf "$MESH_ROOT/bin/mesh-top.py" /usr/local/bin/mesh-top
ln -sf "$MESH_ROOT/bin/mesh-python" /usr/local/bin/mesh-python
ok "$MESH_ROOT; mesh-status, mesh-peers, mesh-observe and mesh-top on PATH"
printf '%s\n' "${MESH_BRANCH:-main}" > "$MESH_ROOT/branch"
printf '%s\n' "${MESH_REPO:-SQCU/mesh}" > "$MESH_ROOT/repo"
printf '%s\n' "$ADMIN_USER" > "$MESH_ROOT/admin-user"

sec "2. Power"
for kv in sleep=0 displaysleep=10 disksleep=0 standby=0 autopoweroff=0 hibernatemode=0 \
          powernap=0 autorestart=1 womp=1 tcpkeepalive=1 ttyskeepawake=1 powermode=2; do
  try "$kv" pmset -a "${kv%%=*}" "${kv##*=}"
done

sec "3. Boot resilience"
try "restart on freeze"   systemsetup -setrestartfreeze on
try "no startup delay"    systemsetup -setwaitforstartupafterpowerfailure 0
try "computer sleep off"  systemsetup -setcomputersleep Never
try "network time"        systemsetup -setusingnetworktime on
try "auto-boot"           nvram auto-boot=true
try "screensaver off"     sudo -H -u "$ADMIN_USER" defaults -currentHost write com.apple.screensaver idleTime -int 0
try "no auto updates"     defaults write /Library/Preferences/com.apple.SoftwareUpdate AutomaticallyInstallMacOSUpdates -bool false
try "no update restart"   defaults write /Library/Preferences/com.apple.commerce AutoUpdateRestartRequired -bool false
try "firewall off"        "$FW" --setglobalstate off
try "stealth off"         "$FW" --setstealthmode off

sec "4. Remote access"
launchctl enable system/com.openssh.sshd 2>/dev/null
launchctl bootstrap system /System/Library/LaunchDaemons/ssh.plist 2>/dev/null
nc -z 127.0.0.1 22 >/dev/null 2>&1 && ok "sshd :22" || bad "sshd :22"
launchctl enable system/com.apple.screensharing 2>/dev/null
launchctl bootstrap system /System/Library/LaunchDaemons/com.apple.screensharing.plist 2>/dev/null
nc -z 127.0.0.1 5900 >/dev/null 2>&1 && ok "screensharing :5900" || bad "screensharing :5900"
"$KS" -activate -configure -allowAccessFor -specifiedUsers >/dev/null 2>&1
"$KS" -configure -users "$ADMIN_USER" -access -on -privs -all >/dev/null 2>&1 \
  && ok "ARD for $ADMIN_USER" || bad "ARD"

sec "5. Pubkey roster"
H=$(eval echo "~$ADMIN_USER"); AK="$H/.ssh/authorized_keys"
mkdir -p "$H/.ssh"; chmod 700 "$H/.ssh"; touch "$AK"
before=$(keys "$AK"); TMPK=$(mktemp)
cat "$AK" "$REPO/keys/authorized_keys" \
  | awk '/^[[:space:]]*#/||/^[[:space:]]*$/{next} !seen[$2]++' >"$TMPK"
install -m 600 "$TMPK" "$AK"; rm -f "$TMPK"; chown -R "$ADMIN_USER":staff "$H/.ssh"
ok "roster merged: $before -> $(keys "$AK") keys, nothing removed"
awk 'FNR==NR{ if(!/^[[:space:]]*#/ && NF) r[$2]=1; next }
     NF && !r[$2]{ print "  --   local key not in roster: " $3 }' "$REPO/keys/authorized_keys" "$AK"

sec "5a. Administrability"
SUDOTMP=$(mktemp)
printf '%s ALL=(ALL) NOPASSWD: ALL\n' "$ADMIN_USER" > "$SUDOTMP"
if visudo -cf "$SUDOTMP" >/dev/null 2>&1; then
  install -m 440 -o root -g wheel "$SUDOTMP" /etc/sudoers.d/mesh
  ok "passwordless sudo for $ADMIN_USER"
else
  bad "sudoers syntax check failed, not installed"
fi
rm -f "$SUDOTMP"
sudo -n -u "$ADMIN_USER" true 2>/dev/null && ok "verified: sudo needs no tty" || echo "  --   verify sudo from a fresh session"

sec "5b. Developer tools"
try "SDK present (verbs.h, librdma)" "$MESH_ROOT/bin/mesh-devtools-init.sh"

sec "5c. Known Wi-Fi networks"
[ -f "$REPO/networks.conf" ] && install -m 600 -o root -g wheel "$REPO/networks.conf" "$MESH_ROOT/networks.conf"
try "preferred networks seeded" "$MESH_ROOT/bin/mesh-networks.sh"

sec "6. Network"
try "fabric: bridge torn down, ports addressed" "$MESH_ROOT/bin/mesh-fabric-init.sh"
oldIFS=$IFS; IFS=$'\n'
svcs=($(networksetup -listallnetworkservices | sed '1d;s/^\*//' | grep -v "^${TB_SERVICE}$" | grep -v '^$'))
IFS=$oldIFS
networksetup -ordernetworkservices "${svcs[@]}" "$TB_SERVICE" >/dev/null 2>&1
last=$(networksetup -listallnetworkservices | sed '1d;s/^\*//' | grep -v '^$' | tail -1)
[ "$last" = "$TB_SERVICE" ] && ok "service order: $TB_SERVICE last of $((${#svcs[@]}+1))" \
                            || bad "service order: last is '$last', a peer could steal the default route"
ok "identity: host UUID $(ioreg -a -rd1 -c IOPlatformExpertDevice | plutil -extract 0.IOPlatformUUID raw -o - -)"

sec "7. RDMA"
st=$(rdma_ctl status 2>&1)
[ "$st" = enabled ] && ok "enabled, nvram $(nvram rdma-enable 2>/dev/null | awk '{print $2}')" \
                    || bad "rdma is '$st', physical Recovery OS visit required"

sec "7a. Tailnet"
[ -f "$REPO/tailscale.key" ] && install -m 600 -o root -g wheel "$REPO/tailscale.key" "$MESH_ROOT/tailscale.key"
"$MESH_ROOT/bin/mesh-tailscale-init.sh"
[ -d /Applications/Tailscale.app ] && ok "tailnet: Tailscale.app" || {
  [ -x "$MESH_ROOT/bin/tailscaled" ] && ok "tailscaled $(cat "$MESH_ROOT/tailscale/version" 2>/dev/null)" || bad "tailscaled not built ($MESH_ROOT/log/tailscale.log)"; }

sec "8. Daemons"
J=$MESH_ROOT/jobs
mkplist(){
  cat >"$J/$1.plist.new" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$1</string>
  <key>ProgramArguments</key><array>$2</array>
  <key>RunAtLoad</key><true/>
$3
  <key>StandardOutPath</key><string>/usr/local/mesh/log/$1.out.log</string>
  <key>StandardErrorPath</key><string>/usr/local/mesh/log/$1.err.log</string>
</dict></plist>
PL
  chmod 644 "$J/$1.plist.new"
}
launchctl bootout system/io.mesh.caffeinate >/dev/null 2>&1; rm -f /Library/LaunchDaemons/io.mesh.caffeinate.plist
mkplist io.mesh.beacon "<string>$MESH_ROOT/bin/mesh-beacon.sh</string>"
mkplist io.mesh.fabric "<string>$MESH_ROOT/bin/mesh-fabric-init.sh</string>"
sed -e "s|__MESH_USER__|$ADMIN_USER|g" -e "s|__MESH_HOME__|$H|g" "$REPO/etc/io.mesh.observer.plist" > "$J/io.mesh.observer.plist.new"
chmod 644 "$J/io.mesh.observer.plist.new"
mkplist io.mesh.router "<string>$MESH_ROOT/bin/mesh-router-init.sh</string>" \
  "  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>1</integer>"
mkplist io.mesh.rdma-init "<string>$MESH_ROOT/bin/mesh-rdma-init.sh</string>"
TSJ=io.mesh.tailscale; [ -d /Applications/Tailscale.app ] && TSJ=
[ -n "$TSJ" ] && mkplist io.mesh.tailscale "<string>$MESH_ROOT/bin/tailscaled</string><string>--statedir=$MESH_ROOT/tailscale</string><string>--socket=/var/run/tailscaled.socket</string>" \
  "  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>1</integer>"
ADMIN_UID=$(id -u "$ADMIN_USER")
mkplist io.mesh.bridge-boot "<string>/bin/launchctl</string><string>bootstrap</string><string>user/$ADMIN_UID</string><string>$H/Library/LaunchAgents/io.mesh.bridge.plist</string>"
for L in io.mesh.beacon io.mesh.fabric io.mesh.router io.mesh.telemetry io.mesh.observer io.mesh.rdma-init io.mesh.bridge-boot $TSJ; do
  port=
  case "$L" in io.mesh.telemetry) port=8788 ;; io.mesh.observer) port=8787 ;; esac
  upid=$(launchctl print "gui/$ADMIN_UID/$L" 2>/dev/null | awk '/^[[:space:]]*pid = /{print $3; exit}')
  if [ -n "$port" ] && [ -n "$upid" ] && lsof -nP -a -p "$upid" -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    mv -f "$J/$L.plist.new" "/Library/LaunchDaemons/$L.plist"
    retained=$((retained+1)); ok "$L listening on :$port (pid $upid); userspace provider retained"
    continue
  fi
  if out=$("$REPO/bin/mesh-agent.sh" "$J/$L.plist.new" system "/Library/LaunchDaemons/$L.plist"); then
    [ -n "$out" ] && { replaced=$((replaced+1)); ok "$out"; } || { retained=$((retained+1)); ok "$L loaded"; }
  else bad "$L NOT LOADED: $out"; fi
done
for label in io.mesh.telemetry io.mesh.observer; do
  launchctl print "system/$label" >/dev/null 2>&1 || launchctl print "gui/$ADMIN_UID/$label" >/dev/null 2>&1 || {
    launchctl bootstrap "gui/$ADMIN_UID" "$H/Library/LaunchAgents/$label.plist" >/dev/null 2>&1 \
      && ok "$label userspace provider loaded" || bad "$label has no provider loaded"; }
done
for P in /Library/LaunchDaemons/io.mesh.*.plist; do
  L=$(basename "$P" .plist)
  launchctl print "gui/$ADMIN_UID/$L" 2>/dev/null | grep -q '^[[:space:]]*pid = ' && continue
  if out=$("$REPO/bin/mesh-agent.sh" "$P" system); then [ -z "$out" ] || ok "$out"; else bad "$L NOT LOADED: $out"; fi
done
for P in "$H/Library/LaunchAgents/"*.plist; do
  [ "$(plutil -extract LimitLoadToSessionType raw "$P" 2>/dev/null)" = Background ] || continue
  if out=$("$REPO/bin/mesh-agent.sh" "$P" "user/$ADMIN_UID"); then [ -z "$out" ] || ok "$out"; else bad "$out"; fi
done
for L in $kick; do launchctl kickstart -k "system/$L" >/dev/null 2>&1 && ok "$L restarted on its new program (kickstart -k)" || bad "$L not restarted"; done
for L in io.mesh.beacon io.mesh.telemetry io.mesh.observer; do launchctl kickstart "system/$L" >/dev/null 2>&1; done
"$MESH_ROOT/bin/mesh-tailscale-init.sh"

sec "9. Status"
try "health report written (/var/run/mesh-health)" "$MESH_ROOT/bin/mesh-health.sh"
"$MESH_ROOT/bin/mesh-status.sh"
printf '%s %s\n' "${MESH_BRANCH:-main}" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$MESH_ROOT/revision"
printf 'reported_failures=%s retained_jobs=%s replaced_jobs=%s\n' "$failures" "$retained" "$replaced" > "$MESH_ROOT/install-status"
echo "installer completed: $failures reported failures; $retained loaded jobs current, $replaced loaded now or with a new definition for their next load"
launchctl bootout system/io.mesh.update >/dev/null 2>&1 || :
