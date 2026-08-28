#!/bin/bash
REPO="$(cd "$(dirname "$0")" && pwd)"
MESH_ROOT=/usr/local/mesh
TB_SERVICE="Thunderbolt Bridge"
KS=/System/Library/CoreServices/RemoteManagement/ARDAgent.app/Contents/Resources/kickstart
FW=/usr/libexec/ApplicationFirewall/socketfilterfw

[ "$(id -u)" -eq 0 ] || exec sudo "$0" "$@"
ADMIN_USER="${MESH_USER:-${SUDO_USER:-$(stat -f%Su /dev/console)}}"
id "$ADMIN_USER" >/dev/null 2>&1 || ADMIN_USER=$(dscl . -read /Groups/admin GroupMembership \
  | tr ' ' '\n' | grep -v 'GroupMembership:\|^root$\|^_\|^$' | head -1)

sec(){ printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
ok(){  printf '  \033[32mOK  \033[0m %s\n' "$*"; }
bad(){ printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
try(){ d="$1"; shift; "$@" >/dev/null 2>&1 && ok "$d" || bad "$d"; }
keys(){ grep -cve '^[[:space:]]*#' -e '^[[:space:]]*$' "$1"; }

echo "provisioning $ADMIN_USER   ($REPO)"

sec "1. Layout"
mkdir -p "$MESH_ROOT"/{bin,jobs,log,templates} /usr/local/bin
install -m 755 -o root -g wheel "$REPO"/bin/*.sh "$MESH_ROOT/bin/"
install -m 755 -o root -g wheel "$REPO"/vendor/babeld-arm64 "$MESH_ROOT/bin/babeld"
install -m 644 -o root -g wheel "$REPO"/templates/* "$MESH_ROOT/templates/" 2>/dev/null
ln -sf "$MESH_ROOT/bin/mesh-status.sh" /usr/local/bin/mesh-status
ln -sf "$MESH_ROOT/bin/mesh-peers.sh"  /usr/local/bin/mesh-peers
ln -sf "$MESH_ROOT/bin/mesh-run.sh"    /usr/local/bin/mesh-run
ok "$MESH_ROOT; mesh-status and mesh-peers on PATH"
printf 'sleep 0\ndisplaysleep 0\ndisksleep 0\nstandby 0\nautorestart 1\nwomp 1\npowermode 2\nfirewall off\n' > "$MESH_ROOT/policy.default"
cp "$MESH_ROOT/policy.default" "$MESH_ROOT/policy"
ok "policy: fleet default (run ./hmi-epilogue.sh after this for a machine someone uses)"
printf '%s %s\n' "${MESH_BRANCH:-main}" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$MESH_ROOT/revision"
ok "converged from branch $(awk '{print $1}' "$MESH_ROOT/revision") at $(awk '{print $2}' "$MESH_ROOT/revision")"

sec "2. Power"
for kv in sleep=0 displaysleep=0 disksleep=0 standby=0 autopoweroff=0 hibernatemode=0 \
          powernap=0 autorestart=1 womp=1 tcpkeepalive=1 ttyskeepawake=1 powermode=2; do
  try "$kv" pmset -a "${kv%%=*}" "${kv##*=}"
done

sec "3. Boot resilience"
try "restart on freeze"   systemsetup -setrestartfreeze on
try "no startup delay"    systemsetup -setwaitforstartupafterpowerfailure 0
try "computer sleep off"  systemsetup -setcomputersleep Never
try "network time"        systemsetup -setusingnetworktime on
try "auto-boot"           nvram auto-boot=true
try "screensaver off"     sudo -u "$ADMIN_USER" defaults -currentHost write com.apple.screensaver idleTime -int 0
try "no auto updates"     defaults write /Library/Preferences/com.apple.SoftwareUpdate AutomaticallyInstallMacOSUpdates -bool false
try "no update restart"   defaults write /Library/Preferences/com.apple.commerce AutoUpdateRestartRequired -bool false
try "firewall off"        "$FW" --setglobalstate off
try "stealth off"         "$FW" --setstealthmode off

sec "4. Remote access"
launchctl enable system/com.openssh.sshd 2>/dev/null
launchctl bootstrap system /System/Library/LaunchDaemons/ssh.plist 2>/dev/null
nc -z -G 2 127.0.0.1 22 >/dev/null 2>&1 && ok "sshd :22" || bad "sshd :22"
launchctl enable system/com.apple.screensharing 2>/dev/null
launchctl bootstrap system /System/Library/LaunchDaemons/com.apple.screensharing.plist 2>/dev/null
nc -z -G 2 127.0.0.1 5900 >/dev/null 2>&1 && ok "screensharing :5900" || bad "screensharing :5900"
"$KS" -activate -configure -allowAccessFor -specifiedUsers >/dev/null 2>&1
"$KS" -configure -users "$ADMIN_USER" -access -on -privs -all >/dev/null 2>&1 \
  && ok "ARD for $ADMIN_USER" || bad "ARD"
"$KS" -restart -agent >/dev/null 2>&1

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
ok "identity: $(scutil --get LocalHostName).local"

sec "7. RDMA"
st=$(rdma_ctl status 2>&1)
[ "$st" = enabled ] && ok "enabled, nvram $(nvram rdma-enable 2>/dev/null | awk '{print $2}')" \
                    || bad "rdma is '$st', physical Recovery OS visit required"

sec "8. Daemons"
mkplist(){
  cat >"/Library/LaunchDaemons/$1.plist" <<PL
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
  chmod 644 "/Library/LaunchDaemons/$1.plist"
}
mkplist io.mesh.caffeinate "<string>/usr/bin/caffeinate</string><string>-dimsu</string>" \
  "  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>"
mkplist io.mesh.beacon "<string>$MESH_ROOT/bin/mesh-beacon.sh</string>" \
  "  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>"
mkplist io.mesh.fabric "<string>$MESH_ROOT/bin/mesh-fabric-init.sh</string>" "  <key>KeepAlive</key><false/>"
install -m 755 -o root -g wheel "$REPO"/user/mesh-nodeinfod.py "$MESH_ROOT/bin/mesh-nodeinfod.py"
MESH_PY=$(command -v python3 || echo /usr/bin/python3)
cat > /Library/LaunchDaemons/io.mesh.nodeinfo.plist <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>io.mesh.nodeinfo</string>
  <key>ProgramArguments</key><array><string>$MESH_PY</string><string>$MESH_ROOT/bin/mesh-nodeinfod.py</string></array>
  <key>EnvironmentVariables</key><dict>
    <key>MESH_NODEINFO</key><string>$MESH_ROOT/bin/mesh-nodeinfo.sh</string>
    <key>MESH_NODEINFO_PORT</key><string>8099</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>5</integer>
</dict></plist>
PL
chmod 644 /Library/LaunchDaemons/io.mesh.nodeinfo.plist
mkplist io.mesh.router "<string>$MESH_ROOT/bin/mesh-router-init.sh</string>" \
  "  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>"
mkplist io.mesh.rdma-init "<string>$MESH_ROOT/bin/mesh-rdma-init.sh</string>" "  <key>KeepAlive</key><false/>"
mkplist io.mesh.update "<string>$MESH_ROOT/bin/mesh-update.sh</string>" "  <key>StartInterval</key><integer>900</integer>"
mkplist io.mesh.keeper "<string>$MESH_ROOT/bin/mesh-keeper.sh</string>" "  <key>StartInterval</key><integer>60</integer>"
for L in io.mesh.caffeinate io.mesh.beacon io.mesh.fabric io.mesh.router io.mesh.nodeinfo io.mesh.rdma-init io.mesh.update io.mesh.keeper; do
  launchctl bootout system/$L >/dev/null 2>&1
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    launchctl print system/$L >/dev/null 2>&1 || break
    sleep 1
  done
  launchctl enable system/$L >/dev/null 2>&1
  for _ in 1 2 3; do
    launchctl bootstrap system "/Library/LaunchDaemons/$L.plist" >/dev/null 2>&1
    launchctl print system/$L >/dev/null 2>&1 && break
    sleep 2
  done
  launchctl print system/$L >/dev/null 2>&1 && ok "$L" || bad "$L NOT LOADED"
done

sec "9. Status"
printf '  waiting for resident daemons'
for _ in $(seq 1 20); do pgrep -qf "dns-sd -R" && break; printf '.'; sleep 1; done
echo
"$MESH_ROOT/bin/mesh-status.sh"
