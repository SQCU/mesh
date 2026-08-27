#!/bin/bash
# install.sh -- provision this Mac as an always-on RDMA mesh node.
# Idempotent; re-running it is also the drift fix. Run as root:
#   sudo ./install.sh
#
# What it does NOT do: enable RDMA. That is Recovery-OS-only (rdma_ctl exits 77
# under a booted system). See README -- it is the one irreducible physical step.

REPO="$(cd "$(dirname "$0")" && pwd)"
MESH_ROOT=/usr/local/mesh
ADMIN_USER="${MESH_USER:-${SUDO_USER:-mdot}}"
TB_SERVICE="Thunderbolt Bridge"

[ "$(id -u)" -eq 0 ] || { echo "must run as root: sudo ./install.sh"; exit 1; }
id "$ADMIN_USER" >/dev/null 2>&1 || { echo "no such user: $ADMIN_USER (set MESH_USER)"; exit 1; }
echo "provisioning node for user: $ADMIN_USER   (repo: $REPO)"

sec(){ printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
ok(){  printf '  \033[32mOK  \033[0m %s\n' "$*"; }
bad(){ printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
try(){ d="$1"; shift; if "$@" >/dev/null 2>&1; then ok "$d"; else bad "$d"; fi; }

sec "1. Layout"
mkdir -p "$MESH_ROOT"/{bin,jobs,log,templates}
install -m 755 -o root -g wheel "$REPO"/bin/*.sh "$MESH_ROOT/bin/"
install -m 644 -o root -g wheel "$REPO"/templates/* "$MESH_ROOT/templates/" 2>/dev/null
mkdir -p /usr/local/bin
ln -sf "$MESH_ROOT/bin/mesh-status.sh" /usr/local/bin/mesh-status
ln -sf "$MESH_ROOT/bin/mesh-peers.sh"  /usr/local/bin/mesh-peers
ok "$MESH_ROOT populated; 'mesh-status' and 'mesh-peers' on PATH"

sec "2. Power -- never sleep, always come back"
for kv in sleep=0 displaysleep=0 disksleep=0 standby=0 autopoweroff=0 hibernatemode=0 \
          powernap=0 autorestart=1 womp=1 tcpkeepalive=1 ttyskeepawake=1 powermode=2; do
  try "${kv}" pmset -a "${kv%%=*}" "${kv##*=}"
done

sec "3. Boot resilience"
try "restart on freeze"        systemsetup -setrestartfreeze on
try "no wait after power fail" systemsetup -setwaitforstartupafterpowerfailure 0
try "computer sleep Never"     systemsetup -setcomputersleep Never
# An unattended OS-update reboot is a silent downtime event.
try "no auto macOS updates"    defaults write /Library/Preferences/com.apple.SoftwareUpdate AutomaticallyInstallMacOSUpdates -bool false
try "no auto update restart"   defaults write /Library/Preferences/com.apple.commerce AutoUpdateRestartRequired -bool false

sec "3b. Anti-withdrawal -- neutralize the defaults that make a node go quiet"
# A node that withdraws itself is the failure this repo exists to prevent. Every
# line here disarms a stock macOS behaviour that can silently remove a working
# machine from the mesh for a reason that sounds locally sensible.
try "network time on"      systemsetup -setusingnetworktime on
try "auto-boot on power"   nvram auto-boot=true
try "screensaver never"    sudo -u "$ADMIN_USER" defaults -currentHost write com.apple.screensaver idleTime -int 0
# The application firewall is disabled deliberately. It can only ever *subtract*
# reachability, and unreachability is the threat we are defending against. The
# security boundary for this fleet is physical access to the room, not a host
# packet filter that might decide to stop answering.
try "app firewall off"     /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off
try "stealth mode off"     /usr/libexec/ApplicationFirewall/socketfilterfw --setstealthmode off

sec "4. Remote access"
# systemsetup -setremotelogin is gated behind Full Disk Access (TCC), which cannot
# be granted headlessly. launchctl needs only root, so it works on a virgin machine.
launchctl enable system/com.openssh.sshd 2>/dev/null
launchctl bootstrap system /System/Library/LaunchDaemons/ssh.plist 2>/dev/null
nc -z -G 2 127.0.0.1 22 >/dev/null 2>&1 && ok "sshd listening" || bad "sshd"
launchctl enable system/com.apple.screensharing 2>/dev/null
launchctl bootstrap system /System/Library/LaunchDaemons/com.apple.screensharing.plist 2>/dev/null
nc -z -G 2 127.0.0.1 5900 >/dev/null 2>&1 && ok "screen sharing listening" || bad "screen sharing"
KS=/System/Library/CoreServices/RemoteManagement/ARDAgent.app/Contents/Resources/kickstart
[ -x "$KS" ] && { "$KS" -activate -configure -allowAccessFor -specifiedUsers >/dev/null 2>&1
                  "$KS" -configure -users "$ADMIN_USER" -access -on -privs -all >/dev/null 2>&1
                  ok "ARD enabled for $ADMIN_USER"; }
/usr/libexec/ApplicationFirewall/socketfilterfw --setstealthmode off >/dev/null 2>&1

sec "5. Pubkey roster"
H=$(eval echo "~$ADMIN_USER")
mkdir -p "$H/.ssh"; chmod 700 "$H/.ssh"
if [ -f "$REPO/keys/authorized_keys" ]; then
  install -m 600 "$REPO/keys/authorized_keys" "$H/.ssh/authorized_keys"
  chown -R "$ADMIN_USER":staff "$H/.ssh"
  ok "roster installed ($(grep -cv '^#\|^$' "$REPO/keys/authorized_keys") keys)"
fi

sec "6. Network -- topology-independent identity"
# Nodes get unplugged, carried, and replugged elsewhere in the mesh. Static IPs encode
# POSITION, not identity, and collide on replug. Bonjour + link-local re-resolve anywhere.
try "TB Bridge auto/link-local" networksetup -setdhcp "$TB_SERVICE"
# TB Bridge last, so a peer can never steal the default route.
networksetup -ordernetworkservices "Ethernet" "Wi-Fi" "$TB_SERVICE" >/dev/null 2>&1 \
  && ok "service order: TB bridge last" || bad "service order"
ok "mesh identity: $(scutil --get LocalHostName).local"

sec "7. RDMA (verify only)"
st=$(rdma_ctl status 2>&1)
if [ "$st" = "enabled" ]; then ok "rdma enabled (nvram $(nvram rdma-enable 2>/dev/null | awk '{print $2}'))"
else bad "RDMA is '$st' -- needs a physical Recovery OS visit: rdma_ctl enable"; fi

sec "8. Daemons"
mkplist(){
  cat > "/Library/LaunchDaemons/$1.plist" <<PL
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
mkplist io.mesh.rdma-init "<string>$MESH_ROOT/bin/mesh-rdma-init.sh</string>" "  <key>KeepAlive</key><false/>"
mkplist io.mesh.keeper    "<string>$MESH_ROOT/bin/mesh-keeper.sh</string>"    "  <key>StartInterval</key><integer>60</integer>"
# The beacon must never exit voluntarily -- its silence is how the mesh detects
# that this node has withdrawn. KeepAlive makes launchd relight it immediately.
mkplist io.mesh.beacon    "<string>$MESH_ROOT/bin/mesh-beacon.sh</string>" \
  "  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>"
for L in io.mesh.caffeinate io.mesh.rdma-init io.mesh.keeper io.mesh.beacon; do
  launchctl bootout system/$L >/dev/null 2>&1
  launchctl bootstrap system "/Library/LaunchDaemons/$L.plist" >/dev/null 2>&1 \
    && { launchctl enable system/$L >/dev/null 2>&1; ok "$L"; } || bad "$L"
done

sec "9. Status"
"$MESH_ROOT/bin/mesh-status.sh"
