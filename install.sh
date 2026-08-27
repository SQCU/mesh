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

# Node profile. Two kinds of machine live on this fabric:
#   appliance -- a node that must never withdraw. Absence IS an alarm. (default)
#   portable  -- a laptop that legitimately sleeps and travels. Reachable when
#                awake, announces itself, holds the roster -- but its absence is
#                expected, not an incident.
# This distinction exists to protect the alarm's meaning. If a laptop closing its
# lid fired the same alarm as a dead appliance, the alarm would be ignored within
# a week, and a genuinely withdrawn node would go unnoticed. Set with:
#   sudo MESH_PROFILE=portable ./install.sh
PROFILE="${MESH_PROFILE:-appliance}"
case "$PROFILE" in appliance|portable) ;; *) echo "MESH_PROFILE must be appliance|portable"; exit 1;; esac

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

echo "$PROFILE" > "$MESH_ROOT/profile"

if [ "$PROFILE" = appliance ]; then
  sec "2. Power -- never sleep, always come back"
  for kv in sleep=0 displaysleep=0 disksleep=0 standby=0 autopoweroff=0 hibernatemode=0 \
            powernap=0 autorestart=1 womp=1 tcpkeepalive=1 ttyskeepawake=1 powermode=2; do
    try "${kv}" pmset -a "${kv%%=*}" "${kv##*=}"
  done
else
  sec "2. Power -- skipped (portable profile)"
  # A portable node is allowed to sleep; forcing a laptop to stay awake in a bag
  # is a thermal and battery problem, not a reliability win. Keep only the pieces
  # that make it reachable the moment it IS awake.
  try "wake on network" pmset -a womp 1
  try "tcp keepalive"   pmset -a tcpkeepalive 1
fi

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
if [ "$PROFILE" = appliance ]; then
  try "screensaver never"  sudo -u "$ADMIN_USER" defaults -currentHost write com.apple.screensaver idleTime -int 0
  # The application firewall is disabled deliberately. It can only ever *subtract*
  # reachability, and unreachability is the threat. The security boundary for an
  # appliance is physical access to the room -- not a host packet filter that might
  # decide to stop answering.
  try "app firewall off"   /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off
  try "stealth mode off"   /usr/libexec/ApplicationFirewall/socketfilterfw --setstealthmode off
else
  # A portable node leaves the room, and with it the physical access control that
  # justified disarming these. Do not strip a travelling machine's defences to buy
  # availability it does not owe us.
  echo "  --   firewall + screen lock left intact (portable leaves the security boundary)"
fi

sec "4. Remote access"
# systemsetup -setremotelogin is gated behind Full Disk Access (TCC), which cannot
# be granted headlessly. launchctl needs only root, so it works on a virgin machine.
launchctl enable system/com.openssh.sshd 2>/dev/null
launchctl bootstrap system /System/Library/LaunchDaemons/ssh.plist 2>/dev/null
nc -z -G 2 127.0.0.1 22 >/dev/null 2>&1 && ok "sshd listening" || bad "sshd"

# Screen sharing is the appliance's REPLACEMENT for the keyboard and monitor we
# want to unplug and never reattach: it is the only remote path to the handful of
# things ssh cannot drive (a post-major-upgrade Setup Assistant pane, a TCC
# prompt, a login window). On an appliance it is a lifeline and the keeper
# re-asserts it.
#
# On a portable it is the opposite. That machine still HAS its keyboard and
# monitor attached -- they are the reason it is portable -- so :5900 buys no
# recoverability and only adds a listening GUI surface on a laptop that leaves
# the room and joins untrusted networks. Same reasoning as the firewall split.
if [ "$PROFILE" = appliance ]; then
  launchctl enable system/com.apple.screensharing 2>/dev/null
  launchctl bootstrap system /System/Library/LaunchDaemons/com.apple.screensharing.plist 2>/dev/null
  nc -z -G 2 127.0.0.1 5900 >/dev/null 2>&1 && ok "screen sharing listening" || bad "screen sharing"
  KS=/System/Library/CoreServices/RemoteManagement/ARDAgent.app/Contents/Resources/kickstart
  [ -x "$KS" ] && { "$KS" -activate -configure -allowAccessFor -specifiedUsers >/dev/null 2>&1
                    "$KS" -configure -users "$ADMIN_USER" -access -on -privs -all >/dev/null 2>&1
                    ok "ARD enabled for $ADMIN_USER"; }
  /usr/libexec/ApplicationFirewall/socketfilterfw --setstealthmode off >/dev/null 2>&1
else
  echo "  --   screen sharing / ARD left untouched (portable keeps its own console)"
fi

sec "5. Pubkey roster"
H=$(eval echo "~$ADMIN_USER")
mkdir -p "$H/.ssh"; chmod 700 "$H/.ssh"
# MERGE the roster in; never truncate what is already there.
#
# An overwrite is a withdrawal vector wearing a hardening costume. The roster is
# a small file in a public repo, so on any node whose operator key is not yet in
# it -- which is EVERY node the moment someone adds a machine -- a straight copy
# revokes the only credential that currently reaches the box, and does it inside
# the provisioning run that was supposed to make the box more reachable. That is
# the exact failure this repo exists to prevent, and it is unrecoverable without
# the physical visit we are trying to spend only once.
#
# So: roster keys are added, local keys are kept, and any local key NOT in the
# roster is REPORTED rather than deleted. Convergence you can see beats
# convergence that locks you out; if a key should die, take it out of the roster
# and remove it deliberately, not as a side effect of a re-run.
AK="$H/.ssh/authorized_keys"
if [ -f "$REPO/keys/authorized_keys" ]; then
  touch "$AK"
  # grep -c prints 0 AND exits 1 on no match; a `|| echo 0` fallback would emit
  # the count twice on a virgin node. Let grep print its own zero.
  before=$(grep -cve '^[[:space:]]*#' -e '^[[:space:]]*$' "$AK" 2>/dev/null)
  TMPK=$(mktemp)
  # Key identity is the base64 body (field 2) -- comments and options drift, and
  # matching on the whole line would re-add a key every time its label changed.
  cat "$AK" "$REPO/keys/authorized_keys" \
    | awk '/^[[:space:]]*#/ || /^[[:space:]]*$/ {next} !seen[$2]++' > "$TMPK"
  install -m 600 "$TMPK" "$AK"; rm -f "$TMPK"
  chown -R "$ADMIN_USER":staff "$H/.ssh"
  after=$(grep -cve '^[[:space:]]*#' -e '^[[:space:]]*$' "$AK")
  ok "roster merged: $before local + roster -> $after keys (nothing removed)"
  # Surface drift instead of silently enforcing it.
  awk 'FNR==NR{ if(!/^[[:space:]]*#/ && NF) r[$2]=1; next }
       NF && !r[$2]{ print "  --   local key not in roster: " $3 " (" substr($2,1,16) "...)" }' \
      "$REPO/keys/authorized_keys" "$AK"
fi

sec "6. Network -- topology-independent identity"
# Nodes get unplugged, carried, and replugged elsewhere in the mesh. Static IPs encode
# POSITION, not identity, and collide on replug. Bonjour + link-local re-resolve anywhere.
try "TB Bridge auto/link-local" networksetup -setdhcp "$TB_SERVICE"
# TB Bridge last, so a peer can never steal the default route.
#
# Do NOT hardcode the service list. It was written from a machine that happened to
# have a service literally named "Ethernet"; a node without one (a laptop, or any
# Mac whose uplink is Wi-Fi) gets an argument list naming a service that does not
# exist, networksetup rejects the whole call, and the reorder becomes a SILENT
# no-op -- which is how a node ends up running with the fabric FIRST in service
# order, exactly the blackhole this line was added to prevent. Hardcoding also
# drops any service not on the list (Tailscale, iPhone USB), which is its own
# outage. Enumerate what is present, preserve relative order, move TB last.
reorder_tb_last(){
  local svcs="" s
  while IFS= read -r s; do
    case "$s" in ""|"An asterisk"*) continue ;; esac
    s="${s#\*}"                       # disabled services are prefixed with '*'
    [ "$s" = "$TB_SERVICE" ] && continue
    svcs="${svcs}${s}
"
  done
  [ -n "$svcs" ] || return 1
  local OIFS="$IFS"
  set -f; IFS='
'
  set -- $svcs "$TB_SERVICE"
  set +f; IFS="$OIFS"
  networksetup -ordernetworkservices "$@" >/dev/null 2>&1
}
if networksetup -listallnetworkservices 2>/dev/null | grep -q "^\*\{0,1\}$TB_SERVICE$"; then
  reorder_tb_last < <(networksetup -listallnetworkservices 2>/dev/null)
  # Verify the RESULT, not the exit code: a reorder that silently did nothing is
  # the failure mode we are fixing, so trusting $? would reproduce it.
  last=$(networksetup -listallnetworkservices 2>/dev/null | grep -v '^An asterisk' \
         | sed 's/^\*//' | grep -v '^$' | tail -1)
  if [ "$last" = "$TB_SERVICE" ]; then ok "service order: '$TB_SERVICE' last (of $(networksetup -listallnetworkservices 2>/dev/null | grep -vc '^An asterisk\|^$'))"
  else bad "service order: last is '$last', wanted '$TB_SERVICE' -- a peer could steal the default route"; fi
else
  echo "  --   no '$TB_SERVICE' service on this node (no fabric cable yet?)"
fi
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
# ThrottleInterval means a just-bootstrapped KeepAlive daemon takes up to ~10s to
# reappear. Wait for it rather than closing a successful run with a false alarm:
# a provisioning tool that cries wolf trains you to ignore the alarm that matters.
printf '  waiting for resident daemons'
for _ in $(seq 1 20); do pgrep -qf "dns-sd -R" && break; printf '.'; sleep 1; done
echo
"$MESH_ROOT/bin/mesh-status.sh"
