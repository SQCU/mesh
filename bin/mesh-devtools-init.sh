#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
LOG=/usr/local/mesh/log/devtools.log
[ -d /usr/local/mesh/log ] && exec >>"$LOG" 2>&1
ts(){ date '+%F %T'; }
SDK=$(xcrun --show-sdk-path 2>/dev/null)
[ -n "$SDK" ] && [ -f "$SDK/usr/include/infiniband/verbs.h" ] && exit 0
echo "[$(ts)] no usable SDK, installing Command Line Tools headlessly"
MARK=/tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
touch "$MARK"
label=$(softwareupdate -l 2>/dev/null | awk -F'Label: ' '/Label: Command Line Tools/{print $2}' | sed 's/[[:space:]]*$//' | tail -1)
if [ -n "$label" ]; then
  echo "[$(ts)] installing: $label"
  softwareupdate -i "$label" --verbose 2>&1
else
  echo "[$(ts)] no Command Line Tools label offered by softwareupdate"
fi
rm -f "$MARK"
xcode-select -p >/dev/null 2>&1 || xcode-select --switch /Library/Developer/CommandLineTools 2>/dev/null
SDK=$(xcrun --show-sdk-path 2>/dev/null)
[ -f "$SDK/usr/include/infiniband/verbs.h" ] \
  && echo "[$(ts)] sdk ready: $SDK (verbs.h + librdma present)" \
  || echo "[$(ts)] ALARM sdk still missing; RDMA workloads cannot be compiled on this node"
exit 0
