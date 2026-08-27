#!/bin/bash
# Announce this node on the fabric continuously, so that its ABSENCE is detectable.
#
# This exists for the failure that matters: the false negative. A node that is
# supposed to be powered, reachable, and accepting work, but silently is not.
# Nothing else in the system would notice that. A node that stops beaconing is the
# signal -- so this daemon must never exit voluntarily, and launchd restarts it if
# it does. Its silence is meaningful; do not make it quiet for any other reason.
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
NAME=$(scutil --get LocalHostName 2>/dev/null || hostname -s)
MODEL=$(sysctl -n hw.model 2>/dev/null)
RDMA=$(rdma_ctl status 2>&1)
exec dns-sd -R "$NAME" _meshnode._tcp local 8099 model="$MODEL" rdma="$RDMA"
