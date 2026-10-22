#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
x(){ ioreg -a -rd1 -c IOPlatformExpertDevice | plutil -extract "0.$1" raw -o - -; }
id=$(x IOPlatformUUID); h=${id//-/}; n=0
for ((i = 0; i < 32; i += 2)); do n=$(( (n * 0x01000193 & 0xffffffff) ^ 0x${h:i:2} )); done
n=$((n & 0x7fffffff))
exec dns-sd -R "$id" _meshnode._tcp local 8099 \
  node="$(( n ? n : 1 ))" node_id="$id" serial="$(x IOPlatformSerialNumber)" model="$(sysctl -n hw.model)" rdma="$(rdma_ctl status 2>&1)"
