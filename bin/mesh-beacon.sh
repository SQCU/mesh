#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
exec dns-sd -R "$(scutil --get LocalHostName)" _meshnode._tcp local 8099 \
  model="$(sysctl -n hw.model)" rdma="$(rdma_ctl status 2>&1)"
