#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
{ ioreg -a -r -c AppleThunderboltRDMAPeerInterface -d 1; echo '<!--devices-->'; ioreg -a -r -c AppleThunderboltRDMAInterface -d 1; } 2>/dev/null | awk '
/<!--devices-->/ { devices = 1; next }
/<dict>/ { depth++; key = ""; next }
/<\/dict>/ {
  if (depth == 1) {
    if (!devices) { if (rid != "") peer[rid] = 1 }
    else if (!(id in seen)) {
      seen[id] = 1; port = name; sub(/^rdma_/, "", port)
      print port " device=" name " rid=" rid " rdma_peer=" ((rid in peer) ? "present" : "absent")
    }
    id = name = rid = ""
  }
  depth--; key = ""; next
}
depth == 1 && /<key>/ { key = $0; sub(/.*<key>/, "", key); sub(/<\/key>.*/, "", key); next }
depth == 1 && key != "" {
  value = $0; sub(/^[ \t]*<[a-z]+>/, "", value); sub(/<\/[a-z]+>[ \t]*$/, "", value)
  if (key == "IORegistryEntryID") id = value
  else if (key == "IORegistryEntryName") name = value
  else if (key == "RID") rid = value
  key = ""
}'
