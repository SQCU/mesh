#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
x(){ printf '%s' "$1" | plutil -extract "$2" raw -o - - 2>/dev/null; }
d=$(ioreg -a -r -c AppleThunderboltRDMAInterface -d 1 2>/dev/null)
q=$(ioreg -a -r -c AppleThunderboltRDMAPeerInterface -d 1 2>/dev/null)
peers=" "; i=0
while x "$q" "$i.IORegistryEntryID" >/dev/null; do peers="$peers$(x "$q" "$i.RID") "; i=$((i+1)); done
seen=" "; i=0
while id=$(x "$d" "$i.IORegistryEntryID"); do
  n=$(x "$d" "$i.IORegistryEntryName"); r=$(x "$d" "$i.RID"); i=$((i+1))
  case "$seen" in *" $id "*) continue ;; esac
  seen="$seen$id "
  case "$peers" in *" $r "*) p=present ;; *) p=absent ;; esac
  echo "${n#rdma_} device=$n rid=$r rdma_peer=$p"
done
