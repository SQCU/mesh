#!/bin/bash
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
H=/usr/local/mesh
OUT=/var/run/mesh-health
GC=/var/run/mesh-health.gc
SAMPLES=/var/run/mesh-health.free
DU=/var/run/mesh-health.du
NOW=$(date +%s)
FLOOR=$(awk '$1=="#define" && $2=="MESH_DISK_FLOOR_GIB"{print $3}' $H/bin/mesh-disk.h 2>/dev/null)
FLOOR=${FLOOR:-20}
FREE=$(df -k /System/Volumes/Data 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
USED=$(df -k /System/Volumes/Data 2>/dev/null | awk 'NR==2{print int($3/1048576)}')
STATE=ok
[ "${FREE:-0}" -lt $((2 * FLOOR)) ] && STATE=low
[ "${FREE:-0}" -lt "$FLOOR" ] && STATE=under_floor
echo "$NOW $FREE" >> "$SAMPLES"
tail -n 1440 "$SAMPLES" > "$SAMPLES.tmp" && mv -f "$SAMPLES.tmp" "$SAMPLES"
read -r T0 F0 < "$SAMPLES"
TREND=$(awk -v t0="$T0" -v f0="$F0" -v t="$NOW" -v f="$FREE" 'BEGIN{h=(t-t0)/3600; if (h < 1) print "unmeasured"; else printf "%.1f", (f0-f)*24/h}')
DAYS=$(awk -v r="$TREND" -v f="$FREE" -v fl="$FLOOR" 'BEGIN{if (r == "unmeasured" || r <= 0) print "none"; else printf "%.1f", (f-fl)/r}')
HOLDERS=$(pmset -g assertions 2>/dev/null | awk '
  /^ +pid [0-9]+\(/ && /(UserIsActive|PreventUserIdleDisplaySleep) named/ {
    match($0, /pid [0-9]+\([^)]*\)/); who=substr($0, RSTART+4, RLENGTH-4); gsub(/[[:space:]]/, "_", who)
    s=0; n=split($0, f, " "); for (i=1; i<=n; i++) if (f[i] ~ /^[0-9]+:[0-9][0-9]:[0-9][0-9]$/) { split(f[i], t, ":"); s=t[1]*3600+t[2]*60+t[3]; kind=f[i+1] }
    if (s >= 3600) printf "%s%s:%s:%.1fh", (c++ ? "," : ""), who, kind, s/3600 }')
if [ ! -f "$GC" ] || [ $((NOW - $(stat -f %m "$GC"))) -gt 3600 ]; then
  log show --last 26h --info --style compact --predicate 'process == "dasd" AND (eventMessage CONTAINS "aned.modelCacheGC" OR eventMessage CONTAINS "aned.danglingModelsGC")' 2>/dev/null \
    | awk '/COMPLETED/{last=$1"T"$2} /Decision: MNP/{refused++} END{printf "%s %d\n", (last == "" ? "none" : substr(last,1,19)), refused}' > "$GC.tmp" && mv -f "$GC.tmp" "$GC"
fi
read -r GCLAST GCREFUSED < "$GC"
if [ "$STATE" != ok ] && { [ ! -f "$DU" ] || [ $((NOW - $(stat -f %m "$DU"))) -gt 3600 ]; }; then
  READABLE=$(du -sxk /System/Volumes/Data 2>/dev/null | awk '{print int($1/1048576)}')
  TOP=$(du -xk -d 2 /System/Volumes/Data/Users 2>/dev/null | awk -F'\t' -v OFS='\t' '{sub(/^\/System\/Volumes\/Data/, "", $2)} split($2, p, "/") == 4' | sort -rn | awk -F'\t' 'NR<=5{printf "%s%s:%dG", (c++ ? "," : ""), $2, $1/1048576}')
  echo "$READABLE $TOP" > "$DU"
fi
[ -f "$DU" ] && read -r READABLE TOP < "$DU"
{
  echo "health_updated=$(date -r "$NOW" '+%FT%T')"
  echo "disk_free_gib=$FREE disk_used_gib=$USED disk_floor_gib=$FLOOR disk_state=$STATE disk_trend_gib_day=$TREND disk_days_to_floor=$DAYS"
  echo "idle_blockers=${HOLDERS:-none} aned_gc_last=$GCLAST aned_gc_refused_26h=$GCREFUSED"
  [ "$STATE" != ok ] && echo "disk_readable_gib=${READABLE:-unmeasured} disk_os_only_gib=$([ -n "$READABLE" ] && echo $((USED - READABLE)) || echo unmeasured) disk_top=${TOP:-unmeasured}"
  [ "$STATE" = low ] && echo "warn=disk: $FREE GiB free, under twice the $FLOOR GiB floor (mesh-disk.h); $USED GiB used, ${READABLE:-?} GiB readable by du, the rest only the OS can account for (aned's model cache, staged updates); largest: ${TOP:-unmeasured}"
  [ "$STATE" = under_floor ] && echo "warn=disk: $FREE GiB free, under the $FLOOR GiB floor (mesh-disk.h): logs, records and compiled shares are being skipped; $USED GiB used, ${READABLE:-?} GiB readable by du; largest: ${TOP:-unmeasured}"
  [ "$DAYS" != none ] && [ "${DAYS%.*}" -lt 7 ] && echo "warn=disk: free space falling ${TREND} GiB a day, the floor in ${DAYS} days at this rate"
  [ "$STATE" != ok ] && [ -n "$HOLDERS" ] && echo "warn=idle: $HOLDERS held over an hour; while a process declares the user active or keeps the display on, dasd runs no idle maintenance and aned's SIP-protected model cache is never collected (README.md, power)"
  [ "$STATE" != ok ] && [ "$GCLAST" = none ] && [ "${GCREFUSED:-0}" -gt 0 ] && echo "warn=idle: dasd refused aned's cache GC $GCREFUSED times in 26 h and it has not completed there; its cache only shrinks when the node is idle (pmset -g assertions)"
  for f in /var/run/mesh-link.*; do
    [ -f "$f" ] || continue; read -r t0 n tb < "$f"
    echo "warn=link: ${f#/var/run/mesh-link.} cable link up but no RDMA peer in the IORegistry, for $((NOW - ${t0:-$NOW})) s; the fabric init has bounced the interface ${n:-0} times, with backoff to 30 min (fabric.log)"
  done
} > "$OUT.tmp"
mv -f "$OUT.tmp" "$OUT"
chmod 644 "$OUT"
cat "$OUT"
