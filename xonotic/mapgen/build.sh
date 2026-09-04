#!/bin/zsh
set -e
SP=${0:A:h}
X=${XONOTIC_DIR:-}
[[ -z $X ]] && for c in $SP/../Xonotic ~/dox/xonotic/Xonotic; do [[ -d $c ]] && X=$c && break; done
Q=${Q3MAP2:-}
[[ -z $Q ]] && for c in $SP/../netradiant-custom ~/dox/xonotic/netradiant-custom; do
  [[ -x $c/install/q3map2.arm64 ]] && Q=$c/install/q3map2.arm64 && break; done
[[ -x $Q ]] || { echo "q3map2 not found (set Q3MAP2=)"; exit 1; }
[[ -d $X ]] || { echo "Xonotic not found (set XONOTIC_DIR=)"; exit 1; }
NAME=${1:-spiral}; shift || true
FS=$SP/fs
MAPS=$FS/data/maps
mkdir -p $MAPS
Q3="$Q -game xonotic -fs_basepath $X -fs_homepath $FS"

$SP/../../bin/mesh-python $SP/spiralgen.py --name $NAME --out $MAPS "$@"

cat > $MAPS/$NAME.mapinfo <<MI
title $NAME
description Procedurally generated helical tunnel.
author spiralgen
cdtrack 2
has weapons
gametype dm
gametype tdm
gametype lms
gametype ka
MI

echo "--- BSP ---";   ${=Q3} -meta $MAPS/$NAME.map            | grep -iE 'leaked|^\*\*\* ERROR' && { echo "LEAK/ERROR"; exit 1; } || true
echo "--- VIS ---";   ${=Q3} -vis  $MAPS/$NAME.map            > /dev/null
echo "--- LIGHT ---"; ${=Q3} -light -fast -bounce 1 $MAPS/$NAME.map > /dev/null

PK3=$SP/build/$NAME.pk3
mkdir -p $SP/build; rm -f $PK3
( cd $FS/data && zip -q -r $PK3 maps/$NAME.bsp maps/$NAME.mapinfo maps/$NAME.waypoints )
echo "OK -> $PK3  ($(du -h $PK3 | cut -f1))"
