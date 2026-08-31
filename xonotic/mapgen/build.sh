#!/bin/zsh
# build.sh -- spiralgen -> q3map2 -> pk3, one command.
#   ./build.sh <name> [spiralgen flags...]
set -e
SP=${0:A:h}
X=$SP/../Xonotic
Q=$SP/../netradiant-custom/install/q3map2.arm64
NAME=${1:-spiral}; shift || true
FS=$SP/fs
MAPS=$FS/data/maps
mkdir -p $MAPS
Q3="$Q -game xonotic -fs_basepath $X -fs_homepath $FS"

python3 $SP/spiralgen.py --name $NAME --out $MAPS "$@"

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
# NOTE: needs the threads.cpp stack-size fix in netradiant-custom, otherwise
# every multithreaded q3map2 stage SIGBUSes on macOS (512 KB worker stacks).
echo "--- LIGHT ---"; ${=Q3} -light -fast -bounce 1 $MAPS/$NAME.map > /dev/null

PK3=$SP/build/$NAME.pk3
mkdir -p $SP/build; rm -f $PK3
( cd $FS/data && zip -q -r $PK3 maps/$NAME.bsp maps/$NAME.mapinfo maps/$NAME.waypoints )
echo "OK -> $PK3  ($(du -h $PK3 | cut -f1))"
