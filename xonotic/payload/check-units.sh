#!/usr/bin/env bash
set -eu
ROOT=${1:?qcsrc path}
QCC=${2:?gmqcc path}
cd "$ROOT/tools"
export QCCFLAGS_WATERMARK=payload
WORKDIR=../.tmp
CPP="cc -xc -E"
export QCC
QCCDEFS='-DNDEBUG=1 -DXONOTIC=1 -DWATERMARK="payload" -DENABLE_EFFECTINFO=0 -DENABLE_DEBUGDRAW=0 -DENABLE_DEBUGTRACE=0'
QCCFLAGS='-std=gmqcc -O3 -Wall -Werror -futf8 -freturn-assignments -frelaxed-switch -Ooverlap-locals -Wno-field-redeclared -Wno-unused-variable -Wno-implicit-function-pointer -Wno-missing-return-values'
. qcc.sh
cd ..
mkdir -p "$WORKDIR"
check_server="common/gamemodes/gamemode/payload/payload.qc common/gamemodes/gamemode/payload/sv_payload.qc"
check_client="common/gamemodes/gamemode/payload/payload.qc common/gamemodes/gamemode/payload/cl_payload.qc"
for prog in server client; do
  MODE=$prog
  includes="-include lib/_all.inc"
  [ -f "${prog}/_all.qh" ] && includes="${includes} -include ${prog}/_all.qh"
  eval "files=\$check_${prog}"
  for f in $files; do
    echo "--- $prog $f"
    qpp "$f" "test-${prog}.dat" ${includes} -I. ${QCCIDENT} ${QCCDEFS} > "${WORKDIR}/${prog}.qc"
    qcc ${QCCFLAGS} -o "../${WORKDIR}/test-${prog}.dat" "../${WORKDIR}/${prog}.qc" > /dev/null
  done
done
echo ALLOK
