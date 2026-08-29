#!/bin/sh
set -e
cd "$(dirname "$0")"
cc -O2 -Wall -Wextra -Wold-style-definition -Wstrict-prototypes -Wsign-compare -Wdeclaration-after-statement -Wmissing-prototypes -I../../../rdma -o meshtest meshtest.c
"$(cd ../../../rdma && pwd)/mesh-stat"
./meshtest "$@"
