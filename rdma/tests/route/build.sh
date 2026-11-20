#!/bin/sh
# builds route/train: the environment (C), the int8 network's Metal library, the Swift runtime. Run from tests/route.
set -e
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-26.4.1.app/Contents/Developer}
xcrun -sdk macosx cc -std=c11 -O2 -DACCELERATE_NEW_LAPACK -I ../.. -I .. -c sim.c -o sim.o
xcrun -sdk macosx cc -std=c11 -O2 -DACCELERATE_NEW_LAPACK -I ../.. -I .. -ffp-contract=off -c forward.c -o forward.o
xcrun -sdk macosx metal -std=metal4.0 -O3 -I . -c int8.metal -o int8.air
xcrun -sdk macosx metal -std=metal4.0 -O3 -I . -c net.metal -o net.air
xcrun -sdk macosx metallib int8.air net.air -o route.metallib
swiftc -O -import-objc-header route.h policy.swift learn.swift net.swift main.swift sim.o forward.o -framework Accelerate -o train
swiftc -O -import-objc-header int8mm.h int8check.swift -o int8check
xcrun -sdk macosx cc -std=c11 -O2 -DROUTE_POLICY -DACCELERATE_NEW_LAPACK -I ../.. -I .. -I . -ffp-contract=off ../switch.c sim.c forward.c -framework Accelerate -lm -o switch-policy
