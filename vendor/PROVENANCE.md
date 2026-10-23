# vendor/babeld-arm64

Babel routing daemon, shipped prebuilt because building on a node requires `cc`,
and invoking `cc` on a virgin Mac triggers the Command Line Tools GUI installer —
the same trap `bootstrap.sh` avoids by not using `git`. A node must never need a
toolchain, a package manager, or a human at a keyboard.

    source   https://www.irif.fr/~jch/software/files/babeld-1.13.1.tar.gz
    tarball  sha256 15f24d26da0ccfc073abcdef0309f281e4684f2aa71126f826572c4c845e8dd9
    built    make LDLIBS=''
    binary   sha256 c4fff03982bb12abc70668058d2b5cb16a883271a1880c92ebbc47d4f4dcb433
    arch     Mach-O 64-bit executable arm64
    links    /usr/lib/libSystem.B.dylib only

`LDLIBS=''` is the only change: the stock Makefile hardcodes `-lrt`, which macOS
does not have — its `clock_gettime` lives in libc. Every object compiles unmodified;
`kernel_socket.c` is the BSD backend and already carries `__APPLE__` handling.

To reproduce:

    curl -fsSL https://www.irif.fr/~jch/software/files/babeld-1.13.1.tar.gz | tar xz
    cd babeld-1.13.1 && make LDLIBS='' && shasum -a 256 babeld

## NetRadiant / q3map2 capacity repair

The existing neighboring checkout is `/Users/mdot/dox/xonotic/netradiant-custom`.
Its only remote is `https://github.com/Garux/netradiant-custom.git`, branch `master`.
Its September 3 local change, "Remove q3map fixed-capacity compiler paths", is now also
named by local branch `codex/mesh-capacity`; no upstream write was made.

`netradiant-capacity.patch` carries the complete nine-file difference against upstream
`master` in mesh version control: thread work counts, VFS/image handling, light tracing,
patch meshes, and visibility storage/flow. Context retains upstream prose unchanged;
it is a source-reconstruction patch, not a new executable copy of the compiler.

From a mesh checkout, run:

```sh
bash bin/mesh-q3map2-build.sh
bin/mesh-q3map2 -help
```

Every build clones the latest `Q3MAP2_BRANCH` (default `master`) from `Q3MAP2_REPO`
(default the upstream URL above), applies the tracked patch or recognizes it already
applied, then invokes `make binaries-q3map2`. It never selects a commit hash. An upstream
change that conflicts with the patch reports a build failure while retaining the existing
compiler; repair the mesh-owned patch or select a named integration branch and rerun.

The macOS tool build uses Clang/Clang++, make, pkg-config, glib, libxml2, libpng, libjpeg,
zlib, and assimp. It builds the compiler target only, bypassing the upstream editor/GUI
dependency check; there is no package installation in the script. `MESH_MACLIBDIR`
defaults to `/opt/homebrew/lib`; `MESH_BUILD_JOBS` defaults to two. These are build-host
dependencies, not new requirements for nodes merely executing an already built game.

Build generations live in `.build/netradiant/` (or `Q3MAP2_BUILD_ROOT`). Successful builds
atomically update `current`; old executables and sources are not removed. The map tools
default to `bin/mesh-q3map2`, which uses the reconstructed compiler, then an available
existing checkout, then realizes the compiler when none exists. `Q3MAP2` remains an
explicit executable override. Rerun the build command to fetch current branch intent;
the compiler launcher itself does not fetch on every map compilation.

On September 4 the fresh upstream-branch clone plus patch completed compilation and
linking as arm64, and its `-help` command ran. No existing map compiler or game process
was interrupted. Stock Xonotic content remains the external `XON_BASEPATH` /
`XONOTIC_DIR` input; mesh does not redistribute the full installed asset tree here.

# vendor/libnccl-mesh.metallib

libnccl-mesh's own Metal kernels as AIR, the Metal compiler's portable form, which every Apple GPU's driver takes.
It is committed because only a machine with the Metal toolchain can build it: an Xcode with its Metal toolchain
component, which needs sudo and an Apple ID to install. On this mesh only the M5 has one. A node needs none: its
`make -C rdma install` uses this file and makes its own binary archive from it (rdma/NCCL.md, "The library's
kernels").

    text     rdma/mesh.metal, then rdma/nccl-mesh-kernels.h's STEPS, KERNELS, TRACKS and TRACK_KERNEL, as
             rdma/.build/nccl-mesh-kernels writes them (rdma/nccl-mesh-kernels.m)
    built    xcrun metal -std=metal4.0 -fmetal-math-mode=safe -mmacosx-version-min=26.0 -Wall -Werror
             -Wno-unused-function -Wno-error=deprecated-declarations -c, then xcrun metallib (rdma/Makefile METAL)
    record   vendor/libnccl-mesh.sha256: line 1 the kernels' SHA-256 (of the compile line above, a newline, and the
             text), line 2 the SHA-256 of vendor/libnccl-mesh.metallib
    with     Xcode 26.4.1, Apple metal version 32023.883 (metalfe-32023.883), MetalToolchain 17.5.188, on the M5
             (macOS 26.3.2); the same text and flags gave the same bytes twice
    arch     AIR (air64), Metal 4.0, macOS 26.0 and later, any Apple GPU

`-fmetal-math-mode=safe` is the build-time form of `MTLMathModeSafe`, which the run-time compile used, and Metal 4.0
is what that compile took by default on macOS 26.3. macOS 26.0 is the floor the library's Metal 4 calls already need.
The toolchain's AIR is not the run-time compiler's: an archive made from the run-time compile of the same text served
none of this file's 72 functions.

`make -C rdma kernels` (which `install` runs) keeps the record in step. Where its two lines are not these kernels' and
this file's, a machine with the toolchain compiles both anew, and a machine without one fails, naming that build. On
the M5:

    DEVELOPER_DIR=/Applications/Xcode-26.4.1.app/Contents/Developer make -C rdma kernels
    git add vendor/libnccl-mesh.metallib vendor/libnccl-mesh.sha256
