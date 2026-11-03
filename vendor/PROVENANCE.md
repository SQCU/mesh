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
