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
