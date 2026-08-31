#!/usr/bin/env bash
# Build the DarkPlaces SDL client on macOS against Homebrew SDL2.
# The stock makefile expects an SDL2.framework in ~/Library/Frameworks; this
# points it at the Homebrew keg instead. Everything else is the stock target.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
dp=$(cd "$here/.." && pwd)/darkplaces-work
sdl_prefix=${SDL2_PREFIX:-/opt/homebrew/opt/sdl2}
target=${1:-sdl-release}
exec make -C "$dp" -j"${JOBS:-8}" "$target" \
  SDLCONFIG_MACOSXCFLAGS="-I$sdl_prefix/include/SDL2 -D_THREAD_SAFE" \
  SDLCONFIG_MACOSXLIBS="-L$sdl_prefix/lib -lSDL2 -framework Cocoa" \
  SDLCONFIG_MACOSXSTATICLIBS="-L$sdl_prefix/lib -lSDL2 -framework Cocoa -I$sdl_prefix/include/SDL2"
