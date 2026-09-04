#!/usr/bin/env bash
set -eu
here=$(cd "$(dirname "$0")" && pwd)
dp=$(cd "$here/.." && pwd)/darkplaces-work
brew_prefix=${HOMEBREW_PREFIX:-$(brew --prefix 2>/dev/null || printf /opt/homebrew)}
sdl_prefix=${SDL2_PREFIX:-$brew_prefix/opt/sdl2}
sdl3_prefix=${SDL3_PREFIX:-$brew_prefix/opt/sdl3}
jpeg_prefix=${JPEG_PREFIX:-$(brew --prefix jpeg-turbo 2>/dev/null || brew --prefix jpeg 2>/dev/null || printf '%s/opt/jpeg' "$brew_prefix")}
png_prefix=${PNG_PREFIX:-$(brew --prefix libpng 2>/dev/null || printf '%s/opt/libpng' "$brew_prefix")}
freetype_prefix=${FREETYPE_PREFIX:-$(brew --prefix freetype 2>/dev/null || printf '%s/opt/freetype' "$brew_prefix")}
target=${1:-sdl-release}
copy_library() {
  cp -fL "$1" "$dp/$2.new"
  mv -f "$dp/$2.new" "$dp/$2"
  chmod u+w "$dp/$2"
}
copy_library "$png_prefix/lib/libpng16.16.dylib" libpng16.16.dylib
copy_library "$freetype_prefix/lib/libfreetype.6.dylib" libfreetype.6.dylib
find "$dp/build-obj" -path '*/darkplaces-sdl/jpeg.o' -delete 2>/dev/null || true
make -C "$dp" -j"${JOBS:-8}" "$target" \
  DP_LINK_JPEG=shared \
  LIB_JPEG="-L$jpeg_prefix/lib -ljpeg" \
  CFLAGS_EXTRA="-I$jpeg_prefix/include" \
  SDLCONFIG_MACOSXCFLAGS="-I$sdl_prefix/include/SDL2 -D_THREAD_SAFE" \
  SDLCONFIG_MACOSXLIBS="-L$sdl_prefix/lib -lSDL2 -framework Cocoa" \
  SDLCONFIG_MACOSXSTATICLIBS="-L$sdl_prefix/lib -lSDL2 -framework Cocoa -I$sdl_prefix/include/SDL2"
copy_library "$jpeg_prefix/lib/libjpeg.8.dylib" libjpeg.8.dylib
copy_library "$sdl_prefix/lib/libSDL2-2.0.0.dylib" libSDL2-2.0.0.dylib
copy_library "$sdl3_prefix/lib/libSDL3.0.dylib" libSDL3.dylib
for library in libjpeg.8.dylib libSDL2-2.0.0.dylib libSDL3.dylib libpng16.16.dylib libfreetype.6.dylib; do
  install_name_tool -id "@loader_path/$library" "$dp/$library"
done
freetype_png=$(otool -L "$dp/libfreetype.6.dylib" | awk '$1 ~ /libpng16.*[.]dylib$/ { print $1; exit }')
install_name_tool -change "$freetype_png" '@loader_path/libpng16.16.dylib' "$dp/libfreetype.6.dylib"
for executable in darkplaces-sdl darkplaces-dedicated; do
  chmod u+w "$dp/$executable"
  jpeg_link=$(otool -L "$dp/$executable" | awk '$1 ~ /libjpeg.*[.]dylib$/ { print $1; exit }')
  install_name_tool -change "$jpeg_link" '@executable_path/libjpeg.8.dylib' "$dp/$executable"
done
sdl_link=$(otool -L "$dp/darkplaces-sdl" | awk '$1 ~ /libSDL2.*[.]dylib$/ { print $1; exit }')
install_name_tool -change "$sdl_link" '@executable_path/libSDL2-2.0.0.dylib' "$dp/darkplaces-sdl"
codesign --force --sign - "$dp/darkplaces-sdl" "$dp/darkplaces-dedicated" "$dp/libjpeg.8.dylib" "$dp/libSDL2-2.0.0.dylib" "$dp/libSDL3.dylib" "$dp/libpng16.16.dylib" "$dp/libfreetype.6.dylib"
