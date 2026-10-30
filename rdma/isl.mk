ISL_CONFIGURATION ?= ../etc/compute/isl.mk
include $(ISL_CONFIGURATION)
ISL_SOURCE = .build/isl-$(MESH_ISL_VERSION)
ISL_TARBALL = .build/isl-$(MESH_ISL_VERSION).tar.xz
ISL_ARCHIVE = $(ISL_SOURCE)/.libs/libisl.a
ISL_HEADERS = -I$(ISL_SOURCE)/include -I$(ISL_SOURCE)
ISL_ALLOCATION = $(abspath mesh-isl-alloc.h)
ISL_CFLAGS = -O2 -fPIC -include $(ISL_ALLOCATION) \
  -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-strdup -fno-builtin-free

$(ISL_TARBALL):
	mkdir -p .build
	curl -fL '$(MESH_ISL_SOURCE)' -o $@

$(ISL_SOURCE)/configure: $(ISL_TARBALL)
	tar -xf $(ISL_TARBALL) -C .build
	touch $@

$(ISL_SOURCE)/.mesh-patched: $(ISL_SOURCE)/configure $(ISL_TARBALL) isl.patch
	tar -xf $(ISL_TARBALL) -C .build isl-$(MESH_ISL_VERSION)/isl_map_simplify.c
	cd $(ISL_SOURCE) && patch -p1 --fuzz=0 < ../../isl.patch
	touch $@

$(ISL_SOURCE)/Makefile: $(ISL_SOURCE)/.mesh-patched $(ISL_CONFIGURATION) isl.mk mesh-isl-alloc.h mesh-isl-memory.h
	@test ! -f $@ || $(MAKE) -C $(ISL_SOURCE) clean
	cd $(ISL_SOURCE) && ./configure --with-int=imath-32 --disable-shared --enable-static --disable-dependency-tracking CFLAGS='-O2 -fPIC'

$(ISL_ARCHIVE): $(ISL_SOURCE)/Makefile
	$(MAKE) -C $(ISL_SOURCE) gitversion.h
	$(MAKE) -C $(ISL_SOURCE) libisl.la CFLAGS='$(ISL_CFLAGS)'
