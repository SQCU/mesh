ISL_CONFIGURATION ?= ../etc/compute/isl.mk
include $(ISL_CONFIGURATION)
ISL_SOURCE = .build/isl-$(MESH_ISL_VERSION)
ISL_ARCHIVE = $(ISL_SOURCE)/.libs/libisl.a
ISL_HEADERS = -I$(ISL_SOURCE)/include -I$(ISL_SOURCE)

$(ISL_SOURCE)/configure:
	mkdir -p .build
	curl -fL '$(MESH_ISL_SOURCE)' -o .build/isl-$(MESH_ISL_VERSION).tar.xz
	tar -xf .build/isl-$(MESH_ISL_VERSION).tar.xz -C .build

$(ISL_SOURCE)/Makefile: $(ISL_SOURCE)/configure $(ISL_CONFIGURATION) isl.mk
	cd $(ISL_SOURCE) && ./configure --with-int=imath-32 --disable-shared --enable-static --disable-dependency-tracking CFLAGS='-O2 -fPIC'

$(ISL_ARCHIVE): $(ISL_SOURCE)/Makefile
	$(MAKE) -C $(ISL_SOURCE) gitversion.h
	$(MAKE) -C $(ISL_SOURCE) libisl.la
