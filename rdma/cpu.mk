CPU_CONFIGURATION ?= ../etc/compute/cpu.mk
include $(CPU_CONFIGURATION)
SLEEF_SOURCE = .build/sleef-$(MESH_SLEEF_REVISION)

$(SLEEF_SOURCE)/CMakeLists.txt: $(CPU_CONFIGURATION)
	mkdir -p $(SLEEF_SOURCE)
	curl -fL '$(MESH_SLEEF_SOURCE)' -o .build/sleef.tar.gz
	tar -xf .build/sleef.tar.gz --strip-components=1 -C $(SLEEF_SOURCE)
	touch $@

.build/sleef/CMakeCache.txt: $(SLEEF_SOURCE)/CMakeLists.txt $(CPU_CONFIGURATION) cpu.mk
	cmake -S $(SLEEF_SOURCE) -B .build/sleef $(MESH_SLEEF_OPTIONS)

.build/libsleef.a: .build/sleef/CMakeCache.txt
	cmake --build .build/sleef --target sleef
	cp .build/sleef/lib/libsleef.a $@
