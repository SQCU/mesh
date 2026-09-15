BINS = mesh-flow mesh-stat libmesh.dylib libmesh_swift.dylib
all: $(BINS)
mesh-flow: mesh-flow.c mesh-verbs.h mesh.h mesh-dataflow.h mesh-dataflow.c Makefile
	mkdir -p .build
	cc $(CPPFLAGS) -fblocks -O2 -o .build/$@ mesh-flow.c mesh-dataflow.c -lrdma
	codesign --force --sign - .build/$@
	mv -f .build/$@ $@
mesh-stat: mesh-stat.c mesh.h
	cc -O2 -o $@ mesh-stat.c
libmesh.dylib: mesh-dataflow.c mesh-dataflow.h mesh-call.c mesh-call.h mesh.h
	cc -fblocks -O2 -dynamiclib -Wl,-install_name,@rpath/libmesh.dylib -o $@ mesh-dataflow.c mesh-call.c
libmesh_swift.dylib: ../swift/Mesh.swift ../swift/Bounds.swift ../swift/Topology.swift libmesh.dylib module.modulemap
	swiftc -O -emit-library -emit-module -module-name Mesh -I . -L . -lmesh -Xlinker -rpath -Xlinker @loader_path -Xlinker -install_name -Xlinker @rpath/libmesh_swift.dylib ../swift/Mesh.swift ../swift/Bounds.swift ../swift/Topology.swift -o $@
sync-on-remote-fill: ../examples/sync-on-remote-fill.swift libmesh_swift.dylib
	swiftc -O -parse-as-library -I . -L . -lmesh_swift -Xlinker -rpath -Xlinker @loader_path $< -o $@
bounds-table: ../examples/bounds-table.swift libmesh_swift.dylib
	swiftc -O -parse-as-library -I . -L . -lmesh_swift -Xlinker -rpath -Xlinker @loader_path $< -o $@
coreml-chain: ../examples/coreml-chain.swift libmesh_swift.dylib
	swiftc -O -parse-as-library -I . -L . -lmesh_swift -Xlinker -rpath -Xlinker @loader_path $< -o $@
indexed-gather: ../examples/indexed-gather.swift libmesh_swift.dylib
	swiftc -O -parse-as-library -I . -L . -lmesh_swift -Xlinker -rpath -Xlinker @loader_path $< -o $@
gram-chain: ../examples/gram-chain.swift libmesh_swift.dylib
	swiftc -O -parse-as-library -Xcc -DACCELERATE_NEW_LAPACK -I . -L . -lmesh_swift -Xlinker -rpath -Xlinker @loader_path $< -o $@
clean:
	rm -f $(BINS) sync-on-remote-fill bounds-table coreml-chain indexed-gather gram-chain Mesh.swiftmodule Mesh.swiftdoc Mesh.swiftsourceinfo Mesh.abi.json
.PHONY: all clean
