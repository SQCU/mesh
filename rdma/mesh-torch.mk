# libmesh-torch.dylib, a torch program's publish on torch's MPS stream (mesh-torch.mm; design/aot-compilation-change-
# request.md, appendix item 7, "A publish waits for nothing": "They need a small Objective-C++ torch extension, built by
# a versioned rule against the node's installed torch, as torch_mesh's `_stream.mm` was (mesh 116b27dc)").
#   make -C rdma -f mesh-torch.mk TORCH=<the installed torch package directory, .../site-packages/torch>
# TORCH is an operand: the torch whose headers it is built against, the program's (config, not hard-coded).  Built as
# torch 2.14.0's own torch.utils.cpp_extension builds an extension on macOS: -std=c++20, and "The darwin linker needs
# explicit consent to ignore unresolved symbols" (`-undefined dynamic_lookup`), so torch's symbols bind to the torch the
# program has already loaded, never a second copy.  A makefile of its own, so rdma/Makefile, a prerequisite of the
# bridge's targets, is unchanged by it (the operator, 2026-11-05: "stop arbitrarily rebuilding unchanged dependencies")
TORCH ?=
# 2026-11-05 (design/compilation-coordination.md, "The revisions of 4e78f6f4"): mesh-section.h, which mesh-torch.mm
# includes since 32d9fcc3, among the inputs.  What must stay: remade only when its own inputs change
libmesh-torch.dylib: mesh-torch.mm mesh-torch.mk mesh-section.h
	$(if $(TORCH),,$(error TORCH: the installed torch package directory, .../site-packages/torch))
	c++ -std=c++20 -fblocks -O2 -Wall -Wextra -Werror -isystem $(TORCH)/include -dynamiclib \
	  -Wl,-install_name,@rpath/libmesh-torch.dylib -undefined dynamic_lookup -framework Foundation -framework Metal \
	  -o $@ mesh-torch.mm
