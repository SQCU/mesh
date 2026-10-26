# Shared build operands for the macOS Metal backend. Hosts may explicitly pass
# METAL_CONFIGURATION=... when realizing a different backend configuration.
# All four current hosts support this API and shader-language floor.
MESH_MACOS_MIN = 26.0
MESH_METAL_LANGUAGE = 4.0
MESH_METAL_LANGUAGE_ENUM = MTLLanguageVersion4_0
MESH_METAL_MATH = safe
export MACOSX_DEPLOYMENT_TARGET = $(MESH_MACOS_MIN)

.PHONY: deployment-target
deployment-target:
	@echo $(MESH_MACOS_MIN)
