# Preparation's exact integer relation algebra. The release includes imath;
# linking it statically avoids a node-local GMP or shared-library dependency.
MESH_ISL_VERSION = 0.27
MESH_ISL_SOURCE = https://libisl.sourceforge.io/isl-$(MESH_ISL_VERSION).tar.xz
# Minimum region lease for a size class during preparation. Larger objects grow
# the lease to fit; this is not a solver capacity or tensor-shape limit.
MESH_ISL_SLAB_BYTES = 65536
