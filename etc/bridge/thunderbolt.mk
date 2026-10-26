# Apple Thunderbolt provider resource input, read at build time.
# The reported max_qp includes one provider-owned queue pair on the observed
# macOS 26 and 27 hosts (design/HETEROGENEOUS-LINKS.md). The device reports the
# total; no host name, OS major version or guessed total enters the planner.
MESH_PROVIDER_RESERVED_QPS = 1
