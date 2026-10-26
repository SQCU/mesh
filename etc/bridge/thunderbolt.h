#pragma once
/* Provider inputs for this build. See design/RDMA-KERNEL-RECOVERY.md and
   design/HETEROGENEOUS-LINKS.md. Device queue capacities are still queried. */
#define MESH_FRAME 4096u
#define MESH_REQUEST_BYTES 16773120u
#define MESH_BANK (UINT64_C(1) << 32)
#define MESH_DMA_EXTENT (UINT64_C(1) << 30)
#define MESH_GID_ADDRESSES 16
#define MESH_GID_ENTRIES 256
/* ibv_port_attr speed/width codes, in bits/second and lanes respectively. */
#define MESH_PORT_SPEEDS { [1]=2500000000, [2]=5000000000, [4]=10000000000, [8]=10000000000, [16]=14000000000, [32]=25000000000, [64]=50000000000, [128]=100000000000 }
#define MESH_PORT_WIDTHS { [1]=1, [2]=4, [4]=8, [8]=12 }

/* Declared ABI facts, checked by the build against its headers. */
#define MESH_SIZEOF_IBV_SEND_WR 128
#define MESH_SIZEOF_IBV_SGE 16
#define MESH_SIZEOF_IBV_RECV_WR 32
#define MESH_SIZEOF_IBV_WC 48
#define MESH_OFFSET_IBV_WC_STATUS 8
#define MESH_OFFSET_IBV_WC_OPCODE 12
