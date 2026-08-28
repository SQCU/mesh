// the only thing a workload may change -- see RDMA-FIRST.md "The invariant core"
#ifndef MESH_F_H
#define MESH_F_H
#include <stdint.h>
struct wire { uint32_t magic, path, stream, seq; uint16_t bytes, src, dst; uint8_t flags, hops; };
#define WIRE_MAGIC 0x4d534831u
#define F_FIRST 1u
#define F_LAST  2u
#define F_META  4u
#define F_NACK  8u
#define HOP_BITS 2u
#define HOP_MASK 3u
#endif
