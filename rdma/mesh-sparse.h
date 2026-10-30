#ifndef MESH_SPARSE_H
#define MESH_SPARSE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Preparation operands. Domains partition the requested output, including
   rectangles with no reduction terms. Intervals name retained input terms. */
struct mesh_sparse_interval { uint32_t first, count; };
struct mesh_sparse_domain {
    uint32_t row, rows, column, columns;
    size_t first_interval, interval_count;
};
struct mesh_sparse_product {
    uint32_t *records, *pieces;
    size_t tiles, piece_count;
    uint32_t groups, full_rows, full_columns, full_inner;
};

/* Geometry is row/column/inner tile width and SIMD groups per threadgroup.
   Records contain two uint4 values per tile; pieces contain uint2 values.
   tiles includes zero-area padding for the final threadgroup. The result owns
   preparation pages until free; copy its arrays into prepared page operands.
   Returns zero, or -1 with errno on an allocation failure. */
int mesh_sparse_product_compile(const uint32_t geometry[4],
    const struct mesh_sparse_domain *domains, size_t domain_count,
    const struct mesh_sparse_interval *intervals, struct mesh_sparse_product *result);
void mesh_sparse_product_free(struct mesh_sparse_product *result);

#ifdef __cplusplus
}
#endif
#endif
