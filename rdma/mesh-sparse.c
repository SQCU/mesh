#include "mesh-sparse.h"
#include "mesh-isl-memory.h"
#include <stdlib.h>
#include <string.h>

struct domain_plan {
    struct mesh_sparse_interval *intervals;
    size_t count;
    struct domain_plan *owner;
    uint32_t first, end;
};
struct product_tile { uint32_t words[8]; };

static int interval_order(const void *a, const void *b) {
    const struct mesh_sparse_interval *x = a, *y = b;
    if (x->first != y->first) return x->first < y->first ? -1 : 1;
    return (x->count > y->count) - (x->count < y->count);
}
static int support_order(const void *a, const void *b) {
    const struct domain_plan *x = *(struct domain_plan *const *)a;
    const struct domain_plan *y = *(struct domain_plan *const *)b;
    size_t count = x->count < y->count ? x->count : y->count;
    for (size_t i = 0; i < count; ++i) {
        int order = interval_order(x->intervals + i, y->intervals + i);
        if (order) return order;
    }
    return (x->count > y->count) - (x->count < y->count);
}
static int tile_order(const void *a, const void *b) {
    const struct product_tile *x = a, *y = b;
    if (x->words[0] != y->words[0]) return x->words[0] < y->words[0] ? -1 : 1;
    return (x->words[1] > y->words[1]) - (x->words[1] < y->words[1]);
}
static size_t divided(size_t count, size_t width) {
    return count / width + (count % width != 0);
}
void mesh_sparse_product_free(struct mesh_sparse_product *result) {
    mesh_isl_free(result->records);
    mesh_isl_free(result->pieces);
    *result = (struct mesh_sparse_product){0};
}
int mesh_sparse_product_compile(const uint32_t geometry[4],
    const struct mesh_sparse_domain *domains, size_t domain_count,
    const struct mesh_sparse_interval *intervals, struct mesh_sparse_product *result) {
    *result = (struct mesh_sparse_product){.full_rows = 1, .full_columns = 1, .full_inner = 1};
    size_t interval_count = 0, tiles = 0, active = 0;
    for (size_t d = 0; d < domain_count; ++d) {
        if (!domains[d].rows || !domains[d].columns) continue;
        interval_count += domains[d].interval_count;
        tiles += divided(domains[d].rows, geometry[0]) * divided(domains[d].columns, geometry[1]);
    }
    if (!tiles) return 0;
    struct domain_plan *plans = mesh_isl_calloc(domain_count, sizeof *plans);
    struct domain_plan **ordered = mesh_isl_malloc(domain_count * sizeof *ordered);
    struct mesh_sparse_interval *normalized = mesh_isl_malloc(interval_count * sizeof *normalized);
    if (!plans || !ordered || !normalized) goto failed;
    size_t position = 0;
    for (size_t d = 0; d < domain_count; ++d) {
        const struct mesh_sparse_domain *domain = domains + d;
        if (!domain->rows || !domain->columns) continue;
        struct domain_plan *plan = plans + d;
        plan->intervals = normalized + position;
        for (size_t i = 0; i < domain->interval_count; ++i) {
            struct mesh_sparse_interval interval = intervals[domain->first_interval + i];
            if (interval.count) plan->intervals[plan->count++] = interval;
        }
        qsort(plan->intervals, plan->count, sizeof *normalized, interval_order);
        size_t merged = 0;
        for (size_t i = 0; i < plan->count; ++i) {
            struct mesh_sparse_interval next = plan->intervals[i];
            if (merged && next.first <= (uint64_t)plan->intervals[merged - 1].first + plan->intervals[merged - 1].count) {
                struct mesh_sparse_interval *last = plan->intervals + merged - 1;
                uint64_t end = (uint64_t)next.first + next.count;
                if (end > (uint64_t)last->first + last->count) last->count = end - last->first;
            } else plan->intervals[merged++] = next;
        }
        plan->count = merged;
        position += merged;
        ordered[active++] = plan;
    }
    qsort(ordered, active, sizeof *ordered, support_order);
    for (size_t first = 0, end; first < active; first = end) {
        struct domain_plan *owner = ordered[first];
        for (end = first + 1; end < active && !support_order(ordered + first, ordered + end); ++end)
            if (ordered[end] < owner) owner = ordered[end];
        for (size_t i = first; i < end; ++i) ordered[i]->owner = owner;
        for (size_t i = 0; i < owner->count; ++i)
            result->piece_count += divided(owner->intervals[i].count, geometry[2]);
    }
    result->groups = divided(tiles, geometry[3]);
    result->tiles = (size_t)result->groups * geometry[3];
    result->records = mesh_isl_calloc(result->tiles, sizeof(struct product_tile));
    result->pieces = mesh_isl_malloc(result->piece_count * 2 * sizeof *result->pieces);
    if (!result->records || !result->pieces) goto failed;
    struct product_tile *records = (struct product_tile *)result->records;
    size_t tile = 0, piece = 0;
    for (size_t d = 0; d < domain_count; ++d) {
        const struct mesh_sparse_domain *domain = domains + d;
        if (!domain->rows || !domain->columns) continue;
        struct domain_plan *plan = plans + d;
        if (plan->owner == plan) {
            plan->first = piece;
            for (size_t i = 0; i < plan->count; ++i) {
                struct mesh_sparse_interval interval = plan->intervals[i];
                for (uint64_t at = 0; at < interval.count; at += geometry[2]) {
                    uint32_t width = interval.count - at < geometry[2] ? interval.count - at : geometry[2];
                    result->pieces[2 * piece] = interval.first + at;
                    result->pieces[2 * piece++ + 1] = width;
                    result->full_inner &= width == geometry[2];
                }
            }
            plan->end = piece;
        }
        for (uint64_t r = 0; r < domain->rows; r += geometry[0]) {
            uint32_t rows = domain->rows - r < geometry[0] ? domain->rows - r : geometry[0];
            result->full_rows &= rows == geometry[0];
            for (uint64_t c = 0; c < domain->columns; c += geometry[1]) {
                uint32_t columns = domain->columns - c < geometry[1] ? domain->columns - c : geometry[1];
                result->full_columns &= columns == geometry[1];
                records[tile++] = (struct product_tile){{domain->row + r, domain->column + c, rows, columns,
                    plan->owner->first, plan->owner->end, 0, 0}};
            }
        }
    }
    qsort(records, tiles, sizeof *records, tile_order);
    mesh_isl_free(normalized);
    mesh_isl_free(ordered);
    mesh_isl_free(plans);
    return 0;
failed:
    mesh_isl_free(normalized);
    mesh_isl_free(ordered);
    mesh_isl_free(plans);
    mesh_sparse_product_free(result);
    return -1;
}
