#pragma once
#import "mesh-tensor-layout.h"
#include <isl/aff.h>
#include <isl/map.h>
// Exact closure for a finite relation over at most count participating values.
isl_map *MeshIndexClosure(isl_map *relation, NSUInteger count);
#include <isl/set.h>

// Preparation only. The relation type and its closed operations are isl's;
// these functions translate the layout declarations and emitted index terms.
// No isl object or relation descriptor is retained by an execution record.
isl_pw_aff *MeshIndexExpression(isl_ctx *ctx, id expression, NSUInteger first, NSUInteger end);
isl_pw_aff *MeshIndexFunction(isl_ctx *ctx, MetalTensorLayout *layout);
isl_map *MeshIndexRelation(isl_ctx *ctx, MetalTensorLayout *layout);
// A resolved selection declaration's exact logical position set. JSON is only
// a preparation-language boundary; no serialized descriptor reaches execution.
// On a declaration exception, error owns a message freed with mesh_isl_free.
isl_set *MeshIndexSelectionDomain(isl_ctx *ctx, const char *declaration, char **error);
isl_map *MeshIndexSelectionRelation(isl_ctx *ctx, const char *declaration, size_t *count, char **error);
// A functional one-output relation as the shared scalar index term JSON.
// Borrow the relation; release the result/error with mesh_isl_free.
char *MeshIndexRelationTerms(isl_map *relation, char **error);
isl_pw_aff *MeshIndexRuns(isl_ctx *ctx, NSArray<NSArray<NSNumber *> *> *runs);
// Sorted unique positions as compact (first,width,step,count) records. Borrow
// the set; release the returned array with mesh_isl_free. NULL reports failure.
int64_t *MeshIndexSetRuns(isl_set *set, size_t *count);
MetalTensorLayout *MeshIndexSelection(isl_set *set); // borrows the set
id MeshIndexTerms(isl_pw_aff *expression); // consumes the expression
BOOL MeshIndexEqual(id left, id right, NSUInteger count);
struct MeshIndexBounds { NSInteger low, high; };
struct MeshIndexRun { NSInteger value, step; NSUInteger end; };
struct MeshIndexBounds MeshIndexImageBounds(isl_map *relation); // consumes a nonempty bounded relation
struct MeshIndexBounds MeshIndexBoundsOf(id expression, NSUInteger first, NSUInteger end);
struct MeshIndexRun MeshIndexRunOf(id expression, NSUInteger first, NSUInteger end);
