#pragma once
#import "mesh-tensor-layout.h"
#include <isl/aff.h>
#include <isl/map.h>
#include <isl/set.h>

// Preparation only. The relation type and its closed operations are isl's;
// these functions translate the layout declarations and emitted index terms.
// No isl object or relation descriptor is retained by an execution record.
isl_pw_aff *MeshIndexExpression(isl_ctx *ctx, id expression, NSUInteger first, NSUInteger end);
isl_pw_aff *MeshIndexFunction(isl_ctx *ctx, MetalTensorLayout *layout);
isl_map *MeshIndexRelation(isl_ctx *ctx, MetalTensorLayout *layout);
isl_pw_aff *MeshIndexRuns(isl_ctx *ctx, NSArray<NSArray<NSNumber *> *> *runs);
id MeshIndexTerms(isl_pw_aff *expression); // consumes the expression
BOOL MeshIndexEqual(id left, id right, NSUInteger count);
struct MeshIndexBounds { NSInteger low, high; };
struct MeshIndexRun { NSInteger value, step; NSUInteger end; };
struct MeshIndexBounds MeshIndexImageBounds(isl_map *relation); // consumes a nonempty bounded relation
struct MeshIndexBounds MeshIndexBoundsOf(id expression, NSUInteger first, NSUInteger end);
struct MeshIndexRun MeshIndexRunOf(id expression, NSUInteger first, NSUInteger end);
