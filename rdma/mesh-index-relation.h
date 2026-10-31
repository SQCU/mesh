#pragma once
#import "mesh-tensor-layout.h"
#include <isl/aff.h>
#include <isl/map.h>
// Exact closure for a finite relation over at most count participating values.
// The construction supplies its direction and forbidden edges (NULL means none).
// Consumes relation, keeps forbidden. A witnessed forbidden edge returns NULL
// without setting an ISL error; an algebra error retains the context error.
enum MeshIndexClosureKind { MeshIndexSymmetricClosure, MeshIndexDirectedClosure };
isl_map *MeshIndexClosure(isl_map *relation, NSUInteger count, enum MeshIndexClosureKind kind, isl_map *forbidden);
#include <isl/set.h>
// Exact states reachable from seeds under the same finite path bound. Derive
// an inductive affine invariant before closing the relation; no seed or edge
// is approximated. Consumes relation and seeds.
isl_set *MeshIndexReachable(isl_map *relation, isl_set *seeds, NSUInteger count);

// Preparation only. The relation type and its closed operations are isl's;
// these functions translate the layout declarations and emitted index terms.
// No isl object or relation descriptor is retained by an execution record.
isl_pw_aff *MeshIndexExpression(isl_ctx *ctx, id expression, NSUInteger first, NSUInteger end);
isl_pw_aff *MeshIndexFunction(isl_ctx *ctx, MetalTensorLayout *layout);
isl_map *MeshIndexRelation(isl_ctx *ctx, MetalTensorLayout *layout);
isl_map *MeshIndexCompose(isl_map *source, isl_map *target); // consumes both
isl_map *MeshIndexNormalize(isl_map *relation); // consumes the relation
// A resolved selection declaration's exact logical position set. JSON is only
// a preparation-language boundary; no serialized descriptor reaches execution.
// On a declaration exception, error owns a message freed with mesh_isl_free.
isl_set *MeshIndexSelectionDomain(isl_ctx *ctx, const char *declaration, char **error);
isl_map *MeshIndexSelectionRelation(isl_ctx *ctx, const char *declaration, size_t *count, char **error);
// The same selection's ordered (offset,count) spans used by initializer
// realization. Repeated and reversed positions keep their declared order.
// Release returned records/error with mesh_isl_free.
int64_t *MeshIndexSelectionSpans(const char *declaration, size_t *count, char **error);
// The existing layout configuration, with its maps and coordinate expressions.
isl_map *MeshIndexLayoutRelation(isl_ctx *ctx, const char *configuration, size_t *count, char **error);
// A functional one-output relation as the shared scalar index term JSON.
// Borrow the relation; release the result/error with mesh_isl_free.
char *MeshIndexRelationTerms(isl_map *relation, char **error);
isl_pw_aff *MeshIndexRuns(isl_ctx *ctx, NSArray<NSArray<NSNumber *> *> *runs);
// Sorted unique positions as compact (first,width,step,count) records. Borrow
// the set; release the returned array with mesh_isl_free. NULL reports failure.
int64_t *MeshIndexSetRuns(isl_set *set, size_t *count);
struct MeshIndexCell { isl_set *domain; size_t *members; size_t count; };
// Borrow input sets. The caller owns each returned domain and releases the
// member arrays and cell array with mesh_isl_free after consuming them.
struct MeshIndexCell *MeshIndexSetCover(isl_set *const *domains, size_t count, size_t *cells);
MetalTensorLayout *MeshIndexSelection(isl_set *set); // borrows the set
id MeshIndexTerms(isl_pw_aff *expression); // consumes the expression
BOOL MeshIndexEqual(id left, id right, NSUInteger count);
struct MeshIndexBounds { NSInteger low, high; };
struct MeshIndexRun { NSInteger value, step; NSUInteger end; };
struct MeshIndexBounds MeshIndexImageBounds(isl_map *relation); // consumes a nonempty bounded relation
struct MeshIndexBounds MeshIndexBoundsOf(id expression, NSUInteger first, NSUInteger end);
struct MeshIndexRun MeshIndexRunOf(id expression, NSUInteger first, NSUInteger end);
