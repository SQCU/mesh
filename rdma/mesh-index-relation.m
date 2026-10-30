#import "mesh-index-relation.h"
#include <isl/ast_build.h>
#include <isl/id.h>
#include <isl/ilp.h>
#include <isl/val.h>
#include "mesh-isl-memory.h"
#include <string.h>

isl_map *MeshIndexNormalize(isl_map *relation) {
    return isl_map_coalesce(isl_map_detect_equalities(isl_map_compute_divs(relation)));
}

isl_map *MeshIndexCompose(isl_map *source,isl_map *target) {
    if(isl_map_is_single_valued(source)==isl_bool_true&&isl_map_is_single_valued(target)==isl_bool_true) {
        isl_pw_multi_aff *first=isl_pw_multi_aff_from_map(source),*second=isl_pw_multi_aff_from_map(target);
        return isl_map_from_pw_multi_aff(isl_pw_multi_aff_pullback_pw_multi_aff(second,first));
    }
    return MeshIndexNormalize(isl_map_apply_range(source,target));
}

static NSDictionary *IndexDeclaration(const char *declaration) {
    NSData *data=[NSData dataWithBytesNoCopy:(void *)declaration length:strlen(declaration) freeWhenDone:NO];
    NSError *error=nil;NSDictionary *value=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if(error)[NSException raise:NSInvalidArgumentException format:@"index declaration: %@",error];
    return value;
}

isl_map *MeshIndexSelectionRelation(isl_ctx *ctx,const char *declaration,size_t *count,char **message) { @autoreleasepool {
    *message=NULL;
    @try {
        MetalTensorLayout *layout=[MetalTensorLayout selection:IndexDeclaration(declaration) extent:nil];*count=layout.count;
        return MeshIndexRelation(ctx,layout);
    } @catch(NSException *error) {
        *message=mesh_isl_strdup(error.reason.UTF8String);
        return NULL;
    }
}}

int64_t *MeshIndexSelectionSpans(const char *declaration,size_t *count,char **message) { @autoreleasepool {
    *count=0;*message=NULL;
    @try {
        NSData *ordered=[MetalTensorLayout selection:IndexDeclaration(declaration) extent:nil].orderedSpans;
        const struct MetalTensorSpan *spans=ordered.bytes;*count=ordered.length/sizeof *spans;
        int64_t *result=mesh_isl_malloc(*count*2*sizeof(int64_t));if(!result)return NULL;
        for(size_t i=0;i<*count;i++){result[i*2]=spans[i].offset;result[i*2+1]=spans[i].count;}
        return result;
    } @catch(NSException *error) {
        *message=mesh_isl_strdup(error.reason.UTF8String);return NULL;
    }
}}

isl_map *MeshIndexLayoutRelation(isl_ctx *ctx,const char *configuration,size_t *count,char **message) { @autoreleasepool {
    *message=NULL;
    @try {
        MetalTensorLayout *layout=[MetalTensorLayout configuration:IndexDeclaration(configuration)];*count=layout.count;
        return MeshIndexRelation(ctx,layout);
    } @catch(NSException *error) {
        *message=mesh_isl_strdup(error.reason.UTF8String);
        return NULL;
    }
}}

isl_set *MeshIndexSelectionDomain(isl_ctx *ctx,const char *declaration,char **message) {
    size_t count=0;isl_map *relation=MeshIndexSelectionRelation(ctx,declaration,&count,message);
    return relation?isl_map_range(relation):NULL;
}

char *MeshIndexRelationTerms(isl_map *relation,char **message) { @autoreleasepool {
    *message=NULL;
    @try {
        isl_pw_multi_aff *function=isl_pw_multi_aff_from_map(isl_map_copy(relation));
        isl_pw_aff *scalar=isl_pw_multi_aff_get_pw_aff(function,0);isl_pw_multi_aff_free(function);
        id terms=MeshIndexTerms(scalar);NSError *error=nil;
        NSData *data=[NSJSONSerialization dataWithJSONObject:terms options:NSJSONWritingFragmentsAllowed error:&error];
        if(error)[NSException raise:NSInvalidArgumentException format:@"index terms: %@",error];
        NSString *text=[[[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] autorelease];
        return mesh_isl_strdup(text.UTF8String);
    } @catch(NSException *error) {
        *message=mesh_isl_strdup(error.reason.UTF8String);return NULL;
    }
}}

static isl_set *IndexDomain(isl_ctx *ctx, NSUInteger first, NSUInteger end) {
    return isl_set_read_from_str(ctx, [[NSString stringWithFormat:@"{ [i] : %lu <= i < %lu }",
        (unsigned long)first, (unsigned long)end] UTF8String]);
}
static isl_pw_aff *IndexLeaf(isl_ctx *ctx, id value, NSUInteger first, NSUInteger end) {
    NSString *text=[NSString stringWithFormat:@"{ [i] -> [(%@)] : %lu <= i < %lu }",
        value, (unsigned long)first, (unsigned long)end];
    return isl_pw_aff_read_from_str(ctx,text.UTF8String);
}
// Compile each branch on the positions where its condition holds. Index
// terms are pure values; constructing an unreachable arm cannot add meaning,
// and rebuilding every arm over the full domain duplicates relation work.
static isl_pw_aff *IndexExpression(isl_ctx *ctx,id expression,isl_set *domain) {
    if([expression isKindOfClass:NSNumber.class]||[expression isEqual:@"i"]) {
        NSString *text=[NSString stringWithFormat:@"{ [i] -> [(%@)] }",expression];
        return isl_pw_aff_intersect_domain(isl_pw_aff_read_from_str(ctx,text.UTF8String),isl_set_copy(domain));
    }
    NSString *op=expression[0];
    isl_pw_aff *a=IndexExpression(ctx,expression[1],domain);
    if([op isEqual:@"mul"]||[op isEqual:@"div"]||[op isEqual:@"mod"]) {
        isl_pw_aff *b=IndexExpression(ctx,expression[2],domain);
        if([op isEqual:@"mul"])return isl_pw_aff_mul(a,b);
        if([op isEqual:@"div"])return isl_pw_aff_floor(isl_pw_aff_div(a,b));
        isl_pw_aff *q=isl_pw_aff_floor(isl_pw_aff_div(isl_pw_aff_copy(a),isl_pw_aff_copy(b)));
        return isl_pw_aff_sub(a,isl_pw_aff_mul(q,b));
    }
    if([op isEqual:@"select"]) {
        isl_set *yes=isl_pw_aff_non_zero_set(a),*no=isl_set_subtract(isl_set_copy(domain),isl_set_copy(yes));
        isl_pw_aff *left=isl_set_is_empty(yes)==isl_bool_true?
            isl_pw_aff_val_on_domain(isl_set_copy(yes),isl_val_zero(ctx)):IndexExpression(ctx,expression[2],yes);
        isl_pw_aff *right=isl_set_is_empty(no)==isl_bool_true?
            isl_pw_aff_val_on_domain(isl_set_copy(no),isl_val_zero(ctx)):IndexExpression(ctx,expression[3],no);
        isl_set_free(yes);isl_set_free(no);
        return isl_pw_aff_union_add(left,right);
    }
    for(NSUInteger k=2;k<[expression count];k++) {
        isl_pw_aff *b=IndexExpression(ctx,expression[k],domain);
        if([op isEqual:@"add"])a=isl_pw_aff_add(a,b);
        else if([op isEqual:@"min"])a=isl_pw_aff_min(a,b);
        else if([op isEqual:@"max"])a=isl_pw_aff_max(a,b);
        else {
            isl_set *condition=NULL;
            if([op isEqual:@"eq"])condition=isl_pw_aff_eq_set(a,b);
            else if([op isEqual:@"le"])condition=isl_pw_aff_le_set(a,b);
            else if([op isEqual:@"lt"])condition=isl_pw_aff_lt_set(a,b);
            else if([op isEqual:@"ge"])condition=isl_pw_aff_ge_set(a,b);
            else if([op isEqual:@"gt"])condition=isl_pw_aff_gt_set(a,b);
            else if([op isEqual:@"and"])condition=isl_set_intersect(isl_pw_aff_non_zero_set(a),isl_pw_aff_non_zero_set(b));
            else if([op isEqual:@"or"])condition=isl_set_union(isl_pw_aff_non_zero_set(a),isl_pw_aff_non_zero_set(b));
            else [NSException raise:NSInvalidArgumentException format:@"unknown index term %@",op];
            a=isl_pw_aff_intersect_domain(isl_set_indicator_function(condition),isl_set_copy(domain));
        }
    }
    return a;
}
isl_pw_aff *MeshIndexExpression(isl_ctx *ctx,id expression,NSUInteger first,NSUInteger end) {
    isl_set *domain=IndexDomain(ctx,first,end);
    isl_pw_aff *value=IndexExpression(ctx,expression,domain);isl_set_free(domain);return value;
}

// A finite map is a union of its affine runs. Its surrounding tensor axes do
// not multiply the table. Composition and inverse use the same relation type.
static isl_pw_aff *IndexTable(isl_ctx *ctx,NSData *table) {
    const NSInteger *values=table.bytes;NSUInteger count=table.length/sizeof *values;
    isl_pw_aff *result=IndexLeaf(ctx,@0,0,0);
    for(NSUInteger first=0;first<count;) {
        NSUInteger end=first+1;NSInteger step=end<count?values[end]-values[first]:0;
        while(end<count&&values[end]==values[first]+(NSInteger)(end-first)*step)end++;
        isl_pw_aff *run=MeshIndexExpression(ctx,@[@"add",@(values[first]-(NSInteger)first*step),@[@"mul",@"i",@(step)]],first,end);
        result=isl_pw_aff_union_add(result,run);first=end;
    }
    return isl_pw_aff_coalesce(result);
}
isl_pw_aff *MeshIndexRuns(isl_ctx *ctx,NSArray<NSArray<NSNumber *> *> *runs) {
    isl_pw_aff *result=IndexLeaf(ctx,@0,0,0);NSUInteger begin=0;
    for(NSArray *run in runs) {
        NSUInteger width=[run[1] unsignedIntegerValue],count=width*[run[3] unsignedIntegerValue];
        if(!count)continue;
        id local=@[@"add",@"i",@(-(NSInteger)begin)];
        id value=@[@"add",run[0],@[@"mul",@[@"div",local,@(width)],run[2]],@[@"mod",local,@(width)]];
        result=isl_pw_aff_union_add(result,MeshIndexExpression(ctx,value,begin,begin+count));begin+=count;
    }
    return isl_pw_aff_coalesce(result);
}
static isl_set *SetInterval(isl_ctx *ctx,NSInteger first,NSInteger end) {
    return isl_set_read_from_str(ctx,[[NSString stringWithFormat:@"{ [x] : %ld <= x < %ld }",(long)first,(long)end] UTF8String]);
}
static NSInteger SetEdge(isl_set *set,BOOL maximum) {
    isl_val *value=maximum?isl_set_dim_max_val(isl_set_copy(set),0):isl_set_dim_min_val(isl_set_copy(set),0);
    NSInteger result=isl_val_get_num_si(value);isl_val_free(value);return result;
}
static isl_set *SetRun(isl_ctx *ctx,NSInteger first,NSInteger width,NSInteger step,NSInteger count) {
    if(count==1)return SetInterval(ctx,first,first+width);
    return isl_set_read_from_str(ctx,[[NSString stringWithFormat:
        @"{ [x] : %ld <= x < %ld and (x - %ld) mod %ld < %ld }",(long)first,(long)(first+step*(count-1)+width),
        (long)first,(long)step,(long)width] UTF8String]);
}
int64_t *MeshIndexSetRuns(isl_set *set,size_t *count) { @autoreleasepool {
    *count=0;
    isl_ctx *ctx=isl_set_get_ctx(set);isl_set *remaining=isl_set_copy(set);
    NSMutableData *runs=[NSMutableData data];
    while(isl_set_is_empty(remaining)==isl_bool_false) {
        NSInteger first=SetEdge(remaining,NO),end=SetEdge(remaining,YES)+1;
        isl_set *holes=isl_set_subtract(SetInterval(ctx,first,end),isl_set_copy(remaining));
        NSInteger width=(isl_set_is_empty(holes)==isl_bool_true?end:SetEdge(holes,NO))-first;
        isl_set_free(holes);
        NSInteger step=0,repeats=1;
        isl_set *after=isl_set_subtract(isl_set_copy(remaining),SetInterval(ctx,first,first+width));
        if(isl_set_is_empty(after)==isl_bool_false) {
            step=SetEdge(after,NO)-first;repeats=(end-first-width)/step+1;
            isl_set *candidate=SetRun(ctx,first,width,step,repeats);
            isl_set *prefix=isl_set_intersect(isl_set_copy(remaining),SetInterval(ctx,first,first+(repeats-1)*step+width));
            isl_set *difference=isl_set_union(isl_set_subtract(isl_set_copy(candidate),isl_set_copy(prefix)),
                isl_set_subtract(prefix,candidate));
            if(isl_set_is_empty(difference)==isl_bool_false)repeats=MAX(1,(SetEdge(difference,NO)-first)/step);
            isl_set_free(difference);
        }
        isl_set_free(after);if(repeats==1)step=0;
        int64_t run[4]={first,width,step,repeats};[runs appendBytes:run length:sizeof run];
        remaining=isl_set_subtract(remaining,SetRun(ctx,first,width,step,repeats));
    }
    isl_set_free(remaining);if(isl_ctx_last_error(ctx)!=isl_error_none)return NULL;
    int64_t *result=mesh_isl_malloc(runs.length);if(!result)return NULL;
    *count=runs.length/(4*sizeof(int64_t));if(runs.length)memcpy(result,runs.bytes,runs.length);return result;
}}
MetalTensorLayout *MeshIndexSelection(isl_set *set) {
    size_t count=0;int64_t *runs=MeshIndexSetRuns(set,&count);NSMutableArray *values=[NSMutableArray array];
    if(!runs)[NSException raise:NSInvalidArgumentException format:@"index selection could not be lowered"];
    for(size_t k=0;k<count;k++)[values addObject:@[@(runs[4*k]),@(runs[4*k+1]),@(runs[4*k+2]),@(runs[4*k+3])]];
    mesh_isl_free(runs);return [MetalTensorLayout runs:values];
}
static void IndexCell(NSMutableArray *cells,isl_set *set,NSArray *members) {
    if(isl_set_is_empty(set)==isl_bool_true){isl_set_free(set);return;}
    [cells addObject:@[[NSValue valueWithPointer:set],members]];
}
struct MeshIndexCell *MeshIndexSetCover(isl_set *const *domains,size_t count,size_t *length) { @autoreleasepool {
    NSMutableArray *cells=[NSMutableArray array];
    for(NSUInteger k=0;k<count;k++) {
        isl_set *domain=isl_set_copy(domains[k]),*remaining=isl_set_copy(domain);
        NSMutableArray *next=[NSMutableArray array];
        for(NSArray *cell in cells) {
            isl_set *prior=[cell[0] pointerValue];
            IndexCell(next,isl_set_intersect(isl_set_copy(prior),isl_set_copy(domain)),[cell[1] arrayByAddingObject:@(k)]);
            IndexCell(next,isl_set_subtract(isl_set_copy(prior),isl_set_copy(domain)),cell[1]);
            remaining=isl_set_subtract(remaining,prior);
        }
        IndexCell(next,remaining,@[@(k)]);isl_set_free(domain);cells=next;
    }
    *length=cells.count;struct MeshIndexCell *result=mesh_isl_calloc(cells.count,sizeof *result);
    for(NSUInteger k=0;k<cells.count;k++) {
        NSArray *cell=cells[k],*members=cell[1];
        result[k].domain=[cell[0] pointerValue];result[k].count=members.count;
        result[k].members=mesh_isl_malloc(members.count*sizeof *result[k].members);
        for(NSUInteger m=0;m<members.count;m++)result[k].members[m]=[members[m] unsignedIntegerValue];
    }
    return result;
}}
NSArray<NSDictionary *> *MetalTensorCover(NSArray<MetalTensorLayout *> *domains) {
    isl_ctx *ctx=isl_ctx_alloc();isl_set *sets[domains.count];
    for(NSUInteger k=0;k<domains.count;k++)sets[k]=isl_map_range(MeshIndexRelation(ctx,domains[k]));
    size_t count=0;struct MeshIndexCell *cells=MeshIndexSetCover(sets,domains.count,&count);
    for(NSUInteger k=0;k<domains.count;k++)isl_set_free(sets[k]);
    NSMutableArray *result=[NSMutableArray array];
    for(size_t k=0;k<count;k++) {
        NSMutableArray *members=[NSMutableArray array];
        for(size_t m=0;m<cells[k].count;m++)[members addObject:@(cells[k].members[m])];
        [result addObject:@{@"members":members,@"domain":MeshIndexSelection(cells[k].domain)}];
        isl_set_free(cells[k].domain);mesh_isl_free(cells[k].members);
    }
    mesh_isl_free(cells);
    isl_ctx_free(ctx);return result;
}
isl_pw_aff *MeshIndexFunction(isl_ctx *ctx,MetalTensorLayout *layout) {
    isl_pw_aff *value=IndexLeaf(ctx,@(layout.offset),0,layout.count);
    NSArray *coordinates=layout.coordinates;
    for(NSUInteger d=0;d<coordinates.count;d++) {
        isl_pw_aff *part=MeshIndexExpression(ctx,coordinates[d],0,layout.count);
        if(layout.maps[@(d)])
            part=isl_pw_aff_pullback_pw_multi_aff(IndexTable(ctx,layout.maps[@(d)]),isl_pw_multi_aff_from_pw_aff(part));
        else part=isl_pw_aff_scale_val(part,isl_val_int_from_si(ctx,[layout.strides[d] integerValue]));
        value=isl_pw_aff_add(value,part);
    }
    return isl_pw_aff_coalesce(value);
}
isl_map *MeshIndexRelation(isl_ctx *ctx,MetalTensorLayout *layout) {
    return isl_map_from_pw_aff(MeshIndexFunction(ctx,layout));
}

MetalTensorLayout *MeshIndexUnion(MetalTensorLayout *sourceRoot,MetalTensorLayout *targetRoot,
    MetalTensorLayout *source,MetalTensorLayout *target,MetalTensorLayout *member,BOOL sourceMember) {
    isl_ctx *ctx=isl_ctx_alloc();
    isl_map *sourceInverse=NULL,*targetInverse=NULL,*targetMap=NULL,*relation=NULL;
    isl_set *remaining=NULL;MetalTensorLayout *result=nil;
    @try {
        sourceInverse=isl_map_reverse(MeshIndexRelation(ctx,sourceRoot));
        targetInverse=isl_map_reverse(MeshIndexRelation(ctx,targetRoot));
        isl_map *read=isl_map_apply_range(MeshIndexRelation(ctx,source),isl_map_copy(sourceInverse));
        isl_map *write=isl_map_apply_range(MeshIndexRelation(ctx,target),isl_map_copy(targetInverse));
        targetMap=isl_map_apply_range(isl_map_reverse(write),read);
        remaining=isl_set_subtract(IndexDomain(ctx,0,targetRoot.count),isl_map_domain(isl_map_copy(targetMap)));
        MetalTensorLayout *extra=MeshIndexSelection(remaining);
        const NSUInteger total=sourceRoot.count+extra.count;
        if(!member)result=[MetalTensorLayout shape:@[@(total)] strides:@[@1] offset:0 maps:@{} indices:nil];
        else if(!member.count)result=[MetalTensorLayout shape:member.shape index:@0 parameters:@{}];
        else {
            if(!sourceMember) {
                isl_map *append=isl_map_from_pw_aff(MeshIndexExpression(ctx,@[@"add",@"i",@(sourceRoot.count)],0,extra.count));
                append=isl_map_apply_range(isl_map_reverse(MeshIndexRelation(ctx,extra)),append);
                targetMap=isl_map_coalesce(isl_map_union(targetMap,append));
            }
            relation=isl_map_apply_range(MeshIndexRelation(ctx,member),isl_map_copy(sourceMember?sourceInverse:targetInverse));
            if(!sourceMember)relation=isl_map_apply_range(relation,isl_map_copy(targetMap));
            isl_pw_multi_aff *function=isl_pw_multi_aff_from_map(relation);relation=NULL;
            isl_pw_aff *index=isl_pw_multi_aff_get_pw_aff(function,0);isl_pw_multi_aff_free(function);
            result=[MetalTensorLayout shape:member.shape index:MeshIndexTerms(index) parameters:@{}];
        }
        if(isl_ctx_last_error(ctx)!=isl_error_none)
            [NSException raise:NSInvalidArgumentException format:@"view union: %s",isl_ctx_last_error_msg(ctx)];
    } @finally {
        isl_map_free(sourceInverse);isl_map_free(targetInverse);isl_map_free(targetMap);isl_map_free(relation);
        isl_set_free(remaining);isl_ctx_free(ctx);
    }
    return result;
}

NSArray<MetalTensorLayout *> *MetalTensorAssignments(NSUInteger count,NSArray<MetalTensorLayout *> *uses,
    NSArray<MetalTensorLayout *> *targets) {
    NSCParameterAssert(uses.count==targets.count);
    isl_ctx *ctx=isl_ctx_alloc();isl_map *assigned=NULL;isl_set *covered=NULL,*missing=NULL;
    NSArray *result=nil;
    @try {
        for(NSUInteger i=0;i<uses.count;i++) {
            NSCParameterAssert(uses[i].count==targets[i].count);
            isl_map *part=isl_map_apply_range(isl_map_reverse(MeshIndexRelation(ctx,uses[i])),MeshIndexRelation(ctx,targets[i]));
            assigned=assigned?isl_map_union(assigned,part):part;
        }
        if(assigned&&isl_map_is_single_valued(assigned)!=isl_bool_true)
            [NSException raise:NSInvalidArgumentException format:@"one logical position has incompatible fixed page assignments"];
        covered=assigned?isl_map_domain(isl_map_copy(assigned)):IndexDomain(ctx,0,0);
        missing=isl_set_subtract(IndexDomain(ctx,0,count),isl_set_copy(covered));
        MetalTensorLayout *positions=MeshIndexSelection(covered),*rest=MeshIndexSelection(missing),*physical;
        if(positions.count) {
            isl_map *mapping=isl_map_apply_range(MeshIndexRelation(ctx,positions),isl_map_copy(assigned));
            isl_pw_multi_aff *function=isl_pw_multi_aff_from_map(mapping);
            isl_pw_aff *index=isl_pw_multi_aff_get_pw_aff(function,0);isl_pw_multi_aff_free(function);
            physical=[MetalTensorLayout shape:positions.shape index:MeshIndexTerms(index) parameters:@{}];
        } else physical=[MetalTensorLayout shape:@[@0] strides:@[@1] offset:0 maps:@{} indices:nil];
        result=@[positions,physical,rest];
    } @finally { isl_map_free(assigned);isl_set_free(covered);isl_set_free(missing);isl_ctx_free(ctx); }
    return result;
}

static isl_map *IndexFeasible(isl_map *relation,isl_map *forbidden) {
    if(!forbidden||isl_map_is_disjoint(relation,forbidden)==isl_bool_true)return relation;
    isl_map_free(relation);return NULL;
}

// Symmetric relations close by finite composition. Directed recurrences use
// ISL's symbolic acceleration, which retains compact paths for large domains.
// Normalize existential locals before acceleration and each composition. If
// acceleration is inexact, the declared vertex bound still supplies every
// simple path and cycle. The caller's forbidden edges make contradictions
// decisive as soon as they follow from these exact compositions.
isl_map *MeshIndexClosure(isl_map *relation,NSUInteger count,enum MeshIndexClosureKind kind,isl_map *forbidden) {
    relation=IndexFeasible(relation,forbidden);
    if(!relation)return NULL;
    relation=isl_map_coalesce(isl_map_compute_divs(relation));
    if(kind==MeshIndexDirectedClosure) {
        isl_bool exact=isl_bool_false;
        isl_map *closed=isl_map_transitive_closure(isl_map_copy(relation),&exact);
        if(!closed||exact==isl_bool_true){isl_map_free(relation);return IndexFeasible(closed,forbidden);}
        isl_map_free(closed);
    }
    for(NSUInteger length=1;relation&&length<count;) {
        isl_map *closed=IndexFeasible(isl_map_union(isl_map_copy(relation),
            isl_map_apply_range(isl_map_copy(relation),isl_map_copy(relation))),forbidden);
        if(!closed){isl_map_free(relation);return NULL;}
        closed=isl_map_coalesce(isl_map_detect_equalities(isl_map_compute_divs(closed)));
        // The union already includes relation; only the converse inclusion
        // remains to establish a fixed point.
        isl_bool same=isl_map_is_subset(closed,relation);isl_map_free(relation);relation=closed;
        if(same==isl_bool_error){isl_map_free(relation);return NULL;}
        if(same==isl_bool_true)break;
        length=length>count/2?count:length*2;
    }
    return relation;
}
NSArray<MetalTensorLayout *> *MetalTensorIdentify(NSUInteger count,MetalTensorLayout *source,MetalTensorLayout *target) {
    NSCParameterAssert([source.shape isEqual:target.shape]);
    isl_ctx *ctx=isl_ctx_alloc();isl_map *relation=NULL;isl_set *representatives=NULL;NSArray *result=nil;
    @try {
        MetalTensorLayout *domain=[MetalTensorLayout shape:@[@(count)] strides:@[@1] offset:0 maps:@{} indices:nil];
        isl_map *links=isl_map_apply_range(isl_map_reverse(MeshIndexRelation(ctx,source)),MeshIndexRelation(ctx,target));
        relation=isl_map_union(MeshIndexRelation(ctx,domain),isl_map_union(isl_map_copy(links),isl_map_reverse(links)));
        relation=MeshIndexClosure(relation,count,MeshIndexSymmetricClosure,NULL);
        relation=isl_map_lexmin(relation);representatives=isl_map_range(isl_map_copy(relation));
        MetalTensorLayout *selected=MeshIndexSelection(representatives);
        MetalTensorLayout *compact=[MetalTensorLayout shape:selected.shape strides:@[@1] offset:0 maps:@{} indices:nil];
        isl_map *numbered=isl_map_apply_range(isl_map_reverse(MeshIndexRelation(ctx,selected)),MeshIndexRelation(ctx,compact));
        isl_pw_multi_aff *function=isl_pw_multi_aff_from_map(isl_map_apply_range(isl_map_copy(relation),numbered));
        isl_pw_aff *index=isl_pw_multi_aff_get_pw_aff(function,0);isl_pw_multi_aff_free(function);
        MetalTensorLayout *mapped=count?[MetalTensorLayout shape:domain.shape index:MeshIndexTerms(index) parameters:@{}]:domain;
        if(!count)isl_pw_aff_free(index);
        result=@[compact,mapped];
    } @finally {isl_map_free(relation);isl_set_free(representatives);isl_ctx_free(ctx);}
    return result;
}

static id IndexTerms(isl_ast_expr *expression) {
    enum isl_ast_expr_type type=isl_ast_expr_get_type(expression);
    if(type==isl_ast_expr_int) {
        isl_val *v=isl_ast_expr_int_get_val(expression);NSNumber *value=@(isl_val_get_num_si(v));isl_val_free(v);return value;
    }
    if(type==isl_ast_expr_id) {
        isl_id *name=isl_ast_expr_id_get_id(expression);NSString *value=@(isl_id_get_name(name));isl_id_free(name);return value;
    }
    enum isl_ast_expr_op_type op=isl_ast_expr_op_get_type(expression);
    NSMutableArray *args=[NSMutableArray array];
    for(int k=0;k<isl_ast_expr_op_get_n_arg(expression);k++) {
        isl_ast_expr *arg=isl_ast_expr_op_get_arg(expression,k);[args addObject:IndexTerms(arg)];isl_ast_expr_free(arg);
    }
    NSString *name=nil;
    switch(op) {
    case isl_ast_expr_op_minus:return @[@"mul",args[0],@(-1)];
    case isl_ast_expr_op_sub:return @[@"add",args[0],@[@"mul",args[1],@(-1)]];
    case isl_ast_expr_op_mul:
        return [args[0] isKindOfClass:NSNumber.class]?@[@"mul",args[1],args[0]]:@[@"mul",args[0],args[1]];
    case isl_ast_expr_op_div:case isl_ast_expr_op_fdiv_q:case isl_ast_expr_op_pdiv_q:name=@"div";break;
    case isl_ast_expr_op_pdiv_r:case isl_ast_expr_op_zdiv_r:name=@"mod";break;
    case isl_ast_expr_op_add:name=@"add";break;
    case isl_ast_expr_op_min:name=@"min";break;case isl_ast_expr_op_max:name=@"max";break;
    case isl_ast_expr_op_and:case isl_ast_expr_op_and_then:name=@"and";break;
    case isl_ast_expr_op_or:case isl_ast_expr_op_or_else:name=@"or";break;
    case isl_ast_expr_op_eq:name=@"eq";break;case isl_ast_expr_op_le:name=@"le";break;
    case isl_ast_expr_op_lt:name=@"lt";break;case isl_ast_expr_op_ge:name=@"ge";break;case isl_ast_expr_op_gt:name=@"gt";break;
    case isl_ast_expr_op_cond:case isl_ast_expr_op_select:name=@"select";break;
    default:[NSException raise:NSInvalidArgumentException format:@"isl index expression is not scalar arithmetic: %d",op];
    }
    [args insertObject:name atIndex:0];return args;
}
id MeshIndexTerms(isl_pw_aff *expression) {
    if(!expression)[NSException raise:NSInvalidArgumentException format:@"index relation could not be lowered"];
    isl_ctx *ctx=isl_pw_aff_get_ctx(expression);
    expression=isl_pw_aff_set_dim_id(expression,isl_dim_in,0,isl_id_alloc(ctx,"i",NULL));
    isl_ast_build *build=isl_ast_build_from_context(isl_pw_aff_domain(isl_pw_aff_copy(expression)));
    isl_ast_expr *ast=isl_ast_build_expr_from_pw_aff(build,isl_pw_aff_coalesce(expression));
    if(!ast)[NSException raise:NSInvalidArgumentException format:@"index expression emission: %s",isl_ctx_last_error_msg(ctx)];
    id terms=IndexTerms(ast);isl_ast_expr_free(ast);isl_ast_build_free(build);return terms;
}
BOOL MeshIndexEqual(id left,id right,NSUInteger count) {
    isl_ctx *ctx=isl_ctx_alloc();
    isl_pw_aff *a=MeshIndexExpression(ctx,left,0,count),*b=MeshIndexExpression(ctx,right,0,count);
    isl_bool equal=isl_pw_aff_is_equal(a,b);isl_pw_aff_free(a);isl_pw_aff_free(b);isl_ctx_free(ctx);
    if(equal==isl_bool_error)[NSException raise:NSInvalidArgumentException format:@"index relation equality could not be resolved"];
    return equal==isl_bool_true;
}
struct MeshIndexBounds MeshIndexImageBounds(isl_map *relation) {
    isl_set *image=isl_map_range(relation);
    isl_val *low=isl_set_dim_min_val(isl_set_copy(image),0),*high=isl_set_dim_max_val(image,0);
    struct MeshIndexBounds result={isl_val_get_num_si(low),isl_val_get_num_si(high)};
    isl_val_free(low);isl_val_free(high);return result;
}
struct MeshIndexBounds MeshIndexBoundsOf(id expression,NSUInteger first,NSUInteger end) {
    isl_ctx *ctx=isl_ctx_alloc();
    struct MeshIndexBounds result=MeshIndexImageBounds(isl_map_from_pw_aff(MeshIndexExpression(ctx,expression,first,end)));
    isl_ctx_free(ctx);return result;
}
struct MeshIndexRun MeshIndexRunOf(id expression,NSUInteger first,NSUInteger end) {
    struct MeshIndexBounds a=MeshIndexBoundsOf(expression,first,first+1);
    NSInteger step=end>first+1?MeshIndexBoundsOf(expression,first+1,first+2).low-a.low:0;
    isl_ctx *ctx=isl_ctx_alloc();
    isl_pw_aff *value=MeshIndexExpression(ctx,expression,first,end);
    isl_pw_aff *line=MeshIndexExpression(ctx,@[@"add",@(a.low-step*(NSInteger)first),@[@"mul",@"i",@(step)]],first,end);
    isl_set *different=isl_pw_aff_ne_set(value,line);
    if(isl_set_is_empty(different)==isl_bool_false) {
        isl_val *last=isl_set_dim_min_val(isl_set_copy(different),0);end=isl_val_get_num_si(last);isl_val_free(last);
    }
    isl_set_free(different);isl_ctx_free(ctx);
    return (struct MeshIndexRun){a.low,step,end};
}
