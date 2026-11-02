#import "mesh-index-relation.h"
#include <isl/ast_build.h>
#include <isl/id.h>
#include <isl/ilp.h>
#include <isl/val.h>
#include <isl/point.h>
#include <isl/constraint.h>
#include <isl/local_space.h>
#include <isl/space.h>
#include "mesh-isl-memory.h"
#include <string.h>
/* isl's disjoint sum (isl_aff_private.h): pieces on domains already disjoint concatenated, where union_add intersects
   and subtracts every pair of pieces */
__isl_give isl_pw_aff *isl_pw_aff_add_disjoint(__isl_take isl_pw_aff *pwaff1,__isl_take isl_pw_aff *pwaff2);

isl_map *MeshIndexNormalize(isl_map *relation) {
    return isl_map_coalesce(isl_map_detect_equalities(isl_map_compute_divs(relation)));
}

isl_map *MeshIndexCompose(isl_map *source,isl_map *target) {
    const isl_bool function=isl_map_is_single_valued(target);
    if(function==isl_bool_true&&isl_map_is_single_valued(source)==isl_bool_true) {
        isl_pw_multi_aff *first=isl_pw_multi_aff_from_map(source),*second=isl_pw_multi_aff_from_map(target);
        return isl_map_from_pw_multi_aff(isl_pw_multi_aff_pullback_pw_multi_aff(second,first));
    }
    /* A relation through an injective function (an access's bytes through its operand's pages): the preimage of the
       relation's range by the function's inverse, { x -> p : (x -> inverse(p)) in source }, the same map with no
       intermediate to eliminate. Changed by the LSTM demonstration session: apply_range left the operand's bytes
       existential and compute_divs eliminated them, a parametric integer program over the layout's flattened
       extents: 4.7 s for a 242-unit shard's transposed weight read and over 35 s at 241 units, against 0.6 s and 1.8 s
       here, the maps equal (scratch isl timing of d-r10's and d-r19's pairs, 2026-11-01). What must stay: the
       relation composed exactly; a non-injective target still goes through apply_range. */
    if(function==isl_bool_true&&isl_map_is_injective(target)==isl_bool_true)
        return MeshIndexNormalize(isl_map_preimage_range_pw_multi_aff(source,isl_pw_multi_aff_from_map(isl_map_reverse(target))));
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

/* The index domain, its leaves and runs built through isl's constructors, not read from text. Changed by the layout
   algebra session: the chain service's preparation spent its whole sample parsing these, isl_pw_aff_read_from_str
   for every leaf of every page selection's runs (2026-11-01). What must stay: the same sets and functions, on the
   one-dimensional space [i]. */
static isl_val *IndexValue(isl_ctx *ctx,NSInteger value) { return isl_val_int_from_si(ctx,(long)value); }
static isl_set *IndexInterval(isl_ctx *ctx,NSInteger first,NSInteger end) {
    isl_set *set=isl_set_universe(isl_space_set_alloc(ctx,0,1));
    set=isl_set_lower_bound_val(set,isl_dim_set,0,IndexValue(ctx,first));
    return isl_set_upper_bound_val(set,isl_dim_set,0,IndexValue(ctx,end-1));
}
static isl_set *IndexDomain(isl_ctx *ctx, NSUInteger first, NSUInteger end) {
    return IndexInterval(ctx,(NSInteger)first,(NSInteger)end);
}
static isl_pw_aff *IndexLeaf(isl_ctx *ctx, NSInteger value, NSUInteger first, NSUInteger end) {
    return isl_pw_aff_val_on_domain(IndexDomain(ctx,first,end),IndexValue(ctx,value));
}
// Compile each branch on the positions where its condition holds. Index
// terms are pure values; constructing an unreachable arm cannot add meaning,
// and rebuilding every arm over the full domain duplicates relation work.
static isl_pw_aff *IndexExpression(isl_ctx *ctx,id expression,isl_set *domain) {
    if([expression isKindOfClass:NSNumber.class])
        return isl_pw_aff_val_on_domain(isl_set_copy(domain),IndexValue(ctx,[expression integerValue]));
    if([expression isEqual:@"i"])
        return isl_pw_aff_intersect_domain(isl_pw_aff_var_on_domain(
            isl_local_space_from_space(isl_set_get_space(domain)),isl_dim_set,0),isl_set_copy(domain));
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
        return isl_pw_aff_add_disjoint(left,right);
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
    isl_pw_aff *result=IndexLeaf(ctx,0,0,0);
    for(NSUInteger first=0;first<count;) {
        NSUInteger end=first+1;NSInteger step=end<count?values[end]-values[first]:0;
        while(end<count&&values[end]==values[first]+(NSInteger)(end-first)*step)end++;
        isl_pw_aff *run=MeshIndexExpression(ctx,@[@"add",@(values[first]-(NSInteger)first*step),@[@"mul",@"i",@(step)]],first,end);
        result=isl_pw_aff_add_disjoint(result,run);first=end;
    }
    return isl_pw_aff_coalesce(result);
}
/* each run (first, width, step, repeats) one piece on its positions [begin, begin + width repeats): first + local +
   (step - width) floor(local / width), local = i - begin, the floor only where the run repeats at a step other than its
   width */
isl_pw_aff *MeshIndexRuns(isl_ctx *ctx,NSArray<NSArray<NSNumber *> *> *runs) {
    isl_pw_aff *result=IndexLeaf(ctx,0,0,0);NSUInteger begin=0;
    isl_local_space *space=isl_local_space_from_space(isl_space_set_alloc(ctx,0,1));
    for(NSArray *run in runs) {
        const NSInteger first=[run[0] integerValue],width=[run[1] integerValue],step=[run[2] integerValue];
        const NSUInteger count=(NSUInteger)width*[run[3] unsignedIntegerValue];
        if(!count)continue;
        isl_aff *local=isl_aff_add_constant_val(isl_aff_var_on_domain(isl_local_space_copy(space),isl_dim_set,0),
            IndexValue(ctx,-(NSInteger)begin));
        isl_aff *value=isl_aff_add_constant_val(isl_aff_copy(local),IndexValue(ctx,first));
        if(count>(NSUInteger)width&&step!=width)
            value=isl_aff_add(value,isl_aff_scale_val(isl_aff_floor(isl_aff_scale_down_val(local,IndexValue(ctx,width))),
                IndexValue(ctx,step-width)));
        else isl_aff_free(local);
        result=isl_pw_aff_add_disjoint(result,isl_pw_aff_alloc(IndexDomain(ctx,begin,begin+count),value));begin+=count;
    }
    isl_local_space_free(space);
    return isl_pw_aff_coalesce(result);
}
static isl_set *SetInterval(isl_ctx *ctx,NSInteger first,NSInteger end) { return IndexInterval(ctx,first,end); }
static NSInteger SetEdge(isl_set *set,BOOL maximum) {
    isl_val *value=maximum?isl_set_dim_max_val(isl_set_copy(set),0):isl_set_dim_min_val(isl_set_copy(set),0);
    NSInteger result=isl_val_get_num_si(value);isl_val_free(value);return result;
}
static isl_set *SetRun(isl_ctx *ctx,NSInteger first,NSInteger width,NSInteger step,NSInteger count) {
    isl_set *interval=SetInterval(ctx,first,first+step*(count-1)+width);
    if(count==1)return interval;
    isl_pw_aff *phase=isl_pw_aff_mod_val(isl_pw_aff_from_aff(isl_aff_add_constant_val(isl_aff_var_on_domain(
        isl_local_space_from_space(isl_set_get_space(interval)),isl_dim_set,0),IndexValue(ctx,-first))),IndexValue(ctx,step));
    isl_pw_aff *bound=isl_pw_aff_val_on_domain(isl_set_universe(isl_set_get_space(interval)),IndexValue(ctx,width));
    return isl_set_intersect(isl_pw_aff_lt_set(phase,bound),interval);
}
static isl_set *SetAbsorbIntervals(isl_set *set) {
    // In one dimension a basic set without local divisions is an interval.
    // Such an explicit component can cover projected components without
    // reconstructing their existential divisions just to take a complement.
    isl_basic_set_list *parts=isl_set_get_basic_set_list(set);
    isl_size count=isl_basic_set_list_size(parts);
    isl_set *intervals=isl_set_empty(isl_set_get_space(set));
    for(isl_size i=0;i<count;i++) {
        isl_basic_set *part=isl_basic_set_list_get_at(parts,i);
        if(isl_basic_set_dim(part,isl_dim_div)==0)
            intervals=isl_set_union(intervals,isl_set_from_basic_set(part));
        else isl_basic_set_free(part);
    }
    intervals=isl_set_coalesce(intervals);
    if(isl_set_is_empty(intervals)!=isl_bool_false) {
        isl_set_free(intervals);isl_basic_set_list_free(parts);return set;
    }
    isl_set *result=isl_set_copy(intervals);
    for(isl_size i=0;i<count;i++) {
        isl_basic_set *basic=isl_basic_set_list_get_at(parts,i);
        if(isl_basic_set_dim(basic,isl_dim_div)==0){isl_basic_set_free(basic);continue;}
        isl_set *part=isl_set_from_basic_set(basic);
        // ISL computes local divisions only for the subtracted right side.
        // Testing this direction leaves the projected component existential.
        isl_bool covered=isl_set_is_subset(part,intervals);
        if(covered==isl_bool_true)isl_set_free(part);
        else result=isl_set_union(result,part);
    }
    isl_basic_set_list_free(parts);isl_set_free(intervals);isl_set_free(set);return result;
}
/* A one-dimensional basic set without divisions is an interval, read from its constraints (a x + k >= 0 and the
   equalities). */
struct SetBounds { int64_t low,high; BOOL lower,upper,empty,general; };
static isl_stat SetBound(isl_constraint *constraint,void *user) {
    struct SetBounds *b=user;
    isl_val *a=isl_constraint_get_coefficient_val(constraint,isl_dim_set,0),*k=isl_constraint_get_constant_val(constraint);
    if(!isl_val_is_int(a)||!isl_val_is_int(k))b->general=YES;
    else {
        const int64_t x=isl_val_get_num_si(a),c=isl_val_get_num_si(k);
        if(isl_constraint_is_equality(constraint)) {
            if(!x){if(c)b->empty=YES;}
            else if(c%x)b->empty=YES;
            else {const int64_t v=-c/x;b->low=b->lower?MAX(b->low,v):v;b->high=b->upper?MIN(b->high,v):v;b->lower=b->upper=YES;}
        } else if(x>0) {
            const int64_t v=(-c>=0?(-c+x-1)/x:-(c/x));b->low=b->lower?MAX(b->low,v):v;b->lower=YES;
        } else if(x<0) {
            const int64_t y=-x,v=(c>=0?c/y:-((-c+y-1)/y));b->high=b->upper?MIN(b->high,v):v;b->upper=YES;
        } else if(c<0)b->empty=YES;
    }
    isl_val_free(a);isl_val_free(k);isl_constraint_free(constraint);return isl_stat_ok;
}
static int SetIntervalOrder(const void *a,const void *b) { const int64_t *x=a,*y=b;return x[0]<y[0]?-1:x[0]>y[0]; }
/* the set's elements as sorted, merged intervals [low, high] into `out` where every basic set is division-free (read
   from its constraints) or where the set spans at most SET_POINT_SPAN positions (its points enumerated into a bitmap):
   no set is built or subtracted, so no division is eliminated. NO otherwise. */
#define SET_POINT_SPAN (UINT64_C(1)<<22)
static isl_stat SetPoint(isl_point *point,void *user) {
    uint64_t **bits=user;isl_val *v=isl_point_get_coordinate_val(point,isl_dim_set,0);
    const int64_t at=isl_val_get_num_si(v)-(int64_t)bits[1][0];isl_val_free(v);isl_point_free(point);
    bits[0][at/64]|=UINT64_C(1)<<(at%64);return isl_stat_ok;
}
static BOOL SetIntervals(isl_set *set,NSMutableData *out) {
    if(isl_set_dim(set,isl_dim_set)!=1||isl_set_dim(set,isl_dim_param)!=0)return NO;
    isl_basic_set_list *parts=isl_set_get_basic_set_list(set);
    const isl_size count=isl_basic_set_list_size(parts);
    NSMutableData *raw=[NSMutableData data];BOOL plain=YES;
    for(isl_size i=0;plain&&i<count;i++) {
        isl_basic_set *part=isl_basic_set_list_get_at(parts,i);struct SetBounds b={0};
        if(isl_basic_set_dim(part,isl_dim_div)!=0)plain=NO;
        else if(isl_basic_set_foreach_constraint(part,SetBound,&b)<0||b.general||(!b.empty&&(!b.lower||!b.upper)))plain=NO;
        else if(!b.empty&&b.low<=b.high){const int64_t iv[2]={b.low,b.high};[raw appendBytes:iv length:sizeof iv];}
        isl_basic_set_free(part);
    }
    isl_basic_set_list_free(parts);
    if(!plain) {
        [raw setLength:0];
        if(isl_set_is_empty(set)==isl_bool_true)return YES;
        isl_val *lo=isl_set_dim_min_val(isl_set_copy(set),0),*hi=isl_set_dim_max_val(isl_set_copy(set),0);
        const BOOL bounded=isl_val_is_int(lo)&&isl_val_is_int(hi);
        const int64_t low=bounded?isl_val_get_num_si(lo):0,high=bounded?isl_val_get_num_si(hi):-1;
        isl_val_free(lo);isl_val_free(hi);
        if(!bounded||high<low||(uint64_t)(high-low)+1>SET_POINT_SPAN)return NO;
        const uint64_t span=(uint64_t)(high-low)+1,words=(span+63)/64;
        uint64_t *map=calloc(words,sizeof *map),origin=(uint64_t)low,*bits[2]={map,&origin};
        if(!map||isl_set_foreach_point(set,SetPoint,bits)<0){free(map);return NO;}
        for(uint64_t at=0;at<span;) {
            if(!(map[at/64]>>(at%64)&1)){at++;continue;}
            uint64_t end=at;while(end+1<span&&(map[(end+1)/64]>>((end+1)%64)&1))end++;
            const int64_t iv[2]={low+(int64_t)at,low+(int64_t)end};[raw appendBytes:iv length:sizeof iv];at=end+1;
        }
        free(map);[out appendData:raw];return YES;
    }
    int64_t *iv=raw.mutableBytes;const size_t n=raw.length/(2*sizeof(int64_t));
    qsort(iv,n,2*sizeof(int64_t),SetIntervalOrder);
    for(size_t k=0;k<n;k++) {
        int64_t *last=out.length?(int64_t *)out.mutableBytes+out.length/sizeof(int64_t)-2:NULL;
        if(last&&iv[2*k]<=last[1]+1){if(iv[2*k+1]>last[1])last[1]=iv[2*k+1];}
        else [out appendBytes:iv+2*k length:2*sizeof(int64_t)];
    }
    return YES;
}
/* sorted disjoint maximal intervals as (first, width, step, repeats) runs: each run an interval and the next ones of
   its width at its step, the peel's own form */
static void SetRunsOfIntervals(const int64_t *iv,size_t n,NSMutableData *runs) {
    for(size_t k=0;k<n;) {
        const int64_t first=iv[2*k],width=iv[2*k+1]-iv[2*k]+1;
        int64_t step=k+1<n?iv[2*(k+1)]-first:0;size_t m=k+1;
        while(m<n&&iv[2*m]==first+(int64_t)(m-k)*step&&iv[2*m+1]-iv[2*m]+1==width)m++;
        const int64_t repeats=(int64_t)(m-k);if(repeats==1)step=0;
        const int64_t run[4]={first,width,step,repeats};[runs appendBytes:run length:sizeof run];k=m;
    }
}

int64_t *MeshIndexSetRuns(isl_set *set,size_t *count) { @autoreleasepool {
    *count=0;
    // Projection can leave many overlapping descriptions of the same bytes.
    // Normalize that set once before repeatedly asking for its next run.
    isl_ctx *ctx=isl_set_get_ctx(set);
    isl_set *remaining=isl_set_remove_redundancies(isl_set_copy(set));
    remaining=SetAbsorbIntervals(remaining);
    remaining=isl_set_coalesce(isl_set_detect_equalities(remaining));
    NSMutableData *runs=[NSMutableData data];
    /* its intervals read whole where they can be (SetIntervals), the runs formed from them; the peel below subtracts each
       run from what remains and eliminates the set's divisions each time, past 60 s for the LSTM's h stored units first
       in placement's layout union (2026-11-01). Changed by the LSTM demonstration session. What must stay: the same
       positions, as runs. */
    NSMutableData *intervals=[NSMutableData data];
    if(SetIntervals(remaining,intervals)) {
        SetRunsOfIntervals(intervals.bytes,intervals.length/(2*sizeof(int64_t)),runs);
        isl_set_free(remaining);
        int64_t *result=mesh_isl_malloc(runs.length?runs.length:1);if(!result)return NULL;
        *count=runs.length/(4*sizeof(int64_t));if(runs.length)memcpy(result,runs.bytes,runs.length);return result;
    }
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
    isl_pw_aff *value=IndexLeaf(ctx,layout.offset,0,layout.count);
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

/* mesh-index-relation.h MeshIndexInverse. The layout's position is offset + sum_d x_d stride_d over the mixed-radix
   digits x of the flat position i = sum_d x_d L_d (L_d the product of the later extents). With the varying axes
   ascending by stride and each stride at least the one before times its extent, the digits are recovered from the
   storage position s by its strides alone: x_k = floor(((s - offset) mod stride_{k+1}) / stride_k), the last axis's
   floor((s - offset) / stride). Added by the LSTM demonstration session: an access composed with its operand's pages
   had isl invert the flattened forward map by a parametric integer program, 1.8 s to 7 s a composition on the 241- and
   58-unit shards (2026-11-01); this inverse is written down, its divisions by the storage strides (powers of two for
   region tensors: a shift and a mask). */
isl_map *MeshIndexInverse(isl_ctx *ctx,MetalTensorLayout *layout) {
    if(!layout.factored||layout.maps.count||!layout.count)return NULL;
    NSArray<NSNumber *> *shape=layout.indexShape,*strides=layout.strides;
    const NSUInteger rank=shape.count;
    NSInteger weight[rank>0?rank:1];NSInteger running=1;
    for(NSUInteger d=rank;d-->0;){weight[d]=running;running*=[shape[d] integerValue];}
    NSMutableArray<NSNumber *> *axes=[NSMutableArray array];
    for(NSUInteger d=0;d<rank;d++)if([shape[d] integerValue]>1) {
        if([strides[d] integerValue]<=0)return NULL;
        [axes addObject:@(d)];
    }
    [axes sortUsingComparator:^NSComparisonResult(NSNumber *a,NSNumber *b){
        const NSInteger x=[strides[a.unsignedIntegerValue] integerValue],y=[strides[b.unsignedIntegerValue] integerValue];
        return x<y?NSOrderedAscending:x>y?NSOrderedDescending:NSOrderedSame;
    }];
    for(NSUInteger k=0;k+1<axes.count;k++) {
        const NSUInteger d=axes[k].unsignedIntegerValue,e=axes[k+1].unsignedIntegerValue;
        if([strides[e] integerValue]<[strides[d] integerValue]*[shape[d] integerValue])return NULL;
    }
    const NSInteger offset=layout.offset;NSInteger last=0;
    NSMutableString *flat=[NSMutableString stringWithString:@"0"],*domain=[NSMutableString string];
    NSString *relative=[NSString stringWithFormat:@"(s - %ld)",(long)offset];
    for(NSUInteger k=0;k<axes.count;k++) {
        const NSUInteger d=axes[k].unsignedIntegerValue;const NSInteger stride=[strides[d] integerValue],extent=[shape[d] integerValue];
        last+=stride*(extent-1);
        NSString *digit=k+1<axes.count
            ?[NSString stringWithFormat:@"floor((%@ mod %ld)/%ld)",relative,(long)[strides[axes[k+1].unsignedIntegerValue] integerValue],(long)stride]
            :[NSString stringWithFormat:@"floor(%@/%ld)",relative,(long)stride];
        [flat appendFormat:@" + %ld*%@",(long)weight[d],digit];
        if(k+1<axes.count)[domain appendFormat:@" and %@ mod %ld < %ld",relative,
            (long)[strides[axes[k+1].unsignedIntegerValue] integerValue],(long)(stride*extent)];
    }
    if(axes.count)[domain appendFormat:@" and %@ mod %ld = 0",relative,(long)[strides[axes[0].unsignedIntegerValue] integerValue]];
    NSString *text=[NSString stringWithFormat:@"{ [s] -> [i] : %ld <= s <= %ld and i = %@%@ }",(long)offset,(long)(offset+last),flat,domain];
    return isl_map_read_from_str(ctx,text.UTF8String);
}

/* mesh-index-relation.h MeshIndexPreimage */
isl_map *MeshIndexPreimage(isl_map *source,isl_map *inverse) {
    return MeshIndexNormalize(isl_map_preimage_range_pw_multi_aff(source,isl_pw_multi_aff_from_map(inverse)));
}

// Number the equivalence classes of a finite ordinal domain. Representative
// order is the original domain order; every equality participates before a
// representative is selected, including many-to-one view relations.
static isl_map *IndexQuotient(isl_ctx *ctx,NSUInteger count,isl_map *links,NSUInteger *size) {
    isl_map *relation=NULL,*result=NULL;isl_set *representatives=NULL;
    @try {
        relation=isl_map_union(isl_set_identity(IndexDomain(ctx,0,count)),
            isl_map_union(isl_map_copy(links),isl_map_reverse(links)));
        relation=MeshIndexClosure(relation,count,MeshIndexSymmetricClosure,NULL);
        relation=isl_map_lexmin(relation);representatives=isl_map_range(isl_map_copy(relation));
        MetalTensorLayout *selected=MeshIndexSelection(representatives);*size=selected.count;
        isl_map *numbered=isl_map_reverse(MeshIndexRelation(ctx,selected));
        result=isl_map_apply_range(isl_map_copy(relation),numbered);
    } @finally {isl_map_free(relation);isl_set_free(representatives);}
    return result;
}

MetalTensorLayout *MeshIndexUnion(MetalTensorLayout *sourceRoot,MetalTensorLayout *targetRoot,
    MetalTensorLayout *source,MetalTensorLayout *target,MetalTensorLayout *member,BOOL sourceMember) {
    isl_ctx *ctx=isl_ctx_alloc();
    isl_map *sourceInverse=NULL,*targetInverse=NULL,*quotient=NULL,*relation=NULL;
    MetalTensorLayout *result=nil;
    @try {
        isl_map *sourceRelation=MeshIndexRelation(ctx,sourceRoot);
        isl_map *shift=isl_map_from_pw_aff(MeshIndexExpression(ctx,@[@"add",@"i",@(sourceRoot.count)],0,targetRoot.count));
        isl_map *targetRelation=isl_map_apply_range(isl_map_reverse(shift),MeshIndexRelation(ctx,targetRoot));
        sourceInverse=isl_map_reverse(isl_map_copy(sourceRelation));
        targetInverse=isl_map_reverse(isl_map_copy(targetRelation));
        isl_map *read=isl_map_apply_range(MeshIndexRelation(ctx,source),isl_map_copy(sourceInverse));
        isl_map *write=isl_map_apply_range(MeshIndexRelation(ctx,target),isl_map_copy(targetInverse));
        isl_map *links=isl_map_apply_range(isl_map_reverse(read),write);
        links=isl_map_union(links,isl_map_apply_range(sourceRelation,isl_map_copy(sourceInverse)));
        links=isl_map_union(links,isl_map_apply_range(targetRelation,isl_map_copy(targetInverse)));
        NSUInteger total=0;
        quotient=IndexQuotient(ctx,sourceRoot.count+targetRoot.count,links,&total);
        if(!member)result=[MetalTensorLayout shape:@[@(total)] strides:@[@1] offset:0 maps:@{} indices:nil];
        else if(!member.count)result=[MetalTensorLayout shape:member.shape index:@0 parameters:@{}];
        else {
            relation=isl_map_apply_range(MeshIndexRelation(ctx,member),isl_map_copy(sourceMember?sourceInverse:targetInverse));
            relation=isl_map_apply_range(relation,isl_map_copy(quotient));
            isl_pw_multi_aff *function=isl_pw_multi_aff_from_map(relation);relation=NULL;
            isl_pw_aff *index=isl_pw_multi_aff_get_pw_aff(function,0);isl_pw_multi_aff_free(function);
            result=[MetalTensorLayout shape:member.shape index:MeshIndexTerms(index) parameters:@{}];
        }
        if(isl_ctx_last_error(ctx)!=isl_error_none)
            [NSException raise:NSInvalidArgumentException format:@"view union: %s",isl_ctx_last_error_msg(ctx)];
    } @finally {
        isl_map_free(sourceInverse);isl_map_free(targetInverse);isl_map_free(quotient);isl_map_free(relation);
        isl_ctx_free(ctx);
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
isl_set *MeshIndexReachable(isl_map *relation,isl_set *seeds,NSUInteger count) {
    isl_set *invariant=isl_set_from_basic_set(isl_set_affine_hull(isl_set_copy(seeds)));
    isl_set *domain=NULL,*result=NULL;
    // Hulls grow from the seeds by one-step images. Equality proves closure
    // under the transition; every actually reachable state is then in this
    // invariant, even when it expands to the full state space.
    for(;;) {
        isl_set *image=isl_set_apply(isl_set_copy(invariant),isl_map_copy(relation));
        isl_set *next=isl_set_from_basic_set(isl_set_affine_hull(isl_set_union(isl_set_copy(invariant),image)));
        isl_bool same=isl_set_is_equal(invariant,next);isl_set_free(invariant);invariant=next;
        if(same==isl_bool_error)goto finished;
        if(same==isl_bool_true)break;
    }
    relation=isl_map_intersect_domain(relation,isl_set_copy(invariant));
    domain=isl_set_union(isl_set_copy(seeds),isl_map_range(isl_map_copy(relation)));
    relation=MeshIndexNormalize(isl_map_intersect_domain(relation,isl_set_copy(domain)));
    {
        // Remove redundant case partitions only when their convex hull is
        // exactly the same integer relation, including all divisibility.
        isl_map *hull=isl_map_from_basic_map(isl_map_convex_hull(isl_map_copy(relation)));
        isl_bool same=isl_map_is_equal(relation,hull);
        if(same==isl_bool_true){isl_map_free(relation);relation=hull;}
        else isl_map_free(hull);
        if(same==isl_bool_error)goto finished;
    }
    // The seed states already supply paths of length zero. Adding identity
    // edges for every intermediate cursor would restore discarded partitions.
    relation=MeshIndexClosure(relation,count,MeshIndexDirectedClosure,NULL);
    result=isl_set_coalesce(isl_set_detect_equalities(isl_set_compute_divs(isl_set_union(
        isl_set_copy(seeds),isl_set_apply(isl_set_copy(seeds),isl_map_copy(relation))))));
finished:
    isl_set_free(invariant);isl_set_free(domain);isl_set_free(seeds);isl_map_free(relation);
    return result;
}
NSArray<MetalTensorLayout *> *MetalTensorIdentify(NSUInteger count,MetalTensorLayout *source,MetalTensorLayout *target) {
    NSCParameterAssert([source.shape isEqual:target.shape]);
    isl_ctx *ctx=isl_ctx_alloc();isl_map *relation=NULL;NSArray *result=nil;
    @try {
        MetalTensorLayout *domain=[MetalTensorLayout shape:@[@(count)] strides:@[@1] offset:0 maps:@{} indices:nil];
        isl_map *links=isl_map_apply_range(isl_map_reverse(MeshIndexRelation(ctx,source)),MeshIndexRelation(ctx,target));
        NSUInteger total=0;relation=IndexQuotient(ctx,count,links,&total);
        MetalTensorLayout *compact=[MetalTensorLayout shape:@[@(total)] strides:@[@1] offset:0 maps:@{} indices:nil];
        isl_pw_multi_aff *function=isl_pw_multi_aff_from_map(isl_map_copy(relation));
        isl_pw_aff *index=isl_pw_multi_aff_get_pw_aff(function,0);isl_pw_multi_aff_free(function);
        MetalTensorLayout *mapped=count?[MetalTensorLayout shape:domain.shape index:MeshIndexTerms(index) parameters:@{}]:domain;
        if(!count)isl_pw_aff_free(index);
        result=@[compact,mapped];
    } @finally {isl_map_free(relation);isl_ctx_free(ctx);}
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
