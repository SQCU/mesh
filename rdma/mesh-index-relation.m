#import "mesh-index-relation.h"
#include <isl/ast_build.h>
#include <isl/id.h>
#include <isl/ilp.h>
#include <isl/val.h>

static isl_set *IndexDomain(isl_ctx *ctx, NSUInteger first, NSUInteger end) {
    return isl_set_read_from_str(ctx, [[NSString stringWithFormat:@"{ [i] : %lu <= i < %lu }",
        (unsigned long)first, (unsigned long)end] UTF8String]);
}
static isl_pw_aff *IndexLeaf(isl_ctx *ctx, id value, NSUInteger first, NSUInteger end) {
    NSString *text=[NSString stringWithFormat:@"{ [i] -> [(%@)] : %lu <= i < %lu }",
        value, (unsigned long)first, (unsigned long)end];
    return isl_pw_aff_read_from_str(ctx,text.UTF8String);
}
isl_pw_aff *MeshIndexExpression(isl_ctx *ctx,id expression,NSUInteger first,NSUInteger end) {
    if([expression isKindOfClass:NSNumber.class]||[expression isEqual:@"i"])
        return IndexLeaf(ctx,expression,first,end);
    NSString *op=expression[0];
    isl_pw_aff *a=MeshIndexExpression(ctx,expression[1],first,end);
    if([op isEqual:@"mul"]||[op isEqual:@"div"]||[op isEqual:@"mod"]) {
        isl_val *n=isl_val_int_from_si(ctx,[expression[2] integerValue]);
        if([op isEqual:@"mul"])return isl_pw_aff_scale_val(a,n);
        if([op isEqual:@"mod"])return isl_pw_aff_mod_val(a,n);
        return isl_pw_aff_floor(isl_pw_aff_scale_down_val(a,n));
    }
    if([op isEqual:@"select"])
        return isl_pw_aff_cond(a,MeshIndexExpression(ctx,expression[2],first,end),
            MeshIndexExpression(ctx,expression[3],first,end));
    for(NSUInteger k=2;k<[expression count];k++) {
        isl_pw_aff *b=MeshIndexExpression(ctx,expression[k],first,end);
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
            a=isl_pw_aff_intersect_domain(isl_set_indicator_function(condition),IndexDomain(ctx,first,end));
        }
    }
    return a;
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
isl_map *MeshIndexRelation(isl_ctx *ctx,MetalTensorLayout *layout) {
    isl_pw_aff *value=IndexLeaf(ctx,@(layout.offset),0,layout.count);
    NSArray *coordinates=layout.coordinates;
    for(NSUInteger d=0;d<coordinates.count;d++) {
        isl_pw_aff *part=MeshIndexExpression(ctx,coordinates[d],0,layout.count);
        if(layout.maps[@(d)])
            part=isl_pw_aff_pullback_pw_multi_aff(IndexTable(ctx,layout.maps[@(d)]),isl_pw_multi_aff_from_pw_aff(part));
        else part=isl_pw_aff_scale_val(part,isl_val_int_from_si(ctx,[layout.strides[d] integerValue]));
        value=isl_pw_aff_add(value,part);
    }
    return isl_map_coalesce(isl_map_from_pw_aff(value));
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
struct MeshIndexBounds MeshIndexBoundsOf(id expression,NSUInteger first,NSUInteger end) {
    isl_ctx *ctx=isl_ctx_alloc();isl_pw_aff *value=MeshIndexExpression(ctx,expression,first,end);
    isl_val *low=isl_pw_aff_min_val(isl_pw_aff_copy(value)),*high=isl_pw_aff_max_val(value);
    struct MeshIndexBounds result={isl_val_get_num_si(low),isl_val_get_num_si(high)};
    isl_val_free(low);isl_val_free(high);isl_ctx_free(ctx);return result;
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
