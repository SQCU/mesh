#import "metal-recording.h"
#import "mesh-index-relation.h"
#include "mesh-graph.h"
#include <isl/ilp.h>
#include <isl/val.h>

struct TensorRun { NSUInteger first, end, part; NSInteger offset; };
static int TensorRunOrder(const void *a,const void *b) {
    NSUInteger x=((const struct TensorRun *)a)->first,y=((const struct TensorRun *)b)->first;
    return x<y?-1:x>y;
}
@implementation MetalTensorTraversal {
    NSData *_runs;
}
+ (instancetype)layouts:(NSArray<MetalTensorLayout *> *)layouts domain:(MetalTensorLayout *)domain selection:(MetalTensorLayout *)selection {
    MetalTensorTraversal *result=[[[self alloc] init] autorelease];
    NSMutableData *runs=[NSMutableData data];NSUInteger ordinal=0;
    domain=[domain reshape:@[@(domain.count)]];
    isl_ctx *ctx=isl_ctx_alloc();
    isl_map *indices=isl_map_apply_range(MeshIndexRelation(ctx,selection),isl_map_reverse(MeshIndexRelation(ctx,domain)));
    indices=isl_map_lexmin(indices);
    for(NSUInteger part=0;part<layouts.count;part++) {
        MetalTensorLayout *storage=[layouts[part] reshape:@[@(layouts[part].count)]];
        NSString *bounds=[NSString stringWithFormat:@"{ [i] : %lu <= i < %lu }",(unsigned long)ordinal,(unsigned long)(ordinal+storage.count)];
        isl_map *selected=isl_map_intersect_range(isl_map_copy(indices),isl_set_read_from_str(ctx,bounds.UTF8String));
        isl_set *positions=isl_map_domain(isl_map_copy(selected));
        MetalTensorLayout *ordered=MeshIndexSelection(positions);isl_set_free(positions);
        NSString *shift=[NSString stringWithFormat:@"{ [i] -> [i - %lu] }",(unsigned long)ordinal];
        selected=isl_map_apply_range(selected,isl_map_read_from_str(ctx,shift.UTF8String));
        selected=isl_map_apply_range(selected,MeshIndexRelation(ctx,storage));
        selected=isl_map_apply_range(MeshIndexRelation(ctx,ordered),selected);
        MetalTensorLayout *placed=ordered.count?[MetalTensorLayout shape:ordered.shape
            index:MeshIndexTerms(isl_map_dim_min(selected,0)) parameters:@{}]:ordered;
        if(!ordered.count)isl_map_free(selected);
        ordinal+=storage.count;
        NSData *logical=[ordered orderedSpans],*physical=[placed orderedSpans];
        const struct MetalTensorSpan *where=logical.bytes,*at=physical.bytes;
        NSUInteger i=0,j=0,usedLogical=0,usedPhysical=0;
        while(i<logical.length/sizeof *where&&j<physical.length/sizeof *at) {
            NSUInteger count=MIN(where[i].count-usedLogical,at[j].count-usedPhysical);
            NSUInteger first=where[i].offset+usedLogical;
            struct TensorRun run={first,first+count,part,at[j].offset+(NSInteger)usedPhysical};
            if(count)[runs appendBytes:&run length:sizeof run];
            usedLogical+=count;usedPhysical+=count;
            if(usedLogical==where[i].count){i++;usedLogical=0;}
            if(usedPhysical==at[j].count){j++;usedPhysical=0;}
        }
    }
    isl_map_free(indices);isl_ctx_free(ctx);NSParameterAssert(ordinal==domain.count);
    qsort(runs.mutableBytes,runs.length/sizeof(struct TensorRun),sizeof(struct TensorRun),TensorRunOrder);
    const struct TensorRun *values=runs.bytes;NSMutableArray *cuts=[NSMutableArray array];
    for(NSUInteger k=0;k<runs.length/sizeof *values;k++) {
        [cuts addObjectsFromArray:@[@(values[k].first),@(values[k].end)]];
    }
    result->_runs=[runs copy];
    result->_cuts=[[[[NSSet setWithArray:cuts] allObjects] sortedArrayUsingSelector:@selector(compare:)] copy];
    return result;
}
- (void)dealloc { [_runs release];[_cuts release];[super dealloc]; }
- (struct MetalTensorFragment)fragment:(NSUInteger)first end:(NSUInteger)end {
    const struct TensorRun *runs=_runs.bytes;NSUInteger left=0,right=_runs.length/sizeof *runs;
    while(left<right){NSUInteger middle=left+(right-left)/2;if(runs[middle].end<=first)left=middle+1;else right=middle;}
    if(left==_runs.length/sizeof *runs||runs[left].first>=end)return (struct MetalTensorFragment){0,0,0};
    /* the part of [first, end) the run it meets holds: a binding's domain may be less than the call's whole (an
       all-gather's source its own segment), and a cut stands at every run end inside a domain, so no other run meets it */
    const NSUInteger from=first>runs[left].first?first:runs[left].first,to=end<runs[left].end?end:runs[left].end;
    return (struct MetalTensorFragment){runs[left].part,to-from,runs[left].offset+(NSInteger)(from-runs[left].first)};
}
@end

static NSArray<NSNumber *> *TensorStrides(NSArray<NSNumber *> *shape) {
    NSMutableArray *steps=[NSMutableArray arrayWithCapacity:shape.count];
    for(NSUInteger d=0;d<shape.count;d++)[steps addObject:@0];
    NSUInteger step=1;
    for(NSUInteger d=shape.count;d--;) { steps[d]=@(step);step*=[shape[d] unsignedIntegerValue]; }
    return steps;
}
static NSUInteger TensorCount(NSArray<NSNumber *> *shape) {
    NSUInteger count=1;for(NSNumber *n in shape)count*=n.unsignedIntegerValue;return count;
}
static NSData *TensorWords(NSArray<NSNumber *> *values) {
    NSMutableData *data=[NSMutableData dataWithLength:values.count*sizeof(NSInteger)];
    NSInteger *out=data.mutableBytes;
    for(NSUInteger i=0;i<values.count;i++)out[i]=values[i].integerValue;
    return data;
}

// Integer index expressions are preparation terms, not executable descriptors.
// Substitution erases each view transform into its factor coordinates. Floor
// division makes the algebra valid for reversed selections as well as forward
// ones; source emission shifts a possibly negative numerator by a known bound.
static NSInteger TensorFloor(NSInteger value,NSInteger divisor) {
    NSInteger q=value/divisor,r=value%divisor;return q-(r<0);
}
static BOOL TensorConstant(id expression) {return [expression isKindOfClass:NSNumber.class];}
static BOOL TensorOp(id expression,NSString *op) {
    return [expression isKindOfClass:NSArray.class]&&[expression[0] isEqual:op];
}
static BOOL TensorGeneral(id expression) {
    if(![expression isKindOfClass:NSArray.class])return NO;
    NSString *op=expression[0];
    return ![op isEqual:@"add"]&&![op isEqual:@"mul"]&&![op isEqual:@"div"]&&![op isEqual:@"mod"];
}
static id TensorSum(NSArray *terms);
static id TensorScale(id value,NSInteger scale) {
    if(!scale)return @0;if(scale==1)return value;
    if(TensorConstant(value))return @([value integerValue]*scale);
    if(TensorOp(value,@"mul"))return TensorScale(value[1],[value[2] integerValue]*scale);
    if(TensorOp(value,@"add")) {
        NSMutableArray *terms=[NSMutableArray array];
        for(NSUInteger k=1;k<[value count];k++)[terms addObject:TensorScale(value[k],scale)];
        return TensorSum(terms);
    }
    return @[@"mul",value,@(scale)];
}
static id TensorSum(NSArray *terms) {
    NSMutableArray *pending=[NSMutableArray arrayWithArray:terms];NSMutableDictionary *coefficients=[NSMutableDictionary dictionary];
    NSInteger constant=0;
    for(NSUInteger k=0;k<pending.count;k++) {
        id value=pending[k];
        if(TensorConstant(value)){constant+=[value integerValue];continue;}
        if(TensorOp(value,@"add")){[pending addObjectsFromArray:[value subarrayWithRange:NSMakeRange(1,[value count]-1)]];continue;}
        NSInteger scale=1;if(TensorOp(value,@"mul")){scale=[value[2] integerValue];value=value[1];}
        coefficients[value]=@([coefficients[value] integerValue]+scale);
    }
    for(id value in [coefficients allKeys])if(TensorOp(value,@"mod")) {
        NSInteger scale=[coefficients[value] integerValue];id quotient=@[@"div",value[1],value[2]];
        if(scale&&[coefficients[quotient] integerValue]==scale*[value[2] integerValue]) {
            [coefficients removeObjectForKey:value];[coefficients removeObjectForKey:quotient];
            NSMutableArray *reduced=[NSMutableArray arrayWithObjects:@(constant),TensorScale(value[1],scale),nil];
            for(id term in coefficients)[reduced addObject:TensorScale(term,[coefficients[term] integerValue])];
            return TensorSum(reduced);
        }
    }
    NSMutableArray *result=[NSMutableArray arrayWithObject:@"add"];
    NSArray *keys=[[coefficients allKeys] sortedArrayUsingComparator:^NSComparisonResult(id a,id b){return [[a description] compare:[b description]];}];
    for(id value in keys)if([coefficients[value] integerValue])[result addObject:TensorScale(value,[coefficients[value] integerValue])];
    if(constant)[result addObject:@(constant)];
    return result.count==1?@0:result.count==2?result[1]:result;
}
static NSInteger TensorGCD(NSInteger a,NSInteger b) {while(b){NSInteger r=a%b;a=b;b=r;}return a;}
static id TensorQuotient(id value,NSInteger divisor,BOOL remainder) {
    if(divisor==1)return remainder?@0:value;
    if(TensorConstant(value)){NSInteger v=[value integerValue],q=TensorFloor(v,divisor);return @(remainder?v-q*divisor:q);}
    if(!remainder&&TensorOp(value,@"div"))return TensorQuotient(value[1],divisor*[value[2] integerValue],NO);
    if(remainder&&TensorOp(value,@"mod")&&[value[2] integerValue]%divisor==0)return TensorQuotient(value[1],divisor,YES);
    NSArray *terms=TensorOp(value,@"add")?[value subarrayWithRange:NSMakeRange(1,[value count]-1)]:@[value];
    NSMutableArray *whole=[NSMutableArray array],*rest=[NSMutableArray array];
    for(id term in terms) {
        NSInteger scale=TensorOp(term,@"mul")?[term[2] integerValue]:1;id factor=TensorOp(term,@"mul")?term[1]:term;
        if(TensorConstant(term)) {
            NSInteger v=[term integerValue],q=TensorFloor(v,divisor);[whole addObject:@(q)];if(v-q*divisor)[rest addObject:@(v-q*divisor)];
        } else if(scale%divisor==0)[whole addObject:TensorScale(factor,scale/divisor)];
        else [rest addObject:term];
    }
    id residual=TensorSum(rest);
    if(!TensorConstant(residual)) {
        NSArray *parts=TensorOp(residual,@"add")?[residual subarrayWithRange:NSMakeRange(1,[residual count]-1)]:@[residual];
        NSInteger common=divisor,constant=0;
        for(id part in parts) {
            if(TensorConstant(part))constant+=[part integerValue];
            else common=TensorGCD(common,TensorOp(part,@"mul")?labs([part[2] integerValue]):1);
        }
        if(common>1) {
            NSMutableArray *reduced=[NSMutableArray arrayWithObject:@(TensorFloor(constant,common))];
            for(id part in parts)if(!TensorConstant(part))
                [reduced addObject:TensorScale(TensorOp(part,@"mul")?part[1]:part,TensorOp(part,@"mul")?[part[2] integerValue]/common:1)];
            residual=TensorQuotient(TensorSum(reduced),divisor/common,remainder);
            if(remainder)residual=TensorSum(@[TensorScale(residual,common),@(constant-TensorFloor(constant,common)*common)]);
        } else residual=@[remainder?@"mod":@"div",residual,@(divisor)];
    }
    else {NSInteger v=[residual integerValue],q=TensorFloor(v,divisor);residual=@(remainder?v-q*divisor:q);}
    if(remainder)return residual;
    [whole addObject:residual];return TensorSum(whole);
}
static id TensorSubstitute(id expression,id index) {
    if(TensorConstant(expression))return expression;
    if([expression isEqual:@"i"])return index;
    NSString *op=expression[0];
    if(TensorGeneral(expression)) {
        NSMutableArray *terms=[NSMutableArray arrayWithObject:op];
        for(NSUInteger k=1;k<[expression count];k++)[terms addObject:TensorSubstitute(expression[k],index)];
        return terms;
    }
    if([op isEqual:@"add"]) {
        NSMutableArray *terms=[NSMutableArray array];for(NSUInteger k=1;k<[expression count];k++)[terms addObject:TensorSubstitute(expression[k],index)];
        return TensorSum(terms);
    }
    id value=TensorSubstitute(expression[1],index);
    return [op isEqual:@"mul"]?TensorScale(value,[expression[2] integerValue]):TensorQuotient(value,[expression[2] integerValue],[op isEqual:@"mod"]);
}
static NSInteger TensorEvaluate(id expression,NSUInteger logical) {
    if(TensorConstant(expression))return [expression integerValue];if([expression isEqual:@"i"])return (NSInteger)logical;
    NSString *op=expression[0];
    if(TensorGeneral(expression)) {
        if([op isEqual:@"select"])return TensorEvaluate(expression[TensorEvaluate(expression[1],logical)?2:3],logical);
        NSInteger result=TensorEvaluate(expression[1],logical);
        for(NSUInteger k=2;k<[expression count];k++) {
            NSInteger value=TensorEvaluate(expression[k],logical);
            if([op isEqual:@"min"])result=MIN(result,value);else if([op isEqual:@"max"])result=MAX(result,value);
            else if([op isEqual:@"eq"])result=result==value;else if([op isEqual:@"le"])result=result<=value;
            else if([op isEqual:@"lt"])result=result<value;else if([op isEqual:@"ge"])result=result>=value;
            else if([op isEqual:@"gt"])result=result>value;else if([op isEqual:@"and"])result=result&&value;
            else if([op isEqual:@"or"])result=result||value;
        }
        return result;
    }
    if([op isEqual:@"add"]){NSInteger sum=0;for(NSUInteger k=1;k<[expression count];k++)sum+=TensorEvaluate(expression[k],logical);return sum;}
    NSInteger value=TensorEvaluate(expression[1],logical),scale=[expression[2] integerValue];
    if([op isEqual:@"mul"])return value*scale;
    NSInteger q=TensorFloor(value,scale);return [op isEqual:@"div"]?q:value-q*scale;
}
// TensorEvaluate's expression as one flat postfix program over the same operations, so a caller that reads an
// expression at every position of a large layout walks no objects: built once, then evaluated with integers only.
// Both arms of a select are evaluated (every scale is a constant, so neither arm can fail).
enum TensorProgramOp { ProgramPushConstant, ProgramPushPosition, ProgramSum, ProgramMultiply, ProgramDivide, ProgramModulo,
    ProgramSelect, ProgramMinimum, ProgramMaximum, ProgramEqual, ProgramAtMost, ProgramBelow, ProgramAtLeast, ProgramAbove,
    ProgramBoth, ProgramEither };
struct TensorProgramStep { enum TensorProgramOp op; NSInteger value; };
static void TensorCompile(id expression,NSMutableData *program) {
    struct TensorProgramStep step={ProgramPushConstant,0};
    if(TensorConstant(expression)){step.value=[expression integerValue];[program appendBytes:&step length:sizeof step];return;}
    if([expression isEqual:@"i"]){step.op=ProgramPushPosition;[program appendBytes:&step length:sizeof step];return;}
    NSString *op=expression[0];const NSUInteger arity=[expression count];
    if([op isEqual:@"select"]) {
        for(NSUInteger k=1;k<4;k++)TensorCompile(expression[k],program);
        step.op=ProgramSelect;[program appendBytes:&step length:sizeof step];return;
    }
    if([op isEqual:@"add"]) {
        for(NSUInteger k=1;k<arity;k++)TensorCompile(expression[k],program);
        step.op=ProgramSum;step.value=(NSInteger)arity-1;[program appendBytes:&step length:sizeof step];return;
    }
    if([op isEqual:@"mul"]||[op isEqual:@"div"]||[op isEqual:@"mod"]) {
        TensorCompile(expression[1],program);
        step.op=[op isEqual:@"mul"]?ProgramMultiply:[op isEqual:@"div"]?ProgramDivide:ProgramModulo;
        step.value=[expression[2] integerValue];[program appendBytes:&step length:sizeof step];return;
    }
    NSDictionary *ops=@{@"min":@(ProgramMinimum),@"max":@(ProgramMaximum),@"eq":@(ProgramEqual),@"le":@(ProgramAtMost),
                        @"lt":@(ProgramBelow),@"ge":@(ProgramAtLeast),@"gt":@(ProgramAbove),@"and":@(ProgramBoth),@"or":@(ProgramEither)};
    if(!ops[op])[NSException raise:NSInvalidArgumentException format:@"unknown index term %@",op];
    TensorCompile(expression[1],program);
    for(NSUInteger k=2;k<arity;k++) {
        TensorCompile(expression[k],program);step.op=(enum TensorProgramOp)[ops[op] intValue];
        [program appendBytes:&step length:sizeof step];
    }
}
static NSInteger TensorRun(const struct TensorProgramStep *program,NSUInteger length,NSInteger position,NSInteger *stack) {
    NSUInteger top=0;
    for(NSUInteger k=0;k<length;k++) {
        const struct TensorProgramStep step=program[k];
        switch(step.op) {
        case ProgramPushConstant: stack[top++]=step.value; break;
        case ProgramPushPosition: stack[top++]=position; break;
        case ProgramSum: {NSInteger sum=0;for(NSInteger j=0;j<step.value;j++)sum+=stack[--top];stack[top++]=sum;} break;
        case ProgramMultiply: stack[top-1]*=step.value; break;
        case ProgramDivide: stack[top-1]=TensorFloor(stack[top-1],step.value); break;
        case ProgramModulo: stack[top-1]-=TensorFloor(stack[top-1],step.value)*step.value; break;
        case ProgramSelect: {NSInteger no=stack[--top],yes=stack[--top],condition=stack[--top];stack[top++]=condition?yes:no;} break;
        default: {
            NSInteger right=stack[--top],left=stack[top-1],result=0;
            switch(step.op) {
            case ProgramMinimum: result=MIN(left,right); break;
            case ProgramMaximum: result=MAX(left,right); break;
            case ProgramEqual: result=left==right; break;
            case ProgramAtMost: result=left<=right; break;
            case ProgramBelow: result=left<right; break;
            case ProgramAtLeast: result=left>=right; break;
            case ProgramAbove: result=left>right; break;
            case ProgramBoth: result=left&&right; break;
            default: result=left||right; break;
            }
            stack[top-1]=result;
        }
        }
    }
    return stack[0];
}
struct TensorExpressionRange { NSInteger low,high; };
static struct TensorExpressionRange TensorRangeBetween(id expression,NSUInteger first,NSUInteger end) {
    if(TensorConstant(expression)){NSInteger v=[expression integerValue];return (struct TensorExpressionRange){v,v};}
    if([expression isEqual:@"i"])return (struct TensorExpressionRange){(NSInteger)first,(NSInteger)(end?end-1:first)};
    if(TensorGeneral(expression)) {
        struct MeshIndexBounds bounds=MeshIndexBoundsOf(expression,first,MAX(first+1,end));
        return (struct TensorExpressionRange){bounds.low,bounds.high};
    }
    NSString *op=expression[0];
    if([op isEqual:@"add"]) {
        struct TensorExpressionRange sum={0,0};
        for(NSUInteger k=1;k<[expression count];k++){struct TensorExpressionRange v=TensorRangeBetween(expression[k],first,end);sum.low+=v.low;sum.high+=v.high;}
        return sum;
    }
    struct TensorExpressionRange v=TensorRangeBetween(expression[1],first,end);NSInteger n=[expression[2] integerValue];
    if([op isEqual:@"mul"])return (struct TensorExpressionRange){MIN(v.low*n,v.high*n),MAX(v.low*n,v.high*n)};
    NSInteger low=TensorFloor(v.low,n),high=TensorFloor(v.high,n);
    if([op isEqual:@"div"])return (struct TensorExpressionRange){low,high};
    return low==high?(struct TensorExpressionRange){v.low-low*n,v.high-low*n}:(struct TensorExpressionRange){0,n-1};
}
static struct TensorExpressionRange TensorRange(id expression,NSUInteger count) {
    return TensorRangeBetween(expression,0,count);
}
static id TensorBound(id expression,NSUInteger count) {
    if(TensorConstant(expression))return expression;
    if([expression isEqual:@"i"])return count<=1?@0:expression;
    if(!count)return @0;
    isl_ctx *ctx=isl_ctx_alloc();id result=nil;
    @try { result=MeshIndexTerms(MeshIndexExpression(ctx,expression,0,count)); }
    @finally { isl_ctx_free(ctx); }
    return result;
}
static id TensorNormalize(id expression,NSUInteger count) {
    return TensorBound(TensorSubstitute(expression,@"i"),count);
}
static NSArray *TensorDigits(NSArray<NSNumber *> *shape) {
    NSMutableArray *result=[NSMutableArray array];NSArray *steps=TensorStrides(shape);
    for(NSUInteger d=0;d<shape.count;d++) {
        id value=TensorQuotient(@"i",[steps[d] integerValue]?:1,NO);
        [result addObject:TensorQuotient(value,[shape[d] integerValue]?:1,YES)];
    }
    return result;
}
static NSString *TensorExpressionSource(id expression,NSUInteger count) {
    if(TensorConstant(expression))return [NSString stringWithFormat:@"%ldl",(long)[expression integerValue]];
    if([expression isEqual:@"i"])return @"long(i)";
    NSString *op=expression[0];
    if(TensorGeneral(expression)) {
        NSMutableArray *args=[NSMutableArray array];
        for(NSUInteger k=1;k<[expression count];k++)[args addObject:TensorExpressionSource(expression[k],count)];
        if([op isEqual:@"select"])return [NSString stringWithFormat:@"select(%@, %@, bool(%@))",args[2],args[1],args[0]];
        NSString *value=args[0];
        for(NSUInteger k=1;k<args.count;k++) {
            if([op isEqual:@"min"]||[op isEqual:@"max"])value=[NSString stringWithFormat:@"%@(long(%@), long(%@))",op,value,args[k]];
            else {
                NSString *symbol=@{@"eq":@"==",@"le":@"<=",@"lt":@"<",@"ge":@">=",@"gt":@">",@"and":@"&",@"or":@"|"}[op];
                value=([op isEqual:@"and"]||[op isEqual:@"or"])?
                    [NSString stringWithFormat:@"long(bool(%@) %@ bool(%@))",value,symbol,args[k]]:
                    [NSString stringWithFormat:@"long(%@ %@ %@)",value,symbol,args[k]];
            }
        }
        return value;
    }
    if([op isEqual:@"add"]) {
        NSMutableArray *terms=[NSMutableArray array];for(NSUInteger k=1;k<[expression count];k++)[terms addObject:TensorExpressionSource(expression[k],count)];
        return [NSString stringWithFormat:@"(%@)",[terms componentsJoinedByString:@" + "]];
    }
    NSString *value=TensorExpressionSource(expression[1],count);NSInteger n=[expression[2] integerValue];
    if([op isEqual:@"mul"])return [NSString stringWithFormat:@"(%@ * %ldl)",value,(long)n];
    struct TensorExpressionRange bounds=TensorRange(expression[1],count);NSInteger shift=MIN(0,TensorFloor(bounds.low,n))*n;
    if(shift)value=[NSString stringWithFormat:@"(%@ - %ldl)",value,(long)shift];
    NSString *result=[NSString stringWithFormat:@"(%@ %@ %ldl)",value,[op isEqual:@"div"]?@"/":@"%",(long)n];
    return shift&&[op isEqual:@"div"]?[NSString stringWithFormat:@"(%@ + %ldl)",result,(long)(shift/n)]:result;
}
static MetalTensorLayout *TensorCompose(MetalTensorLayout *source,NSArray *shape,id index) {
    NSMutableArray *coordinates=[NSMutableArray array];
    for(id value in source.coordinates)[coordinates addObject:TensorSubstitute(value,index)];
    return [source shaped:shape coordinates:coordinates];
}

// Emit the composed index function without reconstructing it from a relation.
// Relations are needed for the inverse, but known functions remain functions.
// The solver's context ends here; the result keeps only index terms and
// declared constants.
static MetalTensorLayout *TensorFunctionLayout(isl_pw_aff *value,NSArray *shape) {
    if(!TensorCount(shape)) {
        isl_pw_aff_free(value);
        return [MetalTensorLayout shape:shape strides:TensorStrides(shape) offset:0 maps:@{} indices:nil];
    }
    const struct MeshIndexBounds bounds=MeshIndexImageBounds(isl_map_from_pw_aff(isl_pw_aff_copy(value)));
    NSInteger low=bounds.low,high=bounds.high;
    id terms=MeshIndexTerms(value);
    MetalTensorLayout *storage=[MetalTensorLayout shape:@[@(high-low+1)] strides:@[@1] offset:low maps:@{} indices:nil];
    return [storage shaped:shape coordinates:@[TensorSum(@[terms,@(-low)])]];
}

// A maximal safe affine piece of an integer term. Dependency construction
// advances over these runs; it does not expand a reshape into scalar metadata.
struct TensorPiece { NSInteger value,step; NSUInteger end; };
static struct TensorPiece TensorExpressionPiece(id expression,NSUInteger at,NSUInteger end) {
    if(TensorConstant(expression))return (struct TensorPiece){[expression integerValue],0,end};
    if([expression isEqual:@"i"])return (struct TensorPiece){(NSInteger)at,1,end};
    if(TensorGeneral(expression)) {
        // The run from `at` read with integers: the compiled expression at each position until it leaves the line its
        // first two positions fix, MeshIndexRunOf's answer without three isl contexts a run. Changed by the LSTM
        // demonstration session: a piecewise coordinate's spans called it once a run, and one LSTM step's realization
        // spent 56 s there on m5ult2 (2026-11-01, all of a 20 s native sample under reaches:).
        // A layout's spans ask this once a run of the same coordinate (16384 runs of two positions in the LSTM's
        // realization): the expression is compiled once and kept, with the expression itself held so its address
        // names no other while it is kept, one a thread.
        static _Thread_local id held=nil;static _Thread_local NSData *kept=nil;
        if(held!=expression) {
            NSMutableData *made=[NSMutableData data];TensorCompile(expression,made);
            [held release];[kept release];held=[expression retain];kept=[made retain];
        }
        const struct TensorProgramStep *program=kept.bytes;const NSUInteger length=kept.length/sizeof *program;
        NSInteger stack[length+1];
        const NSInteger value=TensorRun(program,length,(NSInteger)at,stack);
        const NSInteger step=end>at+1?TensorRun(program,length,(NSInteger)at+1,stack)-value:0;
        NSUInteger stop=at+1;
        while(stop<end&&TensorRun(program,length,(NSInteger)stop,stack)==value+step*(NSInteger)(stop-at))stop++;
        return (struct TensorPiece){value,step,stop};
    }
    NSString *op=expression[0];
    if([op isEqual:@"add"]) {
        struct TensorPiece result={0,0,end};
        for(NSUInteger k=1;k<[expression count];k++) {
            struct TensorPiece piece=TensorExpressionPiece(expression[k],at,end);
            result.value+=piece.value;result.step+=piece.step;result.end=MIN(result.end,piece.end);
        }
        return result;
    }
    struct TensorPiece value=TensorExpressionPiece(expression[1],at,end);NSInteger n=[expression[2] integerValue];
    if([op isEqual:@"mul"]){value.value*=n;value.step*=n;return value;}
    NSInteger q=TensorFloor(value.value,n),r=value.value-q*n;BOOL remainder=[op isEqual:@"mod"];
    if(value.step%n==0)return (struct TensorPiece){remainder?r:q,remainder?0:value.step/n,value.end};
    NSUInteger run=value.step>0?(NSUInteger)((n-1-r)/value.step+1):(NSUInteger)(r/-value.step+1);
    return (struct TensorPiece){remainder?r:q,remainder?value.step:0,MIN(value.end,at+run)};
}
static struct TensorPiece TensorLayoutPiece(MetalTensorLayout *layout,NSArray *coordinates,NSUInteger at,NSUInteger end) {
    struct TensorPiece result={layout.offset,0,end};
    for(NSUInteger d=0;d<coordinates.count;d++) {
        struct TensorPiece part=TensorExpressionPiece(coordinates[d],at,result.end);
        const NSInteger *map=((NSData *)layout.maps[@(d)]).bytes;
        if(map) {
            NSInteger first=map[part.value],step=0;NSUInteger count=part.end-at;
            if(part.step&&count>1) {
                step=map[part.value+part.step]-first;
                for(NSUInteger k=2;k<count;k++)if(map[part.value+(NSInteger)k*part.step]!=first+(NSInteger)k*step){part.end=at+k;break;}
            }
            part.value=first;part.step=step;
        } else {NSInteger scale=[layout.strides[d] integerValue];part.value*=scale;part.step*=scale;}
        result.value+=part.value;result.step+=part.step;result.end=MIN(result.end,part.end);
    }
    return result;
}
static NSData *TensorExpressionSpans(MetalTensorLayout *layout,BOOL ordered) {
    NSMutableData *result=[NSMutableData data];NSArray *coordinates=layout.coordinates;
    for(NSUInteger at=0;at<layout.count;) {
        struct TensorPiece piece=TensorLayoutPiece(layout,coordinates,at,layout.count);NSUInteger count=piece.end-at;
        BOOL range=piece.step==1||(!ordered&&(piece.step==0||piece.step==-1));
        NSUInteger entries=range?1:count;
        for(NSUInteger k=0;k<entries;k++) {
            struct MetalTensorSpan span={piece.value+(NSInteger)k*piece.step,range?count:1};
            if(!ordered&&range){if(!piece.step)span.count=1;else if(piece.step<0)span.offset-=(NSInteger)count-1;}
            struct MetalTensorSpan *last=result.length?(struct MetalTensorSpan *)result.mutableBytes+result.length/sizeof span-1:NULL;
            if(last&&last->offset+(NSInteger)last->count==span.offset)last->count+=span.count;
            else [result appendBytes:&span length:sizeof span];
        }
        at=piece.end;
    }
    return result;
}
// Prove aligned adjacent lanes with the same integer algebra used for source
// emission. Unknown alignment yields scalar accesses, never a runtime check.
static NSUInteger TensorVectorSpan(MetalTensorLayout *layout) {
    if(layout.count%4)return 1;
    NSUInteger groups=layout.count/4;NSMutableArray *base=[NSMutableArray array],*origin=[NSMutableArray arrayWithObject:@(layout.offset)];
    NSArray *coordinates=layout.coordinates;
    for(NSUInteger d=0;d<coordinates.count;d++) {
        id value=TensorBound(TensorSubstitute(coordinates[d],TensorScale(@"i",4)),groups);[base addObject:value];
        NSData *data=layout.maps[@(d)];const NSInteger *map=data.bytes;
        if(map) {
            NSInteger residue=map[TensorConstant(value)?[value integerValue]:0]%4;
            if(!TensorConstant(value))for(NSUInteger k=1;k<data.length/sizeof *map;k++)if((map[k]-residue)%4)return 1;
            [origin addObject:@(residue)];
        } else [origin addObject:TensorScale(value,[layout.strides[d] integerValue])];
    }
    if(![TensorBound(TensorQuotient(TensorSum(origin),4,YES),groups) isEqual:@0])return 1;
    for(NSUInteger lane=1;lane<4;lane++) {
        NSMutableArray *delta=[NSMutableArray array];
        for(NSUInteger d=0;d<coordinates.count;d++) {
            id value=TensorBound(TensorSubstitute(coordinates[d],TensorSum(@[TensorScale(@"i",4),@(lane)])),groups);
            if(layout.maps[@(d)]){if(![value isEqual:base[d]])return 1;}
            else [delta addObject:TensorScale(TensorSum(@[value,TensorScale(base[d],-1)]),[layout.strides[d] integerValue])];
        }
        if(![TensorSum(delta) isEqual:@(lane)])return 1;
    }
    return 4;
}

struct TensorReshapeChunk { NSUInteger count; NSInteger step; NSData *map; };
static MetalTensorLayout *TensorReshape(MetalTensorLayout *source,NSArray<NSNumber *> *shape) {
    if(!source.factored)return nil;
    if(!source.count)return [MetalTensorLayout shape:shape strides:TensorStrides(shape) offset:source.offset maps:@{} indices:nil];
    // Adjacent affine axes form one linear run, even with a stride other than
    // one. A mapped axis remains a bounded factor of the same layout.
    NSMutableData *storage=[NSMutableData dataWithLength:source.indexShape.count*sizeof(struct TensorReshapeChunk)];
    struct TensorReshapeChunk *chunks=storage.mutableBytes;NSUInteger count=0;
    for(NSUInteger d=source.indexShape.count;d--;) {
        const NSUInteger n=[source.indexShape[d] unsignedIntegerValue];if(n==1)continue;
        const NSInteger step=[source.strides[d] integerValue];NSData *map=source.maps[@(d)];
        if(!map&&count&&!chunks[count-1].map&&step==chunks[count-1].step*(NSInteger)chunks[count-1].count)
            chunks[count-1].count*=n;
        else chunks[count++]=(struct TensorReshapeChunk){n,step,map};
    }
    NSMutableArray *steps=[[TensorStrides(shape) mutableCopy] autorelease];
    NSMutableDictionary *maps=[NSMutableDictionary dictionary];NSInteger d=(NSInteger)shape.count-1,origin=source.offset;
    for(NSUInteger c=0;c<count;c++) {
        const struct TensorReshapeChunk chunk=chunks[c];
        const NSInteger end=d;NSUInteger product=1;
        while(d>=0&&(product<chunk.count||[shape[d] unsignedIntegerValue]==1)) {
            const NSUInteger n=[shape[d] unsignedIntegerValue];
            if(n>chunk.count/product||chunk.count%(product*n))return nil;
            steps[d]=@(chunk.step*(NSInteger)product);product*=n;d--;
        }
        if(product!=chunk.count)return nil;
        if(!chunk.map)continue;
        NSMutableArray *axes=[NSMutableArray array];
        for(NSInteger axis=d+1;axis<=end;axis++)if([shape[axis] unsignedIntegerValue]>1)[axes addObject:@(axis)];
        if(axes.count==1){maps[axes[0]]=chunk.map;continue;}
        // Splitting an existing axis map needs only that map's domain, not a
        // table for every outer row. Retain separate factors when they add.
        const NSInteger *values=chunk.map.bytes;const NSInteger base=values[0];
        NSUInteger stride=1;NSMutableArray *logical=[NSMutableArray arrayWithCapacity:axes.count];
        for(NSUInteger axis=0;axis<axes.count;axis++)[logical addObject:@0];
        for(NSUInteger a=axes.count;a--;) {
            NSNumber *axis=axes[a];const NSUInteger n=[shape[axis.unsignedIntegerValue] unsignedIntegerValue];
            NSMutableData *data=[NSMutableData dataWithLength:n*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
            for(NSUInteger i=0;i<n;i++)out[i]=values[i*stride]-base;
            maps[axis]=data;logical[a]=@(stride);stride*=n;
        }
        for(NSUInteger i=0;i<chunk.count;i++) {
            NSInteger value=base;
            for(NSUInteger a=0;a<axes.count;a++) {
                NSNumber *axis=axes[a];const NSInteger *field=((NSData *)maps[axis]).bytes;
                value+=field[(i/[logical[a] unsignedIntegerValue])%[shape[axis.unsignedIntegerValue] unsignedIntegerValue]];
            }
            if(value!=values[i])return nil;
        }
        origin+=base;
    }
    return [MetalTensorLayout shape:shape strides:steps offset:origin maps:maps indices:nil];
}

// Shared logical/index boundaries split a view into independent additive
// domains. Transforms compose only the domains they change during preparation.
struct TensorSection { NSUInteger first,end,indexFirst,indexEnd; };
static NSData *TensorSections(MetalTensorLayout *source) {
    NSMutableDictionary *boundaries=[NSMutableDictionary dictionaryWithObject:@0 forKey:@1];
    NSUInteger product=1;
    for(NSUInteger d=0;d<source.indexShape.count;d++) {
        product*=[source.indexShape[d] unsignedIntegerValue];boundaries[@(product)]=@(d+1);
    }
    NSMutableData *result=[NSMutableData data];product=1;
    NSUInteger first=0,indexFirst=0;
    for(NSUInteger d=0;d<source.shape.count;d++) {
        product*=[source.shape[d] unsignedIntegerValue];NSNumber *boundary=boundaries[@(product)];
        if(!boundary)continue;
        const struct TensorSection section={first,d+1,indexFirst,boundary.unsignedIntegerValue};
        [result appendBytes:&section length:sizeof section];first=d+1;indexFirst=section.indexEnd;
    }
    return result;
}
static MetalTensorLayout *TensorSectionLayout(MetalTensorLayout *source,struct TensorSection section) {
    const NSRange range=NSMakeRange(section.indexFirst,section.indexEnd-section.indexFirst);
    NSMutableDictionary *maps=[NSMutableDictionary dictionary];
    for(NSUInteger d=section.indexFirst;d<section.indexEnd;d++)
        if(source.maps[@(d)])maps[@(d-section.indexFirst)]=source.maps[@(d)];
    MetalTensorLayout *result=[MetalTensorLayout shape:[source.indexShape subarrayWithRange:range]
        strides:[source.strides subarrayWithRange:range] offset:0 maps:maps indices:nil];
    return [result reshape:[source.shape subarrayWithRange:NSMakeRange(section.first,section.end-section.first)]];
}
static MetalTensorLayout *TensorProduct(NSArray<MetalTensorLayout *> *parts) {
    NSMutableArray *shape=[NSMutableArray array],*basis=[NSMutableArray array],*steps=[NSMutableArray array];
    NSMutableDictionary *maps=[NSMutableDictionary dictionary];NSInteger origin=0;
    for(MetalTensorLayout *part in parts) {
        for(NSNumber *axis in part.maps)maps[@(basis.count+axis.unsignedIntegerValue)]=part.maps[axis];
        [shape addObjectsFromArray:part.shape];[basis addObjectsFromArray:part.indexShape];
        [steps addObjectsFromArray:part.strides];origin+=part.offset;
    }
    MetalTensorLayout *result=[MetalTensorLayout shape:basis strides:steps offset:origin maps:maps indices:nil];
    BOOL factored=YES;for(MetalTensorLayout *part in parts)factored&=part.factored;
    if(!factored) {
        NSMutableArray *coordinates=[NSMutableArray array];NSUInteger suffix=TensorCount(shape);
        for(MetalTensorLayout *part in parts) {
            suffix=part.count?suffix/part.count:0;
            id index=TensorQuotient(TensorQuotient(@"i",suffix?:1,NO),part.count?:1,YES);
            for(id value in part.coordinates)[coordinates addObject:TensorSubstitute(value,index)];
        }
        return [result shaped:shape coordinates:coordinates];
    }
    return [result reshape:shape];
}
// An unaffected logical axis may contain an independent index boundary. Expose
// that boundary for composition, then restore the caller's logical shape. A
// changed axis remains whole; its selected values define its new shape.
static MetalTensorLayout *TensorRefineAxes(MetalTensorLayout *source,NSUInteger *selected) {
    NSMutableArray *boundaries=[NSMutableArray array],*shape=[NSMutableArray array];
    NSUInteger product=1,first=1,axis=*selected;
    for(NSNumber *n in source.indexShape){product*=n.unsignedIntegerValue;[boundaries addObject:@(product)];}
    for(NSUInteger d=0;d<source.shape.count;d++) {
        const NSUInteger end=first*[source.shape[d] unsignedIntegerValue];
        if(d==axis)*selected=shape.count;
        else for(NSNumber *boundary in boundaries) {
            const NSUInteger middle=boundary.unsignedIntegerValue;
            if(middle>first&&middle<end&&middle%first==0&&end%middle==0) {
                [shape addObject:@(middle/first)];first=middle;
            }
        }
        [shape addObject:@(end/first)];first=end;
    }
    return [source reshape:shape];
}
static MetalTensorLayout *TensorAxisTransform(MetalTensorLayout *source,NSUInteger axis,
    MetalTensorLayout *(^transform)(MetalTensorLayout *,NSUInteger)) {
    if(!source.count||!source.factored||[source.shape isEqual:source.indexShape])return transform(source,axis);
    NSUInteger refinedAxis=axis;MetalTensorLayout *refined=TensorRefineAxes(source,&refinedAxis);
    if(![refined.shape isEqual:source.shape]) {
        MetalTensorLayout *result=TensorAxisTransform(refined,refinedAxis,transform);
        NSMutableArray *shape=[[source.shape mutableCopy] autorelease];shape[axis]=result.shape[refinedAxis];
        return [result reshape:shape];
    }
    NSData *storage=TensorSections(source);const struct TensorSection *sections=storage.bytes;
    NSMutableArray *parts=[NSMutableArray array];
    for(NSUInteger k=0;k<storage.length/sizeof *sections;k++) {
        const struct TensorSection section=sections[k];MetalTensorLayout *part=TensorSectionLayout(source,section);
        if(axis>=section.first&&axis<section.end)part=transform(part,axis-section.first);
        [parts addObject:part];
    }
    return [TensorProduct(parts) shift:source.offset];
}
// A term repeats after period positions, with a fixed additive change. Zero
// means that no complete period fits this declared domain. This is preparation
// arithmetic; its quotient and finite residue become ordinary index factors.
struct TensorPeriod { NSUInteger period; NSInteger delta; };
static struct TensorPeriod TensorExpressionPeriod(id expression,NSUInteger count) {
    if(TensorConstant(expression))return (struct TensorPeriod){1,0};
    if([expression isEqual:@"i"])return (struct TensorPeriod){1,1};
    if(TensorGeneral(expression)) {
        struct MeshIndexRun run=MeshIndexRunOf(expression,0,count);
        return run.end==count?(struct TensorPeriod){1,run.step}:(struct TensorPeriod){0,0};
    }
    NSString *op=expression[0];
    if([op isEqual:@"add"]) {
        struct TensorPeriod result={1,0};
        for(NSUInteger k=1;k<[expression count];k++) {
            struct TensorPeriod part=TensorExpressionPeriod(expression[k],count);
            if(!part.period)return part;
            NSUInteger factor=part.period/TensorGCD(result.period,part.period);
            if(result.period>count/factor)return (struct TensorPeriod){0,0};
            NSUInteger period=result.period*factor;
            result.delta=result.delta*(NSInteger)factor+part.delta*(NSInteger)(period/part.period);result.period=period;
        }
        return result;
    }
    struct TensorPeriod value=TensorExpressionPeriod(expression[1],count);
    if(!value.period)return value;
    NSInteger n=[expression[2] integerValue];
    if([op isEqual:@"mul"]){value.delta*=n;return value;}
    NSUInteger factor=n/TensorGCD(labs(value.delta),n);
    if(value.period>count/factor)return (struct TensorPeriod){0,0};
    value.period*=factor;value.delta=[op isEqual:@"mod"]?0:value.delta*(NSInteger)factor/n;
    return value;
}

// Append one additive factor, folding its origin now. Maps hold final storage
// indices: selecting a mapped view never emits a lookup into another lookup.
static void TensorSelectionFactor(NSMutableArray *basis,NSMutableArray *steps,NSMutableDictionary *maps,
    NSMutableArray *coordinates,NSInteger *origin,NSData *map,id expression,NSUInteger count) {
    expression=TensorNormalize(expression,count);
    if(TensorConstant(expression)) {
        NSInteger value=[expression integerValue];*origin+=map?((const NSInteger *)map.bytes)[value]:value;return;
    }
    NSUInteger size;
    if(map)size=map.length/sizeof(NSInteger);
    else {
        struct TensorExpressionRange range=TensorRange(expression,count);*origin+=range.low;
        expression=TensorSum(@[expression,@(-range.low)]);size=range.high-range.low+1;
    }
    if(map)maps[@(basis.count)]=map;
    [basis addObject:@(size)];[steps addObject:@1];[coordinates addObject:expression];
}
static MetalTensorLayout *TensorSelect(MetalTensorLayout *source,NSUInteger axis,NSData *indices) {
    const NSUInteger count=indices.length/sizeof(NSInteger);const NSInteger *selected=indices.bytes;
    NSMutableArray *shape=[[source.shape mutableCopy] autorelease];shape[axis]=@(count);
    if(source.factored&&[source.shape isEqual:source.indexShape]) {
        NSMutableDictionary *maps=[[source.maps mutableCopy] autorelease];
        NSMutableData *data=[NSMutableData dataWithLength:indices.length];NSInteger *target=data.mutableBytes;
        const NSInteger *values=((NSData *)source.maps[@(axis)]).bytes;const NSInteger step=[source.strides[axis] integerValue];
        for(NSUInteger i=0;i<count;i++)target[i]=values?values[selected[i]]:selected[i]*step;
        maps[@(axis)]=data;
        return [MetalTensorLayout shape:shape strides:source.strides offset:source.offset maps:maps indices:nil];
    }
    const NSUInteger total=TensorCount(shape);
    if(!total)return [MetalTensorLayout shape:shape strides:TensorStrides(shape) offset:source.offset maps:@{} indices:nil];
    const NSUInteger inner=[TensorStrides(shape)[axis] unsignedIntegerValue],width=[source.shape[axis] unsignedIntegerValue];
    const NSUInteger domain=source.count/width;
    id selectedDigit=TensorDigits(shape)[axis];
    id rest=TensorSum(@[TensorScale(TensorQuotient(@"i",count*inner,NO),inner),TensorQuotient(@"i",inner,YES)]);
    id old=TensorSum(@[TensorScale(TensorQuotient(@"i",inner,NO),width*inner),TensorQuotient(@"i",inner,YES)]);
    NSMutableArray *terms=[NSMutableArray array],*tables=[NSMutableArray array],*linear=[NSMutableArray array];
    NSArray *digits=source.coordinates;
    for(NSUInteger d=0;d<digits.count;d++) {
        NSData *map=source.maps[@(d)];
        if(map){[terms addObject:digits[d]];[tables addObject:map];}
        else [linear addObject:TensorScale(digits[d],[source.strides[d] integerValue])];
    }
    [terms addObject:TensorSum(linear)];[tables addObject:NSNull.null];
    NSMutableArray *basis=[NSMutableArray array],*steps=[NSMutableArray array],*coordinates=[NSMutableArray array],*affine=[NSMutableArray array];
    NSMutableDictionary *maps=[NSMutableDictionary dictionary];NSInteger origin=source.offset;
    for(NSUInteger t=0;t<terms.count;t++) {
        NSData *map=tables[t]==NSNull.null?nil:tables[t];const NSInteger *values=map.bytes;
        NSMutableArray *variants=[NSMutableArray array];
        for(NSUInteger k=0;k<count;k++)
            [variants addObject:TensorNormalize(TensorSubstitute(terms[t],TensorSum(@[old,@(selected[k]*(NSInteger)inner)])),domain)];
        id first=variants[0];BOOL unchanged=YES;
        for(id value in variants)unchanged&=[value isEqual:first];
        if(map&&unchanged) {
            TensorSelectionFactor(basis,steps,maps,coordinates,&origin,map,TensorSubstitute(first,rest),total);continue;
        }
        NSMutableArray *common=[NSMutableArray arrayWithArray:TensorOp(first,@"add")?[first subarrayWithRange:NSMakeRange(1,[first count]-1)]:@[first]];
        for(id value in variants) {
            NSArray *parts=TensorOp(value,@"add")?[value subarrayWithRange:NSMakeRange(1,[value count]-1)]:@[value];
            for(NSUInteger k=common.count;k--;)if(![parts containsObject:common[k]])[common removeObjectAtIndex:k];
        }
        id shared=TensorSum(common);NSMutableArray *residuals=[NSMutableArray array];BOOL constant=YES;
        for(id value in variants) {
            id residual=TensorNormalize(TensorSum(@[value,TensorScale(shared,-1)]),domain);
            [residuals addObject:residual];constant&=TensorConstant(residual);
        }
        if(map&&constant) {
            // Common coordinates plus one constant per selected position. Fuse
            // that small Cartesian map once, without repeating surrounding rows.
            struct TensorExpressionRange range=TensorRange(shared,domain);
            for(NSNumber *shift in residuals){range.low=MAX(range.low,-shift.integerValue);range.high=MIN(range.high,(NSInteger)(map.length/sizeof(NSInteger))-1-shift.integerValue);}
            NSUInteger extent=range.high-range.low+1;
            NSMutableData *data=[NSMutableData dataWithLength:extent*count*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
            for(NSUInteger j=0;j<extent;j++)for(NSUInteger k=0;k<count;k++)out[j*count+k]=values[range.low+(NSInteger)j+[residuals[k] integerValue]];
            id index=TensorSum(@[TensorScale(TensorSum(@[TensorSubstitute(shared,rest),@(-range.low)]),count),selectedDigit]);
            TensorSelectionFactor(basis,steps,maps,coordinates,&origin,data,index,total);continue;
        }
        if(!map) {
            [affine addObject:TensorSubstitute(shared,rest)];
            variants=residuals;
        }
        NSUInteger period=1;NSInteger delta=0;BOOL repeating=YES;NSMutableArray *periods=[NSMutableArray array];
        for(id value in variants) {
            struct TensorPeriod part=TensorExpressionPeriod(value,domain);
            if(!part.period){repeating=NO;break;}
            NSUInteger factor=part.period/TensorGCD(period,part.period);
            if(period>domain/factor){repeating=NO;break;}
            period*=factor;[periods addObject:@[@(part.period),@(part.delta)]];
        }
        if(repeating)for(NSUInteger k=0;k<count;k++) {
            NSInteger change=[periods[k][1] integerValue]*(NSInteger)(period/[periods[k][0] unsignedIntegerValue]);
            if(k==0)delta=change;else repeating&=delta==change;
        }
        if(!repeating||(map&&delta)){period=domain;delta=0;}
        NSMutableData *data=[NSMutableData dataWithLength:period*count*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
        for(NSUInteger j=0;j<period;j++)for(NSUInteger k=0;k<count;k++) {
            NSInteger value=TensorEvaluate(variants[k],j);out[j*count+k]=map?values[value]:value;
        }
        if(delta)[affine addObject:TensorScale(TensorQuotient(rest,period,NO),delta)];
        id index=TensorSum(@[TensorScale(TensorQuotient(rest,period,YES),count),selectedDigit]);
        TensorSelectionFactor(basis,steps,maps,coordinates,&origin,data,index,total);
    }
    TensorSelectionFactor(basis,steps,maps,coordinates,&origin,nil,TensorSum(affine),total);
    MetalTensorLayout *result=[MetalTensorLayout shape:basis strides:steps offset:origin maps:maps indices:nil];
    return [result shaped:shape coordinates:coordinates];
}
// A linear interval can be a rectangular selection of the existing factors.
// Resolve those digits before making a table: huge affine factors stay affine,
// while a mapped factor needs at most its own selected entries. A negative step
// reverses each selected digit; zero repeats one declared value.
static MetalTensorLayout *TensorFactorSlice(MetalTensorLayout *source,NSInteger first,NSUInteger count,NSInteger step) {
    if(!count)return [MetalTensorLayout shape:@[@0] strides:@[@1] offset:source.offset maps:@{} indices:nil];
    if(!step||count==1)return [MetalTensorLayout shape:@[@(count)] strides:@[@0] offset:[source index:first] maps:@{} indices:nil];
    const NSUInteger rank=source.indexShape.count;
    NSMutableData *storage=[NSMutableData dataWithLength:rank*5*sizeof(NSUInteger)];
    NSUInteger *sizes=storage.mutableBytes,*scales=sizes+rank,*residues=scales+rank,*low=residues+rank,*high=low+rank;
    for(NSUInteger d=0;d<rank;d++){sizes[d]=[source.indexShape[d] unsignedIntegerValue];scales[d]=1;}
    const BOOL reverse=step<0;
    NSUInteger begin=(NSUInteger)(reverse?first+(NSInteger)(count-1)*step:first),stride=(NSUInteger)labs(step);
    for(NSUInteger d=rank;d--;) {
        const NSUInteger n=sizes[d];
        if(stride>=n&&stride%n==0) {
            residues[d]=begin%n;begin/=n;stride/=n;sizes[d]=1;scales[d]=0;
        } else {
            if(n%stride)return nil;
            residues[d]=begin%stride;begin/=stride;sizes[d]=n/stride;scales[d]=stride;stride=1;break;
        }
    }
    if(stride!=1)return nil;
    NSUInteger end=begin+count-1;
    for(NSUInteger d=rank;d--;) {low[d]=begin%sizes[d];high[d]=end%sizes[d];begin/=sizes[d];end/=sizes[d];}
    BOOL different=NO;
    for(NSUInteger d=0;d<rank;d++) {
        if(different&&(low[d]||high[d]+1!=sizes[d]))return nil;
        different|=low[d]!=high[d];
    }
    NSMutableArray *shape=[NSMutableArray array],*steps=[NSMutableArray array];
    NSMutableDictionary *maps=[NSMutableDictionary dictionary];NSInteger origin=source.offset;
    for(NSUInteger d=0;d<rank;d++) {
        const NSUInteger n=high[d]-low[d]+1,at=residues[d]+scales[d]*(reverse?high[d]:low[d]);
        const NSInteger delta=(reverse?-1:1)*(NSInteger)scales[d],physical=[source.strides[d] integerValue];
        const NSInteger *values=((NSData *)source.maps[@(d)]).bytes;
        [shape addObject:@(n)];[steps addObject:@(physical*delta)];
        if(values) {
            NSMutableData *data=[NSMutableData dataWithLength:n*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
            for(NSUInteger i=0;i<n;i++)out[i]=values[(NSInteger)at+(NSInteger)i*delta];
            maps[@(d)]=data;
        } else origin+=(NSInteger)at*physical;
    }
    MetalTensorLayout *result=[MetalTensorLayout shape:shape strides:steps offset:origin maps:maps indices:nil];
    return [result reshape:@[@(count)]];
}
static MetalTensorLayout *TensorSlice(MetalTensorLayout *source,NSUInteger axis,NSInteger first,NSUInteger count,NSInteger step) {
    if(source.factored&&[source.shape isEqual:source.indexShape]) {
        NSMutableArray *shape=[[source.shape mutableCopy] autorelease],*strides=[[source.strides mutableCopy] autorelease];
        shape[axis]=@(count);strides[axis]=@(step*[source.strides[axis] integerValue]);
        if(source.maps[@(axis)]) {
            NSMutableDictionary *maps=[[source.maps mutableCopy] autorelease];
            const NSInteger *values=((NSData *)maps[@(axis)]).bytes;
            NSMutableData *data=[NSMutableData dataWithLength:count*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
            for(NSUInteger k=0;k<count;k++)out[k]=values[first+(NSInteger)k*step];
            maps[@(axis)]=data;
            return [MetalTensorLayout shape:shape strides:strides offset:source.offset maps:maps indices:nil];
        }
        return [MetalTensorLayout shape:shape strides:strides offset:source.offset+first*[source.strides[axis] integerValue]
            maps:source.maps indices:nil];
    }
    NSUInteger outer=1,inner=1;
    for(NSUInteger d=0;d<axis;d++)outer*=[source.shape[d] unsignedIntegerValue];
    for(NSUInteger d=axis+1;d<source.shape.count;d++)inner*=[source.shape[d] unsignedIntegerValue];
    if(source.factored&&outer==1&&(inner==1||step==1)) {
        MetalTensorLayout *result=TensorFactorSlice(source,first*(NSInteger)inner,count*inner,step);
        if(result) {
            NSMutableArray *shape=[[source.shape mutableCopy] autorelease];shape[axis]=@(count);
            return [result reshape:shape];
        }
    }
    NSMutableArray *shape=[[source.shape mutableCopy] autorelease];shape[axis]=@(count);
    NSArray *coordinates=TensorDigits(shape),*strides=TensorStrides(source.shape);NSMutableArray *terms=[NSMutableArray array];
    for(NSUInteger d=0;d<shape.count;d++) {
        id value=coordinates[d];if(d==axis)value=TensorSum(@[@(first),TensorScale(value,step)]);
        [terms addObject:TensorScale(value,[strides[d] integerValue])];
    }
    return TensorCompose(source,shape,TensorSum(terms));
}
static BOOL TensorPermutationIsReshape(MetalTensorLayout *source,NSArray<NSNumber *> *axes) {
    NSInteger previous=-1;
    for(NSNumber *axis in axes)if([source.shape[axis.unsignedIntegerValue] unsignedIntegerValue]>1) {
        if(axis.integerValue<previous)return NO;
        previous=axis.integerValue;
    }
    return YES;
}
static MetalTensorLayout *TensorPermute(MetalTensorLayout *source,NSArray<NSNumber *> *axes) {
    NSMutableArray *shape=[NSMutableArray array];
    for(NSNumber *axis in axes)[shape addObject:source.shape[axis.unsignedIntegerValue]];
    if(TensorPermutationIsReshape(source,axes)||!source.count)return [source reshape:shape];
    if(source.factored&&[source.shape isEqual:source.indexShape]) {
        NSMutableArray *strides=[NSMutableArray array];NSMutableDictionary *maps=[NSMutableDictionary dictionary];
        for(NSUInteger d=0;d<axes.count;d++) {
            NSNumber *axis=axes[d];[strides addObject:source.strides[axis.unsignedIntegerValue]];
            if(source.maps[axis])maps[@(d)]=source.maps[axis];
        }
        return [MetalTensorLayout shape:shape strides:strides offset:source.offset maps:maps indices:nil];
    }
    NSArray *old=TensorStrides(source.shape),*coordinates=TensorDigits(shape);NSMutableArray *terms=[NSMutableArray array];
    for(NSUInteger d=0;d<axes.count;d++)[terms addObject:TensorScale(coordinates[d],[old[[axes[d] unsignedIntegerValue]] integerValue])];
    return TensorCompose(source,shape,TensorSum(terms));
}

// Placement stays in the same relation algebra as selection and inversion.
// [a, p] -> [b, q] means position(b) - position(a) = q - p.
static isl_map *TensorSplitRelation(isl_set *set,unsigned inputs) {
    unsigned outputs=isl_set_dim(set,isl_dim_set)-inputs;
    return isl_map_move_dims(isl_map_from_domain(set),isl_dim_out,0,isl_dim_in,inputs,outputs);
}
static isl_map *TensorEquations(MetalTensorLayout *values,isl_pw_aff *group,isl_pw_aff *position,isl_map *roots) {
    isl_ctx *ctx=isl_pw_aff_get_ctx(group);
    // A changed logical shape can expose affine axes in an existing index
    // expression. Use the shared exact normalization before joining fields.
    values=[values shaped:values.shape coordinates:values.coordinates];
    isl_pw_multi_aff *groups=isl_pw_multi_aff_from_pw_aff(group);
    isl_pw_multi_aff *first=isl_pw_multi_aff_from_map(isl_map_lexmin(isl_map_reverse(
        isl_map_from_pw_multi_aff(isl_pw_multi_aff_copy(groups)))));
    first=isl_pw_multi_aff_pullback_pw_multi_aff(first,groups);
    isl_pw_aff *anchor=isl_pw_aff_pullback_pw_multi_aff(MeshIndexFunction(ctx,values),isl_pw_multi_aff_copy(first));
    isl_pw_aff *delta=isl_pw_aff_sub(isl_pw_aff_pullback_pw_multi_aff(isl_pw_aff_copy(position),isl_pw_multi_aff_copy(first)),position);
    // Check the established representatives in the declared iteration order.
    // Permuting complete typed fields adds no equations when their roots and
    // relative positions already agree. This forward check needs no inverse
    // of the value permutation or projected value/anchor/displacement tuple.
    isl_map *current=isl_map_project_out(isl_map_fix_si(isl_map_copy(roots),isl_dim_in,1,0),isl_dim_in,1,1);
    isl_pw_multi_aff *observed=isl_pw_multi_aff_from_map(MeshIndexCompose(MeshIndexRelation(ctx,values),current));
    isl_pw_multi_aff *reference=isl_pw_multi_aff_pullback_pw_multi_aff(isl_pw_multi_aff_copy(observed),first);
    isl_pw_aff *r=isl_pw_multi_aff_get_pw_aff(observed,0),*a=isl_pw_multi_aff_get_pw_aff(reference,0);
    isl_pw_aff *d=isl_pw_aff_sub(isl_pw_multi_aff_get_pw_aff(observed,1),isl_pw_multi_aff_get_pw_aff(reference,1));
    BOOL implied=isl_pw_aff_is_equal(r,a)==isl_bool_true&&isl_pw_aff_is_equal(d,delta)==isl_bool_true;
    isl_pw_aff_free(r);isl_pw_aff_free(a);isl_pw_aff_free(d);
    isl_pw_multi_aff_free(observed);isl_pw_multi_aff_free(reference);
    if(implied){isl_pw_aff_free(anchor);isl_pw_aff_free(delta);return isl_map_empty(isl_map_get_space(roots));}
    isl_pw_multi_aff *relative=isl_pw_multi_aff_flat_range_product(
        isl_pw_multi_aff_from_pw_aff(MeshIndexFunction(ctx,values)),
        isl_pw_multi_aff_flat_range_product(isl_pw_multi_aff_from_pw_aff(anchor),isl_pw_multi_aff_from_pw_aff(delta)));
    isl_set *pairs=isl_set_insert_dims(isl_map_range(isl_map_from_pw_multi_aff(relative)),isl_dim_set,1,1);
    pairs=isl_set_apply(pairs,isl_map_read_from_str(ctx,"{ [v,p,b,d] -> [v,p,b,p+d] }"));
    isl_map *edges=TensorSplitRelation(pairs,2);
    return MeshIndexNormalize(isl_map_union(isl_map_copy(edges),isl_map_reverse(edges)));
}
static isl_map *TensorJoinEquations(isl_map *roots,isl_map *fields,NSUInteger count,isl_map *forbidden) {
    // Each root is a function from a byte and displacement to its component.
    // Relabel the original field edges, hook to a smaller neighbor, and
    // compress that function. There is no need to construct every byte pair.
    while(roots) {
        isl_map *links=MeshIndexNormalize(isl_map_apply_range(isl_map_apply_range(
            isl_map_reverse(isl_map_copy(roots)),isl_map_copy(fields)),isl_map_copy(roots)));
        if(!links||isl_map_is_disjoint(links,forbidden)!=isl_bool_true) {
            isl_map_free(links);isl_map_free(roots);roots=NULL;break;
        }
        if(isl_map_is_identity(links)==isl_bool_true){isl_map_free(links);break;}
        links=isl_map_union(links,isl_set_identity(isl_map_range(isl_map_copy(roots))));
        isl_map *hook=isl_map_lexmin(links);
        // Root identities decrease along every nontrivial edge. A finite
        // component has no longer simple path than its declared byte count.
        for(NSUInteger length=1;hook&&length<count;) {
            isl_map *next=MeshIndexCompose(isl_map_copy(hook),isl_map_copy(hook));
            isl_bool same=isl_map_is_equal(next,hook);isl_map_free(hook);hook=next;
            if(same==isl_bool_error){isl_map_free(hook);hook=NULL;break;}
            if(same==isl_bool_true)break;
            length=length>count/2?count:length*2;
        }
        roots=MeshIndexNormalize(MeshIndexCompose(roots,hook));
        // Distinct bytes must retain distinct positions within a component.
        // Check the zero-displacement slice, rather than all translated pairs.
        isl_map *positions=isl_map_fix_si(isl_map_copy(roots),isl_dim_in,1,0);
        isl_bool distinct=isl_map_is_injective(positions);isl_map_free(positions);
        if(distinct!=isl_bool_true){isl_map_free(roots);roots=NULL;}
    }
    isl_map_free(fields);return roots;
}
/* Whether a typed value's elements already lie in place: each element's `width` positions consecutive ordinals, its
   first a multiple of width. Read from the value's strides (MetalTensorFactored): the flat index's axes below `width`
   dense (each stride its flat weight), every axis above it and the origin multiples of width. */
static BOOL TensorElementsInPlace(MetalTensorLayout *value,NSInteger width) {
    MetalTensorLayout *strided=MetalTensorFactored(value);if(!strided)return NO;
    NSInteger weight=1;
    for(NSUInteger d=strided.indexShape.count;d-->0;) {
        const NSInteger extent=[strided.indexShape[d] integerValue],stride=[strided.strides[d] integerValue];
        if(extent==1)continue;
        if(weight>=width){if(stride%width)return NO;}
        else if(weight*extent<=width){if(stride!=weight)return NO;}
        // an axis across the element boundary: dense below it, and its steps above it are then width apart
        else if(width%weight||extent%(width/weight)||stride!=weight)return NO;
        weight*=extent;
    }
    return strided.offset%width==0;
}
static NSInteger TensorValueAt(MetalTensorLayout *layout,NSArray<NSData *> *programs,NSInteger *stack,NSInteger at);
static MetalTensorLayout *TensorWritten(NSArray *shape,id index,NSInteger low,NSInteger high);
// A layout's positions at every flat index, integers only (its compiled coordinates and their strides or maps).
static NSMutableData *TensorPositions(MetalTensorLayout *layout) {
    NSMutableArray *programs=[NSMutableArray array];NSUInteger longest=1;
    for(id coordinate in layout.coordinates) {
        NSMutableData *program=[NSMutableData data];TensorCompile(coordinate,program);[programs addObject:program];
        longest=MAX(longest,program.length/sizeof(struct TensorProgramStep));
    }
    NSInteger stack[longest+1];NSMutableData *values=[NSMutableData dataWithLength:layout.count*sizeof(NSInteger)];
    NSInteger *out=values.mutableBytes;
    for(NSUInteger i=0;i<layout.count;i++)out[i]=TensorValueAt(layout,programs,stack,(NSInteger)i);
    return values;
}
/* Positions read back as consecutive mixed-radix strided pieces, each proved over its whole range: from a piece's first
   position, its innermost axis the positions on one line, each next axis the whole blocks below it repeated at one step.
   One piece is a factored layout; several, a select over their ranges. */
static MetalTensorLayout *TensorPiecewise(const NSInteger *values,NSUInteger count) {
    NSMutableArray *pieces=[NSMutableArray array];
    for(NSUInteger at=0;at<count;) {
        NSMutableArray *extents=[NSMutableArray array],*steps=[NSMutableArray array];NSUInteger block=1;
        while(at+block<count) {
            const NSInteger step=values[at+block]-values[at];NSUInteger e=1;
            for(;at+(e+1)*block<=count;e++) {
                BOOL same=YES;
                for(NSUInteger j=0;same&&j<block;j++)same=values[at+e*block+j]==values[at+j]+(NSInteger)e*step;
                if(!same)break;
            }
            if(e==1)break;
            [extents insertObject:@(e) atIndex:0];[steps insertObject:@(step) atIndex:0];block*=e;
        }
        [pieces addObject:@[@(at),@(block),extents.count?extents:@[@1],steps.count?steps:@[@0]]];at+=block;
    }
    if(pieces.count==1)return [[MetalTensorLayout shape:pieces[0][2] strides:pieces[0][3] offset:values[0] maps:@{} indices:nil]
        reshape:@[@(count)]];
    NSInteger low=values[0],high=values[0];for(NSUInteger i=0;i<count;i++){low=MIN(low,values[i]);high=MAX(high,values[i]);}
    id index=nil;
    for(NSArray *piece in [pieces reverseObjectEnumerator]) {
        const NSInteger at=[piece[0] integerValue];NSArray *extents=piece[2],*steps=piece[3];
        id local=TensorSum(@[@"i",@(-at)]);NSMutableArray *terms=[NSMutableArray arrayWithObject:@(values[at])];NSInteger weight=1;
        for(NSUInteger k=extents.count;k-->0;) {
            [terms addObject:TensorScale(TensorQuotient(TensorQuotient(local,weight,NO),[extents[k] integerValue],YES),[steps[k] integerValue])];
            weight*=[extents[k] integerValue];
        }
        id value=TensorSum(terms);
        index=index?@[@"select",@[@"lt",@"i",@(at+[piece[1] integerValue])],value,index]:value;
    }
    return TensorWritten(@[@(count)],index,low,high);
}
/* No consumer use, and one typed value whose elements (width W) tile the ordinals (fixed caller pages united into the
   family: placement.Placements.solve's typed `pages`): the packing joins each element's positions into one component
   and places the components in order of their least ordinals, each at the next W-aligned position, its positions in
   element order (the graph module's mesh_graph_tile_slots). Every other typed value's elements must lie inside one tiling element at consecutive element positions
   aligned to their width (dividing W), so they join nothing more. The placement is read back as strided pieces
   (TensorPiecewise). nil where any of that does not hold. Added by the layout algebra session: with the caller's pages a
   permutation of the family's ordinals, the packing's isl walk outlived the 8 s bound on every rank, row-major and
   units first (d-r4, d-rec3, 2026-11-02). That the caller's pages enter placement as fixed bindings at all is the
   page-map change's (design/page-maps-2026-11-02.md). */
static MetalTensorLayout *TensorPlacedOnTiles(NSUInteger count,NSArray<MetalTensorLayout *> *uses,
    NSArray<MetalTensorLayout *> *values,NSArray<NSNumber *> *widths) {
    for(MetalTensorLayout *use in uses)if(use.count)return nil;
    NSInteger tile=-1;NSUInteger width=1;
    for(NSUInteger k=0;k<values.count;k++)if(values[k].count==count&&widths[k].unsignedIntegerValue>width&&
        count%widths[k].unsignedIntegerValue==0){tile=(NSInteger)k;width=widths[k].unsignedIntegerValue;}
    if(tile<0)return nil;
    // the tiles' order and slots are the graph module's (mesh-graph.h mesh_graph_tile_slots)
    NSMutableData *tiled=TensorPositions(values[(NSUInteger)tile]);
    NSMutableData *placedData=[NSMutableData dataWithLength:count*sizeof(NSInteger)];NSInteger *placed=placedData.mutableBytes;
    if(mesh_graph_tile_slots(count,width,(const int64_t *)tiled.bytes,(int64_t *)placed))return nil;
    for(NSUInteger k=0;k<values.count;k++) {
        const NSUInteger w=widths[k].unsignedIntegerValue;
        if(w==1||(NSInteger)k==tile)continue;
        if(width%w)return nil;
        NSMutableData *positions=TensorPositions(values[k]);const NSInteger *v=positions.bytes;
        for(NSUInteger e=0;e<values[k].count;e+=w) {
            if(v[e]<0||v[e]>=(NSInteger)count)return nil;
            const NSInteger first=placed[v[e]];
            if(first%(NSInteger)w)return nil;
            for(NSUInteger t=1;t<w;t++)if(v[e+t]<0||v[e+t]>=(NSInteger)count||placed[v[e+t]]!=first+(NSInteger)t)return nil;
        }
    }
    return TensorPiecewise(placed,count);
}
MetalTensorLayout *MetalTensorPlace(NSUInteger count,NSArray<MetalTensorLayout *> *uses,
    NSArray<MetalTensorLayout *> *targets,NSArray<NSArray<NSNumber *> *> *footprints,
    NSArray<MetalTensorLayout *> *values,NSArray<NSNumber *> *widths) {
    NSCParameterAssert(uses.count==targets.count&&uses.count==footprints.count&&values.count==widths.count);
    if(!count)return [MetalTensorLayout shape:@[@0] strides:@[@1] offset:0 maps:@{} indices:nil];
    /* No consumer use and every typed value's elements in place (TensorElementsInPlace): the components the field
       equations join are intervals tiling the ordinals in order, each at an aligned first position, so the packing below
       places each at its own ordinal, the identity. Added by the layout algebra session: with the LSTM's h stored units
       first a single typed field's packing went through isl's transitive closure of the cursor recurrence over a periodic
       root set, 41 s for a [2, 2, 4] half h and past 3 minutes at [2, 8, 1024] (desk replay, 2026-11-02), row-major
       0.01 s. What must stay: the same placement as the packing. */
    BOOL inPlace=YES;
    for(MetalTensorLayout *use in uses)inPlace&=!use.count;
    for(NSUInteger k=0;inPlace&&k<values.count;k++) {
        const NSInteger width=widths[k].integerValue;
        inPlace=width==1||TensorElementsInPlace(values[k],width);
    }
    if(inPlace)return [MetalTensorLayout shape:@[@(count)] strides:@[@1] offset:0 maps:@{} indices:nil];
    MetalTensorLayout *tiled=TensorPlacedOnTiles(count,uses,values,widths);
    if(tiled)return tiled;
    isl_ctx *ctx=isl_ctx_alloc();MetalTensorLayout *result=nil;
    isl_map *equations=NULL,*roots=NULL,*extents=NULL,*allowed=NULL,*base=NULL,*walk=NULL,*placed=NULL;
    isl_map *forbidden=isl_map_read_from_str(ctx,
        "{ [i,p] -> [j,q] : (i = j and p != q) or (i != j and p = q) }");
    isl_set *ids=isl_set_read_from_str(ctx,[[NSString stringWithFormat:@"{ [i] : 0 <= i < %lu }",(unsigned long)count] UTF8String]);
    isl_set *rootIds=NULL,*states=NULL;
    isl_pw_aff *low=NULL,*high=NULL;
    equations=isl_map_read_from_str(ctx,[[NSString stringWithFormat:
        @"{ [i,p] -> [i,p] : 0 <= i < %lu }",(unsigned long)count] UTF8String]);
    // The typed field relation supplies adjacency and alignment for every
    // element, including fields reached through differently typed aliases.
    NSMutableArray *order=[NSMutableArray array];
    for(NSUInteger k=0;k<values.count;k++)[order addObject:@(k)];
    [order sortUsingComparator:^NSComparisonResult(NSNumber *a,NSNumber *b) {
        return [widths[b.unsignedIntegerValue] compare:widths[a.unsignedIntegerValue]];
    }];
    for(NSNumber *item in order) {
        NSUInteger k=item.unsignedIntegerValue;
        MetalTensorLayout *value=values[k];NSUInteger width=widths[k].unsignedIntegerValue;
        NSCParameterAssert(width&&value.count%width==0);
        if(width==1)continue; // A one-unit field adds no relative-position equation.
        isl_map *fields=TensorEquations(value,
            MeshIndexExpression(ctx,@[@"div",@"i",@(width)],0,value.count),
            MeshIndexExpression(ctx,@[@"mod",@"i",@(width)],0,value.count),equations);
        equations=TensorJoinEquations(equations,fields,count,forbidden);
        if(!equations)goto finished;
    }
    for(NSUInteger k=0;k<uses.count;k++) {
        MetalTensorLayout *use=uses[k],*target=targets[k];NSCParameterAssert(use.count==target.count);
        if(!use.count)continue;
        equations=TensorJoinEquations(equations,TensorEquations(use,
            MeshIndexExpression(ctx,@0,0,use.count),MeshIndexFunction(ctx,target),equations),count,forbidden);
        if(!equations)goto finished;
    }
    equations=isl_map_detect_equalities(isl_map_coalesce(equations));
    {
        isl_set *domain=isl_set_project_out(isl_map_domain(isl_map_copy(equations)),isl_dim_set,1,1);
        isl_bool covered=isl_set_is_equal(domain,ids);isl_set_free(domain);
        if(covered!=isl_bool_true)goto finished;
    }
    // Each value names its least root and its displacement from that root.
    roots=isl_map_project_out(isl_map_fix_si(isl_map_copy(equations),isl_dim_in,1,0),isl_dim_in,1,1);
    roots=isl_map_apply_range(isl_map_lexmin(roots),isl_map_read_from_str(ctx,"{ [r,d] -> [r,-d] }"));
    roots=isl_map_coalesce(roots);
    extents=TensorSplitRelation(isl_map_range(isl_map_copy(roots)),1);
    rootIds=isl_map_domain(isl_map_copy(extents));
    // API write footprints reserve their padding even where no logical field
    // names that storage. They enlarge the same component's physical extent.
    for(NSUInteger k=0;k<uses.count;k++)if(uses[k].count&&[footprints[k][1] unsignedIntegerValue]) {
        NSInteger shift=[footprints[k][0] integerValue]-[targets[k] index:0],extent=[footprints[k][1] integerValue];
        isl_set *anchor=isl_map_range(isl_map_fix_val(isl_map_copy(roots),isl_dim_in,0,
            isl_val_int_from_si(ctx,[uses[k] index:0])));
        isl_map *reserve=isl_map_read_from_str(ctx,[[NSString stringWithFormat:
            @"{ [r,d] -> [r,x] : d + %ld <= x < d + %ld + %ld }",(long)shift,(long)shift,(long)extent] UTF8String]);
        extents=isl_map_union(extents,TensorSplitRelation(isl_set_apply(anchor,reserve),1));
    }
    low=isl_map_dim_min(isl_map_copy(extents),0);high=isl_map_dim_max(isl_map_copy(extents),0);
    allowed=isl_map_intersect_domain(isl_map_read_from_str(ctx,"{ [r] -> [b] }"),isl_set_copy(rootIds));
    for(NSUInteger k=0;k<values.count;k++) {
        NSUInteger width=widths[k].unsignedIntegerValue;
        if(width==1)continue; // Every integer position has unit alignment.
        isl_set *starts=isl_set_read_from_str(ctx,[[NSString stringWithFormat:@"{ [i] : i mod %lu = 0 }",(unsigned long)width] UTF8String]);
        starts=isl_map_range(isl_map_intersect_domain(MeshIndexRelation(ctx,values[k]),starts));
        starts=isl_set_apply(starts,isl_map_copy(roots));
        isl_map *unaligned=isl_map_read_from_str(ctx,[[NSString stringWithFormat:
            @"{ [r,d] -> [r,b] : (b + d) mod %lu != 0 }",(unsigned long)width] UTF8String]);
        allowed=isl_map_subtract(allowed,TensorSplitRelation(isl_set_apply(starts,unaligned),1));
    }
    // Each API constrains the first address of its declared footprint. This is independent
    // of both element alignment and the extent of any padding it may access.
    for(NSUInteger k=0;k<uses.count;k++)if(uses[k].count) {
        NSUInteger alignment=[footprints[k][2] unsignedIntegerValue];
        NSCParameterAssert(alignment);
        isl_set *anchor=isl_map_range(isl_map_fix_val(isl_map_copy(roots),isl_dim_in,0,
            isl_val_int_from_si(ctx,[uses[k] index:0])));
        isl_map *unaligned=isl_map_read_from_str(ctx,[[NSString stringWithFormat:
            @"{ [r,d] -> [r,b] : (b + d - %ld) mod %lu != 0 }",
            (long)([targets[k] index:0]-[footprints[k][0] integerValue]),(unsigned long)alignment] UTF8String]);
        allowed=isl_map_subtract(allowed,TensorSplitRelation(isl_set_apply(anchor,unaligned),1));
    }
    {
        isl_set *domain=isl_map_domain(isl_map_copy(allowed));
        isl_bool covered=isl_set_is_equal(domain,rootIds);isl_set_free(domain);
        if(covered!=isl_bool_true)goto finished;
    }
    // Given the preceding end c, choose the first aligned base whose lowest
    // occupied slot is at least c. Consecutive roots share that same recurrence;
    // no global stride or per-component padding allowance wastes tensor storage.
    {
        isl_pw_aff *bound=isl_pw_aff_sub(isl_pw_aff_read_from_str(ctx,"{ [r,c] -> [c] }"),
            isl_pw_aff_insert_dims(isl_pw_aff_copy(low),isl_dim_in,1,1));
        base=isl_map_apply_range(isl_map_from_pw_aff(bound),isl_map_read_from_str(ctx,"{ [a] -> [b] : b >= a }"));
        base=isl_map_lexmin(isl_map_intersect(base,isl_map_insert_dims(isl_map_copy(allowed),isl_dim_in,1,1)));
        isl_pw_aff *end=isl_pw_aff_add(isl_map_dim_min(isl_map_copy(base),0),
            isl_pw_aff_insert_dims(isl_pw_aff_copy(high),isl_dim_in,1,1));
        end=isl_pw_aff_add_constant_val(end,isl_val_one(ctx));
        isl_map *next=isl_map_from_domain_and_range(isl_set_copy(rootIds),isl_set_copy(rootIds));
        next=isl_map_lexmin(isl_map_intersect(next,isl_map_read_from_str(ctx,"{ [r] -> [s] : s > r }")));
        walk=isl_map_flat_range_product(isl_map_insert_dims(next,isl_dim_in,1,1),isl_map_from_pw_aff(end));
        states=isl_set_apply(isl_set_lexmin(isl_set_copy(rootIds)),isl_map_read_from_str(ctx,"{ [r] -> [r,0] }"));
        // Only states reachable from the declared first component and zero
        // cursor participate in packing; unrelated cursor phases do not.
        states=MeshIndexReachable(walk,states,count);walk=NULL;
        isl_map *bases=isl_map_project_out(isl_map_intersect_domain(isl_map_copy(base),isl_set_copy(states)),isl_dim_in,1,1);
        isl_map *which=isl_map_project_out(isl_map_copy(roots),isl_dim_out,1,1);
        isl_pw_aff *position=isl_map_dim_min(isl_map_apply_range(which,bases),0);
        position=isl_pw_aff_add(position,isl_map_dim_min(isl_map_copy(roots),1));
        placed=isl_map_coalesce(isl_map_from_pw_aff(position));
    }
    if(isl_map_is_injective(placed)!=isl_bool_true)goto finished;
    for(NSUInteger k=0;k<uses.count;k++)if(uses[k].count&&[footprints[k][1] unsignedIntegerValue]) {
        isl_set *foreign=isl_set_subtract(isl_set_copy(ids),isl_map_range(MeshIndexRelation(ctx,uses[k])));
        foreign=isl_set_apply(foreign,isl_map_copy(placed));
        isl_set *anchor=isl_map_range(isl_map_fix_val(isl_map_copy(placed),isl_dim_in,0,
            isl_val_int_from_si(ctx,[uses[k] index:0])));
        NSInteger shift=[footprints[k][0] integerValue]-[targets[k] index:0],extent=[footprints[k][1] integerValue];
        isl_set *reserve=isl_set_apply(anchor,isl_map_read_from_str(ctx,[[NSString stringWithFormat:
            @"{ [p] -> [q] : p + %ld <= q < p + %ld + %ld }",(long)shift,(long)shift,(long)extent] UTF8String]));
        isl_bool disjoint=isl_set_is_disjoint(foreign,reserve);isl_set_free(foreign);isl_set_free(reserve);
        if(disjoint!=isl_bool_true)goto finished;
    }
    result=TensorFunctionLayout(isl_map_dim_min(isl_map_copy(placed),0),@[@(count)]);
finished:;
    NSString *error=isl_ctx_last_error(ctx)==isl_error_none?nil:@(isl_ctx_last_error_msg(ctx));
    isl_map_free(equations);isl_map_free(roots);isl_map_free(extents);isl_map_free(allowed);isl_map_free(forbidden);
    isl_map_free(base);isl_map_free(walk);isl_map_free(placed);
    isl_set_free(ids);isl_set_free(rootIds);isl_set_free(states);isl_pw_aff_free(low);isl_pw_aff_free(high);isl_ctx_free(ctx);
    if(error)[NSException raise:NSInvalidArgumentException format:@"storage placement: %@",error];
    return result;
}

static NSDictionary *TensorPlaceDemands(NSUInteger count,NSArray<NSDictionary *> *demands,
    NSArray<MetalTensorLayout *> *values,NSArray<NSNumber *> *widths) {
    NSMutableArray *uses=[NSMutableArray array],*targets=[NSMutableArray array],*footprints=[NSMutableArray array];
    for(NSDictionary *demand in demands)if(![demand[@"flexible"] boolValue]) {
        [uses addObject:demand[@"use"]];[targets addObject:demand[@"target"]];[footprints addObject:demand[@"footprint"]];
    }
    MetalTensorLayout *placed=MetalTensorPlace(count,uses,targets,footprints,values,widths);
    if(!placed)return nil;
    MetalTensorLayout *domain=[MetalTensorLayout shape:@[@(count)] strides:@[@1] offset:0 maps:@{} indices:nil];
    BOOL changed=NO;
    for(NSDictionary *demand in demands)if([demand[@"flexible"] boolValue]) {
        NSUInteger width=[demand[@"width"] unsignedIntegerValue];
        MetalTensorLayout *view=[[(MetalTensorLayout *)demand[@"use"] rebase:domain onto:placed] elements:width];
        MetalTensorLayout *target=[view requirements:demand[@"requirements"]];
        changed|=![view equivalent:target];
        [uses addObject:demand[@"use"]];[targets addObject:[target components:width]];[footprints addObject:demand[@"footprint"]];
    }
    if(changed)placed=MetalTensorPlace(count,uses,targets,footprints,values,widths);
    if(!placed)return nil;
    NSUInteger extent=0,alignment=1;
    NSMutableArray *alignments=[NSMutableArray arrayWithArray:widths];
    for(NSUInteger k=0;k<uses.count;k++) {
        [alignments addObject:footprints[k][2]];
        if([uses[k] count]&&[footprints[k][1] unsignedIntegerValue]) {
            NSInteger first=[[(MetalTensorLayout *)uses[k] rebase:domain onto:placed] index:0]-[targets[k] index:0]+[footprints[k][0] integerValue];
            extent=MAX(extent,first+[footprints[k][1] unsignedIntegerValue]);
        }
    }
    for(NSNumber *value in alignments) {
        NSUInteger a=alignment,b=value.unsignedIntegerValue;NSCParameterAssert(b);
        while(b){NSUInteger r=a%b;a=b;b=r;}alignment=alignment/a*value.unsignedIntegerValue;
    }
    return @{@"layout":placed,@"extent":@(extent),@"alignment":@(alignment)};
}

NSDictionary *MetalTensorPlacePreferred(NSUInteger count,NSArray<NSDictionary *> *hard,
    NSArray<NSArray<NSDictionary *> *> *groups,NSArray<NSNumber *> *preferences,
    NSArray<MetalTensorLayout *> *values,NSArray<NSNumber *> *widths) {
    NSCParameterAssert(groups.count==preferences.count);
    NSMutableArray *order=[NSMutableArray array],*all=[NSMutableArray arrayWithArray:hard];
    for(NSUInteger k=0;k<groups.count;k++) {
        double weight=[preferences[k] doubleValue];
        if(!isfinite(weight)||weight<0)[NSException raise:NSInvalidArgumentException format:@"layout preferences must be finite and nonnegative"];
        [order addObject:@(k)];[all addObjectsFromArray:groups[k]];
    }
    NSDictionary *solved=TensorPlaceDemands(count,all,values,widths);
    NSMutableArray *accepted=[NSMutableArray array],*conflicts=[NSMutableArray array];
    if(solved)[accepted addObjectsFromArray:order];
    else {
        solved=TensorPlaceDemands(count,hard,values,widths);if(!solved)return nil;
        [order sortUsingComparator:^NSComparisonResult(NSNumber *left,NSNumber *right) {
            NSComparisonResult compared=[preferences[right.unsignedIntegerValue] compare:preferences[left.unsignedIntegerValue]];
            return compared==NSOrderedSame?[left compare:right]:compared;
        }];
        NSMutableArray *held=[NSMutableArray arrayWithArray:hard];
        for(NSNumber *index in order) {
            NSArray *candidate=[held arrayByAddingObjectsFromArray:groups[index.unsignedIntegerValue]];
            NSDictionary *next=TensorPlaceDemands(count,candidate,values,widths);
            if(next){solved=next;[held addObjectsFromArray:groups[index.unsignedIntegerValue]];[accepted addObject:index];}
            else [conflicts addObject:[accepted arrayByAddingObject:index]];
        }
    }
    NSMutableDictionary *result=[[solved mutableCopy] autorelease];
    result[@"accepted"]=accepted;result[@"conflicts"]=conflicts;return result;
}

// Bind configured names into the existing integer term type. The result goes
// straight to isl; this is neither a second algebra nor a runtime evaluator.
static id TensorBind(id expression,NSDictionary *values) {
    if(TensorConstant(expression))return expression;
    if([expression isKindOfClass:NSString.class]) {
        id value=values[expression];
        if(!value)[NSException raise:NSInvalidArgumentException format:@"missing storage operand %@",expression];
        return value;
    }
    NSMutableArray *terms=[NSMutableArray arrayWithObject:expression[0]];
    for(NSUInteger k=1;k<[expression count];k++)[terms addObject:TensorBind(expression[k],values)];
    return terms;
}
static NSMutableDictionary *TensorParameters(NSArray *shape,NSDictionary *parameters) {
    NSMutableDictionary *values=[[parameters mutableCopy] autorelease];values[@"count"]=@(TensorCount(shape));
    for(NSUInteger d=0;d<shape.count;d++)values[[NSString stringWithFormat:@"extent.%lu",(unsigned long)d]]=shape[d];
    return values;
}
NSInteger MetalTensorConstant(id expression,NSArray *shape,NSDictionary *parameters) {
    isl_ctx *ctx=isl_ctx_alloc();
    isl_pw_aff *value=MeshIndexExpression(ctx,TensorBind(expression,TensorParameters(shape,parameters)),0,1);
    isl_val *constant=isl_pw_aff_min_val(value);NSInteger result=isl_val_get_num_si(constant);
    isl_val_free(constant);isl_ctx_free(ctx);return result;
}

// A declared affine producer's axes by descending stride magnitude, and its lowest address (mesh fc6d8716).
static NSArray *TensorInverseAxes(MetalTensorLayout *declared,NSInteger *origin) {
    NSMutableArray *axes=[NSMutableArray array];NSInteger low=declared.offset;
    for(NSUInteger d=0;d<declared.shape.count;d++) {
        const NSInteger n=[declared.shape[d] integerValue],s=[declared.strides[d] integerValue];
        if(n>1&&s){[axes addObject:@(d)];low+=MIN(0,(n-1)*s);}
    }
    [axes sortUsingComparator:^NSComparisonResult(NSNumber *a,NSNumber *b) {
        NSInteger x=labs([declared.strides[a.unsignedIntegerValue] integerValue]),y=labs([declared.strides[b.unsignedIntegerValue] integerValue]);
        return x>y?NSOrderedAscending:x<y?NSOrderedDescending:NSOrderedSame;
    }];
    *origin=low;return axes;
}
// The rebases whose composition is closed-form, as mesh fc6d8716 made them (the last commit cp passed at): the
// placement differs from the declaration by its offset alone; the view is the declaration reshaped; the view shares
// the declaration's factors; or both are affine and the declaration is a dense bijection onto its address range, so
// each view address has one preimage. Each composes `placed` by index terms (TensorCompose), keeping its strides and
// maps, so a mapped placement stays mapped. nil where none applies: the relation composition (rebase) is the general
// case, whose index functions take the solver once per affine piece wherever they are read (TensorExpressionPiece).
// Whether every address of an affine view lies in an affine declaration's image where that image is one dense interval
// (its strides a permutation of the dense steps): 1 or 0 from the two address ranges, -1 where neither is that form.
// Added by the LSTM demonstration session: the offset-only rebase below proved containment with a relation preimage,
// 583 of rebase:'s 1032 samples in a four-step LSTM declaration on m5ult2 (2026-11-01).
static int TensorAffineWithin(MetalTensorLayout *view,MetalTensorLayout *declared) {
    if(!declared.affine||!view.count||!declared.count)return -1;
    NSInteger low;NSArray *axes=TensorInverseAxes(declared,&low);NSUInteger width=1;BOOL dense=YES;
    for(NSNumber *axis in [axes reverseObjectEnumerator]) {
        NSUInteger d=axis.unsignedIntegerValue;dense&=(NSUInteger)labs([declared.strides[d] integerValue])==width;
        width*=[declared.shape[d] unsignedIntegerValue];
    }
    if(!dense||width!=declared.count)return -1;
    /* the view's least and greatest positions, positions it has: from its strides where it is factored without maps,
       else isl's image bounds. Changed by the layout algebra session: any view that was not affine (a factored view of
       more index axes than logical ones, a flat-term view) proved its containment by a relation preimage whose points
       isl enumerated, about 28 ms for each 32 KiB view placement.selected rebases onto its compact root (desk,
       2026-11-02). What must stay: 1 exactly where every position lies in the declaration's interval. */
    NSInteger first=view.offset,last=view.offset;
    if(view.factored&&!view.maps.count) {
        for(NSUInteger d=0;d<view.indexShape.count;d++) {
            const NSInteger span=([view.indexShape[d] integerValue]-1)*[view.strides[d] integerValue];
            first+=MIN(0,span);last+=MAX(0,span);
        }
    } else {const struct MetalTensorSpan span=view.storageSpan;first=span.offset;last=span.offset+(NSInteger)span.count-1;}
    return first>=low&&last<low+(NSInteger)width;
}
static MetalTensorLayout *TensorClosedRebase(MetalTensorLayout *view,MetalTensorLayout *declared,MetalTensorLayout *placed) {
    if([[declared shift:-declared.offset] isEqual:[placed shift:-placed.offset]]) {
        const int within=TensorAffineWithin(view,declared);
        if(within==1||(within<0&&[view preimage:declared].count==view.count))return [view shift:placed.offset-declared.offset];
    }
    NSArray *shape=view.shape;
    if(view.count==declared.count&&[[declared reshape:shape] isEqual:view])return [placed reshape:shape];
    if(declared.factored&&view.offset==declared.offset&&[view.indexShape isEqual:declared.indexShape]&&
        [view.strides isEqual:declared.strides]&&[view.maps isEqual:declared.maps]) {
        NSArray *digits=view.coordinates,*steps=TensorStrides(view.indexShape);NSMutableArray *terms=[NSMutableArray array];
        for(NSUInteger d=0;d<digits.count;d++)[terms addObject:TensorScale(digits[d],[steps[d] integerValue])];
        return TensorCompose(placed,shape,TensorSum(terms));
    }
    if(!view.affine||!declared.affine)return nil;
    NSInteger low;NSArray *axes=TensorInverseAxes(declared,&low);NSUInteger width=1;BOOL dense=YES;
    for(NSNumber *axis in [axes reverseObjectEnumerator]) {
        NSUInteger d=axis.unsignedIntegerValue;dense&=(NSUInteger)labs([declared.strides[d] integerValue])==width;
        width*=[declared.shape[d] unsignedIntegerValue];
    }
    dense&=width==declared.count;NSInteger first=view.offset,last=view.offset;
    NSArray *digits=TensorDigits(shape);NSMutableArray *terms=[NSMutableArray arrayWithObject:@(view.offset-low)];
    for(NSUInteger d=0;d<shape.count;d++) {
        NSInteger step=[view.strides[d] integerValue],span=([shape[d] integerValue]-1)*step;
        first+=MIN(0,span);last+=MAX(0,span);[terms addObject:TensorScale(digits[d],step)];
    }
    if(!dense||first<low||last>=low+(NSInteger)width)return nil;
    id address=TensorSum(terms);NSMutableArray *index=[NSMutableArray array];NSArray *steps=TensorStrides(declared.shape);
    for(NSNumber *axis in axes) {
        NSUInteger d=axis.unsignedIntegerValue;NSInteger step=[declared.strides[d] integerValue],n=[declared.shape[d] integerValue];
        id coordinate=TensorQuotient(TensorQuotient(address,labs(step),NO),n,YES);
        if(step<0)coordinate=TensorSum(@[@(n-1),TensorScale(coordinate,-1)]);
        [index addObject:TensorScale(coordinate,[steps[d] integerValue])];
    }
    return TensorCompose(placed,shape,TensorSum(index));
}
static BOOL TensorHasGeneral(id expression) {
    if(![expression isKindOfClass:NSArray.class])return NO;
    if(TensorGeneral(expression))return YES;
    for(NSUInteger k=1;k<[expression count];k++)if(TensorHasGeneral(expression[k]))return YES;
    return NO;
}
// A component axis's elements as mesh fc6d8716 made them: the first component's layout, each factor's stride or map
// divided by the component count. nil where the components are not each at the first's next address, a factor or the
// origin is not a whole number of elements, or a coordinate is a piecewise term; elements: then composes the relation.
static MetalTensorLayout *TensorClosedElements(MetalTensorLayout *layout,NSUInteger count,NSArray *shape) {
    if(layout.indexShape.count==1&&![layout.shape isEqual:layout.indexShape]) {
        // One index axis (placement's united pages: a mapped run of bytes reshaped into pages): the components'
        // adjacency and the elements' addresses are that axis's addresses at each position's coordinate, read as
        // integers. Added by the LSTM demonstration session: the general form below sliced the 32768-entry map once
        // per component, 7.25 s of a one-step LSTM lowering on the workstation for one call (2026-11-01).
        NSData *map=layout.maps[@0];const NSInteger *values=map.bytes,step=[layout.strides[0] integerValue];
        // The coordinate (piecewise: the pages' runs) compiled once and read at each position with integers (the five
        // ranks' pages are millions of bytes, where walking the expression at each one took most of a minute).
        NSMutableData *compiled=[NSMutableData data];TensorCompile(layout.coordinates[0],compiled);
        const struct TensorProgramStep *program=compiled.bytes;const NSUInteger length=compiled.length/sizeof *program;
        NSMutableData *scratch=[NSMutableData dataWithLength:(length+1)*sizeof(NSInteger)];NSInteger *stack=scratch.mutableBytes;
        const NSUInteger n=layout.count/count;
        NSMutableData *firsts=[NSMutableData dataWithLength:n*sizeof(NSInteger)];NSInteger *out=firsts.mutableBytes;
        for(NSUInteger x=0;x<n;x++) {
            NSInteger at=0;
            for(NSUInteger k=0;k<count;k++) {
                const NSInteger digit=TensorRun(program,length,(NSInteger)(x*count+k),stack),address=values?values[digit]:digit*step;
                if(!k)at=address;
                else if(address!=at+(NSInteger)k)return nil;
            }
            out[x]=at;
        }
        const NSInteger first=out[0],origin=layout.offset+first;
        if(origin%(NSInteger)count)return nil;
        for(NSUInteger x=0;x<n;x++) {
            if((out[x]-first)%(NSInteger)count)return nil;
            out[x]=(out[x]-first)/(NSInteger)count;
        }
        MetalTensorLayout *flat=[MetalTensorLayout shape:@[@(n)] strides:@[@(step/(NSInteger)count)] offset:origin/(NSInteger)count
                                                    maps:@{@0:firsts} indices:nil];
        return [flat reshape:shape];
    }
    MetalTensorLayout *factors=layout.factored&&[layout.indexShape.lastObject unsignedIntegerValue]==count?[layout reshape:layout.indexShape]:layout;
    const NSUInteger last=factors.shape.count-1;
    MetalTensorLayout *base=[factors element:last at:0];
    // On a factored axis each component is the first's next address exactly when the axis steps by 1 or its map rises
    // by 1, read from the axis itself. Added by the LSTM demonstration session: the loop built and compared count
    // layouts a call, 886 of elements:' 991 samples in a four-step LSTM declaration on m5ult2 (2026-11-01).
    // What must stay: the loop for an axis read through coordinates.
    if(factors.factored&&[factors.shape isEqual:factors.indexShape]) {
        NSData *map=factors.maps[@(last)];
        if(map) {
            const NSInteger *values=map.bytes;
            for(NSUInteger k=1;k<count;k++)if(values[k]-values[0]!=(NSInteger)k)return nil;
        } else if([factors.strides[last] integerValue]!=1)return nil;
    } else for(NSUInteger k=1;k<count;k++)if(![[factors element:last at:k] isEqual:[base shift:(NSInteger)k]])return nil;
    NSMutableArray *basis=[NSMutableArray array],*steps=[NSMutableArray array],*coordinates=[NSMutableArray array];
    NSMutableDictionary *maps=[NSMutableDictionary dictionary];NSInteger origin=base.offset;
    NSArray *digits=base.coordinates;
    for(NSUInteger d=0;d<base.indexShape.count;d++) {
        id coordinate=TensorBound(digits[d],base.count);NSData *data=base.maps[@(d)];const NSInteger *values=data.bytes;
        NSInteger step=[base.strides[d] integerValue];
        if(TensorHasGeneral(coordinate))return nil;
        if(TensorConstant(coordinate)) {
            NSInteger at=[coordinate integerValue];origin+=values?values[at]:at*step;continue;
        }
        NSUInteger axis=basis.count;[basis addObject:base.indexShape[d]];[coordinates addObject:coordinate];
        if(values) {
            NSMutableData *divided=[NSMutableData dataWithLength:data.length];NSInteger *out=divided.mutableBytes;
            origin+=values[0];
            for(NSUInteger k=0;k<data.length/sizeof *values;k++) {
                if((values[k]-values[0])%(NSInteger)count)return nil;
                out[k]=(values[k]-values[0])/(NSInteger)count;
            }
            maps[@(axis)]=divided;
        } else if(step%(NSInteger)count)return nil;
        [steps addObject:@(step/(NSInteger)count)];
    }
    if(origin%(NSInteger)count)return nil;
    MetalTensorLayout *result=[MetalTensorLayout shape:basis strides:steps offset:origin/(NSInteger)count maps:maps indices:nil];
    return [result shaped:shape coordinates:coordinates];
}

static NSInteger TensorSelectionExtent(id value) {
  if([value isKindOfClass:NSDictionary.class]&&value[@"parameter"])
    return [value[@"offset"]?:@0 longLongValue]+[value[@"scale"]?:@1 longLongValue]*TensorSelectionExtent(value[@"value"]);
  if([value isKindOfClass:NSArray.class]) {
    NSInteger product=1;for(id factor in value)product*=TensorSelectionExtent(factor);return product;
  }
  if(![value isKindOfClass:NSNumber.class])
    [NSException raise:NSInvalidArgumentException format:@"unbound selection extent %@",value];
  return [value longLongValue];
}
static id TensorSelectionIndex(NSInteger (^extent)(id),NSArray *shape,id expression,NSMutableDictionary *parameters,NSMutableDictionary *lookups) {
  if(![expression isKindOfClass:NSArray.class])return expression;
  NSString *op=expression[0];
  if([op isEqual:@"extent"])return @(extent(expression[1]));
  if([op isEqual:@"lookup"]) {
    if(lookups[expression])return lookups[expression];
    MetalTensorLayout *source=[MetalTensorLayout selection:expression[1] extent:extent];
    id term=TensorSelectionIndex(extent,shape,expression[2],parameters,lookups);
    MetalTensorLayout *indices=[MetalTensorLayout shape:shape index:term parameters:parameters];
    NSString *name=[NSString stringWithFormat:@"selection.%lu",(unsigned long)parameters.count];
    parameters[name]=[indices rebase:[MetalTensorLayout shape:source.shape strides:TensorStrides(source.shape) offset:0 maps:@{} indices:nil] onto:source];
    lookups[expression]=name;
    return name;
  }
  NSMutableArray *terms=[NSMutableArray arrayWithObject:op];
  for(NSUInteger k=1;k<[expression count];k++)[terms addObject:TensorSelectionIndex(extent,shape,expression[k],parameters,lookups)];
  return terms;
}

/* (first, width, step, repeats) runs joined where one continues another: an interval its next one (one interval), single
   positions at one step (one affine progression), a strided run its next row. Each run is i -> first + step
   floor(i / width) + i mod width over its positions, so the joined runs are the same function of i. Equal intervals
   wider than one stay separate pieces: as one strided run their relation's range would carry the floor isl must then
   eliminate (every page range of a transfer is projected from it), where separate pieces' ranges are intervals.
   Added by the layout algebra session: a page selection's spans (the pieces of one interval) were parsed into isl,
   their terms emitted and read back three times, 0.2 ms a single span and 4.5 ms eight (2026-11-01). What must stay:
   the same positions in the same order. */
struct TensorRunPart { NSInteger first,width,step,repeats; };
static NSArray<NSArray<NSNumber *> *> *TensorRunsJoined(NSArray<NSArray<NSNumber *> *> *runs) {
    NSMutableArray *joined=[NSMutableArray array];struct TensorRunPart last={0,0,0,0};
    for(NSArray *run in runs) {
        struct TensorRunPart next={[run[0] integerValue],[run[1] integerValue],[run[2] integerValue],[run[3] integerValue]};
        if(!next.width||!next.repeats)continue;
        if(next.repeats==1)next.step=0;
        else if(next.step==next.width){next.width*=next.repeats;next.repeats=1;next.step=0;}
        const BOOL alone=last.repeats==1,same=next.width==last.width;
        if(!last.repeats)last=next;
        else if(alone&&next.repeats==1&&next.first==last.first+last.width)last.width+=next.width;
        else if(alone&&same&&next.repeats==1&&next.width==1)last=(struct TensorRunPart){last.first,1,next.first-last.first,2};
        else if(!alone&&same&&next.first==last.first+last.repeats*last.step&&(next.repeats==1||next.step==last.step))
            last.repeats+=next.repeats;
        else if(alone&&same&&next.repeats>1&&next.first==last.first+next.step)
            last=(struct TensorRunPart){last.first,last.width,next.step,next.repeats+1};
        else {[joined addObject:@[@(last.first),@(last.width),@(last.step),@(last.repeats)]];last=next;}
    }
    if(last.repeats)[joined addObject:@[@(last.first),@(last.width),@(last.step),@(last.repeats)]];
    return joined;
}
// The joined runs' selection terms, in closed form where they are one run, else from isl.
static id TensorRunsIndex(NSArray<NSArray<NSNumber *> *> *runs) {
    if(runs.count==1) {
        const NSInteger first=[runs[0][0] integerValue],width=[runs[0][1] integerValue],step=[runs[0][2] integerValue];
        if([runs[0][3] integerValue]==1)return TensorSum(@[@(first),@"i"]);
        return TensorSum(@[@(first),TensorScale(TensorQuotient(@"i",width,NO),step),TensorQuotient(@"i",width,YES)]);
    }
    isl_ctx *ctx=isl_ctx_alloc();id index=MeshIndexTerms(MeshIndexRuns(ctx,runs));isl_ctx_free(ctx);
    return index;
}

static NSString *TensorStorageDeclaration(MetalTensorLayout *physical,MetalTensorLayout *source,NSString *name) {
    NSCParameterAssert([source.shape isEqual:physical.shape]);
    if(!physical.count)return [NSString stringWithFormat:@"inline long %@(long) { return 0; }\n",name];
    isl_ctx *ctx=isl_ctx_alloc();
    isl_pw_multi_aff *inverse=isl_map_lexmax_pw_multi_aff(isl_map_reverse(MeshIndexRelation(ctx,source)));
    id terms=MeshIndexTerms(isl_pw_aff_pullback_pw_multi_aff(MeshIndexFunction(ctx,physical),inverse));
    // Storage indices can have gaps and can start below zero. The emitted
    // expression is valid on the declared source image; it adds no validity guard.
    struct MeshIndexBounds bounds=MeshIndexImageBounds(MeshIndexRelation(ctx,source));isl_ctx_free(ctx);
    NSInteger low=bounds.low,high=bounds.high;
    terms=TensorSubstitute(terms,TensorSum(@[@"i",@(low)]));
    return [NSString stringWithFormat:@"inline long %@(long storage) { long i = storage - %ldl; return %@; }\n",
        name,(long)low,TensorExpressionSource(terms,MAX(1,high-low+1))];
}

@implementation MetalTensorLayout
+ (instancetype)selection:(NSDictionary *)declaration extent:(NSInteger (^)(id))extent {
  if(!extent)extent=^NSInteger(id value){return TensorSelectionExtent(value);};
  NSMutableArray *shape=[NSMutableArray array];for(id n in declaration[@"shape"])[shape addObject:@(extent(n))];
  MetalTensorLayout *layout=[self shape:shape strides:TensorStrides(shape) offset:0 maps:@{} indices:nil];
  if(!declaration[@"steps"])
    [NSException raise:NSInvalidArgumentException format:@"selection requires ordered transformations; prepare the program again"];
  for(NSArray *step in declaration[@"steps"]) {
    if([step[0] isEqual:@"select"]) {
      NSMutableArray *runs=[NSMutableArray array];
      for(NSArray *run in step[2]) {
        NSMutableArray *values=[NSMutableArray array];for(id n in run)[values addObject:@(extent(n))];
        [runs addObject:values];
      }
      layout=[layout select:[step[1] unsignedIntegerValue] runs:runs];
    } else if([step[0] isEqual:@"permute"])layout=[layout permute:step[1]];
    else if([step[0] isEqual:@"reshape"]) {
      NSMutableArray *shape=[NSMutableArray array];for(id n in step[1])[shape addObject:@(extent(n))];
      layout=[layout reshape:shape];
    } else if([step[0] isEqual:@"reindex"]) {
      NSMutableArray *shape=[NSMutableArray array];for(id n in step[1][@"shape"])[shape addObject:@(extent(n))];
      NSMutableDictionary *parameters=[NSMutableDictionary dictionary];
      id term=TensorSelectionIndex(extent,shape,step[1][@"index"],parameters,[NSMutableDictionary dictionary]);
      MetalTensorLayout *indices=[MetalTensorLayout shape:shape index:term parameters:parameters];
      layout=[indices rebase:[self shape:layout.shape strides:TensorStrides(layout.shape) offset:0 maps:@{} indices:nil] onto:layout];
    } else if([step[0] isEqual:@"components"]) {
      layout=[layout components:(NSUInteger)extent(step[1])];
    } else if([step[0] isEqual:@"rebase"]) {
      layout=[layout rebase:[self selection:step[1][@"source"] extent:extent]
          onto:[self selection:step[1][@"target"] extent:extent]];
    } else if([step[0] isEqual:@"union"]) {
      NSDictionary *recipe=step[1],*member=recipe[@"member"];
      BOOL hasMember=member!=(id)NSNull.null;
      layout=MeshIndexUnion([self selection:recipe[@"source_root"] extent:extent],
          [self selection:recipe[@"target_root"] extent:extent],
          [self selection:recipe[@"source"] extent:extent],[self selection:recipe[@"target"] extent:extent],
          hasMember?[self selection:member[@"selection"] extent:extent]:nil,
          hasMember&&[member[@"side"] isEqual:@"source"]);
    } else [NSException raise:NSInvalidArgumentException format:@"unknown selection transformation %@",step[0]];
  }
  return layout;
}
+ (instancetype)configuration:(NSDictionary *)value {
    NSMutableDictionary *maps=[NSMutableDictionary dictionary];
    for(NSString *axis in value[@"maps"])maps[@(axis.integerValue)]=TensorWords(value[@"maps"][axis]);
    MetalTensorLayout *layout=[self shape:value[@"index_shape"] strides:value[@"strides"]
        offset:[value[@"offset"] integerValue] maps:maps indices:nil];
    return [layout shaped:value[@"shape"] coordinates:value[@"index_coordinates"]];
}
- (NSDictionary *)configuration {
    NSMutableDictionary *maps=[NSMutableDictionary dictionary];
    for(NSNumber *axis in self.maps) {
        NSData *data=self.maps[axis];const NSInteger *words=data.bytes;NSMutableArray *values=[NSMutableArray array];
        for(NSUInteger k=0;k<data.length/sizeof *words;k++)[values addObject:@(words[k])];
        maps[axis.stringValue]=values;
    }
    return @{@"shape":self.shape,@"index_shape":self.indexShape,@"strides":self.strides,@"offset":@(self.offset),
        @"index_coordinates":self.coordinates,@"maps":maps};
}
+ (instancetype)runs:(NSArray<NSArray<NSNumber *> *> *)runs {
    NSUInteger count=0;for(NSArray *run in runs)count+=[run[1] unsignedIntegerValue]*[run[3] unsignedIntegerValue];
    if(!count)return [self shape:@[@0] strides:@[@1] offset:0 maps:@{} indices:nil];
    runs=TensorRunsJoined(runs);
    // one run is the strided layout [repeats, width] at strides [step, 1], read as one axis
    if(runs.count==1)return [[self shape:@[runs[0][3],runs[0][1]] strides:@[runs[0][2],@1] offset:[runs[0][0] integerValue]
        maps:@{} indices:nil] reshape:@[@(count)]];
    /* several runs: one piecewise function, its terms emitted once relative to its lowest position, the storage its
       closed-form bounds (each run's first and last rows). Joined runs are no affine function (those join to one), so
       shaped:coordinates:'s affine test and its normalization's second emission are not repeated. */
    NSInteger low=NSIntegerMax,high=NSIntegerMin;
    for(NSArray *run in runs) {
        const NSInteger first=[run[0] integerValue],last=first+[run[2] integerValue]*([run[3] integerValue]-1);
        low=MIN(low,MIN(first,last));high=MAX(high,MAX(first,last)+[run[1] integerValue]-1);
    }
    isl_ctx *ctx=isl_ctx_alloc();
    id index=MeshIndexTerms(isl_pw_aff_add_constant_val(MeshIndexRuns(ctx,runs),isl_val_int_from_si(ctx,-(long)low)));
    isl_ctx_free(ctx);
    MetalTensorLayout *result=[[self shape:@[@(high-low+1)] strides:@[@1] offset:low maps:@{} indices:nil]
        withShape:@[@(count)] offset:low];
    result->_count=count;result->_coordinates=[@[index] copy];
    return result;
}
/* Exact layout algebra in a view's coordinates, for placement's unions and identities (MeshIndexUnion,
   MetalTensorIdentify). Added by the layout algebra session: a view's flat isl terms carry the divisions that come only
   from flattening its coordinates into one index (a units-first [2, 8, 1024] half binding's bytes emit as nested selects
   of i mod 2048), and the quotient of two such views had isl eliminate them by parametric lexmin even to enumerate its
   representatives, past 60 s on every rank for the LSTM's h stored units first (2026-11-01). Here a view's strides are
   read back and proved, and the quotient of a pairing is written down, its divisions by the removed view's storage
   strides alone. What must stay: the same positions and the same class numbering as the isl quotient. */

// The layout's position at one flat index, integers only: its compiled coordinates and their strides or maps.
static NSInteger TensorValueAt(MetalTensorLayout *layout,NSArray<NSData *> *programs,NSInteger *stack,NSInteger at) {
    NSInteger value=layout.offset;
    for(NSUInteger d=0;d<programs.count;d++) {
        const struct TensorProgramStep *program=programs[d].bytes;
        const NSInteger c=TensorRun(program,programs[d].length/sizeof *program,at,stack);
        const NSInteger *map=((NSData *)layout.maps[@(d)]).bytes;
        value+=map?map[c]:c*[layout.strides[d] integerValue];
    }
    return value;
}
/* A layout's function as mixed-radix strides, probed from the innermost axis out: an axis' stride is the position one
   block on, its extent the blocks that stay on that line (a contiguous next axis joins it, which is the same function);
   the candidate is then compared with the layout at every position, integers only, so a returned layout is the same
   function exactly. nil where no candidate matches. */
static MetalTensorLayout *TensorProbed(MetalTensorLayout *layout) {
    const NSUInteger n=layout.count;if(!n)return nil;
    NSMutableArray *programs=[NSMutableArray array];NSUInteger longest=1;
    for(id coordinate in layout.coordinates) {
        NSMutableData *program=[NSMutableData data];TensorCompile(coordinate,program);[programs addObject:program];
        longest=MAX(longest,program.length/sizeof(struct TensorProgramStep));
    }
    NSInteger stack[longest+1];
    NSMutableData *levels=[NSMutableData data];
    const NSInteger origin=TensorValueAt(layout,programs,stack,0);
    for(NSUInteger block=1;block<n;) {
        const NSInteger step=TensorValueAt(layout,programs,stack,(NSInteger)block)-origin;NSUInteger e=2;
        while(e<n/block&&TensorValueAt(layout,programs,stack,(NSInteger)(e*block))==origin+(NSInteger)e*step)e++;
        if((n/block)%e)return nil;
        const NSInteger level[3]={(NSInteger)block,(NSInteger)e,step};[levels appendBytes:level length:sizeof level];block*=e;
    }
    const NSInteger *axes=levels.bytes;const NSUInteger m=levels.length/(3*sizeof *axes);
    for(NSUInteger i=0;i<n;i++) {
        NSInteger value=origin;
        for(NSUInteger k=0;k<m;k++)value+=(NSInteger)((i/(NSUInteger)axes[3*k])%(NSUInteger)axes[3*k+1])*axes[3*k+2];
        if(value!=TensorValueAt(layout,programs,stack,(NSInteger)i))return nil;
    }
    NSMutableArray *extents=[NSMutableArray array],*strides=[NSMutableArray array];
    for(NSUInteger k=m;k-->0;){[extents addObject:@(axes[3*k+1])];[strides addObject:@(axes[3*k+2])];}
    if(!m){[extents addObject:@1];[strides addObject:@1];}
    return [[MetalTensorLayout shape:extents strides:strides offset:origin maps:@{} indices:nil] reshape:layout.shape];
}
// mesh-tensor-layout.h MetalTensorFactored
MetalTensorLayout *MetalTensorFactored(MetalTensorLayout *layout) {
    return layout.factored&&!layout.maps.count?layout:TensorProbed(layout);
}
// One axis of a pairing's common index refinement: its storage strides in both views, extent and flat weight, and
// whether the removed view reads it backward (its stride negative, here the stride's magnitude).
struct TensorPairAxis { NSInteger removed,kept,extent,weight; BOOL reversed; };
static int TensorPairAxisOrder(const void *a,const void *b) {
    const NSInteger x=((const struct TensorPairAxis *)a)->removed,y=((const struct TensorPairAxis *)b)->removed;
    return x<y?-1:x>y;
}
// The flat-index boundaries of a factored layout's axes (each axis' weight and the weight above it).
static void TensorBoundaries(MetalTensorLayout *layout,NSMutableSet *bounds) {
    NSArray *weights=TensorStrides(layout.indexShape);
    for(NSUInteger d=0;d<weights.count;d++)if([layout.indexShape[d] integerValue]>1) {
        [bounds addObject:weights[d]];[bounds addObject:@([weights[d] integerValue]*[layout.indexShape[d] integerValue])];
    }
}
// The storage stride of a factored layout along the refined axis of flat weight `weight` (0 where no axis varies there).
static NSInteger TensorStrideAt(MetalTensorLayout *layout,NSInteger weight) {
    NSArray *weights=TensorStrides(layout.indexShape);
    for(NSUInteger d=0;d<weights.count;d++) {
        const NSInteger w=[weights[d] integerValue],e=[layout.indexShape[d] integerValue];
        if(e>1&&w<=weight&&weight<w*e)return [layout.strides[d] integerValue]*(weight/w);
    }
    return 0;
}
// A layout whose index terms are kept as written, positions [low, high]: no isl emission rewrites them.
static MetalTensorLayout *TensorWritten(NSArray *shape,id index,NSInteger low,NSInteger high) {
    MetalTensorLayout *result=[[MetalTensorLayout shape:@[@(high-low+1)] strides:@[@1] offset:low maps:@{} indices:nil]
        withShape:shape offset:low];
    result->_count=TensorCount(shape);result->_coordinates=[@[TensorSum(@[index,@(-low)])] copy];
    return result;
}
// mesh_graph_pair_classes' term constructors: the index terms of this file (TensorSum, TensorScale, TensorQuotient).
struct TensorPairTerms { MetalTensorLayout *factored,*kept; const struct TensorPairAxis *axes; };
static void *TensorPairConstant(void *c,int64_t v){(void)c;return @(v);}
static void *TensorPairPosition(void *c){(void)c;return @"i";}
static void *TensorPairSum(void *c,void *a,void *b){(void)c;return TensorSum(@[(id)a,(id)b]);}
static void *TensorPairScale(void *c,void *a,int64_t f){(void)c;return TensorScale((id)a,(NSInteger)f);}
static void *TensorPairQuotient(void *c,void *a,int64_t d,int r){(void)c;return TensorQuotient((id)a,(NSInteger)d,r!=0);}
static void *TensorPairMinimum(void *c,void *a,int64_t b){(void)c;return @[@"min",(id)a,@(b)];}
static void *TensorPairCompare(void *c,void *a,enum mesh_graph_relation relation,int64_t b) {
    (void)c;NSString *op=relation==MESH_GRAPH_BELOW?@"lt":relation==MESH_GRAPH_AT_MOST?@"le":relation==MESH_GRAPH_AT_LEAST?@"ge":@"eq";
    return @[op,(id)a,@(b)];
}
static void *TensorPairAll(void *c,void *const *conditions,size_t count) {
    (void)c;NSMutableArray *all=[NSMutableArray arrayWithObject:@"and"];
    for(size_t k=0;k<count;k++)[all addObject:(id)conditions[k]];
    return all;
}
static void *TensorPairSelect(void *c,void *condition,void *yes,void *no){(void)c;return @[@"select",(id)condition,(id)yes,(id)no];}
static void *TensorPairKept(void *c,void *const *digits,size_t m) {
    const struct TensorPairTerms *pair=c;NSMutableArray *value=[NSMutableArray array],*logical=[NSMutableArray array];
    // the view's own digit along each axis: the lattice's digit, or extent - 1 - it where the axis is read backward
    for(size_t k=0;k<m;k++)[logical addObject:pair->axes[k].reversed?
        TensorSum(@[@(pair->axes[k].extent-1),TensorScale((id)digits[k],-1)]):(id)digits[k]];
    if(pair->factored) {
        [value addObject:@(pair->factored.offset)];
        for(size_t k=0;k<m;k++)[value addObject:TensorScale(logical[k],pair->axes[k].kept)];
    } else {
        NSMutableArray *index=[NSMutableArray array],*flat=[NSMutableArray arrayWithObject:@(pair->kept.offset)];
        for(size_t k=0;k<m;k++)[index addObject:TensorScale(logical[k],pair->axes[k].weight)];
        NSArray *coordinates=pair->kept.coordinates;
        for(NSUInteger d=0;d<coordinates.count;d++)[flat addObject:TensorScale(coordinates[d],[pair->kept.strides[d] integerValue])];
        [value addObject:TensorSubstitute(TensorSum(flat),TensorSum(index))];
    }
    return TensorSum(value);
}
static const struct mesh_graph_terms TensorPairTermConstructors={TensorPairConstant,TensorPairPosition,TensorPairSum,
    TensorPairScale,TensorPairQuotient,TensorPairMinimum,TensorPairCompare,TensorPairAll,TensorPairSelect,TensorPairKept};
/* mesh-tensor-layout.h MetalTensorPairQuotient. With every kept value below every removed position, each class is one
   kept value with the removed positions paired to it, its least member that kept value, so the classes in order of their
   least members are the ordinals that are not removed, in order: p numbers as p - C(p), C(p) the removed positions below
   p, and removed(x) + shift as kept(x) (below every removed position, so C is 0 there). The removed view is read through
   its own storage strides: y = p - origin is in its image when y's digit at each stride is within its extent, its flat
   index is the digits' weighted sum, and C(y) counts axis by axis from the top. */
NSArray<MetalTensorLayout *> *MetalTensorPairQuotient(NSUInteger count,MetalTensorLayout *kept,MetalTensorLayout *removed,
    NSInteger shift,NSUInteger first,NSUInteger extent) {
    MetalTensorLayout *B=MetalTensorFactored(removed);if(!B||!B.count)return nil;
    MetalTensorLayout *A=MetalTensorFactored(kept);
    NSMutableSet *bounds=[NSMutableSet setWithObjects:@1,@(B.count),nil];TensorBoundaries(B,bounds);
    if(A) {
        NSMutableSet *joint=[[bounds mutableCopy] autorelease];TensorBoundaries(A,joint);
        NSArray *sorted=[joint.allObjects sortedArrayUsingSelector:@selector(compare:)];BOOL chain=YES;
        for(NSUInteger k=0;chain&&k+1<sorted.count;k++)chain=[sorted[k+1] integerValue]%[sorted[k] integerValue]==0;
        if(chain)bounds=joint;else A=nil;
    }
    NSArray *weights=[bounds.allObjects sortedArrayUsingSelector:@selector(compare:)];
    NSMutableData *storage=[NSMutableData data];NSInteger reversal=0;
    for(NSUInteger k=0;k+1<weights.count;k++) {
        const NSInteger w=[weights[k] integerValue];
        struct TensorPairAxis axis={TensorStrideAt(B,w),A?TensorStrideAt(A,w):0,[weights[k+1] integerValue]/w,w,NO};
        /* an axis read backward is the same positions read forward from its last one: the removed lattice takes the
           stride's magnitude and an origin moved by the axis' span, the kept side the digit extent - 1 - x. Changed by
           the layout algebra session: the closed form refused every negative stride, and the LSTM's row-major h with its
           layer axis reversed went to the isl quotient, a lexmin over a 32768-point permutation past 8 s on every rank
           (d-rec2, 2026-11-02). */
        if(!axis.removed)return nil;
        if(axis.removed<0){reversal+=axis.removed*(axis.extent-1);axis.removed=-axis.removed;axis.reversed=YES;}
        [storage appendBytes:&axis length:sizeof axis];
    }
    // a single removed position is one axis of extent 1
    if(!storage.length){const struct TensorPairAxis point={1,0,1,1,NO};[storage appendBytes:&point length:sizeof point];}
    struct TensorPairAxis *axes=storage.mutableBytes;const NSUInteger m=storage.length/sizeof *axes;
    qsort(axes,m,sizeof *axes,TensorPairAxisOrder);
    // a lattice: each axis' stride at least the span of those below
    for(NSUInteger k=0;k+1<m;k++)if(axes[k+1].removed<axes[k].removed*axes[k].extent)return nil;
    const NSInteger origin=B.offset+shift+reversal;
    NSInteger high=0;
    if(A){high=A.offset;for(NSUInteger k=0;k<m;k++)high+=MAX(0,axes[k].kept*(axes[k].extent-1));}
    else {
        if(kept.maps.count)return nil;
        const struct MetalTensorSpan span=kept.storageSpan;high=span.offset+(NSInteger)span.count-1;
    }
    if(high>=origin)return nil;
    /* the class rule and the lattice's count are the graph module's (mesh-graph.h mesh_graph_pair_classes); the terms
       are built here, the kept side's position at a removed position's digits by the kept view's strides along the
       common axes, or its own terms at their flat index */
    NSMutableData *strides=[NSMutableData dataWithLength:m*sizeof(int64_t)],*extents=[NSMutableData dataWithLength:m*sizeof(int64_t)];
    for(NSUInteger k=0;k<m;k++){((int64_t *)strides.mutableBytes)[k]=axes[k].removed;((int64_t *)extents.mutableBytes)[k]=axes[k].extent;}
    const struct mesh_graph_lattice lattice={origin,m,strides.bytes,extents.bytes};
    struct TensorPairTerms pair={A,kept,axes};
    id classes=(id)mesh_graph_pair_classes(&lattice,(int64_t)first,(int64_t)(first+extent),&TensorPairTermConstructors,&pair);
    const NSInteger total=(NSInteger)count-(NSInteger)B.count;
    id index=first?TensorSubstitute(classes,TensorSum(@[@"i",@((NSInteger)first)])):classes;
    return @[[MetalTensorLayout shape:@[@(total)] strides:@[@1] offset:0 maps:@{} indices:nil],
        TensorWritten(@[@(extent)],index,0,total-1)];
}
- (MetalTensorLayout *)intersection:(MetalTensorLayout *)other {
    isl_ctx *ctx=isl_ctx_alloc();
    isl_set *a=isl_map_range(MeshIndexRelation(ctx,self)),*b=isl_map_range(MeshIndexRelation(ctx,other));
    isl_set *common=isl_set_intersect(a,b);MetalTensorLayout *result=MeshIndexSelection(common);
    isl_set_free(common);isl_ctx_free(ctx);return result;
}
- (MetalTensorLayout *)union:(MetalTensorLayout *)other {
    isl_ctx *ctx=isl_ctx_alloc();
    isl_set *a=isl_map_range(MeshIndexRelation(ctx,self)),*b=isl_map_range(MeshIndexRelation(ctx,other));
    isl_set *common=isl_set_union(a,b);MetalTensorLayout *result=MeshIndexSelection(common);
    isl_set_free(common);isl_ctx_free(ctx);return result;
}
- (MetalTensorLayout *)preimage:(MetalTensorLayout *)values {
    isl_ctx *ctx=isl_ctx_alloc();
    isl_set *image=isl_map_range(MeshIndexRelation(ctx,values));
    isl_set *positions=isl_map_domain(isl_map_intersect_range(MeshIndexRelation(ctx,self),image));
    MetalTensorLayout *result=MeshIndexSelection(positions);
    isl_set_free(positions);isl_ctx_free(ctx);return result;
}
@synthesize coordinates=_coordinates;
+ (instancetype)shape:(NSArray<NSNumber *> *)shape index:(id)expression parameters:(NSDictionary<NSString *,id> *)parameters {
    const NSUInteger count=TensorCount(shape);
    if(!count)return [self shape:shape strides:TensorStrides(shape) offset:0 maps:@{} indices:nil];
    NSMutableDictionary *values=TensorParameters(shape,parameters);values[@"i"]=@"i";
    NSArray *digits=TensorDigits(shape);
    for(NSUInteger d=0;d<shape.count;d++) {
        values[[NSString stringWithFormat:@"axis.%lu",(unsigned long)d]]=digits[d];
    }
    isl_ctx *ctx=isl_ctx_alloc();
    for(NSString *name in parameters)if([parameters[name] isKindOfClass:MetalTensorLayout.class])
        values[name]=MeshIndexTerms(MeshIndexFunction(ctx,[parameters[name] reshape:shape]));
    isl_pw_aff *relation=MeshIndexExpression(ctx,TensorBind(expression,values),0,count);
    MetalTensorLayout *result=TensorFunctionLayout(relation,shape);isl_ctx_free(ctx);return result;
}
+ (instancetype)shape:(NSArray<NSNumber *> *)shape strides:(NSArray<NSNumber *> *)strides
    offset:(NSInteger)offset maps:(NSDictionary<NSNumber *,NSData *> *)maps indices:(NSData *)indices {
    NSParameterAssert(shape.count==strides.count);
    MetalTensorLayout *result=[[[self alloc] init] autorelease];
    result->_shape=[shape copy];result->_indexShape=[shape copy];result->_count=TensorCount(shape);
    // A joint map may simplify to the same separable representation as a
    // strided tensor. This affects emission, never the operand/dependency type.
    if(indices) {
        NSParameterAssert(indices.length==result->_count*sizeof(NSInteger));
        const NSInteger *values=indices.bytes;
        if(result->_count) {
            NSMutableDictionary *axes=[NSMutableDictionary dictionary];
            NSArray *logical=TensorStrides(shape);
            for(NSUInteger d=0;d<shape.count;d++) {
                const NSUInteger n=[shape[d] unsignedIntegerValue],step=[logical[d] unsignedIntegerValue];
                NSMutableData *data=[NSMutableData dataWithLength:n*sizeof(NSInteger)];
                NSInteger *axis=data.mutableBytes;
                for(NSUInteger j=0;j<n;j++)axis[j]=values[j*step]-values[0];
                axes[@(d)]=data;
            }
            BOOL separable=YES;
            for(NSUInteger i=0;separable&&i<result->_count;i++) {
                NSInteger value=values[0];
                for(NSUInteger d=0;d<shape.count;d++) {
                    const NSInteger *axis=((NSData *)axes[@(d)]).bytes;
                    value+=axis[(i/[logical[d] unsignedIntegerValue])%[shape[d] unsignedIntegerValue]];
                }
                separable=value==values[i];
            }
            if(separable){offset+=values[0];maps=axes;strides=logical;indices=nil;}
        } else {maps=@{};strides=TensorStrides(shape);indices=nil;}
    }
    if(indices) {
        // A joint table is one index factor of this same representation.
        // Logical axes stay in shape; no separate table path survives lowering.
        [result->_indexShape release];result->_indexShape=[@[@(result->_count)] copy];
        shape=result->_indexShape;strides=@[@1];maps=@{@0:indices};
    }
    NSData *words=TensorWords(strides);
    NSDictionary *normalized=MetalViewNormalizeAxes(words.bytes,shape.count,maps);
    const NSInteger *steps=((NSData *)normalized[@"strides"]).bytes;
    NSMutableArray *held=[NSMutableArray arrayWithCapacity:shape.count];
    for(NSUInteger d=0;d<shape.count;d++)[held addObject:@(steps[d])];
    result->_strides=[held copy];result->_maps=[normalized[@"maps"] copy];
    result->_offset=offset+[normalized[@"offset"] integerValue];
    return result;
}
- (void)dealloc { [_shape release];[_indexShape release];[_strides release];[_maps release];[_coordinates release];[super dealloc]; }
// Copy the flat index factors, never a parent layout or a chain of transforms.
- (MetalTensorLayout *)withShape:(NSArray<NSNumber *> *)shape offset:(NSInteger)offset {
    MetalTensorLayout *result=[[[MetalTensorLayout alloc] init] autorelease];
    result->_shape=[shape copy];result->_indexShape=[_indexShape copy];result->_count=_count;
    result->_strides=[_strides copy];result->_maps=[_maps copy];result->_offset=offset;result->_coordinates=[_coordinates copy];
    return result;
}
- (BOOL)factored { return !_coordinates; }
- (NSArray *)coordinates { return _coordinates?:TensorDigits(_indexShape); }
- (MetalTensorLayout *)shaped:(NSArray<NSNumber *> *)shape coordinates:(NSArray *)coordinates {
    NSParameterAssert(coordinates.count==_indexShape.count);
    const NSUInteger count=TensorCount(shape);NSMutableArray *values=[NSMutableArray array],*standard=[NSMutableArray array];
    /* piecewise coordinates (a composition through a written quotient, an isl emission's selects) that are one strided
       function: its strides read back and proved at every position (TensorProbed), before any normalization. Added by
       the layout algebra session: a units-first h's views composed with their union's written map spent placement's
       declaration in normalizing the composite through isl (MeshIndexExpression of nested selects, 2026-11-01), where
       the composite is the view's own strides in the joint domain. What must stay: the same function. */
    BOOL piecewise=NO;for(id value in coordinates)piecewise|=TensorHasGeneral(value);
    if(piecewise&&count>1) {
        MetalTensorLayout *raw=[self withShape:shape offset:_offset];raw->_count=count;
        [raw->_coordinates release];raw->_coordinates=[coordinates copy];
        MetalTensorLayout *strided=TensorProbed(raw);if(strided)return strided;
    }
    for(id value in coordinates)[values addObject:TensorNormalize(value,count)];
    for(id value in TensorDigits(_indexShape))[standard addObject:TensorNormalize(value,count)];
    if(count==TensorCount(_indexShape)&&[values isEqual:standard]) {
        MetalTensorLayout *result=[self withShape:_indexShape offset:_offset];result->_count=count;
        [result->_coordinates release];result->_coordinates=nil;return [result reshape:shape];
    }
    // A selection may remove every irregular part of a map. Retain a direct
    // strided API binding when the resulting expression has that form; no
    // payload copy is needed to satisfy a consumer's layout requirement.
    if(count) {
        NSMutableArray *terms=[NSMutableArray arrayWithObject:@(_offset)];BOOL linear=YES;
        for(NSUInteger d=0;linear&&d<values.count;d++) {
            NSInteger step=[_strides[d] integerValue],origin=0;const NSInteger *map=((NSData *)_maps[@(d)]).bytes;
            if(map) {
                struct TensorExpressionRange range=TensorRange(values[d],count);
                range.low=MAX(0,range.low);range.high=MIN([_indexShape[d] integerValue]-1,range.high);
                step=range.high>range.low?map[range.low+1]-map[range.low]:0;origin=map[range.low]-range.low*step;
                for(NSInteger k=range.low+1;linear&&k<=range.high;k++)linear=map[k]==origin+k*step;
            }
            [terms addObject:TensorSum(@[@(origin),TensorScale(values[d],step)])];
        }
        if(linear) {
            id expression=TensorNormalize(TensorSum(terms),count);NSInteger origin=TensorEvaluate(expression,0);
            NSArray *logical=TensorStrides(shape),*digits=TensorDigits(shape);NSMutableArray *steps=[NSMutableArray array],*expected=[NSMutableArray arrayWithObject:@(origin)];
            for(NSUInteger d=0;d<shape.count;d++) {
                NSInteger step=[shape[d] unsignedIntegerValue]>1?TensorEvaluate(expression,[logical[d] unsignedIntegerValue])-origin:0;
                [steps addObject:@(step)];[expected addObject:TensorScale(digits[d],step)];
            }
            if(MeshIndexEqual(expression,TensorSum(expected),count))
                return [MetalTensorLayout shape:shape strides:steps offset:origin maps:@{} indices:nil];
        }
    }
    MetalTensorLayout *result=[self withShape:shape offset:_offset];result->_count=count;
    [result->_coordinates release];result->_coordinates=[values copy];
    return result;
}
- (id)copyWithZone:(NSZone *)zone { return [self retain]; }
- (BOOL)affine { return !_coordinates&&!_maps.count&&[_shape isEqual:_indexShape]; }
- (struct MetalTensorSpan)storageSpan {
    if(!_count)return (struct MetalTensorSpan){_offset,0};
    isl_ctx *ctx=isl_ctx_alloc();struct MeshIndexBounds bounds=MeshIndexImageBounds(MeshIndexRelation(ctx,self));
    isl_ctx_free(ctx);return (struct MetalTensorSpan){bounds.low,(NSUInteger)(bounds.high-bounds.low+1)};
}
- (BOOL)equivalent:(MetalTensorLayout *)other {
    if(![_shape isEqual:other.shape])return NO;
    if([self isEqual:other]||!_count)return YES;
    isl_ctx *ctx=isl_ctx_alloc();isl_map *a=MeshIndexRelation(ctx,self),*b=MeshIndexRelation(ctx,other);
    isl_bool same=isl_map_is_equal(a,b);isl_map_free(a);isl_map_free(b);isl_ctx_free(ctx);
    if(same==isl_bool_error)[NSException raise:NSInvalidArgumentException format:@"layout relation equality could not be resolved"];
    return same==isl_bool_true;
}
- (BOOL)isEqual:(id)object {
    if(![object isKindOfClass:MetalTensorLayout.class])return NO;
    MetalTensorLayout *other=object;
    if(_offset!=other.offset||![_shape isEqual:other.shape]||![_indexShape isEqual:other.indexShape]||![_maps isEqual:other.maps]||![self.coordinates isEqual:other.coordinates])return NO;
    for(NSUInteger d=0;d<_indexShape.count;d++)if([_indexShape[d] unsignedIntegerValue]>1&&![_strides[d] isEqual:other.strides[d]])return NO;
    return YES;
}
- (NSUInteger)hash {
    NSUInteger hash=(_shape.hash+_indexShape.hash)^_maps.hash^(NSUInteger)_offset^self.coordinates.hash;
    for(NSUInteger d=0;d<_indexShape.count;d++)if([_indexShape[d] unsignedIntegerValue]>1)hash^=[_strides[d] hash];
    return hash;
}
- (NSInteger)index:(NSUInteger)logical {
    NSInteger result=_offset;
    if(_coordinates) {
        for(NSUInteger d=0;d<_indexShape.count;d++) {
            NSInteger at=TensorEvaluate(_coordinates[d],logical);const NSInteger *values=((NSData *)_maps[@(d)]).bytes;
            result+=values?values[at]:at*[_strides[d] integerValue];
        }
        return result;
    }
    for(NSUInteger d=_indexShape.count;d--;) {
        const NSUInteger n=[_indexShape[d] unsignedIntegerValue],at=logical%n;logical/=n;
        const NSInteger *values=((NSData *)_maps[@(d)]).bytes;
        result+=values?values[at]:(NSInteger)at*[_strides[d] integerValue];
    }
    return result;
}
- (MetalTensorLayout *)gather:(NSArray<NSNumber *> *)shape indices:(NSData *)indices {
    const NSUInteger count=TensorCount(shape);NSParameterAssert(indices.length==count*sizeof(NSInteger));
    NSMutableData *data=[NSMutableData dataWithLength:indices.length];
    const NSInteger *source=indices.bytes;NSInteger *target=data.mutableBytes;
    for(NSUInteger i=0;i<count;i++)target[i]=[self index:source[i]];
    return [MetalTensorLayout shape:shape strides:TensorStrides(shape) offset:0 maps:@{} indices:data];
}
- (MetalTensorLayout *)select:(NSUInteger)axis indices:(NSData *)indices {
    return TensorAxisTransform(self,axis,^MetalTensorLayout *(MetalTensorLayout *part,NSUInteger local) {
        return TensorSelect(part,local,indices);
    });
}
- (MetalTensorLayout *)select:(NSUInteger)axis runs:(NSArray<NSArray<NSNumber *> *> *)runs {
    NSUInteger count=0;for(NSArray *run in runs)count+=[run[1] unsignedIntegerValue]*[run[3] unsignedIntegerValue];
    if(!count)return [self slice:axis first:0 count:0 step:1];
    NSMutableArray *shape=[[_shape mutableCopy] autorelease];shape[axis]=@(count);
    NSUInteger inner=1;for(NSUInteger d=axis+1;d<_shape.count;d++)inner*=[_shape[d] unsignedIntegerValue];
    if(!TensorCount(shape))return [MetalTensorLayout shape:shape strides:TensorStrides(shape) offset:_offset maps:@{} indices:nil];
    id selection=TensorRunsIndex(TensorRunsJoined(runs));
    id digit=TensorQuotient(TensorQuotient(@"i",inner,NO),count,YES);
    id selected=TensorSubstitute(selection,digit);
    id index=TensorSum(@[TensorScale(TensorQuotient(@"i",count*inner,NO),[_shape[axis] integerValue]*inner),
                        TensorScale(selected,inner),TensorQuotient(@"i",inner,YES)]);
    return TensorCompose(self,shape,index);
}
- (MetalTensorLayout *)slice:(NSUInteger)axis first:(NSInteger)first count:(NSUInteger)count step:(NSInteger)step {
    if(first==0&&count==[_shape[axis] unsignedIntegerValue]&&step==1)return self;
    return TensorAxisTransform(self,axis,^MetalTensorLayout *(MetalTensorLayout *part,NSUInteger local) {
        return TensorSlice(part,local,first,count,step);
    });
}
- (MetalTensorLayout *)element:(NSUInteger)axis at:(NSUInteger)index {
    MetalTensorLayout *one=[self slice:axis first:index count:1 step:1];
    NSMutableArray *shape=[[one.shape mutableCopy] autorelease];[shape removeObjectAtIndex:axis];
    return [one reshape:shape];
}
- (MetalTensorLayout *)permute:(NSArray<NSNumber *> *)axes {
    if(!self.count||!self.factored||[_shape isEqual:_indexShape]||TensorPermutationIsReshape(self,axes))return TensorPermute(self,axes);
    NSData *storage=TensorSections(self);const struct TensorSection *sections=storage.bytes;
    const NSUInteger count=storage.length/sizeof *sections;
    NSMutableArray *owners=[NSMutableArray array],*ends=[NSMutableArray array];
    for(NSUInteger k=0;k<count;k++) {
        [ends addObject:@0];
        for(NSUInteger d=sections[k].first;d<sections[k].end;d++)[owners addObject:@(k)];
    }
    for(NSUInteger d=0;d<axes.count;d++)ends[[owners[[axes[d] unsignedIntegerValue]] unsignedIntegerValue]]=@(d+1);
    NSMutableArray *parts=[NSMutableArray array];
    for(NSUInteger first=0;first<axes.count;) {
        // Interleaved coupled axes form one domain. Independent domains keep
        // their own factors, even when moved across a coupled domain.
        NSUInteger end=first+1;NSMutableIndexSet *members=[NSMutableIndexSet indexSet];
        for(NSUInteger d=first;d<end;d++) {
            const NSUInteger member=[owners[[axes[d] unsignedIntegerValue]] unsignedIntegerValue];
            [members addIndex:member];end=MAX(end,[ends[member] unsignedIntegerValue]);
        }
        NSMutableArray *inputs=[NSMutableArray array],*local=[NSMutableArray array];
        NSMutableDictionary *positions=[NSMutableDictionary dictionary];NSUInteger position=0;
        for(NSUInteger k=members.firstIndex;k!=NSNotFound;k=[members indexGreaterThanIndex:k]) {
            [inputs addObject:TensorSectionLayout(self,sections[k])];
            for(NSUInteger d=sections[k].first;d<sections[k].end;d++)positions[@(d)]=@(position++);
        }
        for(NSUInteger d=first;d<end;d++)[local addObject:positions[axes[d]]];
        [parts addObject:TensorPermute(TensorProduct(inputs),local)];first=end;
    }
    return [TensorProduct(parts) shift:_offset];
}
- (MetalTensorLayout *)reshape:(NSArray<NSNumber *> *)shape {
    NSParameterAssert(TensorCount(shape)==_count);
    if([_shape isEqual:shape])return self;
    MetalTensorLayout *factored=TensorReshape(self,shape);if(factored)return factored;
    return [self withShape:shape offset:_offset];
}
- (MetalTensorLayout *)components:(NSUInteger)count {
    NSMutableArray *shape=[[_shape mutableCopy] autorelease];[shape addObject:@(count)];
    NSMutableArray *basis=[NSMutableArray arrayWithArray:_indexShape];[basis addObject:@(count)];
    NSMutableArray *strides=[NSMutableArray array];for(NSNumber *step in _strides)[strides addObject:@(step.integerValue*(NSInteger)count)];
    [strides addObject:@1];NSMutableDictionary *maps=[NSMutableDictionary dictionary];
    for(NSNumber *axis in _maps) {
        NSMutableData *data=[[_maps[axis] mutableCopy] autorelease];NSInteger *values=data.mutableBytes;
        for(NSUInteger i=0;i<data.length/sizeof(NSInteger);i++)values[i]*=count;
        maps[axis]=data;
    }
    MetalTensorLayout *result=[MetalTensorLayout shape:basis strides:strides offset:_offset*(NSInteger)count maps:maps indices:nil];
    if(_coordinates) {
        NSMutableArray *coordinates=[NSMutableArray array];id index=TensorQuotient(@"i",count,NO);
        for(id value in _coordinates)[coordinates addObject:TensorSubstitute(value,index)];
        [coordinates addObject:TensorQuotient(@"i",count,YES)];return [result shaped:shape coordinates:coordinates];
    }
    return [result reshape:shape];
}
- (MetalTensorLayout *)shift:(NSInteger)elements {
    return elements?[self withShape:_shape offset:_offset+elements]:self;
}
- (MetalTensorLayout *)elements:(NSUInteger)count {
    NSParameterAssert(count&&[_shape.lastObject unsignedIntegerValue]==count);
    NSMutableArray *shape=[[_shape mutableCopy] autorelease];[shape removeLastObject];
    if(!_count)return [MetalTensorLayout shape:shape strides:TensorStrides(shape) offset:_offset/(NSInteger)count maps:@{} indices:nil];
    MetalTensorLayout *closed=nil;
    @try{closed=TensorClosedElements(self,count,shape);}@catch(NSException *unused){(void)unused;closed=nil;}
    if(closed)return closed;
    isl_ctx *ctx=isl_ctx_alloc();
    isl_pw_aff *first=MeshIndexExpression(ctx,@[@"mul",@"i",@(count)],0,_count/count);
    isl_pw_aff *value=isl_pw_aff_pullback_pw_multi_aff(MeshIndexFunction(ctx,self),isl_pw_multi_aff_from_pw_aff(first));
    value=isl_pw_aff_floor(isl_pw_aff_scale_down_val(value,isl_val_int_from_ui(ctx,count)));
    MetalTensorLayout *result=TensorFunctionLayout(value,shape);isl_ctx_free(ctx);
    if(![[result components:count] equivalent:self])
        [NSException raise:NSInvalidArgumentException format:@"typed element components are not adjacent and aligned in their declared storage"];
    return result;
}
/* A rebase through a declaration that is a factored injection, its inverse written down: a view position p is the
   declaration's at flat index x(p), its digits read at the declaration's own strides (sorted by magnitude, an axis read
   backward from its last position, as MetalTensorPairQuotient's lattice) and weighted by their flat weights. x is
   proved at every view position by integers (x in range and the declaration's position at x the view's), then composed
   into the placement, its strides read back exactly where it has them (TensorProbed). nil where the declaration is not
   such an injection or a view position is not one of its positions. Added by the layout algebra session: the LSTM's
   placement rebased each layer's h, at region positions, through the binding's declaration with the layers 1.6M
   elements apart, and isl's lexmax inverse took 0.7 s a rank (d-rec3) and outlived the 8 s bound (d-r3, 2026-11-02);
   this is integer arithmetic on strides and offsets, whatever their size. What must stay: the same layout. */
static MetalTensorLayout *TensorInverseRebase(MetalTensorLayout *view,MetalTensorLayout *declared,MetalTensorLayout *placed) {
    if(view.maps.count||placed.maps.count)return nil;
    declared=MetalTensorFactored(declared);if(!declared)return nil;
    struct TensorInverseAxis { NSInteger stride,extent,weight; BOOL reversed; };
    NSArray *weights=TensorStrides(declared.indexShape);NSMutableData *storage=[NSMutableData data];
    NSInteger origin=declared.offset;
    for(NSUInteger d=0;d<declared.indexShape.count;d++) {
        struct TensorInverseAxis axis={[declared.strides[d] integerValue],[declared.indexShape[d] integerValue],
            [weights[d] integerValue],NO};
        if(axis.extent==1)continue;
        if(!axis.stride)return nil;
        if(axis.stride<0){origin+=axis.stride*(axis.extent-1);axis.stride=-axis.stride;axis.reversed=YES;}
        [storage appendBytes:&axis length:sizeof axis];
    }
    struct TensorInverseAxis *axes=storage.mutableBytes;const NSUInteger m=storage.length/sizeof *axes;
    for(NSUInteger k=1;k<m;k++)for(NSUInteger j=k;j>0&&axes[j].stride<axes[j-1].stride;j--){
        const struct TensorInverseAxis t=axes[j];axes[j]=axes[j-1];axes[j-1]=t;
    }
    for(NSUInteger k=0;k+1<m;k++)if(axes[k+1].stride<axes[k].stride*axes[k].extent)return nil;
    NSMutableArray *address=[NSMutableArray arrayWithObject:@(view.offset-origin)];NSArray *coordinates=view.coordinates;
    for(NSUInteger d=0;d<coordinates.count;d++)[address addObject:TensorScale(coordinates[d],[view.strides[d] integerValue])];
    id y=TensorSum(address);NSMutableArray *index=[NSMutableArray array];
    for(NSUInteger k=0;k<m;k++) {
        id digit=TensorQuotient(k+1<m?TensorQuotient(y,axes[k+1].stride,YES):y,axes[k].stride,NO);
        if(axes[k].reversed)digit=TensorSum(@[@(axes[k].extent-1),TensorScale(digit,-1)]);
        [index addObject:TensorScale(digit,axes[k].weight)];
    }
    id x=TensorSum(index);
    NSMutableData *program=[NSMutableData data];TensorCompile(x,program);
    const struct TensorProgramStep *steps=program.bytes;const NSUInteger length=program.length/sizeof *steps;
    NSInteger stack[length+1];
    for(NSUInteger i=0;i<view.count;i++) {
        const NSInteger at=TensorRun(steps,length,(NSInteger)i,stack);
        if(at<0||at>=(NSInteger)declared.count||[declared index:(NSUInteger)at]!=[view index:i])return nil;
    }
    NSMutableArray *composed=[NSMutableArray array];
    for(id value in placed.coordinates)[composed addObject:TensorSubstitute(value,x)];
    MetalTensorLayout *raw=[placed withShape:view.shape offset:placed.offset];raw->_count=view.count;
    [raw->_coordinates release];raw->_coordinates=[composed copy];
    return TensorProbed(raw)?:[placed shaped:view.shape coordinates:composed];
}
/* The same rebase as integer arithmetic on strides and offsets alone, where the view is factored, the declaration a
   factored injection and the placement dense (its index axes the row-major steps, any origin). Each view axis moves
   along one axis of the declaration's lattice (sorted by stride magnitude, reversed axes read forward) by a whole
   multiple of its stride, a view axis wider than that lattice axis split across contiguous lattice axes it fills from
   digit 0. With the view's origin's digits as the base, each lattice digit of a view position is its base plus its
   view axes' steps when their ranges stay within the axis' extent (no carry, so every view position is in the lattice):
   then the flat index is strided over the view's split axes, each stride the step times the lattice axis' flat weight
   (negated where it is read backward). nil where any of that does not hold. */
static MetalTensorLayout *TensorStridedRebase(MetalTensorLayout *view,MetalTensorLayout *declared,MetalTensorLayout *placed) {
    if(!view.factored||view.maps.count||!declared.factored||declared.maps.count||!placed.factored||placed.maps.count)return nil;
    {
        NSInteger step=1;
        for(NSUInteger d=placed.indexShape.count;d-->0;) {
            const NSInteger n=[placed.indexShape[d] integerValue];
            if(n>1&&[placed.strides[d] integerValue]!=step)return nil;
            step*=n;
        }
    }
    struct TensorLatticeAxis { NSInteger stride,extent,weight,base,low,high; BOOL reversed; };
    NSArray *weights=TensorStrides(declared.indexShape);NSMutableData *storage=[NSMutableData data];
    NSInteger origin=declared.offset;
    for(NSUInteger d=0;d<declared.indexShape.count;d++) {
        struct TensorLatticeAxis axis={[declared.strides[d] integerValue],[declared.indexShape[d] integerValue],
            [weights[d] integerValue],0,0,0,NO};
        if(axis.extent==1)continue;
        if(!axis.stride)return nil;
        if(axis.stride<0){origin+=axis.stride*(axis.extent-1);axis.stride=-axis.stride;axis.reversed=YES;}
        [storage appendBytes:&axis length:sizeof axis];
    }
    struct TensorLatticeAxis *axes=storage.mutableBytes;const NSUInteger m=storage.length/sizeof *axes;
    for(NSUInteger k=1;k<m;k++)for(NSUInteger j=k;j>0&&axes[j].stride<axes[j-1].stride;j--){
        const struct TensorLatticeAxis t=axes[j];axes[j]=axes[j-1];axes[j-1]=t;
    }
    for(NSUInteger k=0;k+1<m;k++)if(axes[k+1].stride<axes[k].stride*axes[k].extent)return nil;
    const NSInteger first=view.offset-origin;
    if(first<0||(m&&first%axes[0].stride))return nil;
    NSInteger rest=first;
    for(NSUInteger k=m;k-->0;) {
        axes[k].base=rest/axes[k].stride;rest-=axes[k].base*axes[k].stride;
        if(axes[k].base>=axes[k].extent&&k+1<m)return nil;
        axes[k].low=axes[k].high=axes[k].base;
    }
    if(rest||(m&&axes[m-1].base>=axes[m-1].extent)||(!m&&first))return nil;
    NSMutableArray *shape=[NSMutableArray array],*strides=[NSMutableArray array];
    for(NSUInteger d=0;d<view.indexShape.count;d++) {
        NSInteger extent=[view.indexShape[d] integerValue],step=[view.strides[d] integerValue];
        NSMutableArray *partShape=[NSMutableArray array],*partStrides=[NSMutableArray array];
        if(extent>1&&!step){[partShape addObject:@(extent)];[partStrides addObject:@0];extent=1;}
        const NSInteger sign=step<0?-1:1;NSInteger magnitude=labs(step);
        while(extent>1) {
            NSInteger k=-1;
            for(NSUInteger j=0;j<m;j++)if(axes[j].stride<=magnitude)k=(NSInteger)j;
            if(k<0||magnitude%axes[k].stride)return nil;
            struct TensorLatticeAxis *axis=axes+k;const NSInteger c=magnitude/axis->stride;
            const NSInteger along=sign*c*axis->weight*(axis->reversed?-1:1);
            if(c*(extent-1)<axis->extent) {
                if(sign>0)axis->high+=c*(extent-1);else axis->low-=c*(extent-1);
                [partShape insertObject:@(extent) atIndex:0];[partStrides insertObject:@(along) atIndex:0];extent=1;
            } else {
                // the inner part fills this lattice axis from digit 0, the rest goes on along the next (contiguous) one
                if(sign<0||axis->extent%c||axis->base||axis->low||axis->high||(NSUInteger)k+1>=m||
                   axes[k+1].stride!=axis->stride*axis->extent)return nil;
                const NSInteger part=axis->extent/c;if(extent%part)return nil;
                axis->high=c*(part-1);
                [partShape insertObject:@(part) atIndex:0];[partStrides insertObject:@(along) atIndex:0];
                extent/=part;magnitude=axes[k+1].stride;
            }
        }
        [shape addObjectsFromArray:partShape];[strides addObjectsFromArray:partStrides];
    }
    NSInteger offset=placed.offset;
    for(NSUInteger k=0;k<m;k++) {
        if(axes[k].low<0||axes[k].high>=axes[k].extent)return nil;
        offset+=axes[k].weight*(axes[k].reversed?axes[k].extent-1-axes[k].base:axes[k].base);
    }
    if(!shape.count){[shape addObject:@1];[strides addObject:@0];}
    return [[MetalTensorLayout shape:shape strides:strides offset:offset maps:@{} indices:nil] reshape:view.shape];
}
- (MetalTensorLayout *)rebase:(MetalTensorLayout *)declared onto:(MetalTensorLayout *)placed {
    NSParameterAssert([declared.shape isEqual:placed.shape]);
    if([self isEqual:declared])return placed;
    if(!self.count)return [MetalTensorLayout shape:_shape strides:TensorStrides(_shape) offset:0 maps:@{} indices:nil];
    MetalTensorLayout *closed=nil;
    @try{closed=TensorClosedRebase(self,declared,placed);}@catch(NSException *unused){(void)unused;closed=nil;}
    if(closed)return closed;
    closed=TensorStridedRebase(self,declared,placed)?:TensorInverseRebase(self,declared,placed);
    if(closed)return closed;
    isl_ctx *ctx=isl_ctx_alloc();
    isl_pw_aff *view=MeshIndexFunction(ctx,self),*target=MeshIndexFunction(ctx,placed);
    isl_set *domain=isl_pw_aff_domain(isl_pw_aff_copy(view));
    // The inverse is a relation. Selecting its last preimage retains the
    // existing alias convention where several logical positions name a value.
    // Repeated consumer indices remain repeated in the composed result.
    isl_pw_multi_aff *inverse=isl_map_lexmax_pw_multi_aff(isl_map_reverse(MeshIndexRelation(ctx,declared)));
    isl_pw_multi_aff *selected=isl_pw_multi_aff_pullback_pw_multi_aff(inverse,isl_pw_multi_aff_from_pw_aff(view));
    isl_pw_aff *result=isl_pw_aff_pullback_pw_multi_aff(target,selected);
    isl_set *resolved=isl_pw_aff_domain(isl_pw_aff_copy(result));
    isl_bool complete=isl_set_is_equal(domain,resolved);isl_set_free(domain);isl_set_free(resolved);
    if(complete!=isl_bool_true) {
        isl_pw_aff_free(result);isl_ctx_free(ctx);
        [NSException raise:NSInvalidArgumentException format:@"operand alias names storage outside its declared producer"];
    }
    MetalTensorLayout *layout=TensorFunctionLayout(result,_shape);isl_ctx_free(ctx);return layout;
}
- (BOOL)matches:(NSArray<NSNumber *> *)strides {
    if(!_count)return YES;
    if(!self.affine)return NO;
    for(NSUInteger d=0;d<_shape.count;d++)if([_shape[d] unsignedIntegerValue]>1&&![_strides[d] isEqual:strides[d]])return NO;
    return YES;
}
- (MetalTensorLayout *)required:(NSString *)kind strides:(NSArray<NSNumber *> *)strides {
    if([kind isEqual:@"indexed"]||([kind isEqual:@"strided"]&&self.affine))return self;
    if([kind isEqual:@"positive_strided"]&&self.affine) {
        NSMutableArray *steps=[NSMutableArray arrayWithArray:_strides];BOOL positive=YES;
        NSArray *packed=TensorStrides(_shape);
        for(NSUInteger d=0;d<_shape.count;d++)if([steps[d] integerValue]<=0) {
            if([_shape[d] unsignedIntegerValue]<=1)steps[d]=packed[d];else positive=NO;
        }
        if(positive)return [MetalTensorLayout shape:_shape strides:steps offset:_offset maps:@{} indices:nil];
    }
    NSArray *required=[kind isEqual:@"fixed"]?strides:TensorStrides(_shape);
    NSParameterAssert([kind isEqual:@"fixed"]||[kind isEqual:@"strided"]||[kind isEqual:@"contiguous"]||[kind isEqual:@"positive_strided"]);
    NSParameterAssert(required.count==_shape.count);
    if([self matches:required])return self;
    NSInteger offset=0;
    for(NSUInteger d=0;d<_shape.count;d++)if([_shape[d] integerValue]>0)offset-=MIN(0,([_shape[d] integerValue]-1)*[required[d] integerValue]);
    return [MetalTensorLayout shape:_shape strides:required offset:offset maps:@{} indices:nil];
}
- (MetalTensorLayout *)requirements:(NSArray<NSDictionary *> *)requirements {
    MetalTensorLayout *(^required)(MetalTensorLayout *,NSDictionary *)=^MetalTensorLayout *(MetalTensorLayout *value,NSDictionary *r) {
        MetalTensorLayout *exact=r[@"layout"];
        if(!exact)return [value required:r[@"kind"] strides:r[@"strides"]];
        exact=[exact reshape:value.shape];
        return exact.count?[exact shift:[value index:0]-[exact index:0]]:exact;
    };
    MetalTensorLayout *result=self;
    for(NSDictionary *r in requirements)result=required(result,r);
    for(NSDictionary *r in requirements)
        if(![required(result,r) equivalent:result])
            [NSException raise:NSInvalidArgumentException format:@"conflicting operand layout requirements: %@",requirements];
    return result;
}
- (NSString *)declaration:(NSUInteger)binding {
    return [self declaration:binding origin:_offset];
}
- (NSString *)declaration:(NSUInteger)binding origin:(NSInteger)origin {
    NSMutableString *result=[NSMutableString string];
    NSData *shape=TensorWords(_indexShape),*steps=TensorWords(_strides);
    NSMutableArray *coordinates=nil;
    if(_coordinates){coordinates=[NSMutableArray array];for(id value in _coordinates)[coordinates addObject:TensorExpressionSource(value,_count)];}
    [result appendString:MetalViewMappedIndexDeclaration(binding,shape.bytes,steps.bytes,_indexShape.count,(NSUInteger)_offset,_maps,
        coordinates,coordinates?TensorVectorSpan(self):0,_offset-origin)];
    // APIs that require affine operands consume the strides of the placed
    // view, not strides captured before placement. These are shader constants.
    [result appendFormat:@"template<> constexpr bool mesh_affine<%lu>() { return %s; }\n",
        (unsigned long)binding,self.affine?"true":"false"];
    if(self.affine)for(NSUInteger axis=0;axis<_shape.count;axis++)
        [result appendFormat:@"template<> constexpr long mesh_stride<%lu, %lu>() { return %ldl; }\n",
            (unsigned long)binding,(unsigned long)axis,(long)[_strides[axis] integerValue]];
    return result;
}
- (NSArray<NSArray<NSNumber *> *> *)constants:(NSUInteger)binding {
    return [self constants:binding origin:_offset];
}
- (NSArray<NSArray<NSNumber *> *> *)constants:(NSUInteger)binding origin:(NSInteger)origin {
    if(_maps.count||_coordinates)return nil; // The fixed-axis constant form cannot express composed coordinates.
    NSData *shape=TensorWords(_indexShape),*steps=TensorWords(_strides);
    return MetalViewIndexConstants(binding,shape.bytes,steps.bytes,_indexShape.count,(NSUInteger)_offset,_offset-origin);
}
- (NSString *)positionDeclaration:(NSUInteger)binding {
    isl_ctx *ctx=isl_ctx_alloc();isl_pw_aff *relation=MeshIndexFunction(ctx,self);
    NSUInteger span=0;
    if(_count>1) {
        // Every break lies in one residue class. Its gcd gives the aligned
        // run width without walking the relation's logical elements.
        NSString *shift=[NSString stringWithFormat:@"{ [i] -> [i - 1] : 1 <= i < %lu }",(unsigned long)_count];
        isl_pw_aff *previous=isl_pw_aff_pullback_pw_multi_aff(isl_pw_aff_copy(relation),
            isl_pw_multi_aff_from_map(isl_map_read_from_str(ctx,shift.UTF8String)));
        isl_set *breaks=isl_pw_aff_ne_set(isl_pw_aff_copy(relation),isl_pw_aff_add_constant_val(previous,isl_val_one(ctx)));
        if(isl_set_is_empty(breaks)!=isl_bool_true) {
            isl_val *modulo=NULL,*residue=NULL;isl_set_dim_residue_class_val(breaks,0,&modulo,&residue);
            isl_val *divisor=isl_val_gcd(modulo,residue);NSUInteger value=(NSUInteger)isl_val_get_num_si(divisor);
            span=value&(~value+1);isl_val_free(divisor);
        }
        isl_set_free(breaks);
    }
    isl_pw_aff_free(relation);isl_ctx_free(ctx);
    return MetalViewPositionDeclaration(binding,[self declaration:0],_count,span);
}
- (NSString *)declaration:(NSUInteger)binding fromStorage:(MetalTensorLayout *)source {
    return [[self declaration:binding] stringByAppendingString:TensorStorageDeclaration(self,source,
        [NSString stringWithFormat:@"mesh_storage_%lu",(unsigned long)binding])];
}
- (NSString *)scalarStorageDeclaration:(NSUInteger)binding fromBytes:(MetalTensorLayout *)source elementBytes:(NSUInteger)width {
    NSParameterAssert(width>0);
    NSString *name=[NSString stringWithFormat:@"mesh_storage_bytes_%lu",(unsigned long)binding];
    return [TensorStorageDeclaration(self,source,name) stringByAppendingFormat:
        @"inline long mesh_storage_%lu(long i) { return %@(i * %lul) / %lul; }\n",
        (unsigned long)binding,name,(unsigned long)width,(unsigned long)width];
}
- (NSData *)orderedSpans {
    if(_coordinates)return TensorExpressionSpans(self,YES);
    NSMutableData *data=[NSMutableData data];NSUInteger width=1;
    for(NSUInteger d=_indexShape.count;d--;) {
        NSUInteger n=[_indexShape[d] unsignedIntegerValue];
        if(n>1&&(_maps[@(d)]||[_strides[d] integerValue]!=(NSInteger)width))break;
        width*=n;
    }
    for(NSUInteger i=0;i<_count;i+=width) {
        const struct MetalTensorSpan span={[self index:i],width};
        struct MetalTensorSpan *last=data.length?(struct MetalTensorSpan *)data.mutableBytes+data.length/sizeof span-1:NULL;
        if(last&&last->offset+(NSInteger)last->count==span.offset)last->count+=span.count;
        else [data appendBytes:&span length:sizeof span];
    }
    return data;
}
- (NSData *)reaches:(id<MTLBuffer>)buffer base:(NSUInteger)base elementBytes:(NSUInteger)bytes access:(uint32_t)access binding:(int32_t)binding {
    // a binding reach names its slot (metal-recording.m MetalTensorSpanReaches)
    if(!_count){struct MetalReach empty={buffer,base+_offset*(NSInteger)bytes,0,access|(binding>=0?MetalReachBinding:0),binding};return [NSData dataWithBytes:&empty length:sizeof empty];}
    if(_coordinates)return MetalTensorSpanReaches(buffer,base,TensorExpressionSpans(self,NO),bytes,access,binding);
    NSArray *shape=_indexShape,*strides=_strides;
    NSDictionary *maps=_maps;
    NSMutableDictionary *scaled=[NSMutableDictionary dictionary];
    for(NSNumber *axis in maps) {
        NSMutableData *data=[[maps[axis] mutableCopy] autorelease];NSInteger *values=data.mutableBytes;
        for(NSUInteger i=0;i<data.length/sizeof(NSInteger);i++)values[i]*=bytes;
        scaled[axis]=data;
    }
    NSMutableArray *steps=[NSMutableArray array];for(NSNumber *step in strides)[steps addObject:@(step.integerValue*(NSInteger)bytes)];
    NSData *sizes=TensorWords(shape),*words=TensorWords(steps);
    return MetalViewMappedReaches(buffer,base+_offset*(NSInteger)bytes,sizes.bytes,words.bytes,shape.count,bytes,access,binding,scaled);
}
@end

@implementation MetalTensorPages {
    id _owner;
}
+ (instancetype)adopt:(ncclMeshPageTable_t)table {
    MetalTensorPages *result=[[[self alloc] init] autorelease];
    result->_table=table;
    ncclMeshBuffer binding={0};
    ncclResult_t status=ncclMeshPageTableShape(result->_table,&result->_pageCount,&result->_pageBytes);
    if(!status)status=ncclMeshPageTableBinding(result->_table,&binding);
    if(status)[NSException raise:NSInvalidArgumentException format:@"page operand: %s",ncclGetLastError(NULL)];
    result->_buffer=[(id<MTLBuffer>)binding.buffer retain];
    result->_origin=binding.offset;
    return result;
}
+ (instancetype)host:(const void *)address {
    ncclMeshPageTable_t table=NULL;
    if(ncclMeshPageTableCreate(address,NULL,0,&table))
        [NSException raise:NSInvalidArgumentException format:@"page operand: %s",ncclGetLastError(NULL)];
    MetalTensorPages *result=[self adopt:table];
    result->_origin=(uintptr_t)address-(uintptr_t)result->_buffer.contents;
    return result;
}
+ (instancetype)storage:(id<MTLBuffer>)buffer {
    MetalTensorPages *result=[self host:buffer.contents];
    result->_owner=[buffer retain];return result;
}
- (MetalTensorLayout *)backing {
    size_t count=0;
    if(ncclMeshPageTableBacking(_table,NULL,&count))
        [NSException raise:NSInvalidArgumentException format:@"page backing: %s",ncclGetLastError(NULL)];
    NSMutableData *data=[NSMutableData dataWithLength:count*sizeof(ncclMeshPageBacking)];
    ncclMeshPageBacking *runs=data.mutableBytes;
    if(ncclMeshPageTableBacking(_table,runs,&count))
        [NSException raise:NSInvalidArgumentException format:@"page backing: %s",ncclGetLastError(NULL)];
    NSMutableArray *parts=[NSMutableArray array];
    for(size_t i=0;i<count;i++)[parts addObject:@[@(runs[i].page),@(runs[i].count),@0,@1]];
    return [MetalTensorLayout runs:parts];
}
+ (MetalTensorLayout *)domain:(NSArray<MetalTensorPages *> *)sources {
    NSMutableArray *domains=[NSMutableArray array],*parts=[NSMutableArray array];
    for(MetalTensorPages *source in sources)[domains addObject:[source backing]];
    for(NSDictionary *cell in MetalTensorCover(domains)) {
        NSData *physical=[cell[@"domain"] orderedSpans];const struct MetalTensorSpan *pages=physical.bytes;
        for(NSUInteger k=0;k<physical.length/sizeof *pages;k++)[parts addObject:@[@(pages[k].offset),@(pages[k].count),@0,@1]];
    }
    return [MetalTensorLayout runs:parts];
}
+ (instancetype)compose:(NSArray<MetalTensorPages *> *)sources order:(MetalTensorLayout *)order alignment:(NSUInteger)alignment {
    NSMutableArray *parts=[NSMutableArray array],*local=[NSMutableArray array];
    for(MetalTensorPages *source in sources) {
        MetalTensorLayout *domain=[source backing];
        [local addObject:[MetalTensorLayout shape:domain.shape strides:@[@1] offset:0 maps:@{} indices:nil]];
        NSData *physical=[domain orderedSpans];const struct MetalTensorSpan *pages=physical.bytes;
        for(NSUInteger k=0;k<physical.length/sizeof *pages;k++)[parts addObject:@[@(pages[k].offset),@(pages[k].count),@0,@1]];
    }
    MetalTensorTraversal *traversal=[MetalTensorTraversal layouts:local domain:[MetalTensorLayout runs:parts] selection:order];
    NSMutableData *selections=[NSMutableData data];NSArray<NSNumber *> *cuts=traversal.cuts;
    for(NSUInteger k=1;k<cuts.count;k++) {
        const struct MetalTensorFragment part=[traversal fragment:cuts[k-1].unsignedIntegerValue end:cuts[k].unsignedIntegerValue];
        if(!part.count)continue;
        const ncclMeshPageSelection selection={sources[part.part].table,(uint64_t)part.offset,part.count};
        [selections appendBytes:&selection length:sizeof selection];
    }
    ncclMeshPageTable_t table=NULL;
    if(ncclMeshPageTableCompose(selections.bytes,selections.length/sizeof(ncclMeshPageSelection),alignment,NULL,0,&table))
        [NSException raise:NSInvalidArgumentException format:@"page composition: %s",ncclGetLastError(NULL)];
    return [self adopt:table];
}
- (MetalTensorLayout *)physical:(MetalTensorLayout *)layout elementBytes:(NSUInteger)bytes {
    MetalTensorLayout *physical=[[[self backing] components:_pageBytes] reshape:@[@(_pageCount*_pageBytes)]];
    MetalTensorLayout *ordinals=[MetalTensorLayout shape:physical.shape strides:@[@1] offset:0 maps:@{} indices:nil];
    return [[layout components:bytes] rebase:ordinals onto:physical];
}
- (void)dealloc { ncclMeshPageTableDestroy(_table);[_buffer release];[_owner release];[super dealloc]; }
- (MetalTensorLayout *)bind:(MetalTensorLayout *)layout elementBytes:(NSUInteger)bytes {
    NSParameterAssert(bytes&&_origin%bytes==0);
    return [layout shift:_origin/bytes];
}
@end

/* mesh-tensor-layout.h MetalTensorLayoutRecord. A layout is its exact state: logical and index shapes, strides, origin,
   maps, and its coordinates where it is not factored (a factored layout's are its digits). Numbers, strings, arrays and
   dictionaries are as they are. Added by the layout algebra session: the LSTM's declaration spent 14.5 s on rank 0 in
   layout calls that a desk replay of hand-built inputs did not reproduce (2026-11-02). A log nothing reads back. */
static id TensorRecordValue(id value) {
    if([value isKindOfClass:MetalTensorLayout.class]) {
        MetalTensorLayout *layout=value;NSMutableDictionary *maps=[NSMutableDictionary dictionary];
        for(NSNumber *axis in layout.maps) {
            NSData *data=layout.maps[axis];const NSInteger *words=data.bytes;NSMutableArray *values=[NSMutableArray array];
            for(NSUInteger k=0;k<data.length/sizeof *words;k++)[values addObject:@(words[k])];
            maps[axis.stringValue]=values;
        }
        NSMutableDictionary *state=[NSMutableDictionary dictionaryWithDictionary:@{@"shape":layout.shape,
            @"index_shape":layout.indexShape,@"strides":layout.strides,@"offset":@(layout.offset),@"maps":maps,
            @"factored":@(layout.factored)}];
        if(!layout.factored)state[@"index_coordinates"]=layout.coordinates;
        return state;
    }
    if([value isKindOfClass:NSArray.class]) {
        NSMutableArray *items=[NSMutableArray array];for(id item in value)[items addObject:TensorRecordValue(item)];return items;
    }
    if([value isKindOfClass:NSDictionary.class]) {
        NSMutableDictionary *items=[NSMutableDictionary dictionary];
        for(id key in value)items[[key description]]=TensorRecordValue(value[key]);
        return items;
    }
    if([value isKindOfClass:NSNumber.class]||[value isKindOfClass:NSString.class])return value;
    return value?[value description]:NSNull.null;
}
/* mesh-tensor-layout.h MetalTensorLayoutEnter and MetalTensorLayoutRecord. The operands are serialized once, at entry:
   written whole to layouts-current-<pid>.json (a temporary file renamed over it, so a rank ended mid-call leaves that
   call's operands complete), and kept for the line appended at return where the call took over 50 ms. Changed by the
   layout algebra session: the LSTM's stalled calls never returned within the 8 s quiet bound, so a line written at
   return left nothing (d-rec1, 2026-11-02). */
static const char *TensorRecordDirectory(void) {
    static const char *dir;static int read;
    if(!read){read=1;dir=getenv("MESH_PREPARE_RECORD");}
    return dir;
}
static _Thread_local NSData *TensorRecordOperands;
uint64_t MetalTensorLayoutEnter(const char *call,NSDictionary *operands) { @autoreleasepool {
    const char *dir=TensorRecordDirectory();
    if(dir) {
        NSData *data=[NSJSONSerialization dataWithJSONObject:TensorRecordValue(operands) options:0 error:nil];
        [TensorRecordOperands release];TensorRecordOperands=[data retain];
        char path[1024],temporary[1040];const int pid=(int)getpid();
        if(data&&snprintf(path,sizeof path,"%s/layouts-current-%d.json",dir,pid)<(int)sizeof path&&
           snprintf(temporary,sizeof temporary,"%s.partial",path)<(int)sizeof temporary) {
            FILE *out=fopen(temporary,"w");
            if(out) {
                struct timespec wall;clock_gettime(CLOCK_REALTIME,&wall);
                fprintf(out,"{\"call\":\"%s\",\"entered\":%.6f,\"operands\":",call,(double)wall.tv_sec+wall.tv_nsec/1e9);
                fwrite(data.bytes,1,data.length,out);fputs("}\n",out);
                if(!fclose(out))rename(temporary,path);
            }
        }
    }
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}}
void MetalTensorLayoutRecord(const char *call,uint64_t started) {
    const double us=(double)(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)-started)/1e3;
    NSData *data=TensorRecordOperands;
    if(us<=50000||!data)return;
    static FILE *out;static int opened;
    if(!opened) {
        opened=1;const char *dir=TensorRecordDirectory();char path[1024];
        if(dir&&snprintf(path,sizeof path,"%s/layouts-%d.jsonl",dir,(int)getpid())<(int)sizeof path)out=fopen(path,"w");
    }
    if(!out)return;
    fprintf(out,"{\"call\":\"%s\",\"us\":%.0f,\"operands\":",call,us);
    fwrite(data.bytes,1,data.length,out);fputs("}\n",out);fflush(out);
}

