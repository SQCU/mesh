#import "metal-recording.h"
#import "mesh-index-relation.h"
#include <isl/ilp.h>
#include <isl/val.h>

struct TensorRun { NSUInteger first, end, part; NSInteger offset; };
@implementation MetalTensorTraversal {
    NSData *_runs;
    NSArray *_domains;
}
+ (instancetype)layouts:(NSArray<MetalTensorLayout *> *)layouts domains:(NSArray<NSArray<NSNumber *> *> *)domains {
    MetalTensorTraversal *result=[[[self alloc] init] autorelease];
    NSMutableData *runs=[NSMutableData data];NSMutableArray *cuts=[NSMutableArray array];
    NSUInteger count=0;
    for(NSUInteger part=0;part<layouts.count;part++) {
        NSData *data=[layouts[part] orderedSpans];const struct MetalTensorSpan *spans=data.bytes;
        for(NSUInteger k=0;k<data.length/sizeof *spans;k++) {
            struct TensorRun run={count,count+spans[k].count,part,spans[k].offset};
            if(spans[k].count)[runs appendBytes:&run length:sizeof run];
            count=run.end;
        }
    }
    NSUInteger total=0;
    for(NSArray *domain in domains) {
        NSUInteger first=[domain[0] unsignedIntegerValue],end=[domain[1] unsignedIntegerValue];
        NSParameterAssert(end>=first);total+=end-first;
    }
    NSParameterAssert(total==count);
    const struct TensorRun *values=runs.bytes;
    for(NSUInteger k=0;k+1<runs.length/sizeof *values;k++) {
        NSUInteger local=values[k].end;
        for(NSArray *domain in domains) {
            NSUInteger first=[domain[0] unsignedIntegerValue],end=[domain[1] unsignedIntegerValue];
            if(local<=end-first){[cuts addObject:@(first+local)];break;}
            local-=end-first;
        }
    }
    result->_runs=[runs copy];result->_domains=[domains copy];
    result->_cuts=[[[[NSSet setWithArray:cuts] allObjects] sortedArrayUsingSelector:@selector(compare:)] copy];
    return result;
}
- (void)dealloc { [_runs release];[_domains release];[_cuts release];[super dealloc]; }
- (struct MetalTensorFragment)fragment:(NSUInteger)first end:(NSUInteger)end {
    NSUInteger lo=0,hi=0;
    for(NSArray *domain in _domains) {
        NSUInteger start=[domain[0] unsignedIntegerValue],stop=[domain[1] unsignedIntegerValue];
        lo+=MIN(MAX(first,start),stop)-start;hi+=MIN(MAX(end,start),stop)-start;
    }
    if(lo==hi)return (struct MetalTensorFragment){0,0,0};
    const struct TensorRun *runs=_runs.bytes;NSUInteger left=0,right=_runs.length/sizeof *runs;
    while(left<right){NSUInteger middle=left+(right-left)/2;if(runs[middle].end<=lo)left=middle+1;else right=middle;}
    NSParameterAssert(left<_runs.length/sizeof *runs&&hi<=runs[left].end);
    return (struct MetalTensorFragment){runs[left].part,hi-lo,runs[left].offset+(NSInteger)(lo-runs[left].first)};
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
    NSString *op=expression[0];
    if(TensorGeneral(expression)) {
        struct TensorExpressionRange range=TensorRange(expression,count);
        if(range.low==range.high)return @(range.low);
        NSMutableArray *terms=[NSMutableArray arrayWithObject:op];
        for(NSUInteger k=1;k<[expression count];k++)[terms addObject:TensorBound(expression[k],count)];
        return terms;
    }
    if([op isEqual:@"add"]){NSMutableArray *terms=[NSMutableArray array];for(NSUInteger k=1;k<[expression count];k++)[terms addObject:TensorBound(expression[k],count)];return TensorSum(terms);}
    id value=TensorBound(expression[1],count);NSInteger n=[expression[2] integerValue];
    if([op isEqual:@"mul"])return TensorScale(value,n);
    struct TensorExpressionRange bounds=TensorRange(value,count);NSInteger low=TensorFloor(bounds.low,n),high=TensorFloor(bounds.high,n);
    if(low==high)return [op isEqual:@"div"]?@(low):TensorSum(@[value,@(-low*n)]);
    return TensorQuotient(value,n,[op isEqual:@"mod"]);
}
static id TensorNormalize(id expression,NSUInteger count) {
    id value=TensorSubstitute(expression,@"i");
    for(;;) {
        id bounded=TensorBound(value,count);if([bounded isEqual:value])return bounded;
        value=bounded;
    }
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

// Reduce a composed relation to the ordinary emitted layout. The solver's
// context ends here; the result keeps only index terms and declared constants.
static MetalTensorLayout *TensorRelationLayout(isl_map *relation,NSArray *shape) {
    if(!TensorCount(shape)) {
        isl_map_free(relation);
        return [MetalTensorLayout shape:shape strides:TensorStrides(shape) offset:0 maps:@{} indices:nil];
    }
    isl_pw_multi_aff *function=isl_pw_multi_aff_from_map(relation);
    isl_pw_aff *value=isl_pw_multi_aff_get_pw_aff(function,0);isl_pw_multi_aff_free(function);
    isl_val *minimum=isl_pw_aff_min_val(isl_pw_aff_copy(value)),*maximum=isl_pw_aff_max_val(isl_pw_aff_copy(value));
    NSInteger low=isl_val_get_num_si(minimum),high=isl_val_get_num_si(maximum);isl_val_free(minimum);isl_val_free(maximum);
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
        struct MeshIndexRun run=MeshIndexRunOf(expression,at,end);
        return (struct TensorPiece){run.value,run.step,run.end};
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

// Weighted union-find: position(b) - position(a) = distance. These equations
// exist only during placement; the result is the ordinary shared layout.
struct TensorPosition { NSUInteger parent, size; NSInteger delta; };
static NSUInteger TensorRoot(struct TensorPosition *nodes,NSUInteger i) {
    if(nodes[i].parent!=i) {
        const NSUInteger p=nodes[i].parent;
        nodes[i].parent=TensorRoot(nodes,p);nodes[i].delta+=nodes[p].delta;
    }
    return nodes[i].parent;
}
static BOOL TensorJoin(struct TensorPosition *nodes,NSUInteger a,NSUInteger b,NSInteger distance) {
    const NSUInteger x=TensorRoot(nodes,a),y=TensorRoot(nodes,b);
    const NSInteger shift=distance+nodes[a].delta-nodes[b].delta;
    if(x==y)return shift==0;
    if(nodes[x].size>=nodes[y].size) {
        nodes[y].parent=x;nodes[y].delta=shift;nodes[x].size+=nodes[y].size;
    } else { nodes[x].parent=y;nodes[x].delta=-shift;nodes[y].size+=nodes[x].size; }
    return YES;
}
static NSInteger TensorMod(NSInteger value,NSInteger modulus) {
    const NSInteger remainder=value%modulus;return remainder<0?remainder+modulus:remainder;
}
struct TensorPlacementGroup { NSInteger low, high, phase, modulus, base; };
static BOOL TensorAlign(struct TensorPlacementGroup *group,NSInteger phase,NSInteger modulus) {
    // Combine two alignment congruences, including widths with different factors.
    NSInteger a=group->modulus,b=modulus,x=1,y=0;
    while(b){const NSInteger q=a/b,t=a%b;a=b;b=t;const NSInteger next=x-q*y;x=y;y=next;}
    const NSInteger difference=phase-group->phase;
    if(difference%a)return NO;
    const NSInteger factor=modulus/a;
    const NSInteger step=factor==1?0:TensorMod((difference/a)*x,factor);
    group->phase+=group->modulus*step;group->modulus*=factor;
    group->phase=TensorMod(group->phase,group->modulus);return YES;
}
static int TensorPositionCompare(const void *left,const void *right) {
    const NSInteger a=((const struct TensorPosition *)left)->delta,b=((const struct TensorPosition *)right)->delta;
    return (a>b)-(a<b);
}
MetalTensorLayout *MetalTensorPlace(NSUInteger count,NSArray<MetalTensorLayout *> *uses,
    NSArray<MetalTensorLayout *> *targets,NSArray<MetalTensorLayout *> *values,NSArray<NSNumber *> *widths) {
    NSCParameterAssert(uses.count==targets.count&&values.count==widths.count);
    struct TensorPosition *nodes=calloc(count?count:1,sizeof *nodes);
    struct TensorPlacementGroup *groups=calloc(count?count:1,sizeof *groups);
    MetalTensorLayout *result=nil;
    if(!nodes||!groups){free(nodes);free(groups);[NSException raise:NSMallocException format:@"tensor placement metadata"];}
    for(NSUInteger i=0;i<count;i++) {
        nodes[i]=(struct TensorPosition){i,1,0};
        groups[i]=(struct TensorPlacementGroup){NSIntegerMax,NSIntegerMin,0,1,0};
    }
    // A typed scalar's components stay adjacent even when no API constrains
    // that part of the allocation. Differently typed aliases contribute here too.
    for(NSUInteger k=0;k<values.count;k++) {
        MetalTensorLayout *value=values[k];const NSUInteger width=widths[k].unsignedIntegerValue;
        NSCParameterAssert(width&&value.count%width==0);
        for(NSUInteger i=0;i<value.count;i+=width) {
            const NSUInteger first=(NSUInteger)[value index:i];NSCParameterAssert(first<count);
            for(NSUInteger j=1;j<width;j++) {
                const NSUInteger next=(NSUInteger)[value index:i+j];NSCParameterAssert(next<count);
                if(!TensorJoin(nodes,first,next,(NSInteger)j))goto finished;
            }
        }
    }
    for(NSUInteger k=0;k<uses.count;k++) {
        MetalTensorLayout *use=uses[k],*target=targets[k];NSCParameterAssert(use.count==target.count);
        if(!use.count)continue;
        const NSUInteger first=(NSUInteger)[use index:0];const NSInteger origin=[target index:0];
        NSCParameterAssert(first<count);
        for(NSUInteger i=1;i<use.count;i++) {
            const NSUInteger next=(NSUInteger)[use index:i];NSCParameterAssert(next<count);
            if(!TensorJoin(nodes,first,next,[target index:i]-origin))goto finished;
        }
    }
    for(NSUInteger i=0;i<count;i++) {
        const NSUInteger root=TensorRoot(nodes,i);const NSInteger offset=nodes[i].delta;
        groups[root].low=MIN(groups[root].low,offset);groups[root].high=MAX(groups[root].high,offset);
    }
    for(NSUInteger k=0;k<values.count;k++) {
        MetalTensorLayout *value=values[k];const NSUInteger width=widths[k].unsignedIntegerValue;
        for(NSUInteger i=0;i<value.count;i+=width) {
            const NSUInteger first=(NSUInteger)[value index:i],root=nodes[first].parent;
            if(!TensorAlign(groups+root,-nodes[first].delta,(NSInteger)width))goto finished;
        }
    }
    {
        NSInteger cursor=0;
        // Lay disjoint equation components beside one another. Their internal
        // positions are already fixed; unconstrained values retain distinct slots.
        for(NSUInteger i=0;i<count;i++)if(nodes[i].parent==i) {
            struct TensorPlacementGroup *g=groups+i;
            g->base=cursor-g->low;g->base+=TensorMod(g->phase-g->base,g->modulus);
            cursor=g->base+g->high+1;
        }
        NSMutableData *data=[NSMutableData dataWithLength:count*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
        for(NSUInteger i=0;i<count;i++)out[i]=nodes[i].delta+=groups[nodes[i].parent].base;
        // Distinct source values cannot occupy the same physical scalar slot.
        qsort(nodes,count,sizeof *nodes,TensorPositionCompare);
        for(NSUInteger i=1;i<count;i++)if(nodes[i-1].delta==nodes[i].delta)goto finished;
        result=[MetalTensorLayout shape:@[@(count)] strides:@[@1] offset:0 maps:@{} indices:data];
    }
finished:
    free(nodes);free(groups);return result;
}

@implementation MetalTensorLayout
@synthesize coordinates=_coordinates;
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
    isl_ctx *ctx=isl_ctx_alloc();id selection=MeshIndexTerms(MeshIndexRuns(ctx,runs));isl_ctx_free(ctx);
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
    isl_ctx *ctx=isl_ctx_alloc();
    isl_map *first=isl_map_from_pw_aff(MeshIndexExpression(ctx,@[@"mul",@"i",@(count)],0,_count/count));
    isl_map *relation=isl_map_apply_range(first,MeshIndexRelation(ctx,self));
    isl_pw_multi_aff *function=isl_pw_multi_aff_from_map(relation);
    isl_pw_aff *value=isl_pw_multi_aff_get_pw_aff(function,0);isl_pw_multi_aff_free(function);
    value=isl_pw_aff_floor(isl_pw_aff_scale_down_val(value,isl_val_int_from_ui(ctx,count)));
    MetalTensorLayout *result=TensorRelationLayout(isl_map_from_pw_aff(value),shape);isl_ctx_free(ctx);
    if(![[result components:count] equivalent:self])
        [NSException raise:NSInvalidArgumentException format:@"typed element components are not adjacent and aligned in their declared storage"];
    return result;
}
- (MetalTensorLayout *)rebase:(MetalTensorLayout *)declared onto:(MetalTensorLayout *)placed {
    NSParameterAssert([declared.shape isEqual:placed.shape]);
    if([self isEqual:declared])return placed;
    if(!self.count)return [MetalTensorLayout shape:_shape strides:TensorStrides(_shape) offset:0 maps:@{} indices:nil];
    isl_ctx *ctx=isl_ctx_alloc();
    isl_map *view=MeshIndexRelation(ctx,self),*source=MeshIndexRelation(ctx,declared),*target=MeshIndexRelation(ctx,placed);
    isl_set *domain=isl_map_domain(isl_map_copy(view));
    // The inverse is a relation. Selecting its last preimage retains the
    // existing alias convention where several logical positions name a value.
    // Repeated consumer indices remain repeated in the composed result.
    isl_map *inverse=isl_map_lexmax(isl_map_reverse(source));
    isl_map *result=isl_map_apply_range(isl_map_apply_range(view,inverse),target);
    isl_set *resolved=isl_map_domain(isl_map_copy(result));
    isl_bool complete=isl_set_is_equal(domain,resolved);isl_set_free(domain);isl_set_free(resolved);
    if(complete!=isl_bool_true) {
        isl_map_free(result);isl_ctx_free(ctx);
        [NSException raise:NSInvalidArgumentException format:@"operand alias names storage outside its declared producer"];
    }
    MetalTensorLayout *layout=TensorRelationLayout(result,_shape);isl_ctx_free(ctx);return layout;
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
    MetalTensorLayout *result=self;
    for(NSDictionary *r in requirements)result=[result required:r[@"kind"] strides:r[@"strides"]];
    for(NSDictionary *r in requirements)
        if(![[result required:r[@"kind"] strides:r[@"strides"]] isEqual:result])
            [NSException raise:NSInvalidArgumentException format:@"conflicting operand layout requirements: %@",requirements];
    return result;
}
- (NSString *)declaration:(NSUInteger)binding {
    NSMutableString *result=[NSMutableString string];
    NSData *shape=TensorWords(_indexShape),*steps=TensorWords(_strides);
    NSMutableArray *coordinates=nil;
    if(_coordinates){coordinates=[NSMutableArray array];for(id value in _coordinates)[coordinates addObject:TensorExpressionSource(value,_count)];}
    [result appendString:MetalViewMappedIndexDeclaration(binding,shape.bytes,steps.bytes,_indexShape.count,(NSUInteger)_offset,_maps,
        coordinates,coordinates?TensorVectorSpan(self):0)];
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
    if(_maps.count||_coordinates)return nil; // The fixed-axis constant form cannot express composed coordinates.
    NSData *shape=TensorWords(_indexShape),*steps=TensorWords(_strides);
    return MetalViewIndexConstants(binding,shape.bytes,steps.bytes,_indexShape.count,(NSUInteger)_offset);
}
- (NSString *)declaration:(NSUInteger)binding fromStorage:(MetalTensorLayout *)source {
    NSParameterAssert([source.shape isEqual:_shape]);
    if(!_count)return [[self declaration:binding] stringByAppendingFormat:@"inline long mesh_storage_%lu(long) { return 0; }\n",(unsigned long)binding];
    isl_ctx *ctx=isl_ctx_alloc();
    isl_map *inverse=isl_map_lexmax(isl_map_reverse(MeshIndexRelation(ctx,source)));
    isl_map *relation=isl_map_apply_range(inverse,MeshIndexRelation(ctx,self));
    isl_pw_multi_aff *function=isl_pw_multi_aff_from_map(relation);
    id terms=MeshIndexTerms(isl_pw_multi_aff_get_pw_aff(function,0));isl_pw_multi_aff_free(function);
    // Storage indices can have gaps and can start below zero. The emitted
    // expression is valid on the declared source image; it adds no validity guard.
    isl_set *image=isl_map_range(MeshIndexRelation(ctx,source));
    isl_val *minimum=isl_set_dim_min_val(isl_set_copy(image),0),*maximum=isl_set_dim_max_val(image,0);
    NSInteger low=isl_val_get_num_si(minimum),high=isl_val_get_num_si(maximum);
    isl_val_free(minimum);isl_val_free(maximum);isl_ctx_free(ctx);
    terms=TensorSubstitute(terms,TensorSum(@[@"i",@(low)]));
    return [[self declaration:binding] stringByAppendingFormat:@"inline long mesh_storage_%lu(long storage) { long i = storage - %ldl; return %@; }\n",
        (unsigned long)binding,(long)low,TensorExpressionSource(terms,MAX(1,high-low+1))];
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
    if(!_count){struct MetalReach empty={buffer,base+_offset*(NSInteger)bytes,0,access|MetalReachBinding,binding};return [NSData dataWithBytes:&empty length:sizeof empty];}
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
+ (instancetype)host:(const void *)address {
    MetalTensorPages *result=[[[self alloc] init] autorelease];
    ncclMeshBuffer binding={0};
    ncclResult_t status=ncclMeshPageTableCreate(address,NULL,0,&result->_table);
    if(!status)status=ncclMeshPageTableShape(result->_table,&result->_pageCount,&result->_pageBytes);
    if(!status)status=ncclMeshPageTableBinding(result->_table,&binding);
    if(status)[NSException raise:NSInvalidArgumentException format:@"page operand: %s",ncclGetLastError(NULL)];
    result->_buffer=[(id<MTLBuffer>)binding.buffer retain];
    result->_origin=(uintptr_t)address-(uintptr_t)result->_buffer.contents;
    return result;
}
+ (instancetype)storage:(id<MTLBuffer>)buffer {
    MetalTensorPages *result=[self host:buffer.contents];
    result->_owner=[buffer retain];return result;
}
- (void)dealloc { ncclMeshPageTableDestroy(_table);[_buffer release];[_owner release];[super dealloc]; }
- (MetalTensorLayout *)bind:(MetalTensorLayout *)layout elementBytes:(NSUInteger)bytes {
    NSParameterAssert(bytes&&_origin%bytes==0);
    return [layout shift:_origin/bytes];
}
@end
