#import "metal-recording.h"

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

struct TensorReshapeChunk { NSUInteger count; NSInteger step; NSData *map; };
static MetalTensorLayout *TensorReshape(MetalTensorLayout *source,NSArray<NSNumber *> *shape) {
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
    return [result reshape:shape];
}
static MetalTensorLayout *TensorAxisTransform(MetalTensorLayout *source,NSUInteger axis,
    MetalTensorLayout *(^transform)(MetalTensorLayout *,NSUInteger)) {
    if(!source.count||[source.shape isEqual:source.indexShape])return transform(source,axis);
    NSData *storage=TensorSections(source);const struct TensorSection *sections=storage.bytes;
    NSMutableArray *parts=[NSMutableArray array];
    for(NSUInteger k=0;k<storage.length/sizeof *sections;k++) {
        const struct TensorSection section=sections[k];MetalTensorLayout *part=TensorSectionLayout(source,section);
        if(axis>=section.first&&axis<section.end)part=transform(part,axis-section.first);
        [parts addObject:part];
    }
    return [TensorProduct(parts) shift:source.offset];
}
static MetalTensorLayout *TensorSelect(MetalTensorLayout *source,NSUInteger axis,NSData *indices) {
    const NSUInteger count=indices.length/sizeof(NSInteger);const NSInteger *selected=indices.bytes;
    NSMutableArray *shape=[[source.shape mutableCopy] autorelease];shape[axis]=@(count);
    if([source.shape isEqual:source.indexShape]) {
        NSMutableDictionary *maps=[[source.maps mutableCopy] autorelease];
        NSMutableData *data=[NSMutableData dataWithLength:indices.length];NSInteger *target=data.mutableBytes;
        const NSInteger *values=((NSData *)source.maps[@(axis)]).bytes;const NSInteger step=[source.strides[axis] integerValue];
        for(NSUInteger i=0;i<count;i++)target[i]=values?values[selected[i]]:selected[i]*step;
        maps[@(axis)]=data;
        return [MetalTensorLayout shape:shape strides:source.strides offset:source.offset maps:maps indices:nil];
    }
    const NSUInteger total=TensorCount(shape);NSArray *steps=TensorStrides(source.shape),*newSteps=TensorStrides(shape);
    NSMutableData *data=[NSMutableData dataWithLength:total*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
    for(NSUInteger i=0;i<total;i++) {
        NSUInteger old=0;
        for(NSUInteger d=0;d<shape.count;d++) {
            NSUInteger at=(i/[newSteps[d] unsignedIntegerValue])%[shape[d] unsignedIntegerValue];
            old+=(d==axis?selected[at]:at)*[steps[d] unsignedIntegerValue];
        }
        out[i]=[source index:old];
    }
    return [MetalTensorLayout shape:shape strides:newSteps offset:0 maps:@{} indices:data];
}
static MetalTensorLayout *TensorSlice(MetalTensorLayout *source,NSUInteger axis,NSInteger first,NSUInteger count,NSInteger step) {
    if([source.shape isEqual:source.indexShape]&&!source.maps[@(axis)]) {
        NSMutableArray *shape=[[source.shape mutableCopy] autorelease],*strides=[[source.strides mutableCopy] autorelease];
        shape[axis]=@(count);strides[axis]=@(step*[source.strides[axis] integerValue]);
        return [MetalTensorLayout shape:shape strides:strides offset:source.offset+first*[source.strides[axis] integerValue]
            maps:source.maps indices:nil];
    }
    NSMutableData *data=[NSMutableData dataWithLength:count*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
    for(NSUInteger i=0;i<count;i++)out[i]=first+(NSInteger)i*step;
    return TensorSelect(source,axis,data);
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
    if([source.shape isEqual:source.indexShape]) {
        NSMutableArray *strides=[NSMutableArray array];NSMutableDictionary *maps=[NSMutableDictionary dictionary];
        for(NSUInteger d=0;d<axes.count;d++) {
            NSNumber *axis=axes[d];[strides addObject:source.strides[axis.unsignedIntegerValue]];
            if(source.maps[axis])maps[@(d)]=source.maps[axis];
        }
        return [MetalTensorLayout shape:shape strides:strides offset:source.offset maps:maps indices:nil];
    }
    NSArray *old=TensorStrides(source.shape),*steps=TensorStrides(shape);
    NSMutableData *data=[NSMutableData dataWithLength:source.count*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
    for(NSUInteger i=0;i<source.count;i++) {
        NSUInteger at=0;for(NSUInteger d=0;d<axes.count;d++)at+=(i/[steps[d] unsignedIntegerValue])%[shape[d] unsignedIntegerValue]*[old[[axes[d] unsignedIntegerValue]] unsignedIntegerValue];
        out[i]=[source index:at];
    }
    return [MetalTensorLayout shape:shape strides:steps offset:0 maps:@{} indices:data];
}

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

static BOOL TensorCoordinates(MetalTensorLayout *layout,NSArray *axes,NSInteger low,NSInteger address,NSInteger *out) {
    NSInteger remainder=address-low;if(remainder<0||!layout.count)return NO;
    memset(out,0,layout.shape.count*sizeof *out);
    for(NSNumber *axis in axes) {
        const NSUInteger d=axis.unsignedIntegerValue,n=[layout.shape[d] unsignedIntegerValue];
        const NSInteger s=[layout.strides[d] integerValue],q=remainder/labs(s);
        if((NSUInteger)q>=n)return NO;
        remainder-=q*labs(s);out[d]=s<0?(NSInteger)n-1-q:q;
    }
    return remainder==0;
}

static MetalTensorLayout *TensorCoordinateRebase(MetalTensorLayout *view,MetalTensorLayout *declared,MetalTensorLayout *placed) {
    if(!view.affine||!declared.affine)return nil;
    for(NSUInteger d=0;d<declared.shape.count;d++)
        if([declared.shape[d] unsignedIntegerValue]>1&&![declared.strides[d] integerValue])return nil;
    NSInteger low;NSArray *axes=TensorInverseAxes(declared,&low);NSInteger reach=1;
    // The coordinate inverse must be unique. Gaps, negative strides and
    // singleton axes are all allowed; overlapping source axes are not a basis.
    for(NSNumber *axis in [axes reverseObjectEnumerator]) {
        const NSUInteger d=axis.unsignedIntegerValue;const NSInteger step=labs([declared.strides[d] integerValue]);
        if(step<reach)return nil;
        reach+=([declared.shape[d] integerValue]-1)*step;
    }
    const NSUInteger dimensions=declared.shape.count;
    NSMutableData *storage=[NSMutableData dataWithLength:MAX(1,dimensions*(view.shape.count+1))*sizeof(NSInteger)];
    NSInteger *base=storage.mutableBytes;
    if(!TensorCoordinates(declared,axes,low,view.offset,base))return nil;
    NSMutableArray *steps=[NSMutableArray array];
    for(NSUInteger d=0;d<view.shape.count;d++) {
        NSInteger *delta=base+(d+1)*dimensions,step=0;
        if([view.shape[d] unsignedIntegerValue]>1) {
            if(!TensorCoordinates(declared,axes,low,view.offset+[view.strides[d] integerValue],delta))return nil;
            for(NSUInteger a=0;a<dimensions;a++) {
                delta[a]-=base[a];if(placed.affine)step+=delta[a]*[placed.strides[a] integerValue];
            }
        }
        [steps addObject:@(step)];
    }
    NSInteger origin=placed.offset;
    for(NSUInteger a=0;a<dimensions;a++) {
        NSInteger first=base[a],last=base[a];
        for(NSUInteger d=0;d<view.shape.count;d++) {
            const NSInteger span=base[(d+1)*dimensions+a]*([view.shape[d] integerValue]-1);
            first+=MIN(0,span);last+=MAX(0,span);
        }
        if(first<0||last>=[declared.shape[a] integerValue])return nil;
        if(placed.affine)origin+=base[a]*[placed.strides[a] integerValue];
    }
    if(placed.affine)return [MetalTensorLayout shape:view.shape strides:steps offset:origin maps:@{} indices:nil];
    // A rectangular alias of an affine declaration selects the same logical
    // coordinates after mapped placement. Compose the ordinary view operations;
    // do not enumerate the producer's full storage domain to invert its map.
    NSMutableDictionary *variables=[NSMutableDictionary dictionary];NSMutableIndexSet *used=[NSMutableIndexSet indexSet];
    for(NSUInteger d=0;d<view.shape.count;d++) {
        NSInteger found=-1;const NSInteger *delta=base+(d+1)*dimensions;
        for(NSUInteger a=0;a<dimensions;a++)if(delta[a]) {
            if(found>=0||variables[@(a)])return nil;
            found=(NSInteger)a;
        }
        if(found>=0){variables[@(found)]=@(d);[used addIndex:d];}
    }
    MetalTensorLayout *result=placed;NSMutableArray *order=[NSMutableArray array];
    for(NSUInteger a=0;a<dimensions;a++) {
        NSNumber *variable=variables[@(a)];const NSUInteger d=variable.unsignedIntegerValue;
        result=[result slice:a first:base[a] count:variable?[view.shape[d] unsignedIntegerValue]:1
            step:variable?base[(d+1)*dimensions+a]:1];
        if(variable)[order addObject:variable];
    }
    for(NSUInteger a=dimensions;a--;)if(!variables[@(a)])result=[result element:a at:0];
    NSMutableArray *broadcast=[NSMutableArray array],*zero=[NSMutableArray array],*permutation=[NSMutableArray array];
    NSMutableArray *positions=[NSMutableArray array];
    for(NSUInteger d=0;d<view.shape.count;d++)if(![used containsIndex:d]) {
        [broadcast addObject:view.shape[d]];[zero addObject:@0];[positions addObject:@(d)];
    }
    [positions addObjectsFromArray:order];
    for(NSUInteger d=0;d<view.shape.count;d++)[permutation addObject:@([positions indexOfObject:@(d)])];
    MetalTensorLayout *repeated=[MetalTensorLayout shape:broadcast strides:zero offset:0 maps:@{} indices:nil];
    return [TensorProduct(@[repeated,result]) permute:permutation];
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
- (void)dealloc { [_shape release];[_indexShape release];[_strides release];[_maps release];[super dealloc]; }
// Copy the flat index factors, never a parent layout or a chain of transforms.
- (MetalTensorLayout *)withShape:(NSArray<NSNumber *> *)shape offset:(NSInteger)offset {
    MetalTensorLayout *result=[[[MetalTensorLayout alloc] init] autorelease];
    result->_shape=[shape copy];result->_indexShape=[_indexShape copy];result->_count=_count;
    result->_strides=[_strides copy];result->_maps=[_maps copy];result->_offset=offset;
    return result;
}
- (id)copyWithZone:(NSZone *)zone { return [self retain]; }
- (BOOL)affine { return !_maps.count&&[_shape isEqual:_indexShape]; }
- (BOOL)isEqual:(id)object {
    if(![object isKindOfClass:MetalTensorLayout.class])return NO;
    MetalTensorLayout *other=object;
    if(_offset!=other.offset||![_shape isEqual:other.shape]||![_indexShape isEqual:other.indexShape]||![_maps isEqual:other.maps])return NO;
    for(NSUInteger d=0;d<_indexShape.count;d++)if([_indexShape[d] unsignedIntegerValue]>1&&![_strides[d] isEqual:other.strides[d]])return NO;
    return YES;
}
- (NSUInteger)hash {
    NSUInteger hash=(_shape.hash+_indexShape.hash)^_maps.hash^(NSUInteger)_offset;
    for(NSUInteger d=0;d<_indexShape.count;d++)if([_indexShape[d] unsignedIntegerValue]>1)hash^=[_strides[d] hash];
    return hash;
}
- (NSInteger)index:(NSUInteger)logical {
    NSInteger result=_offset;
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
    if(!self.count||[_shape isEqual:_indexShape]||TensorPermutationIsReshape(self,axes))return TensorPermute(self,axes);
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
    return [result reshape:shape];
}
- (MetalTensorLayout *)shift:(NSInteger)elements {
    return elements?[self withShape:_shape offset:_offset+elements]:self;
}
- (MetalTensorLayout *)elements:(NSUInteger)count {
    NSParameterAssert(count&&[_shape.lastObject unsignedIntegerValue]==count);
    NSMutableArray *shape=[[_shape mutableCopy] autorelease];[shape removeLastObject];
    MetalTensorLayout *factors=[_indexShape.lastObject unsignedIntegerValue]==count?[self reshape:_indexShape]:self;
    MetalTensorLayout *base=[factors element:factors.shape.count-1 at:0];
    NSParameterAssert(base.offset%(NSInteger)count==0);
    NSMutableArray *steps=[NSMutableArray array];
    for(NSUInteger d=0;d<base.indexShape.count;d++) {
        NSInteger step=[base.strides[d] integerValue];
        NSParameterAssert([base.indexShape[d] unsignedIntegerValue]<=1||base.maps[@(d)]||step%(NSInteger)count==0);
        [steps addObject:@(step/(NSInteger)count)];
    }
    NSData *(^divide)(NSData *)=^NSData *(NSData *data) {
        if(!data)return nil;
        NSMutableData *result=[NSMutableData dataWithLength:data.length];
        const NSInteger *source=data.bytes;NSInteger *target=result.mutableBytes;
        for(NSUInteger i=0;i<data.length/sizeof(NSInteger);i++) {
            NSParameterAssert(source[i]%(NSInteger)count==0);target[i]=source[i]/(NSInteger)count;
        }
        return result;
    };
    NSMutableDictionary *maps=[NSMutableDictionary dictionary];
    for(NSNumber *axis in base.maps)maps[axis]=divide(base.maps[axis]);
    MetalTensorLayout *result=[MetalTensorLayout shape:base.indexShape strides:steps offset:base.offset/(NSInteger)count maps:maps indices:nil];
    return [result reshape:shape];
}
- (MetalTensorLayout *)rebase:(MetalTensorLayout *)declared onto:(MetalTensorLayout *)placed {
    NSParameterAssert([declared.shape isEqual:placed.shape]);
    if([self isEqual:declared])return placed;
    if([[declared shift:-declared.offset] isEqual:[placed shift:-placed.offset]])
        return [self shift:placed.offset-declared.offset];
    if(_count==declared.count) {
        MetalTensorLayout *reshaped=[declared reshape:_shape];
        if([reshaped isEqual:self])return [placed reshape:_shape];
    }
    MetalTensorLayout *rebased=TensorCoordinateRebase(self,declared,placed);if(rebased)return rebased;
    NSArray *logical=TensorStrides(declared.shape);
    NSInteger low=0;NSArray *axes=declared.affine?TensorInverseAxes(declared,&low):@[];
    NSMutableDictionary *inverse=nil;
    NSMutableData *coordinates=[NSMutableData dataWithLength:MAX(1,declared.shape.count)*sizeof(NSInteger)];
    NSInteger *coordinate=coordinates.mutableBytes;
    NSMutableData *data=[NSMutableData dataWithLength:_count*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
    for(NSUInteger i=0;i<_count;i++) {
        const NSInteger address=[self index:i];NSUInteger at=0;
        const BOOL found=declared.affine&&TensorCoordinates(declared,axes,low,address,coordinate);
        if(found)for(NSUInteger d=0;d<declared.shape.count;d++)at+=coordinate[d]*[logical[d] unsignedIntegerValue];
        if(!found) {
            if(!inverse){inverse=[NSMutableDictionary dictionary];for(NSUInteger j=0;j<declared.count;j++)inverse[@([declared index:j])]=@(j);}
            NSNumber *index=inverse[@(address)];
            if(!index)[NSException raise:NSInvalidArgumentException format:@"operand alias names storage outside its declared producer"];
            at=index.unsignedIntegerValue;
        }
        out[i]=[placed index:at];
    }
    return [MetalTensorLayout shape:_shape strides:TensorStrides(_shape) offset:0 maps:@{} indices:data];
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
    [result appendString:MetalViewMappedIndexDeclaration(binding,shape.bytes,steps.bytes,_indexShape.count,(NSUInteger)_offset,_maps)];
    // APIs that require affine operands consume the strides of the placed
    // view, not strides captured before placement. These are shader constants.
    if(self.affine)for(NSUInteger axis=0;axis<_shape.count;axis++)
        [result appendFormat:@"template<> constexpr long mesh_stride<%lu, %lu>() { return %ldl; }\n",
            (unsigned long)binding,(unsigned long)axis,(long)[_strides[axis] integerValue]];
    return result;
}
- (NSString *)declaration:(NSUInteger)binding fromStorage:(MetalTensorLayout *)source {
    NSParameterAssert(source.affine&&[source.shape isEqual:_shape]);
    NSMutableString *result=[NSMutableString stringWithString:[self declaration:binding]];
    [result appendFormat:@"inline long mesh_storage_%lu(long i) {\n",(unsigned long)binding];
    if([source matches:TensorStrides(source.shape)]) {
        [result appendFormat:@"return mesh_index<%lu>(i - %ldl);\n}\n",(unsigned long)binding,(long)source.offset];
        return result;
    }
    NSArray *logical=TensorStrides(source.shape);
    NSInteger low;NSArray *axes=TensorInverseAxes(source,&low);
    [result appendFormat:@"long r = i - %ldl, at = 0;\n",(long)low];
    for(NSNumber *axis in axes) {
        NSUInteger d=axis.unsignedIntegerValue;NSInteger s=[source.strides[d] integerValue];
        [result appendFormat:@"{ long q = r / %ldl; r -= q * %ldl; at += ",(long)labs(s),(long)labs(s)];
        if(s<0)[result appendFormat:@"(%ldl - q)",(long)[source.shape[d] integerValue]-1];else [result appendString:@"q"];
        [result appendFormat:@" * %lul; }\n",(unsigned long)[logical[d] unsignedIntegerValue]];
    }
    [result appendFormat:@"return mesh_index<%lu>(at);\n}\n",(unsigned long)binding];
    return result;
}
- (NSData *)orderedSpans {
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
