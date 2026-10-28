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
    result->_shape=[shape copy];result->_count=TensorCount(shape);
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
        // One final physical-offset map, not a lookup through another map.
        const NSInteger *values=indices.bytes;const NSInteger first=values[0];
        NSMutableData *relative=[indices mutableCopy];NSInteger *out=relative.mutableBytes;
        for(NSUInteger i=0;i<result->_count;i++)out[i]-=first;
        result->_indices=relative;result->_offset=offset+first;
        result->_strides=[TensorStrides(shape) copy];result->_maps=[@{} copy];
    } else {
        NSData *words=TensorWords(strides);
        NSDictionary *normalized=MetalViewNormalizeAxes(words.bytes,shape.count,maps);
        const NSInteger *steps=((NSData *)normalized[@"strides"]).bytes;
        NSMutableArray *held=[NSMutableArray arrayWithCapacity:shape.count];
        for(NSUInteger d=0;d<shape.count;d++)[held addObject:@(steps[d])];
        result->_strides=[held copy];result->_maps=[normalized[@"maps"] copy];
        result->_offset=offset+[normalized[@"offset"] integerValue];
    }
    return result;
}
- (void)dealloc { [_shape release];[_strides release];[_maps release];[_indices release];[super dealloc]; }
- (id)copyWithZone:(NSZone *)zone { return [self retain]; }
- (BOOL)affine { return !_indices&&!_maps.count; }
- (BOOL)isEqual:(id)object {
    if(![object isKindOfClass:MetalTensorLayout.class])return NO;
    MetalTensorLayout *other=object;
    if(_offset!=other.offset||![_shape isEqual:other.shape]||![_maps isEqual:other.maps]
        ||!(_indices==other.indices||[_indices isEqual:other.indices]))return NO;
    for(NSUInteger d=0;d<_shape.count;d++)if([_shape[d] unsignedIntegerValue]>1&&![_strides[d] isEqual:other.strides[d]])return NO;
    return YES;
}
- (NSUInteger)hash {
    NSUInteger hash=_shape.hash^_maps.hash^_indices.hash^(NSUInteger)_offset;
    for(NSUInteger d=0;d<_shape.count;d++)if([_shape[d] unsignedIntegerValue]>1)hash^=[_strides[d] hash];
    return hash;
}
- (NSInteger)index:(NSUInteger)logical {
    if(_indices)return _offset+((const NSInteger *)_indices.bytes)[logical];
    NSInteger result=_offset;
    for(NSUInteger d=_shape.count;d--;) {
        const NSUInteger n=[_shape[d] unsignedIntegerValue],at=logical%n;logical/=n;
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
    const NSUInteger count=indices.length/sizeof(NSInteger);const NSInteger *selected=indices.bytes;
    NSMutableArray *shape=[[_shape mutableCopy] autorelease];shape[axis]=@(count);
    if(!_indices) {
        NSMutableDictionary *maps=[[_maps mutableCopy] autorelease];
        NSMutableData *data=[NSMutableData dataWithLength:indices.length];NSInteger *target=data.mutableBytes;
        const NSInteger *source=((NSData *)_maps[@(axis)]).bytes;const NSInteger step=[_strides[axis] integerValue];
        for(NSUInteger i=0;i<count;i++)target[i]=source?source[selected[i]]:selected[i]*step;
        maps[@(axis)]=data;
        return [MetalTensorLayout shape:shape strides:_strides offset:_offset maps:maps indices:nil];
    }
    const NSUInteger total=TensorCount(shape);NSArray *steps=TensorStrides(_shape),*newSteps=TensorStrides(shape);
    NSMutableData *data=[NSMutableData dataWithLength:total*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
    for(NSUInteger i=0;i<total;i++) {
        NSUInteger old=0;
        for(NSUInteger d=0;d<shape.count;d++) {
            NSUInteger at=(i/[newSteps[d] unsignedIntegerValue])%[shape[d] unsignedIntegerValue];
            old+=(d==axis?selected[at]:at)*[steps[d] unsignedIntegerValue];
        }
        out[i]=[self index:old];
    }
    return [MetalTensorLayout shape:shape strides:newSteps offset:0 maps:@{} indices:data];
}
- (MetalTensorLayout *)slice:(NSUInteger)axis first:(NSInteger)first count:(NSUInteger)count step:(NSInteger)step {
    if(!_indices&&!_maps[@(axis)]) {
        NSMutableArray *shape=[[_shape mutableCopy] autorelease],*strides=[[_strides mutableCopy] autorelease];
        shape[axis]=@(count);strides[axis]=@(step*[_strides[axis] integerValue]);
        return [MetalTensorLayout shape:shape strides:strides offset:_offset+first*[_strides[axis] integerValue] maps:_maps indices:nil];
    }
    NSMutableData *data=[NSMutableData dataWithLength:count*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
    for(NSUInteger i=0;i<count;i++)out[i]=first+(NSInteger)i*step;
    return [self select:axis indices:data];
}
- (MetalTensorLayout *)element:(NSUInteger)axis at:(NSUInteger)index {
    MetalTensorLayout *one=[self slice:axis first:index count:1 step:1];
    NSMutableArray *shape=[[one.shape mutableCopy] autorelease],*strides=[[one.strides mutableCopy] autorelease];
    [shape removeObjectAtIndex:axis];[strides removeObjectAtIndex:axis];
    NSMutableDictionary *maps=[NSMutableDictionary dictionary];
    for(NSNumber *d in one.maps)if(d.unsignedIntegerValue!=axis)maps[@(d.unsignedIntegerValue>axis?d.unsignedIntegerValue-1:d.unsignedIntegerValue)]=one.maps[d];
    return [MetalTensorLayout shape:shape strides:strides offset:one.offset maps:maps indices:one.indices];
}
- (MetalTensorLayout *)permute:(NSArray<NSNumber *> *)axes {
    NSMutableArray *shape=[NSMutableArray array],*strides=[NSMutableArray array];NSMutableDictionary *maps=[NSMutableDictionary dictionary];
    for(NSUInteger d=0;d<axes.count;d++) {
        const NSUInteger at=[axes[d] unsignedIntegerValue];[shape addObject:_shape[at]];[strides addObject:_strides[at]];
        if(_maps[@(at)])maps[@(d)]=_maps[@(at)];
    }
    if(!_indices)return [MetalTensorLayout shape:shape strides:strides offset:_offset maps:maps indices:nil];
    NSArray *old=TensorStrides(_shape),*steps=TensorStrides(shape);
    NSMutableData *data=[NSMutableData dataWithLength:_count*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
    for(NSUInteger i=0;i<_count;i++) {
        out[i]=0;for(NSUInteger d=0;d<axes.count;d++)out[i]+=(i/[steps[d] unsignedIntegerValue])%[shape[d] unsignedIntegerValue]*[old[[axes[d] unsignedIntegerValue]] unsignedIntegerValue];
    }
    return [self gather:shape indices:data];
}
- (MetalTensorLayout *)reshape:(NSArray<NSNumber *> *)shape {
    NSParameterAssert(TensorCount(shape)==_count);
    if([_shape isEqual:shape])return self;
    if([self matches:TensorStrides(_shape)])return [MetalTensorLayout shape:shape strides:TensorStrides(shape) offset:_offset maps:@{} indices:nil];
    NSMutableData *data=[NSMutableData dataWithLength:_count*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
    for(NSUInteger i=0;i<_count;i++)out[i]=[self index:i];
    return [MetalTensorLayout shape:shape strides:TensorStrides(shape) offset:0 maps:@{} indices:data];
}
- (MetalTensorLayout *)components:(NSUInteger)count {
    NSMutableArray *shape=[[_shape mutableCopy] autorelease];[shape addObject:@(count)];
    if(_indices) {
        NSMutableData *data=[NSMutableData dataWithLength:_count*count*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
        for(NSUInteger i=0;i<_count;i++)for(NSUInteger j=0;j<count;j++)out[i*count+j]=[self index:i]*(NSInteger)count+j;
        return [MetalTensorLayout shape:shape strides:TensorStrides(shape) offset:0 maps:@{} indices:data];
    }
    NSMutableArray *strides=[NSMutableArray array];for(NSNumber *step in _strides)[strides addObject:@(step.integerValue*(NSInteger)count)];
    [strides addObject:@1];NSMutableDictionary *maps=[NSMutableDictionary dictionary];
    for(NSNumber *axis in _maps) {
        NSMutableData *data=[[_maps[axis] mutableCopy] autorelease];NSInteger *values=data.mutableBytes;
        for(NSUInteger i=0;i<data.length/sizeof(NSInteger);i++)values[i]*=count;
        maps[axis]=data;
    }
    return [MetalTensorLayout shape:shape strides:strides offset:_offset*(NSInteger)count maps:maps indices:nil];
}
- (MetalTensorLayout *)shift:(NSInteger)elements {
    if(!elements)return self;
    return [MetalTensorLayout shape:_shape strides:_strides offset:_offset+elements maps:_maps indices:_indices];
}
- (MetalTensorLayout *)elements:(NSUInteger)count {
    NSParameterAssert(count&&[_shape.lastObject unsignedIntegerValue]==count);
    MetalTensorLayout *base=[self element:_shape.count-1 at:0];
    NSParameterAssert(base.offset%(NSInteger)count==0);
    NSMutableArray *steps=[NSMutableArray array];
    for(NSUInteger d=0;d<base.shape.count;d++) {
        NSInteger step=[base.strides[d] integerValue];
        NSParameterAssert([base.shape[d] unsignedIntegerValue]<=1||base.maps[@(d)]||step%(NSInteger)count==0);
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
    return [MetalTensorLayout shape:base.shape strides:steps offset:base.offset/(NSInteger)count maps:maps indices:divide(base.indices)];
}
- (MetalTensorLayout *)rebase:(MetalTensorLayout *)declared onto:(MetalTensorLayout *)placed {
    NSParameterAssert([declared.shape isEqual:placed.shape]);
    if([self isEqual:declared])return placed;
    if([[declared shift:-declared.offset] isEqual:[placed shift:-placed.offset]])
        return [self shift:placed.offset-declared.offset];
    NSArray *logical=TensorStrides(declared.shape);
    NSInteger low;NSArray *axes=TensorInverseAxes(declared,&low);
    NSMutableDictionary *inverse=nil;
    NSMutableData *data=[NSMutableData dataWithLength:_count*sizeof(NSInteger)];NSInteger *out=data.mutableBytes;
    for(NSUInteger i=0;i<_count;i++) {
        const NSInteger address=[self index:i];NSInteger remainder=address-low;NSUInteger at=0;
        BOOL found=declared.affine&&remainder>=0;
        for(NSNumber *axis in axes)if(found) {
            const NSUInteger d=axis.unsignedIntegerValue,n=[declared.shape[d] unsignedIntegerValue];
            const NSInteger s=[declared.strides[d] integerValue],q=remainder/labs(s);
            found=(NSUInteger)q<n;remainder-=q*labs(s);
            at+=(s<0?n-1-q:q)*[logical[d] unsignedIntegerValue];
        }
        found=found&&remainder==0;
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
    if(_indices){const NSInteger step=1;[result appendString:MetalViewMappedIndexDeclaration(binding,&_count,&step,1,(NSUInteger)_offset,@{@0:_indices})];}
    else {
        NSData *shape=TensorWords(_shape),*steps=TensorWords(_strides);
        [result appendString:MetalViewMappedIndexDeclaration(binding,shape.bytes,steps.bytes,_shape.count,(NSUInteger)_offset,_maps)];
    }
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
    if(!_indices)for(NSUInteger d=_shape.count;d--;) {
        NSUInteger n=[_shape[d] unsignedIntegerValue];
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
    NSArray *shape=_indices?@[@(_count)]:_shape,*strides=_indices?@[@1]:_strides;
    NSDictionary *maps=_indices?@{@0:_indices}:_maps;
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
