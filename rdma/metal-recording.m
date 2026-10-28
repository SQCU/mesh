#import "metal-recording.h"
#include "nccl.h"
#import <objc/runtime.h>
#include <stdatomic.h>

BOOL MetalViewLayoutMatches(const NSUInteger *shape,const NSInteger *strides,const NSInteger *required,size_t rank) {
    for(size_t d=0;d<rank;d++)if(!shape[d])return YES;
    for(size_t d=0;d<rank;d++)if(shape[d]>1&&strides[d]!=required[d])return NO;
    return YES;
}

NSString *MetalViewIndexExpression(const NSUInteger *shape,const NSInteger *strides,size_t rank,NSString *index) {
    // Collapse adjacent axes before emitting arithmetic. Contiguous views
    // become the original index, with no table, descriptor load or branch.
    struct { NSUInteger count, divisor; NSInteger stride; } axes[rank?rank:1];
    NSUInteger divisor=1,used=0;
    for(size_t d=rank;d--;) {
        if(!shape[d])return @"0";
        if(shape[d]==1)continue;
        if(used&&strides[d]==axes[used-1].stride*(NSInteger)axes[used-1].count)
            axes[used-1].count*=shape[d];
        else {axes[used].count=shape[d];axes[used].divisor=divisor;axes[used++].stride=strides[d];}
        divisor*=shape[d];
    }
    NSMutableArray *terms=[NSMutableArray array];
    for(NSUInteger a=0;a<used;a++) {
        if(!axes[a].stride)continue;
        NSString *term=[NSString stringWithFormat:@"long(%@)",index];
        if(axes[a].divisor!=1)term=[NSString stringWithFormat:@"(%@ / %lul)",term,(unsigned long)axes[a].divisor];
        if(a+1<used)term=[NSString stringWithFormat:@"(%@ %% %lul)",term,(unsigned long)axes[a].count];
        if(axes[a].stride!=1)term=[NSString stringWithFormat:@"(%@ * %ldl)",term,(long)axes[a].stride];
        [terms addObject:term];
    }
    return terms.count?[terms componentsJoinedByString:@" + "]:@"0";
}

static NSString *MetalViewIndexTemplates(void) {
    return @"template<uint B> inline long mesh_index(ulong i) { return long(i); }\n"
        "template<uint B> constexpr bool mesh_vector4() { return true; }\n";
}

NSString *MetalViewIndexPrelude(void) {
    return [@"#ifndef MESH_INDEX_DECLARED\n#include <metal_stdlib>\nusing namespace metal;\n#define MESH_INDEX_DECLARED 1\n"
        stringByAppendingFormat:@"%@#endif\n",MetalViewIndexTemplates()];
}

static NSDictionary<NSString *,NSData *> *MetalViewBlockSelection(const int32_t *order,size_t count,
    NSUInteger block,NSUInteger copies,int32_t (^select)(int32_t,NSUInteger)) {
    NSMutableData *blocks=[NSMutableData data],*mapped=[NSMutableData data];
    int32_t *lanes=malloc(block*sizeof *lanes);
    for(size_t start=0;start<count;start+=block)for(NSUInteger copy=0;copy<copies;copy++) {
        BOOL used=NO;
        for(NSUInteger i=0;i<block;i++) {
            const int32_t value=start+i<count?order[start+i]:-1;
            lanes[i]=value>=0?select(value,copy):-1;used|=lanes[i]>=0;
        }
        if(used) {
            const int32_t index=(int32_t)(start/block);
            [blocks appendBytes:&index length:sizeof index];
            [mapped appendBytes:lanes length:block*sizeof *lanes];
        } else break;
    }
    free(lanes);return @{@"blocks":blocks,@"order":mapped};
}

NSDictionary<NSString *,NSData *> *MetalViewBlockSlice(const int32_t *order,size_t count,
    NSInteger first,NSInteger end,NSInteger step,NSUInteger block) {
    return MetalViewBlockSelection(order,count,block,1,^int32_t(int32_t value,NSUInteger copy) {
        (void)copy;
        const NSInteger delta=value-first;
        return (step>0?(value>=first&&value<end):(value<=first&&value>end))&&delta%step==0?(int32_t)(delta/step):-1;
    });
}

typedef struct { int32_t value, output; } MetalViewSelection;
static int MetalViewSelectionCompare(const void *a,const void *b) {
    const MetalViewSelection *x=a,*y=b;
    if(x->value!=y->value)return x->value<y->value?-1:1;
    return x->output<y->output?-1:x->output>y->output;
}
NSDictionary<NSString *,NSData *> *MetalViewBlockGather(const int32_t *order,size_t count,
    const int32_t *selected,size_t selectedCount,NSUInteger block) {
    if(selectedCount) {
        const NSInteger step=selectedCount>1?(NSInteger)selected[1]-selected[0]:1;
        BOOL affine=step!=0;
        for(size_t i=1;affine&&i<selectedCount;i++)affine=selected[i]==(NSInteger)selected[0]+(NSInteger)i*step;
        if(affine)return MetalViewBlockSlice(order,count,selected[0],(NSInteger)selected[selectedCount-1]+step,step,block);
    }
    MetalViewSelection *entries=malloc(selectedCount*sizeof *entries);
    for(size_t i=0;i<selectedCount;i++)entries[i]=(MetalViewSelection){selected[i],(int32_t)i};
    qsort(entries,selectedCount,sizeof *entries,MetalViewSelectionCompare);
    NSUInteger copies=0;
    for(size_t first=0;first<selectedCount;) {
        size_t end=first+1;while(end<selectedCount&&entries[end].value==entries[first].value)end++;
        copies=MAX(copies,end-first);first=end;
    }
    NSDictionary *result=MetalViewBlockSelection(order,count,block,copies,^int32_t(int32_t value,NSUInteger copy) {
        size_t lo=0,hi=selectedCount;
        while(lo<hi){size_t mid=lo+(hi-lo)/2;if(entries[mid].value<value)lo=mid+1;else hi=mid;}
        const size_t at=lo+copy;
        return at<selectedCount&&entries[at].value==value?entries[at].output:-1;
    });
    free(entries);return result;
}

NSData *MetalViewTiles(const NSUInteger *shape,size_t rank,NSUInteger elements) {
    for(size_t d=0;d<rank;d++)if(!shape[d])return [NSData data];
    size_t axis=rank;NSUInteger tail=1;
    while(axis&&tail*shape[axis-1]<=elements){axis--;tail*=shape[axis];}
    NSMutableData *result=[NSMutableData data];
    NSUInteger *tile=calloc(2*rank+1,sizeof *tile);
    memcpy(tile+rank,shape,rank*sizeof *shape);
    if(!axis){[result appendBytes:tile length:2*rank*sizeof *tile];free(tile);return result;}
    axis--;NSUInteger chunk=MAX(1,elements/tail);
    for(size_t d=0;d<axis;d++)tile[rank+d]=1;
    BOOL done=NO;
    while(!done){
        tile[rank+axis]=MIN(chunk,shape[axis]-tile[axis]);
        [result appendBytes:tile length:2*rank*sizeof *tile];
        tile[axis]+=chunk;
        if(tile[axis]>=shape[axis]){
            tile[axis]=0;done=YES;
            for(size_t d=axis;d--;){if(++tile[d]<shape[d]){done=NO;break;}tile[d]=0;}
        }
    }
    free(tile);return result;
}

static NSString *MetalViewAffineIndexDeclaration(NSUInteger binding,const NSUInteger *shape,const NSInteger *strides,
    size_t rank,NSUInteger elementOffset,NSInteger base) {
    NSString *expression=MetalViewIndexExpression(shape,strides,rank,@"i");
    if(base)expression=[NSString stringWithFormat:@"(%@ + %ldl)",expression,(long)base];
    NSMutableString *source=[NSMutableString string];
    if(![expression isEqualToString:@"long(i)"])
        [source appendFormat:@"template<> inline long mesh_index<%lu>(ulong i) { return %@; }\n",(unsigned long)binding,expression];
    // A logical group of four can retain its vector instruction only when it
    // is physically adjacent and aligned. This is a compilation decision.
    NSUInteger inner=1,prefix=rank;
    for(size_t d=rank;d--;) {
        if(shape[d]>1&&strides[d]!=(NSInteger)inner)break;
        inner*=shape[d];prefix=d;
    }
    BOOL vector=((NSInteger)elementOffset+base)%4==0&&(!prefix||inner%4==0);
    for(size_t d=0;d<prefix;d++)if(shape[d]>1&&strides[d]%4)vector=NO;
    if(!vector)[source appendFormat:@"template<> constexpr bool mesh_vector4<%lu>() { return false; }\n",(unsigned long)binding];
    return source;
}

NSString *MetalViewIndexDeclaration(NSUInteger binding,const NSUInteger *shape,const NSInteger *strides,
    size_t rank,NSUInteger elementOffset) {
    return MetalViewAffineIndexDeclaration(binding,shape,strides,rank,elementOffset,0);
}

NSDictionary<NSString *,id> *MetalViewNormalizeAxes(const NSInteger *strides,size_t rank,NSDictionary<NSNumber *,NSData *> *maps) {
    NSMutableData *steps=[NSMutableData dataWithBytes:strides length:rank*sizeof *strides];
    NSInteger *held=steps.mutableBytes,base=0;
    NSMutableDictionary *remaining=maps?[maps mutableCopy]:[NSMutableDictionary dictionary];
    for(NSNumber *axis in maps) {
        NSData *data=maps[axis];const NSInteger *values=data.bytes;
        const NSUInteger count=data.length/sizeof *values;
        const NSInteger first=count?values[0]:0,step=count>1?values[1]-first:held[axis.unsignedIntegerValue];
        BOOL affine=YES;
        for(NSUInteger i=1;affine&&i<count;i++)affine=values[i]==first+(NSInteger)i*step;
        if(affine){held[axis.unsignedIntegerValue]=step;base+=first;[remaining removeObjectForKey:axis];}
    }
    return @{@"strides":steps,@"offset":@(base),@"maps":remaining};
}

NSString *MetalViewMappedIndexDeclaration(NSUInteger binding,const NSUInteger *shape,const NSInteger *strides,
    size_t rank,NSUInteger elementOffset,NSDictionary<NSNumber *,NSData *> *maps) {
    if(!maps.count)return MetalViewIndexDeclaration(binding,shape,strides,rank,elementOffset);
    NSDictionary *normalized=MetalViewNormalizeAxes(strides,rank,maps);
    strides=(const NSInteger *)((NSData *)normalized[@"strides"]).bytes;maps=normalized[@"maps"];
    const NSInteger shift=[normalized[@"offset"] integerValue];
    if(!maps.count)return MetalViewAffineIndexDeclaration(binding,shape,strides,rank,elementOffset,shift);
    NSMutableString *source=[NSMutableString string];
    NSMutableArray *terms=[NSMutableArray array];
    if(shift)[terms addObject:[NSString stringWithFormat:@"%ldl",(long)shift]];
    NSUInteger divisor=1,contiguous=1;NSInteger origin=(NSInteger)elementOffset+shift;BOOL inner=YES,vector=YES;
    for(size_t d=rank;d--;) {
        if(!shape[d])return MetalViewIndexDeclaration(binding,shape,strides,rank,elementOffset);
        NSString *coordinate=divisor==1?@"long(i)":[NSString stringWithFormat:@"(long(i) / %lul)",(unsigned long)divisor];
        if(d)coordinate=[NSString stringWithFormat:@"(%@ %% %lul)",coordinate,(unsigned long)shape[d]];
        const NSInteger *values=(const NSInteger *)maps[@(d)].bytes;
        NSInteger base=values?values[0]:0,step=values&&shape[d]>1?values[1]-base:strides[d];
        NSUInteger period=0;
        if(values)for(NSUInteger j=1;j<shape[d];j++)if(values[j]!=base+(NSInteger)j*step){period=j;break;}
        NSString *term=nil;
        if(period) {
            // A slice can start inside a storage group. Preserve that phase
            // in the emitted arithmetic instead of turning it into a LUT.
            NSMutableDictionary *counts=[NSMutableDictionary dictionary];NSUInteger most=0;
            for(NSUInteger j=1;j<shape[d];j++) {
                NSNumber *delta=@(values[j]-values[j-1]);NSUInteger count=[counts[delta] unsignedIntegerValue]+1;
                counts[delta]=@(count);if(count>most){most=count;step=delta.integerValue;}
            }
            NSUInteger first=0,last=0,span=0;NSInteger jump=0;BOOL grouped=YES;
            for(NSUInteger j=1;j<shape[d]&&grouped;j++) {
                const NSInteger delta=values[j]-values[j-1]-step;if(!delta)continue;
                if(!first){first=j;jump=delta;}
                else {if(!span)span=j-last;grouped=delta==jump&&j-last==span;}
                last=j;
            }
            if(!span)span=MAX(first,shape[d]-first);
            const NSUInteger phase=span>=first?span-first:0;
            grouped=grouped&&first&&span>=first;
            for(NSUInteger j=0;grouped&&j<shape[d];j++)
                grouped=values[j]==base+(NSInteger)j*step+(NSInteger)((j+phase)/span)*jump;
            if(grouped)term=[NSString stringWithFormat:@"(%@ * %ldl + ((%@ + %lul) / %lul) * %ldl)",coordinate,
                (long)step,coordinate,(unsigned long)phase,(unsigned long)span,(long)jump];
            else {
                NSString *name=[NSString stringWithFormat:@"mesh_map_%lu_%lu",(unsigned long)binding,(unsigned long)d];
                [source appendFormat:@"constant long %@[%lu] = {",name,(unsigned long)shape[d]];
                for(NSUInteger j=0;j<shape[d];j++)[source appendFormat:@"%s%ldl",j?",":"",(long)(values[j]-base)];
                [source appendString:@"};\n"];
                term=[NSString stringWithFormat:@"%@[%@]",name,coordinate];
            }
        } else if(shape[d]>1&&step)term=step==1?coordinate:[NSString stringWithFormat:@"(%@ * %ldl)",coordinate,(long)step];
        if(term)[terms addObject:term];
        if(base)[terms addObject:[NSString stringWithFormat:@"%ldl",(long)base]];
        origin+=base;
        if(shape[d]>1) {
            if(inner&&!period&&step==(NSInteger)contiguous)contiguous*=shape[d];
            else {inner=NO;vector&=contiguous%4==0;}
            if(!inner) {
                if(values){for(NSUInteger j=0;j<shape[d];j++)vector&=(values[j]-base)%4==0;}
                else vector&=step%4==0;
            }
        }
        divisor*=shape[d];
    }
    NSString *expression=terms.count?[terms componentsJoinedByString:@" + "]:@"0";
    [source appendFormat:@"template<> inline long mesh_index<%lu>(ulong i) { return %@; }\n",(unsigned long)binding,expression];
    if(!vector||origin%4)[source appendFormat:@"template<> constexpr bool mesh_vector4<%lu>() { return false; }\n",(unsigned long)binding];
    return source;
}

struct MetalViewAxis { NSUInteger count, stride; const NSInteger *offsets; NSInteger minimum; };
struct MetalViewSpan { NSUInteger offset, length; };
static int MetalViewAxisOrder(const void *a,const void *b) {
    const NSUInteger x=((const struct MetalViewAxis *)a)->stride,y=((const struct MetalViewAxis *)b)->stride;
    return (x>y)-(x<y);
}
static int MetalViewSpanOrder(const void *a,const void *b) {
    const NSUInteger x=((const struct MetalViewSpan *)a)->offset,y=((const struct MetalViewSpan *)b)->offset;
    return (x>y)-(x<y);
}

NSData *MetalViewReaches(id<MTLBuffer> buffer,NSUInteger offset,const NSUInteger *shape,
    const NSInteger *strides,size_t rank,NSUInteger elementBytes,uint32_t access,int32_t binding) {
    return MetalViewMappedReaches(buffer,offset,shape,strides,rank,elementBytes,access,binding,nil);
}

NSData *MetalViewMappedReaches(id<MTLBuffer> buffer,NSUInteger offset,const NSUInteger *shape,
    const NSInteger *strides,size_t rank,NSUInteger elementBytes,uint32_t access,int32_t binding,
    NSDictionary<NSNumber *,NSData *> *maps) {
    NSDictionary *normalized=maps.count?MetalViewNormalizeAxes(strides,rank,maps):nil;
    if(normalized){strides=(const NSInteger *)((NSData *)normalized[@"strides"]).bytes;maps=normalized[@"maps"];}
    NSMutableData *axes=[NSMutableData dataWithLength:rank*sizeof(struct MetalViewAxis)];
    struct MetalViewAxis *axis=axes.mutableBytes;
    BOOL empty=!elementBytes;
    NSInteger base=(NSInteger)offset+[normalized[@"offset"] integerValue];
    for(size_t i=0;i<rank;i++) {
        empty|=!shape[i];
        const NSInteger *values=(const NSInteger *)maps[@(i)].bytes;
        if(values&&shape[i]) {
            NSInteger minimum=values[0];
            for(NSUInteger j=1;j<shape[i];j++)minimum=MIN(minimum,values[j]);
            base+=minimum;
            axis[i]=(struct MetalViewAxis){shape[i],NSUIntegerMax,values,minimum};
        } else {
            if(shape[i]&&strides[i]<0)base+=(NSInteger)(shape[i]-1)*strides[i];
            axis[i]=(struct MetalViewAxis){strides[i]?shape[i]:MIN(shape[i],1),
                (NSUInteger)(strides[i]<0?-strides[i]:strides[i]),NULL,0};
        }
    }
    struct MetalReach reach={buffer,empty?offset:(NSUInteger)base,0,access|MetalReachBinding,binding};
    if(empty)return [NSData dataWithBytes:&reach length:sizeof reach];
    qsort(axis,rank,sizeof *axis,MetalViewAxisOrder);
    const struct MetalViewSpan first={0,elementBytes};
    NSMutableData *spans=[NSMutableData dataWithBytes:&first length:sizeof first];
    for(size_t i=0;i<rank;i++) {
        if(axis[i].count<=1)continue;
        struct MetalViewSpan *values=spans.mutableBytes;
        const NSUInteger count=spans.length/sizeof *values;
        if(!axis[i].offsets&&count==1&&axis[i].stride<=values[0].length) {
            values[0].length+=(axis[i].count-1)*axis[i].stride;
            continue;
        }
        NSMutableData *next=[NSMutableData dataWithLength:spans.length*axis[i].count];
        struct MetalViewSpan *out=next.mutableBytes;
        for(NSUInteger n=0;n<axis[i].count;n++) {
            const NSUInteger at=axis[i].offsets?(NSUInteger)(axis[i].offsets[n]-axis[i].minimum):n*axis[i].stride;
            for(NSUInteger j=0;j<count;j++)out[n*count+j]=(struct MetalViewSpan){values[j].offset+at,values[j].length};
        }
        const NSUInteger total=count*axis[i].count;
        qsort(out,total,sizeof *out,MetalViewSpanOrder);
        NSUInteger used=0;
        for(NSUInteger j=0;j<total;j++) {
            if(used&&out[j].offset<=out[used-1].offset+out[used-1].length)
                out[used-1].length=MAX(out[used-1].length,out[j].offset+out[j].length-out[used-1].offset);
            else out[used++]=out[j];
        }
        next.length=used*sizeof *out;spans=next;
    }
    NSMutableData *result=[NSMutableData data];
    const struct MetalViewSpan *values=spans.bytes;
    for(NSUInteger i=0;i<spans.length/sizeof *values;i++) {
        reach.offset=(NSUInteger)base+values[i].offset;reach.length=values[i].length;
        [result appendBytes:&reach length:sizeof reach];
    }
    return result;
}

struct MetalRecordedDispatch {
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> buffers[MESH_BUFFER_BINDINGS];
    NSUInteger offsets[MESH_BUFFER_BINDINGS], memory[MESH_MEMORY_BINDINGS],
        strides[MESH_BUFFER_BINDINGS], instances[MESH_BUFFER_BINDINGS], lengths[MESH_BUFFER_BINDINGS];
    MTLSize grid, group;
    BOOL threads, barrier;
    uint64_t access;
    // docs/kernels.md Any program: the reaches the dispatch declared (MetalDispatchReaching), `reaches` of the
    // capture's from `reach`, and the bindings whose extent one of them declares (bit j: binding j)
    uint32_t reach, reaches, extents, bounded;
};

@interface MetalCapture : NSObject {
@public
    NSMutableData *dispatches, *reaches;
    NSMutableSet *owners, *resources;
    NSMutableArray *preparation, *hosts;
    NSUInteger buffers, memory;
}
@end

static _Thread_local MetalCapture *activeCapture;
static _Thread_local const struct MetalBindingView *bindingViews;
static _Thread_local size_t bindingViewCount;
static void MetalDispatchAccesses(id<MTLComputeCommandEncoder>,MTLSize,MTLSize,BOOL,const struct MetalReach *,size_t);

void MetalRecordViews(const struct MetalBindingView *views,size_t count,void (^encode)(void)) {
    const struct MetalBindingView *previous=bindingViews;size_t before=bindingViewCount;
    bindingViews=views;bindingViewCount=count;
    @try { encode(); } @finally { bindingViews=previous;bindingViewCount=before; }
}

static char replaySegmentsKey,replayRecordsKey,replayResourcesKey,replayHostsKey,pipelineAccessKey,pipelineDescriptorKey,pipelineBindingsKey,
    pipelineAddressesKey,operandStrideKey,operandInstancesKey,operandOutputOffsetKey;

static void MetalReplayDestroy(struct MetalReplay *replay) {
    [replay->commands release]; [replay->residency release]; [replay->owner release];
    *replay = (struct MetalReplay){0};
}

// docs/kernels.md Any program: whether a reflected structure holds an address (a pointer member,
// or one in a nested structure or array), through which a kernel reaches memory no binding names.
static BOOL MetalStructAddresses(MTLStructType *structure) {
    for (MTLStructMember *member in structure.members) {
        MTLDataType type = member.dataType;
        MTLStructType *inner = member.structType;
        for (MTLArrayType *array = member.arrayType; array; array = array.elementArrayType) {
            type = array.elementType; inner = array.elementStructType;
        }
        if (type == MTLDataTypePointer || (type == MTLDataTypeStruct && MetalStructAddresses(inner))) return YES;
    }
    return NO;
}

// ../dox/mesh/design/prepared-machine.md#M06
// A pipeline of a recording device, with its reflection kept: the buffers it reads and writes, the
// bindings and threadgroup memory it declares, and whether an argument holds an address.
static id<MTLComputePipelineState> MetalCreatePipeline(id<MTLDevice> device,
    MTLComputePipelineDescriptor *descriptor,BOOL indirect,MTLPipelineOption options,
    MTLComputePipelineReflection **reflection,NSError **error) {
    descriptor = [descriptor copy];
    if(indirect)descriptor.supportIndirectCommandBuffers = YES;
    MTLComputePipelineReflection *bindings = nil;
    id result = [device newComputePipelineStateWithDescriptor:descriptor options:options|MTLPipelineOptionBindingInfo|MTLPipelineOptionBufferTypeInfo reflection:&bindings error:error];
    if (reflection) *reflection = bindings;
    if (!result) { [descriptor release]; return nil; }
    uint64_t access = 0, declared = 0;
    BOOL addresses = NO;
    for (id<MTLBinding> binding in bindings.bindings) {
        if(binding.type==MTLBindingTypeThreadgroupMemory && binding.isUsed)
            declared |= UINT64_C(1) << (MESH_ACCESS_WRITE_SHIFT+binding.index);
        if(binding.type!=MTLBindingTypeBuffer)continue;
        declared |= UINT64_C(1) << binding.index;
        if (!binding.isUsed) continue;
        id<MTLBufferBinding> buffer = (id)binding;
        addresses |= buffer.bufferPointerType.elementIsArgumentBuffer || buffer.bufferDataType == MTLDataTypePointer ||
            (buffer.bufferDataType == MTLDataTypeStruct && MetalStructAddresses(buffer.bufferStructType));
        if (binding.access != MTLBindingAccessWriteOnly) access |= UINT64_C(1) << binding.index;
        if (binding.access != MTLBindingAccessReadOnly) access |= UINT64_C(1) << (MESH_ACCESS_WRITE_SHIFT + binding.index);
    }
    objc_setAssociatedObject(result, &pipelineAccessKey, @(access), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    if (addresses) objc_setAssociatedObject(result, &pipelineAddressesKey, @YES, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    objc_setAssociatedObject(result, &pipelineBindingsKey, @(declared), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    objc_setAssociatedObject(result, &pipelineDescriptorKey, descriptor, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [descriptor release]; return result;
}

// docs/kernels.md Any program: the dataflow's identity of a buffer, its allocation.  A heap's
// buffers may alias one another (makeAliasable), so each is its heap.
static NSValue *MetalAllocationKey(id buffer) {
    if(!buffer||[buffer storageMode]==MTLStorageModeShared)return [NSValue valueWithPointer:NULL];
    id heap=[buffer respondsToSelector:@selector(heap)]?[buffer heap]:nil;
    return [NSValue valueWithPointer:heap?:buffer];
}

// docs/kernels.md Any program: where a buffer starts in its allocation
static NSUInteger MetalAllocationBase(id<MTLBuffer> buffer) {
    if(!buffer||buffer.storageMode==MTLStorageModeShared)return (NSUInteger)buffer.contents;
    return [buffer respondsToSelector:@selector(heap)]&&[buffer heap]?buffer.heapOffset:0;
}

typedef void (^MetalAccess)(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger bytes, BOOL load, BOOL store);
// docs/kernels.md Any program: a command's accesses, each a buffer's bytes [offset, offset + bytes) read or written:
// Each binding has an explicit extent or a union of binding reaches. Slot strides
// relocate those views; neither slot strides nor allocation sizes are accesses.
static void MetalAccesses(const struct MetalRecordedDispatch *command, const struct MetalReach *reaches, NSUInteger invocation, MetalAccess each) {
    for(NSUInteger j=0;j<MESH_BUFFER_BINDINGS;j++) {
        id<MTLBuffer> buffer=command->buffers[j];
        const BOOL load=(command->access>>j)&1,store=(command->access>>(MESH_ACCESS_WRITE_SHIFT+j))&1;
        if(!buffer||(!load&&!store)||((command->extents>>j)&1))continue;
        const NSUInteger offset=MetalInvocationOffset(command->offsets[j],command->strides[j],command->instances[j],invocation);
        if(command->lengths[j])each(buffer,offset,command->lengths[j],load,store);
    }
    for(uint32_t k=0;k<command->reaches;k++) {
        const struct MetalReach *reach=reaches+command->reach+k;
        if(!reach->length)continue;
        NSUInteger offset=reach->offset;
        if(reach->access&MetalReachBinding)
            offset=MetalInvocationOffset(offset,command->strides[reach->binding],command->instances[reach->binding],invocation);
        each(reach->buffer,offset,reach->length,(reach->access&MetalReachRead)!=0,(reach->access&MetalReachWritten)!=0);
    }
}

// Canonical storage identity and ranges for every dependency lowering.
static void MetalViewAccesses(NSData *accesses,void (^each)(id,NSUInteger,NSUInteger,uint32_t)) {
    const struct MetalReach *views=accesses.bytes;
    for(NSUInteger i=0;i<accesses.length/sizeof *views;i++) {
        const struct MetalReach v=views[i];if(!v.length||!(v.access&3))continue;
        const NSUInteger base=MetalAllocationBase(v.buffer);
        each(MetalAllocationKey(v.buffer),base+v.offset,base+v.offset+v.length,v.access);
    }
}

struct MetalSpan { NSUInteger lo, hi, node; };
static void MetalSpanAdd(NSMutableDictionary *spans,id key,NSUInteger lo,NSUInteger hi,NSUInteger node) {
    NSMutableData *held=spans[key]?:(spans[key]=[NSMutableData data]);
    const struct MetalSpan span={lo,hi,node};[held appendBytes:&span length:sizeof span];
}

// A new writer replaces only the covered portion of the previous frontier.
static void MetalSpanErase(NSMutableData *data,NSUInteger lo,NSUInteger hi) {
    const NSUInteger count=data.length/sizeof(struct MetalSpan);if(!count)return;NSUInteger kept=0;
    NSMutableData *tails=[NSMutableData data];struct MetalSpan *spans=data.mutableBytes;
    for(NSUInteger i=0;i<count;i++) {
        struct MetalSpan span=spans[i];
        if(span.lo>=hi||span.hi<=lo){spans[kept++]=span;continue;}
        if(span.hi>hi){const struct MetalSpan tail={hi,span.hi,span.node};[tails appendBytes:&tail length:sizeof tail];}
        if(span.lo<lo){span.hi=lo;spans[kept++]=span;}
    }
    data.length=kept*sizeof *spans;[data appendData:tails];
}

static NSIndexSet *MetalViewPrior(NSMutableDictionary *read,NSMutableDictionary *written,NSData *views) {
    NSMutableIndexSet *prior=[NSMutableIndexSet indexSet];
    MetalViewAccesses(views,^(id key,NSUInteger lo,NSUInteger hi,uint32_t access){
        for(NSMutableDictionary *set in (access&MetalReachWritten?@[written,read]:@[written])) {
            NSData *data=set[key];const struct MetalSpan *spans=data.bytes;
            for(NSUInteger i=0;i<data.length/sizeof *spans;i++)
                if(spans[i].lo<hi&&lo<spans[i].hi)[prior addIndex:spans[i].node];
        }
    });
    return prior;
}

static void MetalViewDefine(NSMutableDictionary *read,NSMutableDictionary *written,NSData *views,NSUInteger node) {
    MetalViewAccesses(views,^(id key,NSUInteger lo,NSUInteger hi,uint32_t access){
        if(access&MetalReachWritten){MetalSpanErase(read[key],lo,hi);MetalSpanErase(written[key],lo,hi);}
    });
    MetalViewAccesses(views,^(id key,NSUInteger lo,NSUInteger hi,uint32_t access){
        if(access&MetalReachWritten)MetalSpanAdd(written,key,lo,hi,node);
        else if(access&MetalReachRead)MetalSpanAdd(read,key,lo,hi,node);
    });
}

NSArray<NSIndexSet *> *MetalViewDependencies(NSArray<NSData *> *accesses) {
    NSMutableDictionary *read=[NSMutableDictionary dictionary],*written=[NSMutableDictionary dictionary];
    NSMutableArray *dependencies=[NSMutableArray arrayWithCapacity:accesses.count];
    for(NSUInteger i=0;i<accesses.count;i++) {
        [dependencies addObject:MetalViewPrior(read,written,accesses[i])];
        MetalViewDefine(read,written,accesses[i],i);
    }
    return dependencies;
}

static NSData *MetalCommandReaches(const struct MetalRecordedDispatch *command,const struct MetalReach *reaches,NSUInteger invocation) {
    NSMutableData *views=[NSMutableData data];
    MetalAccesses(command,reaches,invocation,^(id<MTLBuffer> buffer,NSUInteger offset,NSUInteger bytes,BOOL load,BOOL store){
        const struct MetalReach view={buffer,offset,bytes,(load?MetalReachRead:0)|(store?MetalReachWritten:0),-1};
        [views appendBytes:&view length:sizeof view];
    });
    return views;
}

// Accesses still relevant to later encoders. Weak allocation keys avoid keeping
// a finished function's pages alive through its queue. The function itself owns
// every buffer until completion. Overlapping ranges are merged at preparation.
@interface MetalQueueAccess : NSObject {
@public NSMapTable *read, *written;
}
@end
@implementation MetalQueueAccess
- (instancetype)init {
    self=[super init];
    read=[[NSMapTable weakToStrongObjectsMapTable] retain];
    written=[[NSMapTable weakToStrongObjectsMapTable] retain];return self;
}
- (void)dealloc { [read release];[written release];[super dealloc]; }
@end
static char queueAccessKey,encoderAccessKey;

static void MetalRangeAdd(NSMapTable *map,id allocation,NSUInteger lo,NSUInteger hi) {
    if(lo==hi)return;
    NSMutableData *data=[map objectForKey:allocation];
    if(!data){data=[NSMutableData data];[map setObject:data forKey:allocation];}
    struct MetalSpan *spans=data.mutableBytes;
    for(NSUInteger i=0;i<data.length/sizeof(*spans);) {
        if(spans[i].lo<=hi&&lo<=spans[i].hi) {
            lo=MIN(lo,spans[i].lo);hi=MAX(hi,spans[i].hi);
            spans[i]=spans[data.length/sizeof(*spans)-1];data.length-=sizeof(*spans);
            spans=data.mutableBytes;i=0;
        } else i++;
    }
    const struct MetalSpan span={lo,hi,0};[data appendBytes:&span length:sizeof span];
}

static void MetalRangesMerge(NSMapTable *into,NSMapTable *from) {
    for(id allocation in from.keyEnumerator) {
        NSData *data=[from objectForKey:allocation];const struct MetalSpan *spans=data.bytes;
        for(NSUInteger i=0;i<data.length/sizeof(*spans);i++)MetalRangeAdd(into,allocation,spans[i].lo,spans[i].hi);
    }
}

static NSData *MetalQueueViews(MetalQueueAccess *scope) {
    NSMutableData *views=[NSMutableData data];
    for(NSUInteger mode=MetalReachRead;mode<=MetalReachWritten;mode++) {
        NSMapTable *set=mode==MetalReachRead?scope->read:scope->written;
        for(id buffer in set.keyEnumerator) {
            NSData *data=[set objectForKey:buffer];const struct MetalSpan *spans=data.bytes;
            for(NSUInteger i=0;i<data.length/sizeof *spans;i++) {
                const struct MetalReach view={buffer,spans[i].lo,spans[i].hi-spans[i].lo,(uint32_t)mode,-1};
                [views appendBytes:&view length:sizeof view];
            }
        }
    }
    return views;
}

void MetalEncoderReaches(id<MTL4ComputeCommandEncoder> encoder,const struct MetalReach *reaches,size_t count) {
    NSArray *scope=objc_getAssociatedObject(encoder,&encoderAccessKey);
    if(!scope)[NSException raise:NSInvalidArgumentException format:@"Metal accesses require a prepared schedule encoder"];
    MetalQueueAccess *previous=scope[0],*current=scope[1];
    NSData *views=[NSData dataWithBytes:reaches length:count*sizeof *reaches];
    NSArray *dependencies=MetalViewDependencies(@[MetalQueueViews(previous),views]);
    if([dependencies[1] count]) {
        [encoder barrierAfterQueueStages:MTLStageDispatch beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        [previous->read removeAllObjects];[previous->written removeAllObjects];
    }
    for(size_t i=0;i<count;i++) {
        const struct MetalReach r=reaches[i];if(!r.length)continue;
        if(r.access&MetalReachRead)MetalRangeAdd(current->read,r.buffer,r.offset,r.offset+r.length);
        if(r.access&MetalReachWritten)MetalRangeAdd(current->written,r.buffer,r.offset,r.offset+r.length);
    }
}

static void MetalEncoderAccesses(id<MTL4ComputeCommandEncoder> encoder,const struct MetalRecordedDispatch *command,
    const struct MetalReach *reaches,NSUInteger invocation) {
    NSData *views=MetalCommandReaches(command,reaches,invocation);
    MetalEncoderReaches(encoder,views.bytes,views.length/sizeof(struct MetalReach));
}

// Lower the same relation to a serial sequence of GPU phases. An emitted
// barrier already covers all earlier phases, so their frontier can be dropped.
static BOOL MetalOrder(NSMutableDictionary *read,NSMutableDictionary *written,NSUInteger *phase,
    const struct MetalRecordedDispatch *command,const struct MetalReach *reaches,NSUInteger invocation) {
    NSData *views=MetalCommandReaches(command,reaches,invocation);
    NSIndexSet *prior=MetalViewPrior(read,written,views);
    const NSUInteger level=prior.count?MAX(*phase,prior.lastIndex+1):*phase;
    BOOL barrier=level!=*phase;
    if(barrier){[read removeAllObjects];[written removeAllObjects];}
    MetalViewDefine(read,written,views,level);*phase=level;return barrier;
}

static void MetalDataflow(MetalCapture *capture) {
    struct MetalRecordedDispatch *commands=capture->dispatches.mutableBytes;
    NSUInteger phase=0,count=capture->dispatches.length/sizeof(*commands);
    NSMutableDictionary *written=[NSMutableDictionary dictionary],*read=[NSMutableDictionary dictionary];
    for(NSUInteger i=0;i<count;i++)commands[i].barrier=MetalOrder(read,written,&phase,commands+i,capture->reaches.bytes,0);
}

@interface MetalForwarder : NSProxy {
@public id target;
}
- (instancetype)initWithTarget:(id)object;
@end

@interface MetalDeviceRecorder : MetalForwarder { @public BOOL indirect; }
@end
@interface MetalComputeRecorder : MetalForwarder {
@public MetalCapture *capture;
    struct MetalRecordedDispatch state;
}
- (void)dispatchGrid:(MTLSize)grid threadsPerThreadgroup:(MTLSize)group threadGrid:(BOOL)threads reaches:(const struct MetalReach *)reaches count:(size_t)count;
@end
@interface MetalCommandRecorder : MetalForwarder
@end
@interface MetalBlitPreparation : MetalForwarder
@end
@interface MetalQueueRecorder : MetalForwarder
@end

@implementation MetalCapture
- (instancetype)init {
    self = [super init];
    dispatches = [NSMutableData new];
    reaches = [NSMutableData new];
    owners = [NSMutableSet new];
    resources = [NSMutableSet new];
    preparation = [NSMutableArray new];
    hosts = [NSMutableArray new];
    return self;
}
- (void)dealloc {
    [dispatches release]; [reaches release]; [owners release]; [resources release]; [preparation release]; [hosts release]; [super dealloc];
}
@end

@implementation MetalForwarder
- (instancetype)initWithTarget:(id)object { target = [object retain]; return self; }
- (void)dealloc { [target release]; [super dealloc]; }
- (NSMethodSignature *)methodSignatureForSelector:(SEL)selector {
    return [target methodSignatureForSelector:selector];
}
- (void)forwardInvocation:(NSInvocation *)invocation { [invocation invokeWithTarget:target]; }
- (BOOL)respondsToSelector:(SEL)selector { return [target respondsToSelector:selector]; }
@end

@implementation MetalDeviceRecorder
- (id<MTLCommandQueue>)newCommandQueue {
    id queue = [target newCommandQueue];
    id recorder = [[MetalQueueRecorder alloc] initWithTarget:queue];
    [queue release]; return recorder;
}
- (id<MTLCommandQueue>)newCommandQueueWithDescriptor:(MTLCommandQueueDescriptor *)descriptor {
    id queue = [target newCommandQueueWithDescriptor:descriptor];
    id recorder = [[MetalQueueRecorder alloc] initWithTarget:queue];
    [queue release]; return recorder;
}
- (id<MTLCommandQueue>)newCommandQueueWithMaxCommandBufferCount:(NSUInteger)count {
    id queue = [target newCommandQueueWithMaxCommandBufferCount:count];
    id recorder = [[MetalQueueRecorder alloc] initWithTarget:queue];
    [queue release]; return recorder;
}
- (id<MTLComputePipelineState>)newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)descriptor
    options:(MTLPipelineOption)options reflection:(MTLComputePipelineReflection **)reflection error:(NSError **)error {
    return MetalCreatePipeline(target,descriptor,indirect,options,reflection,error);
}
- (id<MTLComputePipelineState>)newComputePipelineStateWithFunction:(id<MTLFunction>)function
    options:(MTLPipelineOption)options reflection:(MTLComputePipelineReflection **)reflection error:(NSError **)error {
    MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = function;
    id result = [self newComputePipelineStateWithDescriptor:descriptor options:options reflection:reflection error:error];
    [descriptor release]; return result;
}
- (id<MTLComputePipelineState>)newComputePipelineStateWithFunction:(id<MTLFunction>)function error:(NSError **)error {
    return [self newComputePipelineStateWithFunction:function options:MTLPipelineOptionNone reflection:NULL error:error];
}
// docs/placement.md#memory
- (id<MTLBuffer>)newBufferWithBytesNoCopy:(void *)pointer length:(NSUInteger)length options:(MTLResourceOptions)options
    deallocator:(void (^)(void *pointer, NSUInteger length))deallocator {
    return [target newBufferWithBytesNoCopy:pointer length:length options:options deallocator:deallocator];
}
@end

@implementation MetalComputeRecorder
- (void)setComputePipelineState:(id<MTLComputePipelineState>)pipeline {
    state.pipeline = pipeline;
    state.access = [objc_getAssociatedObject(pipeline, &pipelineAccessKey) unsignedLongLongValue];
}
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    // ../dox/mesh/design/prepared-machine.md#M06: a binding past the argument table fails with its index
    if (index >= MESH_BUFFER_BINDINGS)
        [NSException raise:NSInvalidArgumentException
            format:@"Buffer binding %lu exceeds Metal capacity %d", index, MESH_BUFFER_BINDINGS];
    state.buffers[index] = buffer; state.offsets[index] = offset;
    state.lengths[index]=0;state.bounded&=~(UINT32_C(1)<<index);
    state.strides[index] = [objc_getAssociatedObject(buffer,&operandStrideKey) unsignedIntegerValue];
    capture->buffers = MAX(capture->buffers, index + 1);
    if (buffer) { [capture->resources addObject:buffer]; [capture->owners addObject:buffer]; }
}
- (void)setBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index {
    if (index >= MESH_BUFFER_BINDINGS)
        [NSException raise:NSInvalidArgumentException
            format:@"Buffer binding %lu exceeds Metal capacity %d", index, MESH_BUFFER_BINDINGS];
    state.offsets[index] = offset;
    state.lengths[index]=0;state.bounded&=~(UINT32_C(1)<<index);
}
- (void)setBuffers:(const id<MTLBuffer> *)buffers offsets:(const NSUInteger *)offsets withRange:(NSRange)range {
    for (NSUInteger i = 0; i < range.length; ++i) [self setBuffer:buffers[i] offset:offsets[i] atIndex:range.location + i];
}
- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    id<MTLBuffer> buffer = [[target device] newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
    [self setBuffer:buffer offset:0 atIndex:index];
    state.lengths[index]=length;state.bounded|=UINT32_C(1)<<index;[buffer release];
}
- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index {
    if (index >= MESH_MEMORY_BINDINGS)
        [NSException raise:NSInvalidArgumentException
            format:@"Threadgroup memory binding %lu exceeds Metal capacity %d", index, MESH_MEMORY_BINDINGS];
    state.memory[index] = length; capture->memory = MAX(capture->memory, index + 1);
}
// docs/kernels.md Any program: what a dispatch reaches beyond its bindings is declared with it (MetalDispatchReaching)
- (void)refuseUse {
    [NSException raise:NSInvalidArgumentException format:@"command %lu: a resource used by its encoder; a reach is declared with "
        "its dispatch (MetalDispatchReaching, docs/kernels.md Any program)", (unsigned long)(capture->dispatches.length / sizeof state)];
}
- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage { (void)resource; (void)usage; [self refuseUse]; }
- (void)useResources:(const id<MTLResource> *)resources count:(NSUInteger)count usage:(MTLResourceUsage)usage {
    (void)resources; (void)count; (void)usage; [self refuseUse];
}
- (void)useHeap:(id<MTLHeap>)heap { (void)heap; [self refuseUse]; }
- (void)useHeaps:(const id<MTLHeap> *)heaps count:(NSUInteger)count { (void)heaps; (void)count; [self refuseUse]; }
- (void)dispatchThreadgroups:(MTLSize)grid threadsPerThreadgroup:(MTLSize)group {
    state.grid = grid; state.group = group; state.threads = NO; state.reach = 0; state.reaches = 0;
    [self recordDispatch];
}
- (void)dispatchThreads:(MTLSize)grid threadsPerThreadgroup:(MTLSize)group {
    state.grid = grid; state.group = group; state.threads = YES; state.reach = 0; state.reaches = 0;
    [self recordDispatch];
}
// docs/kernels.md Any program: the reaches kept with the capture (their buffers held resident by the replay)
- (void)dispatchGrid:(MTLSize)grid threadsPerThreadgroup:(MTLSize)group threadGrid:(BOOL)threads reaches:(const struct MetalReach *)reaches count:(size_t)count {
    state.grid = grid; state.group = group; state.threads = threads;
    state.reach = (uint32_t)(capture->reaches.length / sizeof *reaches); state.reaches = (uint32_t)count;
    if (count) [capture->reaches appendBytes:reaches length:count * sizeof *reaches];
    for (size_t k = 0; k < count; k++) { [capture->resources addObject:reaches[k].buffer]; [capture->owners addObject:reaches[k].buffer]; }
    [self recordDispatch];
}
// ../dox/mesh/design/prepared-machine.md#M06
// docs/kernels.md Any program: a pipeline of the recording device, whose reflection shows what its
// bindings read and write; one whose arguments hold addresses declares what they reach, and is
// refused where it declares nothing. Binding reaches describe the named argument's
// access ranges, including holes and aliases. Threadgroup memory the pipeline does not declare is not bound. A pipeline the
// recording device did not make is refused.
- (void)recordDispatch {
    NSNumber *declared = objc_getAssociatedObject(state.pipeline, &pipelineBindingsKey);
    const unsigned long index = (unsigned long)(capture->dispatches.length / sizeof state);
    NSString *name = state.pipeline.label ?: @"unnamed";
    if (!declared)
        [NSException raise:NSInvalidArgumentException format:@"command %lu (%@): a pipeline the recording device did not make "
            "(a crossing ends a segment: mesh_rank.h MeshRankCross)", index, name];
    if (declared && !state.reaches && objc_getAssociatedObject(state.pipeline, &pipelineAddressesKey))
        [NSException raise:NSInvalidArgumentException format:@"command %lu (%@): its arguments hold addresses and it declares no reach "
            "(MetalDispatchReaching, docs/kernels.md Any program)", index, name];
    state.extents = 0;
    uint64_t access=state.access;
    const struct MetalReach *reaches = (const struct MetalReach *)capture->reaches.bytes + state.reach;
    for (uint32_t k = 0; k < state.reaches; k++) {
        const struct MetalReach *reach = reaches + k;
        const NSUInteger slots = [objc_getAssociatedObject(reach->buffer, &operandInstancesKey) unsignedIntegerValue],
            stride = [objc_getAssociatedObject(reach->buffer, &operandStrideKey) unsignedIntegerValue];
        if (!(reach->access & (MetalReachRead | MetalReachWritten)) || reach->offset + reach->length > reach->buffer.length || (stride && slots != 1 && !(reach->access & MetalReachBinding)))
            [NSException raise:NSInvalidArgumentException format:@"command %lu (%@): reach %u (%@, %lu bytes at %lu of %lu, access %u) is "
                "no access, past its buffer, or into an operand's slots (%lu), which an address cannot follow (docs/kernels.md Any program)",
                index, name, k, reach->buffer.label ?: @"unnamed", (unsigned long)reach->length, (unsigned long)reach->offset,
                (unsigned long)reach->buffer.length, reach->access, (unsigned long)slots];
        if (!(reach->access & MetalReachBinding)) continue;
        if (reach->binding < 0 || reach->binding >= MESH_BUFFER_BINDINGS || state.buffers[reach->binding] != reach->buffer)
            [NSException raise:NSInvalidArgumentException format:@"command %lu (%@): reach %u names no matching buffer binding", index, name, k];
        if(!(state.extents&(UINT32_C(1)<<reach->binding)))
            access&=~((UINT64_C(1)<<reach->binding)|(UINT64_C(1)<<(MESH_ACCESS_WRITE_SHIFT+reach->binding)));
        if(reach->access&MetalReachRead)access|=UINT64_C(1)<<reach->binding;
        if(reach->access&MetalReachWritten)access|=UINT64_C(1)<<(MESH_ACCESS_WRITE_SHIFT+reach->binding);
        state.extents |= UINT32_C(1) << reach->binding;
    }
    struct MetalRecordedDispatch recorded = state;
    recorded.access=access;
    for(NSUInteger j=0;j<MESH_BUFFER_BINDINGS;j++) {
        const uint32_t bit=UINT32_C(1)<<j;
        const uint32_t mode=((state.access>>j)&1?MetalReachRead:0)|
            ((state.access>>(MESH_ACCESS_WRITE_SHIFT+j))&1?MetalReachWritten:0);
        if(!state.buffers[j]||!mode||(recorded.extents&bit)||(recorded.bounded&bit))continue;
        for(size_t v=0;v<bindingViewCount;v++) {
            const struct MetalBindingView *view=bindingViews+v;
            if(view->buffer!=state.buffers[j]||state.offsets[j]<view->offset||
                (view->length?state.offsets[j]>=view->offset+view->length:state.offsets[j]!=view->offset))continue;
            const struct MetalReach range={view->buffer,state.offsets[j],view->offset+view->length-state.offsets[j],mode|MetalReachBinding,(int32_t)j};
            if(!recorded.reaches)recorded.reach=(uint32_t)(capture->reaches.length/sizeof range);
            [capture->reaches appendBytes:&range length:sizeof range];recorded.reaches++;recorded.extents|=bit;
        }
        if(!(recorded.extents&bit))
            [NSException raise:NSInvalidArgumentException format:@"command %lu (%@): binding %lu has no prepared byte view",index,name,(unsigned long)j];
    }
    for (NSUInteger i = 0; i < MESH_MEMORY_BINDINGS; i++)
        if (!(declared.unsignedLongLongValue & (UINT64_C(1) << (MESH_ACCESS_WRITE_SHIFT + i)))) recorded.memory[i] = 0;
    [capture->owners addObject:state.pipeline];
    [capture->dispatches appendBytes:&recorded length:sizeof recorded];
}
- (void)forwardInvocation:(NSInvocation *)invocation {
    // a selector without arguments encodes nothing (a query, endEncoding, popDebugGroup); the
    // dataflow orders the recording, so a scope barrier adds no edge
    NSString *name = NSStringFromSelector(invocation.selector);
    if ([name containsString:@":"] && ![name isEqualToString:@"setLabel:"] &&
        ![name isEqualToString:@"pushDebugGroup:"] && ![name isEqualToString:@"memoryBarrierWithScope:"]) {
        [NSException raise:NSInvalidArgumentException format:@"Unrecorded compute command %@", name];
    }
    [invocation invokeWithTarget:target];
}
@end

@implementation MetalBlitPreparation
- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range value:(uint8_t)value {
    [target fillBuffer:buffer range:range value:value];
}
- (void)forwardInvocation:(NSInvocation *)invocation {
    NSString *name = NSStringFromSelector(invocation.selector);
    if (![name isEqualToString:@"endEncoding"] && ![name isEqualToString:@"setLabel:"] &&
        ![name isEqualToString:@"pushDebugGroup:"] && ![name isEqualToString:@"popDebugGroup"])
        [NSException raise:NSInvalidArgumentException format:@"Unrecorded blit command %@", name];
    [invocation invokeWithTarget:target];
}
@end

@implementation MetalCommandRecorder
- (void)commit {
    if (activeCapture) [activeCapture->preparation addObject:target];
    [target commit];
}
- (id<MTLBlitCommandEncoder>)blitCommandEncoder {
    id encoder = [target blitCommandEncoder];
    if (!activeCapture) return encoder;
    return (id)[[[MetalBlitPreparation alloc] initWithTarget:encoder] autorelease];
}
- (void)encodeSignalEvent:(id<MTLEvent>)event value:(uint64_t)value {
    [target encodeSignalEvent:event value:value];
}
- (id<MTLComputeCommandEncoder>)computeCommandEncoder {
    if (!activeCapture) return [target computeCommandEncoder];
    MetalComputeRecorder *encoder = [[MetalComputeRecorder alloc] initWithTarget:[target computeCommandEncoder]];
    encoder->capture = activeCapture;
    return (id)[encoder autorelease];
}
- (id<MTLComputeCommandEncoder>)computeCommandEncoderWithDescriptor:(MTLComputePassDescriptor *)descriptor {
    id encoder = [target computeCommandEncoderWithDescriptor:descriptor];
    if (!activeCapture) return encoder;
    MetalComputeRecorder *recorder = [[MetalComputeRecorder alloc] initWithTarget:encoder];
    recorder->capture = activeCapture;
    return (id)[recorder autorelease];
}
// a serial or concurrent encoder records alike: the recording's order is its dataflow
- (id<MTLComputeCommandEncoder>)computeCommandEncoderWithDispatchType:(MTLDispatchType)type {
    id encoder = [target computeCommandEncoderWithDispatchType:type];
    if (!activeCapture) return encoder;
    MetalComputeRecorder *recorder = [[MetalComputeRecorder alloc] initWithTarget:encoder];
    recorder->capture = activeCapture;
    return (id)[recorder autorelease];
}
- (void)forwardInvocation:(NSInvocation *)invocation {
    NSString *name = NSStringFromSelector(invocation.selector);
    if (activeCapture && ([name hasPrefix:@"encode"] || [name containsString:@"CommandEncoder"]))
        [NSException raise:NSInvalidArgumentException format:@"Unrecorded command %@", name];
    [invocation invokeWithTarget:target];
}
@end

@implementation MetalQueueRecorder
- (id<MTLCommandBuffer>)commandBuffer {
    id command = [target commandBuffer];
    MetalCommandRecorder *recorder = [[MetalCommandRecorder alloc] initWithTarget:command];
    return (id)[recorder autorelease];
}
- (id<MTLCommandBuffer>)commandBufferWithUnretainedReferences {
    id command = [target commandBufferWithUnretainedReferences];
    MetalCommandRecorder *recorder = [[MetalCommandRecorder alloc] initWithTarget:command];
    return (id)[recorder autorelease];
}
- (id<MTLCommandBuffer>)commandBufferWithDescriptor:(MTLCommandBufferDescriptor *)descriptor {
    id command = [target commandBufferWithDescriptor:descriptor];
    MetalCommandRecorder *recorder = [[MetalCommandRecorder alloc] initWithTarget:command];
    return (id)[recorder autorelease];
}
@end

// ../dox/mesh/design/prepared-machine.md#M06
// A recording's replay: its indirect command buffers holding every repetition's commands (the
// one modulus), their barriers, and the residency set.
static void MetalReplayBuild(MetalCapture *capture,id<MTLDevice> device,NSUInteger repetitions,enum MetalForm form,
    struct MetalReplay *result) {
    result->count = (uint32_t)(capture->dispatches.length / sizeof(struct MetalRecordedDispatch));
    if (!result->count) [NSException raise:NSInvalidArgumentException format:@"No compute commands recorded"];
    [capture->owners unionSet:capture->resources];
    result->buffers = (uint16_t)capture->buffers; result->memory = (uint16_t)capture->memory;
    const struct MetalRecordedDispatch *commands = capture->dispatches.bytes;
    NSMutableData *barriers=[NSMutableData dataWithLength:form==MetalFormIndirect?(NSUInteger)result->count*repetitions:0];
    uint8_t *ordered=barriers.mutableBytes;
    NSMutableDictionary *written=[NSMutableDictionary dictionary],*read=[NSMutableDictionary dictionary];
    NSUInteger phase=0;
    for(NSUInteger t=0;form==MetalFormIndirect&&t<repetitions;t++) @autoreleasepool {
        for(NSUInteger i=0;i<result->count;i++)ordered[t*result->count+i]=MetalOrder(read,written,&phase,commands+i,capture->reaches.bytes,t);
    }
    result->owner = [capture->owners retain];
    objc_setAssociatedObject(result->owner,&replayHostsKey,[NSArray arrayWithArray:capture->hosts],OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    NSMutableData *resources=[NSMutableData dataWithLength:capture->resources.count*sizeof(id)];
    id __unsafe_unretained *resourceList=resources.mutableBytes;NSUInteger resourceIndex=0;
    for(id resource in capture->resources)resourceList[resourceIndex++]=resource;
    objc_setAssociatedObject(result->owner,&replayResourcesKey,resources,OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    // docs/kernels.md E2: the records a direct replay encodes (MetalReplayDispatch4)
    objc_setAssociatedObject(result->owner, &replayRecordsKey, @[[NSData dataWithBytes:commands length:capture->dispatches.length],barriers,@(form),[NSData dataWithData:capture->reaches],@(repetitions)],
        OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    MTLResidencySetDescriptor *residency = [MTLResidencySetDescriptor new];
    residency.initialCapacity = capture->owners.count + 1;
    NSError *error = nil;
    result->residency = [device newResidencySetWithDescriptor:residency error:&error];
    [residency release];
    if (!result->residency) [NSException raise:NSMallocException format:@"%@", error];
    for (id allocation in capture->owners) [result->residency addAllocation:allocation];
    if (form != MetalFormIndirect) { [result->residency commit]; [result->residency requestResidency]; return; }
    MTLIndirectCommandBufferDescriptor *descriptor = [MTLIndirectCommandBufferDescriptor new];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
    descriptor.inheritBuffers = NO; descriptor.inheritPipelineState = NO;
    descriptor.maxKernelBufferBindCount = capture->buffers;
    descriptor.maxKernelThreadgroupMemoryBindCount = capture->memory;
    // One indirect command buffer holds at most MESH_INDIRECT_COMMANDS commands, so the
    // repetitions fill as many buffers as they need, `span` repetitions each, executed in order
    // by MetalReplayExecute.
    NSUInteger ceiling = MESH_INDIRECT_COMMANDS;
    NSUInteger span = MAX((NSUInteger)1, ceiling / result->count);
    if (result->count > ceiling) [NSException raise:NSInvalidArgumentException format:@"%u commands exceed one indirect buffer's %lu", result->count, (unsigned long)ceiling];
    NSMutableArray *segments = [NSMutableArray array];
    for (NSUInteger first = 0; first < repetitions; first += span) {
        id<MTLIndirectCommandBuffer> segment = [device newIndirectCommandBufferWithDescriptor:descriptor
            maxCommandCount:result->count*MIN(span, repetitions-first) options:0];
        if (!segment) [NSException raise:NSMallocException format:@"indirect command buffer of %lu commands",
            (unsigned long)(result->count*MIN(span, repetitions-first))];
        [segments addObject:segment]; [segment release];
    }
    result->commands = [segments[0] retain];
    objc_setAssociatedObject(result->commands, &replaySegmentsKey, @[segments, @(span*result->count)], OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [descriptor release];
    for (NSUInteger t = 0; t < repetitions; t++) for (uint32_t i = 0; i < result->count; ++i) {
        const struct MetalRecordedDispatch *source = commands + i;
        if (!source->pipeline.supportIndirectCommandBuffers)
            [NSException raise:NSInvalidArgumentException format:@"command %u (%@) does not support indirect commands: the encoded form runs it (docs/kernels.md Any program)",
                i, source->pipeline.label ?: @"unnamed"];
        id<MTLIndirectComputeCommand> command = [segments[t / span] indirectComputeCommandAtIndex:(t % span)*result->count+i];
        [command setComputePipelineState:source->pipeline];
        // ../dox/mesh/design/prepared-machine.md#M13: the one modulus.  Invocation t binds slot
        // (t % instances), and instances == repetitions is a bare t.
        for (NSUInteger j = 0; j < capture->buffers; ++j)
            if (((source->access | (source->access >> MESH_ACCESS_WRITE_SHIFT)) >> j) & 1) {
                [command setKernelBuffer:source->buffers[j]
                    offset:MetalInvocationOffset(source->offsets[j],source->strides[j],source->instances[j],t) atIndex:j];
            }
        for (NSUInteger j = 0; j < capture->memory; ++j)
            if (source->memory[j]) [command setThreadgroupMemoryLength:source->memory[j] atIndex:j];
        // a segment's first command waits on the segment before it
        if (ordered[t*result->count+i]) [command setBarrier];
        if (source->threads) [command concurrentDispatchThreads:source->grid threadsPerThreadgroup:source->group];
        else [command concurrentDispatchThreadgroups:source->grid threadsPerThreadgroup:source->group];
    }
    for (id<MTLIndirectCommandBuffer> segment in segments) [result->residency addAllocation:segment];
    [result->residency commit];
    [result->residency requestResidency];
}

// docs/kernels.md Any program: one recorded command's line: its level (barriers before it), entry,
// grid and group (`t` a thread grid), threadgroup memory, the bytes of its accesses (MetalAccesses:
// its bindings, then its reaches), and the named operands.
static NSString *MetalCommandLine(const struct MetalRecordedDispatch *command,const struct MetalReach *reaches,NSUInteger i,NSUInteger level,
    NSMutableDictionary *allocations) {
    MTLComputePipelineDescriptor *descriptor=objc_getAssociatedObject(command->pipeline,&pipelineDescriptorKey);
    NSUInteger memory=command->pipeline.staticThreadgroupMemoryLength;
    __block NSUInteger read=0,written=0;
    for(NSUInteger j=0;j<MESH_MEMORY_BINDINGS;j++)memory+=command->memory[j];
    NSMutableString *names=[NSMutableString string],*bindings=[NSMutableString string];
    MetalAccesses(command,reaches,0,^(id<MTLBuffer> buffer,NSUInteger offset,NSUInteger bytes,BOOL load,BOOL store) {
        if(load)read+=bytes;
        if(store)written+=bytes;
        if(buffer.label.length)[names appendFormat:@" %@:%s%s",buffer.label,load?"r":"-",store?"w":"-"];
        NSValue *key=MetalAllocationKey(buffer);
        if(!allocations[key])allocations[key]=@(allocations.count);
        [bindings appendFormat:@" @%lu+%lu:%lu:%s%s",[allocations[key] unsignedLongValue],(unsigned long)(MetalAllocationBase(buffer)+offset),
            (unsigned long)bytes,load?"r":"-",store?"w":"-"];
    });
    return [NSString stringWithFormat:@"command %lu level %lu %s grid %lu,%lu,%lu%s group %lu,%lu,%lu memory %lu read %lu written %lu%s%s%s",
        i,level,descriptor.computeFunction.name.UTF8String?:"?",command->grid.width,command->grid.height,command->grid.depth,
        command->threads?"t":"",command->group.width,command->group.height,command->group.depth,memory,read,written,
        "",bindings.UTF8String,names.UTF8String];
}

// tools/mesh/schedule.py: whether the recordings that follow list their commands (a record-only call's), beside RECORD_ONLY
static _Atomic int listing;
void MetalRecordingList(int on) { atomic_store(&listing,on); }


// ../dox/mesh/design/prepared-machine.md#M06
// Records the compute commands `encode` issues through queues of the recording `device` as one
// repetition: the dataflow's barriers, each operand's slots and output offset (the slot count is
// the buffer's only where the binding's stride is the buffer's own), the replay built in `form`;
// the program's census line, and under RECORD_ONLY one line a command.
static void MetalRecordReplay(id<MTLDevice> device,void (^encode)(void),NSUInteger repetitions,enum MetalForm form,struct MetalReplay *result) {
    MetalCapture *capture = [MetalCapture new];
    @try {
        activeCapture = capture;
        @try { encode(); } @finally { activeCapture = nil; }
        for (id<MTLCommandBuffer> command in capture->preparation) {
            [command waitUntilCompleted];
            if (command.error) [NSException raise:NSInternalInconsistencyException format:@"%@", command.error];
        }
        struct MetalRecordedDispatch *prepared=capture->dispatches.mutableBytes;
        const NSUInteger recorded=capture->dispatches.length/sizeof(*prepared);
        for(NSUInteger i=0;i<recorded;i++)for(NSUInteger j=0;j<capture->buffers;j++) {
            if(prepared[i].access&(UINT64_C(1)<<(MESH_ACCESS_WRITE_SHIFT+j))) {
                NSUInteger shift=[objc_getAssociatedObject(prepared[i].buffers[j],&operandOutputOffsetKey) unsignedIntegerValue];
                struct MetalReach *reaches=capture->reaches.mutableBytes;
                for(uint32_t k=0;k<prepared[i].reaches;k++) {
                    struct MetalReach *reach=reaches+prepared[i].reach+k;
                    if((reach->access&MetalReachBinding)&&reach->binding==(int32_t)j)
                        reach->offset+=shift;
                }
                prepared[i].offsets[j]+=shift;
            }
            NSUInteger own=[objc_getAssociatedObject(prepared[i].buffers[j],&operandStrideKey) unsignedIntegerValue];
            prepared[i].instances[j]=prepared[i].strides[j]==own?
                [objc_getAssociatedObject(prepared[i].buffers[j],&operandInstancesKey) unsignedIntegerValue]:0;
        }
        MetalDataflow(capture);
        MetalReplayBuild(capture, ((MetalForwarder *)device)->target, repetitions, form, result);
        NSUInteger levels=1;
        for(NSUInteger i=1;i<recorded;i++)levels+=prepared[i].barrier;
        printf("prepared program commands %lu levels %lu form %s\n",recorded,levels,(const char *[]){"icb","encoded","serial"}[form]);
        NSMutableDictionary *allocations=[NSMutableDictionary dictionary];
        if(getenv("RECORD_ONLY")||atomic_load(&listing))for(NSUInteger i=0,level=0;i<recorded;i++) @autoreleasepool {
            level+=i&&prepared[i].barrier;
            printf("prepared %s\n",MetalCommandLine(prepared+i,capture->reaches.bytes,i,level,allocations).UTF8String);
        }
    } @catch (NSException *error) {
        MetalReplayDestroy(result);
        @throw;
    } @finally {
        [capture release];
    }
}

static void MetalReplaySegments(struct MetalReplay *replay, NSRange range, void (^each)(id<MTLIndirectCommandBuffer>, NSRange)) {
    if (!replay->commands) return;
    NSArray *held = objc_getAssociatedObject(replay->commands, &replaySegmentsKey);
    NSArray *segments = held ? held[0] : @[replay->commands];
    NSUInteger size = held ? [held[1] unsignedIntegerValue] : NSUIntegerMax;
    for (NSUInteger at = range.location, end = NSMaxRange(range); at < end;) {
        NSUInteger index = at / size, local = at % size, length = MIN(end - at, size - local);
        each(segments[index], NSMakeRange(local, length));
        at += length;
    }
}

// The same recorded operands bind a direct Metal 4 dispatch during function preparation.
static void MetalReplayDispatch4(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,
    struct MetalReplay *replay,const struct MetalRecordedDispatch *source,NSUInteger t) {
    [encoder setArgumentTable:table];[encoder setComputePipelineState:source->pipeline];
    for(NSUInteger j=0;j<replay->buffers;j++)if(((source->access|(source->access>> MESH_ACCESS_WRITE_SHIFT))>>j)&1) {
        [table setAddress:source->buffers[j].gpuAddress+MetalInvocationOffset(source->offsets[j],source->strides[j],source->instances[j],t) atIndex:j];
    }
    for(NSUInteger j=0;j<replay->memory;j++)if(source->memory[j])[encoder setThreadgroupMemoryLength:source->memory[j] atIndex:j];
    if(source->threads)[encoder dispatchThreads:source->grid threadsPerThreadgroup:source->group];
    else [encoder dispatchThreadgroups:source->grid threadsPerThreadgroup:source->group];
}

// ../dox/mesh/design/prepared-machine.md#M13
static void MetalReplayDispatch(id<MTLComputeCommandEncoder> encoder, struct MetalReplay *replay,
    const struct MetalRecordedDispatch *source, NSUInteger t, const struct MetalReach *reaches) {
    [encoder setComputePipelineState:source->pipeline];
    for (NSUInteger j = 0; j < replay->buffers; j++)
        if (((source->access | (source->access >> MESH_ACCESS_WRITE_SHIFT)) >> j) & 1) {
            [encoder setBuffer:source->buffers[j] offset:MetalInvocationOffset(source->offsets[j],source->strides[j],source->instances[j],t) atIndex:j];
        }
    for (NSUInteger j = 0; j < replay->memory; j++)
        if (source->memory[j]) [encoder setThreadgroupMemoryLength:source->memory[j] atIndex:j];
    if(activeCapture) {
        // Recapture preserves scalar/dense binding extents too. This is graph
        // construction, never a numerical submission operation.
        NSMutableData *views=[NSMutableData data];
        if(reaches)[views appendBytes:reaches+source->reach length:source->reaches*sizeof *reaches];
        for(NSUInteger j=0;j<replay->buffers;j++) {
            const uint32_t mode=((source->access>>j)&1?MetalReachRead:0)|
                ((source->access>>(MESH_ACCESS_WRITE_SHIFT+j))&1?MetalReachWritten:0);
            if(!source->buffers[j]||!mode||((source->extents>>j)&1))continue;
            const struct MetalReach view={source->buffers[j],source->offsets[j],source->lengths[j],mode|MetalReachBinding,(int32_t)j};
            [views appendBytes:&view length:sizeof view];
        }
        struct MetalReach *bound=views.mutableBytes;
        for(NSUInteger j=0;j<views.length/sizeof *bound;j++)if(bound[j].access&MetalReachBinding)
            bound[j].offset=MetalInvocationOffset(bound[j].offset,source->strides[bound[j].binding],source->instances[bound[j].binding],t);
        MetalDispatchAccesses(encoder,source->grid,source->group,source->threads,views.bytes,views.length/sizeof *bound);
    } else MetalDispatchAccesses(encoder,source->grid,source->group,source->threads,
        reaches?reaches+source->reach:NULL,reaches?source->reaches:0);
}

struct MetalReplay *MetalReplayCommands(id<MTLDevice> device,const struct MetalCommandDescription *commands,size_t count,enum MetalForm form) {
    if(object_getClass(device)==[MetalDeviceRecorder class])device=((MetalForwarder *)device)->target;
    MetalCapture *capture=[MetalCapture new];struct MetalReplay *replay=NULL;
    @try {
        for(size_t i=0;i<count;i++) {
            const struct MetalCommandDescription *description=commands+i;
            struct MetalRecordedDispatch command={.pipeline=description->pipeline,.grid=description->groups,.group=description->threads,.threads=description->threadGrid,
                .reach=(uint32_t)(capture->reaches.length/sizeof(struct MetalReach)),.reaches=(uint32_t)description->reachCount};
            [capture->owners addObject:command.pipeline];
            for(size_t k=0;k<description->bindingCount;k++) {
                const struct MetalCommandBinding binding=description->bindings[k];
                if(binding.index>=MESH_BUFFER_BINDINGS)[NSException raise:NSInvalidArgumentException format:@"binding index exceeds Metal capacity"];
                const uint32_t at=binding.index;
                command.buffers[at]=binding.buffer;command.offsets[at]=binding.offset;command.lengths[at]=binding.length;
                command.access|=(uint64_t)(binding.access&MetalReachRead?1:0)<<at;
                command.access|=(uint64_t)(binding.access&MetalReachWritten?1:0)<<(MESH_ACCESS_WRITE_SHIFT+at);
                capture->buffers=MAX(capture->buffers,at+1);
                [capture->resources addObject:binding.buffer];
            }
            if(description->memoryCount>MESH_MEMORY_BINDINGS)[NSException raise:NSInvalidArgumentException format:@"threadgroup-memory index exceeds Metal capacity"];
            for(size_t k=0;k<description->memoryCount;k++)command.memory[k]=description->memory[k];
            capture->memory=MAX(capture->memory,description->memoryCount);
            [capture->reaches appendBytes:description->reaches length:description->reachCount*sizeof(struct MetalReach)];
            for(size_t k=0;k<description->reachCount;k++) {
                const struct MetalReach reach=description->reaches[k];
                [capture->resources addObject:reach.buffer];
                if(reach.access&MetalReachBinding) {
                    if(reach.binding<0||reach.binding>=MESH_BUFFER_BINDINGS||command.buffers[reach.binding]!=reach.buffer)
                        [NSException raise:NSInvalidArgumentException format:@"command reach names no matching buffer binding"];
                    command.extents|=UINT32_C(1)<<reach.binding;
                }
            }
            [capture->dispatches appendBytes:&command length:sizeof command];
        }
        MetalDataflow(capture);
        if(posix_memalign((void **)&replay,_Alignof(struct MetalReplay),sizeof *replay))
            [NSException raise:NSMallocException format:@"allocating replay"];
        memset(replay,0,sizeof *replay);
        MetalReplayBuild(capture,device,1,form,replay);
    } @catch(NSException *error) {
        if(replay){MetalReplayDestroy(replay);free(replay);replay=NULL;}
        fprintf(stderr,"record: %s\n",error.reason.UTF8String);
    } @finally { [capture release]; }
    return replay;
}

void MetalReplayEncode(struct MetalReplay *replay,id<MTLCommandBuffer> command,NSRange range) {
    NSArray *held=objc_getAssociatedObject(replay->owner,&replayRecordsKey);
    id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
    if(activeCapture || [held[2] unsignedIntegerValue]!=MetalFormIndirect) {
        const struct MetalRecordedDispatch *records=[held[0] bytes];
        const struct MetalReach *reaches=[held[3] bytes];
        for(NSUInteger i=range.location;i<NSMaxRange(range);i++)MetalReplayDispatch(encoder,replay,records+i,0,reaches);
    } else {
        NSData *resources=objc_getAssociatedObject(replay->owner,&replayResourcesKey);
        [encoder useResources:resources.bytes count:resources.length/sizeof(id) usage:MTLResourceUsageRead|MTLResourceUsageWrite];
        MetalReplaySegments(replay,range,^(id<MTLIndirectCommandBuffer> segment,NSRange local){
            [encoder executeCommandsInBuffer:segment withRange:local];
        });
    }
    [encoder endEncoding];
}

// docs/kernels.md#collectives
static NSError *MetalReplayFailure(NSInteger code, NSString *reason) {
    return [NSError errorWithDomain:@"MetalReplay" code:code userInfo:@{NSLocalizedDescriptionKey:reason}];
}

// One prepared function owns its command stream. Append and Finish run during preparation;
// Enqueue commits the finished stream. Each part keeps the actions it encoded, so a later
// invocation encodes them again into the same command buffer, allocator, argument table,
// residency set and events; invocation g signals and awaits value g.
typedef int (^MetalAction)(struct MetalSchedule *);
@interface MetalWork : NSObject {
@public
    NSData *accesses;
    MetalAction encode;
    NSError *(^operation)(void);
    NSMutableIndexSet *dependencies, *successors;
    struct MetalSchedule *part;
    // Objective-C allocation need not have MetalReplay's extended alignment.
    unsigned char replay[sizeof(struct MetalReplay)];
    NSUInteger invocation, first, end;
    id<MTLSharedEvent> event;
    NSMutableArray *prerequisites;
    _Atomic unsigned remaining;
    _Atomic int status;
    NSError *error;
}
@end
@implementation MetalWork
- (void)dealloc {
    [accesses release];[encode release];[operation release];[dependencies release];[successors release];
    [event release];[prerequisites release];[error release];[super dealloc];
}
@end

struct MetalSchedule {
    struct MetalSchedule *next, *tail, *root;
    struct MetalSchedule *submit, *firstSubmission, *lastSubmission;
    id<MTLDevice> device;
    enum MetalForm form;
    BOOL finished, prepared, again, ran;
    id queue, command, encoder;
    id<MTL4CommandAllocator> allocator;
    id<MTL4ArgumentTable> table;
    id<MTLResidencySet> residency;
    NSMutableSet *owners;
    NSMutableOrderedSet *communicators;
    NSMutableDictionary *read, *written;
    NSUInteger phase;
    id<MTLEvent> after;
    uint64_t value, generation;
    id<MTLSharedEvent> event, start;
    NSMutableArray *work, *hosts, *waits, *actions;
    NSMutableSet *waited;
    struct MetalSchedule *boundary;
    NSUInteger commands, ordinal;
    _Atomic int status;
    _Atomic unsigned references;
    _Atomic unsigned pending;
    NSError *error;
    void (^completed)(NSError *);
    id<MTLBuffer> data;
    NSUInteger used;
};
// Host operations' prerequisites notify through one listener for the process.
static MTLSharedEventListener *MetalListener(void) {
    static MTLSharedEventListener *listener;static dispatch_once_t once;
    dispatch_once(&once,^{listener=[[MTLSharedEventListener alloc] initWithDispatchQueue:dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0)];});
    return listener;
}

MTLCompileOptions *MetalDeviceCompilerOptions(id<MTLDevice> device) {
    MTLCompileOptions *options=[MTLCompileOptions new];
    options.languageVersion=MESH_METAL_LANGUAGE;
    options.preprocessorMacros=@{
        @"MESH_DEVICE_THREADGROUP_THREADS": @(device.maxThreadsPerThreadgroup.width),
        @"MESH_DEVICE_THREADGROUP_BYTES": @(device.maxThreadgroupMemoryLength),
#define MESH_METAL_VALUE(name, value) @"MESH_" #name: @(value),
#include "../etc/compute/metal.def"
#undef MESH_METAL_VALUE
    };
    return [options autorelease];
}

static struct MetalSchedule *MetalScheduleTail(struct MetalSchedule *s) { return s->tail?:s; }
static void MetalScheduleFlush(struct MetalSchedule *);
static struct MetalSchedule *MetalScheduleWritable(struct MetalSchedule *);
static void MetalScheduleFinishPart(struct MetalSchedule *);
static void MetalScheduleArrived(struct MetalSchedule *);
static void MetalScheduleCompleted(struct MetalSchedule *,NSError *);
static void MetalScheduleWait(struct MetalSchedule *root,id<MTLSharedEvent> event) {
    if([root->waited containsObject:event])return;
    struct MetalSchedule *s=MetalScheduleWritable(root);
    if(s->commands){MetalScheduleFinishPart(s);s=MetalScheduleWritable(root);}
    [s->waits addObject:event];[root->waited addObject:event];
    if(s->form==MetalFormSerial)[(id<MTLCommandBuffer>)s->command encodeWaitForEvent:event value:root->generation];
}

// A part's command buffer opened for its actions: Metal 4's begun again on its own allocator, a
// serial part's made anew with its input waits, which precede its encoder.
static void MetalPartBegin(struct MetalSchedule *s) {
    s->read=[NSMutableDictionary new];s->written=[NSMutableDictionary new];s->phase=0;s->commands=0;
    if(s->form==MetalFormSerial) {
        [s->command release];s->command=[[(id<MTLCommandQueue>)s->queue commandBufferWithUnretainedReferences] retain];
        if(s->again&&s->after)[(id<MTLCommandBuffer>)s->command encodeWaitForEvent:s->after value:s->root->value];
        for(id<MTLSharedEvent> event in s->again?s->waits:nil)[(id<MTLCommandBuffer>)s->command encodeWaitForEvent:event value:s->root->generation];
        return;
    }
    [s->allocator reset];
    [(id<MTL4CommandBuffer>)s->command beginCommandBufferWithAllocator:s->allocator];
    s->encoder=[[(id<MTL4CommandBuffer>)s->command computeCommandEncoder] retain];
    MetalQueueAccess *previous=objc_getAssociatedObject(s->queue,&queueAccessKey);
    if(!previous) {
        previous=[MetalQueueAccess new];
        objc_setAssociatedObject(s->queue,&queueAccessKey,previous,OBJC_ASSOCIATION_RETAIN_NONATOMIC);[previous release];
    }
    MetalQueueAccess *current=[MetalQueueAccess new];
    objc_setAssociatedObject(s->encoder,&encoderAccessKey,@[previous,current],OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [current release];
}

static void MetalPartEnd(struct MetalSchedule *s) {
    [s->residency commit];
    if(s->form==MetalFormSerial) {
        if(s->residency)[(id<MTLCommandBuffer>)s->command useResidencySet:s->residency];
        [s->encoder endEncoding];
        if(s->event)[(id<MTLCommandBuffer>)s->command encodeSignalEvent:s->event value:s->root->generation];
        if(s->commands)[(id<MTLCommandBuffer>)s->command addCompletedHandler:^(id<MTLCommandBuffer> done){
            MetalScheduleCompleted(s,done.error);
        }];
    } else {
        [(id<MTL4CommandBuffer>)s->command useResidencySet:s->residency];
        NSArray *scope=objc_getAssociatedObject(s->encoder,&encoderAccessKey);
        MetalQueueAccess *previous=scope[0],*current=scope[1];
        MetalRangesMerge(previous->read,current->read);MetalRangesMerge(previous->written,current->written);
        objc_setAssociatedObject(s->encoder,&encoderAccessKey,nil,OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        [s->encoder endEncoding];
        [(id<MTL4CommandBuffer>)s->command endCommandBuffer];
    }
    [s->encoder release];s->encoder=nil;
    [s->read release];s->read=nil;[s->written release];s->written=nil;
}

// A later invocation's part: the actions it recorded, encoded again in their order.
static int MetalPartEncode(struct MetalSchedule *s) {
    s->again=YES;MetalPartBegin(s);
    int error=0;
    for(MetalAction action in s->actions)if((error=action(s)))break;
    MetalPartEnd(s);
    return error;
}

struct MetalSchedule *MetalScheduleCreateOn(id queue, enum MetalForm form) {
    if(!queue)return NULL;
    id<MTLDevice> device=[queue device];
    struct MetalSchedule *s=calloc(1,sizeof *s);
    if(!s)return NULL;
    atomic_init(&s->references,1);
    atomic_init(&s->pending,0);s->root=s;s->generation=1;
    atomic_init(&s->status,0);
    s->device=[device retain];s->form=form;s->queue=[queue retain];
    s->owners=[NSMutableSet new];s->actions=[NSMutableArray new];
    s->work=[NSMutableArray new];s->hosts=[NSMutableArray new];s->waits=[NSMutableArray new];s->waited=[NSMutableSet new];
    MTLResidencySetDescriptor *resident=[MTLResidencySetDescriptor new];
    s->residency=[device newResidencySetWithDescriptor:resident error:nil];[resident release];
    if(form!=MetalFormSerial) {
        s->allocator=[device newCommandAllocator];
        MTL4ArgumentTableDescriptor *descriptor=[MTL4ArgumentTableDescriptor new];descriptor.maxBufferBindCount=MESH_BUFFER_BINDINGS;
        s->table=[device newArgumentTableWithDescriptor:descriptor error:nil];[descriptor release];
        s->command=[device newCommandBuffer];
    }
    if(!s->residency||(form!=MetalFormSerial&&(!s->allocator||!s->table||!s->command))) {
        MetalScheduleFree(s);return NULL;
    }
    // The serial encoder is opened at the first append, after the external input event.
    MetalPartBegin(s);
    if(!s->command||(form!=MetalFormSerial&&!s->encoder)){MetalScheduleFree(s);return NULL;}
    return s;
}

// An external command is committed as it was given, once: it has no recorded actions.
struct MetalSchedule *MetalScheduleCommand(id<MTLCommandBuffer> command) {
    struct MetalSchedule *s=calloc(1,sizeof *s);
    if(!s)return NULL;
    atomic_init(&s->references,1);atomic_init(&s->status,0);atomic_init(&s->pending,0);s->root=s;s->generation=1;
    s->device=[command.device retain];s->queue=[command.commandQueue retain];
    s->command=[command retain];s->form=MetalFormSerial;s->commands=1;
    return s;
}

// Before Finish, the input event the function's first part awaits; after it, the value a later
// invocation awaits on that event.
void MetalScheduleAfter(struct MetalSchedule *s,id<MTLEvent> after,uint64_t value) {
    s->root->value=value;
    if(s->root->prepared)return;
    s=MetalScheduleTail(s);
    [s->after release];s->after=[after retain];
    if(s->form==MetalFormSerial&&after)[(id<MTLCommandBuffer>)s->command encodeWaitForEvent:after value:value];
}

void MetalScheduleCommunicator(struct MetalSchedule *s,struct ncclComm *comm) {
    s=s->root;
    if(!s->communicators)s->communicators=[NSMutableOrderedSet new];
    [s->communicators addObject:[NSValue valueWithPointer:comm]];
}

id<MTLResidencySet> MetalScheduleResources(struct MetalSchedule *s) { return MetalScheduleTail(s)->residency; }

struct MetalCommandBinding MetalScheduleData(struct MetalSchedule *s,const void *data,size_t bytes,uint32_t index) {
    s=MetalScheduleTail(s);
    NSUInteger size=(bytes+MESH_COMMAND_DATA_ALIGNMENT-1)/MESH_COMMAND_DATA_ALIGNMENT*MESH_COMMAND_DATA_ALIGNMENT;
    if(!s->data || s->used+size>s->data.length) {
        [s->data release];s->data=[s->device newBufferWithLength:MAX((NSUInteger)MESH_COMMAND_DATA_BYTES,size) options:MTLResourceStorageModeShared];
        s->used=0;[s->owners addObject:s->data];[s->residency addAllocation:s->data];
    }
    struct MetalCommandBinding result={s->data,s->used,bytes,index,MetalReachRead};
    memcpy((char *)s->data.contents+s->used,data,bytes);s->used+=size;
    return result;
}

// An action encodes into the part it is given; at preparation the writable part, which records it.
static int MetalScheduleDo(struct MetalSchedule *root,MetalAction action) {
    struct MetalSchedule *s=MetalScheduleWritable(root);
    [s->actions addObject:action];
    return action(s);
}

static void MetalPartCompute(struct MetalSchedule *s,id<MTLComputePipelineState> pipeline,
    const struct MetalCommandBinding *bindings,size_t count,MTLSize groups,MTLSize threads) {
    s->commands++;
    struct MetalRecordedDispatch command={.pipeline=pipeline,.grid=groups,.group=threads};
    struct MetalReplay view={0};
    if(!s->again)[s->owners addObject:pipeline];
    for(size_t k=0;k<count;k++) {
        const struct MetalCommandBinding b=bindings[k];
        if(b.index>=MESH_BUFFER_BINDINGS)[NSException raise:NSInvalidArgumentException format:@"binding index exceeds Metal capacity"];
        command.buffers[b.index]=b.buffer;command.offsets[b.index]=b.offset;command.lengths[b.index]=b.length;
        command.access|=(uint64_t)(b.access&MetalReachRead?1:0)<<b.index;
        command.access|=(uint64_t)(b.access&MetalReachWritten?1:0)<<(MESH_ACCESS_WRITE_SHIFT+b.index);
        view.buffers=MAX(view.buffers,b.index+1);
        if(!s->again){[s->owners addObject:b.buffer];[s->residency addAllocation:b.buffer];}
    }
    BOOL barrier=MetalOrder(s->read,s->written,&s->phase,&command,NULL,0);
    if(s->form==MetalFormSerial) {
        if(!s->encoder)s->encoder=[[(id<MTLCommandBuffer>)s->command computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial] retain];
        MetalReplayDispatch(s->encoder,&view,&command,0,NULL);
    } else {
        MetalEncoderAccesses(s->encoder,&command,NULL,0);
        if(barrier)[(id<MTL4ComputeCommandEncoder>)s->encoder barrierAfterEncoderStages:MTLStageDispatch
            beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        MetalReplayDispatch4(s->encoder,s->table,&view,&command,0);
    }
}

static MetalWork *MetalScheduleWork(struct MetalSchedule *root,NSData *accesses,MetalAction encode,NSError *(^operation)(void)) {
    MetalWork *work=[MetalWork new];work->accesses=[accesses copy];work->encode=[encode copy];work->operation=[operation copy];
    work->dependencies=[NSMutableIndexSet new];work->successors=[NSMutableIndexSet new];
    const struct MetalReach *views=accesses.bytes;
    for(NSUInteger k=0;k<accesses.length/sizeof *views;k++)if(views[k].buffer)[root->owners addObject:views[k].buffer];
    [root->work addObject:work];[work release];return work;
}

void MetalScheduleCompute(struct MetalSchedule *root,id<MTLComputePipelineState> pipeline,
    const struct MetalCommandBinding *bindings,size_t count,MTLSize groups,MTLSize threads) { @autoreleasepool {
    NSData *bound=[NSData dataWithBytes:bindings length:count*sizeof *bindings];
    NSMutableData *accesses=[NSMutableData data];
    for(size_t k=0;k<count;k++) {
        const struct MetalCommandBinding b=bindings[k];
        const struct MetalReach reach={b.buffer,b.offset,b.length,b.access,-1};
        [accesses appendBytes:&reach length:sizeof reach];
    }
    MetalScheduleWork(root,accesses,^int(struct MetalSchedule *s){
        MetalPartCompute(s,pipeline,bound.bytes,count,groups,threads);return 0;
    },nil);
}
}

int MetalScheduleCross(struct MetalSchedule *root,MetalCrossing cross) {
    MetalScheduleFlush(root);
    // Collective marks preserve the declared transport order. All preceding
    // host outputs are available before this opaque transport declaration.
    for(MetalWork *host in root->hosts)MetalScheduleWait(root,host->event);
    int error=MetalScheduleDo(root,[[^int(struct MetalSchedule *s){
        s->commands++;
        int failed=cross(s->encoder,s->table,s->residency,0,0);
        [s->read removeAllObjects];[s->written removeAllObjects];
        return failed;
    } copy] autorelease]);
    root->boundary=MetalScheduleTail(root);
    return error;
}

void MetalScheduleHost(struct MetalSchedule *root,const struct MetalReach *views,size_t count,NSError *(^operation)(void)) { @autoreleasepool {
    if(!operation||!count)[NSException raise:NSInvalidArgumentException format:@"host operation needs declared data views"];
    MetalScheduleWork(root,[NSData dataWithBytes:views length:count*sizeof *views],nil,operation);
}
}

static void MetalPartRange(struct MetalSchedule *s,struct MetalReplay *replay,NSUInteger t,NSUInteger from,NSUInteger to) {
    s->commands+=to-from;
    NSArray *held=objc_getAssociatedObject(replay->owner,&replayRecordsKey);
    const struct MetalRecordedDispatch *records=[held[0] bytes];
    const struct MetalReach *reaches=[held[3] bytes];
    const uint8_t *barriers=[held[1] bytes];
    const NSUInteger n=replay->count;
    if(!s->again) {
        [s->owners addObject:replay->owner];
        for(id<MTLAllocation> allocation in replay->residency.allAllocations)[s->residency addAllocation:allocation];
    }
    if(s->form==MetalFormSerial&&!s->encoder)
        s->encoder=[[(id<MTLCommandBuffer>)s->command computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial] retain];
    id<MTL4ComputeCommandEncoder> encoder=s->encoder;
    NSUInteger pending=t*n+from;
    for(NSUInteger c=pending;c<t*n+to;c++) @autoreleasepool {
        const struct MetalRecordedDispatch *source=records+c%n;
        if(s->form!=MetalFormSerial)MetalEncoderAccesses(encoder,source,reaches,t);
        BOOL barrier=MetalOrder(s->read,s->written,&s->phase,source,reaches,t);
        if(s->form==MetalFormSerial)MetalReplayDispatch(s->encoder,replay,source,t,NULL);
        else if(s->form==MetalFormEncoded) {
            if(barrier)[encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
            MetalReplayDispatch4(encoder,s->table,replay,source,t);
        } else if(barrier&&(!barriers[c]||c==pending)) {
            MetalReplaySegments(replay,NSMakeRange(pending,c-pending),^(id<MTLIndirectCommandBuffer> segment,NSRange local){
                [encoder executeCommandsInBuffer:segment withRange:local];
            });
            [encoder barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
            pending=c;
        }
    }
    if(s->form==MetalFormIndirect&&t*n+to>pending)MetalReplaySegments(replay,NSMakeRange(pending,t*n+to-pending),^(id<MTLIndirectCommandBuffer> segment,NSRange local){
        [encoder executeCommandsInBuffer:segment withRange:local];
    });
}

static void MetalScheduleRangeNow(struct MetalSchedule *root,struct MetalReplay *replay,NSUInteger t,NSUInteger from,NSUInteger to) {
    if(!replay||from==to){MetalScheduleWritable(root);return;}
    NSData *kept=[NSData dataWithBytes:replay length:sizeof *replay];
    MetalScheduleDo(root,[[^int(struct MetalSchedule *s){
        struct MetalReplay copy;memcpy(&copy,kept.bytes,sizeof copy);
        MetalPartRange(s,&copy,t,from,to);return 0;
    } copy] autorelease]);
}

static void MetalScheduleRange(struct MetalSchedule *root,struct MetalReplay *replay,NSUInteger t,NSUInteger from,NSUInteger to) {
    if(!replay||from==to)return;
    [root->owners addObject:replay->owner];[root->owners addObject:replay->residency];
    MetalWork *work=MetalScheduleWork(root,nil,nil,nil);
    memcpy(work->replay,replay,sizeof *replay);work->invocation=t;work->first=from;work->end=to;
}

// Only mixed host/GPU spans need per-command dependency nodes. GPU-only
// functions keep the original compact replay ranges throughout preparation.
static NSArray *MetalScheduleExpand(struct MetalSchedule *root,NSArray *ranges) {
    for(MetalWork *range in ranges) {
        if(!range->end){[root->work addObject:range];continue;}
        struct MetalReplay replay;memcpy(&replay,range->replay,sizeof replay);
        NSArray *held=objc_getAssociatedObject(replay.owner,&replayRecordsKey);
        const struct MetalRecordedDispatch *records=[held[0] bytes];
        const struct MetalReach *reaches=[held[3] bytes];
        for(NSUInteger c=range->first;c<range->end;c++) {
            NSData *accesses=MetalCommandReaches(records+c,reaches,range->invocation);
            MetalWork *work=MetalScheduleWork(root,accesses,nil,nil);
            memcpy(work->replay,&replay,sizeof replay);work->invocation=range->invocation;work->first=c;work->end=c+1;
        }
    }
    NSArray *nodes=[[root->work copy] autorelease];[root->work removeAllObjects];return nodes;
}

static struct MetalSchedule *MetalScheduleWritable(struct MetalSchedule *root) {
    struct MetalSchedule *tail=MetalScheduleTail(root);
    if(!tail->finished)return tail;
    struct MetalSchedule *next=MetalScheduleCreateOn(tail->queue,tail->form);
    if(!next)[NSException raise:NSMallocException format:@"preparing data-dependent command part"];
    next->root=root;next->ordinal=tail->ordinal+1;tail->next=next;root->tail=next;
    return next;
}

NSString *MetalViewDependencyDeclaration(NSString *name,NSArray<NSIndexSet *> *dependencies,NSUInteger producers) {
    NSMutableArray *successors=[NSMutableArray arrayWithCapacity:producers];
    for(NSUInteger i=0;i<producers;i++)[successors addObject:[NSMutableIndexSet indexSet]];
    NSMutableData *offsets=[NSMutableData data],*targets=[NSMutableData data],*holders=[NSMutableData data];
    for(NSUInteger i=producers;i<dependencies.count;i++) {
        NSIndexSet *prior=dependencies[i];const uint32_t count=(uint32_t)prior.count;
        [holders appendBytes:&count length:sizeof count];
        [prior enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){(void)stop;[successors[p] addIndex:i-producers];}];
    }
    uint32_t offset=0;[offsets appendBytes:&offset length:sizeof offset];
    for(NSIndexSet *next in successors) {
        for(NSUInteger i=next.firstIndex;i!=NSNotFound;i=[next indexGreaterThanIndex:i]) {
            const uint32_t target=(uint32_t)i;[targets appendBytes:&target length:sizeof target];offset++;
        }
        [offsets appendBytes:&offset length:sizeof offset];
    }
    NSArray *arrays=@[offsets,targets,holders];
    NSMutableString *source=[NSMutableString stringWithFormat:@"namespace %@_indices {\n%@",name,MetalViewIndexTemplates()];
    for(NSUInteger binding=0;binding<arrays.count;binding++) {
        NSData *data=arrays[binding];const uint32_t *values=data.bytes;
        const NSUInteger count=data.length/sizeof *values;const NSInteger stride=0;
        NSMutableData *map=[NSMutableData dataWithLength:count*sizeof(NSInteger)];NSInteger *indices=map.mutableBytes;
        for(NSUInteger i=0;i<count;i++)indices[i]=values[i];
        [source appendString:MetalViewMappedIndexDeclaration(binding,&count,&stride,1,0,@{@0:map})];
    }
    [source appendFormat:@"}\nstruct %@ { uint producer;\n"
        "uint begin() const { return uint(%@_indices::mesh_index<0>(producer)); }\n"
        "uint end() const { return uint(%@_indices::mesh_index<0>(producer+1)); }\n"
        "static uint target(uint i) { return uint(%@_indices::mesh_index<1>(i)); }\n"
        "static uint holders(uint g) { return uint(%@_indices::mesh_index<2>(g)); }\n};\n",name,name,name,name,name];
    return source;
}

static void MetalWorkDependencies(NSArray *work) {
    NSMutableArray *accesses=[NSMutableArray arrayWithCapacity:work.count];
    for(MetalWork *w in work)[accesses addObject:w->accesses];
    NSArray *dependencies=MetalViewDependencies(accesses);
    for(NSUInteger i=0;i<work.count;i++) {
        MetalWork *w=work[i];[w->dependencies addIndexes:dependencies[i]];
        [w->dependencies enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){(void)stop;[((MetalWork *)work[p])->successors addIndex:i];}];
    }
}

static id<MTLSharedEvent> MetalSchedulePublication(struct MetalSchedule *s) {
    if(!s->event) {
        s->event=[s->device newSharedEvent];
        if(!s->event)[NSException raise:NSMallocException format:@"preparing GPU input publication"];
        // A later host node can name an already closed part. It is still
        // unsubmitted, so its publication can be encoded during preparation.
        if(s->finished&&s->form==MetalFormSerial)
            [(id<MTLCommandBuffer>)s->command encodeSignalEvent:s->event value:s->root->generation];
    }
    MetalScheduleFinishPart(s);return s->event;
}

// A host operation's prerequisites: each an event and whether its value is the caller's input
// value (MetalScheduleAfter), else the invocation's.
static void MetalScheduleBindHost(struct MetalSchedule *root,MetalWork *work,NSArray *nodes) {
    NSMutableDictionary *events=[NSMutableDictionary dictionary];
    void (^add)(id,BOOL)=^(id event,BOOL input){events[[NSValue valueWithPointer:event]]=@[event,@(input)];};
    __block struct MetalSchedule *latest=root->boundary;
    [work->dependencies enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){
        (void)stop;MetalWork *producer=nodes[p];
        if(producer->operation)add(producer->event,NO);
        else if(!latest||producer->part->ordinal>latest->ordinal)latest=producer->part;
    }];
    // Queue order makes the latest needed GPU part cover earlier GPU parts.
    // An internal predecessor already carries the function's input dependency.
    if(latest)add(MetalSchedulePublication(latest),NO);
    if(!events.count) {
        if(!root->start)root->start=[root->device newSharedEvent];
        if(!root->start)[NSException raise:NSMallocException format:@"preparing native host start"];
        add(root->start,NO);
        if(root->after)add(root->after,YES);
    }
    work->prerequisites=[events.allValues mutableCopy];
}

// Each invocation's notifications: only the last actual prerequisite invokes the native
// operation; no Python runs. A task's block retains it until the block runs.
static void MetalScheduleArm(struct MetalSchedule *root) {
    const uint64_t g=root->generation,input=root->value;
    for(MetalWork *task in root->hosts) {
        [task->error release];task->error=nil;
        atomic_store_explicit(&task->status,0,memory_order_relaxed);
        atomic_store_explicit(&task->remaining,(unsigned)task->prerequisites.count,memory_order_release);
    }
    for(MetalWork *task in root->hosts)for(NSArray *prerequisite in task->prerequisites)
        [(id<MTLSharedEvent>)prerequisite[0] notifyListener:MetalListener() atValue:[prerequisite[1] boolValue]?input:g
            block:^(id<MTLSharedEvent> reached,uint64_t value){
                (void)reached;(void)value;
                if(atomic_fetch_sub_explicit(&task->remaining,1,memory_order_acq_rel)!=1)return;
                NSError *error=task->operation();task->error=[error retain];
                atomic_store_explicit(&task->status,error?-1:1,memory_order_release);
                task->event.signaledValue=g;
                MetalScheduleArrived(root);
            }];
}

// Adjacent selected dispatches retain the recorder's batched ICB execution.
// The dependency walk only chooses an order; it does not split a range into
// individual executeCommands calls.
static void MetalScheduleEmit(struct MetalSchedule *root,NSArray *batch) {
    for(NSUInteger i=0;i<batch.count;) {
        MetalWork *work=batch[i];NSUInteger end=work->end,j=i+1;
        if(work->end) {
            struct MetalReplay replay;memcpy(&replay,work->replay,sizeof replay);
            while(j<batch.count) {
                MetalWork *next=batch[j];
                if(!next->end||next->invocation!=work->invocation||next->first!=end||
                   memcmp(next->replay,work->replay,sizeof replay))break;
                end=next->end;j++;
            }
            MetalScheduleRangeNow(root,&replay,work->invocation,work->first,end);
        } else MetalScheduleDo(root,work->encode);
        for(;i<j;i++)((MetalWork *)batch[i])->part=MetalScheduleTail(root);
    }
}

static void MetalScheduleFlush(struct MetalSchedule *root) { @autoreleasepool {
    NSArray *nodes=[[root->work copy] autorelease];[root->work removeAllObjects];
    BOOL hosts=NO;for(MetalWork *work in nodes)hosts|=work->operation!=nil;
    if(!hosts){MetalScheduleEmit(root,nodes);return;}
    nodes=MetalScheduleExpand(root,nodes);
    for(MetalWork *work in nodes)if(work->operation) {
        work->event=[root->device newSharedEvent];
        if(!work->event)[NSException raise:NSMallocException format:@"preparing host output event"];
        atomic_init(&work->status,0);[root->hosts addObject:work];
    }
    MetalWorkDependencies(nodes);
    NSMutableData *counts=[NSMutableData dataWithLength:nodes.count*sizeof(NSUInteger)];NSUInteger *left=counts.mutableBytes;
    NSMutableIndexSet *ready=[NSMutableIndexSet indexSet];
    for(NSUInteger i=0;i<nodes.count;i++){left[i]=((MetalWork *)nodes[i])->dependencies.count;if(!left[i])[ready addIndex:i];}
    NSMutableArray *batch=[NSMutableArray array];
    while(ready.count) {
        __block NSUInteger selected=NSNotFound;
        [ready enumerateIndexesUsingBlock:^(NSUInteger i,BOOL *stop){
            MetalWork *work=nodes[i];
            if(work->operation){selected=i;*stop=YES;return;}
            __block BOOL available=YES;
            [work->dependencies enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *unused){
                (void)unused;MetalWork *producer=nodes[p];
                if(producer->operation&&![root->waited containsObject:producer->event])available=NO;
            }];
            if(available&&selected==NSNotFound)selected=i;
        }];
        if(selected==NSNotFound)selected=ready.firstIndex;
        [ready removeIndex:selected];MetalWork *work=nodes[selected];
        if(work->operation) {
            MetalScheduleEmit(root,batch);[batch removeAllObjects];
            MetalScheduleBindHost(root,work,nodes);
        } else {
            [work->dependencies enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){
                (void)stop;MetalWork *producer=nodes[p];
                if(producer->operation&&![root->waited containsObject:producer->event]) {
                    MetalScheduleEmit(root,batch);[batch removeAllObjects];
                    MetalScheduleWait(root,producer->event);
                }
            }];
            [batch addObject:work];
        }
        [work->successors enumerateIndexesUsingBlock:^(NSUInteger i,BOOL *stop){(void)stop;if(!--left[i])[ready addIndex:i];}];
    }
    MetalScheduleEmit(root,batch);
}
}

NSError *MetalScheduleAppend(struct MetalSchedule *root,struct MetalReplay *replay,NSRange invocations,
    NSArray<NSNumber *> *marks,MetalCrossing cross) {
    struct MetalSchedule *s=MetalScheduleTail(root);
    NSArray *held=replay?objc_getAssociatedObject(replay->owner,&replayRecordsKey):nil;
    if(replay&&[held[2] unsignedIntegerValue]!=s->form)return MetalReplayFailure(EINVAL,@"mixed replay forms in one function");
    if(s->form==MetalFormSerial&&marks.count)return MetalReplayFailure(EINVAL,@"crossings require the configured Metal 4 form");
    if(replay&&NSMaxRange(invocations)>[held[4] unsignedIntegerValue])return MetalReplayFailure(EINVAL,@"invocations exceed the recorded slots");
    NSArray *hosts=replay?objc_getAssociatedObject(replay->owner,&replayHostsKey):nil;
    const NSUInteger n=replay?replay->count:0;
    for(NSUInteger t=invocations.location;t<NSMaxRange(invocations);t++) {
        NSUInteger from=0,k=0,h=0;
        while(k<marks.count||h<hosts.count) {
            NSUInteger at=k<marks.count?marks[k].unsignedIntegerValue:n;
            if(h<hosts.count)at=MIN(at,[hosts[h][@"at"] unsignedIntegerValue]);
            if(at<from||at>n)return MetalReplayFailure(EINVAL,@"dependency outside its recorded command range");
            MetalScheduleRange(root,replay,t,from,at);
            while(h<hosts.count&&[hosts[h][@"at"] unsignedIntegerValue]==at) {
                NSDictionary *host=hosts[h++];NSData *views=host[@"accesses"];
                MetalScheduleHost(root,views.bytes,views.length/sizeof(struct MetalReach),host[@"operation"]);
            }
            while(k<marks.count&&marks[k].unsignedIntegerValue==at) {
                const NSUInteger mark=k++;
                int failed=MetalScheduleCross(root,^(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,
                    id<MTLResidencySet> residency,NSUInteger unused,NSUInteger ignored){
                    (void)unused;(void)ignored;return cross(encoder,table,residency,t,mark);
                });
                if(failed)return MetalReplayFailure(failed,@"the declared crossing could not be encoded");
            }
            from=at;
        }
        MetalScheduleRange(root,replay,t,from,n);
    }
    return nil;
}

// An invocation ends when its submission, every part's completion and publication, and every
// host operation have arrived.
static void MetalScheduleArrived(struct MetalSchedule *root) {
    if(atomic_fetch_sub_explicit(&root->pending,1,memory_order_acq_rel)==1) {
        NSError *failure=nil;
        MetalScheduleStatus(root,&failure);
        if(root->completed)root->completed(failure);
        MetalScheduleFree(root);
    }
}

static void MetalScheduleCompleted(struct MetalSchedule *s,NSError *error) {
    [s->error release];s->error=[error retain];
    atomic_store_explicit(&s->status,error?-1:1,memory_order_release);
    MetalScheduleArrived(s->root);
    MetalScheduleFree(s);
}

static void MetalScheduleFinishPart(struct MetalSchedule *s) {
    if(s->finished)return;
    [s->residency commit];[s->residency requestResidency];
    MetalPartEnd(s);s->finished=YES;
    if(!s->commands){atomic_store_explicit(&s->status,1,memory_order_relaxed);return;}
    struct MetalSchedule *root=s->root;
    if(root->lastSubmission)root->lastSubmission->submit=s;else root->firstSubmission=s;
    root->lastSubmission=s;
}

int MetalScheduleFinish(struct MetalSchedule *root,void (^completed)(NSError *)) {
    MetalScheduleFlush(root);
    [root->completed release];root->completed=[completed copy];
    MetalScheduleFinishPart(MetalScheduleTail(root));
    root->prepared=YES;
    for(NSValue *value in root->communicators) {
        const ncclResult_t error=ncclMeshCommRealized(value.pointerValue);
        if(error)return error;
    }
    return 0;
}

int MetalScheduleStatus(const struct MetalSchedule *s,NSError **error) {
    *error=nil;
    const struct MetalSchedule *root=s->root;
    int result=1;
    for(;s;s=s->next){int state=atomic_load_explicit(&s->status,memory_order_acquire);if(state<0){*error=s->error;return -1;}if(!state)result=0;}
    for(MetalWork *host in root->hosts) {
        int state=atomic_load_explicit(&host->status,memory_order_acquire);
        if(state<0){*error=host->error;return -1;}if(!state)result=0;
    }
    for(NSValue *value in root->communicators) {
        ncclResult_t status=ncclSuccess;
        ncclResult_t call=ncclCommGetAsyncError(value.pointerValue,&status);
        if(call||status!=ncclSuccess) {
            *error=MetalReplayFailure(ENOTCONN,@(ncclGetLastError(value.pointerValue)));
            return -1;
        }
    }
    return result;
}

int MetalScheduleEnqueue(struct MetalSchedule *root) {
    if(root!=root->root||!root->prepared)return EINVAL;
    if(atomic_load_explicit(&root->pending,memory_order_acquire))return EBUSY;
    struct MetalSchedule *s;
    if(root->ran) {
        if(!root->actions)return EALREADY;
        root->generation++;
        for(s=root->firstSubmission;s;s=s->submit){const int error=MetalPartEncode(s);if(error)return error;}
    }
    root->ran=YES;
    const uint64_t g=root->generation;
    unsigned count=1+(unsigned)root->hosts.count;
    for(s=root->firstSubmission;s;s=s->submit){count+=1+(s->event&&s->form!=MetalFormSerial);atomic_store_explicit(&s->status,0,memory_order_relaxed);}
    atomic_store_explicit(&root->pending,count,memory_order_release);
    // One reference spans submission, another spans all native/GPU results.
    atomic_fetch_add_explicit(&root->references,2,memory_order_relaxed);
    MetalScheduleArm(root);
    for(s=root->firstSubmission;s;) {
        // Completion may resume the owner and release the whole function
        // before commit returns. Read the compiled successor before commit.
        struct MetalSchedule *next=s->submit;
        atomic_fetch_add_explicit(&s->references,1,memory_order_relaxed);
        if(s->form==MetalFormSerial)[(id<MTLCommandBuffer>)s->command commit];
        else {
            if(s->after)[(id<MTL4CommandQueue>)s->queue waitForEvent:s->after value:root->value];
            for(id<MTLSharedEvent> event in s->waits)[(id<MTL4CommandQueue>)s->queue waitForEvent:event value:g];
            // The queue's signal runs after the feedback handler may have; the function, and so
            // its event, lives until the signal is seen.
            if(s->event)[s->event notifyListener:MetalListener() atValue:g block:^(id<MTLSharedEvent> e,uint64_t v){
                (void)e;(void)v;MetalScheduleArrived(root);
            }];
            // A commit's options call their feedback handler once.
            MTL4CommitOptions *options=[MTL4CommitOptions new];
            struct MetalSchedule *part=s;
            [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback){MetalScheduleCompleted(part,feedback.error);}];
            [(id<MTL4CommandQueue>)s->queue commit:(id<MTL4CommandBuffer> *)&s->command count:1 options:options];
            [options release];
            if(s->event)[(id<MTL4CommandQueue>)s->queue signalEvent:s->event value:g];
        }
        s=next;
    }
    root->start.signaledValue=g;
    // The initial pending count covers submission itself, including a function
    // with only native host work or no commands. Empty GPU parts are not sent.
    MetalScheduleArrived(root);
    MetalScheduleFree(root);
    return 0;
}

void MetalScheduleFree(struct MetalSchedule *s) {
    if(!s)return;
    if(atomic_fetch_sub_explicit(&s->references,1,memory_order_acq_rel)!=1)return;
    MetalScheduleFree(s->next);
    if(!s->finished&&s->encoder) {
        [s->encoder endEncoding];
        if(s->form!=MetalFormSerial)[(id<MTL4CommandBuffer>)s->command endCommandBuffer];
    }
    [s->encoder release];[s->command release];[s->queue release];[s->allocator release];[s->table release];[s->data release];
    [s->residency release];[s->owners release];[s->read release];[s->written release];[s->actions release];
    [s->event release];[s->start release];[s->work release];[s->hosts release];[s->waits release];[s->waited release];
    [s->completed release];[s->error release];[s->after release];[s->device release];[s->communicators release];free(s);
}

id<MTLDevice> MetalRecordingDevice(id<MTLDevice> device,BOOL indirectPipelines) {
    MetalDeviceRecorder *recording=[[MetalDeviceRecorder alloc] initWithTarget:device];
    recording->indirect=indirectPipelines;
    return [(id)recording autorelease];
}

// docs/kernels.md Any program: recorded with the dispatch on a recording encoder; on any other, each reach but a
// binding's extent (bound already) a resource the dispatch uses as its access says
static void MetalDispatchAccesses(id<MTLComputeCommandEncoder> encoder, MTLSize groups, MTLSize threads,
    BOOL threadGrid, const struct MetalReach *reaches, size_t count) {
    if (object_getClass(encoder) == [MetalComputeRecorder class]) {
        [(MetalComputeRecorder *)encoder dispatchGrid:groups threadsPerThreadgroup:threads threadGrid:threadGrid reaches:reaches count:count];
        return;
    }
    for (size_t k = 0; k < count; k++)
        if (!(reaches[k].access & MetalReachBinding))
            [encoder useResource:reaches[k].buffer usage:((reaches[k].access & MetalReachRead) ? MTLResourceUsageRead : 0) |
                ((reaches[k].access & MetalReachWritten) ? MTLResourceUsageWrite : 0)];
    if(threadGrid)[encoder dispatchThreads:groups threadsPerThreadgroup:threads];
    else [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

void MetalDispatchReaching(id<MTLComputeCommandEncoder> encoder, MTLSize groups, MTLSize threads,
    const struct MetalReach *reaches, size_t count) {
    MetalDispatchAccesses(encoder,groups,threads,NO,reaches,count);
}

NSUInteger MetalRecordMark(void) {
    return activeCapture ? activeCapture->dispatches.length / sizeof(struct MetalRecordedDispatch) : NSNotFound;
}

void MetalRecordHost(const struct MetalReach *views,size_t count,NSError *(^operation)(void)) {
    if(!activeCapture)[NSException raise:NSInvalidArgumentException format:@"host shares require whole-function recording"];
    if(!operation||!count)[NSException raise:NSInvalidArgumentException format:@"host operation needs declared data views"];
    for(size_t k=0;k<count;k++)if(views[k].buffer)[activeCapture->owners addObject:views[k].buffer];
    [activeCapture->hosts addObject:@{@"at":@(MetalRecordMark()),@"accesses":[NSData dataWithBytes:views length:count*sizeof *views],
        @"operation":[[operation copy] autorelease]}];
}

// docs/kernels.md Any program
struct MetalReplay *MetalRecord(id<MTLDevice> device, void (^encode)(void), const struct MetalOperand *operands,
    size_t count, NSUInteger invocations, enum MetalForm form) {
    if (object_getClass(device) != [MetalDeviceRecorder class]) {
        fprintf(stderr, "record: the device is not a recording device (MetalRecordingDevice)\n");
        return NULL;
    }
    struct MetalReplay *replay = NULL;
    @autoreleasepool {
        // ../dox/mesh/design/prepared-machine.md#M13: each operand's slots and output offset travel
        // with its buffer, a later declaration of the same buffer replacing an earlier one
        for (size_t k = 0; k < count; k++) {
            objc_setAssociatedObject(operands[k].buffer, &operandStrideKey, @(operands[k].stride), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            objc_setAssociatedObject(operands[k].buffer, &operandInstancesKey, @(operands[k].instances), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
            objc_setAssociatedObject(operands[k].buffer, &operandOutputOffsetKey, @(operands[k].outputOffset), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        }
        @try {
            struct MetalReplay recorded = {0};
            MetalRecordReplay(device, encode, invocations, form, &recorded);
            NSArray *held = objc_getAssociatedObject(recorded.owner, &replayRecordsKey);
            const struct MetalRecordedDispatch *records = [held[0] bytes];
            for (size_t k = 0; k < count; k++) {
                NSUInteger binding = 0;
                for (uint32_t i = 0; i < recorded.count; i++) for (NSUInteger j = 0; j < MESH_BUFFER_BINDINGS; j++)
                    binding += records[i].buffers[j] == operands[k].buffer && (((records[i].access | (records[i].access >> MESH_ACCESS_WRITE_SHIFT)) >> j) & 1);
                printf("prepared operand %s bindings %lu instances %lu\n", operands[k].name, binding, operands[k].instances);
            }
            if (posix_memalign((void **)&replay, _Alignof(struct MetalReplay), sizeof *replay)) MetalReplayDestroy(&recorded);
            else {
                *replay = recorded;
                if(replay->commands) {
                    id<MTLCommandQueue> queue=[replay->residency.device newCommandQueue];
                    id<MTLCommandBuffer> command=[queue commandBuffer];
                    id<MTLBlitCommandEncoder> encoder=[command blitCommandEncoder];
                    MetalReplaySegments(replay,NSMakeRange(0,(NSUInteger)replay->count*invocations),^(id<MTLIndirectCommandBuffer> segment,NSRange local){
                        [encoder optimizeIndirectCommandBuffer:segment withRange:local];
                    });
                    [encoder endEncoding];[command commit];[command waitUntilCompleted];
                    NSError *error=command.error;[queue release];
                    if(error){MetalReplayFree(replay);replay=NULL;[NSException raise:NSInternalInconsistencyException format:@"%@",error];}
                }


            }
        } @catch (NSException *error) {
            fprintf(stderr, "record: %s\n", error.reason.UTF8String);
        }
    }
    return replay;
}

void MetalReplayFree(struct MetalReplay *replay) {
    if (!replay) return;
    MetalReplayDestroy(replay);
    free(replay);
}
