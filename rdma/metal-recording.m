#import "metal-recording.h"
#include "nccl.h"
#import <objc/runtime.h>
#include <stdatomic.h>
#include <time.h>
#include <os/os_sync_wait_on_address.h>
#include <limits.h>

BOOL MetalViewLayoutMatches(const NSUInteger *shape,const NSInteger *strides,const NSInteger *required,size_t rank) {
    for(size_t d=0;d<rank;d++)if(!shape[d])return YES;
    for(size_t d=0;d<rank;d++)if(shape[d]>1&&strides[d]!=required[d])return NO;
    return YES;
}

// The axes a view's index sums, adjacent ones collapsed, innermost first: each its count, divisor and stride. Their
// number, or SIZE_MAX for an empty view (whose every index is 0). Shared by the source form (MetalViewIndexExpression)
// and the constants form (MetalViewIndexConstants).
typedef struct { NSUInteger count, divisor; NSInteger stride; } MetalViewAxis;
static size_t MetalViewAxes(const NSUInteger *shape,const NSInteger *strides,size_t rank,MetalViewAxis *axes) {
    NSUInteger divisor=1,used=0;
    for(size_t d=rank;d--;) {
        if(!shape[d])return SIZE_MAX;
        if(shape[d]==1)continue;
        if(used&&strides[d]==axes[used-1].stride*(NSInteger)axes[used-1].count)
            axes[used-1].count*=shape[d];
        else {axes[used].count=shape[d];axes[used].divisor=divisor;axes[used++].stride=strides[d];}
        divisor*=shape[d];
    }
    return used;
}

NSString *MetalViewIndexExpression(const NSUInteger *shape,const NSInteger *strides,size_t rank,NSString *index) {
    // Collapse adjacent axes before emitting arithmetic. Contiguous views
    // become the original index, with no table, descriptor load or branch.
    MetalViewAxis axes[rank?rank:1];
    const size_t used=MetalViewAxes(shape,strides,rank,axes);
    if(used==SIZE_MAX)return @"0";
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

// design/compile-once-2026-10-29.md: binding B's affine layout as function constants, at MESH_FC_FIRST + 32 B + its
// field (0 set, 1 axes, 2 base, 3 vector span, then each axis a's count, divisor and stride at 4 + 3a), so one
// library of a source serves every command and each command's pipeline is specialized from its own values.
#define MESH_FC_FIRST 1024u
#define MESH_FC_AXES 8u
#define MESH_FC_BINDINGS 16u
static NSUInteger MetalViewConstant(NSUInteger binding,NSUInteger field) { return MESH_FC_FIRST+32u*binding+field; }

NSString *MetalViewIndexConstantPrelude(void) {
    NSMutableString *s=[NSMutableString stringWithString:@"#ifndef MESH_INDEX_DECLARED\n#include <metal_stdlib>\nusing namespace metal;\n"
        "#define MESH_INDEX_DECLARED 1\n"
        "template<uint B> inline long mesh_index(ulong i) { return long(i); }\ntemplate<uint B, uint A> constexpr long mesh_stride();\n"
        "template<uint B> inline ulong mesh_vector_span() { return 0; }\ntemplate<uint B> inline bool mesh_vector4() { return mesh_vector_span<B>() % 4 == 0; }\n"];
    const char *names[3]={"count","divisor","stride"},*types[3]={"ulong","ulong","long"};
    for(NSUInteger b=0;b<MESH_FC_BINDINGS;b++) {
        [s appendFormat:@"constant bool mesh_set_%lu [[function_constant(%lu)]];\n",(unsigned long)b,(unsigned long)MetalViewConstant(b,0)];
        [s appendFormat:@"constant bool mesh_on_%lu = is_function_constant_defined(mesh_set_%lu) && mesh_set_%lu;\n",(unsigned long)b,(unsigned long)b,(unsigned long)b];
        [s appendFormat:@"constant uint mesh_axes_%lu_v [[function_constant(%lu)]];\nconstant uint mesh_axes_%lu = mesh_on_%lu ? mesh_axes_%lu_v : 0;\n",
            (unsigned long)b,(unsigned long)MetalViewConstant(b,1),(unsigned long)b,(unsigned long)b,(unsigned long)b];
        [s appendFormat:@"constant long mesh_base_%lu_v [[function_constant(%lu)]];\nconstant long mesh_base_%lu = mesh_on_%lu ? mesh_base_%lu_v : 0;\n",
            (unsigned long)b,(unsigned long)MetalViewConstant(b,2),(unsigned long)b,(unsigned long)b,(unsigned long)b];
        [s appendFormat:@"constant ulong mesh_span_%lu_v [[function_constant(%lu)]];\nconstant ulong mesh_span_%lu = mesh_on_%lu ? mesh_span_%lu_v : 0;\n",
            (unsigned long)b,(unsigned long)MetalViewConstant(b,3),(unsigned long)b,(unsigned long)b,(unsigned long)b];
        for(NSUInteger a=0;a<MESH_FC_AXES;a++) for(int f=0;f<3;f++)
            [s appendFormat:@"constant %s mesh_%s_%lu_%lu_v [[function_constant(%lu)]];\nconstant %s mesh_%s_%lu_%lu = mesh_axes_%lu > %lu ? mesh_%s_%lu_%lu_v : %s;\n",
                types[f],names[f],(unsigned long)b,(unsigned long)a,(unsigned long)MetalViewConstant(b,4+3*a+(NSUInteger)f),types[f],names[f],
                (unsigned long)b,(unsigned long)a,(unsigned long)b,(unsigned long)a,names[f],(unsigned long)b,(unsigned long)a,f==2?"0":"1"];
        [s appendFormat:@"template<> inline long mesh_index<%lu>(ulong i) {\n  if (!mesh_on_%lu) return long(i);\n  long r = mesh_base_%lu;\n",
            (unsigned long)b,(unsigned long)b,(unsigned long)b];
        for(NSUInteger a=0;a<MESH_FC_AXES;a++)
            [s appendFormat:@"  if (mesh_axes_%lu > %lu) { long q = long(i) / long(mesh_divisor_%lu_%lu); if (mesh_axes_%lu > %lu) q %%= long(mesh_count_%lu_%lu); r += q * mesh_stride_%lu_%lu; }\n",
                (unsigned long)b,(unsigned long)a,(unsigned long)b,(unsigned long)a,(unsigned long)b,(unsigned long)a+1,(unsigned long)b,(unsigned long)a,(unsigned long)b,(unsigned long)a];
        [s appendFormat:@"  return r;\n}\ntemplate<> inline ulong mesh_vector_span<%lu>() { return mesh_span_%lu; }\n",(unsigned long)b,(unsigned long)b];
    }
    [s appendString:@"#endif\n"];
    return s;
}

static NSUInteger MetalViewAlignment(NSUInteger span,NSInteger distance);
// binding's function-constant values (the constants form of MetalViewAffineIndexDeclaration, base 0): each as its
// index, type (0 bool, 1 uint, 2 long, 3 ulong) and value; nil where the view does not fit the form (more axes than
// MESH_FC_AXES, a binding past MESH_FC_BINDINGS, or an empty view)
NSArray<NSArray<NSNumber *> *> *MetalViewIndexConstants(NSUInteger binding,const NSUInteger *shape,const NSInteger *strides,
    size_t rank,NSUInteger elementOffset,NSInteger relativeOffset) {
    MetalViewAxis axes[rank?rank:1];
    const size_t used=MetalViewAxes(shape,strides,rank,axes);
    if(binding>=MESH_FC_BINDINGS||used==SIZE_MAX||used>MESH_FC_AXES)return nil;
    NSUInteger inner=1,prefix=rank;
    for(size_t d=rank;d--;) {
        if(shape[d]>1&&strides[d]!=(NSInteger)inner)break;
        inner*=shape[d];prefix=d;
    }
    NSUInteger vector=MetalViewAlignment(prefix?inner:0,(NSInteger)elementOffset);
    for(size_t d=0;d<prefix;d++)if(shape[d]>1)vector=MetalViewAlignment(vector,strides[d]);
    NSMutableArray *values=[NSMutableArray arrayWithObjects:@[@(MetalViewConstant(binding,0)),@0,@1],@[@(MetalViewConstant(binding,1)),@1,@(used)],
        @[@(MetalViewConstant(binding,2)),@2,@(relativeOffset)],@[@(MetalViewConstant(binding,3)),@3,@(vector)],nil];
    for(size_t a=0;a<used;a++) {
        [values addObject:@[@(MetalViewConstant(binding,4+3*a)),@3,@(axes[a].count)]];
        [values addObject:@[@(MetalViewConstant(binding,5+3*a)),@3,@(axes[a].divisor)]];
        [values addObject:@[@(MetalViewConstant(binding,6+3*a)),@2,@(axes[a].stride)]];
    }
    return values;
}

static NSString *MetalViewIndexTemplates(void) {
    return @"template<uint B> inline long mesh_index(ulong i) { return long(i); }\n"
        "template<uint B, uint A> constexpr long mesh_stride();\n"
        "template<uint B> constexpr bool mesh_affine();\n"
        "template<uint B> constexpr ulong mesh_vector_span() { return 0; }\n"
        "template<uint B> constexpr bool mesh_vector4() { return mesh_vector_span<B>() % 4 == 0; }\n";
}

NSString *MetalViewIndexPrelude(void) {
    return [@"#ifndef MESH_INDEX_DECLARED\n#include <metal_stdlib>\nusing namespace metal;\n#define MESH_INDEX_DECLARED 1\n"
        stringByAppendingFormat:@"%@#endif\n",MetalViewIndexTemplates()];
}

// Each block of `block` stored lanes, `copies` times, keeps the present lanes `select` maps to an output; a block
// (or copy) with none is not retained.
static NSDictionary<NSString *,NSData *> *MetalViewBlockSelection(const int32_t *present,const int32_t *order,size_t count,
    size_t stored,NSUInteger block,NSUInteger copies,BOOL (^select)(int32_t,NSUInteger,int32_t *)) {
    NSMutableData *blocks=[NSMutableData data],*positions=[NSMutableData data],*mapped=[NSMutableData data];
    int32_t retained=0;
    for(size_t start=0,at=0;start<stored;start+=block) {
        const size_t first=at;
        while(at<count&&(size_t)present[at]<start+block)at++;
        for(NSUInteger copy=0;copy<copies;copy++) {
            BOOL used=NO;
            for(size_t k=first;k<at;k++) {
                int32_t output;
                if(!select(order[k],copy,&output))continue;
                const int32_t position=retained*(int32_t)block+present[k]-(int32_t)start;
                [positions appendBytes:&position length:sizeof position];
                [mapped appendBytes:&output length:sizeof output];used=YES;
            }
            if(!used)break;
            const int32_t index=(int32_t)(start/block);
            [blocks appendBytes:&index length:sizeof index];retained++;
        }
    }
    return @{@"blocks":blocks,@"present":positions,@"order":mapped};
}

NSDictionary<NSString *,NSData *> *MetalViewBlockSlice(const int32_t *present,const int32_t *order,size_t count,
    size_t stored,NSInteger first,NSInteger end,NSInteger step,NSUInteger block) {
    return MetalViewBlockSelection(present,order,count,stored,block,1,^BOOL(int32_t value,NSUInteger copy,int32_t *output) {
        (void)copy;
        const NSInteger delta=value-first;
        if(!((step>0?(value>=first&&value<end):(value<=first&&value>end))&&delta%step==0))return NO;
        *output=(int32_t)(delta/step);return YES;
    });
}

typedef struct { int32_t value, output; } MetalViewSelection;
static int MetalViewSelectionCompare(const void *a,const void *b) {
    const MetalViewSelection *x=a,*y=b;
    if(x->value!=y->value)return x->value<y->value?-1:1;
    return x->output<y->output?-1:x->output>y->output;
}
NSDictionary<NSString *,NSData *> *MetalViewBlockGather(const int32_t *present,const int32_t *order,size_t count,
    size_t stored,const int32_t *selected,size_t selectedCount,NSUInteger block) {
    if(selectedCount) {
        const NSInteger step=selectedCount>1?(NSInteger)selected[1]-selected[0]:1;
        BOOL affine=step!=0;
        for(size_t i=1;affine&&i<selectedCount;i++)affine=selected[i]==(NSInteger)selected[0]+(NSInteger)i*step;
        if(affine)return MetalViewBlockSlice(present,order,count,stored,selected[0],(NSInteger)selected[selectedCount-1]+step,step,block);
    }
    MetalViewSelection *entries=malloc(selectedCount*sizeof *entries);
    for(size_t i=0;i<selectedCount;i++)entries[i]=(MetalViewSelection){selected[i],(int32_t)i};
    qsort(entries,selectedCount,sizeof *entries,MetalViewSelectionCompare);
    NSUInteger copies=0;
    for(size_t first=0;first<selectedCount;) {
        size_t end=first+1;while(end<selectedCount&&entries[end].value==entries[first].value)end++;
        copies=MAX(copies,end-first);first=end;
    }
    NSDictionary *result=MetalViewBlockSelection(present,order,count,stored,block,copies,^BOOL(int32_t value,NSUInteger copy,int32_t *output) {
        size_t lo=0,hi=selectedCount;
        while(lo<hi){size_t mid=lo+(hi-lo)/2;if(entries[mid].value<value)lo=mid+1;else hi=mid;}
        const size_t at=lo+copy;
        if(!(at<selectedCount&&entries[at].value==value))return NO;
        *output=entries[at].output;return YES;
    });
    free(entries);return result;
}

BOOL MetalHadamardLocal(NSUInteger columns,NSUInteger mix,NSUInteger threads,NSUInteger memory) {
    enum {
#define MESH_METAL_VALUE(name, value) Local_##name = value,
#include "../etc/compute/metal.def"
#undef MESH_METAL_VALUE
    };
    if(!mix || columns%mix)return NO;
    NSUInteger n=columns/mix,t=Local_HADAMARD_THREADS,s=Local_SIMD_WIDTH;
    if(!n || (n&(n-1)) || n<s || n>Local_HADAMARD_VALUES*s || n>t || t%n)return NO;
    return t<=threads && mix<=Local_HADAMARD_MIX && mix<=sizeof(uint32_t)*CHAR_BIT && t/s<=s && mix%(t/n)==0
        && mix<=Local_HADAMARD_BLOCKS*(t/s) && mix/(t/n)<=Local_HADAMARD_MIX_ROWS
        && columns<=Local_HADAMARD_STORES*t && columns<=Local_HADAMARD_STORAGE/2
        && (Local_HADAMARD_STORAGE+s+Local_HADAMARD_MIX)*sizeof(float)<=memory;
}

NSData *MetalButterflyPlan(NSUInteger width,NSUInteger localElements) {
    if(!width || (width&(width-1)) || (width>1 && localElements<2))
        [NSException raise:NSInvalidArgumentException format:@"butterfly needs a power-of-two width and storage for a pair"];
    NSUInteger tile=1;
    while(tile<=localElements/2 && tile<width)tile*=2;
    NSMutableData *plan=[NSMutableData data];
    NSUInteger step=1;
    do {
        struct MetalButterflyStage stage={MIN(tile,width/step),step};
        [plan appendBytes:&stage length:sizeof stage];
        step*=stage.width;
    } while(step<width);
    return plan;
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

static NSUInteger MetalViewAlignment(NSUInteger span,NSInteger distance) {
    NSUInteger d=distance<0?(NSUInteger)(-(distance+1))+1:(NSUInteger)distance;
    while(d){NSUInteger next=span%d;span=d;d=next;}
    return span;
}

static NSString *MetalViewAffineIndexDeclaration(NSUInteger binding,const NSUInteger *shape,const NSInteger *strides,
    size_t rank,NSUInteger elementOffset,NSInteger base) {
    NSString *expression=MetalViewIndexExpression(shape,strides,rank,@"i");
    if(base)expression=[NSString stringWithFormat:@"(%@ + %ldl)",expression,(long)base];
    NSMutableString *source=[NSMutableString string];
    if(![expression isEqualToString:@"long(i)"])
        [source appendFormat:@"template<> inline long mesh_index<%lu>(ulong i) { return %@; }\n",(unsigned long)binding,expression];
    // The aligned contiguous span is measured in this view's elements. A byte
    // view and a scalar view can therefore describe the same vector correctly.
    NSUInteger inner=1,prefix=rank;
    for(size_t d=rank;d--;) {
        if(shape[d]>1&&strides[d]!=(NSInteger)inner)break;
        inner*=shape[d];prefix=d;
    }
    NSUInteger vector=MetalViewAlignment(prefix?inner:0,(NSInteger)elementOffset+base);
    for(size_t d=0;d<prefix;d++)if(shape[d]>1)vector=MetalViewAlignment(vector,strides[d]);
    if(vector)[source appendFormat:@"template<> constexpr ulong mesh_vector_span<%lu>() { return %lul; }\n",(unsigned long)binding,(unsigned long)vector];
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

NSString *MetalViewIndexValuesDeclaration(NSUInteger binding,const NSInteger *values,NSUInteger count) {
    NSData *indices=[NSData dataWithBytes:values length:count*sizeof *values];
    return [[MetalTensorLayout shape:@[@(count)] strides:@[@1] offset:0 maps:@{} indices:indices] positionDeclaration:binding];
}

NSString *MetalViewPositionDeclaration(NSUInteger binding,NSString *indices,NSUInteger count,NSUInteger span) {
    return [NSString stringWithFormat:@"template<uint B> inline long mesh_position(ulong i);\n"
        "template<uint B> constexpr ulong mesh_position_span();\n"
        "template<uint B> constexpr ulong mesh_position_count();\n"
        "namespace mesh_positions_%lu {\n%@%@}\n"
        "template<> inline long mesh_position<%lu>(ulong i) { return mesh_positions_%lu::mesh_index<0>(i); }\n"
        "template<> constexpr ulong mesh_position_span<%lu>() { return %luul; }\n"
        "template<> constexpr ulong mesh_position_count<%lu>() { return %luul; }\n",
        (unsigned long)binding,MetalViewIndexTemplates(),indices,(unsigned long)binding,(unsigned long)binding,
        (unsigned long)binding,(unsigned long)span,(unsigned long)binding,(unsigned long)count];
}

NSString *MetalViewMappedIndexDeclaration(NSUInteger binding,const NSUInteger *shape,const NSInteger *strides,
    size_t rank,NSUInteger elementOffset,NSDictionary<NSNumber *,NSData *> *maps,
    NSArray<NSString *> *coordinates,NSUInteger vectorSpan,NSInteger relativeOffset) {
    if(!coordinates&&!maps.count)return MetalViewAffineIndexDeclaration(binding,shape,strides,rank,elementOffset-relativeOffset,relativeOffset);
    NSDictionary *normalized=MetalViewNormalizeAxes(strides,rank,maps);
    strides=(const NSInteger *)((NSData *)normalized[@"strides"]).bytes;maps=normalized[@"maps"];
    const NSInteger shift=[normalized[@"offset"] integerValue];
    if(!coordinates&&!maps.count)return MetalViewAffineIndexDeclaration(binding,shape,strides,rank,elementOffset-relativeOffset,shift+relativeOffset);
    NSMutableString *source=[NSMutableString string];
    NSMutableArray *terms=[NSMutableArray array];
    if(shift+relativeOffset)[terms addObject:[NSString stringWithFormat:@"%ldl",(long)(shift+relativeOffset)]];
    NSUInteger divisor=1,contiguous=1,vector=0;NSInteger origin=(NSInteger)elementOffset+shift;BOOL inner=YES;
    for(size_t d=rank;d--;) {
        if(!shape[d])return MetalViewIndexDeclaration(binding,shape,strides,rank,elementOffset);
        NSString *coordinate=divisor==1?@"long(i)":[NSString stringWithFormat:@"(long(i) / %lul)",(unsigned long)divisor];
        if(d)coordinate=[NSString stringWithFormat:@"(%@ %% %lul)",coordinate,(unsigned long)shape[d]];
        if(coordinates)coordinate=coordinates[d];
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
            else {inner=NO;vector=MetalViewAlignment(vector,(NSInteger)contiguous);}
            if(!inner) {
                if(values){for(NSUInteger j=0;j<shape[d];j++)vector=MetalViewAlignment(vector,values[j]-base);}
                else vector=MetalViewAlignment(vector,step);
            }
        }
        divisor*=shape[d];
    }
    NSString *expression=terms.count?[terms componentsJoinedByString:@" + "]:@"0";
    [source appendFormat:@"template<> inline long mesh_index<%lu>(ulong i) { return %@; }\n",(unsigned long)binding,expression];
    vector=coordinates?vectorSpan:MetalViewAlignment(vector,origin);
    if(vector)[source appendFormat:@"template<> constexpr ulong mesh_vector_span<%lu>() { return %lul; }\n",(unsigned long)binding,(unsigned long)vector];
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
static void MetalViewUnionSpans(NSMutableData *data) {
    struct MetalViewSpan *out=data.mutableBytes;NSUInteger total=data.length/sizeof *out,used=0;
    qsort(out,total,sizeof *out,MetalViewSpanOrder);
    for(NSUInteger j=0;j<total;j++) {
        if(!out[j].length)continue;
        if(used&&out[j].offset<=out[used-1].offset+out[used-1].length)
            out[used-1].length=MAX(out[used-1].length,out[j].offset+out[j].length-out[used-1].offset);
        else out[used++]=out[j];
    }
    data.length=used*sizeof *out;
}
NSData *MetalTensorSpanReaches(id<MTLBuffer> buffer,NSUInteger base,NSData *spans,
    NSUInteger elementBytes,uint32_t access,int32_t binding) {
    const struct MetalTensorSpan *input=spans.bytes;NSUInteger count=spans.length/sizeof *input;
    NSMutableData *bytes=[NSMutableData dataWithLength:count*sizeof(struct MetalViewSpan)];
    struct MetalViewSpan *out=bytes.mutableBytes;
    for(NSUInteger k=0;k<count;k++)out[k]=(struct MetalViewSpan){base+input[k].offset*elementBytes,input[k].count*elementBytes};
    MetalViewUnionSpans(bytes);
    NSMutableData *result=[NSMutableData data];const struct MetalViewSpan *values=bytes.bytes;
    struct MetalReach reach={buffer,base,0,access|MetalReachBinding,binding};
    for(NSUInteger k=0;k<bytes.length/sizeof *values;k++) {
        reach.offset=values[k].offset;reach.length=values[k].length;[result appendBytes:&reach length:sizeof reach];
    }
    if(!result.length)[result appendBytes:&reach length:sizeof reach];
    return result;
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
        MetalViewUnionSpans(next);spans=next;
    }
    NSMutableData *result=[NSMutableData data];
    const struct MetalViewSpan *values=spans.bytes;
    for(NSUInteger i=0;i<spans.length/sizeof *values;i++) {
        reach.offset=(NSUInteger)base+values[i].offset;reach.length=values[i].length;
        [result appendBytes:&reach length:sizeof reach];
    }
    return result;
}

struct MetalRecordedAddress {
    struct MetalCommandAddress value;
    NSUInteger stride, instances;
    id<MTLBuffer> uniform;
    NSUInteger uniformOffset, uniformLength, uniformStride, uniformInstances;
    NSInteger uniformInvocation;
};
struct MetalRecordedDispatch {
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> buffers[MESH_BUFFER_BINDINGS];
    NSUInteger offsets[MESH_BUFFER_BINDINGS], memory[MESH_MEMORY_BINDINGS],
        strides[MESH_BUFFER_BINDINGS], instances[MESH_BUFFER_BINDINGS], lengths[MESH_BUFFER_BINDINGS];
    NSInteger invocations[MESH_BUFFER_BINDINGS];
    const struct MetalRecordedAddress *addresses;
    NSUInteger addressCount;
    MTLSize grid, group;
    BOOL threads;
    id<MTLBuffer> indirectGrid;
    NSUInteger indirectOffset, indirectStride, indirectInstances;
    NSInteger indirectInvocation;
    NSUInteger level;
    uint64_t access;
    // docs/kernels.md Any program: the reaches the command's description declared, `reaches` of the capture's from
    // `reach`, and the bindings whose extent one of them declares (bit j: binding j)
    uint32_t reach, reaches, extents;
};

@interface MetalCapture : NSObject {
@public
    NSMutableData *dispatches, *reaches;
    NSMutableSet *owners, *resources;
    NSMutableArray *hosts,*addressRecords;
    NSUInteger buffers, memory;
}
@end

static char pipelineSizesKey;
static char replaySegmentsKey,replayRecordsKey,replayHostsKey,replayAddressRecordsKey;

static void MetalReplayDestroy(struct MetalReplay *replay) {
    [replay->commands release]; [replay->residency release]; [replay->owner release];
    *replay = (struct MetalReplay){0};
}

// The bytes of the type each buffer binding of a pipeline's kernel declares, from its reflection, kept with it
// (MetalPipelineDeclaredBytes): the one statement of what each binding must hold.
static void MetalPipelineSizes(id<MTLComputePipelineState> pipeline,MTLComputePipelineReflection *reflection) {
    uint32_t sizes[MESH_BUFFER_BINDINGS] = {0};
    for (id<MTLBinding> binding in reflection.bindings)
        if (binding.type==MTLBindingTypeBuffer && binding.isUsed && binding.index<MESH_BUFFER_BINDINGS)
            sizes[binding.index]=(uint32_t)((id<MTLBufferBinding>)binding).bufferDataSize;
    objc_setAssociatedObject(pipeline,&pipelineSizesKey,[NSData dataWithBytes:sizes length:sizeof sizes],OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

// The bytes of the type a pipeline's kernel declares at buffer `index` (its reflection's bufferDataSize: a `constant T&`
// record's sizeof(T), a `device T*` array's element), and its label (MetalPipelineMake's: its function's name): 0 where its
// pipeline was not made by MetalPipelineMake, or the kernel uses no buffer there.
NSUInteger MetalPipelineDeclaredBytes(id<MTLComputePipelineState> pipeline,NSUInteger index) {
    NSData *sizes=objc_getAssociatedObject(pipeline,&pipelineSizesKey);
    return sizes && index<sizes.length/sizeof(uint32_t)?((const uint32_t *)sizes.bytes)[index]:0;
}
NSString *MetalPipelineName(id<MTLComputePipelineState> pipeline) { return pipeline.label; }

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

typedef void (^MetalAccess)(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger bytes, uint32_t access);
// docs/kernels.md Any program: a command's accesses, each a buffer's bytes [offset, offset + bytes) read or written:
// Each binding has an explicit extent or a union of binding reaches. Slot strides
// relocate those views; neither slot strides nor allocation sizes are accesses.
static void MetalAccesses(const struct MetalRecordedDispatch *command, const struct MetalReach *reaches, NSUInteger invocation, MetalAccess each) {
    if(command->indirectGrid)each(command->indirectGrid,
        MetalInvocationOffset(command->indirectOffset,command->indirectStride,command->indirectInstances,invocation,command->indirectInvocation),
        command->threads?sizeof(MTLDispatchThreadsIndirectArguments):sizeof(MTLDispatchThreadgroupsIndirectArguments),MetalReachRead);
    for(NSUInteger j=0;j<MESH_BUFFER_BINDINGS;j++) {
        id<MTLBuffer> buffer=command->buffers[j];
        const BOOL load=(command->access>>j)&1,store=(command->access>>(MESH_ACCESS_WRITE_SHIFT+j))&1;
        if(!buffer||(!load&&!store)||((command->extents>>j)&1))continue;
        const NSUInteger offset=MetalInvocationOffset(command->offsets[j],command->strides[j],command->instances[j],invocation,command->invocations[j]);
        if(command->lengths[j])each(buffer,offset,command->lengths[j],(load?MetalReachRead:0)|(store?MetalReachWritten:0));
    }
    for(uint32_t k=0;k<command->reaches;k++) {
        const struct MetalReach *reach=reaches+command->reach+k;
        if(!reach->length)continue;
        NSUInteger offset=reach->offset;
        if(reach->access&MetalReachAddress) {
            const struct MetalRecordedAddress *address=command->addresses+reach->binding;
            offset=MetalInvocationOffset(offset,address->stride,address->instances,invocation,address->value.invocation);
        } else if(reach->access&MetalReachBinding)
            offset=MetalInvocationOffset(offset,command->strides[reach->binding],command->instances[reach->binding],invocation,command->invocations[reach->binding]);
        each(reach->buffer,offset,reach->length,reach->access);
    }
}

// Every bridge lease and composed view names pages in the process's one region.
// The virtual address is a binding; physical page identity orders its users.
static char MetalRegionPageIdentity;
static BOOL MetalPageAccesses(NSUInteger address,NSUInteger length,uint32_t access,
    void (^each)(id,NSUInteger,NSUInteger,uint32_t)) {
    ncclMeshPageTable_t table=NULL;
    ncclResult_t status=ncclMeshPageTableCreate((const void *)address,NULL,0,&table);
    if(status==ncclInvalidArgument)return NO; // Ordinary nonbridge storage.
    if(status)[NSException raise:NSMallocException format:@"dependency page table: %s",ncclGetLastError(NULL)];
    size_t pages=0,pageBytes=0,count=0;const void *base=NULL;
    status=ncclMeshPageTableShape(table,&pages,&pageBytes);
    if(!status)status=ncclMeshPageTableAddress(table,&base);
    if(!status)status=ncclMeshPageTableBacking(table,NULL,&count);
    NSMutableData *storage=status?nil:[NSMutableData dataWithLength:count*sizeof(ncclMeshPageBacking)];
    if(!status)status=ncclMeshPageTableBacking(table,storage.mutableBytes,&count);
    ncclMeshPageTableDestroy(table);
    if(status)[NSException raise:NSMallocException format:@"dependency page backing: %s",ncclGetLastError(NULL)];
    const NSUInteger first=address-(NSUInteger)base,last=first+length;
    const ncclMeshPageBacking *runs=storage.bytes;
    NSValue *key=[NSValue valueWithPointer:&MetalRegionPageIdentity];
    for(size_t r=0;r<count;r++) {
        const NSUInteger begin=runs[r].first*pageBytes,end=(runs[r].first+runs[r].count)*pageBytes;
        const NSUInteger lo=MAX(first,begin),hi=MIN(last,end);
        if(lo<hi){const NSUInteger physical=runs[r].page*pageBytes+lo-begin;each(key,physical,physical+hi-lo,access);}
    }
    return YES;
}

// Canonical storage identity and ranges for every dependency lowering.
static void MetalViewAccesses(NSData *accesses,void (^each)(id,NSUInteger,NSUInteger,uint32_t)) {
    const struct MetalReach *views=accesses.bytes;
    for(NSUInteger i=0;i<accesses.length/sizeof *views;i++) {
        const struct MetalReach v=views[i];if(!v.length||!(v.access&3))continue;
        const NSUInteger base=MetalAllocationBase(v.buffer),first=base+v.offset;
        if((!v.buffer||v.buffer.storageMode==MTLStorageModeShared)&&MetalPageAccesses(first,v.length,v.access,each))continue;
        each(MetalAllocationKey(v.buffer),first,first+v.length,v.access);
    }
}

struct MetalSpan { NSUInteger lo, hi, node; };
// A key's frontier: its spans clipped into buckets of 2^MESH_FRONTIER_BITS bytes (a dictionary of bucket index to pieces),
// so adding a span appends to the buckets it covers, and a lookup or an erase reads only the buckets [lo, hi) covers. A
// span meets [lo, hi) exactly when one of its pieces does. A segment of many tiles in any order then costs per access
// its buckets' pieces, not every span (cp's recording, 2026-10-29: MetalOrder's frontier, profiled four times).
#define MESH_FRONTIER_BITS 20
static void MetalSpanAdd(NSMutableDictionary *spans,id key,NSUInteger lo,NSUInteger hi,NSUInteger node) {
    if(lo>=hi)return;
    NSMutableDictionary *buckets=spans[key]?:(spans[key]=[NSMutableDictionary dictionary]);
    for(NSUInteger b=lo>>MESH_FRONTIER_BITS;b<=(hi-1)>>MESH_FRONTIER_BITS;b++) {
        NSNumber *at=@(b);NSMutableData *pieces=buckets[at]?:(buckets[at]=[NSMutableData data]);
        const struct MetalSpan piece={MAX(lo,b<<MESH_FRONTIER_BITS),MIN(hi,(b+1)<<MESH_FRONTIER_BITS),node};
        [pieces appendBytes:&piece length:sizeof piece];
    }
}

// A new writer replaces only the covered portion of the previous frontier.
static void MetalSpanErase(NSMutableDictionary *spans,id key,NSUInteger lo,NSUInteger hi) {
    NSMutableDictionary *buckets=spans[key];
    if(!buckets||lo>=hi)return;
    for(NSUInteger b=lo>>MESH_FRONTIER_BITS;b<=(hi-1)>>MESH_FRONTIER_BITS;b++) {
        NSMutableData *data=buckets[@(b)];
        const NSUInteger count=data.length/sizeof(struct MetalSpan);if(!count)continue;NSUInteger kept=0;
        NSMutableData *tails=nil;struct MetalSpan *pieces=data.mutableBytes;
        for(NSUInteger i=0;i<count;i++) {
            struct MetalSpan piece=pieces[i];
            if(piece.lo>=hi||piece.hi<=lo){pieces[kept++]=piece;continue;}
            if(piece.hi>hi){
                const struct MetalSpan tail={hi,piece.hi,piece.node};
                if(!tails)tails=[[NSMutableData alloc] init];
                [tails appendBytes:&tail length:sizeof tail];
            }
            if(piece.lo<lo){piece.hi=lo;pieces[kept++]=piece;}
        }
        data.length=kept*sizeof *pieces;
        if(tails){[data appendData:tails];[tails release];}
    }
}

static NSIndexSet *MetalViewPrior(NSMutableDictionary *read,NSMutableDictionary *written,
    NSMutableDictionary *atomic,NSData *views) {
    NSMutableIndexSet *prior=[NSMutableIndexSet indexSet];
    MetalViewAccesses(views,^(id key,NSUInteger lo,NSUInteger hi,uint32_t access){
        if((access&MetalReachNext)&&!(access&MetalReachWritten))return;
        NSMutableDictionary *const sets[3]={written,read,atomic};
        for(int s=0;s<3;s++) {
            const BOOL observes=(access&MetalReachObserve)&&!(access&MetalReachWritten);
            if((s==1&&!(access&MetalReachWritten))||(s==2&&((access&MetalReachAtomic)||observes)))continue;
            NSMutableDictionary *buckets=sets[s][key];if(!buckets||lo>=hi)continue;
            for(NSUInteger b=lo>>MESH_FRONTIER_BITS;b<=(hi-1)>>MESH_FRONTIER_BITS;b++) {
                NSData *data=buckets[@(b)];const struct MetalSpan *pieces=data.bytes;const NSUInteger n=data.length/sizeof *pieces;
                for(NSUInteger i=0;i<n;i++)if(pieces[i].lo<hi&&lo<pieces[i].hi)[prior addIndex:pieces[i].node];
            }
        }
    });
    return prior;
}

static void MetalViewDefine(NSMutableDictionary *read,NSMutableDictionary *written,
    NSMutableDictionary *atomic,NSData *views,NSUInteger node) {
    MetalViewAccesses(views,^(id key,NSUInteger lo,NSUInteger hi,uint32_t access){
        if((access&MetalReachWritten)&&!(access&MetalReachAtomic)) {
            MetalSpanErase(read,key,lo,hi);MetalSpanErase(written,key,lo,hi);MetalSpanErase(atomic,key,lo,hi);
        }
    });
    MetalViewAccesses(views,^(id key,NSUInteger lo,NSUInteger hi,uint32_t access){
        if(access&MetalReachWritten)MetalSpanAdd(access&MetalReachAtomic?atomic:written,key,lo,hi,node);
        else if((access&MetalReachRead)&&!(access&(MetalReachObserve|MetalReachNext)))MetalSpanAdd(read,key,lo,hi,node);
    });
}

// Bind forward reads by byte intersection, not by declaration order. A later
// writer can supply part of a view; the remaining spans await their own writer.
static void MetalViewStep(NSMutableDictionary *read,NSMutableDictionary *written,NSMutableDictionary *atomic,
    NSMutableDictionary *next,NSMutableDictionary *observed,NSData *views,NSUInteger node,
    void (^edge)(NSUInteger,NSUInteger)) {
    [MetalViewPrior(read,written,atomic,views) enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){(void)stop;edge(node,p);}];
    NSMutableArray *resolved=[NSMutableArray array];
    MetalViewAccesses(views,^(id key,NSUInteger lo,NSUInteger hi,uint32_t access){
        if(!(access&MetalReachWritten))return;
        NSMutableDictionary *const pending[2]={next,observed};
        for(NSUInteger s=0;s<2;s++) {
            NSDictionary *buckets=pending[s][key];
            for(NSUInteger b=lo>>MESH_FRONTIER_BITS;b<=(hi-1)>>MESH_FRONTIER_BITS;b++) {
                NSData *data=buckets[@(b)];const struct MetalSpan *pieces=data.bytes;
                for(NSUInteger i=0;i<data.length/sizeof *pieces;i++)if(pieces[i].lo<hi&&lo<pieces[i].hi) {
                    edge(pieces[i].node,node);
                    if(!s)[resolved addObject:@[key,@(MAX(lo,pieces[i].lo)),@(MIN(hi,pieces[i].hi)),@(pieces[i].node)]];
                }
            }
            MetalSpanErase(pending[s],key,lo,hi);
        }
    });
    MetalViewDefine(read,written,atomic,views,node);
    for(NSArray *span in resolved)MetalSpanAdd(read,span[0],[span[1] unsignedIntegerValue],
        [span[2] unsignedIntegerValue],[span[3] unsignedIntegerValue]);
    MetalViewAccesses(views,^(id key,NSUInteger lo,NSUInteger hi,uint32_t access){
        if((access&(MetalReachRead|MetalReachNext))==(MetalReachRead|MetalReachNext))
            MetalSpanAdd(access&MetalReachObserve?observed:next,key,lo,hi,node);
    });
}

static BOOL MetalViewPending(NSDictionary *pending) {
    for(NSDictionary *buckets in pending.allValues)for(NSData *spans in buckets.allValues)if(spans.length)return YES;
    return NO;
}

static NSData *MetalViewQualify(NSData *views,NSMutableDictionary *domain) {
    NSMutableData *qualified=[NSMutableData data];const struct MetalReach *reaches=views.bytes;
    for(NSUInteger r=0;r<views.length/sizeof *reaches;r++) {
        const struct MetalReach original=reaches[r];
        if(!(original.access&MetalReachRead)||!original.length) {
            [qualified appendBytes:&original length:sizeof original];continue;
        }
        __block NSUInteger offset=original.offset;
        MetalViewAccesses([NSData dataWithBytes:&original length:sizeof original],^(id key,NSUInteger lo,NSUInteger hi,uint32_t access){
            NSMutableIndexSet *cuts=[NSMutableIndexSet indexSet];[cuts addIndex:lo];[cuts addIndex:hi];
            NSDictionary *buckets=domain[key];
            for(NSUInteger b=lo>>MESH_FRONTIER_BITS;b<=(hi-1)>>MESH_FRONTIER_BITS;b++) {
                NSData *data=buckets[@(b)];const struct MetalSpan *pieces=data.bytes;
                for(NSUInteger i=0;i<data.length/sizeof *pieces;i++)if(pieces[i].lo<hi&&lo<pieces[i].hi) {
                    [cuts addIndex:MAX(lo,pieces[i].lo)];[cuts addIndex:MIN(hi,pieces[i].hi)];
                }
            }
            NSUInteger from=cuts.firstIndex;
            for(NSUInteger to=[cuts indexGreaterThanIndex:from];to!=NSNotFound;to=[cuts indexGreaterThanIndex:to]) {
                NSData *data=buckets[@(from>>MESH_FRONTIER_BITS)];const struct MetalSpan *pieces=data.bytes;
                BOOL next=NO;for(NSUInteger i=0;i<data.length/sizeof *pieces;i++)next|=pieces[i].lo<=from&&from<pieces[i].hi;
                const struct MetalReach part={original.buffer,offset+from-lo,to-from,access|(next?MetalReachNext:0),original.binding};
                [qualified appendBytes:&part length:sizeof part];from=to;
            }
            offset+=hi-lo;
        });
    }
    MetalViewAccesses(views,^(id key,NSUInteger lo,NSUInteger hi,uint32_t access){
        if(access&MetalReachWritten)MetalSpanErase(domain,key,lo,hi);
    });
    return qualified;
}

NSArray<NSIndexSet *> *MetalViewDependencies(NSArray<NSData *> *accesses) {
    NSMutableDictionary *read=[NSMutableDictionary dictionary],*written=[NSMutableDictionary dictionary];
    NSMutableDictionary *atomic=[NSMutableDictionary dictionary];
    NSMutableDictionary *next=[NSMutableDictionary dictionary],*observed=[NSMutableDictionary dictionary];
    NSMutableArray *dependencies=[NSMutableArray arrayWithCapacity:accesses.count];
    for(NSUInteger i=0;i<accesses.count;i++)[dependencies addObject:[NSMutableIndexSet indexSet]];
    for(NSUInteger i=0;i<accesses.count;i++) @autoreleasepool {
        MetalViewStep(read,written,atomic,next,observed,accesses[i],i,^(NSUInteger consumer,NSUInteger producer){
            [dependencies[consumer] addIndex:producer];
        });
    }
    return dependencies;
}

static NSData *MetalCommandReaches(const struct MetalRecordedDispatch *command,const struct MetalReach *reaches,NSUInteger invocation) {
    NSMutableData *views=[NSMutableData data];
    MetalAccesses(command,reaches,invocation,^(id<MTLBuffer> buffer,NSUInteger offset,NSUInteger bytes,uint32_t access){
        const struct MetalReach view={buffer,offset,bytes,access&~(MetalReachBinding|MetalReachAddress),-1};
        [views appendBytes:&view length:sizeof view];
    });
    return views;
}

// The queue retains the last producer/reader of each declared page range.
// These are dependency operands, not a union requiring a whole-queue barrier.
// Snapshots retain fences/events only; they never keep numerical pages alive.
@interface MetalQueueAccess : NSObject {
@public NSMutableDictionary *read, *written, *atomic, *resources;
    NSArray *producers;
}
@end
@implementation MetalQueueAccess
- (instancetype)init {
    self=[super init];
    read=[NSMutableDictionary new];written=[NSMutableDictionary new];atomic=[NSMutableDictionary new];
    resources=[NSMutableDictionary new];producers=[NSArray new];return self;
}
- (void)dealloc { [read release];[written release];[atomic release];[resources release];[producers release];[super dealloc]; }
@end
static char queueAccessKey,encoderScheduleKey;

static void MetalDataflow(MetalCapture *capture) {
    struct MetalRecordedDispatch *commands=capture->dispatches.mutableBytes;
    const NSUInteger count=capture->dispatches.length/sizeof(*commands);
    NSMutableArray *views=[NSMutableArray arrayWithCapacity:count];
    for(NSUInteger i=0;i<count;i++)[views addObject:MetalCommandReaches(commands+i,capture->reaches.bytes,0)];
    NSArray *dependencies=MetalViewDependencies(views);
    NSMutableArray *successors=[NSMutableArray arrayWithCapacity:count];
    NSMutableData *counts=[NSMutableData dataWithLength:count*sizeof(NSUInteger)];NSUInteger *left=counts.mutableBytes;
    NSMutableIndexSet *ready=[NSMutableIndexSet indexSet];
    for(NSUInteger i=0;i<count;i++){[successors addObject:[NSMutableIndexSet indexSet]];commands[i].level=0;}
    for(NSUInteger i=0;i<count;i++) {
        left[i]=[dependencies[i] count];if(!left[i])[ready addIndex:i];
        [dependencies[i] enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){(void)stop;[successors[p] addIndex:i];}];
    }
    while(ready.count) {
        const NSUInteger i=ready.firstIndex;[ready removeIndex:i];
        [successors[i] enumerateIndexesUsingBlock:^(NSUInteger next,BOOL *stop){
            (void)stop;commands[next].level=MAX(commands[next].level,commands[i].level+1);
            if(!--left[next])[ready addIndex:next];
        }];
    }
}

@implementation MetalCapture
- (instancetype)init {
    self = [super init];
    dispatches = [NSMutableData new];
    reaches = [NSMutableData new];
    owners = [NSMutableSet new];
    resources = [NSMutableSet new];
    hosts = [NSMutableArray new];addressRecords=[NSMutableArray new];
    return self;
}
- (void)dealloc {
    [dispatches release]; [reaches release]; [owners release]; [resources release]; [hosts release]; [addressRecords release]; [super dealloc];
}
@end

// A shared buffer over a region lease of its own (ncclMemAlloc, nccl.h Memory), retained: no copy, the lease returned
// (ncclMemFree) when the buffer is deallocated, after every command that bound it.
id<MTLBuffer> MetalLeaseBuffer(id<MTLDevice> device,NSUInteger length) {
    const NSUInteger page=(NSUInteger)getpagesize(),bytes=(MAX(length,(NSUInteger)1)+page-1)/page*page;
    void *memory=NULL;
    if(ncclMemAlloc(&memory,bytes))
        [NSException raise:NSMallocException format:@"a region lease of %lu bytes: %s",(unsigned long)bytes,ncclGetLastError(NULL)];
    id<MTLBuffer> buffer=[device newBufferWithBytesNoCopy:memory length:bytes options:MTLResourceStorageModeShared
        deallocator:^(void *pointer,NSUInteger leased){(void)leased;ncclMemFree(pointer);}];
    if(!buffer) {
        ncclMemFree(memory);
        [NSException raise:NSMallocException format:@"a Metal buffer over a region lease of %lu bytes",(unsigned long)bytes];
    }
    return buffer;
}

// Embedded addresses are data in ordinary prepared uniform pages. Their exact
// target views retain their own invocation relation for dependency ordering.
static void MetalCaptureAddresses(MetalCapture *capture,id<MTLDevice> device,NSUInteger repetitions) {
    struct MetalRecordedDispatch *commands=capture->dispatches.mutableBytes;
    struct MetalReach *reaches=capture->reaches.mutableBytes;
    for(NSUInteger c=0;c<capture->dispatches.length/sizeof *commands;c++) {
        struct MetalRecordedDispatch *command=commands+c;
        for(NSUInteger slot=0;slot<MESH_BUFFER_BINDINGS;slot++) {
            const struct MetalRecordedAddress *uniform=NULL;
            for(NSUInteger a=0;a<command->addressCount;a++)if(command->addresses[a].value.binding==slot) {uniform=command->addresses+a;break;}
            if(!uniform)continue;
            const NSUInteger alignment=_Alignof(uint64_t),stride=(uniform->uniformLength+alignment-1)/alignment*alignment;
            id<MTLBuffer> table=MetalLeaseBuffer(device,stride*repetitions);
            for(NSUInteger t=0;t<repetitions;t++) {
                char *destination=(char *)table.contents+t*stride;
                const NSUInteger from=MetalInvocationOffset(uniform->uniformOffset,uniform->uniformStride,uniform->uniformInstances,t,uniform->uniformInvocation);
                memcpy(destination,(const char *)uniform->uniform.contents+from,uniform->uniformLength);
                for(NSUInteger a=0;a<command->addressCount;a++) {
                    const struct MetalRecordedAddress *address=command->addresses+a;
                    if(address->value.binding!=slot)continue;
                    const uint64_t pointer=address->value.buffer.gpuAddress+
                        MetalInvocationOffset(address->value.origin,address->stride,address->instances,t,address->value.invocation);
                    memcpy(destination+address->value.offset,&pointer,sizeof pointer);
                }
            }
            // The same declared ranges now refer to their prepared table slot.
            for(NSUInteger r=0;r<command->reaches;r++) {
                struct MetalReach *reach=reaches+command->reach+r;
                if((reach->access&MetalReachBinding)&&!(reach->access&MetalReachAddress)&&reach->binding==(int32_t)slot) {
                    reach->buffer=table;reach->offset-=command->offsets[slot];
                }
            }
            command->buffers[slot]=table;command->offsets[slot]=0;command->lengths[slot]=uniform->uniformLength;
            command->strides[slot]=stride;command->instances[slot]=repetitions;command->invocations[slot]=0;
            [capture->resources addObject:table];[table release];
        }
    }
}

// ../dox/mesh/design/prepared-machine.md#M06
// A recording's replay: indirect commands holding every repetition's operands
// (the one modulus), plus residency. The function graph owns their ordering.
static void MetalReplayBuild(MetalCapture *capture,id<MTLDevice> device,NSUInteger repetitions,enum MetalForm form,
    struct MetalReplay *result) {
    result->count = (uint32_t)(capture->dispatches.length / sizeof(struct MetalRecordedDispatch));
    MetalCaptureAddresses(capture,device,repetitions);
    [capture->owners unionSet:capture->resources];
    result->buffers = (uint16_t)capture->buffers; result->memory = (uint16_t)capture->memory;
    const struct MetalRecordedDispatch *commands = capture->dispatches.bytes;
    result->owner = [capture->owners retain];
    objc_setAssociatedObject(result->owner,&replayAddressRecordsKey,[NSArray arrayWithArray:capture->addressRecords],OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    objc_setAssociatedObject(result->owner,&replayHostsKey,[NSArray arrayWithArray:capture->hosts],OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    // docs/kernels.md E2: the records a direct replay encodes (MetalReplayDispatch4)
    objc_setAssociatedObject(result->owner, &replayRecordsKey, @[[NSData dataWithBytes:commands length:capture->dispatches.length],[NSData data],@(form),[NSData dataWithData:capture->reaches],@(repetitions)],
        OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    MTLResidencySetDescriptor *residency = [MTLResidencySetDescriptor new];
    residency.initialCapacity = capture->owners.count + 1;
    NSError *error = nil;
    result->residency = [device newResidencySetWithDescriptor:residency error:&error];
    [residency release];
    if (!result->residency) [NSException raise:NSMallocException format:@"%@", error];
    for (id allocation in capture->owners) [result->residency addAllocation:allocation];
    BOOL fixed=NO;
    for(uint32_t i=0;i<result->count;i++)fixed|=!commands[i].indirectGrid;
    if (form != MetalFormIndirect || !fixed || !result->count || !repetitions) {
        [result->residency commit]; [result->residency requestResidency]; return;
    }
    MTLIndirectCommandBufferDescriptor *descriptor = [MTLIndirectCommandBufferDescriptor new];
    descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch | MTLIndirectCommandTypeConcurrentDispatchThreads;
    descriptor.inheritBuffers = NO; descriptor.inheritPipelineState = NO;
    descriptor.maxKernelBufferBindCount = capture->buffers;
    descriptor.maxKernelThreadgroupMemoryBindCount = capture->memory;
    // Storage capacity divides the flat command list; it never limits one
    // invocation or adds a dependency at a storage-segment boundary.
    const NSUInteger ceiling=MESH_INDIRECT_COMMANDS,total=(NSUInteger)result->count*repetitions;
    NSMutableArray *segments = [NSMutableArray array];
    // Metal makes an indirect command buffer only from the device, in storage it allocates itself: no call places one
    // over a region lease, so these segments are the replay's memory outside the region.
    for (NSUInteger first=0;first<total;first+=ceiling) {
        id<MTLIndirectCommandBuffer> segment = [device newIndirectCommandBufferWithDescriptor:descriptor
            maxCommandCount:MIN(ceiling,total-first) options:0];
        if (!segment) [NSException raise:NSMallocException format:@"indirect command buffer of %lu commands",
            (unsigned long)MIN(ceiling,total-first)];
        [segments addObject:segment]; [segment release];
    }
    result->commands = [segments[0] retain];
    objc_setAssociatedObject(result->commands, &replaySegmentsKey, @[segments, @(ceiling)], OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [descriptor release];
    for (NSUInteger t = 0; t < repetitions; t++) for (uint32_t i = 0; i < result->count; ++i) {
        const struct MetalRecordedDispatch *source = commands + i;
        // ICB compute records have fixed grids. A page-valued grid is emitted
        // by the enclosing Metal4 encoder's indirect-dispatch instruction.
        if(source->indirectGrid)continue;
        if (!source->pipeline.supportIndirectCommandBuffers)
            [NSException raise:NSInvalidArgumentException format:@"command %u (%@) does not support indirect commands: the encoded form runs it (docs/kernels.md Any program)",
                i, source->pipeline.label ?: @"unnamed"];
        const NSUInteger index=t*result->count+i;
        id<MTLIndirectComputeCommand> command = [segments[index/ceiling] indirectComputeCommandAtIndex:index%ceiling];
        [command setComputePipelineState:source->pipeline];
        // ../dox/mesh/design/prepared-machine.md#M13: the one modulus.  Invocation t binds slot
        // (t % instances), and instances == repetitions is a bare t.
        for (NSUInteger j = 0; j < capture->buffers; ++j)
            if (((source->access | (source->access >> MESH_ACCESS_WRITE_SHIFT)) >> j) & 1) {
                [command setKernelBuffer:source->buffers[j]
                    offset:MetalInvocationOffset(source->offsets[j],source->strides[j],source->instances[j],t,source->invocations[j]) atIndex:j];
            }
        for (NSUInteger j = 0; j < capture->memory; ++j)
            if (source->memory[j]) [command setThreadgroupMemoryLength:source->memory[j] atIndex:j];
        // Dependencies belong to the enclosing function's command graph.
        // An ICB command never fences unrelated commands in its storage order.
        if (source->threads) [command concurrentDispatchThreads:source->grid threadsPerThreadgroup:source->group];
        else [command concurrentDispatchThreadgroups:source->grid threadsPerThreadgroup:source->group];
    }
    for (id<MTLIndirectCommandBuffer> segment in segments) [result->residency addAllocation:segment];
    [result->residency commit];
    [result->residency requestResidency];
}

// docs/kernels.md Any program: one recorded command's line: its dependency depth, entry,
// grid and group (`t` a thread grid), threadgroup memory, the bytes of its accesses (MetalAccesses:
// its bindings, then its reaches), and the named operands.
static NSString *MetalCommandLine(const struct MetalRecordedDispatch *command,const struct MetalReach *reaches,NSUInteger i,NSUInteger level,
    NSMutableDictionary *allocations) {
    NSUInteger memory=command->pipeline.staticThreadgroupMemoryLength;
    __block NSUInteger read=0,written=0;
    for(NSUInteger j=0;j<MESH_MEMORY_BINDINGS;j++)memory+=command->memory[j];
    NSMutableString *names=[NSMutableString string],*bindings=[NSMutableString string];
    MetalAccesses(command,reaches,0,^(id<MTLBuffer> buffer,NSUInteger offset,NSUInteger bytes,uint32_t access) {
        const BOOL load=(access&MetalReachRead)!=0,store=(access&MetalReachWritten)!=0;
        if(load)read+=bytes;
        if(store)written+=bytes;
        if(buffer.label.length)[names appendFormat:@" %@:%s%s",buffer.label,load?"r":"-",store?"w":"-"];
        NSValue *key=MetalAllocationKey(buffer);
        if(!allocations[key])allocations[key]=@(allocations.count);
        [bindings appendFormat:@" @%lu+%lu:%lu:%s%s",[allocations[key] unsignedLongValue],(unsigned long)(MetalAllocationBase(buffer)+offset),
            (unsigned long)bytes,load?"r":"-",store?"w":"-"];
    });
    return [NSString stringWithFormat:@"command %lu level %lu %s grid %lu,%lu,%lu%s group %lu,%lu,%lu memory %lu read %lu written %lu%s%s%s",
        i,level,MetalPipelineName(command->pipeline).UTF8String?:"?",command->grid.width,command->grid.height,command->grid.depth,
        command->threads?"t":"",command->group.width,command->group.height,command->group.depth,memory,read,written,
        "",bindings.UTF8String,names.UTF8String];
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
        [table setAddress:source->buffers[j].gpuAddress+MetalInvocationOffset(source->offsets[j],source->strides[j],source->instances[j],t,source->invocations[j]) atIndex:j];
    }
    for(NSUInteger j=0;j<replay->memory;j++)if(source->memory[j])[encoder setThreadgroupMemoryLength:source->memory[j] atIndex:j];
    if(source->indirectGrid) {
        const MTLGPUAddress grid=source->indirectGrid.gpuAddress+
            MetalInvocationOffset(source->indirectOffset,source->indirectStride,source->indirectInstances,t,source->indirectInvocation);
        if(source->threads)[encoder dispatchThreadsWithIndirectBuffer:grid];
        else [encoder dispatchThreadgroupsWithIndirectBuffer:grid threadsPerThreadgroup:source->group];
    } else if(source->threads)[encoder dispatchThreads:source->grid threadsPerThreadgroup:source->group];
    else [encoder dispatchThreadgroups:source->grid threadsPerThreadgroup:source->group];
}

// docs/kernels.md Any program: each description a recorded command, its pipeline an owner and the buffers it names the
// capture's resources
static void MetalCaptureAppend(MetalCapture *capture,const struct MetalCommandDescription *commands,size_t count) {
    for(size_t i=0;i<count;i++) {
        const struct MetalCommandDescription *description=commands+i;
        struct MetalRecordedDispatch command={.pipeline=description->pipeline,.grid=description->groups,.group=description->threads,.threads=description->threadGrid,
            .indirectGrid=description->indirectGrid,.indirectOffset=description->indirectOffset,.indirectInvocation=description->indirectInvocation,
            .reach=(uint32_t)(capture->reaches.length/sizeof(struct MetalReach)),.reaches=(uint32_t)description->reachCount};
        [capture->owners addObject:command.pipeline];
        if(command.indirectGrid)[capture->resources addObject:command.indirectGrid];
        for(size_t k=0;k<description->bindingCount;k++) {
            const struct MetalCommandBinding binding=description->bindings[k];
            if(binding.index>=MESH_BUFFER_BINDINGS)[NSException raise:NSInvalidArgumentException format:@"binding index exceeds Metal capacity"];
            const uint32_t at=binding.index;
            command.buffers[at]=binding.buffer;command.offsets[at]=binding.offset;command.lengths[at]=binding.length;command.invocations[at]=binding.invocation;
            command.access|=(uint64_t)(binding.access&MetalReachRead?1:0)<<at;
            command.access|=(uint64_t)(binding.access&MetalReachWritten?1:0)<<(MESH_ACCESS_WRITE_SHIFT+at);
            capture->buffers=MAX(capture->buffers,at+1);
            [capture->resources addObject:binding.buffer];
        }
        if(description->addressCount) {
            NSMutableData *addresses=[NSMutableData dataWithLength:description->addressCount*sizeof(struct MetalRecordedAddress)];
            struct MetalRecordedAddress *records=addresses.mutableBytes;
            for(NSUInteger a=0;a<description->addressCount;a++) {
                records[a].value=description->addresses[a];
                const NSUInteger slot=records[a].value.binding;
                records[a].uniform=command.buffers[slot];records[a].uniformOffset=command.offsets[slot];
                records[a].uniformLength=command.lengths[slot];
                [capture->resources addObject:records[a].value.buffer];[capture->owners addObject:records[a].uniform];
            }
            command.addresses=addresses.bytes;command.addressCount=description->addressCount;
            [capture->addressRecords addObject:addresses];
        }
        if(description->memoryCount>MESH_MEMORY_BINDINGS)[NSException raise:NSInvalidArgumentException format:@"threadgroup-memory index exceeds Metal capacity"];
        for(size_t k=0;k<description->memoryCount;k++)command.memory[k]=description->memory[k];
        capture->memory=MAX(capture->memory,description->memoryCount);
        [capture->reaches appendBytes:description->reaches length:description->reachCount*sizeof(struct MetalReach)];
        for(size_t k=0;k<description->reachCount;k++) {
            const struct MetalReach reach=description->reaches[k];
            [capture->resources addObject:reach.buffer];
            if(reach.access&MetalReachAddress) {
                if(reach.binding<0||(NSUInteger)reach.binding>=command.addressCount||command.addresses[reach.binding].value.buffer!=reach.buffer)
                    [NSException raise:NSInvalidArgumentException format:@"command reach names no matching address operand"];
            } else if(reach.access&MetalReachBinding) {
                if(reach.binding<0||reach.binding>=MESH_BUFFER_BINDINGS||command.buffers[reach.binding]!=reach.buffer)
                    [NSException raise:NSInvalidArgumentException format:@"command reach names no matching buffer binding"];
                command.extents|=UINT32_C(1)<<reach.binding;
            }
        }
        [capture->dispatches appendBytes:&command length:sizeof command];
    }
}

static MetalCapture *MetalCaptureDescribed(const struct MetalCommandDescription *commands,size_t count) {
    MetalCapture *capture=[[MetalCapture new] autorelease];
    MetalCaptureAppend(capture,commands,count);return capture;
}

// metal-recording.h MetalOperand: the operand a buffer is, the last one declared of it
static const struct MetalOperand *MetalOperandOf(const struct MetalOperand *operands,size_t count,id<MTLBuffer> buffer) {
    for(size_t k=count;buffer&&k--;)if(operands[k].buffer==buffer)return operands+k;
    return NULL;
}

// ../dox/mesh/design/prepared-machine.md#M13: the operands bound in the capture's commands. A binding of an operand takes
// its stride and slots; every binding keeps its compiler-supplied origin. Its mode is its reaches' where it has any. A
// reach with neither access nor residency, past its buffer or, not a binding's, into an operand's slots (which an address cannot follow), and a
// binding's extent past its buffer, are refused.
static void MetalCaptureOperands(MetalCapture *capture,const struct MetalOperand *operands,size_t count) {
    struct MetalRecordedDispatch *commands=capture->dispatches.mutableBytes;
    struct MetalReach *reaches=capture->reaches.mutableBytes;
    for(NSUInteger i=0;i<capture->dispatches.length/sizeof *commands;i++) {
        struct MetalRecordedDispatch *command=commands+i;
        const struct MetalOperand *grid=MetalOperandOf(operands,count,command->indirectGrid);
        command->indirectStride=grid?grid->stride:0;command->indirectInstances=grid?grid->instances:0;
        NSString *name=MetalPipelineName(command->pipeline)?:@"unnamed";
        uint64_t access=command->access;uint32_t moded=0;
        struct MetalRecordedAddress *addresses=(struct MetalRecordedAddress *)command->addresses;
        for(NSUInteger a=0;a<command->addressCount;a++) {
            const struct MetalOperand *operand=MetalOperandOf(operands,count,addresses[a].value.buffer);
            addresses[a].stride=operand?operand->stride:0;addresses[a].instances=operand?operand->instances:0;
            const struct MetalOperand *uniform=MetalOperandOf(operands,count,addresses[a].uniform);
            addresses[a].uniformStride=uniform?uniform->stride:0;addresses[a].uniformInstances=uniform?uniform->instances:0;
            addresses[a].uniformInvocation=command->invocations[addresses[a].value.binding];
        }
        for(uint32_t k=0;k<command->reaches;k++) {
            const struct MetalReach *reach=reaches+command->reach+k;
            const struct MetalOperand *operand=MetalOperandOf(operands,count,reach->buffer);
            const NSUInteger slots=operand?operand->instances:0,stride=operand?operand->stride:0;
            if(!(reach->access&(MetalReachRead|MetalReachWritten|MetalReachResident))||reach->offset+reach->length>reach->buffer.length||
               (stride&&slots!=1&&!(reach->access&(MetalReachBinding|MetalReachAddress))))
                [NSException raise:NSInvalidArgumentException format:@"command %lu (%@): reach %u (%@, %lu bytes at %lu of %lu, access %u) is "
                    "no access or residency, past its buffer, or into an operand's slots (%lu), which an address cannot follow (docs/kernels.md Any program)",
                    (unsigned long)i,name,k,reach->buffer.label?:@"unnamed",(unsigned long)reach->length,(unsigned long)reach->offset,
                    (unsigned long)reach->buffer.length,reach->access,(unsigned long)slots];
            if((reach->access&MetalReachAddress)||!(reach->access&MetalReachBinding))continue;
            const uint32_t bit=UINT32_C(1)<<reach->binding;
            if(!(moded&bit))access&=~((UINT64_C(1)<<reach->binding)|(UINT64_C(1)<<(MESH_ACCESS_WRITE_SHIFT+reach->binding)));
            if(reach->access&MetalReachRead)access|=UINT64_C(1)<<reach->binding;
            if(reach->access&MetalReachWritten)access|=UINT64_C(1)<<(MESH_ACCESS_WRITE_SHIFT+reach->binding);
            moded|=bit;
        }
        command->access=access;
        for(NSUInteger j=0;j<MESH_BUFFER_BINDINGS;j++) {
            id<MTLBuffer> buffer=command->buffers[j];
            if(buffer&&(((access|(access>>MESH_ACCESS_WRITE_SHIFT))>>j)&1)&&!((command->extents>>j)&1)&&
               command->offsets[j]+command->lengths[j]>buffer.length)
                [NSException raise:NSInvalidArgumentException format:@"command %lu (%@): binding %lu (%@, %lu bytes at %lu of %lu) is past its buffer",
                    (unsigned long)i,name,(unsigned long)j,buffer.label?:@"unnamed",(unsigned long)command->lengths[j],
                    (unsigned long)command->offsets[j],(unsigned long)buffer.length];
            const struct MetalOperand *operand=MetalOperandOf(operands,count,buffer);
            command->strides[j]=operand?operand->stride:0;command->instances[j]=operand?operand->instances:0;
        }
    }
}

// ../dox/mesh/design/prepared-machine.md#M06: the capture's replay, its dataflow's barriers and `invocations` of its
// commands in `form`
static struct MetalReplay *MetalReplayMade(MetalCapture *capture,id<MTLDevice> device,NSUInteger invocations,enum MetalForm form) {
    MetalDataflow(capture);
    struct MetalReplay *replay=NULL;
    if(posix_memalign((void **)&replay,_Alignof(struct MetalReplay),sizeof *replay))
        [NSException raise:NSMallocException format:@"allocating replay"];
    memset(replay,0,sizeof *replay);
    @try { MetalReplayBuild(capture,device,invocations,form,replay); }
    @catch(NSException *error){MetalReplayFree(replay);@throw;}
    return replay;
}

// ../dox/mesh/design/prepared-machine.md#M06: Apple's range optimizer over the replay's indirect command buffers, its
// `invocations` of commands, run on a Metal 4 queue of its own; its end its commit feedback's
static void MetalReplayOptimize(struct MetalReplay *replay,NSUInteger invocations) {
    if(!replay->commands)return;
    id<MTLDevice> device=replay->residency.device;
    id<MTL4CommandQueue> queue=[[device newMTL4CommandQueue] autorelease];
    id<MTL4CommandAllocator> allocator=[[device newCommandAllocator] autorelease];
    id<MTL4CommandBuffer> command=[[device newCommandBuffer] autorelease];
    id<MTLSharedEvent> ended=[[device newSharedEvent] autorelease];
    if(!queue||!allocator||!command||!ended)[NSException raise:NSMallocException format:@"a Metal 4 submission optimizing the replay"];
    [command beginCommandBufferWithAllocator:allocator];
    [command useResidencySet:replay->residency];
    id<MTL4ComputeCommandEncoder> encoder=[command computeCommandEncoder];
    MetalReplaySegments(replay,NSMakeRange(0,(NSUInteger)replay->count*invocations),^(id<MTLIndirectCommandBuffer> segment,NSRange local){
        [encoder optimizeIndirectCommandBuffer:segment withRange:local];
    });
    [encoder endEncoding];[command endCommandBuffer];
    MTL4CommitOptions *options=[[MTL4CommitOptions new] autorelease];
    __block NSError *failed=nil;
    [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback){ failed=[feedback.error retain]; ended.signaledValue=1; }];
    [queue commit:&command count:1 options:options];
    [ended waitUntilSignaledValue:1 timeoutMS:UINT64_MAX];
    [failed autorelease];
    if(failed)[NSException raise:NSInternalInconsistencyException format:@"%@",failed];
}

struct MetalReplay *MetalReplayCommands(id<MTLDevice> device,const struct MetalCommandDescription *commands,size_t count,enum MetalForm form) {
    @autoreleasepool {
        @try { return MetalReplayMade(MetalCaptureDescribed(commands,count),device,1,form); }
        @catch(NSException *error) { fprintf(stderr,"record: %s\n",error.reason.UTF8String); return NULL; }
    }
}

// tools/mesh/schedule.py: whether the recordings that follow list their commands (a record-only call's), beside RECORD_ONLY
static _Atomic int listing;
void MetalRecordingList(int on) { atomic_store(&listing,on); }

// ../dox/mesh/design/prepared-machine.md#M06, #M13: a program's commands as one invocation, its operands bound by slot
// and its host operations at their commands, prepared as `invocations` of them in `form`; the program's census line,
// under RECORD_ONLY (or MetalRecordingList) one line a command, and each operand's bindings.
struct MetalReplay *MetalReplayDescribed(id<MTLDevice> device,const struct MetalCommandDescription *commands,size_t count,
    const struct MetalHostDescription *hosts,size_t hostCount,const struct MetalOperand *operands,size_t operandCount,
    NSUInteger invocations,enum MetalForm form) {
    struct MetalReplay *replay=NULL;
    @autoreleasepool {
        @try {
            MetalCapture *capture=MetalCaptureDescribed(commands,count);
            MetalCaptureOperands(capture,operands,operandCount);
            for(size_t h=0;h<hostCount;h++) {
                const struct MetalHostDescription *host=hosts+h;
                if(!host->instances||!host->bindings)[NSException raise:NSInvalidArgumentException format:@"host operation needs prepared invocation bindings"];
                NSMutableArray *bindings=[NSMutableArray arrayWithCapacity:host->instances];
                for(size_t j=0;j<host->instances;j++) {
                    const struct MetalHostBinding *binding=host->bindings+j;
                    if(!binding->operation||!binding->count)[NSException raise:NSInvalidArgumentException format:@"host operation needs declared data views"];
                    for(size_t k=0;k<binding->count;k++)if(binding->reaches[k].buffer)[capture->owners addObject:binding->reaches[k].buffer];
                    [bindings addObject:@{@"accesses":[NSData dataWithBytes:binding->reaches length:binding->count*sizeof *binding->reaches],
                        @"operation":[[binding->operation copy] autorelease]}];
                }
                [capture->hosts addObject:@{@"at":@(host->at),@"bindings":bindings,@"invocation":@(host->invocation)}];
            }
            replay=MetalReplayMade(capture,device,invocations,form);
            const struct MetalRecordedDispatch *recorded=capture->dispatches.bytes;
            const NSUInteger n=replay->count;
            NSUInteger levels=n?1:0;
            for(NSUInteger i=0;i<n;i++)levels=MAX(levels,recorded[i].level+1);
            printf("prepared program commands %lu levels %lu form %s\n",(unsigned long)n,(unsigned long)levels,(const char *[]){"icb","encoded"}[form]);
            NSMutableDictionary *allocations=[NSMutableDictionary dictionary];
            if(getenv("RECORD_ONLY")||atomic_load(&listing))for(NSUInteger i=0;i<n;i++) @autoreleasepool {
                printf("prepared %s\n",MetalCommandLine(recorded+i,capture->reaches.bytes,i,recorded[i].level,allocations).UTF8String);
            }
            for(size_t k=0;k<operandCount;k++) {
                NSUInteger bindings=0;
                for(NSUInteger i=0;i<n;i++)for(NSUInteger j=0;j<MESH_BUFFER_BINDINGS;j++)
                    bindings+=recorded[i].buffers[j]==operands[k].buffer&&(((recorded[i].access|(recorded[i].access>>MESH_ACCESS_WRITE_SHIFT))>>j)&1);
                printf("prepared operand %s bindings %lu instances %lu\n",operands[k].name,(unsigned long)bindings,(unsigned long)operands[k].instances);
            }
            MetalReplayOptimize(replay,invocations);
        } @catch(NSException *error) {
            MetalReplayFree(replay);replay=NULL;
            fprintf(stderr,"record: %s\n",error.reason.UTF8String);
        }
    }
    return replay;
}

// docs/kernels.md#collectives
static NSError *MetalReplayFailure(NSInteger code, NSString *reason) {
    return [NSError errorWithDomain:@"MetalReplay" code:code userInfo:@{NSLocalizedDescriptionKey:reason}];
}

// One prepared function owns its command stream. Append and Finish run during preparation;
// Enqueue only commits the finished stream. The observer never controls subsequent work.
@interface MetalWork : NSObject {
@public
    NSData *accesses;
    NSData *activity;
    NSArray *readVersions;
    void (^encode)(struct MetalSchedule *);
    MetalHostOperation operation;
    MeshHostTask task;
    struct MetalSchedule *root;
    NSMutableIndexSet *dependencies, *successors, *hostDependencies, *gpuDependencies;
    struct MetalSchedule *part;
    // Objective-C allocation need not have MetalReplay's extended alignment.
    unsigned char replay[sizeof(struct MetalReplay)];
    NSUInteger invocation, first, end, ordinal;
    id<MTLSharedEvent> event;
    id<MTLFence> fence;
    uint64_t eventValue;
    MTLStages stages;
    BOOL prepared, external, declaredInput, hostConsumer, exported;
    NSUInteger declaredCommand;
    _Atomic unsigned remaining;
    _Atomic int status;
    NSError *error;
    double ended;
}
@end
@implementation MetalWork
- (void)dealloc {
    if(task.release)task.release(task.context);
    [accesses release];[activity release];[readVersions release];[encode release];[operation release];[dependencies release];[successors release];
    [event release];[fence release];[error release];[hostDependencies release];[gpuDependencies release];[super dealloc];
}
@end

struct MetalSchedule {
    struct MetalSchedule *next, *tail, *root;
    struct MetalSchedule *submit, *firstSubmission, *lastSubmission;
    id<MTLDevice> device;
    enum MetalForm form;
    BOOL finished;
    id<MTL4CommandQueue> queue;
    id<MTL4CommandBuffer> command;
    id<MTL4ComputeCommandEncoder> encoder;
    id<MTL4CommandAllocator> allocator;
    id<MTL4ArgumentTable> table;
    id<MTLResidencySet> residency;
    MTL4CommitOptions *options;
    NSMutableSet *owners;
    NSMutableSet *replayResidency;
    NSMutableOrderedSet *communicators;
    NSMutableIndexSet *hostDependencies, *partDependencies;
    id<MTLSharedEvent> event, start;
    MTLSharedEventListener *listener;
    NSMutableArray *work, *hosts, *history;
    NSMutableArray *readVersions;
    MetalCapture *declarations;
    NSMutableData *waits;
    NSMutableDictionary *frontierRead, *frontierWritten, *frontierAtomic, *hostResources;
    MetalQueueAccess *queueAccesses;
    NSMutableArray *parts;
    NSUInteger commands, ordinal;
    _Atomic int status;
    // the part's end, in the statistics clock (MetalScheduleEnd), stored before its status
    double ended;
    // the root's: 1 once every part and host task has ended (MetalScheduleArrived), woken for MetalScheduleEnd
    _Atomic uint32_t arrived;
    _Atomic unsigned references;
    _Atomic unsigned pending;
    NSError *error;
    void (^completed)(NSError *);
    id<MTLBuffer> data;
    NSUInteger used;
    // design/replay-templates-2026-10-29.md: the calls that built the function, on its root (MetalScheduleTemplate),
    // and each communicator's plan declaration it last recorded
    NSMutableArray *script;
    NSMutableDictionary *plans;
};

static void MetalScheduleCall(struct MetalSchedule *s,NSDictionary *call) {
    s=s->root;
    if(!s->script)s->script=[NSMutableArray new];
    [s->script addObject:call];
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
// All wait records are produced during preparation; submission reads this flat list.
struct MetalScheduleWait { id<MTLEvent> event; uint64_t value; };

static NSMutableDictionary *MetalFrontierCopy(NSDictionary *source,NSDictionary *indices) {
    NSMutableDictionary *copy=[NSMutableDictionary dictionaryWithCapacity:source.count];
    for(id key in source) {
        NSMutableDictionary *buckets=[NSMutableDictionary dictionary];
        for(id bucket in source[key]) {
            NSMutableData *data=[[source[key][bucket] mutableCopy] autorelease];
            if(!data.length)continue;
            if(indices) {
                struct MetalSpan *spans=data.mutableBytes;NSUInteger kept=0;
                for(NSUInteger i=0;i<data.length/sizeof *spans;i++) {
                    NSNumber *mapped=indices[@(spans[i].node)];if(!mapped)continue;
                    spans[kept]=spans[i];spans[kept++].node=mapped.unsignedIntegerValue;
                }
                data.length=kept*sizeof *spans;
            }
            buckets[bucket]=data;
        }
        if(buckets.count)copy[key]=buckets;
    }
    return copy;
}

static struct MetalSchedule *MetalScheduleCreatePart(id<MTL4CommandQueue> queue,enum MetalForm form,BOOL initial) {
    if(!queue)return NULL;
    id<MTLDevice> device=queue.device;
    struct MetalSchedule *s=calloc(1,sizeof *s);
    if(!s)return NULL;
    atomic_init(&s->references,1);
    atomic_init(&s->pending,1);s->root=s;
    atomic_init(&s->status,0);
    s->device=[device retain];s->form=form;s->queue=[queue retain];
    s->owners=[NSMutableSet new];
    s->work=[NSMutableArray new];s->hosts=[NSMutableArray new];s->waits=[NSMutableData new];
    s->hostDependencies=[NSMutableIndexSet new];s->partDependencies=[NSMutableIndexSet new];
    s->history=[NSMutableArray new];s->frontierRead=[NSMutableDictionary new];s->frontierWritten=[NSMutableDictionary new];
    s->frontierAtomic=[NSMutableDictionary new];s->hostResources=[NSMutableDictionary new];
    s->parts=[[NSMutableArray alloc] initWithObjects:[NSValue valueWithPointer:s],nil];
    MTLResidencySetDescriptor *resident=[MTLResidencySetDescriptor new];
    s->residency=[device newResidencySetWithDescriptor:resident error:nil];[resident release];
    s->allocator=[device newCommandAllocator];
    MTL4ArgumentTableDescriptor *descriptor=[MTL4ArgumentTableDescriptor new];descriptor.maxBufferBindCount=MESH_BUFFER_BINDINGS;
    s->table=[device newArgumentTableWithDescriptor:descriptor error:nil];[descriptor release];
    s->command=[device newCommandBuffer];
    if(s->command&&s->allocator) {
        [s->command beginCommandBufferWithAllocator:s->allocator];
        s->encoder=[[s->command computeCommandEncoder] retain];
        if(initial) {
            @synchronized(queue) {
                MetalQueueAccess *previous=objc_getAssociatedObject(queue,&queueAccessKey);
                if(!previous) {
                    previous=[MetalQueueAccess new];
                    objc_setAssociatedObject(queue,&queueAccessKey,previous,OBJC_ASSOCIATION_RETAIN_NONATOMIC);[previous release];
                }
                s->queueAccesses=[previous retain];
            }
        }
        objc_setAssociatedObject(s->encoder,&encoderScheduleKey,[NSValue valueWithPointer:s],OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    if(!s->queue||!s->command||!s->residency||!s->allocator||!s->table||!s->encoder) {
        MetalScheduleFree(s);return NULL;
    }
    return s;
}

struct MetalSchedule *MetalScheduleCreateOn(id<MTL4CommandQueue> queue,enum MetalForm form) {
    return MetalScheduleCreatePart(queue,form,YES);
}

// design/replay-templates-2026-10-29.md: the plan `comm`'s crossings assume (ncclMeshCommPlanDeclaration), as a call
// where it differs from the last one this function recorded: its communicator a place, then its bytes, each binding
// a place
static void MetalSchedulePlan(struct MetalSchedule *root,struct ncclComm *comm) {
    size_t length=0,count=0;
    if(ncclMeshCommPlanDeclaration(comm,NULL,&length,NULL,&count)||!length)return;
    NSMutableData *bytes=[NSMutableData dataWithLength:sizeof comm+length],*at=[NSMutableData dataWithLength:count*sizeof(size_t)];
    if(ncclMeshCommPlanDeclaration(comm,(char *)bytes.mutableBytes+sizeof comm,&length,at.mutableBytes,&count))return;
    NSData *plan=[bytes subdataWithRange:NSMakeRange(sizeof comm,length)];
    NSValue *key=[NSValue valueWithPointer:comm];
    if(!root->plans)root->plans=[NSMutableDictionary new];
    if([root->plans[key] isEqualToData:plan])return;
    root->plans[key]=plan;
    memcpy(bytes.mutableBytes,&comm,sizeof comm);
    NSMutableData *places=[NSMutableData dataWithLength:(count+1)*sizeof(struct MetalPlace)];
    struct MetalPlace *into=places.mutableBytes;const size_t *offsets=at.bytes;
    into[0]=(struct MetalPlace){0,MetalRelocationCommunicator};
    for(size_t k=0;k<count;k++)into[k+1]=(struct MetalPlace){sizeof comm+offsets[k],MetalRelocationBuffer};
    MetalScheduleCall(root,@{@"call":@"plan",@"kind":@"plan",@"bytes":bytes,@"places":places});
}

void MetalScheduleCommunicator(struct MetalSchedule *s,struct ncclComm *comm) {
    s=s->root;
    if(!s->communicators)s->communicators=[NSMutableOrderedSet new];
    [s->communicators addObject:[NSValue valueWithPointer:comm]];
    MetalSchedulePlan(s,comm);
}

id<MTLResidencySet> MetalScheduleResources(struct MetalSchedule *s) { return MetalScheduleTail(s)->residency; }

struct MetalCommandBinding MetalScheduleData(struct MetalSchedule *s,const void *data,size_t bytes,uint32_t index) {
    s=MetalScheduleTail(s);
    NSUInteger size=(bytes+MESH_COMMAND_DATA_ALIGNMENT-1)/MESH_COMMAND_DATA_ALIGNMENT*MESH_COMMAND_DATA_ALIGNMENT;
    if(!s->data || s->used+size>s->data.length) {
        [s->data release];s->data=MetalLeaseBuffer(s->device,MAX((NSUInteger)MESH_COMMAND_DATA_BYTES,size));
        s->used=0;[s->owners addObject:s->data];[s->residency addAllocation:s->data];
    }
    struct MetalCommandBinding result={s->data,s->used,bytes,index,MetalReachRead,0};
    memcpy((char *)s->data.contents+s->used,data,bytes);s->used+=size;
    MetalScheduleCall(s,@{@"call":@"data",@"bytes":[NSData dataWithBytes:data length:bytes],@"index":@(index),
        @"at":[NSString stringWithFormat:@"%p+%lu",(void *)result.buffer,(unsigned long)result.offset]});
    return result;
}

static void MetalScheduleComputeNow(struct MetalSchedule *s,id<MTLComputePipelineState> pipeline,
    const struct MetalCommandBinding *bindings,size_t count,MTLSize groups,MTLSize threads) {
    s=MetalScheduleWritable(s);s->commands++;
    struct MetalRecordedDispatch command={.pipeline=pipeline,.grid=groups,.group=threads};
    struct MetalReplay view={0};
    [s->owners addObject:pipeline];
    for(size_t k=0;k<count;k++) {
        const struct MetalCommandBinding b=bindings[k];
        if(b.index>=MESH_BUFFER_BINDINGS)[NSException raise:NSInvalidArgumentException format:@"binding index exceeds Metal capacity"];
        command.buffers[b.index]=b.buffer;command.offsets[b.index]=b.offset;command.lengths[b.index]=b.length;command.invocations[b.index]=b.invocation;
        command.access|=(uint64_t)(b.access&MetalReachRead?1:0)<<b.index;
        command.access|=(uint64_t)(b.access&MetalReachWritten?1:0)<<(MESH_ACCESS_WRITE_SHIFT+b.index);
        view.buffers=MAX(view.buffers,b.index+1);
        [s->owners addObject:b.buffer];
        [s->residency addAllocation:b.buffer];
    }
    MetalReplayDispatch4(s->encoder,s->table,&view,&command,0);
}

static MetalWork *MetalScheduleWork(struct MetalSchedule *root,NSData *accesses,
    void (^encode)(struct MetalSchedule *),MetalHostOperation operation) {
    MetalWork *work=[MetalWork new];work->accesses=[accesses copy];work->encode=[encode copy];work->operation=[operation copy];
    work->readVersions=[root->readVersions copy];
    work->root=root;
    work->stages=MTLStageDispatch;work->eventValue=1;
    work->dependencies=[NSMutableIndexSet new];work->successors=[NSMutableIndexSet new];
    work->hostDependencies=[NSMutableIndexSet new];work->gpuDependencies=[NSMutableIndexSet new];
    const struct MetalReach *views=accesses.bytes;
    for(NSUInteger k=0;k<accesses.length/sizeof *views;k++)if(views[k].buffer)[root->owners addObject:views[k].buffer];
    [root->work addObject:work];[work release];return work;
}

void MetalScheduleRetainObject(struct MetalSchedule *root,id object) {
    [root->root->owners addObject:object];
}

void MetalScheduleReadVersions(struct MetalSchedule *root,const struct MetalReach *views,size_t count,void (^declare)(void)) {
    root=root->root;
    NSMutableDictionary *domain=[NSMutableDictionary dictionary];
    MetalViewAccesses([NSData dataWithBytes:views length:count*sizeof *views],^(id key,NSUInteger lo,NSUInteger hi,uint32_t access){
        if(access&MetalReachNext)MetalSpanAdd(domain,key,lo,hi,0);
    });
    if(!root->readVersions)root->readVersions=[NSMutableArray new];
    [root->readVersions addObject:domain];
    @try { declare(); } @finally { [root->readVersions removeLastObject]; }
}

void MetalScheduleInput(struct MetalSchedule *root,const struct MetalReach *views,size_t count,id<MTLEvent> event,uint64_t value) {
    if(!event)[NSException raise:NSInvalidArgumentException format:@"page producer requires its completion event"];
    MetalWork *work=MetalScheduleWork(root,[NSData dataWithBytes:views length:count*sizeof *views],nil,nil);
    work->external=YES;work->declaredInput=YES;work->event=[(id)event retain];work->eventValue=value;
}

void MetalEncoderCompute(id<MTL4ComputeCommandEncoder> encoder,const struct MetalCommandDescription *command) {
    struct MetalSchedule *root=MetalEncoderSchedule(encoder);
    if(!root->declarations)root->declarations=[MetalCapture new];
    const NSUInteger at=root->declarations->dispatches.length/sizeof(struct MetalRecordedDispatch);
    MetalCaptureAppend(root->declarations,command,1);
    MetalWork *work=MetalScheduleWork(root,nil,nil,nil);work->declaredCommand=at+1;
}

struct MetalSchedule *MetalEncoderSchedule(id<MTL4ComputeCommandEncoder> encoder) {
    struct MetalSchedule *root=[objc_getAssociatedObject(encoder,&encoderScheduleKey) pointerValue];
    if(!root)[NSException raise:NSInvalidArgumentException format:@"command needs its function preparation context"];
    return root->root;
}

void MetalScheduleCompute(struct MetalSchedule *root,id<MTLComputePipelineState> pipeline,
    const struct MetalCommandBinding *bindings,size_t count,MTLSize groups,MTLSize threads) { @autoreleasepool {
    NSData *bound=[NSData dataWithBytes:bindings length:count*sizeof *bindings];
    MetalScheduleCall(root,@{@"call":@"compute",@"pipeline":pipeline,@"bindings":bound,
        @"groups":@[@(groups.width),@(groups.height),@(groups.depth)],@"threads":@[@(threads.width),@(threads.height),@(threads.depth)]});
    NSMutableData *accesses=[NSMutableData data];
    for(size_t k=0;k<count;k++) {
        const struct MetalCommandBinding b=bindings[k];
        const struct MetalReach reach={b.buffer,b.offset,b.length,b.access,-1};
        [accesses appendBytes:&reach length:sizeof reach];
    }
    MetalScheduleWork(root,accesses,^(struct MetalSchedule *s){
        MetalScheduleComputeNow(s,pipeline,bound.bytes,count,groups,threads);
    },nil);
}
}

static int MetalScheduleCrossWork(struct MetalSchedule *root,NSData *accesses,const ncclMeshGroup *group,MetalCrossing cross) {
    (void)accesses;(void)group;
    // The callback realizes immutable bindings and declares each inner command.
    // It runs now, before dependency compilation, and cannot hide a collective
    // behind an opaque runtime operation or force a partial compiler flush.
    struct MetalSchedule *s=MetalScheduleTail(root);
    return cross(s->encoder,s->table,s->residency,0,0);
}

// design/replay-templates-2026-10-29.md: a call with its declaration, its bytes and places copied
static NSMutableDictionary *MetalScheduleDeclared(NSString *call,const struct MetalDeclaration *declared) {
    NSMutableDictionary *entry=[NSMutableDictionary dictionaryWithObject:call forKey:@"call"];
    if(declared) {
        entry[@"kind"]=@(declared->kind);entry[@"bytes"]=[NSData dataWithBytes:declared->bytes length:declared->length];
        entry[@"places"]=[NSData dataWithBytes:declared->places length:declared->count*sizeof *declared->places];
    }
    return entry;
}

int MetalScheduleCross(struct MetalSchedule *root,const struct MetalReach *views,size_t count,
    const ncclMeshGroup *group,const struct MetalDeclaration *declared,MetalCrossing cross) {
    NSData *accesses=[NSData dataWithBytes:views length:count*sizeof *views];
    NSMutableDictionary *entry=MetalScheduleDeclared(@"cross",declared);entry[@"accesses"]=accesses;
    MetalScheduleCall(root,entry);
    return MetalScheduleCrossWork(root,accesses,group,cross);
}

static void MetalScheduleHostWork(struct MetalSchedule *root,NSData *views,MetalHostOperation operation) {
    if(!operation||!views.length)[NSException raise:NSInvalidArgumentException format:@"host operation needs declared data views"];
    MetalScheduleWork(root,views,nil,operation);
}

void MetalScheduleHost(struct MetalSchedule *root,const struct MetalReach *views,size_t count,const struct MetalDeclaration *declared,
    MetalHostOperation operation) { @autoreleasepool {
    NSData *accesses=[NSData dataWithBytes:views length:count*sizeof *views];
    NSMutableDictionary *entry=MetalScheduleDeclared(@"host",declared);entry[@"accesses"]=accesses;
    MetalScheduleCall(root,entry);
    MetalScheduleHostWork(root,accesses,operation);
}
}

void MetalEncoderHost(id<MTL4ComputeCommandEncoder> encoder,const struct MetalReach *views,size_t count,
    const struct MetalDeclaration *declared,MetalHostOperation operation) {
    struct MetalSchedule *root=MetalEncoderSchedule(encoder);
    (void)declared; // The containing crossing owns the template declaration.
    MetalScheduleHostWork(root->root,[NSData dataWithBytes:views length:count*sizeof *views],operation);
}

static void MetalScheduleRangeNow(struct MetalSchedule *root,struct MetalReplay *replay,NSUInteger t,NSUInteger from,NSUInteger to) {
    struct MetalSchedule *s=MetalScheduleWritable(root);
    if(!replay||from==to)return;
    s->commands+=to-from;
    NSArray *held=objc_getAssociatedObject(replay->owner,&replayRecordsKey);
    const struct MetalRecordedDispatch *records=[held[0] bytes];
    /* the replay's own form decides its dispatch, direct (Encoded) or its indirect command buffers (Indirect), so one
       Metal 4 schedule carries replays of either */
    const enum MetalForm made=(enum MetalForm)[held[2] unsignedIntegerValue];
    const NSUInteger n=replay->count;
    [s->owners addObject:replay->owner];
    if(!s->replayResidency)s->replayResidency=[NSMutableSet new];
    if(![s->replayResidency containsObject:replay->residency]) {
        [s->replayResidency addObject:replay->residency];
        [s->command useResidencySet:replay->residency];
    }
    id<MTL4ComputeCommandEncoder> encoder=s->encoder;
    for(NSUInteger c=t*n+from;c<t*n+to;c++) @autoreleasepool {
        const struct MetalRecordedDispatch *source=records+c%n;
        if(made==MetalFormEncoded||source->indirectGrid) {
            MetalReplayDispatch4(encoder,s->table,replay,source,t);
        } else {
            MetalReplaySegments(replay,NSMakeRange(c,1),^(id<MTLIndirectCommandBuffer> segment,NSRange local){
                [encoder executeCommandsInBuffer:segment withRange:local];
            });
        }
    }
}

static void MetalScheduleRange(struct MetalSchedule *root,struct MetalReplay *replay,NSUInteger t,NSUInteger from,NSUInteger to) {
    if(!replay||from==to)return;
    [root->owners addObject:replay->owner];[root->owners addObject:replay->residency];
    MetalWork *work=MetalScheduleWork(root,nil,nil,nil);
    memcpy(work->replay,replay,sizeof *replay);work->invocation=t;work->first=from;work->end=to;
}

// metal-recording.h MetalScheduleReplay: commands `commands` of a replay's first invocation, a range of the schedule's work
void MetalScheduleReplay(struct MetalSchedule *root,struct MetalReplay *replay,NSRange commands) {
    MetalScheduleRange(root,replay,0,commands.location,NSMaxRange(commands));
}

static void MetalScheduleCopyNow(struct MetalSchedule *root,id<MTLBuffer> source,NSUInteger from,id<MTLBuffer> destination,NSUInteger to,
    NSUInteger bytes) {
    struct MetalSchedule *s=MetalScheduleWritable(root);
    s->commands++;
    [s->owners addObject:source];[s->owners addObject:destination];
    [s->residency addAllocation:source];[s->residency addAllocation:destination];
    id<MTL4ComputeCommandEncoder> encoder=s->encoder;
    [encoder copyFromBuffer:source sourceOffset:from toBuffer:destination destinationOffset:to size:bytes];
}

// metal-recording.h MetalScheduleCopy
void MetalScheduleCopy(struct MetalSchedule *root,id<MTLBuffer> source,NSUInteger from,id<MTLBuffer> destination,NSUInteger to,
    NSUInteger bytes) { @autoreleasepool {
    if(!bytes)return;
    const struct MetalReach reaches[2]={{source,from,bytes,MetalReachRead,-1},{destination,to,bytes,MetalReachWritten,-1}};
    MetalWork *work=MetalScheduleWork(root,[NSData dataWithBytes:reaches length:sizeof reaches],^(struct MetalSchedule *now){
        MetalScheduleCopyNow(now,source,from,destination,to,bytes);
    },nil);
    work->stages=MTLStageBlit;
}
}

// Every backend enters one graph with one node for each command. Grouping
// commands for storage never changes the dependency relation.
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
            [work->readVersions release];work->readVersions=[range->readVersions copy];
            memcpy(work->replay,&replay,sizeof replay);work->invocation=range->invocation;work->first=c;work->end=c+1;
        }
    }
    NSArray *nodes=[[root->work copy] autorelease];[root->work removeAllObjects];return nodes;
}

static struct MetalSchedule *MetalScheduleWritable(struct MetalSchedule *root) {
    struct MetalSchedule *tail=MetalScheduleTail(root);
    if(!tail->finished)return tail;
    struct MetalSchedule *next=MetalScheduleCreatePart(tail->queue,tail->form,NO);
    if(!next)[NSException raise:NSMallocException format:@"preparing data-dependent command part"];
    next->root=root;next->ordinal=tail->ordinal+1;tail->next=next;root->tail=next;
    [root->parts addObject:[NSValue valueWithPointer:next]];
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
        [source appendString:MetalViewMappedIndexDeclaration(binding,&count,&stride,1,0,@{@0:map},nil,0,0)];
    }
    [source appendFormat:@"}\nstruct %@ { uint producer;\n"
        "uint begin() const { return uint(%@_indices::mesh_index<0>(producer)); }\n"
        "uint end() const { return uint(%@_indices::mesh_index<0>(producer+1)); }\n"
        "static uint target(uint i) { return uint(%@_indices::mesh_index<1>(i)); }\n"
        "static uint holders(uint g) { return uint(%@_indices::mesh_index<2>(g)); }\n};\n",name,name,name,name,name];
    return source;
}

static void MetalHostFinished(void *context,const char *failure) {
    MetalWork *task=context;
    NSError *error=failure?MetalReplayFailure(EIO,@(failure)):nil;
    task->error=[error retain];
    task->ended=(double)clock_gettime_nsec_np(CLOCK_MONOTONIC)*1e-9;
    atomic_store_explicit(&task->status,error?-1:1,memory_order_release);
    task->event.signaledValue=task->eventValue;
    MetalScheduleArrived(task->root);
}

// Expand compound declarations once. Their only runtime leaves are the same
// prepared native tasks used by individual host operations. The wrapper and
// declaration arrays remain preparation metadata, not an instruction loop.
static void MetalScheduleExpandHost(struct MetalSchedule *root,MetalWork *work,NSMutableArray *nodes) {
    if(!work->operation){[nodes addObject:work];return;}
    if(!work->prepared) {
        const char *failure=work->operation(MetalHostFinished,work,&work->task);
        if(failure)[NSException raise:NSInvalidArgumentException format:@"host preparation: %s",failure];
        work->prepared=YES;
    }
    if(work->task.activity) {
        NSMutableData *activity=work->activity?[[work->activity mutableCopy] autorelease]:[NSMutableData data];
        [activity appendBytes:&work->task.activity length:sizeof work->task.activity];
        [work->activity release];work->activity=[activity copy];
    }
    if(!work->task.commands) {
        if(!work->task.submit||!work->task.release)
            [NSException raise:NSInvalidArgumentException format:@"host preparation needs submission and lifetime operands"];
        if(work->activity.length) {
            NSMutableData *views=[[work->accesses mutableCopy] autorelease];
            const uint32_t *const *activity=work->activity.bytes;
            for(NSUInteger i=0;i<work->activity.length/sizeof *activity;i++) {
                const struct MetalReach reach={nil,(NSUInteger)activity[i],sizeof(uint32_t),MetalReachRead,-1};
                [views appendBytes:&reach length:sizeof reach];
            }
            [work->accesses release];work->accesses=[views copy];
        }
        [nodes addObject:work];return;
    }
    if(work->task.submit)[NSException raise:NSInvalidArgumentException format:@"compound host declaration cannot supply an executor"];
    for(size_t i=0;i<work->task.command_count;i++) {
        const MeshHostCommand command=work->task.commands[i];
        if(!command.prepare)[NSException raise:NSInvalidArgumentException format:@"host instruction requires its preparation entry"];
        NSMutableData *views=[NSMutableData dataWithCapacity:command.access_count*sizeof(struct MetalReach)];
        for(size_t j=0;j<command.access_count;j++) {
            const MeshHostAccess access=command.accesses[j];
            const struct MetalReach view={nil,(NSUInteger)access.pages+access.offset,access.length,access.access,-1};
            [views appendBytes:&view length:sizeof view];
        }
        MetalWork *child=MetalScheduleWork(root,views,nil,^const char *(MeshHostCompletion complete,void *context,MeshHostTask *task){
            (void)work; // Retain the compound declaration until this prepared leaf is released.
            return command.prepare(command.declaration,complete,context,task);
        });
        child->activity=[work->activity copy];
        [child->readVersions release];child->readVersions=[work->readVersions copy];
        MetalScheduleExpandHost(root,child,nodes);
    }
}

static BOOL MetalWorkHost(MetalWork *work) { return work->operation!=nil||(work->external&&!work->fence); }

static NSUInteger MetalWorkDependencies(struct MetalSchedule *root,NSArray *work) {
    const NSUInteger first=root->history.count;
    NSMutableDictionary *next=[NSMutableDictionary dictionary],*observed=[NSMutableDictionary dictionary];
    for(NSUInteger i=0;i<work.count;i++) {
        MetalWork *w=work[i];w->ordinal=first+i;[root->history addObject:w];
    }
    for(NSUInteger i=0;i<work.count;i++) {
        MetalWork *w=work[i];
        for(NSMutableDictionary *domain in w->external?@[]:w->readVersions) {
            NSData *qualified=MetalViewQualify(w->accesses,domain);
            [w->accesses release];w->accesses=[qualified retain];
        }
        [w->readVersions release];w->readVersions=nil;
        if(w->operation) {
            for(size_t r=0;r<w->task.resource_count;r++) {
                const MeshHostResource resource=w->task.resources[r];
                if(!resource.identity||!resource.capacity)[NSException raise:NSInvalidArgumentException format:@"host resource needs identity and capacity"];
                NSValue *key=[NSValue valueWithPointer:resource.identity];
                NSMutableDictionary *held=root->hostResources[key];
                if(!held)root->hostResources[key]=held=[[@{@"capacity":@(resource.capacity),@"uses":[NSMutableArray array]} mutableCopy] autorelease];
                if([held[@"capacity"] unsignedLongLongValue]!=resource.capacity)[NSException raise:NSInvalidArgumentException format:@"host resource has conflicting capacities"];
                NSMutableArray *uses=held[@"uses"];
                if(uses.count>=resource.capacity)[w->dependencies addIndex:[uses[uses.count-resource.capacity] unsignedIntegerValue]];
                [uses addObject:@(first+i)];
            }
        }
        MetalViewStep(root->frontierRead,root->frontierWritten,root->frontierAtomic,next,observed,w->accesses,first+i,
            ^(NSUInteger consumer,NSUInteger producer){
                [((MetalWork *)root->history[consumer])->dependencies addIndex:producer];
            });
        [w->accesses release];w->accesses=nil;
    }
    // A forward read without a later writer may name an explicitly supplied
    // function input. Only that still-current version qualifies; a previous
    // numerical write or another function's queue history cannot supply it.
    NSMutableDictionary *inputsAvailable=MetalFrontierCopy(root->frontierWritten,nil);
    for(id key in root->frontierAtomic)for(NSData *data in [root->frontierAtomic[key] allValues]) {
        const struct MetalSpan *contributions=data.bytes;
        for(NSUInteger i=0;i<data.length/sizeof *contributions;i++)
            MetalSpanErase(inputsAvailable,key,contributions[i].lo,contributions[i].hi);
    }
    for(id key in inputsAvailable)for(NSData *data in [inputsAvailable[key] allValues]) {
        const struct MetalSpan *inputs=data.bytes;
        for(NSUInteger i=0;i<data.length/sizeof *inputs;i++) {
            const struct MetalSpan input=inputs[i];MetalWork *producer=root->history[input.node];
            if(!producer->declaredInput)continue;
            NSMutableDictionary *const pending[2]={next,observed};
            for(NSUInteger s=0;s<2;s++) {
                NSDictionary *buckets=pending[s][key];
                for(NSUInteger b=input.lo>>MESH_FRONTIER_BITS;b<=(input.hi-1)>>MESH_FRONTIER_BITS;b++) {
                    NSData *piecesData=buckets[@(b)];const struct MetalSpan *pieces=piecesData.bytes;
                    for(NSUInteger j=0;j<piecesData.length/sizeof *pieces;j++)if(pieces[j].lo<input.hi&&input.lo<pieces[j].hi) {
                        [((MetalWork *)root->history[pieces[j].node])->dependencies addIndex:input.node];
                        if(!s)MetalSpanAdd(root->frontierRead,key,MAX(input.lo,pieces[j].lo),MIN(input.hi,pieces[j].hi),pieces[j].node);
                    }
                }
                MetalSpanErase(pending[s],key,input.lo,input.hi);
            }
        }
    }
    if(MetalViewPending(next)||MetalViewPending(observed))
        [NSException raise:NSInvalidArgumentException format:@"next-version page read has no declared producer or input"];
    NSMutableData *counts=[NSMutableData dataWithLength:work.count*sizeof(NSUInteger)];NSUInteger *left=counts.mutableBytes;
    NSMutableIndexSet *ready=[NSMutableIndexSet indexSet];
    for(NSUInteger i=0;i<work.count;i++) {
        MetalWork *w=work[i];
        [w->dependencies enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){
            (void)stop;if(p>=first)[((MetalWork *)root->history[p])->successors addIndex:i];
        }];
        left[i]=[w->dependencies countOfIndexesInRange:NSMakeRange(first,work.count)];
        if(!left[i])[ready addIndex:i];
    }
    NSUInteger compiled=0;
    while(ready.count) {
        const NSUInteger i=ready.firstIndex;[ready removeIndex:i];MetalWork *w=work[i];compiled++;
        [w->dependencies enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){
            (void)stop;MetalWork *producer=root->history[p];
            [w->hostDependencies addIndexes:producer->hostDependencies];
            if(MetalWorkHost(producer)) {
                [w->hostDependencies addIndex:p];
                [w->gpuDependencies addIndexes:producer->gpuDependencies];
            } else {
                [w->gpuDependencies addIndex:p];
                if(w->operation)producer->hostConsumer=YES;
            }
        }];
        [w->successors enumerateIndexesUsingBlock:^(NSUInteger successor,BOOL *stop){
            (void)stop;if(!--left[successor])[ready addIndex:successor];
        }];
    }
    if(compiled!=work.count)[NSException raise:NSInvalidArgumentException format:@"page-version dependencies contain a cycle"];
    return first;
}

static id<MTLSharedEvent> MetalSchedulePublication(struct MetalSchedule *s) {
    if(!s->event) {
        s->event=[s->device newSharedEvent];
        if(!s->event)[NSException raise:NSMallocException format:@"preparing GPU input publication"];
        // A later host node can name an already closed part. It is still
        // unsubmitted: its queue signals the publication after it (MetalScheduleEnqueue).
    }
    MetalScheduleFinishPart(s);return s->event;
}

static void MetalScheduleBindHost(struct MetalSchedule *root,MetalWork *work,NSArray *nodes) {
    NSMutableDictionary *events=[NSMutableDictionary dictionary];
    __block BOOL internal=NO;
    void (^add)(id,uint64_t)=^(id event,uint64_t value){
        NSValue *key=[NSValue valueWithPointer:event];
        events[key]=@[event,@(MAX(value,[events[key][1] unsignedLongLongValue]))];
    };
    [work->dependencies enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){
        (void)stop;MetalWork *producer=nodes[p];
        internal|=!producer->external;
        if(MetalWorkHost(producer)||producer->external)add(producer->event,producer->eventValue);
        else add(MetalSchedulePublication(producer->part),1);
    }];
    // Each actual GPU predecessor has a publication. Encoding order is not
    // execution order, so an ordinal cannot stand for those dependencies.
    if(!internal) {
        if(!root->start)root->start=[root->device newSharedEvent];
        if(!root->start)[NSException raise:NSMallocException format:@"preparing native host start"];
        add(root->start,1);
    }
    atomic_init(&work->remaining,(unsigned)events.count);
    // The listeners and all operands are retained during preparation. Only the
    // last actual prerequisite invokes the native operation; no Python runs.
    // External inputs may already be ready during preparation. A task without
    // an internal predecessor also names this function's entry event, so it
    // cannot start before Enqueue. Every other task inherits that entry through
    // its actual producer chain.
    MetalWork *task=work;
    for(NSArray *dependency in events.allValues) {
        id<MTLSharedEvent> event=dependency[0];
        [event notifyListener:root->listener atValue:[dependency[1] unsignedLongLongValue]
            block:^(id<MTLSharedEvent> reached,uint64_t value){
                (void)reached;(void)value;
                if(atomic_fetch_sub_explicit(&task->remaining,1,memory_order_acq_rel)!=1)return;
                // A fixed-width mask over declared page operands, evaluated
                // once at this leaf's data-ready continuation. It never polls
                // a dependency or chooses a different numerical implementation.
                const uint32_t *const *activity=task->activity.bytes;
                uint32_t active=1;
                for(NSUInteger i=0;i<task->activity.length/sizeof *activity;i++)active&=!!*activity[i];
                if(active)task->task.submit(task->task.context);
                else MetalHostFinished(task,NULL);
            }];
    }
}

// Choose command cuts from required waits, not from an incidental compiler flush.
// An independent later command can then move before an earlier host-dependent
// part. Commands with the same dependencies keep their existing batched replay.
static struct MetalSchedule *MetalSchedulePart(struct MetalSchedule *root,MetalWork *work,NSIndexSet *gpuDependencies) {
    NSMutableIndexSet *parts=[NSMutableIndexSet indexSet];
    NSMutableIndexSet *hosts=[[work->hostDependencies mutableCopy] autorelease];
    [gpuDependencies enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){
        (void)stop;struct MetalSchedule *part=((MetalWork *)root->history[p])->part;
        if(!part)return; // A preceding prepared function supplies its fence directly.
        [parts addIndex:part->ordinal];[parts addIndexes:part->partDependencies];
        [hosts addIndexes:part->hostDependencies];
    }];
    struct MetalSchedule *s=MetalScheduleWritable(root);
    if(s->commands) {
        BOOL additional=![s->hostDependencies containsIndexes:hosts];
        BOOL inherited=![hosts containsIndexes:s->hostDependencies];
        if(additional||(inherited&&![parts containsIndex:s->ordinal])) {
            MetalScheduleFinishPart(s);s=MetalScheduleWritable(root);
        }
    }
    [parts removeIndex:s->ordinal];
    [s->partDependencies addIndexes:parts];[s->hostDependencies addIndexes:hosts];
    return s;
}

static BOOL MetalScheduleSameInputs(MetalWork *left,MetalWork *right) {
    return [left->hostDependencies isEqual:right->hostDependencies];
}

static NSIndexSet *MetalScheduleFrontierNodes(struct MetalSchedule *root) {
    NSMutableIndexSet *live=[NSMutableIndexSet indexSet];
    for(NSDictionary *frontier in @[root->frontierRead,root->frontierWritten,root->frontierAtomic])
        for(NSDictionary *buckets in frontier.allValues)for(NSData *data in buckets.allValues) {
            const struct MetalSpan *spans=data.bytes;
            for(NSUInteger i=0;i<data.length/sizeof *spans;i++)[live addIndex:spans[i].node];
        }
    return live;
}

static void MetalScheduleEndEncoder(struct MetalSchedule *s) {
    if(!s->encoder)return;
    objc_setAssociatedObject(s->encoder,&encoderScheduleKey,nil,OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [s->encoder endEncoding];[s->encoder release];s->encoder=nil;
}

// A fence covers just its producer's encoder. Independent commands have their
// own scopes in the same command buffer, so no stage-prefix barrier orders them.
static void MetalScheduleBeginCommand(struct MetalSchedule *s,MetalWork *work) {
    MetalScheduleEndEncoder(s);
    s->encoder=[[s->command computeCommandEncoder] retain];
    if(!s->encoder)[NSException raise:NSMallocException format:@"preparing command encoder"];
    objc_setAssociatedObject(s->encoder,&encoderScheduleKey,[NSValue valueWithPointer:s],OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [work->gpuDependencies enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){
        (void)stop;MetalWork *producer=s->root->history[p];
        [s->encoder waitForFence:producer->fence beforeEncoderStages:work->stages];
    }];
    work->fence=[s->device newFence];
    if(!work->fence)[NSException raise:NSMallocException format:@"preparing command dependency fence"];
}

// Storage adjacency in an ICB is not an execution dependency. Each command
// consumes the same compiled edges, whether its dispatch is direct or indirect.
static void MetalScheduleEmit(struct MetalSchedule *root,NSArray *batch) {
    NSMutableArray *groups=[NSMutableArray array];NSMutableDictionary *equivalent=[NSMutableDictionary dictionary];
    for(MetalWork *work in batch) {
        if(work->external)continue;
        // Internal commands with exactly the same incoming and outgoing edges
        // are a complete bipartite block. One publication preserves every edge.
        // Escaping page values and host publications keep individual operands.
        NSArray *key=work->exported||work->hostConsumer?nil:@[work->dependencies,work->successors,work->hostDependencies,@(work->stages)];
        NSMutableArray *group=key?equivalent[key]:nil;
        if(!group){group=[NSMutableArray array];[groups addObject:group];if(key)equivalent[key]=group;}
        [group addObject:work];
    }
    for(NSArray *group in groups) {
        MetalWork *work=group[0];
        struct MetalSchedule *part=MetalSchedulePart(root,work,work->gpuDependencies);
        // Shared events are queue publications. Isolate a GPU value consumed
        // by a host task so its publication never includes unrelated commands.
        if(work->hostConsumer&&part->commands) {
            MetalScheduleFinishPart(part);
            part=MetalSchedulePart(root,work,work->gpuDependencies);
        }
        MetalScheduleBeginCommand(part,work);
        for(MetalWork *member in group) {
            if(member->end) {
                struct MetalReplay replay;memcpy(&replay,member->replay,sizeof replay);
                MetalScheduleRangeNow(root,&replay,member->invocation,member->first,member->end);
            } else member->encode(root);
            member->part=part;[member->encode release];member->encode=nil;
            if(member!=work)member->fence=[work->fence retain];
        }
        [part->encoder updateFence:work->fence afterEncoderStages:work->stages];
        MetalScheduleEndEncoder(part);
        if(work->hostConsumer)MetalSchedulePublication(part);
    }
}

static void MetalScheduleFlush(struct MetalSchedule *root) { @autoreleasepool {
    NSArray *nodes=[[root->work copy] autorelease];[root->work removeAllObjects];
    if(!nodes.count)return;
    if(root->declarations) {
        // One shared capture/residency/ICB storage for all transport commands,
        // regardless of how many dependency cells the compiler emits.
        MetalCaptureOperands(root->declarations,NULL,0);
        struct MetalReplay *replay=MetalReplayMade(root->declarations,root->device,1,root->form);
        [root->owners addObject:replay->owner];[root->owners addObject:replay->residency];
        if(replay->commands)[root->owners addObject:replay->commands];
        for(MetalWork *work in nodes)if(work->declaredCommand) {
            memcpy(work->replay,replay,sizeof *replay);work->first=work->declaredCommand-1;work->end=work->declaredCommand;
            work->declaredCommand=0;
        }
        MetalReplayFree(replay);[root->declarations release];root->declarations=nil;
    }
    nodes=MetalScheduleExpand(root,nodes);
    NSMutableArray *expanded=[NSMutableArray array];
    for(MetalWork *work in nodes)MetalScheduleExpandHost(root,work,expanded);
    [root->work removeAllObjects];nodes=expanded;
    BOOL hosts=NO;for(MetalWork *work in nodes)hosts|=work->operation!=nil;
    const NSUInteger first=MetalWorkDependencies(root,nodes);
    [MetalScheduleFrontierNodes(root) enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){
        (void)stop;((MetalWork *)root->history[p])->exported=YES;
    }];
    if(hosts&&!root->listener) {
        root->listener=[[MTLSharedEventListener alloc] initWithDispatchQueue:dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0)];
        if(!root->listener)[NSException raise:NSMallocException format:@"preparing native host data events"];
    }
    for(MetalWork *work in nodes)if(work->operation) {
        work->event=[root->device newSharedEvent];
        if(!work->event)[NSException raise:NSMallocException format:@"preparing host output event"];
        atomic_init(&work->status,0);[root->hosts addObject:work];atomic_fetch_add(&root->pending,1);
    }
    NSMutableData *counts=[NSMutableData dataWithLength:nodes.count*sizeof(NSUInteger)];NSUInteger *left=counts.mutableBytes;
    NSMutableIndexSet *ready=[NSMutableIndexSet indexSet];
    for(NSUInteger i=0;i<nodes.count;i++) {
        left[i]=[((MetalWork *)nodes[i])->dependencies countOfIndexesInRange:NSMakeRange(first,nodes.count)];
        if(!left[i])[ready addIndex:i];
    }
    NSMutableArray *batch=[NSMutableArray array];
    while(ready.count) {
        __block NSUInteger selected=NSNotFound;
        [ready enumerateIndexesUsingBlock:^(NSUInteger i,BOOL *stop){
            MetalWork *work=nodes[i];
            if(work->operation){selected=i;*stop=YES;return;}
            if([MetalScheduleTail(root)->hostDependencies containsIndexes:work->hostDependencies]&&selected==NSNotFound)selected=i;
        }];
        if(selected==NSNotFound)selected=ready.firstIndex;
        [ready removeIndex:selected];MetalWork *work=nodes[selected];
        if(work->external) {
            // An external producer is a typed input, not executable work.
        } else if(work->operation) {
            MetalScheduleEmit(root,batch);[batch removeAllObjects];
            MetalScheduleBindHost(root,work,root->history);
        } else {
            // Emit preceding work first so each named GPU predecessor already
            // has its command part when the next batch chooses its wait scope.
            if(batch.count&&!MetalScheduleSameInputs(batch.lastObject,work)) {
                MetalScheduleEmit(root,batch);[batch removeAllObjects];
            }
            [batch addObject:work];
        }
        [work->successors enumerateIndexesUsingBlock:^(NSUInteger i,BOOL *stop){(void)stop;if(!--left[i])[ready addIndex:i];}];
    }
    MetalScheduleEmit(root,batch);
}
}

NSError *MetalScheduleAppend(struct MetalSchedule *root,struct MetalReplay *replay,NSRange invocations,
    NSArray<NSNumber *> *marks,MetalCrossingViews views,MetalCrossing cross) {
    if(marks.count&&(!views||!cross))return MetalReplayFailure(EINVAL,@"crossing marks require their page accesses and encoder");
    NSArray *held=replay?objc_getAssociatedObject(replay->owner,&replayRecordsKey):nil;
    if(replay&&NSMaxRange(invocations)>[held[4] unsignedIntegerValue])return MetalReplayFailure(EINVAL,@"invocations exceed the recorded slots");
    if(replay)MetalScheduleCall(root,@{@"call":@"append",@"owner":replay->owner,@"buffers":@(replay->buffers),@"memory":@(replay->memory),
        @"invocations":@[@(invocations.location),@(invocations.length)],@"marks":marks?:@[]});
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
                NSDictionary *host=hosts[h++];
                NSArray *bindings=host[@"bindings"];
                NSDictionary *binding=bindings[ncclMeshInvocationIndex(t,[host[@"invocation"] longLongValue],bindings.count)];
                MetalScheduleHostWork(root,binding[@"accesses"],binding[@"operation"]);
            }
            while(k<marks.count&&marks[k].unsignedIntegerValue==at) {
                const NSUInteger mark=k++;
                ncclMeshGroup group={0};NSData *accesses=views(t,mark,&group);
                int failed=MetalScheduleCrossWork(root,accesses,&group,^(id<MTL4ComputeCommandEncoder> encoder,id<MTL4ArgumentTable> table,
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

static void MetalScheduleArrived(struct MetalSchedule *root) {
    if(atomic_fetch_sub_explicit(&root->pending,1,memory_order_acq_rel)==1) {
        NSError *failure=nil;
        MetalScheduleStatus(root,&failure);
        atomic_store_explicit(&root->arrived,1,memory_order_release);
        os_sync_wake_by_address_all(&root->arrived,sizeof root->arrived,OS_SYNC_WAKE_BY_ADDRESS_NONE);
        if(root->completed)root->completed(failure);
        MetalScheduleFree(root);
    }
}

// a GPU host time (mach_absolute_time's seconds, MTL4CommitFeedback) in the statistics clock, CLOCK_MONOTONIC
// (mesh-statistics.c), which also counts the time the machine slept: the two clocks' difference signed, since on a
// machine that never slept it is about 0 and the second reading may be the later one (2ff98b3b's unsigned difference
// wrapped to 2^64 ns there)
static double MetalMonotonic(CFTimeInterval host) {
    const uint64_t uptime=clock_gettime_nsec_np(CLOCK_UPTIME_RAW),monotonic=clock_gettime_nsec_np(CLOCK_MONOTONIC);
    return host+(double)(int64_t)(monotonic-uptime)*1e-9;
}
static void MetalScheduleCompleted(struct MetalSchedule *s,NSError *error,double ended) {
    s->error=[error retain];
    s->ended=ended;
    atomic_store_explicit(&s->status,error?-1:1,memory_order_release);
    MetalScheduleArrived(s->root);
    MetalScheduleFree(s);
}

static void MetalScheduleFinishPart(struct MetalSchedule *s) {
    if(s->finished)return;
    [s->residency commit];[s->residency requestResidency];
    if(s->residency)[s->command useResidencySet:s->residency];
    MetalScheduleEndEncoder(s);s->finished=YES;
    if(!s->commands) {
        [s->command endCommandBuffer];
        atomic_store_explicit(&s->status,1,memory_order_relaxed);
        return;
    }
    struct MetalSchedule *root=s->root;
    atomic_fetch_add_explicit(&root->pending,1,memory_order_relaxed);
    [s->command endCommandBuffer];
    s->options=[MTL4CommitOptions new];
    [s->options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback){
        MetalScheduleCompleted(s,feedback.error,MetalMonotonic(feedback.GPUEndTime));
    }];
}

// Lower the persistent dependency graph to one flat commit order. A host wait
// follows every GPU part needed to produce that host's inputs. Among ready
// parts, work whose host results are already covered is committed first.
static void MetalScheduleOrder(struct MetalSchedule *root) {
    NSArray *parts=root->parts;
    NSMutableIndexSet *remaining=[NSMutableIndexSet indexSet],*done=[NSMutableIndexSet indexSet],*hosts=[NSMutableIndexSet indexSet];
    for(struct MetalSchedule *s=root;s;s=s->next) {
        [s->waits setLength:0];
        if(s->commands)[remaining addIndex:s->ordinal];else [done addIndex:s->ordinal];
    }
    root->firstSubmission=root->lastSubmission=NULL;
    while(remaining.count) {
        __block NSUInteger selected=NSNotFound,cost=NSNotFound;
        [remaining enumerateIndexesUsingBlock:^(NSUInteger i,BOOL *stop){
            struct MetalSchedule *s=[parts[i] pointerValue];
            if(![done containsIndexes:s->partDependencies])return;
            NSMutableIndexSet *needed=[[s->hostDependencies mutableCopy] autorelease];[needed removeIndexes:hosts];
            NSUInteger waits=needed.count;
            if(waits<cost){selected=i;cost=waits;if(!cost)*stop=YES;}
        }];
        if(selected==NSNotFound)[NSException raise:NSInternalInconsistencyException format:@"cyclic prepared command parts"];
        struct MetalSchedule *s=[parts[selected] pointerValue];
        NSMutableIndexSet *needed=[[s->hostDependencies mutableCopy] autorelease];[needed removeIndexes:hosts];
        NSIndexSet *all=[[needed copy] autorelease];
        [all enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){(void)stop;[needed removeIndexes:((MetalWork *)root->history[p])->hostDependencies];}];
        [needed enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){
            (void)stop;MetalWork *producer=root->history[p];
            const struct MetalScheduleWait wait={producer->event,producer->eventValue};
            [s->waits appendBytes:&wait length:sizeof wait];
        }];
        [hosts addIndexes:s->hostDependencies];
        [done addIndex:selected];[remaining removeIndex:selected];
        s->submit=NULL;
        if(root->lastSubmission)root->lastSubmission->submit=s;else root->firstSubmission=s;
        root->lastSubmission=s;
    }
}

// Carry exact outstanding page relations into the next prepared function on
// this queue. Compact the frontier's node identities, retaining only publication
// operands, so the queue never retains a completed function or its allocations.
static void MetalSchedulePublishFrontier(struct MetalSchedule *root) {
    NSMutableIndexSet *live=[[MetalScheduleFrontierNodes(root) mutableCopy] autorelease];
    for(NSDictionary *resource in root->hostResources.allValues) {
        NSArray *uses=resource[@"uses"];const NSUInteger count=MIN(uses.count,[resource[@"capacity"] unsignedIntegerValue]);
        for(NSUInteger i=uses.count-count;i<uses.count;i++)[live addIndex:[uses[i] unsignedIntegerValue]];
    }
    NSMutableDictionary *indices=[NSMutableDictionary dictionary];
    NSMutableArray *producers=[NSMutableArray arrayWithCapacity:live.count];
    [live enumerateIndexesUsingBlock:^(NSUInteger p,BOOL *stop){
        (void)stop;MetalWork *source=root->history[p],*producer=[MetalWork new];
        producer->external=YES;producer->ordinal=producers.count;
        producer->eventValue=source->eventValue;producer->fence=[source->fence retain];
        producer->event=[(source->event?:MetalSchedulePublication(source->part)) retain];
        producer->hostDependencies=[NSMutableIndexSet new];producer->gpuDependencies=[NSMutableIndexSet new];
        indices[@(p)]=@(producers.count);[producers addObject:producer];[producer release];
    }];
    MetalQueueAccess *queue=root->queueAccesses;
    [queue->read release];queue->read=[MetalFrontierCopy(root->frontierRead,indices) retain];
    [queue->written release];queue->written=[MetalFrontierCopy(root->frontierWritten,indices) retain];
    [queue->atomic release];queue->atomic=[MetalFrontierCopy(root->frontierAtomic,indices) retain];
    [queue->producers release];queue->producers=[producers copy];
    [queue->resources removeAllObjects];
    for(id key in root->hostResources) {
        NSDictionary *resource=root->hostResources[key];NSArray *uses=resource[@"uses"];
        const NSUInteger count=MIN(uses.count,[resource[@"capacity"] unsignedIntegerValue]);
        NSMutableArray *kept=[NSMutableArray arrayWithCapacity:count];
        for(NSUInteger i=uses.count-count;i<uses.count;i++)[kept addObject:indices[uses[i]]];
        queue->resources[key]=@{@"capacity":resource[@"capacity"],@"uses":kept};
    }
}

int MetalScheduleFinish(struct MetalSchedule *root,void (^completed)(NSError *)) {
    // Functions may be declared by independent preparation workers. Import and
    // publish exact queue relations in one preparation transaction; no lock or
    // frontier discovery enters submission or numerical execution.
    @synchronized(root->queueAccesses) {
        MetalQueueAccess *previous=root->queueAccesses;
        NSMutableDictionary *active=[NSMutableDictionary dictionary];
        for(NSUInteger i=0;i<previous->producers.count;i++) {
            MetalWork *producer=previous->producers[i];
            const BOOL shared=[producer->event conformsToProtocol:@protocol(MTLSharedEvent)];
            if(shared&&producer->event.signaledValue>=producer->eventValue)continue;
            active[@(i)]=@(root->history.count);[root->history addObject:producer];
        }
        [root->frontierRead addEntriesFromDictionary:MetalFrontierCopy(previous->read,active)];
        [root->frontierWritten addEntriesFromDictionary:MetalFrontierCopy(previous->written,active)];
        [root->frontierAtomic addEntriesFromDictionary:MetalFrontierCopy(previous->atomic,active)];
        for(id key in previous->resources) {
            NSDictionary *resource=previous->resources[key];
            NSMutableArray *uses=[NSMutableArray array];
            for(NSNumber *source in resource[@"uses"])if(active[source])[uses addObject:active[source]];
            root->hostResources[key]=[[@{@"capacity":resource[@"capacity"],@"uses":uses} mutableCopy] autorelease];
        }
        MetalScheduleFlush(root);
        [root->completed release];root->completed=[completed copy];
        MetalScheduleFinishPart(MetalScheduleTail(root));
        MetalScheduleOrder(root);
        MetalSchedulePublishFrontier(root);
        for(NSValue *value in root->communicators) {
            const ncclResult_t error=ncclMeshCommRealized(value.pointerValue);
            if(error)return error;
        }
    }
    return 0;
}

int MetalScheduleEnd(const struct MetalSchedule *s,double *ended,NSError **error) {
    struct MetalSchedule *root=s->root;
    *error=nil;
    while(!atomic_load_explicit(&root->arrived,memory_order_acquire)) {
        // woken by the last arrival; each second without it, its communicators' errors read (a lost link's GPU waits
        // end at their own bounds, and with them the parts)
        os_sync_wait_on_address_with_timeout(&root->arrived,0,sizeof root->arrived,OS_SYNC_WAIT_ON_ADDRESS_NONE,
                                             OS_CLOCK_MACH_ABSOLUTE_TIME,1000000000ull);
        if(MetalScheduleStatus(s,error)<0)return -1;
    }
    double latest=0;
    for(const struct MetalSchedule *part=s;part;part=part->next)if(part->ended>latest)latest=part->ended;
    for(MetalWork *host in root->hosts)if(host->ended>latest)latest=host->ended;
    *ended=latest;
    return MetalScheduleStatus(s,error);
}

int MetalScheduleStatus(const struct MetalSchedule *s,NSError **error) {
    *error=nil;
    const struct MetalSchedule *root=s->root;
    // A part publishes its status before its arrival. Completion polling must
    // observe the final arrival too, so reading the reported end cannot wait.
    int result=atomic_load_explicit(&root->arrived,memory_order_acquire)?1:0;
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

void MetalScheduleEnqueue(struct MetalSchedule *s) {
    struct MetalSchedule *root=s;
    // One reference spans submission, another spans all native/GPU results.
    atomic_fetch_add_explicit(&root->references,2,memory_order_relaxed);
    s=root->firstSubmission;
    while(s) {
        // Completion may resume the owner and release the whole function
        // before commit returns. Read the compiled successor before commit.
        struct MetalSchedule *next=s->submit;
        atomic_fetch_add_explicit(&s->references,1,memory_order_relaxed);
        const struct MetalScheduleWait *waits=s->waits.bytes;
        if(getenv("MESH_SCHEDULE_TRACE")) {
            fprintf(stderr,"schedule %.3f part %lu: %lu commands, %lu waits:",clock_gettime_nsec_np(CLOCK_REALTIME)/1e9,(unsigned long)s->ordinal,(unsigned long)s->commands,
                    (unsigned long)(s->waits.length/sizeof *waits));
            for(NSUInteger i=0;i<s->waits.length/sizeof *waits;i++)
                fprintf(stderr," [%p needs %llu, at %llu]",(void *)waits[i].event,(unsigned long long)waits[i].value,
                        [(id)waits[i].event respondsToSelector:@selector(signaledValue)]?(unsigned long long)((id<MTLSharedEvent>)waits[i].event).signaledValue:~0ull);
            fprintf(stderr,"\n");
        }
        for(NSUInteger i=0;i<s->waits.length/sizeof *waits;i++)[s->queue waitForEvent:waits[i].event value:waits[i].value];
        [s->queue commit:&s->command count:1 options:s->options];
        if(s->event)[s->queue signalEvent:s->event value:1];
        s=next;
    }
    root->start.signaledValue=1;
    // The initial pending count covers submission itself, including a function
    // with only native host work or no commands. Empty GPU parts are not sent.
    MetalScheduleArrived(root);
    MetalScheduleFree(root);
}

void MetalScheduleFree(struct MetalSchedule *s) {
    if(!s)return;
    if(atomic_fetch_sub_explicit(&s->references,1,memory_order_acq_rel)!=1)return;
    MetalScheduleFree(s->next);
    if(!s->finished&&s->encoder) {
        [s->encoder endEncoding];
        [s->command endCommandBuffer];
    }
    [s->encoder release];[s->command release];[s->queue release];[s->allocator release];[s->table release];[s->data release];
    [s->residency release];[s->owners release];[s->replayResidency release];[s->options release];
    [s->event release];[s->start release];[s->listener release];[s->work release];[s->hosts release];[s->waits release];
    [s->declarations release];
    [s->hostDependencies release];[s->partDependencies release];
    [s->history release];[s->readVersions release];[s->frontierRead release];[s->frontierWritten release];[s->frontierAtomic release];[s->hostResources release];
    [s->queueAccesses release];[s->parts release];
    [s->completed release];[s->error release];[s->device release];[s->communicators release];
    [s->script release];[s->plans release];free(s);
}

void MetalReplayFree(struct MetalReplay *replay) {
    if (!replay) return;
    MetalReplayDestroy(replay);
    free(replay);
}

// design/replay-templates-2026-10-29.md: templates

static char pipelineDeclarationKey;

void MetalPipelineDeclare(id<MTLComputePipelineState> pipeline,id<MTLLibrary> library,NSString *function,
    NSArray<NSArray *> *constants,NSDictionary<NSString *,id> *descriptor) {
    objc_setAssociatedObject(pipeline,&pipelineDeclarationKey,@{@"library":library,@"function":function,
        @"constants":constants?:@[],@"descriptor":descriptor?:@{}},OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

static MTLComputePipelineDescriptor *MetalPipelineDescriptor(id<MTLLibrary> library,NSString *function,NSArray *constants,
    NSDictionary *fields,NSError **error) {
    id<MTLFunction> made=nil;
    if(constants.count) {
        MTLFunctionConstantValues *values=[[MTLFunctionConstantValues new] autorelease];
        for(NSArray *constant in constants)
            [values setConstantValue:[constant[2] bytes] type:(MTLDataType)[constant[1] unsignedIntegerValue] atIndex:[constant[0] unsignedIntegerValue]];
        made=[library newFunctionWithName:function constantValues:values error:error];
    } else made=[library newFunctionWithName:function];
    if(!made) {
        if(error&&!*error)*error=MetalReplayFailure(ENOENT,[NSString stringWithFormat:@"no function %@",function]);
        return nil;
    }
    MTLComputePipelineDescriptor *descriptor=[[MTLComputePipelineDescriptor new] autorelease];
    descriptor.computeFunction=made;[made release];
    if(fields[@"label"])descriptor.label=fields[@"label"];
    if(fields[@"threads"])descriptor.maxTotalThreadsPerThreadgroup=[fields[@"threads"] unsignedIntegerValue];
    descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth=[fields[@"multiple"] boolValue];
    descriptor.supportIndirectCommandBuffers=[fields[@"indirect"] boolValue];
    return descriptor;
}

id<MTLComputePipelineState> MetalPipelineMake(id<MTLLibrary> library,NSString *function,NSArray<NSArray *> *constants,
    NSDictionary<NSString *,id> *fields,NSArray<id<MTLBinaryArchive>> *archives,NSError **error) {
    MTLComputePipelineDescriptor *descriptor=MetalPipelineDescriptor(library,function,constants,fields,error);
    if(!descriptor)return nil;
    if(archives.count)descriptor.binaryArchives=archives;
    if(!descriptor.label)descriptor.label=function;   // the kernel a refusal names (MetalPipelineName)
    MTLComputePipelineReflection *reflection=nil;
    id<MTLComputePipelineState> pipeline=[library.device newComputePipelineStateWithDescriptor:descriptor
        options:MTLPipelineOptionBindingInfo|MTLPipelineOptionBufferTypeInfo reflection:&reflection error:error];
    if(pipeline){MetalPipelineDeclare(pipeline,library,function,constants,fields);MetalPipelineSizes(pipeline,reflection);}
    return pipeline;
}

// A template names operands by index: a binding or reach its operand (or, for a binding, the command data a data call
// made), each beside its recorded form with the pointers cleared; -1 where it names none.
struct MetalTemplateBinding { int32_t operand, data; struct MetalCommandBinding binding; };
struct MetalTemplateDispatch { uint32_t pipeline; int32_t operands[MESH_BUFFER_BINDINGS], indirectGrid; NSUInteger address; struct MetalRecordedDispatch record; };
struct MetalTemplateReach { int32_t operand, address; struct MetalReach reach; };
struct MetalTemplateAddress { int32_t operand, uniform; struct MetalRecordedAddress record; };

static NSArray *MetalTemplateSizes(void) {
    return @[@(sizeof(struct MetalTemplateDispatch)),@(sizeof(struct MetalTemplateReach)),@(sizeof(struct MetalTemplateBinding)),
        @(sizeof(struct MetalTemplateAddress))];
}

// A buffer's operand is its region lease (ncclMeshMemBuffer), so buffers over one lease's bytes are one operand;
// `delta` is where the buffer starts in it. A buffer outside every lease is its own.
static id<MTLBuffer> MetalTemplateLease(id<MTLBuffer> buffer,NSUInteger *delta) {
    void *lease=NULL;size_t offset=0;*delta=0;
    if(!buffer.contents||ncclMeshMemBuffer(buffer.contents,&lease,&offset))return buffer;
    *delta=offset;return (id<MTLBuffer>)lease;
}

// The compile facts a device's libraries take (MetalDeviceCompilerOptions)
static NSDictionary *MetalTemplateDevice(id<MTLDevice> device) {
    return @{@"threads":@(device.maxThreadsPerThreadgroup.width),@"bytes":@(device.maxThreadgroupMemoryLength)};
}

NSDictionary *MetalScheduleTemplate(struct MetalSchedule *const *schedules,size_t count,NSArray<id<MTLBuffer>> *bound,
    NSArray<NSNumber *> *offsets,NSError **error) {
    if(!count){if(error)*error=MetalReplayFailure(EINVAL,@"a template of no function");return nil;}
    struct MetalSchedule *root=schedules[0]->root;
    NSMutableArray *operands=[NSMutableArray array],*pipelines=[NSMutableArray array],*libraries=[NSMutableArray array],
        *functions=[NSMutableArray array];
    NSMutableDictionary *operandAt=[NSMutableDictionary dictionary],*pipelineAt=[NSMutableDictionary dictionary],
        *libraryAt=[NSMutableDictionary dictionary],*dataAt=[NSMutableDictionary dictionary];
    // a bound operand's offsets are its lease's; where its buffer starts in the lease (delta) moves them to the buffer
    // that instantiation binds in its place
    for(NSUInteger i=0;i<bound.count;i++) {
        NSUInteger delta=0;NSValue *key=[NSValue valueWithPointer:(void *)MetalTemplateLease(bound[i],&delta)];
        if(!operandAt[key])operandAt[key]=@(operands.count);
        [operands addObject:@{@"bound":@(operands.count),@"delta":@(delta+[offsets[i] unsignedIntegerValue])}];
    }
    __block NSString *failure=nil;
    int32_t (^operand)(id<MTLBuffer>,NSUInteger *)=^int32_t(id<MTLBuffer> buffer,NSUInteger *delta){
        *delta=0;
        if(!buffer)return -1;
        id<MTLBuffer> lease=MetalTemplateLease(buffer,delta);
        NSValue *key=[NSValue valueWithPointer:(void *)lease];
        NSNumber *at=operandAt[key];
        if(at)return at.intValue;
        if(!lease.contents){failure=@"a kept operand is not in shared storage";return -1;}
        operandAt[key]=at=@(operands.count);
        [operands addObject:@{@"length":@(lease.length),@"contents":[NSMutableData dataWithBytes:lease.contents length:lease.length]}];
        return at.intValue;
    };
    uint32_t (^pipelineIndex)(id)=^uint32_t(id pipeline){
        NSValue *key=[NSValue valueWithPointer:(void *)pipeline];
        NSNumber *at=pipelineAt[key];
        if(at)return at.unsignedIntValue;
        NSDictionary *declared=objc_getAssociatedObject(pipeline,&pipelineDeclarationKey);
        if(!declared){failure=[NSString stringWithFormat:@"pipeline %@ has no declaration",[pipeline label]?:@"(unlabelled)"];return 0;}
        NSValue *library=[NSValue valueWithPointer:(void *)declared[@"library"]];
        if(!libraryAt[library]){libraryAt[library]=@(libraries.count);[libraries addObject:declared[@"library"]];}
        pipelineAt[key]=at=@(pipelines.count);
        [pipelines addObject:@{@"library":libraryAt[library],@"function":declared[@"function"],
            @"constants":declared[@"constants"],@"descriptor":declared[@"descriptor"],
            @"threads":@([pipeline maxTotalThreadsPerThreadgroup])}];
        return at.unsignedIntValue;
    };
    // a reach's operand: its buffer's lease, or (a host view: no buffer, its address its offset) its address's
    struct MetalTemplateReach (^placed)(struct MetalReach)=^struct MetalTemplateReach(struct MetalReach view){
        struct MetalTemplateReach made={.operand=-1,.address=0,.reach=view};made.reach.buffer=nil;
        NSUInteger delta=0;
        if(view.buffer){made.operand=operand(view.buffer,&delta);made.reach.offset+=delta;}
        else if(view.offset) {
            void *lease=NULL;size_t at=0;
            if(ncclMeshMemBuffer((const void *)view.offset,&lease,&at)){failure=@"a host view outside every lease";return made;}
            made.operand=operand((id<MTLBuffer>)lease,&delta);made.address=1;made.reach.offset=at+delta;
        }
        return made;
    };
    // a declaration's places as (place, relocation, operand, offset in the operand), cleared in its bytes
    NSMutableArray *communicators=[NSMutableArray array];
    NSDictionary *(^declared)(NSDictionary *)=^NSDictionary *(NSDictionary *call){
        NSMutableData *bytes=[[call[@"bytes"] mutableCopy] autorelease];NSData *listed=call[@"places"];
        const struct MetalPlace *places=listed.bytes;NSMutableArray *relocations=[NSMutableArray array];
        for(NSUInteger k=0;k<listed.length/sizeof *places;k++) {
            const BOOL buffer=places[k].relocation==MetalRelocationBuffer;
            if(places[k].at+sizeof(void *)+(buffer?sizeof(size_t):0)>bytes.length){failure=@"a declared place past its bytes";break;}
            char *field=(char *)bytes.mutableBytes+places[k].at;
            NSUInteger delta=0;int32_t at=-1;size_t offset=0;
            if(places[k].relocation==MetalRelocationCommunicator) {
                void *comm=NULL;memcpy(&comm,field,sizeof comm);
                if(!comm)continue;
                NSValue *key=[NSValue valueWithPointer:comm];NSUInteger index=[communicators indexOfObject:key];
                if(index==NSNotFound){index=communicators.count;[communicators addObject:key];}
                at=(int32_t)index;memset(field,0,sizeof comm);
            } else if(!buffer) {
                void *address=NULL,*lease=NULL;memcpy(&address,field,sizeof address);
                if(!address)continue;
                if(ncclMeshMemBuffer(address,&lease,&offset)){failure=@"a declared address outside every lease";break;}
                at=operand((id<MTLBuffer>)lease,&delta);memset(field,0,sizeof address);
            } else {
                id named=nil;memcpy(&named,field,sizeof named);memcpy(&offset,field+sizeof named,sizeof offset);
                if(!named)continue;
                at=operand(named,&delta);memset(field,0,sizeof named+sizeof offset);
            }
            [relocations addObject:@[@(places[k].at),@(places[k].relocation),@(at),@(offset+delta)]];
        }
        return @{@"kind":call[@"kind"],@"bytes":bytes,@"relocations":relocations};
    };
    // a plan is the program's: emitted where it differs from the last one emitted for its communicator
    NSMutableDictionary *planned=[NSMutableDictionary dictionary];
    for(size_t f=0;f<count&&!failure;f++) {
    struct MetalSchedule *function=schedules[f]->root;
    NSMutableArray *script=[NSMutableArray array];
    [functions addObject:script];[dataAt removeAllObjects];
    for(NSDictionary *call in function->script) {
        NSString *kind=call[@"call"];
        if([kind isEqual:@"data"]) {
            dataAt[call[@"at"]]=@(dataAt.count);
            [script addObject:@{@"call":kind,@"bytes":call[@"bytes"],@"index":call[@"index"]}];
        } else if([kind isEqual:@"compute"]) {
            NSData *bindings=call[@"bindings"];
            const struct MetalCommandBinding *source=bindings.bytes;const NSUInteger n=bindings.length/sizeof *source;
            NSMutableData *made=[NSMutableData dataWithLength:n*sizeof(struct MetalTemplateBinding)];
            struct MetalTemplateBinding *target=made.mutableBytes;
            for(NSUInteger k=0;k<n;k++) {
                NSNumber *data=dataAt[[NSString stringWithFormat:@"%p+%lu",(void *)source[k].buffer,(unsigned long)source[k].offset]];
                NSUInteger delta=0;
                target[k]=(struct MetalTemplateBinding){.operand=data?-1:operand(source[k].buffer,&delta),.data=data?data.intValue:-1,.binding=source[k]};
                target[k].binding.buffer=nil;target[k].binding.offset+=delta;
            }
            [script addObject:@{@"call":kind,@"pipeline":@(pipelineIndex(call[@"pipeline"])),@"bindings":made,
                @"groups":call[@"groups"],@"threads":call[@"threads"]}];
        } else if([kind isEqual:@"append"]) {
            id owner=call[@"owner"];
            NSArray *held=objc_getAssociatedObject(owner,&replayRecordsKey);
            if([objc_getAssociatedObject(owner,&replayHostsKey) count]){failure=@"a recorded host operation has no declaration";break;}
            if([call[@"marks"] count]){failure=@"a replay's crossing has no declaration";break;}
            NSData *records=held[0],*reaches=held[3];
            const struct MetalRecordedDispatch *commands=records.bytes;const NSUInteger n=records.length/sizeof *commands;
            NSMutableData *made=[NSMutableData dataWithLength:n*sizeof(struct MetalTemplateDispatch)];
            struct MetalTemplateDispatch *target=made.mutableBytes;
            NSMutableData *addresses=[NSMutableData data];
            NSUInteger deltas[MESH_BUFFER_BINDINGS];
            const struct MetalReach *views=reaches.bytes;
            NSMutableData *spans=[NSMutableData dataWithLength:reaches.length/sizeof *views*sizeof(struct MetalTemplateReach)];
            struct MetalTemplateReach *into=spans.mutableBytes;
            for(NSUInteger i=0;i<n;i++) {
                target[i].record=commands[i];target[i].record.pipeline=nil;
                target[i].record.addresses=NULL;target[i].address=addresses.length/sizeof(struct MetalTemplateAddress);
                target[i].pipeline=pipelineIndex(commands[i].pipeline);
                target[i].indirectGrid=-1;target[i].record.indirectGrid=nil;
                if(commands[i].indirectGrid) {
                    NSUInteger delta=0;target[i].indirectGrid=operand(commands[i].indirectGrid,&delta);
                    target[i].record.indirectOffset+=delta;
                }
                for(NSUInteger a=0;a<commands[i].addressCount;a++) {
                    const struct MetalRecordedAddress *source=commands[i].addresses+a;
                    struct MetalTemplateAddress address={.record=*source};
                    NSUInteger delta=0;
                    address.operand=operand(source->value.buffer,&delta);
                    address.record.value.buffer=nil;address.record.value.origin+=delta;
                    address.uniform=operand(source->uniform,&delta);
                    address.record.uniform=nil;address.record.uniformOffset+=delta;
                    // Only symbolic targets belong in a template. Clear the
                    // declared pointer words in its saved constant pages,
                    // preserving every scalar and the live source allocation.
                    NSMutableData *contents=address.uniform<0?nil:operands[address.uniform][@"contents"];
                    if(contents) {
                        const NSUInteger repetitions=[held[4] unsignedIntegerValue];
                        const NSUInteger slots=!source->uniformStride?1:source->uniformInstances?
                            MIN(repetitions,source->uniformInstances):repetitions;
                        for(NSUInteger t=0;t<slots;t++) {
                            const NSUInteger at=MetalInvocationOffset(address.record.uniformOffset,source->uniformStride,
                                source->uniformInstances,t,source->uniformInvocation)+source->value.offset;
                            memset((char *)contents.mutableBytes+at,0,sizeof(uint64_t));
                        }
                    }
                    [addresses appendBytes:&address length:sizeof address];
                }
                for(NSUInteger j=0;j<MESH_BUFFER_BINDINGS;j++) {
                    id<MTLBuffer> buffer=commands[i].buffers[j];
                    for(NSUInteger a=0;a<commands[i].addressCount;a++)if(commands[i].addresses[a].value.binding==j) {
                        const struct MetalRecordedAddress *address=commands[i].addresses+a;
                        buffer=address->uniform;target[i].record.offsets[j]=address->uniformOffset;
                        target[i].record.lengths[j]=address->uniformLength;target[i].record.strides[j]=address->uniformStride;
                        target[i].record.instances[j]=address->uniformInstances;target[i].record.invocations[j]=address->uniformInvocation;
                        break;
                    }
                    target[i].operands[j]=operand(buffer,deltas+j);
                    target[i].record.buffers[j]=nil;target[i].record.offsets[j]+=deltas[j];
                }
                for(uint32_t k=0;k<commands[i].reaches;k++) {
                    struct MetalReach reach=views[commands[i].reach+k];
                    if((reach.access&MetalReachBinding)&&!(reach.access&MetalReachAddress))for(NSUInteger a=0;a<commands[i].addressCount;a++)
                        if((int32_t)commands[i].addresses[a].value.binding==reach.binding) {
                            reach.buffer=commands[i].addresses[a].uniform;
                            reach.offset+=commands[i].addresses[a].uniformOffset;break;
                        }
                    into[commands[i].reach+k]=placed(reach);
                }
            }
            [script addObject:@{@"call":kind,@"records":made,@"reaches":spans,@"addresses":addresses,@"form":held[2],@"repetitions":held[4],
                @"buffers":call[@"buffers"],@"memory":call[@"memory"],@"invocations":call[@"invocations"]}];
        } else if(call[@"kind"]&&([kind isEqual:@"host"]||[kind isEqual:@"cross"]||[kind isEqual:@"plan"])) {
            NSMutableDictionary *entry=[[declared(call) mutableCopy] autorelease];entry[@"call"]=kind;
            if([kind isEqual:@"plan"]) {
                NSNumber *comm=[entry[@"relocations"] firstObject][2]?:@(-1);
                if([planned[comm] isEqual:entry])continue;
                planned[comm]=entry;
            }
            if([kind isEqual:@"host"]||[kind isEqual:@"cross"]) {
                NSData *views=call[@"accesses"];const struct MetalReach *source=views.bytes;const NSUInteger n=views.length/sizeof *source;
                NSMutableData *made=[NSMutableData dataWithLength:n*sizeof(struct MetalTemplateReach)];
                struct MetalTemplateReach *into=made.mutableBytes;
                for(NSUInteger k=0;k<n;k++)into[k]=placed(source[k]);
                entry[@"accesses"]=made;
            }
            [script addObject:entry];
        } else failure=[NSString stringWithFormat:@"a %@ call has no declaration",kind];
        if(failure)break;
    }
    }
    if(failure){if(error)*error=MetalReplayFailure(EINVAL,failure);return nil;}
    return @{@"format":@2,@"form":@(root->form),@"sizes":MetalTemplateSizes(),@"device":MetalTemplateDevice(root->device),
        @"libraries":libraries,@"pipelines":pipelines,@"operands":operands,@"communicators":@(communicators.count),
        @"functions":functions};
}

// The pipelines of the device its libraries are on, as pipelines.<architecture>.metallib in `directory`
static NSError *MetalTemplateWriteArchive(NSDictionary *made,NSString *directory) {
    NSArray *libraries=made[@"libraries"];NSError *error=nil;
    if(!libraries.count)return nil;
    id<MTLDevice> device=[libraries[0] device];
    id<MTLBinaryArchive> archive=[[device newBinaryArchiveWithDescriptor:[[MTLBinaryArchiveDescriptor new] autorelease] error:&error] autorelease];
    if(!archive)return error;
    for(NSDictionary *pipeline in made[@"pipelines"]) {
        MTLComputePipelineDescriptor *descriptor=MetalPipelineDescriptor(libraries[[pipeline[@"library"] unsignedIntegerValue]],
            pipeline[@"function"],pipeline[@"constants"],pipeline[@"descriptor"],&error);
        if(!descriptor||![archive addComputePipelineFunctionsWithDescriptor:descriptor error:&error])return error;
    }
    NSString *name=[NSString stringWithFormat:@"pipelines.%@.metallib",device.architecture.name];
    if(![archive serializeToURL:[NSURL fileURLWithPath:[directory stringByAppendingPathComponent:name]] error:&error])return error;
    return nil;
}

NSError *MetalTemplateArchive(NSDictionary *made,NSString *directory) {
    return made[@"archive"]?nil:MetalTemplateWriteArchive(made,directory);
}

NSError *MetalTemplateWrite(NSDictionary *made,NSString *directory) {
    NSError *error=nil;
    if(![NSFileManager.defaultManager createDirectoryAtPath:directory withIntermediateDirectories:YES attributes:nil error:&error])return error;
    NSArray *libraries=made[@"libraries"];NSMutableArray *names=[NSMutableArray array];
    for(NSUInteger i=0;i<libraries.count;i++) {
        id library=libraries[i];NSString *name=[NSString stringWithFormat:@"library.%lu.metallib",(unsigned long)i];
        if(![library respondsToSelector:@selector(serializeToURL:error:)])return MetalReplayFailure(ENOTSUP,@"this system's Metal libraries cannot be written");
        if(![library serializeToURL:[NSURL fileURLWithPath:[directory stringByAppendingPathComponent:name]] error:&error])return error;
        [names addObject:name];
    }
    if((error=MetalTemplateWriteArchive(made,directory)))return error;
    NSMutableDictionary *listed=[[made mutableCopy] autorelease];
    listed[@"libraries"]=names;[listed removeObjectForKey:@"archive"];
    NSData *data=[NSPropertyListSerialization dataWithPropertyList:listed format:NSPropertyListBinaryFormat_v1_0 options:0 error:&error];
    if(!data||![data writeToFile:[directory stringByAppendingPathComponent:@"template.plist"] options:NSDataWritingAtomic error:&error])return error;
    return nil;
}

NSDictionary *MetalTemplateRead(id<MTLDevice> device,NSString *directory,NSError **error) {
    NSData *data=[NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"template.plist"] options:0 error:error];
    NSDictionary *listed=data?[NSPropertyListSerialization propertyListWithData:data options:NSPropertyListImmutable format:NULL error:error]:nil;
    if(!listed)return nil;
    if(![listed isKindOfClass:NSDictionary.class]||![listed[@"format"] isEqual:@2]||![listed[@"sizes"] isEqual:MetalTemplateSizes()]) {
        if(error)*error=MetalReplayFailure(EINVAL,[NSString stringWithFormat:@"%@ was not written by this recorder's build",directory]);
        return nil;
    }
    NSMutableDictionary *made=[[listed mutableCopy] autorelease];NSMutableArray *libraries=[NSMutableArray array];
    for(NSString *name in listed[@"libraries"]) {
        id<MTLLibrary> library=[device newLibraryWithURL:[NSURL fileURLWithPath:[directory stringByAppendingPathComponent:name]] error:error];
        if(!library)return nil;
        [libraries addObject:library];[library release];
    }
    made[@"libraries"]=libraries;
    NSString *archived=[directory stringByAppendingPathComponent:[NSString stringWithFormat:@"pipelines.%@.metallib",device.architecture.name]];
    if([NSFileManager.defaultManager fileExistsAtPath:archived]) {
        MTLBinaryArchiveDescriptor *descriptor=[[MTLBinaryArchiveDescriptor new] autorelease];
        descriptor.url=[NSURL fileURLWithPath:archived];
        id<MTLBinaryArchive> archive=[device newBinaryArchiveWithDescriptor:descriptor error:error];
        if(!archive)return nil;
        made[@"archive"]=archive;[archive release];
    }
    return made;
}

// A replay an instantiated function made, freed with it
@interface MetalTemplateReplay : NSObject { @public struct MetalReplay *replay; }
@end
@implementation MetalTemplateReplay
- (void)dealloc { MetalReplayFree(replay); [super dealloc]; }
@end

// A template reach on the instance's operands: a buffer's offset, or a host view's address
static struct MetalReach MetalTemplateReachOf(struct MetalTemplateReach placed,NSArray *buffers,const NSInteger *shifts) {
    struct MetalReach reach=placed.reach;
    if(placed.operand<0)return reach;
    id<MTLBuffer> buffer=buffers[(NSUInteger)placed.operand];
    const NSUInteger offset=(NSUInteger)((NSInteger)reach.offset+shifts[placed.operand]);
    if(placed.address){reach.buffer=nil;reach.offset=(NSUInteger)buffer.contents+offset;}
    else {reach.buffer=buffer;reach.offset=offset;}
    return reach;
}

// A declaration's bytes with its places rewritten to the instance's operands; `places` gets those places
static NSData *MetalTemplatePlaced(NSDictionary *declared,NSArray *buffers,const NSInteger *shifts,NSArray<NSValue *> *communicators,
    NSMutableData *places) {
    NSMutableData *bytes=[[declared[@"bytes"] mutableCopy] autorelease];
    for(NSArray *relocation in declared[@"relocations"]) {
        const struct MetalPlace place={[relocation[0] unsignedLongValue],(enum MetalRelocation)[relocation[1] intValue]};
        const NSUInteger at=[relocation[2] unsignedIntegerValue];
        if(place.relocation==MetalRelocationCommunicator) {
            void *comm=[communicators[at] pointerValue];
            memcpy((char *)bytes.mutableBytes+place.at,&comm,sizeof comm);
            [places appendBytes:&place length:sizeof place];
            continue;
        }
        const size_t offset=(size_t)((NSInteger)[relocation[3] unsignedLongValue]+shifts[at]);
        id<MTLBuffer> buffer=buffers[at];char *field=(char *)bytes.mutableBytes+place.at;
        if(place.relocation==MetalRelocationAddress){void *address=(char *)buffer.contents+offset;memcpy(field,&address,sizeof address);}
        else {memcpy(field,&buffer,sizeof buffer);memcpy(field+sizeof buffer,&offset,sizeof offset);}
        [places appendBytes:&place length:sizeof place];
    }
    return bytes;
}

// A template's replay, built as MetalReplayCommands builds one: its records' pipelines and operands resolved, its
// order compiled anew (MetalDataflow)
static struct MetalReplay *MetalTemplateReplayMake(id<MTLDevice> device,NSDictionary *call,NSArray *pipelines,NSArray *buffers,
    const NSInteger *shifts) {
    MetalCapture *capture=[[MetalCapture new] autorelease];
    NSData *records=call[@"records"],*reaches=call[@"reaches"];
    NSData *addresses=call[@"addresses"];
    const struct MetalTemplateAddress *listed=addresses.bytes;
    NSMutableData *addressData=[NSMutableData dataWithLength:addresses.length/sizeof *listed*sizeof(struct MetalRecordedAddress)];
    struct MetalRecordedAddress *boundAddresses=addressData.mutableBytes;
    for(NSUInteger a=0;a<addresses.length/sizeof *listed;a++) {
        struct MetalRecordedAddress address=listed[a].record;
        address.value.buffer=buffers[listed[a].operand];address.value.origin+=shifts[listed[a].operand];
        address.uniform=buffers[listed[a].uniform];address.uniformOffset+=shifts[listed[a].uniform];
        boundAddresses[a]=address;
        [capture->resources addObject:address.value.buffer];[capture->owners addObject:address.uniform];
    }
    if(addressData.length)[capture->addressRecords addObject:addressData];
    const struct MetalTemplateDispatch *source=records.bytes;
    for(NSUInteger i=0;i<records.length/sizeof *source;i++) {
        struct MetalRecordedDispatch command=source[i].record;
        command.addresses=command.addressCount?boundAddresses+source[i].address:NULL;
        command.pipeline=pipelines[source[i].pipeline];[capture->owners addObject:command.pipeline];
        if(source[i].indirectGrid>=0) {
            const NSUInteger at=(NSUInteger)source[i].indirectGrid;
            command.indirectGrid=buffers[at];command.indirectOffset+=shifts[at];
            [capture->resources addObject:command.indirectGrid];
        }
        for(NSUInteger j=0;j<MESH_BUFFER_BINDINGS;j++)if(source[i].operands[j]>=0) {
            const NSUInteger at=(NSUInteger)source[i].operands[j];
            command.buffers[j]=buffers[at];command.offsets[j]+=shifts[at];[capture->resources addObject:command.buffers[j]];
        }
        [capture->dispatches appendBytes:&command length:sizeof command];
    }
    const struct MetalTemplateReach *views=reaches.bytes;
    for(NSUInteger i=0;i<reaches.length/sizeof *views;i++) {
        const struct MetalReach reach=MetalTemplateReachOf(views[i],buffers,shifts);
        if(reach.buffer)[capture->resources addObject:reach.buffer];
        [capture->reaches appendBytes:&reach length:sizeof reach];
    }
    capture->buffers=[call[@"buffers"] unsignedIntegerValue];capture->memory=[call[@"memory"] unsignedIntegerValue];
    return MetalReplayMade(capture,device,[call[@"repetitions"] unsignedIntegerValue],(enum MetalForm)[call[@"form"] integerValue]);
}

NSError *MetalScheduleInstantiate(struct MetalSchedule *const *schedules,size_t count,NSDictionary *made,
    NSArray<id<MTLBuffer>> *bound,NSArray<NSNumber *> *offsets,NSArray<NSValue *> *communicators,
    NSArray<NSData *> *operators,MetalTemplateHost host,MetalTemplateCross cross) {
    NSArray *functions=made[@"functions"];
    if(!count||functions.count!=count)
        return MetalReplayFailure(EINVAL,[NSString stringWithFormat:@"the template holds %lu functions, %lu given",
            (unsigned long)functions.count,(unsigned long)count]);
    struct MetalSchedule *root=schedules[0]->root;
    @try {
        NSDictionary *facts=MetalTemplateDevice(root->device);
        if(![made[@"device"] isEqual:facts])
            return MetalReplayFailure(EINVAL,[NSString stringWithFormat:@"the template's kernels were compiled for threadgroups of %@ threads and "
                "%@ bytes; this device's are %@ and %@",made[@"device"][@"threads"],made[@"device"][@"bytes"],facts[@"threads"],facts[@"bytes"]]);
        if(communicators.count<[made[@"communicators"] unsignedIntegerValue])
            return MetalReplayFailure(EINVAL,[NSString stringWithFormat:@"the template names %@ communicators, %lu given",
                made[@"communicators"],(unsigned long)communicators.count]);
        NSArray *libraries=made[@"libraries"],*archives=made[@"archive"]?@[made[@"archive"]]:nil;
        NSMutableArray *pipelines=[NSMutableArray array];
        for(NSDictionary *declared in made[@"pipelines"]) {
            NSError *error=nil;
            id<MTLComputePipelineState> pipeline=MetalPipelineMake(libraries[[declared[@"library"] unsignedIntegerValue]],
                declared[@"function"],declared[@"constants"],declared[@"descriptor"],archives,&error);
            if(!pipeline)return error;
            [pipelines addObject:pipeline];[pipeline release];
        }
        // a dispatch's threadgroup was chosen for the recording's pipeline; this device's may take fewer threads
        NSString *(^fits)(NSUInteger,MTLSize)=^NSString *(NSUInteger p,MTLSize group){
            const NSUInteger threads=group.width*group.height*group.depth,most=[pipelines[p] maxTotalThreadsPerThreadgroup];
            return threads<=most?nil:[NSString stringWithFormat:@"pipeline %@ takes %lu threads a threadgroup on this device, its dispatch %lu",
                made[@"pipelines"][p][@"function"],(unsigned long)most,(unsigned long)threads];
        };
        NSMutableArray *buffers=[NSMutableArray array];
        NSMutableData *moved=[NSMutableData dataWithLength:[made[@"operands"] count]*sizeof(NSInteger)];
        NSInteger *shifts=moved.mutableBytes;
        for(NSDictionary *operand in made[@"operands"]) {
            if(operand[@"bound"]) {
                const NSUInteger at=[operand[@"bound"] unsignedIntegerValue];
                if(at>=bound.count)return MetalReplayFailure(EINVAL,[NSString stringWithFormat:@"the template binds operand %lu of %lu given",
                    (unsigned long)at,(unsigned long)bound.count]);
                NSUInteger delta=0;id<MTLBuffer> lease=MetalTemplateLease(bound[at],&delta);
                shifts[buffers.count]=(NSInteger)(delta+[offsets[at] unsignedIntegerValue])-[operand[@"delta"] integerValue];
                [buffers addObject:lease];
            } else {
                NSData *contents=operand[@"contents"];
                id<MTLBuffer> kept=MetalLeaseBuffer(root->device,contents.length);
                memcpy(kept.contents,contents.bytes,contents.length);
                for(size_t f=0;f<count;f++)[schedules[f]->root->owners addObject:kept];
                [buffers addObject:kept];[kept release];
            }
        }
        for(size_t f=0;f<count;f++) {
        struct MetalSchedule *s=schedules[f],*root=s->root;
        NSMutableArray *data=[NSMutableArray array];
        for(NSDictionary *call in functions[f]) {
            NSString *kind=call[@"call"];
            if([kind isEqual:@"data"]) {
                NSData *bytes=call[@"bytes"];
                const struct MetalCommandBinding binding=MetalScheduleData(s,bytes.bytes,bytes.length,[call[@"index"] unsignedIntValue]);
                [data addObject:[NSData dataWithBytes:&binding length:sizeof binding]];
            } else if([kind isEqual:@"compute"]) {
                NSData *bindings=call[@"bindings"];
                const struct MetalTemplateBinding *source=bindings.bytes;const NSUInteger n=bindings.length/sizeof *source;
                NSMutableData *resolved=[NSMutableData dataWithLength:n*sizeof(struct MetalCommandBinding)];
                struct MetalCommandBinding *target=resolved.mutableBytes;
                for(NSUInteger k=0;k<n;k++) {
                    if(source[k].data>=0)target[k]=*(const struct MetalCommandBinding *)[data[(NSUInteger)source[k].data] bytes];
                    else {
                        target[k]=source[k].binding;target[k].buffer=buffers[(NSUInteger)source[k].operand];
                        target[k].offset+=shifts[source[k].operand];
                    }
                }
                NSArray *groups=call[@"groups"],*threads=call[@"threads"];
                const MTLSize group=MTLSizeMake([threads[0] unsignedIntegerValue],[threads[1] unsignedIntegerValue],[threads[2] unsignedIntegerValue]);
                NSString *unfit=fits([call[@"pipeline"] unsignedIntegerValue],group);
                if(unfit)return MetalReplayFailure(EINVAL,unfit);
                MetalScheduleCompute(s,pipelines[[call[@"pipeline"] unsignedIntegerValue]],target,n,
                    MTLSizeMake([groups[0] unsignedIntegerValue],[groups[1] unsignedIntegerValue],[groups[2] unsignedIntegerValue]),group);
            } else if([kind isEqual:@"append"]) {
                NSData *records=call[@"records"];const struct MetalTemplateDispatch *recorded=records.bytes;
                for(NSUInteger i=0;i<records.length/sizeof *recorded;i++) {
                    NSString *unfit=fits(recorded[i].pipeline,recorded[i].record.group);
                    if(unfit)return MetalReplayFailure(EINVAL,unfit);
                }
                MetalTemplateReplay *held=[[MetalTemplateReplay new] autorelease];
                held->replay=MetalTemplateReplayMake(root->device,call,pipelines,buffers,shifts);
                [root->owners addObject:held];
                NSArray *range=call[@"invocations"];
                NSError *error=MetalScheduleAppend(s,held->replay,NSMakeRange([range[0] unsignedIntegerValue],[range[1] unsignedIntegerValue]),@[],nil,nil);
                if(error)return error;
            } else if([kind isEqual:@"plan"]) {
                // the communicator's plan declared again, its bindings the instance's: before the crossings that follow
                NSData *bytes=MetalTemplatePlaced(call,buffers,shifts,communicators,[NSMutableData data]);
                struct ncclComm *comm=NULL;memcpy(&comm,bytes.bytes,sizeof comm);
                const NSUInteger at=[communicators indexOfObject:[NSValue valueWithPointer:comm]];
                NSData *bindings=at<operators.count?operators[at]:nil;
                const ncclResult_t failed=ncclMeshCommPlanDeclared(comm,(const char *)bytes.bytes+sizeof comm,bytes.length-sizeof comm,
                    bindings.bytes,bindings.length/sizeof(ncclMeshRedOpBinding));
                if(failed)return MetalReplayFailure(failed,[NSString stringWithFormat:@"the template's plan: %s",ncclGetLastError(comm)]);
            } else if([kind isEqual:@"host"]||[kind isEqual:@"cross"]) {
                const BOOL hosted=[kind isEqual:@"host"];
                if(hosted?!host:!cross)
                    return MetalReplayFailure(EINVAL,[NSString stringWithFormat:@"a %@ %@ call, and no frontend to make it",call[@"kind"],kind]);
                NSMutableData *places=[NSMutableData data];
                NSData *bytes=MetalTemplatePlaced(call,buffers,shifts,communicators,places);
                const struct MetalDeclaration again={[call[@"kind"] UTF8String],bytes.bytes,bytes.length,places.bytes,
                    places.length/sizeof(struct MetalPlace)};
                NSError *error=nil;
                NSData *views=call[@"accesses"];
                if(!views)return MetalReplayFailure(EINVAL,@"a prepared call has no declared accesses");
                const struct MetalTemplateReach *source=views.bytes;const NSUInteger n=views.length/sizeof *source;
                NSMutableData *resolved=[NSMutableData dataWithLength:n*sizeof(struct MetalReach)];
                struct MetalReach *into=resolved.mutableBytes;
                for(NSUInteger k=0;k<n;k++)into[k]=MetalTemplateReachOf(source[k],buffers,shifts);
                if(hosted) {
                    MetalHostOperation operation=host(call[@"kind"],bytes,&error);
                    if(!operation)return error?:MetalReplayFailure(EINVAL,@"the frontend made no host operation");
                    MetalScheduleHost(s,into,n,&again,operation);
                } else {
                    ncclMeshGroup group={0};
                    MetalCrossing crossing=cross(call[@"kind"],bytes,s,&group,&error);
                    if(!crossing)return error?:MetalReplayFailure(EINVAL,@"the frontend made no crossing");
                    [root->owners addObject:crossing];
                    const int failed=MetalScheduleCross(s,into,n,&group,&again,crossing);
                    if(failed)return MetalReplayFailure(failed,@"the template's crossing could not be encoded");
                }
            } else return MetalReplayFailure(EINVAL,[NSString stringWithFormat:@"a template's %@ call",kind]);
        }
        }
    } @catch(NSException *error) { return MetalReplayFailure(EINVAL,error.reason); }
    return nil;
}
