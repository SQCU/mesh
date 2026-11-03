#import "metal-recording.h"
#include "mesh-graph.h"
#include "nccl.h"
#import <objc/runtime.h>
#include <stdatomic.h>
#include <time.h>
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
    /* a binding reach names its slot; one built with binding -1 (an input's or a transfer payload's, reached by no
       argument) is a plain reach. Changed by the LSTM demonstration session: every reach was marked a binding, and
       the recorder refused the 8-step LSTM's transfer-payload reach as slot -1 (and would shift by -1 marking its
       access), 2026-11-01. What must stay: MetalReachBinding exactly when binding >= 0. */
    struct MetalReach reach={buffer,base,0,access|(binding>=0?MetalReachBinding:0),binding};
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
    struct MetalReach reach={buffer,empty?offset:(NSUInteger)base,0,access|(binding>=0?MetalReachBinding:0),binding};
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

static char pipelineSizesKey;
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
        if(error&&!*error)*error=[NSError errorWithDomain:@"mesh.pipeline" code:ENOENT
            userInfo:@{NSLocalizedDescriptionKey:[NSString stringWithFormat:@"no function %@",function]}];
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
    if(pipeline)MetalPipelineSizes(pipeline,reflection);
    return pipeline;
}
