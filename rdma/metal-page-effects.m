#import "metal-page-effects.h"
#import "mesh-index-relation.h"
#include "mesh-isl-memory.h"
#include "mesh-page-sites.h"
#include <isl/val.h>
#include <isl/space.h>
#include <isl/point.h>
#include <isl/constraint.h>

id MetalPageRegionRoot(void) { return @"mesh-region"; }

static ncclMeshBuffer PageBuffer(ncclMeshBuffer value) {
    if(!value.buffer&&ncclMeshMemBuffer((void *)value.offset,&value.buffer,&value.offset))
        [NSException raise:NSInvalidArgumentException format:@"page effects: operand has no region page binding"];
    return value;
}

static NSData *PageOperandIdentity(ncclMeshBuffer value,NSUInteger bytes) {
    id<MTLBuffer> buffer=(__bridge id<MTLBuffer>)value.buffer;size_t count=0;
    const void *address=(const char *)buffer.contents+value.offset;
    if(ncclMeshRegionRuns(address,bytes,NULL,&count))
        [NSException raise:NSInvalidArgumentException format:@"page effects: coverage operand has no canonical pages"];
    NSMutableData *raw=[NSMutableData dataWithLength:count*sizeof(ncclMeshRegionRun)];
    if(ncclMeshRegionRuns(address,bytes,raw.mutableBytes,&count))
        [NSException raise:NSInvalidArgumentException format:@"page effects: coverage page binding changed during preparation"];
    NSMutableData *result=[NSMutableData data];const ncclMeshRegionRun *runs=raw.bytes;
    for(size_t i=0;i<count;i++) {
        ncclMeshRegionRun *last=result.length?(ncclMeshRegionRun *)result.mutableBytes+result.length/sizeof(*last)-1:NULL;
        if(last&&last->offset+last->bytes==runs[i].offset)last->bytes+=runs[i].bytes;
        else [result appendBytes:runs+i length:sizeof(*runs)];
    }
    return result;
}

@implementation MetalPageCoverage
+ (instancetype)declaration:(ncclMeshPageCoverage)value {
    MetalPageCoverage *v=[[[self alloc] init] autorelease];
    v->declaration=value;v->declaration.active=PageBuffer(value.active);v->declaration.expected=PageBuffer(value.expected);
    [(__bridge id)v->declaration.active.buffer retain];[(__bridge id)v->declaration.expected.buffer retain];
    v->identity=[@[@(value.cuts),PageOperandIdentity(v->declaration.active,value.cuts*sizeof(uint32_t)),
        PageOperandIdentity(v->declaration.expected,value.cuts*sizeof(uint32_t))] retain];return v;
}
- (BOOL)isEqual:(id)other { return other==self||([other isKindOfClass:MetalPageCoverage.class]&&[identity isEqual:((MetalPageCoverage *)other)->identity]); }
- (NSUInteger)hash { return identity.hash; }
- (void)dealloc { [(__bridge id)declaration.active.buffer release];[(__bridge id)declaration.expected.buffer release];[identity release];[super dealloc]; }
@end
@implementation MetalPageWriteDemand
+ (instancetype)before:(NSUInteger)b selection:(isl_map *)s {
    MetalPageWriteDemand *v=[[[self alloc] init] autorelease];v->before=b;v->selection=isl_map_copy(s);return v;
}
- (void)bind:(MetalPageCoverage *)c cut:(NSUInteger)k { [coverage release];coverage=[c retain];cut=k; }
- (void)dealloc { isl_map_free(selection);[coverage release];[super dealloc]; }
@end
@implementation MetalPageContribution
+ (instancetype)coverage:(MetalPageCoverage *)c word:(ncclMeshBuffer)ready phase:(NSUInteger)p cells:(NSUInteger)n
    width:(NSUInteger)w indices:(ncclMeshBuffer)i {
    MetalPageContribution *v=[[[self alloc] init] autorelease];v->coverage=[c retain];v->phase=p;
    v->cells=n;v->width=w;v->word=PageBuffer(ready);v->indices=PageBuffer(i);
    [(__bridge id)v->word.buffer retain];[(__bridge id)v->indices.buffer retain];return v;
}
- (void)dealloc { [coverage release];[(__bridge id)word.buffer release];[(__bridge id)indices.buffer release];[super dealloc]; }
@end
@implementation MetalPageAnnouncement
+ (instancetype)coverage:(MetalPageCoverage *)c phase:(NSUInteger)p flags:(uint32_t)f relation:(isl_map *)r {
    MetalPageAnnouncement *v=[[[self alloc] init] autorelease];v->coverage=[c retain];v->phase=p;v->flags=f;
    v->relation=isl_map_copy(r);return v;
}
- (void)dealloc { [coverage release];isl_map_free(relation);[super dealloc]; }
@end

isl_map *MetalPageRegionRelation(isl_map *local,id<MTLBuffer> buffer) {
    if(isl_map_is_empty(local)==isl_bool_true)return isl_map_copy(local);
    const struct MeshIndexBounds bounds=MeshIndexImageBounds(isl_map_copy(local));
    const NSUInteger first=(NSUInteger)bounds.low,length=(NSUInteger)(bounds.high-bounds.low)+1;
    size_t count=0;
    const void *address=(const char *)buffer.contents+first;
    if(ncclMeshRegionRuns(address,length,NULL,&count))
        [NSException raise:NSInvalidArgumentException format:@"page effects: selected bytes have no bridge page mapping"];
    NSMutableData *storage=[NSMutableData dataWithLength:count*sizeof(ncclMeshRegionRun)];
    ncclMeshRegionRun *runs=storage.mutableBytes;
    if(ncclMeshRegionRuns(address,length,runs,&count))
        [NSException raise:NSInvalidArgumentException format:@"page effects: page mapping changed during preparation"];
    /* Query the bounded footprint once, not the whole allocation or each
       selected scalar. Composition below retains all holes and repetitions. */
    NSMutableString *text=[NSMutableString stringWithString:@"{ [b] -> [p] : "];
    NSUInteger at=first;
    for(size_t k=0;k<count;k++) {
        [text appendFormat:@"%@( %lu <= b < %lu and p = b - %lu + %llu )",k?@" or ":@"",
            (unsigned long)at,(unsigned long)(at+runs[k].bytes),(unsigned long)at,
            (unsigned long long)runs[k].offset];
        at+=runs[k].bytes;
    }
    if(!count)[text appendString:@"false"];
    [text appendString:@" }"];
    return MeshIndexCompose(isl_map_copy(local),isl_map_read_from_str(isl_map_get_ctx(local),text.UTF8String));
}

isl_map *MetalPageRegionSelection(isl_ctx *context,id<MTLBuffer> buffer,MetalTensorLayout *bytes) {
    isl_map *local=MeshIndexRelation(context,bytes);
    isl_map *physical=MetalPageRegionRelation(local,buffer);
    isl_map_free(local);return physical;
}

/* metal-page-effects.h MetalPageRegionInverse: each page run's bytes back to the buffer's, then the layout's inverse;
   the bounds are the layout's image's, the ones MetalPageRegionRelation queries */
isl_map *MetalPageRegionInverse(isl_ctx *context,id<MTLBuffer> buffer,MetalTensorLayout *bytes) {
    isl_map *layout=MeshIndexInverse(context,bytes);
    if(!layout)return NULL;
    const struct MeshIndexBounds bounds=MeshIndexImageBounds(isl_map_reverse(isl_map_copy(layout)));
    const NSUInteger first=(NSUInteger)bounds.low,length=(NSUInteger)(bounds.high-bounds.low)+1;
    size_t count=0;
    const void *address=(const char *)buffer.contents+first;
    if(ncclMeshRegionRuns(address,length,NULL,&count))
        [NSException raise:NSInvalidArgumentException format:@"page effects: selected bytes have no bridge page mapping"];
    NSMutableData *storage=[NSMutableData dataWithLength:count*sizeof(ncclMeshRegionRun)];
    ncclMeshRegionRun *runs=storage.mutableBytes;
    if(ncclMeshRegionRuns(address,length,runs,&count))
        [NSException raise:NSInvalidArgumentException format:@"page effects: page mapping changed during preparation"];
    NSMutableString *text=[NSMutableString stringWithString:@"{ [p] -> [b] : "];
    NSUInteger at=first;
    for(size_t k=0;k<count;k++) {
        [text appendFormat:@"%@( %llu <= p < %llu and b = p - %llu + %lu )",k?@" or ":@"",
            (unsigned long long)runs[k].offset,(unsigned long long)(runs[k].offset+runs[k].bytes),
            (unsigned long long)runs[k].offset,(unsigned long)at];
        at+=runs[k].bytes;
    }
    if(!count)[text appendString:@"false"];
    [text appendString:@" }"];
    return MeshIndexCompose(isl_map_read_from_str(context,text.UTF8String),layout);
}

isl_map *MetalPageGroupRelation(isl_map *access,const NSUInteger threads[3],const NSUInteger counts[3],
    const NSInteger *axes,NSUInteger axisCount,const NSUInteger *reductions,NSUInteger reductionCount) {
    NSCParameterAssert((NSUInteger)isl_map_dim(access,isl_dim_in)==axisCount);
    isl_ctx *context=isl_map_get_ctx(access);
    isl_space *domain=isl_space_domain(isl_map_get_space(access));
    isl_local_space *local=isl_local_space_from_space(isl_space_copy(domain));
    isl_aff *flat=isl_aff_zero_on_domain(isl_local_space_copy(local));
    for(NSUInteger dim=0;dim<axisCount;dim++) {
        NSInteger axis=axes[dim];BOOL reduction=NO;
        for(NSUInteger r=0;r<reductionCount;r++)reduction|=reductions[r]==dim;
        if(axis<0||reduction)continue;
        NSCParameterAssert(axis<3&&threads[axis]>0);
        NSUInteger stride=axis==0?1:axis==1?counts[0]:counts[0]*counts[1];
        isl_aff *coordinate=isl_aff_var_on_domain(isl_local_space_copy(local),isl_dim_set,(unsigned)dim);
        coordinate=isl_aff_floor(isl_aff_scale_down_val(coordinate,isl_val_int_from_ui(context,threads[axis])));
        coordinate=isl_aff_scale_val(coordinate,isl_val_int_from_ui(context,stride));
        flat=isl_aff_add(flat,coordinate);
    }
    isl_local_space_free(local);isl_space_free(domain);
    isl_map *groups=isl_map_from_aff(flat);
    groups=isl_map_intersect_domain(groups,isl_map_domain(isl_map_copy(access)));
    return MeshIndexCompose(isl_map_reverse(groups),isl_map_copy(access));
}

MetalPageAccess *MetalPageImportAccess(isl_ctx *context,NSDictionary *declaration,id<MTLBuffer> buffer,
    MetalTensorLayout *bytes,const NSUInteger threads[3],const NSUInteger counts[3]) {
    isl_map *logical=isl_map_read_from_str(context,[declaration[@"relation"] UTF8String]);
    if(![declaration[@"grouped"] boolValue]) {
        NSUInteger rank=[declaration[@"shape"] count],reductionCount=[declaration[@"reductions"] count];
        NSInteger axes[rank];NSUInteger reductions[reductionCount];
        for(NSUInteger k=0;k<rank;k++)axes[k]=-1;
        NSArray *active=declaration[@"active"];
        for(NSUInteger k=0;k<active.count;k++)axes[[active[k] unsignedIntegerValue]]=(NSInteger)k;
        for(NSUInteger k=0;k<reductionCount;k++)reductions[k]=[declaration[@"reductions"][k] unsignedIntegerValue];
        isl_map *projected=MetalPageGroupRelation(logical,threads,counts,axes,rank,reductions,reductionCount);
        isl_map_free(logical);logical=projected;
    }
    isl_map *local=MeshIndexCompose(logical,MeshIndexRelation(context,[bytes reshape:@[@(bytes.count)]]));
    isl_map *physical=MetalPageRegionRelation(local,buffer);
    uint32_t mode=[declaration[@"side"] isEqual:@"reads"]?MetalReachRead:MetalReachWritten;
    MetalPageAccess *result=[MetalPageAccess root:MetalPageRegionRoot() slot:[declaration[@"slot"] integerValue]
        access:mode relation:physical];
    isl_map_free(local);isl_map_free(physical);return result;
}

@implementation MetalPageAccess
+ (instancetype)root:(id)r slot:(NSInteger)s access:(uint32_t)a relation:(isl_map *)m {
    MetalPageAccess *v=[[[self alloc] init] autorelease];v->root=[r retain];v->slot=s;v->access=a;v->relation=isl_map_copy(m);return v;
}
- (void)dealloc { [root release];isl_map_free(relation);[super dealloc]; }
@end
@implementation MetalPageAccessLayout
+ (instancetype)buffer:(id<MTLBuffer>)b slot:(NSInteger)s access:(uint32_t)a
    groups:(MetalTensorLayout *)g bytes:(MetalTensorLayout *)p invocation:(NSInteger)i {
    NSCParameterAssert(g.count==p.count);
    MetalPageAccessLayout *v=[[[self alloc] init] autorelease];
    v->buffer=[b retain];v->slot=s;v->access=a;v->groups=[g retain];v->bytes=[p retain];v->invocation=i;return v;
}
- (MetalPageAccess *)resolve:(isl_ctx *)context displacement:(NSInteger)displacement {
    isl_map *local=MeshIndexCompose(isl_map_reverse(MeshIndexRelation(context,groups)),
        MeshIndexRelation(context,[bytes shift:displacement]));
    isl_map *relation=MetalPageRegionRelation(local,buffer);
    MetalPageAccess *result=[MetalPageAccess root:MetalPageRegionRoot() slot:slot access:access relation:relation];
    isl_map_free(local);isl_map_free(relation);return result;
}
- (void)dealloc { [buffer release];[groups release];[bytes release];[super dealloc]; }
@end
@implementation MetalPageWriter
+ (instancetype)producer:(MetalPageCommand *)p command:(MetalPageCommand *)c
    phase:(NSUInteger)f cells:(NSUInteger)n selected:(ncclMeshBuffer)s destination:(ncclMeshBuffer)d
    cellBytes:(NSUInteger)b groups:(MetalTensorLayout *)g ownership:(enum MetalPageWriteOwnership)o
    sequence:(MetalTensorLayout *)q {
    MetalPageWriter *v=[[[self alloc] init] autorelease];
    /* Commands own their attached writer declarations. The enclosing source
       array owns both endpoint commands, so these links are non-owning. */
    v->producer=p;v->command=c;v->phase=f;v->cells=n;v->selected=PageBuffer(s);v->destination=PageBuffer(d);
    [(__bridge id)v->selected.buffer retain];[(__bridge id)v->destination.buffer retain];
    v->cellBytes=b;v->groups=[g retain];v->ownership=o;v->sequence=[q retain];return v;
}
- (void)dealloc { [(__bridge id)selected.buffer release];[(__bridge id)destination.buffer release];
    [groups release];[sequence release];[super dealloc]; }
@end
@implementation MetalPageCommand
@synthesize coherentAccesses, argumentNames, frame;
+ (instancetype)order:(NSUInteger)o groups:(NSUInteger)g backend:(enum MetalPageBackend)b source:(NSString *)s
    function:(NSString *)f grid:(MTLSize)d metadataSlot:(NSUInteger)slot accesses:(NSArray<MetalPageAccess *> *)a {
    MetalPageCommand *v=[[[self alloc] init] autorelease];v->order=o;v->groups=g;v->backend=b;
    v->source=[s copy];v->function=[f copy];v->grid=d;v->metadataSlot=slot;v->accesses=[a copy];return v;
}
- (void)dealloc { [(__bridge id)activity.buffer release];[source release];[function release];[accesses release];[argumentNames release];[sites release];[contributions release];[announcements release];[writers release];[super dealloc]; }
@end
void MetalPageCommandActivity(MetalPageCommand *command,id<MTLBuffer> buffer,NSUInteger offset,isl_ctx *context) {
    [(__bridge id)command->activity.buffer release];command->activity=(ncclMeshBuffer){[buffer retain],offset};
    NSUInteger groups=command->sites.count?[command->sites[0][@"first"] unsignedIntegerValue]:command->groups;
    NSString *text=[NSString stringWithFormat:@"{ [g] -> [b] : 0<=g<%lu and %lu<=b<%lu }",
        (unsigned long)groups,(unsigned long)offset,(unsigned long)(offset+sizeof(uint32_t))];
    isl_map *local=isl_map_read_from_str(context,text.UTF8String),*relation=MetalPageRegionRelation(local,buffer);
    MetalPageAccess *access=[MetalPageAccess root:MetalPageRegionRoot() slot:-1 access:MetalReachRead relation:relation];
    command->activityAccess=access;
    NSArray *all=[command->accesses arrayByAddingObject:access];[command->accesses release];command->accesses=[all copy];
    isl_map_free(relation);isl_map_free(local);
}
void MetalPageCommandSite(MetalPageCommand *command,NSString *name,NSUInteger cells,enum MetalPageSiteScope scope,
    enum MetalPageSiteScope joinScope,NSArray<MetalPageAccess *> *accesses) {
    NSMutableArray *joined=[command->accesses mutableCopy];
    for(MetalPageAccess *access in accesses) {
        isl_ctx *context=isl_map_get_ctx(access->relation);
        NSString *text=[NSString stringWithFormat:@"{ [g] -> [c] : c=g-%lu and 0<=c<%lu }",
            (unsigned long)command->groups,(unsigned long)cells];
        isl_map *shift=isl_map_read_from_str(context,text.UTF8String);
        isl_map *relation=MeshIndexCompose(shift,isl_map_copy(access->relation));
        MetalPageAccess *selected=[MetalPageAccess root:access->root slot:access->slot access:access->access relation:relation];
        selected->phase=command->sites.count+1;[joined addObject:selected];
        isl_map_free(relation);
    }
    [command->accesses release];command->accesses=joined;
    NSArray *sites=[(command->sites?:@[]) arrayByAddingObject:@{@"name":name,@"first":@(command->groups),
        @"count":@(cells),@"scope":@(scope),@"join_scope":@(joinScope)}];
    [command->sites release];command->sites=[sites copy];command->groups+=cells;
}
void MetalPageCommandSiteActivity(MetalPageCommand *command,NSUInteger phase,MetalPageAccessLayout *activity,
    isl_ctx *context) {
    NSMutableArray *sites=[command->sites mutableCopy];
    NSMutableDictionary *site=[sites[phase-1] mutableCopy];site[@"activity"]=activity;
    sites[phase-1]=site;[site release];[command->sites release];command->sites=sites;
    MetalPageAccess *local=[activity resolve:context displacement:0];
    const NSUInteger first=[site[@"first"] unsignedIntegerValue];
    NSString *shift=[NSString stringWithFormat:@"{ [g] -> [c] : c=g-%lu }",(unsigned long)first];
    isl_map *relation=MeshIndexCompose(isl_map_read_from_str(context,shift.UTF8String),isl_map_copy(local->relation));
    MetalPageAccess *read=[MetalPageAccess root:local->root slot:local->slot access:MetalReachRead relation:relation];
    read->phase=phase;isl_map_free(relation);
    NSArray *all=[command->accesses arrayByAddingObject:read];[command->accesses release];command->accesses=[all copy];
}
@implementation MetalPageTransfer
@synthesize coverage,demand,coverageCut;
+ (instancetype)owner:(id)o order:(NSUInteger)n effect:(ncclMeshPageEffect)e root:(id)r selection:(isl_map *)s
    payload:(id<MTLBuffer>)p layout:(MetalTensorLayout *)l {
    if(!e.word.buffer&&ncclMeshMemBuffer((void *)e.word.offset,&e.word.buffer,&e.word.offset))
        [NSException raise:NSInvalidArgumentException format:@"page effects: publication word has no region binding"];
    if(!e.violation.buffer&&e.violation.offset&&ncclMeshMemBuffer((void *)e.violation.offset,&e.violation.buffer,&e.violation.offset))
        [NSException raise:NSInvalidArgumentException format:@"page effects: diagnostic operand has no region binding"];
    MetalPageTransfer *v=[[[self alloc] init] autorelease];v->owner=[o retain];v->order=n;v->effect=e;v->observeAccess=MetalReachRead;
    [(__bridge id)e.word.buffer retain];[(__bridge id)e.violation.buffer retain];
    v->root=[r retain];v->selection=isl_map_copy(s);v->payload=[p retain];v->layout=[l retain];return v;
}
- (void)dealloc { [owner release];[root release];[payload release];[layout release];[coverage release];[demand release];
    [(__bridge id)effect.word.buffer release];[(__bridge id)effect.violation.buffer release];isl_map_free(selection);[super dealloc]; }
@end
@implementation MetalPageInitial
- (void)dealloc { [transfer release];[observations release];[super dealloc]; }
@end
@implementation MetalPageEffects
- (void)dealloc { [commands release];[initial release];[results release];[domains release];[solved release];
    [declarationCommands release];[declarationTransfers release];[super dealloc]; }
@end
@implementation MetalPageSource
@synthesize source, metadata, metadataOffset, reaches;
- (void)dealloc { [source release];[metadata release];[observedSlots release];[publishedSlots release];[reaches release];[owners release];[super dealloc]; }
@end

@interface MetalPageDomain : NSObject {
@public NSArray<MetalPageTransfer *> *rows;
    BOOL publication;
    NSMutableArray<NSMutableArray<NSMutableIndexSet *> *> *announcements;
    NSMutableArray<NSMutableArray<NSMutableIndexSet *> *> *activity;
    NSMutableArray<NSMutableArray<NSMutableIndexSet *> *> *contributions;
    NSMutableArray<NSMutableData *> *thresholds;
    NSArray<NSMutableData *> *observationRows; /* selected declared row, never an absolute counter */
    NSMutableArray<NSMutableIndexSet *> *observedSlots,*publishedSlots;
    NSMutableData *expected,*completed;
    NSArray<MetalPageInitial *> *initial;
    NSArray *pages;
}
@end
@implementation MetalPageDomain
- (void)dealloc { [rows release];[contributions release];[thresholds release];[observedSlots release];[publishedSlots release];
    [announcements release];[activity release];
    [observationRows release];[expected release];[completed release];[initial release];[pages release];[super dealloc]; }
@end

static void PageFailure(NSString *reason) { [NSException raise:NSInvalidArgumentException format:@"page effects: %@",reason]; }
static isl_set *PageRange(isl_map *relation) { return isl_map_range(isl_map_copy(relation)); }
static BOOL PageEmpty(isl_set *set) {
    isl_bool value=isl_set_is_empty(set);if(value<0)PageFailure(@"invalid relation");return value;
}
static isl_set *PageDomain(isl_map *relation,isl_set *range) {
    return isl_map_domain(isl_map_intersect_range(isl_map_copy(relation),isl_set_copy(range)));
}
/* Each group set's runs, their count and microseconds, and a slow set's isl text, one JSON line each in
   MESH_PREPARE_RECORD's directory (none where it is unset): a log of preparation that nothing reads back. Added for the
   LSTM demonstration, whose page-effects solve spent rank 0's whole sample here (2026-11-01). */
static void PageGroupsRecord(isl_set *groups,size_t count,uint64_t started) {
    static FILE *out;static int opened;static uint64_t calls;static double total;
    if(!opened){
        opened=1;const char *dir=getenv("MESH_PREPARE_RECORD");char path[1024];
        if(dir&&snprintf(path,sizeof path,"%s/page-groups-%d.jsonl",dir,(int)getpid())<(int)sizeof path)out=fopen(path,"w");
    }
    if(!out)return;
    const double us=(double)(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)-started)/1e3;
    calls++;total+=us;
    /* a slow set alone, with the running count and microseconds of all of them: every set was a line, 468151 of
       them on rank 0 before its bound (2026-11-01), and the text is isl's, freed by isl's allocator, never free() */
    if(us<=1000)return;
    char *text=isl_set_to_str(groups);
    fprintf(out,"{\"runs\":%zu,\"us\":%.1f,\"calls\":%llu,\"total_us\":%.1f,\"set\":\"%.600s\"}\n",count,us,
            (unsigned long long)calls,total,text?text:"");
    mesh_isl_free(text);fflush(out);
}
/* A one-dimensional basic set without divisions is an interval, read from its constraints: the bounds a x + k >= 0
   (and equalities) give. NO where it has divisions, parameters or an unbounded side. */
struct PageBounds { int64_t low,high; BOOL lower,upper,empty,general; };
static isl_stat PageBound(isl_constraint *constraint,void *user) {
    struct PageBounds *b=user;
    isl_val *a=isl_constraint_get_coefficient_val(constraint,isl_dim_set,0),*k=isl_constraint_get_constant_val(constraint);
    if(!isl_val_is_int(a)||!isl_val_is_int(k)){b->general=YES;}
    else {
        const int64_t x=isl_val_get_num_si(a),c=isl_val_get_num_si(k);
        if(isl_constraint_is_equality(constraint)) {
            if(!x){if(c)b->empty=YES;}
            else if(c%x){b->empty=YES;}
            else {const int64_t v=-c/x;b->low=b->lower?MAX(b->low,v):v;b->high=b->upper?MIN(b->high,v):v;b->lower=b->upper=YES;}
        } else if(x>0) {
            const int64_t v=(-c>=0?(-c+x-1)/x:-((c)/x));b->low=b->lower?MAX(b->low,v):v;b->lower=YES;
        } else if(x<0) {
            const int64_t y=-x,v=(c>=0?c/y:-((-c+y-1)/y));b->high=b->upper?MIN(b->high,v):v;b->upper=YES;
        } else if(c<0)b->empty=YES;
    }
    isl_val_free(a);isl_val_free(k);isl_constraint_free(constraint);return isl_stat_ok;
}
static int PageIntervalOrder(const void *a,const void *b) {
    const int64_t *x=a,*y=b;return x[0]<y[0]?-1:x[0]>y[0];
}
/* the set's elements as merged intervals, where every basic set is a division-free interval: one pass over its
   constraints, no isl set built or subtracted (MeshIndexSetRuns peels runs with both, 2.4 ms for one interval) */
static BOOL PageIntervalsEach(isl_set *set,void (^each)(NSUInteger)) {
    if(isl_set_dim(set,isl_dim_set)!=1||isl_set_dim(set,isl_dim_param)!=0)return NO;
    isl_basic_set_list *parts=isl_set_get_basic_set_list(set);
    const isl_size count=isl_basic_set_list_size(parts);
    int64_t *intervals=malloc(2*sizeof(int64_t)*(size_t)(count>0?count:1));size_t made=0;BOOL plain=intervals!=NULL;
    for(isl_size i=0;plain&&i<count;i++) {
        isl_basic_set *part=isl_basic_set_list_get_at(parts,i);
        struct PageBounds b={0};
        if(isl_basic_set_dim(part,isl_dim_div)!=0)plain=NO;
        else if(isl_basic_set_foreach_constraint(part,PageBound,&b)<0||b.general||(!b.empty&&(!b.lower||!b.upper)))plain=NO;
        else if(!b.empty&&b.low<=b.high){intervals[2*made]=b.low;intervals[2*made+1]=b.high;made++;}
        isl_basic_set_free(part);
    }
    isl_basic_set_list_free(parts);
    if(plain) {
        qsort(intervals,made,2*sizeof(int64_t),PageIntervalOrder);
        int64_t next=INT64_MIN;
        for(size_t k=0;k<made;k++)for(int64_t v=MAX(intervals[2*k],next);v<=intervals[2*k+1];v++){each((NSUInteger)v);next=v+1;}
    }
    free(intervals);return plain;
}
static void PageGroups(isl_set *groups,void (^each)(NSUInteger)) {
    const uint64_t started=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    if(PageIntervalsEach(groups,each)){PageGroupsRecord(groups,0,started);return;}
    size_t count=0;int64_t *runs=MeshIndexSetRuns(groups,&count);
    PageGroupsRecord(groups,count,started);
    if(!runs&&!PageEmpty(groups))PageFailure(@"cannot lower dispatch selection");
    for(size_t i=0;i<count;i++)for(int64_t r=0;r<runs[4*i+3];r++)for(int64_t j=0;j<runs[4*i+1];j++)
        each((NSUInteger)(runs[4*i]+r*runs[4*i+2]+j));
    mesh_isl_free(runs);
}
/* every (input, output) pair of a map from one integer to one integer, its points read once */
struct PagePairsUser { void (^each)(NSUInteger,NSUInteger); };
static isl_stat PagePairPoint(isl_point *point,void *user) {
    isl_val *a=isl_point_get_coordinate_val(point,isl_dim_set,0),*b=isl_point_get_coordinate_val(point,isl_dim_set,1);
    ((struct PagePairsUser *)user)->each((NSUInteger)isl_val_get_num_si(a),(NSUInteger)isl_val_get_num_si(b));
    isl_val_free(a);isl_val_free(b);isl_point_free(point);return isl_stat_ok;
}
static void PagePairs(isl_map *map,void (^each)(NSUInteger,NSUInteger)) {
    struct PagePairsUser user={each};
    isl_set *pairs=isl_set_flatten(isl_map_wrap(isl_map_copy(map)));
    if(isl_set_foreach_point(pairs,PagePairPoint,&user)<0)PageFailure(@"cannot enumerate a group's cuts");
    isl_set_free(pairs);
}
static isl_set *PageInterval(isl_set *domain,NSUInteger first,NSUInteger bytes) {
    isl_set *v=isl_set_universe(isl_set_get_space(domain));
    v=isl_set_lower_bound_si(v,isl_dim_set,0,(int)0);
    isl_ctx *ctx=isl_set_get_ctx(domain);
    v=isl_set_lower_bound_val(v,isl_dim_set,0,isl_val_int_from_ui(ctx,first));
    return isl_set_upper_bound_val(v,isl_dim_set,0,isl_val_int_from_ui(ctx,first+bytes-1));
}
static NSNumber *PageWord(ncclMeshBuffer word) {
    id<MTLBuffer> buffer=(__bridge id<MTLBuffer>)word.buffer;
    ncclMeshRegionRun run;size_t count=1;
    if(ncclMeshRegionRuns((const char *)buffer.contents+word.offset,sizeof(uint64_t),&run,&count)||count!=1)
        PageFailure(@"publication word must name one declared region span");
    return @(run.offset);
}
static NSArray *PageSorted(NSArray *values,NSComparator compare) { return [values sortedArrayWithOptions:NSSortStable usingComparator:compare]; }
static NSArray *PageSemanticCut(MetalPageTransfer *t) {
    return @[t->owner,t->root,@(t->effect.call),@(t->effect.frame),@(t->effect.operand),t->demand?:NSNull.null,t->coverage,@(t->coverageCut)];
}
static NSMutableArray *PageSlots(NSUInteger count) {
    NSMutableArray *values=[NSMutableArray arrayWithCapacity:count];
    for(NSUInteger i=0;i<count;i++)[values addObject:[NSMutableIndexSet indexSet]];return values;
}

/* Map each coverage value to the exact byte word which defines it. The
   domain is a cut ordinal, not a possible numerical destination address. */
static isl_map *PageCoverageWords(isl_ctx *context,ncclMeshBuffer word,NSUInteger count) {
    NSString *text=[NSString stringWithFormat:@"{ [c] -> [b] : 0<=c<%lu and %llu+4*c<=b<%llu+4*c+4 }",
        (unsigned long)count,(unsigned long long)word.offset,(unsigned long long)word.offset];
    isl_map *local=isl_map_read_from_str(context,text.UTF8String);
    isl_map *physical=MetalPageRegionRelation(local,(__bridge id<MTLBuffer>)word.buffer);
    isl_map_free(local);return physical;
}

static void PageCoverageAccesses(MetalPageCommand *command,isl_ctx *context) {
    if(!command->contributions.count&&!command->announcements.count)return;
    NSMutableArray *accesses=[command->accesses mutableCopy];
    void (^append)(isl_map *,NSUInteger)=^(isl_map *relation,NSUInteger phase) {
        MetalPageAccess *access=[MetalPageAccess root:MetalPageRegionRoot() slot:-1 access:MetalReachRead relation:relation];
        access->phase=phase;[accesses addObject:access];isl_map_free(relation);
    };
    for(MetalPageContribution *part in command->contributions) {
        NSUInteger first=part->phase?[command->sites[part->phase-1][@"first"] unsignedIntegerValue]:0;
        NSString *text=[NSString stringWithFormat:@"{ [g] -> [b] : %lu<=g<%lu and %llu+%lu*(g-%lu)<=b<%llu+%lu*(g-%lu+1) }",
            (unsigned long)first,(unsigned long)(first+part->cells),(unsigned long long)part->indices.offset,
            (unsigned long)(part->width*sizeof(int64_t)),(unsigned long)first,(unsigned long long)part->indices.offset,
            (unsigned long)(part->width*sizeof(int64_t)),(unsigned long)first];
        isl_map *local=isl_map_read_from_str(context,text.UTF8String);
        append(MetalPageRegionRelation(local,(__bridge id<MTLBuffer>)part->indices.buffer),part->phase);isl_map_free(local);
    }
    for(MetalPageAnnouncement *part in command->announcements) {
        NSUInteger first=part->phase?[command->sites[part->phase-1][@"first"] unsignedIntegerValue]:0;
        NSString *text=[NSString stringWithFormat:@"{ [g] -> [c] : c=g-%lu }",(unsigned long)first];
        isl_map *group=MeshIndexCompose(isl_map_read_from_str(context,text.UTF8String),isl_map_copy(part->relation));
        if(part->flags&1)append(MeshIndexCompose(isl_map_copy(group),PageCoverageWords(context,part->coverage->declaration.expected,part->coverage->declaration.cuts)),part->phase);
        if(part->flags&4)append(MeshIndexCompose(isl_map_copy(group),PageCoverageWords(context,part->coverage->declaration.active,part->coverage->declaration.cuts)),part->phase);
        isl_map_free(group);
    }
    [command->accesses release];command->accesses=accesses;
}

static void PageCoverageAnnouncements(NSArray<MetalPageCommand *> *commands,MetalPageDomain *domain) {
    domain->announcements=[[NSMutableArray alloc] initWithCapacity:commands.count];
    domain->activity=[[NSMutableArray alloc] initWithCapacity:commands.count];
    for(MetalPageCommand *command in commands) {
        [domain->announcements addObject:PageSlots(command->groups)];
        [domain->activity addObject:PageSlots(command->groups)];
    }
    NSMutableSet *values=[NSMutableSet set];
    for(MetalPageTransfer *row in domain->rows)if(row->coverage)[values addObject:row->coverage];
    for(MetalPageCoverage *coverage in values) {
      NSMutableSet *orders=[NSMutableSet set];
      for(MetalPageTransfer *row in domain->rows)if([row->coverage isEqual:coverage])[orders addObject:@(row->order)];
      for(NSNumber *order in orders)for(NSUInteger kind=0;kind<2;kind++) {
        ncclMeshBuffer word=kind?coverage->declaration.active:coverage->declaration.expected;
        isl_ctx *context=isl_map_get_ctx(domain->rows[0]->selection);
        isl_map *words=PageCoverageWords(context,word,coverage->declaration.cuts);
        isl_set *selected=isl_set_empty(isl_space_domain(isl_map_get_space(words)));
        for(MetalPageTransfer *row in domain->rows)if([row->coverage isEqual:coverage]&&row->order==order.unsignedIntegerValue) {
            isl_set *cell=isl_set_universe(isl_set_get_space(selected));
            cell=isl_set_fix_val(cell,isl_dim_set,0,isl_val_int_from_ui(context,row->coverageCut));
            selected=isl_set_union(selected,cell);
        }
        words=isl_map_intersect_domain(words,selected);
        isl_set *remaining=PageRange(words);
        NSMutableArray *target=kind?domain->activity:domain->announcements;
        /* this coverage's rows at this order, by their cut, once. Changed by the LSTM demonstration session: each
           group of each writing access fixed its group in the map, took its range's runs and scanned every row for
           every cut, groups x rows of isl and of scans (776515 group sets on rank 0 in the 64-step plan, 2026-11-01).
           Now a map's (group, cut) pairs are its points, read once (PagePairs). What must stay: the same rows added
           to the same groups. */
        NSMutableDictionary<NSNumber *,NSMutableIndexSet *> *byCut=[NSMutableDictionary dictionary];
        for(NSUInteger row=0;row<domain->rows.count;row++) {
            MetalPageTransfer *r=domain->rows[row];
            if(![r->coverage isEqual:coverage]||r->order!=order.unsignedIntegerValue)continue;
            NSMutableIndexSet *rows=byCut[@(r->coverageCut)];if(!rows)byCut[@(r->coverageCut)]=rows=[NSMutableIndexSet indexSet];
            [rows addIndex:row];
        }
        for(NSUInteger c=commands.count;c-->0;) {
            MetalPageCommand *command=commands[c];if(command->order>=order.unsignedIntegerValue)continue;
            for(NSUInteger phase=command->sites.count+1;phase-->0;) {
                for(MetalPageAccess *access in command->accesses) {
                    if(access->phase!=phase||![access->root isEqual:MetalPageRegionRoot()]||!(access->access&MetalReachWritten))continue;
                    isl_map *writes=isl_map_intersect_range(isl_map_copy(access->relation),isl_set_copy(remaining));
                    isl_map *cuts=MeshIndexCompose(writes,isl_map_reverse(isl_map_copy(words)));
                    isl_set *groups=isl_map_domain(isl_map_copy(cuts));
                    if(!PageEmpty(groups)&&access->slot>=0)[domain->publishedSlots[c] addIndex:(NSUInteger)access->slot];
                    PagePairs(cuts,^(NSUInteger group,NSUInteger cut){
                        NSIndexSet *rows=byCut[@(cut)];if(rows)[target[c][group] addIndexes:rows];
                    });
                    isl_set_free(groups);isl_map_free(cuts);
                }
                for(MetalPageAccess *access in command->accesses)
                    if(access->phase==phase&&[access->root isEqual:MetalPageRegionRoot()]&&(access->access&MetalReachWritten))
                        remaining=isl_set_subtract(remaining,PageRange(access->relation));
            }
            for(MetalPageAnnouncement *announcement in command->announcements) {
                if(![announcement->coverage isEqual:coverage]||!(announcement->flags&(kind?4u:1u)))continue;
                NSUInteger first=announcement->phase?[command->sites[announcement->phase-1][@"first"] unsignedIntegerValue]:0;
                PagePairs(announcement->relation,^(NSUInteger group,NSUInteger cut){
                    NSIndexSet *rows=byCut[@(cut)];if(rows)[target[c][first+group] addIndexes:rows];
                });
                isl_map *reached=MeshIndexCompose(isl_map_copy(announcement->relation),isl_map_copy(words));
                remaining=isl_set_subtract(remaining,PageRange(reached));isl_map_free(reached);
            }
        }
        if(!PageEmpty(remaining))PageFailure(@"coverage values need their declared producer or imported-value continuation");
        isl_set_free(remaining);isl_map_free(words);
      }
    }
}

void MetalPageEffectsOrder(NSArray<MetalPageCommand *> *commands,NSArray<MetalPageTransfer *> *transfers,NSUInteger *next) {
    NSMutableSet *values=[NSMutableSet set];
    for(MetalPageCommand *c in commands)[values addObject:@(c->order)];
    for(MetalPageTransfer *t in transfers)[values addObject:@(t->order)];
    NSMutableDictionary *positions=[NSMutableDictionary dictionary];
    for(NSNumber *order in [values.allObjects sortedArrayUsingSelector:@selector(compare:)])positions[order]=@((*next)++);
    for(MetalPageCommand *c in commands)c->order=[positions[@(c->order)] unsignedIntegerValue];
    for(MetalPageTransfer *t in transfers)t->order=[positions[@(t->order)] unsignedIntegerValue];
}

NSArray<NSArray<MetalPageTransfer *> *> *MetalPagePublicationDomains(NSArray<MetalPageTransfer *> *transfers) {
    NSMutableDictionary *words=[NSMutableDictionary dictionary];
    for(MetalPageTransfer *t in transfers)if(t->effect.kind==ncclMeshPagePublish) {
        NSNumber *word=PageWord(t->effect.word);NSMutableArray *rows=words[word];
        if(!rows)words[word]=rows=[NSMutableArray array];[rows addObject:t];
    }
    NSMutableArray *domains=[NSMutableArray array];
    for(NSNumber *word in [words.allKeys sortedArrayUsingSelector:@selector(compare:)])
        [domains addObject:PageSorted(words[word],^NSComparisonResult(MetalPageTransfer *a,MetalPageTransfer *b){
            return a->effect.threshold<b->effect.threshold?NSOrderedAscending:a->effect.threshold>b->effect.threshold?NSOrderedDescending:NSOrderedSame;
        })];
    return domains;
}

/* A value arrival constrains reads. Completion of a prior reader constrains
   overwrites of those same pages. Both retain the source version until its
   first numerical write; neither is a command-wide ordering rule. */
/* Changed by the LSTM demonstration session: the walk went on past the command whose writes cover the transfer's last
   remaining pages, so every transfer walked every later command of the function, transfers x commands of isl work
   (MetalPageEffectsSolve under PageObservationGroups, the whole of rank 0's sample in the 64-step plan, 2026-11-01).
   Once nothing remains no later access meets it and nothing is subtracted, so the walk ends there with the same
   groups. What must stay: commands visited in order until the transfer's pages are all overwritten. */
static void PageObservationGroups(NSArray<MetalPageCommand *> *commands,MetalPageTransfer *t,
    void (^each)(NSUInteger,NSUInteger,MetalPageAccess *,isl_set *)) {
    isl_set *remaining=PageRange(t->selection);
    for(NSUInteger c=0;c<commands.count&&!PageEmpty(remaining);c++) {
        MetalPageCommand *command=commands[c];if(command->order<=t->order)continue;
        for(NSUInteger phase=0;phase<=command->sites.count&&!PageEmpty(remaining);phase++) {
            for(MetalPageAccess *access in command->accesses)
                if(access->phase==phase&&[access->root isEqual:t->root]&&(access->access&t->observeAccess)) {
                    isl_set *groups=PageDomain(access->relation,remaining);
                    if(!PageEmpty(groups))each(c,phase,access,groups);
                    isl_set_free(groups);
                }
            for(MetalPageAccess *access in command->accesses)
                if(access->phase==phase&&[access->root isEqual:t->root]&&(access->access&MetalReachWritten))
                    remaining=isl_set_subtract(remaining,PageRange(access->relation));
        }
    }
    isl_set_free(remaining);
}
static isl_set *PageActivityRange(MetalPageTransfer *t) {
    isl_map *words=PageCoverageWords(isl_map_get_ctx(t->selection),t->coverage->declaration.active,t->coverage->declaration.cuts);
    words=isl_map_fix_val(words,isl_dim_in,0,isl_val_int_from_ui(isl_map_get_ctx(words),t->coverageCut));
    return isl_map_range(words);
}
/* A predicate is itself a page read. Close those source relations before the
   single solve, including a mask which arrives under another declared mask.
   Subtracting retained reads makes this finite over the declared mask pages;
   no runtime metadata discovery or polling scheduler is introduced. */
static void PageObservationMasks(NSArray<MetalPageCommand *> *commands,NSArray<MetalPageTransfer *> *transfers) {
    BOOL changed;
    do {
        __block BOOL added=NO;
        for(MetalPageTransfer *t in transfers)if(t->effect.kind==ncclMeshPageObserve&&t->coverage) {
            isl_set *mask=PageActivityRange(t);
            NSMutableArray *extras=[NSMutableArray array];for(NSUInteger c=0;c<commands.count;c++)[extras addObject:[NSMutableArray array]];
            PageObservationGroups(commands,t,^(NSUInteger c,NSUInteger phase,MetalPageAccess *use,isl_set *groups) {
                (void)use;MetalPageCommand *command=commands[c];
                isl_map *relation=isl_map_from_domain_and_range(isl_set_copy(groups),isl_set_copy(mask));
                for(MetalPageAccess *known in command->accesses)
                    if(known->phase==phase&&[known->root isEqual:MetalPageRegionRoot()]&&(known->access&MetalReachRead))
                        relation=isl_map_subtract(relation,isl_map_copy(known->relation));
                if(!isl_map_is_empty(relation)) {
                    MetalPageAccess *access=[MetalPageAccess root:MetalPageRegionRoot() slot:-1 access:MetalReachRead relation:relation];
                    access->phase=phase;[extras[c] addObject:access];added=YES;
                }
                isl_map_free(relation);
            });
            for(NSUInteger c=0;c<commands.count;c++)if([extras[c] count]) {
                MetalPageCommand *command=commands[c];NSArray *all=[command->accesses arrayByAddingObjectsFromArray:extras[c]];
                [command->accesses release];command->accesses=[all copy];
            }
            isl_set_free(mask);
        }
        changed=added;
    } while(changed);
}

static NSArray *PageSourceIncidence(MetalPageEffects *effects);

MetalPageEffects *MetalPageEffectsSolve(NSArray<MetalPageCommand *> *given,NSArray<MetalPageTransfer *> *transfers) {
    MetalPageEffects *result=[[[MetalPageEffects alloc] init] autorelease];
    result->declarationCommands=[given copy];result->declarationTransfers=[transfers copy];
    result->commands=[PageSorted(given,^NSComparisonResult(MetalPageCommand *a,MetalPageCommand *b){
        return a->order<b->order?NSOrderedAscending:a->order>b->order?NSOrderedDescending:NSOrderedSame;
    }) retain];
    NSArray *commands=result->commands;
    if(transfers.count) {
        isl_ctx *context=isl_map_get_ctx(((MetalPageTransfer *)transfers[0])->selection);
        for(MetalPageCommand *command in commands)PageCoverageAccesses(command,context);
    }
    PageObservationMasks(commands,transfers);
    NSMutableDictionary *observers=[NSMutableDictionary dictionary],*coverageSelections=[NSMutableDictionary dictionary];
    NSMutableArray *arrivals=[NSMutableArray array];
    for(MetalPageTransfer *t in transfers) {
        if(t->effect.bytes!=t->layout.count)PageFailure(@"transfer layout must name its exact selected bytes");
        NSNumber *word=PageWord(t->effect.word);
        if(t->effect.kind==ncclMeshPageObserve) {
            id key=@[@(t->order),t->owner,@(t->effect.call),word,@(t->observeAccess),
                t->coverage?:NSNull.null,t->coverage?@(t->coverageCut):NSNull.null];
            NSMutableArray *rows=observers[key];if(!rows)observers[key]=rows=[NSMutableArray array];[rows addObject:t];
            if(t->observeAccess&MetalReachRead)[arrivals addObject:t];
        } else if(t->coverage) {
            NSArray *key=PageSemanticCut(t);isl_set *prior=[coverageSelections[key] pointerValue];
            isl_set *selected=PageRange(t->selection);
            coverageSelections[key]=[NSValue valueWithPointer:prior?isl_set_union(prior,selected):selected];
        }
    }
    arrivals=[PageSorted(arrivals,^NSComparisonResult(MetalPageTransfer *a,MetalPageTransfer *b){
        return a->order>b->order?NSOrderedAscending:a->order<b->order?NSOrderedDescending:NSOrderedSame;
    }) mutableCopy];
    NSMutableArray *initial=[NSMutableArray array],*domains=[NSMutableArray array];
    for(NSArray *rows in MetalPagePublicationDomains(transfers)) {
        NSNumber *word=PageWord(((MetalPageTransfer *)rows[0])->effect.word);
        MetalPageDomain *domain=[[[MetalPageDomain alloc] init] autorelease];domain->publication=YES;
        domain->rows=[rows retain];
        domain->expected=[[NSMutableData alloc] initWithLength:domain->rows.count*sizeof(uint32_t)];
        domain->completed=[[NSMutableData alloc] initWithLength:(domain->rows.count+1)*sizeof(uint32_t)];
        domain->contributions=[[NSMutableArray alloc] initWithCapacity:commands.count];
        domain->publishedSlots=[PageSlots(commands.count) retain];
        for(MetalPageCommand *command in commands)[domain->contributions addObject:PageSlots(command->groups)];
        PageCoverageAnnouncements(commands,domain);
        for(NSUInteger c=0;c<commands.count;c++) {
            MetalPageCommand *command=commands[c];
            for(MetalPageContribution *part in command->contributions) {
                if(![PageWord(part->word) isEqual:word])continue;
                for(MetalPageAccess *access in command->accesses)
                    if(access->phase==part->phase&&(access->access&MetalReachWritten)&&access->slot>=0)
                        [domain->publishedSlots[c] addIndex:(NSUInteger)access->slot];
            }
        }
        NSMutableArray *inputs=[NSMutableArray array];
        for(NSUInteger cut=0;cut<domain->rows.count;cut++) {
            MetalPageTransfer *t=domain->rows[cut];isl_set *all=PageRange(t->selection);
            /* Physical routing may split one semantic coverage cut across
               cables and packet boundaries. Its declared count is shared by
               those children, so each child retains the same exact semantic
               contributor domain, without counting a possible-address union. */
            if(t->coverage){isl_set_free(all);all=isl_set_copy([coverageSelections[PageSemanticCut(t)] pointerValue]);}
            isl_set *remaining=isl_set_copy(all);
            for(NSUInteger c=commands.count;c-->0;) {
                MetalPageCommand *command=commands[c];if(command->order>=t->order)continue;
                for(NSUInteger phase=command->sites.count+1;phase-->0;) {
                BOOL indexed=NO;
                for(MetalPageContribution *part in command->contributions)
                    indexed|=part->phase==phase&&[PageWord(part->word) isEqual:word]&&[part->coverage isEqual:t->coverage];
                for(MetalPageAccess *access in command->accesses)if(access->phase==phase&&[access->root isEqual:t->root]&&(access->access&MetalReachWritten)) {
                    isl_set *groups=PageDomain(access->relation,remaining);
                    if(!PageEmpty(groups)&&access->slot>=0)[domain->publishedSlots[c] addIndex:(NSUInteger)access->slot];
                    if(!indexed)PageGroups(groups,^(NSUInteger group){[domain->contributions[c][group] addIndex:cut];});isl_set_free(groups);
                }
                for(MetalPageAccess *access in command->accesses)if(access->phase==phase&&[access->root isEqual:t->root]&&(access->access&MetalReachWritten))
                    remaining=isl_set_subtract(remaining,PageRange(access->relation));
                }
            }
            if(!t->coverage) {
                uint32_t expected=0;
                for(NSArray *groups in domain->contributions)for(NSIndexSet *cuts in groups)expected+=[cuts containsIndex:cut];
                ((uint32_t *)domain->expected.mutableBytes)[cut]=expected;
                ((uint32_t *)domain->completed.mutableBytes)[cut]=expected?5u:7u;
            }
            if(!PageEmpty(remaining)) {
                isl_set *indices=PageDomain(t->selection,remaining);size_t count=0;
                int64_t *runs=MeshIndexSetRuns(indices,&count);
                for(size_t r=0;r<count;r++)for(int64_t j=0;j<runs[4*r+3];j++) {
                    NSUInteger first=(NSUInteger)(runs[4*r]+j*runs[4*r+2]),bytes=(NSUInteger)runs[4*r+1];
                    isl_set *span=PageInterval(indices,first,bytes);
                    isl_set *unbound=isl_map_range(isl_map_intersect_domain(isl_map_copy(t->selection),span));
                    NSMutableArray *dependencies=[NSMutableArray array];
                    for(MetalPageTransfer *arrival in arrivals) {
                        if(arrival->order>=t->order||![arrival->root isEqual:t->root])continue;
                        isl_set *covered=isl_set_intersect(isl_set_copy(unbound),PageRange(arrival->selection));
                        if(!PageEmpty(covered)){[dependencies addObject:arrival];unbound=isl_set_subtract(unbound,isl_set_copy(covered));}
                        isl_set_free(covered);
                    }
                    isl_set_free(unbound);
                    MetalPageInitial *input=[[[MetalPageInitial alloc] init] autorelease];input->transfer=[t retain];
                    input->first=first;input->bytes=bytes;input->observations=[dependencies copy];
                    [initial addObject:input];[inputs addObject:input];
                }
                mesh_isl_free(runs);isl_set_free(indices);
            }
            isl_set_free(remaining);isl_set_free(all);
        }
        domain->initial=[inputs copy];[domains addObject:domain];
    }
    NSMutableDictionary *consumed=[NSMutableDictionary dictionary];
    for(id key in observers) {
        MetalPageDomain *domain=[[[MetalPageDomain alloc] init] autorelease];domain->rows=[observers[key] copy];
        domain->thresholds=[[NSMutableArray alloc] initWithCapacity:commands.count];
        NSMutableArray *selected=[NSMutableArray arrayWithCapacity:commands.count];
        domain->observedSlots=[PageSlots(commands.count) retain];
        for(MetalPageCommand *command in commands) {
            [domain->thresholds addObject:[NSMutableData dataWithLength:command->groups*sizeof(uint64_t)]];
            NSMutableData *rows=[NSMutableData dataWithLength:command->groups*sizeof(int64_t)];
            memset(rows.mutableBytes,0xff,rows.length);[selected addObject:rows];
        }
        for(NSUInteger row=0;row<domain->rows.count;row++) {
            MetalPageTransfer *t=domain->rows[row];
            PageObservationGroups(commands,t,^(NSUInteger c,NSUInteger phase,MetalPageAccess *access,isl_set *groups) {
                (void)phase;
                if(access->slot>=0)[domain->observedSlots[c] addIndex:(NSUInteger)access->slot];
                uint64_t *thresholds=domain->thresholds[c].mutableBytes;
                int64_t *indices=[selected[c] mutableBytes];
                PageGroups(groups,^(NSUInteger group){
                    if(indices[group]<0||t->effect.threshold>thresholds[group]) {
                        thresholds[group]=t->effect.threshold;indices[group]=(int64_t)row;
                    }
                });
            });
        }
        domain->observationRows=[selected copy];
        uint64_t greatest=0;
        for(NSData *row in domain->thresholds)for(NSUInteger i=0;i<row.length/sizeof(uint64_t);i++)greatest=MAX(greatest,((const uint64_t *)row.bytes)[i]);
        NSNumber *word=PageWord(domain->rows[0]->effect.word);consumed[word]=@(MAX(greatest,[consumed[word] unsignedLongLongValue]));
        [domains addObject:domain];
    }
    NSMutableDictionary *terminal=[NSMutableDictionary dictionary];
    for(MetalPageTransfer *t in arrivals) {
        if(t->effect.operand==ncclMeshPageInternal)continue;
        NSNumber *word=PageWord(t->effect.word);MetalPageTransfer *prior=terminal[word];
        if(t->effect.threshold>[consumed[word] unsignedLongLongValue]&&(!prior||t->effect.threshold>prior->effect.threshold))terminal[word]=t;
    }
    result->initial=[initial copy];result->results=[terminal.allValues copy];result->domains=[domains copy];
    result->solved=[PageSourceIncidence(result) retain];
    for(NSValue *selected in coverageSelections.allValues)isl_set_free(selected.pointerValue);
    [arrivals release];return result;
}

MetalPageEffects *MetalPageEffectsRebind(MetalPageEffects *declaration,
    NSArray<MetalPageCommand *> *given,NSArray<MetalPageTransfer *> *transfers) {
    if(given.count!=declaration->declarationCommands.count||transfers.count!=declaration->declarationTransfers.count)
        PageFailure(@"occurrence does not bind the retained declaration's command/transfer ordinals");
    NSMutableDictionary *commands=[NSMutableDictionary dictionary],*bindings=[NSMutableDictionary dictionary];
    for(NSUInteger i=0;i<given.count;i++)commands[[NSValue valueWithPointer:declaration->declarationCommands[i]]]=given[i];
    for(NSUInteger i=0;i<transfers.count;i++)bindings[[NSValue valueWithPointer:declaration->declarationTransfers[i]]]=transfers[i];
    NSArray *(^bound)(NSArray *)=^NSArray *(NSArray *values) {
        NSMutableArray *made=[NSMutableArray arrayWithCapacity:values.count];
        for(id value in values)[made addObject:bindings[[NSValue valueWithPointer:value]]];return made;
    };
    MetalPageEffects *result=[[[MetalPageEffects alloc] init] autorelease];
    result->declarationCommands=[given copy];result->declarationTransfers=[transfers copy];
    NSMutableArray *ordered=[NSMutableArray arrayWithCapacity:given.count];
    for(MetalPageCommand *command in declaration->commands)[ordered addObject:commands[[NSValue valueWithPointer:command]]];
    result->commands=[ordered copy];result->results=[bound(declaration->results) copy];
    NSMutableDictionary *initials=[NSMutableDictionary dictionary];NSMutableArray *initial=[NSMutableArray array];
    for(MetalPageInitial *original in declaration->initial) {
        MetalPageInitial *input=[[[MetalPageInitial alloc] init] autorelease];
        input->transfer=[bindings[[NSValue valueWithPointer:original->transfer]] retain];
        input->first=original->first;input->bytes=original->bytes;input->observations=[bound(original->observations) copy];
        initials[[NSValue valueWithPointer:original]]=input;[initial addObject:input];
    }
    result->initial=[initial copy];NSMutableArray *domains=[NSMutableArray array];
    for(MetalPageDomain *original in declaration->domains) {
        MetalPageDomain *domain=[[[MetalPageDomain alloc] init] autorelease];
        domain->publication=original->publication;domain->rows=[bound(original->rows) copy];
        domain->announcements=[original->announcements retain];domain->activity=[original->activity retain];
        domain->contributions=[original->contributions retain];domain->observedSlots=[original->observedSlots retain];
        domain->publishedSlots=[original->publishedSlots retain];domain->observationRows=[original->observationRows retain];
        domain->expected=[original->expected mutableCopy];domain->completed=[original->completed mutableCopy];
        NSMutableArray *inputs=[NSMutableArray array];for(id input in original->initial)[inputs addObject:initials[[NSValue valueWithPointer:input]]];
        domain->initial=[inputs copy];
        if(!domain->publication) {
            domain->thresholds=[[NSMutableArray alloc] initWithCapacity:given.count];
            for(NSData *selected in domain->observationRows) {
                NSMutableData *values=[NSMutableData dataWithLength:selected.length];
                const int64_t *rows=selected.bytes;uint64_t *thresholds=values.mutableBytes;
                for(NSUInteger group=0;group<selected.length/sizeof *rows;group++)
                    thresholds[group]=rows[group]<0?0:domain->rows[(NSUInteger)rows[group]]->effect.threshold;
                [domain->thresholds addObject:values];
            }
        }
        [domains addObject:domain];
    }
    result->domains=[domains copy];result->solved=[declaration->solved retain];return result;
}

@interface MetalPageComposition : NSObject {
@public ncclMeshPageTable_t table;
}
@end
@implementation MetalPageComposition
- (void)dealloc { if(table)ncclMeshPageTableDestroy(table);[super dealloc]; }
@end

static id<MTLBuffer> PageData(id<MTLDevice> device,NSData *data) {
    id<MTLBuffer> buffer=[MetalLeaseBuffer(device,data.length) autorelease];
    if(data.length)memcpy(buffer.contents,data.bytes,data.length);return buffer;
}
static NSArray *PageDomainPages(MetalPageDomain *domain,id<MTLDevice> device) {
    @synchronized(domain) {
        if(!domain->pages) {
            NSMutableData *thresholds=[NSMutableData dataWithLength:domain->rows.count*sizeof(uint64_t)];
            for(NSUInteger i=0;i<domain->rows.count;i++)((uint64_t *)thresholds.mutableBytes)[i]=domain->rows[i]->effect.threshold;
            NSData *arrivals=[NSMutableData dataWithLength:domain->rows.count*sizeof(uint32_t)];
            domain->pages=[@[PageData(device,thresholds),PageData(device,domain->completed),
                PageData(device,domain->expected),
                PageData(device,arrivals)] retain];
        }
        return domain->pages;
    }
}

/* Only preparation has a variable number of fields. The numerical entry sees
   one page operand and fixed byte offsets. Page composition retains existing
   storage and does not copy the observed or published words. */
@interface MetalPageFields : NSObject {
@public NSMutableArray *fields,*owners;
    NSMutableArray *offsetTables;
    NSMutableData *reaches;
    NSUInteger composedOffset;
    NSMutableDictionary<NSArray *,NSNumber *> *fieldIndex;
}
- (NSUInteger)add:(id<MTLBuffer>)buffer offset:(NSUInteger)offset bytes:(NSUInteger)bytes access:(uint32_t)access;
- (id<MTLBuffer>)compose;
@end
@implementation MetalPageFields
- (instancetype)init {
    if((self=[super init])){fields=[NSMutableArray new];owners=[NSMutableArray new];offsetTables=[NSMutableArray new];reaches=[NSMutableData new];
        fieldIndex=[NSMutableDictionary new];}return self;
}
- (void)dealloc { [fields release];[owners release];[offsetTables release];[reaches release];[fieldIndex release];[super dealloc]; }
- (NSUInteger)add:(id<MTLBuffer>)buffer offset:(NSUInteger)offset bytes:(NSUInteger)bytes access:(uint32_t)access {
    const struct MetalReach reach={buffer,offset,bytes,access,-1};[reaches appendBytes:&reach length:sizeof reach];
    /* a field already added, found by its buffer, offset and bytes (each field was compared with every one before it,
       fields squared a command) */
    NSArray *key=@[[NSValue valueWithPointer:(__bridge const void *)buffer],@(offset),@(bytes)];
    NSNumber *prior=fieldIndex[key];if(prior)return prior.unsignedIntegerValue;
    fieldIndex[key]=@(fields.count);
    [fields addObject:@[buffer,@(offset),@(bytes)]];[owners addObject:buffer];return fields.count-1;
}
- (id<MTLBuffer>)compose {
    NSMutableData *selection=[NSMutableData dataWithLength:fields.count*sizeof(ncclMeshPageSelection)];
    ncclMeshPageSelection *selected=selection.mutableBytes;size_t cursor=0,alignment=0;
    for(NSUInteger i=0;i<fields.count;i++) {
        NSArray *field=fields[i];id<MTLBuffer> buffer=field[0];const size_t offset=[field[1] unsignedIntegerValue],bytes=[field[2] unsignedIntegerValue];
        MetalPageComposition *source=[[[MetalPageComposition alloc] init] autorelease];
        /* a Metal table is created from the field's {buffer, offset} descriptor (nccl.h ncclMeshPageTableCreate: metal 1
           reads `local` as an ncclMeshBuffer). Changed by the LSTM demonstration session: 7d3aa6cf passed the buffer's
           contents, whose first bytes were then read as a buffer and faulted in address_of on ranks 0 and 2 of the
           8-step LSTM (ncclMeshPageTableCreate + 152 beneath -[MetalPageFields compose], 2026-11-01 19:2x). What must
           stay: the descriptor, its offset 0, the table's base the buffer's. */
        const ncclMeshBuffer whole={(__bridge void *)buffer,0};
        if(ncclMeshPageTableCreate(&whole,NULL,1,&source->table))PageFailure([NSString stringWithFormat:@"cannot bind metadata pages: %s",ncclGetLastError(NULL)]);
        size_t pages=0,width=0;if(ncclMeshPageTableShape(source->table,&pages,&width))PageFailure([NSString stringWithFormat:@"metadata page shape: %s",ncclGetLastError(NULL)]);
        if(alignment&&width!=alignment)PageFailure(@"metadata fields have different local page widths");alignment=width;
        const size_t first=offset/width,count=(offset%width+bytes+width-1)/width;
        selected[i]=(ncclMeshPageSelection){source->table,first,count};
        fields[i]=@[buffer,@(offset),@(bytes),@(cursor*width+offset%width)];cursor+=count;[owners addObject:source];
    }
    for(NSArray *table in offsetTables) {
        id<MTLBuffer> buffer=table[0];NSArray *indices=table[1];uint64_t *words=buffer.contents;
        for(NSUInteger i=0;i<indices.count;i++)words[i]=[fields[[indices[i][0] unsignedIntegerValue]][3] unsignedLongLongValue]+[indices[i][1] unsignedLongLongValue];
    }
    MetalPageComposition *combined=[[[MetalPageComposition alloc] init] autorelease];
    if(ncclMeshPageTableCompose(selected,fields.count,alignment,NULL,1,&combined->table))PageFailure([NSString stringWithFormat:@"cannot compose metadata pages: %s",ncclGetLastError(NULL)]);
    ncclMeshBuffer bound={0};if(ncclMeshPageTableBinding(combined->table,&bound))PageFailure([NSString stringWithFormat:@"cannot bind composed metadata: %s",ncclGetLastError(NULL)]);
    composedOffset=bound.offset;[owners addObject:combined];return (__bridge id<MTLBuffer>)bound.buffer;
}
@end

static NSString *PageField(NSUInteger field) { return [NSString stringWithFormat:@"MESH_EFFECT_FIELD_%lu",(unsigned long)field]; }
static NSString *PageWordField(MetalPageFields *fields,ncclMeshBuffer word,uint32_t access) {
    return PageField([fields add:(__bridge id<MTLBuffer>)word.buffer offset:word.offset bytes:sizeof(uint64_t) access:access]);
}
static NSString *PageViolationField(MetalPageFields *fields,ncclMeshBuffer word) {
    id<MTLBuffer> buffer=(__bridge id<MTLBuffer>)word.buffer;
    return PageField([fields add:buffer offset:word.offset bytes:buffer.length-word.offset access:MetalReachRead|MetalReachWritten|MetalReachAtomic]);
}
/* every MESH_EFFECT_FIELD_<n> of the text its field's pointer, in one pass, each token's whole number read. Changed by
   the LSTM demonstration session: each field rescanned the whole text, fields x text a command, the 64-step LSTM's
   source generation's whole sample (2026-11-01). What must stay: a token of no field kept as written. */
static NSString *PageReplaceFields(NSString *text,MetalPageFields *fields,NSString *base) {
    static NSString *const prefix=@"MESH_EFFECT_FIELD_";
    const NSUInteger length=text.length,count=fields->fields.count;
    NSMutableString *out=[NSMutableString stringWithCapacity:length];
    for(NSUInteger at=0;at<length;) {
        const NSRange found=[text rangeOfString:prefix options:NSLiteralSearch range:NSMakeRange(at,length-at)];
        if(found.location==NSNotFound){[out appendString:[text substringFromIndex:at]];break;}
        [out appendString:[text substringWithRange:NSMakeRange(at,found.location-at)]];
        NSUInteger end=NSMaxRange(found),value=0;BOOL digits=NO;
        for(unichar ch;end<length&&(ch=[text characterAtIndex:end])>='0'&&ch<='9';end++){value=value*10+(ch-'0');digits=YES;}
        if(digits&&value<count)[out appendFormat:@"(%@ + %@ul)",base,fields->fields[value][3]];
        else [out appendString:[text substringWithRange:NSMakeRange(found.location,end-found.location)]];
        at=end;
    }
    return out;
}
static NSArray<NSNumber *> *PageIndices(NSIndexSet *set) {
    NSMutableArray *values=[NSMutableArray array];[set enumerateIndexesUsingBlock:^(NSUInteger i,BOOL *stop){(void)stop;[values addObject:@(i)];}];return values;
}

static void PageCoverageSource(MetalPageDomain *domain,MetalPageCommand *command,NSUInteger index,
    id<MTLDevice> device,MetalPageFields *fields,NSMutableString *helpers,NSMutableArray *after,NSUInteger *ordinal) {
    NSMutableArray<MetalPageContribution *> *contributions=[NSMutableArray array];
    NSMutableArray<NSDictionary *> *compiled=[NSMutableArray array];
    NSNumber *wordIdentity=PageWord(domain->rows[0]->effect.word);
    for(MetalPageContribution *part in command->contributions)
        if([PageWord(part->word) isEqual:wordIdentity])[contributions addObject:part];
    NSMutableSet *coverages=[NSMutableSet setWithObject:NSNull.null];
    for(MetalPageTransfer *row in domain->rows)if(row->coverage)[coverages addObject:row->coverage];
    for(NSUInteger phase=0;phase<after.count;phase++)for(id coverage in coverages) {
        NSUInteger first=phase?[command->sites[phase-1][@"first"] unsignedIntegerValue]:0;
        NSUInteger end=phase?first+[command->sites[phase-1][@"count"] unsignedIntegerValue]:
            command->sites.count?[command->sites[0][@"first"] unsignedIntegerValue]:command->groups;
        NSMutableArray *rows=[NSMutableArray array];NSUInteger width=0;
        for(NSUInteger g=first;g<end;g++) {
            NSMutableArray *row=[NSMutableArray array];
            for(NSNumber *cut in PageIndices(domain->contributions[index][g])) {
                id selected=domain->rows[cut.unsignedIntegerValue]->coverage?:NSNull.null;
                if([selected isEqual:coverage])[row addObject:cut];
            }
            [rows addObject:row];width=MAX(width,row.count);
        }
        if(width)[compiled addObject:@{@"phase":@(phase),@"first":@(first),@"rows":rows,@"width":@(width),@"coverage":coverage}];
    }
    NSMutableArray<NSMutableDictionary *> *announcements=[NSMutableArray array];NSUInteger announcementWidth=0;
    for(NSUInteger group=0;group<command->groups;group++) {
        NSMutableDictionary *cuts=[NSMutableDictionary dictionary];
        for(NSNumber *cut in PageIndices(domain->announcements[index][group]))cuts[cut]=@1;
        for(NSNumber *cut in PageIndices(domain->activity[index][group]))cuts[cut]=@([cuts[cut] unsignedIntValue]|4u);
        [announcements addObject:cuts];announcementWidth=MAX(announcementWidth,cuts.count);
    }
    if(!contributions.count&&!compiled.count&&!announcementWidth)return;
    for(MetalPageTransfer *row in domain->rows)if(row->coverage)[fields->owners addObject:row->coverage];
    for(MetalPageAnnouncement *part in command->announcements)[fields->owners addObject:part->coverage];
    BOOL gpu=command->backend==MetalPageMetal;NSUInteger count=domain->rows.count;
    NSArray *pages=PageDomainPages(domain,device);
    NSString *thresholds=PageField([fields add:pages[0] offset:0 bytes:count*sizeof(uint64_t) access:MetalReachRead]);
    NSString *state=PageField([fields add:pages[1] offset:0 bytes:(count+1)*sizeof(uint32_t)
        access:MetalReachRead|MetalReachWritten|MetalReachAtomic]);
    NSString *arrivals=PageField([fields add:pages[3] offset:0 bytes:count*sizeof(uint32_t)
        access:MetalReachRead|MetalReachWritten|MetalReachAtomic]);
    NSMutableArray *projection=[NSMutableArray array];BOOL dynamic=NO;
    for(NSUInteger cut=0;cut<count;cut++) {
        MetalPageTransfer *row=domain->rows[cut];ncclMeshBuffer value={pages[2],0};NSUInteger cells=count,at=cut;
        if(row->coverage){value=row->coverage->declaration.expected;cells=row->coverage->declaration.cuts;at=row->coverageCut;dynamic=YES;}
        NSUInteger field=[fields add:(__bridge id<MTLBuffer>)value.buffer offset:value.offset bytes:cells*sizeof(uint32_t)
            access:MetalReachRead|MetalReachObserve];
        [projection addObject:@[@(field),@(at*sizeof(uint32_t))]];
    }
    NSString *ordinals=nil,*totals=nil;
    if(dynamic) {
        id<MTLBuffer> projected=PageData(device,[NSMutableData dataWithLength:count*sizeof(uint64_t)]);
        [fields->offsetTables addObject:@[projected,projection]];
        ordinals=PageField([fields add:projected offset:0 bytes:count*sizeof(uint64_t) access:MetalReachRead]);
    } else totals=PageField([projection[0][0] unsignedIntegerValue]);
    NSString *word=PageWordField(fields,domain->rows[0]->effect.word,MetalReachWritten|MetalReachAtomic);
    NSString *prefix=[NSString stringWithFormat:@"MeshPageCoveragePrefix%lu",(unsigned long)(*ordinal)++];
    [helpers appendFormat:gpu?@"struct %@ { device const ulong *cuts; volatile coherent(system) device ulong *word; void at(uint p) {mesh_publish(word,cuts[p-1]);} };\n":
        @"struct %@ { const uint64_t *cuts; volatile uint64_t *word; void at(uint32_t p) {mesh_publish(word,cuts[p-1]);} };\n",prefix];
    NSString *(^contribute)(NSString *,NSString *,NSUInteger,BOOL,NSString *)=^NSString *(NSString *cut,NSString *flags,NSUInteger delta,BOOL known,NSString *fixedAddress) {
        NSString *address=fixedAddress?:dynamic?[NSString stringWithFormat:gpu?@"(mesh_effects+((device const ulong *)%@)[%@])":
            @"(mesh_effects+((const uint64_t *)%@)[%@])",ordinals,cut]:
            [NSString stringWithFormat:@"(%@+4ul*%@)",totals,cut];
        if(gpu)return [NSString stringWithFormat:
            @"mesh_graph_coverage((volatile coherent(system) device atomic_uint *)%@+%@,(volatile coherent(system) device atomic_uint *)%@+%@,(volatile coherent(system) device const uint *)%@,%@,%luu,%@);\n",
            state,cut,arrivals,cut,address,flags,(unsigned long)delta,known?@"true":@"false"];
        return [NSString stringWithFormat:
            @"mesh_graph_coverage((uint32_t *)%@+%@,(uint32_t *)%@+%@,(const uint32_t *)%@,%@,%luu,%uu);\n",
            state,cut,arrivals,cut,address,flags,(unsigned long)delta,known?1u:0u];
    };
    NSMutableIndexSet *phases=[NSMutableIndexSet indexSet];
    for(NSDictionary *part in compiled) {
        NSUInteger phase=[part[@"phase"] unsignedIntegerValue],first=[part[@"first"] unsignedIntegerValue];
        NSArray *rows=part[@"rows"];BOOL known=part[@"coverage"]==NSNull.null;
        for(NSUInteger column=0;column<[part[@"width"] unsignedIntegerValue];column++) {
            BOOL affine=YES;int64_t base=0,step=0,expectedBase=0,expectedStep=0;
            for(NSUInteger g=0;g<rows.count;g++) {
                if(column>=[rows[g] count]){affine=NO;break;}
                int64_t cut=[rows[g][column] longLongValue];
                int64_t expected=known?cut:(int64_t)domain->rows[(NSUInteger)cut]->coverageCut;
                if(!g){base=cut;expectedBase=expected;}
                else if(g==1){step=cut-base;expectedStep=expected-expectedBase;}
                else affine&=cut==base+(int64_t)g*step&&expected==expectedBase+(int64_t)g*expectedStep;
            }
            if(affine) {
                NSString *cell=[NSString stringWithFormat:@"((long)mesh_effect_group-%ldl)",(long)first];
                NSString *cut=step?[NSString stringWithFormat:@"(%lldl+%lldl*%@)",(long long)base,(long long)step,cell]:
                    [NSString stringWithFormat:@"%lldl",(long long)base];
                NSString *expected=expectedStep?[NSString stringWithFormat:@"(%lldl+%lldl*%@)",(long long)expectedBase,(long long)expectedStep,cell]:
                    [NSString stringWithFormat:@"%lldl",(long long)expectedBase];
                NSUInteger field=[projection[(NSUInteger)base][0] unsignedIntegerValue];
                NSString *address=[NSString stringWithFormat:@"(%@+4ul*%@)",PageField(field),expected];
                [after[phase] appendString:contribute(cut,@"0u",1,known,address)];
            } else {
                NSMutableData *table=[NSMutableData dataWithLength:rows.count*sizeof(int64_t)];
                int64_t *cuts=table.mutableBytes;
                for(NSUInteger g=0;g<rows.count;g++)cuts[g]=column<[rows[g] count]?[rows[g][column] longLongValue]:-1;
                id<MTLBuffer> buffer=PageData(device,table);
                NSString *indices=PageField([fields add:buffer offset:0 bytes:table.length access:MetalReachRead]);
                [after[phase] appendFormat:gpu?@"{const long cut=((device const long *)%@)[mesh_effect_group-%luul];if(cut>=0){\n":
                    @"{const int64_t cut=((const int64_t *)%@)[mesh_effect_group-%luul];if(cut>=0){\n",indices,(unsigned long)first];
                [after[phase] appendString:contribute(@"cut",@"0u",1,known,nil)];[after[phase] appendString:@"}}\n"];
            }
        }
        [phases addIndex:phase];
    }
    for(MetalPageContribution *part in contributions) {
        [fields->owners addObject:part];
        NSUInteger first=part->phase?[command->sites[part->phase-1][@"first"] unsignedIntegerValue]:0;
        NSUInteger cells=part->phase?[command->sites[part->phase-1][@"count"] unsignedIntegerValue]:
            command->sites.count?[command->sites[0][@"first"] unsignedIntegerValue]:command->groups;
        if(cells!=part->cells)PageFailure(@"contribution rows must retain the source phase's declared cell domain");
        NSString *indices=PageField([fields add:(__bridge id<MTLBuffer>)part->indices.buffer offset:part->indices.offset
            bytes:part->cells*part->width*sizeof(int64_t) access:MetalReachRead]);
        for(NSUInteger column=0;column<part->width;column++) {
            [after[part->phase] appendFormat:gpu?@"{ const long cut=((device const long *)%@)[(mesh_effect_group-%luul)*%luul+%luul]; if(cut>=0) {\n":
                @"{ const int64_t cut=((const int64_t *)%@)[(mesh_effect_group-%luul)*%luul+%luul]; if(cut>=0) {\n",
                indices,(unsigned long)first,(unsigned long)part->width,(unsigned long)column];
            [after[part->phase] appendString:contribute(@"cut",@"0u",1,part->coverage==nil,nil)];[after[part->phase] appendString:@"} }\n"];
        }
        [phases addIndex:part->phase];
    }
    if(announcementWidth) {
        NSMutableData *table=[NSMutableData dataWithLength:command->groups*announcementWidth*2*sizeof(int64_t)];
        memset(table.mutableBytes,0xff,table.length);
        for(NSUInteger group=0;group<command->groups;group++) {
            NSUInteger column=0;
            for(NSNumber *cut in [announcements[group].allKeys sortedArrayUsingSelector:@selector(compare:)]) {
                int64_t *row=(int64_t *)table.mutableBytes+(group*announcementWidth+column++)*2;
                row[0]=cut.longLongValue;row[1]=[announcements[group][cut] longLongValue];
            }
        }
        id<MTLBuffer> buffer=PageData(device,table);
        NSString *indices=PageField([fields add:buffer offset:0 bytes:table.length access:MetalReachRead]);
        for(NSUInteger phase=0;phase<after.count;phase++) {
            NSUInteger first=phase?[command->sites[phase-1][@"first"] unsignedIntegerValue]:0;
            NSUInteger end=phase?first+[command->sites[phase-1][@"count"] unsignedIntegerValue]:
                command->sites.count?[command->sites[0][@"first"] unsignedIntegerValue]:command->groups;
            BOOL needed=NO;for(NSUInteger g=first;g<end;g++)needed|=announcements[g].count!=0;if(!needed)continue;
            for(NSUInteger column=0;column<announcementWidth;column++) {
                [after[phase] appendFormat:gpu?@"{ device const long *entry=(device const long *)%@+(mesh_effect_group*%luul+%luul)*2ul; const long cut=entry[0]; if(cut>=0) {\n":
                    @"{ const int64_t *entry=(const int64_t *)%@+(mesh_effect_group*%luul+%luul)*2ul; const int64_t cut=entry[0]; if(cut>=0) {\n",
                    indices,(unsigned long)announcementWidth,(unsigned long)column];
                [after[phase] appendString:contribute(@"cut",@"(unsigned int)entry[1]",0,NO,nil)];[after[phase] appendString:@"} }\n"];
            }
            [phases addIndex:phase];
        }
    }
    for(NSNumber *phase in PageIndices(phases))[after[phase.unsignedIntegerValue] appendFormat:gpu?
        @"{ %@ prefix={(device const ulong *)%@,(volatile coherent(system) device ulong *)%@};mesh_graph_settle((device atomic_uint *)%@,0,%luu,(device atomic_uint *)%@+%luu,prefix); }\n":
        @"{ %@ prefix={(const uint64_t *)%@,(volatile uint64_t *)%@};mesh_graph_settle((uint32_t *)%@,0,%luu,(uint32_t *)%@+%luu,prefix); }\n",
        prefix,thresholds,word,state,(unsigned long)count,state,(unsigned long)count];
}
static NSRegularExpression *PageRegex(NSString *pattern) { return [NSRegularExpression regularExpressionWithPattern:pattern options:0 error:NULL]; }
static NSString *PageSub(NSString *source,NSString *pattern,NSString *replacement) {
    return [PageRegex(pattern) stringByReplacingMatchesInString:source options:0 range:NSMakeRange(0,source.length) withTemplate:replacement];
}
NSString *MetalPageSitesPrelude(void) {
    return @MESH_PAGE_SITES_SOURCE;
}
/* Balanced source scopes, skipping comments and quoted literals. This pass
   changes the one declared entry, including every conditional definition and
   its explicitly instantiated template; it does not discover another kernel. */
static NSUInteger PageClose(NSString *text,NSUInteger opening,unichar left,unichar right) {
    NSUInteger depth=1;unichar quote=0;BOOL line=NO,block=NO,escape=NO;
    for(NSUInteger i=opening+1;i<text.length;i++) {
        unichar c=[text characterAtIndex:i],next=i+1<text.length?[text characterAtIndex:i+1]:0;
        if(line){if(c=='\n')line=NO;continue;}
        if(block){if(c=='*'&&next=='/'){block=NO;i++;}continue;}
        if(quote){if(escape)escape=NO;else if(c=='\\')escape=YES;else if(c==quote)quote=0;continue;}
        if(c=='/'&&next=='/'){line=YES;i++;continue;}
        if(c=='/'&&next=='*'){block=YES;i++;continue;}
        if(c=='\''||c=='"'){quote=c;continue;}
        if(c==left)depth++;else if(c==right&&!--depth)return i;
    }
    PageFailure(@"unclosed numerical source scope");return NSNotFound;
}
static NSString *PageNumericalScope(NSString *body) {
    if([body rangeOfString:@"return"].location==NSNotFound)return body;
    NSMutableString *declarations=[NSMutableString string],*inside=[[body mutableCopy] autorelease];
    NSArray *matches=[PageRegex(@"(?m)^\\s*(?:threadgroup|constexpr|typedef|using)\\s+[^;{}]+;") matchesInString:body options:0 range:NSMakeRange(0,body.length)];
    NSMutableArray *outer=[NSMutableArray array];
    for(NSTextCheckingResult *match in matches) {
        NSString *prefix=[body substringToIndex:match.range.location];
        if([prefix componentsSeparatedByString:@"{"].count!=[prefix componentsSeparatedByString:@"}"].count)continue;
        [outer addObject:match];[declarations appendString:[body substringWithRange:match.range]];
    }
    for(NSTextCheckingResult *match in outer.reverseObjectEnumerator)[inside deleteCharactersInRange:match.range];
    return [NSString stringWithFormat:@"%@\n[&]() {%@}();\n",declarations,inside];
}
static NSArray<NSTextCheckingResult *> *PageMetalEntries(NSString *source,NSString *name) {
    NSString *escaped=[NSRegularExpression escapedPatternForString:name];
    NSTextCheckingResult *instance=[PageRegex([NSString stringWithFormat:@"host_name\\(\"%@\"\\).*?decltype\\((\\w+)\\s*<",escaped])
        firstMatchInString:source options:0 range:NSMakeRange(0,source.length)];
    if(instance)name=[source substringWithRange:[instance rangeAtIndex:1]];
    return [PageRegex([NSString stringWithFormat:@"kernel\\s+void\\s+%@\\s*\\(",[NSRegularExpression escapedPatternForString:name]])
        matchesInString:source options:0 range:NSMakeRange(0,source.length)];
}
NSUInteger MetalPageMetadataSlot(NSString *source,NSString *function,NSIndexSet *generatedBindings,NSUInteger capacity) {
    NSMutableIndexSet *occupied=[[generatedBindings mutableCopy] autorelease];
    NSArray *entries=PageMetalEntries(source,function);
    if(!entries.count)PageFailure(@"numerical declaration has no retained source entry");
    for(NSTextCheckingResult *entry in entries) {
        NSUInteger opening=NSMaxRange(entry.range)-1,end=PageClose(source,opening,'(',')');
        NSRange parameters=NSMakeRange(opening+1,end-opening-1);
        for(NSTextCheckingResult *binding in [PageRegex(@"\\[\\[\\s*buffer\\s*\\(\\s*([0-9]+)\\s*\\)\\s*\\]\\]")
            matchesInString:source options:0 range:parameters])
            [occupied addIndex:[[source substringWithRange:[binding rangeAtIndex:1]] integerValue]];
    }
    for(NSUInteger slot=0;slot<capacity;slot++)if(![occupied containsIndex:slot])return slot;
    return NSNotFound;
}
static NSString *PageMetalSource(MetalPageCommand *command,NSString *helpers,NSString *before,NSString *after,
    NSIndexSet *observed,NSIndexSet *published) {
    NSString *source=command->source;
    NSArray *entries=PageMetalEntries(source,command->function);
    if(!entries.count)PageFailure(@"numerical declaration has no retained source entry");
    if(command->metadataSlot==NSNotFound)PageFailure(@"page metadata needs a free declared buffer binding");
    NSMutableIndexSet *coherent=[[observed mutableCopy] autorelease];[coherent addIndexes:published];
    for(NSTextCheckingResult *entry in entries.reverseObjectEnumerator) {
        const NSUInteger opening=NSMaxRange(entry.range)-1,end=PageClose(source,opening,'(',')');
        const NSUInteger body=[source rangeOfString:@"{" options:0 range:NSMakeRange(end,source.length-end)].location;
        if(body==NSNotFound)PageFailure(@"numerical entry has no source body");
        const NSUInteger close=PageClose(source,body,'{','}');
        NSString *parameters=[source substringWithRange:NSMakeRange(opening+1,end-opening-1)];
        NSMutableString *arguments=[[parameters mutableCopy] autorelease],*aliases=[NSMutableString string];
        for(NSArray *item in @[@[@"thread_index_in_threadgroup",@"mesh_lane",@"uint"],@[@"threadgroup_position_in_grid",@"mesh_group",@"uint3"]]) {
            if([before rangeOfString:item[1]].location==NSNotFound&&[after rangeOfString:item[1]].location==NSNotFound)continue;
            NSTextCheckingResult *found=[PageRegex([NSString stringWithFormat:@"\\b(uint[123]?|ushort[123]?)\\s+(\\w+)\\s*\\[\\[%@\\]\\]",item[0]])
                firstMatchInString:parameters options:0 range:NSMakeRange(0,parameters.length)];
            if(found) {
                NSString *variable=[parameters substringWithRange:[found rangeAtIndex:2]];
                if(![variable isEqual:item[1]]) {
                    NSString *type=[parameters substringWithRange:[found rangeAtIndex:1]],*value=variable;
                    if([item[1] isEqual:@"mesh_group"]&&![type hasSuffix:@"3"])value=[NSString stringWithFormat:[type hasSuffix:@"2"]?@"uint3(%@,0)":@"uint3(%@,0,0)",variable];
                    [aliases appendFormat:@"const auto %@=%@;\n",item[1],value];
                }
            } else {
                NSString *type=item[2],*variable=item[1];
                if([item[1] isEqual:@"mesh_group"]) {
                    NSTextCheckingResult *geometry=[PageRegex(@"\\b(uint[123]?|ushort[123]?)\\s+\\w+\\s*\\[\\[(?:thread_position_in_grid|thread_position_in_threadgroup|threads_per_threadgroup|threads_per_grid)\\]\\]")
                        firstMatchInString:parameters options:0 range:NSMakeRange(0,parameters.length)];
                    if(geometry)type=[parameters substringWithRange:[geometry rangeAtIndex:1]];
                    if(![type hasSuffix:@"3"]) {
                        variable=@"mesh_group_argument";
                        [aliases appendFormat:[type hasSuffix:@"2"]?@"const uint3 mesh_group=uint3(%@,0);\n":@"const uint3 mesh_group=uint3(%@,0,0);\n",variable];
                    }
                }
                [arguments appendFormat:@"%@%@ %@ [[%@]]",arguments.length?@", ":@"",type,variable,item[0]];
            }
        }
        [arguments appendFormat:@", device uchar *mesh_effects [[buffer(%lu)]]",(unsigned long)command->metadataSlot];
        if(!command->coherentAccesses)for(NSNumber *slot in PageIndices(coherent)) {
            NSString *name=slot.unsignedIntegerValue<command->argumentNames.count?command->argumentNames[slot.unsignedIntegerValue]:nil;
            NSString *target=name.length?[NSString stringWithFormat:@"[^,]*?\\b%@\\b[^,]*",[NSRegularExpression escapedPatternForString:name]]:
                [NSString stringWithFormat:@"[^,]*?\\[\\[buffer\\(%@\\)\\]\\]",slot];
            NSString *constant=[NSString stringWithFormat:@"\\bconstant\\s+(%@)",target];
            NSString *device=[NSString stringWithFormat:@"(?<!coherent\\(system\\) )\\bdevice\\s+(%@)",target];
            arguments=[[PageSub(PageSub(arguments,constant,@"volatile coherent(system) device const $1"),device,@"volatile coherent(system) device $1") mutableCopy] autorelease];
        }
        NSString *numerical=PageNumericalScope([[source substringWithRange:NSMakeRange(body+1,close-body-1)]
            stringByReplacingOccurrencesOfString:@"MESH_PAGE_OPERAND" withString:@"mesh_effects"]);
        NSTextCheckingResult *template=[PageRegex(@"template\\s*<[^;{}]*>\\s*$") firstMatchInString:source options:0 range:NSMakeRange(0,entry.range.location)];
        NSUInteger declaration=template?template.range.location:entry.range.location;
        NSString *guard=PageSub(command->function,@"[^A-Za-z0-9_]",@"_");
        NSString *prefix=[NSString stringWithFormat:@"#ifndef MESH_PAGE_EFFECTS_%@\n#define MESH_PAGE_EFFECTS_%@\n%@\n#endif\n",guard,guard,helpers];
        source=[NSString stringWithFormat:@"%@%@%@%@%@%@%@%@%@%@",[source substringToIndex:declaration],prefix,
            [source substringWithRange:NSMakeRange(declaration,opening+1-declaration)],arguments,
            [source substringWithRange:NSMakeRange(end,body-end+1)],aliases,before,numerical,after,[source substringFromIndex:close]];
    }
    uint64_t reads=0,writes=0;for(NSNumber *slot in PageIndices(observed))if(slot.unsignedIntegerValue<64)reads|=UINT64_C(1)<<slot.unsignedIntegerValue;
    for(NSNumber *slot in PageIndices(published))if(slot.unsignedIntegerValue<64)writes|=UINT64_C(1)<<slot.unsignedIntegerValue;
    return [NSString stringWithFormat:@"#pragma METAL internals : enable\n#undef MESH_OBSERVED_BINDINGS\n#define MESH_OBSERVED_BINDINGS %lluul\n#undef MESH_PUBLISHED_BINDINGS\n#define MESH_PUBLISHED_BINDINGS %lluul\n%@",(unsigned long long)reads,(unsigned long long)writes,source];
}

/* One site relation and body for either target. The backend supplies only its
   participant visibility operation and address-space spelling. Host cells are
   source iterations within one native invocation, not a task scheduler. */
static NSString *PageSites(MetalPageCommand *command,NSMutableString *declarations,
    NSArray<NSString *> *before,NSArray<NSString *> *after) {
    if(!command->sites.count)return @"";
    const BOOL gpu=command->backend==MetalPageMetal;
    NSString *pointer=gpu?@"device unsigned char *":@"unsigned char *";
    NSMutableString *source=[NSMutableString stringWithString:gpu?MetalPageSitesPrelude():@MESH_PAGE_HOST_SITES_SOURCE];
    NSUInteger phase=0;for(NSDictionary *site in command->sites) {phase++;
        NSString *name=site[@"name"];
        [source appendFormat:@"MESH_PAGE_SITE(%@)\ntemplate<> struct mesh_page_site<mesh_page_site_%@> {static inline void enter(%@,unsigned long,unsigned int); static inline void leave(%@,unsigned long,unsigned int); static inline void join(%@,unsigned int);};\n",name,name,pointer,pointer,pointer];
        for(NSString *side in @[@"enter",@"leave"]) {
            NSString *body=[side isEqual:@"enter"]?before[phase]:after[phase];
            [declarations appendFormat:@"inline void mesh_page_site<mesh_page_site_%@>::%@(%@mesh_effects, unsigned long cell, unsigned int lane) { [[maybe_unused]] const unsigned long mesh_effect_group=%@ul+cell;\n",name,side,pointer,site[@"first"]];
            enum MetalPageSiteScope scope=(enum MetalPageSiteScope)[site[@"scope"] unsignedIntegerValue];
            NSString *visibility=gpu?@"atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE));":@"";
            NSString *barrier=scope==MetalPageSiteSimdgroup?@"simdgroup_barrier(mem_flags::mem_device);":@"threadgroup_barrier(mem_flags::mem_device);";
            if(!body.length)[declarations appendString:@"(void)mesh_effects;(void)mesh_effect_group;(void)lane;\n"];
            else if(gpu&&scope!=MetalPageSiteLane) {
                if([side isEqual:@"enter"])[declarations appendFormat:@"if(lane==0){%@}%@%@\n",body,barrier,visibility];
                else [declarations appendFormat:@"%@if(lane==0){%@%@}\n",barrier,visibility,body];
            } else [declarations appendFormat:@"(void)lane;%@%@\n",visibility,body];
            [declarations appendString:@"}\n"];
        }
        enum MetalPageSiteScope join=(enum MetalPageSiteScope)[site[@"join_scope"] unsignedIntegerValue];
        [declarations appendFormat:@"inline void mesh_page_site<mesh_page_site_%@>::join(%@mesh_effects,unsigned int lane) {(void)mesh_effects;(void)lane;",name,pointer];
        if(gpu&&before[phase].length&&join!=MetalPageSiteLane) {
            [declarations appendString:join==MetalPageSiteSimdgroup?@"simdgroup_barrier(mem_flags::mem_device);":@"threadgroup_barrier(mem_flags::mem_device);"];
            [declarations appendString:@"atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE));"];
        }
        [declarations appendString:@"}\n"];
    }
    return source;
}

NSString *MetalPageEffectsCompose(MetalPageCommand *command,NSString *declarations,NSArray<NSString *> *before,NSArray<NSString *> *after,
    NSIndexSet *observed,NSIndexSet *published,NSString *activityInputs,NSString *activityPredicate) {
    NSString *enter=before[0],*leave=after[0];
    NSMutableString *siteDeclarations=[[declarations mutableCopy] autorelease];
    NSString *siteSource=PageSites(command,siteDeclarations,before,after);
    if(command->backend==MetalPageMetal) {
        BOOL indexed=[enter rangeOfString:@"mesh_effect_group"].location!=NSNotFound||[leave rangeOfString:@"mesh_effect_group"].location!=NSNotFound||
            (activityInputs.length&&[activityInputs rangeOfString:@"mesh_effect_group"].location!=NSNotFound);
        NSString *group=indexed?[NSString stringWithFormat:@"const ulong mesh_effect_group=mesh_group.x+%luul*(mesh_group.y+%luul*mesh_group.z);\n",(unsigned long)command->grid.width,(unsigned long)command->grid.height]:@"";
        NSString *visibility=@"atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE));";
        NSString *activation=activityInputs.length?[NSString stringWithFormat:@"if(mesh_lane==0){%@}threadgroup_barrier(mem_flags::mem_device);%@\n",activityInputs,visibility]:@"";
        NSString *guard=activityPredicate.length?[NSString stringWithFormat:@"if(%@){\n",activityPredicate]:@"";
        NSString *starts=[NSString stringWithFormat:@"%@%@%@%@",group,activation,guard,enter.length?
            [NSString stringWithFormat:@"if(mesh_lane==0){%@}threadgroup_barrier(mem_flags::mem_device);%@\n",enter,visibility]:@""];
        NSString *ends=leave.length?[NSString stringWithFormat:@"threadgroup_barrier(mem_flags::mem_device);if(mesh_lane==0){%@%@}\n",visibility,leave]:@"";
        if(activityPredicate.length)ends=[ends stringByAppendingString:@"}\n"];
        NSString *result=PageMetalSource(command,siteDeclarations,starts,ends,observed,published);
        return [siteSource stringByAppendingString:result];
    } else if(command->backend==MetalPageANE) {
        NSString *activation=activityPredicate.length?[NSString stringWithFormat:@"%@if(!(%@))return false;",activityInputs?:@"",activityPredicate]:@"";
        return [NSString stringWithFormat:@"#include \"mesh-publication.h\"\n%@\nstruct MeshCpuArguments {uint8_t *mesh_effects;};\n#define MESH_ANE_ENTER(arguments) ([&](){const auto &a=*static_cast<const MeshCpuArguments *>(arguments);[[maybe_unused]] auto *mesh_effects=a.mesh_effects;[[maybe_unused]] const unsigned long mesh_effect_group=0;%@%@return true;}())\n#define MESH_ANE_LEAVE(arguments) do {const auto &a=*static_cast<const MeshCpuArguments *>(arguments);[[maybe_unused]] auto *mesh_effects=a.mesh_effects;[[maybe_unused]] const unsigned long mesh_effect_group=0;%@} while(0)\n#include \"mesh-ane-entry.inc\"\n",declarations,PageSub(activation,@"\\n",@" "),PageSub(enter,@"\\n",@" "),PageSub(leave,@"\\n",@" ")];
    } else {
        NSString *source=command->source;NSRange structure=[source rangeOfString:@"struct MeshCpuArguments {"];
        if(structure.location==NSNotFound)PageFailure(@"CPU source requires its declared MeshCpuArguments ABI");
        NSUInteger end=[source rangeOfString:@"};" options:0 range:NSMakeRange(NSMaxRange(structure),source.length-NSMaxRange(structure))].location;
        source=[source stringByReplacingCharactersInRange:NSMakeRange(end,0) withString:@"uint8_t *mesh_effects;"];
        NSRange entry=[source rangeOfString:@"extern \"C\" void mesh_cpu_entry("];if(entry.location==NSNotFound)PageFailure(@"CPU source requires its declared numerical entry");
        NSUInteger opening=[source rangeOfString:@"{" options:0 range:NSMakeRange(NSMaxRange(entry),source.length-NSMaxRange(entry))].location;
        NSUInteger close=PageClose(source,opening,'{','}'),binding=NSMaxRange([source rangeOfString:@";" options:0 range:NSMakeRange(opening,close-opening)]);
        NSString *numerical=PageNumericalScope([[source substringWithRange:NSMakeRange(binding,close-binding)]
            stringByReplacingOccurrencesOfString:@"MESH_PAGE_OPERAND" withString:@"mesh_effects"]);
        NSString *activation=activityPredicate.length?[NSString stringWithFormat:@"%@if(%@){\n",activityInputs?:@"",activityPredicate]:@"";
        return [NSString stringWithFormat:@"%@#include \"mesh-publication.h\"\n%@\n%@\n[[maybe_unused]] auto *mesh_effects=a.mesh_effects;[[maybe_unused]] const unsigned long mesh_effect_group=0;\n%@%@%@%@%@%@",siteSource,siteDeclarations,[source substringToIndex:binding],activation,enter,numerical,leave,activityPredicate.length?@"}\n":@"",[source substringFromIndex:close]];
    }
}

/* Predicate availability precedes evaluating that predicate. This is a
   preparation topological order of the already solved, source-local reads;
   independent words retain no edge. */
static NSArray *PageObservationOrder(NSArray *ordinals,NSArray *domains,NSUInteger first,NSUInteger end,NSUInteger command) {
    NSMutableArray *pending=[ordinals mutableCopy],*ordered=[NSMutableArray array];
    while(pending.count) {
        NSUInteger selected=NSNotFound;
        for(NSUInteger i=0;i<pending.count;i++) {
            MetalPageDomain *consumer=domains[[pending[i] unsignedIntegerValue]];MetalPageTransfer *use=consumer->rows[0];
            BOOL blocked=NO;
            if(use->coverage) {
                isl_set *mask=PageActivityRange(use);
                const int64_t *needed=consumer->observationRows[command].bytes;
                for(NSUInteger j=0;j<pending.count&&!blocked;j++)if(j!=i) {
                    MetalPageDomain *producer=domains[[pending[j] unsignedIntegerValue]];
                    const int64_t *ready=producer->observationRows[command].bytes;
                    BOOL overlaps=NO;for(NSUInteger g=first;g<end;g++)overlaps|=needed[g]>=0&&ready[g]>=0;
                    if(!overlaps)continue;
                    for(MetalPageTransfer *t in producer->rows)if(t->observeAccess&MetalReachRead) {
                        isl_set *meet=isl_set_intersect(isl_set_copy(mask),PageRange(t->selection));
                        blocked|=!PageEmpty(meet);isl_set_free(meet);
                    }
                }
                isl_set_free(mask);
            }
            if(!blocked){selected=i;break;}
        }
        if(selected==NSNotFound)PageFailure(@"source-local observation predicates form a dependency cycle");
        [ordered addObject:pending[selected]];[pending removeObjectAtIndex:selected];
    }
    [pending release];return ordered;
}
/* The activity producer and its own predicates are unconditional inputs of a
   masked entry. Retain domain ordinals, not occurrence words or generated
   strings. Rebinding preserves this solved incidence without another query. */
static NSArray *PageActivityInputs(MetalPageCommand *command,NSMutableArray *ordinals,NSArray *domains) {
    if(!command->activityAccess)return @[];
    isl_set *needed=PageRange(command->activityAccess->relation);
    NSMutableArray *inputs=[NSMutableArray array];BOOL changed;
    do {
        changed=NO;
        for(NSNumber *ordinal in [[ordinals copy] autorelease]) {
            MetalPageDomain *domain=domains[ordinal.unsignedIntegerValue];BOOL supplies=NO;
            for(MetalPageTransfer *row in domain->rows)if(row->observeAccess&MetalReachRead) {
                isl_set *meet=isl_set_intersect(isl_set_copy(needed),PageRange(row->selection));
                supplies|=!PageEmpty(meet);isl_set_free(meet);
            }
            if(!supplies)continue;
            for(MetalPageTransfer *row in domain->rows)if(row->coverage)
                needed=isl_set_union(needed,PageActivityRange(row));
            [inputs addObject:ordinal];[ordinals removeObject:ordinal];changed=YES;
        }
    } while(changed);
    isl_set_free(needed);return inputs;
}
static NSArray *PageSourceIncidence(MetalPageEffects *effects) {
    NSMutableArray *compiled=[NSMutableArray arrayWithCapacity:effects->commands.count];
    for(NSUInteger index=0;index<effects->commands.count;index++) {
        MetalPageCommand *command=effects->commands[index];
        NSMutableArray *phases=[NSMutableArray array];
        for(NSUInteger phase=0;phase<=command->sites.count;phase++) {
            NSUInteger first=phase?[command->sites[phase-1][@"first"] unsignedIntegerValue]:0;
            NSUInteger end=phase?first+[command->sites[phase-1][@"count"] unsignedIntegerValue]:
                command->sites.count?[command->sites[0][@"first"] unsignedIntegerValue]:command->groups;
            NSMutableArray *selected=[NSMutableArray array];
            for(NSUInteger ordinal=0;ordinal<effects->domains.count;ordinal++) {
                MetalPageDomain *domain=effects->domains[ordinal];if(domain->publication)continue;
                const int64_t *rows=domain->observationRows[index].bytes;
                BOOL uses=NO;for(NSUInteger g=first;g<end;g++)uses|=rows[g]>=0;
                if(uses)[selected addObject:@(ordinal)];
            }
            [phases addObject:selected];
        }
        NSUInteger entry=command->sites.count?[command->sites[0][@"first"] unsignedIntegerValue]:command->groups;
        NSArray *activity=PageObservationOrder(PageActivityInputs(command,phases[0],effects->domains),
            effects->domains,0,entry,index);
        for(NSUInteger phase=0;phase<phases.count;phase++) {
            NSUInteger first=phase?[command->sites[phase-1][@"first"] unsignedIntegerValue]:0;
            NSUInteger end=phase?first+[command->sites[phase-1][@"count"] unsignedIntegerValue]:entry;
            phases[phase]=PageObservationOrder(phases[phase],effects->domains,first,end,index);
        }
        [compiled addObject:@{@"activity":activity,@"phases":phases}];
    }
    return compiled;
}
MetalPageSource *MetalPageEffectsSource(MetalPageEffects *effects,NSUInteger index,id<MTLDevice> device) {
    MetalPageCommand *command=effects->commands[index];const BOOL gpu=command->backend==MetalPageMetal;
    MetalPageSource *result=[[[MetalPageSource alloc] init] autorelease];
    MetalPageFields *fields=[[[MetalPageFields alloc] init] autorelease];
    NSMutableString *helpers=[NSMutableString string];NSMutableArray *before=[NSMutableArray array],*after=[NSMutableArray array];
    for(NSUInteger phase=0;phase<=command->sites.count;phase++){[before addObject:[NSMutableString string]];[after addObject:[NSMutableString string]];}
    NSMutableDictionary *observations=[NSMutableDictionary dictionary];
    NSMutableIndexSet *observed=[NSMutableIndexSet indexSet],*published=[NSMutableIndexSet indexSet];
    NSUInteger ordinal=0;
    for(NSUInteger domainIndex=0;domainIndex<effects->domains.count;domainIndex++) {
        MetalPageDomain *domain=effects->domains[domainIndex];
        if(!domain->publication) {
            NSData *values=domain->thresholds[index];const uint64_t *thresholds=values.bytes;BOOL needed=NO;
            const int64_t *rows=domain->observationRows[index].bytes;
            for(NSUInteger g=0;g<command->groups;g++)needed|=rows[g]>=0;if(!needed)continue;
            [observed addIndexes:domain->observedSlots[index]];
            MetalPageTransfer *t=domain->rows[0];
            NSString *word=PageWordField(fields,t->effect.word,MetalReachRead|MetalReachObserve);
            NSString *violation=PageViolationField(fields,t->effect.violation);
            BOOL uniform=YES;for(NSUInteger g=1;g<command->groups;g++)uniform&=rows[g]==rows[0];
            /* Occurrence values are ordinary operands. Source text retains
               the solved row/field structure, never absolute counter values
               or the diagnostic entry of a previous realization. */
            const NSUInteger count=uniform?1:command->groups;
            NSMutableData *record=[NSMutableData dataWithLength:(3+count)*sizeof(uint64_t)];
            uint64_t *data=record.mutableBytes;data[0]=t->effect.polls;data[1]=t->effect.channel;data[2]=t->effect.entry;
            memcpy(data+3,thresholds,count*sizeof(uint64_t));
            id<MTLBuffer> table=PageData(device,record);
            NSString *operand=PageField([fields add:table offset:0 bytes:record.length access:MetalReachRead]);
            NSString *array=[NSString stringWithFormat:gpu?@"((device const ulong *)%@)":@"((const uint64_t *)%@)",operand];
            NSString *threshold=[NSString stringWithFormat:@"%@[3ul%@]",array,uniform?@"":@"+mesh_effect_group"];
            NSString *fragment=[NSString stringWithFormat:@"mesh_step_observe(%@%@, %@, %@, (unsigned int)%@[0], (unsigned int)%@[1], %@[2]);\n",
                gpu?@"":@"(volatile uint64_t *)",word,threshold,violation,array,array,array];
            if(t->coverage) {
                ncclMeshBuffer active=t->coverage->declaration.active;active.offset+=t->coverageCut*sizeof(uint32_t);
                NSString *mask=PageField([fields add:(__bridge id<MTLBuffer>)active.buffer offset:active.offset
                    bytes:sizeof(uint32_t) access:MetalReachRead]);
                [fields->owners addObject:t->coverage];
                NSString *predicate=[NSString stringWithFormat:gpu?
                    @"mesh_payload((volatile coherent(system) device const uint *)%@)!=0":
                    @"__atomic_load_n((const uint32_t *)%@,__ATOMIC_ACQUIRE)!=0",mask];
                fragment=[NSString stringWithFormat:@"if(%@){%@}\n",predicate,fragment];
            }
            if(!uniform)fragment=[NSString stringWithFormat:@"if(%@){%@}\n",threshold,fragment];
            observations[@(domainIndex)]=fragment;
            continue;
        }
        [published addIndexes:domain->publishedSlots[index]];
        PageCoverageSource(domain,command,index,device,fields,helpers,after,&ordinal);
        for(MetalPageInitial *input in domain->initial) {
            MetalTensorLayout *flat=[input->transfer->layout reshape:@[@(input->transfer->layout.count)]];
            MetalTensorLayout *view=[flat slice:0 first:(NSInteger)input->first count:input->bytes step:1];
            [fields->reaches appendData:[view reaches:input->transfer->payload base:0 elementBytes:1 access:MetalReachRead binding:-1]];
            [fields->owners addObject:input->transfer->payload];
        }
    }
    NSMutableString *activityInputs=[NSMutableString string];NSString *activityPredicate=@"";
    if(command->activity.buffer) {
        NSString *mask=PageField([fields add:(__bridge id<MTLBuffer>)command->activity.buffer offset:command->activity.offset
            bytes:sizeof(uint32_t) access:MetalReachRead]);
        activityPredicate=[NSString stringWithFormat:gpu?@"mesh_payload((volatile coherent(system) device const uint *)%@)!=0":
            @"__atomic_load_n((const uint32_t *)%@,__ATOMIC_ACQUIRE)!=0",mask];
        for(NSNumber *ordinal in effects->solved[index][@"activity"])
            [activityInputs appendString:observations[ordinal]];
    }
    NSArray *phases=effects->solved[index][@"phases"];
    for(NSUInteger phase=0;phase<before.count;phase++)
        for(NSNumber *ordinal in phases[phase])[before[phase] appendString:observations[ordinal]];
    result->observedSlots=[PageIndices(observed) copy];result->publishedSlots=[PageIndices(published) copy];
    if(!fields->fields.count){result->source=[(command->backend==MetalPageCPU&&command->sites.count?
        [@MESH_PAGE_HOST_SITES_SOURCE stringByAppendingString:command->source]:command->source) copy];result->reaches=[[NSData alloc] init];return result;}
    result->metadata=[[fields compose] retain];result->metadataOffset=fields->composedOffset;
    result->owners=[fields->owners copy];result->reaches=[fields->reaches copy];
    NSString *base=@"mesh_effects";
    NSString *declarations=PageReplaceFields(helpers,fields,base);
    for(NSUInteger phase=0;phase<before.count;phase++) {
        before[phase]=PageReplaceFields(before[phase],fields,base);after[phase]=PageReplaceFields(after[phase],fields,base);
    }
    result->source=[MetalPageEffectsCompose(command,declarations,before,after,observed,published,
        PageReplaceFields(activityInputs,fields,base),PageReplaceFields(activityPredicate,fields,base)) copy];
    return result;
}
