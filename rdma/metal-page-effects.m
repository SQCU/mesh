#import "metal-page-effects.h"
#import "mesh-index-relation.h"
#include "mesh-isl-memory.h"
#include "mesh-page-sites.h"
#include <isl/val.h>
#include <isl/space.h>

id MetalPageRegionRoot(void) { return @"mesh-region"; }

static ncclMeshBuffer PageBuffer(ncclMeshBuffer value) {
    if(!value.buffer&&ncclMeshMemBuffer((void *)value.offset,&value.buffer,&value.offset))
        [NSException raise:NSInvalidArgumentException format:@"page effects: operand has no region page binding"];
    return value;
}

@implementation MetalPageCoverage
+ (instancetype)declaration:(ncclMeshPageCoverage)value {
    MetalPageCoverage *v=[[[self alloc] init] autorelease];
    v->declaration=value;v->declaration.active=PageBuffer(value.active);v->declaration.expected=PageBuffer(value.expected);
    [(__bridge id)v->declaration.active.buffer retain];[(__bridge id)v->declaration.expected.buffer retain];return v;
}
- (void)dealloc { [(__bridge id)declaration.active.buffer release];[(__bridge id)declaration.expected.buffer release];[super dealloc]; }
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
@implementation MetalPageCommand
@synthesize coherentAccesses, argumentNames, frame;
+ (instancetype)order:(NSUInteger)o groups:(NSUInteger)g backend:(enum MetalPageBackend)b source:(NSString *)s
    function:(NSString *)f grid:(MTLSize)d metadataSlot:(NSUInteger)slot accesses:(NSArray<MetalPageAccess *> *)a {
    MetalPageCommand *v=[[[self alloc] init] autorelease];v->order=o;v->groups=g;v->backend=b;
    v->source=[s copy];v->function=[f copy];v->grid=d;v->metadataSlot=slot;v->accesses=[a copy];return v;
}
- (void)dealloc { [source release];[function release];[accesses release];[argumentNames release];[sites release];[contributions release];[announcements release];[super dealloc]; }
@end
void MetalPageCommandSite(MetalPageCommand *command,NSString *name,NSUInteger cells,enum MetalPageSiteScope scope,
    NSArray<MetalPageAccess *> *accesses) {
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
        @"count":@(cells),@"scope":@(scope)}];
    [command->sites release];command->sites=[sites copy];command->groups+=cells;
}
@implementation MetalPageTransfer
@synthesize coverage,coverageCut;
+ (instancetype)owner:(id)o order:(NSUInteger)n effect:(ncclMeshPageEffect)e root:(id)r selection:(isl_map *)s
    payload:(id<MTLBuffer>)p layout:(MetalTensorLayout *)l {
    if(!e.word.buffer&&ncclMeshMemBuffer((void *)e.word.offset,&e.word.buffer,&e.word.offset))
        [NSException raise:NSInvalidArgumentException format:@"page effects: publication word has no region binding"];
    if(!e.violation.buffer&&e.violation.offset&&ncclMeshMemBuffer((void *)e.violation.offset,&e.violation.buffer,&e.violation.offset))
        [NSException raise:NSInvalidArgumentException format:@"page effects: diagnostic operand has no region binding"];
    MetalPageTransfer *v=[[[self alloc] init] autorelease];v->owner=[o retain];v->order=n;v->effect=e;
    [(__bridge id)e.word.buffer retain];[(__bridge id)e.violation.buffer retain];
    v->root=[r retain];v->selection=isl_map_copy(s);v->payload=[p retain];v->layout=[l retain];return v;
}
- (void)dealloc { [owner release];[root release];[payload release];[layout release];[coverage release];
    [(__bridge id)effect.word.buffer release];[(__bridge id)effect.violation.buffer release];isl_map_free(selection);[super dealloc]; }
@end
@implementation MetalPageInitial
- (void)dealloc { [transfer release];[observations release];[super dealloc]; }
@end
@implementation MetalPageEffects
- (void)dealloc { [commands release];[initial release];[results release];[domains release];[solved release];[super dealloc]; }
@end
@implementation MetalPageSource
@synthesize source, metadata, metadataOffset, reaches;
- (void)dealloc { [source release];[metadata release];[observedSlots release];[publishedSlots release];[reaches release];[owners release];[super dealloc]; }
@end

@interface MetalPageDomain : NSObject {
@public NSArray<MetalPageTransfer *> *rows;
    BOOL publication;
    MetalPageCoverage *coverage;
    NSMutableArray<NSMutableArray<NSMutableIndexSet *> *> *announcements;
    NSMutableArray<NSMutableArray<NSMutableIndexSet *> *> *activity;
    NSMutableArray<NSMutableArray<NSMutableIndexSet *> *> *contributions;
    NSMutableArray<NSMutableData *> *thresholds;
    NSMutableArray<NSMutableIndexSet *> *observedSlots,*publishedSlots;
    NSMutableData *expected,*completed;
    NSArray<MetalPageInitial *> *initial;
    NSArray *pages;
}
@end
@implementation MetalPageDomain
- (void)dealloc { [rows release];[contributions release];[thresholds release];[observedSlots release];[publishedSlots release];
    [coverage release];[announcements release];[activity release];
    [expected release];[completed release];[initial release];[pages release];[super dealloc]; }
@end

static void PageFailure(NSString *reason) { [NSException raise:NSInvalidArgumentException format:@"page effects: %@",reason]; }
static isl_set *PageRange(isl_map *relation) { return isl_map_range(isl_map_copy(relation)); }
static BOOL PageEmpty(isl_set *set) {
    isl_bool value=isl_set_is_empty(set);if(value<0)PageFailure(@"invalid relation");return value;
}
static isl_set *PageDomain(isl_map *relation,isl_set *range) {
    return isl_map_domain(isl_map_intersect_range(isl_map_copy(relation),isl_set_copy(range)));
}
static void PageGroups(isl_set *groups,void (^each)(NSUInteger)) {
    size_t count=0;int64_t *runs=MeshIndexSetRuns(groups,&count);
    if(!runs&&!PageEmpty(groups))PageFailure(@"cannot lower dispatch selection");
    for(size_t i=0;i<count;i++)for(int64_t r=0;r<runs[4*i+3];r++)for(int64_t j=0;j<runs[4*i+1];j++)
        each((NSUInteger)(runs[4*i]+r*runs[4*i+2]+j));
    mesh_isl_free(runs);
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
    for(NSUInteger kind=0;kind<2;kind++) {
        ncclMeshBuffer word=kind?domain->coverage->declaration.active:domain->coverage->declaration.expected;
        isl_ctx *context=isl_map_get_ctx(domain->rows[0]->selection);
        isl_map *words=PageCoverageWords(context,word,domain->coverage->declaration.cuts);
        isl_set *selected=isl_set_empty(isl_space_domain(isl_map_get_space(words)));
        for(MetalPageTransfer *row in domain->rows) {
            isl_set *cell=isl_set_universe(isl_set_get_space(selected));
            cell=isl_set_fix_val(cell,isl_dim_set,0,isl_val_int_from_ui(context,row->coverageCut));
            selected=isl_set_union(selected,cell);
        }
        words=isl_map_intersect_domain(words,selected);
        isl_set *remaining=PageRange(words);
        NSMutableArray *target=kind?domain->activity:domain->announcements;
        for(NSUInteger c=commands.count;c-->0;) {
            MetalPageCommand *command=commands[c];if(command->order>=domain->rows[0]->order)continue;
            for(NSUInteger phase=command->sites.count+1;phase-->0;) {
                for(MetalPageAccess *access in command->accesses) {
                    if(access->phase!=phase||![access->root isEqual:MetalPageRegionRoot()]||!(access->access&MetalReachWritten))continue;
                    isl_map *writes=isl_map_intersect_range(isl_map_copy(access->relation),isl_set_copy(remaining));
                    isl_map *cuts=MeshIndexCompose(writes,isl_map_reverse(isl_map_copy(words)));
                    isl_set *groups=isl_map_domain(isl_map_copy(cuts));
                    if(!PageEmpty(groups)&&access->slot>=0)[domain->publishedSlots[c] addIndex:(NSUInteger)access->slot];
                    PageGroups(groups,^(NSUInteger group) {
                        isl_map *selected=isl_map_fix_val(isl_map_copy(cuts),isl_dim_in,0,isl_val_int_from_ui(context,group));
                        isl_set *range=isl_map_range(selected);
                        PageGroups(range,^(NSUInteger cut){
                            for(NSUInteger row=0;row<domain->rows.count;row++)
                                if(domain->rows[row]->coverageCut==cut)[target[c][group] addIndex:row];
                        });isl_set_free(range);
                    });
                    isl_set_free(groups);isl_map_free(cuts);
                }
                for(MetalPageAccess *access in command->accesses)
                    if(access->phase==phase&&[access->root isEqual:MetalPageRegionRoot()]&&(access->access&MetalReachWritten))
                        remaining=isl_set_subtract(remaining,PageRange(access->relation));
            }
            for(MetalPageAnnouncement *announcement in command->announcements) {
                if(announcement->coverage!=domain->coverage||!(announcement->flags&(kind?4u:1u)))continue;
                NSUInteger first=announcement->phase?[command->sites[announcement->phase-1][@"first"] unsignedIntegerValue]:0;
                isl_set *groups=isl_map_domain(isl_map_copy(announcement->relation));
                PageGroups(groups,^(NSUInteger group) {
                    isl_map *selected=isl_map_fix_val(isl_map_copy(announcement->relation),isl_dim_in,0,isl_val_int_from_ui(context,group));
                    isl_set *range=isl_map_range(selected);
                    PageGroups(range,^(NSUInteger cut) {
                        for(NSUInteger row=0;row<domain->rows.count;row++)
                            if(domain->rows[row]->coverageCut==cut)[target[c][first+group] addIndex:row];
                    });isl_set_free(range);
                });isl_set_free(groups);
                isl_map *reached=MeshIndexCompose(isl_map_copy(announcement->relation),isl_map_copy(words));
                remaining=isl_set_subtract(remaining,PageRange(reached));isl_map_free(reached);
            }
        }
        if(!PageEmpty(remaining))PageFailure(@"coverage values need their declared producer or imported-value continuation");
        isl_set_free(remaining);isl_map_free(words);
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


MetalPageEffects *MetalPageEffectsSolve(NSArray<MetalPageCommand *> *given,NSArray<MetalPageTransfer *> *transfers) {
    MetalPageEffects *result=[[[MetalPageEffects alloc] init] autorelease];
    result->commands=[PageSorted(given,^NSComparisonResult(MetalPageCommand *a,MetalPageCommand *b){
        return a->order<b->order?NSOrderedAscending:a->order>b->order?NSOrderedDescending:NSOrderedSame;
    }) retain];
    NSArray *commands=result->commands;
    if(transfers.count) {
        isl_ctx *context=isl_map_get_ctx(((MetalPageTransfer *)transfers[0])->selection);
        for(MetalPageCommand *command in commands)PageCoverageAccesses(command,context);
    }
    NSMutableDictionary *publishers=[NSMutableDictionary dictionary],*observers=[NSMutableDictionary dictionary];
    NSMutableArray *arrivals=[NSMutableArray array];
    for(MetalPageTransfer *t in transfers) {
        if(t->effect.bytes!=t->layout.count)PageFailure(@"transfer layout must name its exact selected bytes");
        NSNumber *word=PageWord(t->effect.word);
        id key=t->effect.kind==ncclMeshPagePublish?word:@[@(t->order),t->owner,@(t->effect.call),word];
        NSMutableDictionary *table=t->effect.kind==ncclMeshPagePublish?publishers:observers;
        NSMutableArray *rows=table[key];if(!rows)table[key]=rows=[NSMutableArray array];[rows addObject:t];
        if(t->effect.kind==ncclMeshPageObserve)[arrivals addObject:t];
    }
    arrivals=[PageSorted(arrivals,^NSComparisonResult(MetalPageTransfer *a,MetalPageTransfer *b){
        return a->order>b->order?NSOrderedAscending:a->order<b->order?NSOrderedDescending:NSOrderedSame;
    }) mutableCopy];
    NSMutableArray *initial=[NSMutableArray array],*domains=[NSMutableArray array];
    for(NSNumber *word in publishers) {
        MetalPageDomain *domain=[[[MetalPageDomain alloc] init] autorelease];domain->publication=YES;
        domain->rows=[PageSorted(publishers[word],^NSComparisonResult(MetalPageTransfer *a,MetalPageTransfer *b){
            return a->effect.threshold<b->effect.threshold?NSOrderedAscending:a->effect.threshold>b->effect.threshold?NSOrderedDescending:NSOrderedSame;
        }) retain];
        domain->coverage=[domain->rows[0]->coverage retain];
        for(MetalPageTransfer *row in domain->rows)
            if(row->coverage!=domain->coverage)PageFailure(@"one publication domain must retain one coverage operand");
        domain->expected=[[NSMutableData alloc] initWithLength:domain->rows.count*sizeof(uint32_t)];
        domain->completed=[[NSMutableData alloc] initWithLength:(domain->rows.count+1)*sizeof(uint32_t)];
        domain->contributions=[[NSMutableArray alloc] initWithCapacity:commands.count];
        domain->publishedSlots=[PageSlots(commands.count) retain];
        for(MetalPageCommand *command in commands)[domain->contributions addObject:PageSlots(command->groups)];
        if(domain->coverage) {
            PageCoverageAnnouncements(commands,domain);
            for(NSUInteger c=0;c<commands.count;c++) {
                MetalPageCommand *command=commands[c];
                for(MetalPageContribution *part in command->contributions) {
                    if(part->coverage!=domain->coverage||![PageWord(part->word) isEqual:word])continue;
                    for(MetalPageAccess *access in command->accesses)
                        if(access->phase==part->phase&&(access->access&MetalReachWritten)&&access->slot>=0)
                            [domain->publishedSlots[c] addIndex:(NSUInteger)access->slot];
                }
            }
            [domains addObject:domain];continue;
        }
        NSMutableArray *inputs=[NSMutableArray array];
        for(NSUInteger cut=0;cut<domain->rows.count;cut++) {
            MetalPageTransfer *t=domain->rows[cut];isl_set *all=PageRange(t->selection),*remaining=isl_set_copy(all);
            for(NSUInteger c=commands.count;c-->0;) {
                MetalPageCommand *command=commands[c];if(command->order>=t->order)continue;
                for(NSUInteger phase=command->sites.count+1;phase-->0;) {
                for(MetalPageAccess *access in command->accesses)if(access->phase==phase&&[access->root isEqual:t->root]&&(access->access&MetalReachWritten)) {
                    isl_set *groups=PageDomain(access->relation,remaining);
                    if(!PageEmpty(groups)&&access->slot>=0)[domain->publishedSlots[c] addIndex:(NSUInteger)access->slot];
                    PageGroups(groups,^(NSUInteger group){[domain->contributions[c][group] addIndex:cut];});isl_set_free(groups);
                }
                for(MetalPageAccess *access in command->accesses)if(access->phase==phase&&[access->root isEqual:t->root]&&(access->access&MetalReachWritten))
                    remaining=isl_set_subtract(remaining,PageRange(access->relation));
                }
            }
            uint32_t expected=0;
            for(NSArray *groups in domain->contributions)for(NSIndexSet *cuts in groups)expected+=[cuts containsIndex:cut];
            ((uint32_t *)domain->expected.mutableBytes)[cut]=expected;
            if(!expected)((uint32_t *)domain->completed.mutableBytes)[cut]=2;
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
        domain->observedSlots=[PageSlots(commands.count) retain];
        for(MetalPageCommand *command in commands)[domain->thresholds addObject:[NSMutableData dataWithLength:command->groups*sizeof(uint64_t)]];
        for(MetalPageTransfer *t in domain->rows) {
            isl_set *remaining=PageRange(t->selection);
            for(NSUInteger c=0;c<commands.count;c++) {
                MetalPageCommand *command=commands[c];if(command->order<=t->order)continue;
                for(NSUInteger phase=0;phase<=command->sites.count;phase++) {
                for(MetalPageAccess *access in command->accesses)if(access->phase==phase&&[access->root isEqual:t->root]&&(access->access&MetalReachRead)) {
                    isl_set *groups=PageDomain(access->relation,remaining);
                    if(!PageEmpty(groups)&&access->slot>=0)[domain->observedSlots[c] addIndex:(NSUInteger)access->slot];
                    uint64_t *thresholds=domain->thresholds[c].mutableBytes;
                    PageGroups(groups,^(NSUInteger group){thresholds[group]=MAX(thresholds[group],t->effect.threshold);});isl_set_free(groups);
                }
                for(MetalPageAccess *access in command->accesses)if(access->phase==phase&&[access->root isEqual:t->root]&&(access->access&MetalReachWritten))
                    remaining=isl_set_subtract(remaining,PageRange(access->relation));
                }
            }
            isl_set_free(remaining);
        }
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
    [arrivals release];return result;
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
                domain->coverage?(__bridge id<MTLBuffer>)domain->coverage->declaration.expected.buffer:PageData(device,domain->expected),
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
    NSMutableData *reaches;
    NSUInteger composedOffset;
}
- (NSUInteger)add:(id<MTLBuffer>)buffer offset:(NSUInteger)offset bytes:(NSUInteger)bytes access:(uint32_t)access;
- (id<MTLBuffer>)compose;
@end
@implementation MetalPageFields
- (instancetype)init {
    if((self=[super init])){fields=[NSMutableArray new];owners=[NSMutableArray new];reaches=[NSMutableData new];}return self;
}
- (void)dealloc { [fields release];[owners release];[reaches release];[super dealloc]; }
- (NSUInteger)add:(id<MTLBuffer>)buffer offset:(NSUInteger)offset bytes:(NSUInteger)bytes access:(uint32_t)access {
    const struct MetalReach reach={buffer,offset,bytes,access,-1};[reaches appendBytes:&reach length:sizeof reach];
    for(NSUInteger i=0;i<fields.count;i++) {
        NSArray *prior=fields[i];if(prior[0]==buffer&&[prior[1] unsignedIntegerValue]==offset&&[prior[2] unsignedIntegerValue]==bytes)return i;
    }
    [fields addObject:@[buffer,@(offset),@(bytes)]];[owners addObject:buffer];return fields.count-1;
}
- (id<MTLBuffer>)compose {
    NSMutableData *selection=[NSMutableData dataWithLength:fields.count*sizeof(ncclMeshPageSelection)];
    ncclMeshPageSelection *selected=selection.mutableBytes;size_t cursor=0,alignment=0;
    for(NSUInteger i=0;i<fields.count;i++) {
        NSArray *field=fields[i];id<MTLBuffer> buffer=field[0];const size_t offset=[field[1] unsignedIntegerValue],bytes=[field[2] unsignedIntegerValue];
        MetalPageComposition *source=[[[MetalPageComposition alloc] init] autorelease];
        if(ncclMeshPageTableCreate(buffer.contents,NULL,1,&source->table))PageFailure(@"cannot bind metadata pages");
        size_t pages=0,width=0;if(ncclMeshPageTableShape(source->table,&pages,&width))PageFailure(@"metadata page shape");
        if(alignment&&width!=alignment)PageFailure(@"metadata fields have different local page widths");alignment=width;
        const size_t first=offset/width,count=(offset%width+bytes+width-1)/width;
        selected[i]=(ncclMeshPageSelection){source->table,first,count};
        fields[i]=@[buffer,@(offset),@(bytes),@(cursor*width+offset%width)];cursor+=count;[owners addObject:source];
    }
    MetalPageComposition *combined=[[[MetalPageComposition alloc] init] autorelease];
    if(ncclMeshPageTableCompose(selected,fields.count,alignment,NULL,1,&combined->table))PageFailure(@"cannot compose metadata pages");
    ncclMeshBuffer bound={0};if(ncclMeshPageTableBinding(combined->table,&bound))PageFailure(@"cannot bind composed metadata");
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
static NSString *PageReplaceFields(NSString *text,MetalPageFields *fields,NSString *base) {
    for(NSUInteger i=fields->fields.count;i-->0;) {
        NSString *pointer=[NSString stringWithFormat:@"(%@ + %@ul)",base,fields->fields[i][3]];
        text=[text stringByReplacingOccurrencesOfString:PageField(i) withString:pointer];
    }
    return text;
}
static NSArray<NSNumber *> *PageIndices(NSIndexSet *set) {
    NSMutableArray *values=[NSMutableArray array];[set enumerateIndexesUsingBlock:^(NSUInteger i,BOOL *stop){(void)stop;[values addObject:@(i)];}];return values;
}

static void PageCoverageSource(MetalPageDomain *domain,MetalPageCommand *command,NSUInteger index,
    id<MTLDevice> device,MetalPageFields *fields,NSMutableString *helpers,NSMutableArray *after,NSUInteger *ordinal) {
    NSMutableArray<MetalPageContribution *> *contributions=[NSMutableArray array];
    NSNumber *wordIdentity=PageWord(domain->rows[0]->effect.word);
    for(MetalPageContribution *part in command->contributions)
        if(part->coverage==domain->coverage&&[PageWord(part->word) isEqual:wordIdentity])[contributions addObject:part];
    NSMutableArray<NSMutableDictionary *> *announcements=[NSMutableArray array];NSUInteger announcementWidth=0;
    for(NSUInteger group=0;group<command->groups;group++) {
        NSMutableDictionary *cuts=[NSMutableDictionary dictionary];
        for(NSNumber *cut in PageIndices(domain->announcements[index][group]))cuts[cut]=@1;
        for(NSNumber *cut in PageIndices(domain->activity[index][group]))cuts[cut]=@([cuts[cut] unsignedIntValue]|4u);
        [announcements addObject:cuts];announcementWidth=MAX(announcementWidth,cuts.count);
    }
    if(!contributions.count&&!announcementWidth)return;
    [fields->owners addObject:domain->coverage];
    BOOL gpu=command->backend==MetalPageMetal;NSUInteger count=domain->rows.count;
    NSArray *pages=PageDomainPages(domain,device);
    NSString *thresholds=PageField([fields add:pages[0] offset:0 bytes:count*sizeof(uint64_t) access:MetalReachRead]);
    NSString *state=PageField([fields add:pages[1] offset:0 bytes:(count+1)*sizeof(uint32_t)
        access:MetalReachRead|MetalReachWritten|MetalReachAtomic]);
    NSString *arrivals=PageField([fields add:pages[3] offset:0 bytes:count*sizeof(uint32_t)
        access:MetalReachRead|MetalReachWritten|MetalReachAtomic]);
    ncclMeshBuffer expected=domain->coverage->declaration.expected;
    NSString *totals=PageField([fields add:(__bridge id<MTLBuffer>)expected.buffer offset:expected.offset
        bytes:domain->coverage->declaration.cuts*sizeof(uint32_t) access:MetalReachRead|MetalReachObserve]);
    NSMutableData *projection=[NSMutableData dataWithLength:count*sizeof(uint64_t)];
    for(NSUInteger cut=0;cut<count;cut++)((uint64_t *)projection.mutableBytes)[cut]=domain->rows[cut]->coverageCut;
    id<MTLBuffer> projected=PageData(device,projection);
    NSString *ordinals=PageField([fields add:projected offset:0 bytes:projection.length access:MetalReachRead]);
    NSString *word=PageWordField(fields,domain->rows[0]->effect.word,MetalReachWritten|MetalReachAtomic);
    NSString *prefix=[NSString stringWithFormat:@"MeshPageCoveragePrefix%lu",(unsigned long)(*ordinal)++];
    [helpers appendFormat:gpu?@"struct %@ { device const ulong *cuts; volatile coherent(system) device ulong *word; void at(uint p) {mesh_publish(word,cuts[p-1]);} };\n":
        @"struct %@ { const uint64_t *cuts; volatile uint64_t *word; void at(uint32_t p) {mesh_publish(word,cuts[p-1]);} };\n",prefix];
    NSString *(^contribute)(NSString *,NSString *,NSUInteger)=^NSString *(NSString *cut,NSString *flags,NSUInteger delta) {
        if(gpu)return [NSString stringWithFormat:
            @"mesh_graph_coverage((volatile coherent(system) device atomic_uint *)%@+%@,(volatile coherent(system) device atomic_uint *)%@+%@,(volatile coherent(system) device const uint *)%@+((device const ulong *)%@)[%@],%@,%luu);\n",
            state,cut,arrivals,cut,totals,ordinals,cut,flags,(unsigned long)delta];
        return [NSString stringWithFormat:
            @"mesh_graph_coverage((uint32_t *)%@+%@,(uint32_t *)%@+%@,(const uint32_t *)%@+((const uint64_t *)%@)[%@],%@,%luu);\n",
            state,cut,arrivals,cut,totals,ordinals,cut,flags,(unsigned long)delta];
    };
    NSMutableIndexSet *phases=[NSMutableIndexSet indexSet];
    for(MetalPageContribution *part in contributions) {
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
            [after[part->phase] appendString:contribute(@"cut",@"0u",1)];[after[part->phase] appendString:@"} }\n"];
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
                [after[phase] appendString:contribute(@"cut",@"(unsigned int)entry[1]",0)];[after[phase] appendString:@"} }\n"];
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
    NSArray *matches=[PageRegex(@"(?m)^\\s*(?:threadgroup|constexpr)\\s+[^;{}]+;") matchesInString:body options:0 range:NSMakeRange(0,body.length)];
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
        [source appendFormat:@"MESH_PAGE_SITE(%@)\ntemplate<> struct mesh_page_site<mesh_page_site_%@> {static inline void enter(%@,unsigned long,unsigned int); static inline void leave(%@,unsigned long,unsigned int);};\n",name,name,pointer,pointer];
        for(NSString *side in @[@"enter",@"leave"]) {
            NSString *body=[side isEqual:@"enter"]?before[phase]:after[phase];
            [declarations appendFormat:@"inline void mesh_page_site<mesh_page_site_%@>::%@(%@mesh_effects, unsigned long cell, unsigned int lane) { const unsigned long mesh_effect_group=%@ul+cell;\n",name,side,pointer,site[@"first"]];
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
    }
    return source;
}

NSString *MetalPageEffectsCompose(MetalPageCommand *command,NSString *declarations,NSArray<NSString *> *before,NSArray<NSString *> *after,
    NSIndexSet *observed,NSIndexSet *published) {
    NSString *enter=before[0],*leave=after[0];
    NSMutableString *siteDeclarations=[[declarations mutableCopy] autorelease];
    NSString *siteSource=PageSites(command,siteDeclarations,before,after);
    if(command->backend==MetalPageMetal) {
        BOOL indexed=[enter rangeOfString:@"mesh_effect_group"].location!=NSNotFound||[leave rangeOfString:@"mesh_effect_group"].location!=NSNotFound;
        NSString *group=indexed?[NSString stringWithFormat:@"const ulong mesh_effect_group=mesh_group.x+%luul*(mesh_group.y+%luul*mesh_group.z);\n",(unsigned long)command->grid.width,(unsigned long)command->grid.height]:@"";
        NSString *visibility=@"atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,static_cast<thread_scope>(MESH_SYSTEM_THREAD_SCOPE));";
        NSString *starts=enter.length?[NSString stringWithFormat:@"%@if(mesh_lane==0){%@}threadgroup_barrier(mem_flags::mem_device);%@\n",group,enter,visibility]:group;
        NSString *ends=leave.length?[NSString stringWithFormat:@"threadgroup_barrier(mem_flags::mem_device);if(mesh_lane==0){%@%@}\n",visibility,leave]:@"";
        NSString *result=PageMetalSource(command,siteDeclarations,starts,ends,observed,published);
        return [siteSource stringByAppendingString:result];
    } else if(command->backend==MetalPageANE) {
        return [NSString stringWithFormat:@"#include \"mesh-publication.h\"\n%@\nstruct MeshCpuArguments {uint8_t *mesh_effects;};\n#define MESH_ANE_ENTER(arguments) do {const auto &a=*static_cast<const MeshCpuArguments *>(arguments);[[maybe_unused]] auto *mesh_effects=a.mesh_effects;[[maybe_unused]] const unsigned long mesh_effect_group=0;%@} while(0)\n#define MESH_ANE_LEAVE(arguments) do {const auto &a=*static_cast<const MeshCpuArguments *>(arguments);[[maybe_unused]] auto *mesh_effects=a.mesh_effects;[[maybe_unused]] const unsigned long mesh_effect_group=0;%@} while(0)\n#include \"mesh-ane-entry.inc\"\n",declarations,PageSub(enter,@"\\n",@" "),PageSub(leave,@"\\n",@" ")];
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
        return [NSString stringWithFormat:@"%@#include \"mesh-publication.h\"\n%@\n%@\n[[maybe_unused]] auto *mesh_effects=a.mesh_effects;[[maybe_unused]] const unsigned long mesh_effect_group=0;\n%@%@%@%@",siteSource,siteDeclarations,[source substringToIndex:binding],enter,numerical,leave,[source substringFromIndex:close]];
    }
}

MetalPageSource *MetalPageEffectsSource(MetalPageEffects *effects,NSUInteger index,id<MTLDevice> device) {
    MetalPageCommand *command=effects->commands[index];const BOOL gpu=command->backend==MetalPageMetal;
    MetalPageSource *result=[[[MetalPageSource alloc] init] autorelease];
    MetalPageFields *fields=[[[MetalPageFields alloc] init] autorelease];
    NSMutableString *helpers=[NSMutableString string];NSMutableArray *before=[NSMutableArray array],*after=[NSMutableArray array];
    for(NSUInteger phase=0;phase<=command->sites.count;phase++){[before addObject:[NSMutableString string]];[after addObject:[NSMutableString string]];}
    NSMutableIndexSet *observed=[NSMutableIndexSet indexSet],*published=[NSMutableIndexSet indexSet];
    NSUInteger ordinal=0;
    for(MetalPageDomain *domain in effects->domains) {
        if(!domain->publication) {
            NSData *values=domain->thresholds[index];const uint64_t *thresholds=values.bytes;BOOL needed=NO;
            for(NSUInteger g=0;g<command->groups;g++)needed|=thresholds[g]!=0;if(!needed)continue;
            [observed addIndexes:domain->observedSlots[index]];
            MetalPageTransfer *t=domain->rows[0];
            NSString *word=PageWordField(fields,t->effect.word,MetalReachRead|MetalReachObserve);
            NSString *violation=PageViolationField(fields,t->effect.violation),*threshold;
            BOOL uniform=YES;for(NSUInteger g=1;g<command->groups;g++)uniform&=thresholds[g]==thresholds[0];
            if(!uniform) {
                id<MTLBuffer> table=PageData(device,values);
                threshold=[NSString stringWithFormat:gpu?@"((device const ulong *)%@)[mesh_effect_group]":@"((const uint64_t *)%@)[mesh_effect_group]",
                    PageField([fields add:table offset:0 bytes:values.length access:MetalReachRead])];
            } else threshold=[NSString stringWithFormat:@"%lluul",(unsigned long long)thresholds[0]];
            NSString *fragment=[NSString stringWithFormat:@"mesh_step_observe(%@%@, %@, %@, %uu, %uu, %lluul);\n",
                gpu?@"":@"(volatile uint64_t *)",word,threshold,violation,t->effect.polls,t->effect.channel,(unsigned long long)t->effect.entry];
            for(NSUInteger phase=0;phase<before.count;phase++) {
                NSUInteger first=phase?[command->sites[phase-1][@"first"] unsignedIntegerValue]:0;
                NSUInteger end=phase?[command->sites[phase-1][@"count"] unsignedIntegerValue]+first:
                    command->sites.count?[command->sites[0][@"first"] unsignedIntegerValue]:command->groups;
                BOOL uses=NO;for(NSUInteger g=first;g<end;g++)uses|=thresholds[g]!=0;
                if(uses)[before[phase] appendString:fragment];
            }
            continue;
        }
        if(domain->coverage) {
            [published addIndexes:domain->publishedSlots[index]];
            PageCoverageSource(domain,command,index,device,fields,helpers,after,&ordinal);continue;
        }
        NSArray<NSIndexSet *> *cuts=domain->contributions[index];NSUInteger width=0;
        for(NSIndexSet *value in cuts)width=MAX(width,value.count);if(!width)continue;
        [published addIndexes:domain->publishedSlots[index]];
        NSArray *pages=PageDomainPages(domain,device);const NSUInteger count=domain->rows.count;
        NSString *thresholds=PageField([fields add:pages[0] offset:0 bytes:count*sizeof(uint64_t) access:MetalReachRead]);
        NSString *state=PageField([fields add:pages[1] offset:0 bytes:(count+1)*sizeof(uint32_t) access:MetalReachRead|MetalReachWritten|MetalReachAtomic]);
        NSString *totals=PageField([fields add:pages[2] offset:0 bytes:count*sizeof(uint32_t) access:MetalReachRead]);
        NSString *counters=PageField([fields add:pages[3] offset:0 bytes:count*sizeof(uint32_t) access:MetalReachRead|MetalReachWritten|MetalReachAtomic]);
        NSString *word=PageWordField(fields,domain->rows[0]->effect.word,MetalReachWritten|MetalReachAtomic);
        NSString *prefix=[NSString stringWithFormat:@"MeshPagePrefix%lu",(unsigned long)ordinal++];
        [helpers appendFormat:gpu?@"struct %@ { device const ulong *cuts; volatile coherent(system) device ulong *word; void at(uint p) {mesh_publish(word,cuts[p-1]);} };\n":
            @"struct %@ { const uint64_t *cuts; volatile uint64_t *word; void at(uint32_t p) {mesh_publish(word,cuts[p-1]);} };\n",prefix];
        NSMutableString *fragment=[NSMutableString stringWithString:@"{\n"];
        NSString *indices=nil;
        if(gpu||command->groups>1) {
            NSMutableData *table=[NSMutableData dataWithLength:command->groups*width*sizeof(int64_t)];memset(table.mutableBytes,0xff,table.length);
            for(NSUInteger g=0;g<cuts.count;g++){__block NSUInteger c=0;[cuts[g] enumerateIndexesUsingBlock:^(NSUInteger value,BOOL *stop){
                (void)stop;((int64_t *)table.mutableBytes)[g*width+c++]=(int64_t)value;}];}
            id<MTLBuffer> operand=PageData(device,table);
            indices=PageField([fields add:operand offset:0 bytes:table.length access:MetalReachRead]);
        }
        if(gpu) {
            [fragment appendFormat:@"device atomic_uint *state=(device atomic_uint *)%@;\n",state];
            for(NSUInteger c=0;c<width;c++)[fragment appendFormat:
                @"{ const long cut=((device const long *)%@)[mesh_effect_group*%luul+%luul]; if(cut>=0) { uint n=atomic_fetch_add_explicit((device atomic_uint *)%@+cut,1u,memory_order_relaxed); if(n+1u==((device const uint *)%@)[cut]) atomic_store_explicit(state+cut,2u,memory_order_relaxed); } }\n",
                indices,(unsigned long)width,(unsigned long)c,counters,totals];
            [fragment appendFormat:@"%@ prefix={(device const ulong *)%@,(volatile coherent(system) device ulong *)%@};mesh_graph_settle(state,0,%luu,state+%luu,prefix);\n",prefix,thresholds,word,(unsigned long)count,(unsigned long)count];
        } else {
            [fragment appendFormat:@"uint32_t *state=(uint32_t *)%@;\n",state];
            if(indices)for(NSUInteger c=0;c<width;c++)[fragment appendFormat:
                @"{const int64_t cut=((const int64_t *)%@)[mesh_effect_group*%luul+%luul];if(cut>=0&&__atomic_add_fetch((uint32_t *)%@+cut,1u,__ATOMIC_ACQ_REL)==((const uint32_t *)%@)[cut])__atomic_store_n(state+cut,2u,__ATOMIC_RELEASE);}\n",
                indices,(unsigned long)width,(unsigned long)c,counters,totals];
            else for(NSNumber *cut in PageIndices(cuts[0]))[fragment appendFormat:
                @"if(__atomic_add_fetch((uint32_t *)%@+%@ul,1u,__ATOMIC_ACQ_REL)==((const uint32_t *)%@)[%@ul]) __atomic_store_n(state+%@ul,2u,__ATOMIC_RELEASE);\n",counters,cut,totals,cut,cut];
            [fragment appendFormat:@"%@ prefix={(const uint64_t *)%@,(volatile uint64_t *)%@};mesh_graph_settle(state,0,%luu,state+%luu,prefix);\n",prefix,thresholds,word,(unsigned long)count,(unsigned long)count];
        }
        [fragment appendString:@"}\n"];
        for(NSUInteger phase=0;phase<after.count;phase++) {
            NSUInteger first=phase?[command->sites[phase-1][@"first"] unsignedIntegerValue]:0;
            NSUInteger end=phase?[command->sites[phase-1][@"count"] unsignedIntegerValue]+first:
                command->sites.count?[command->sites[0][@"first"] unsignedIntegerValue]:command->groups;
            BOOL uses=NO;for(NSUInteger g=first;g<end;g++)uses|=[cuts[g] count]!=0;
            if(uses)[after[phase] appendString:fragment];
        }
        for(MetalPageInitial *input in domain->initial) {
            MetalTensorLayout *flat=[input->transfer->layout reshape:@[@(input->transfer->layout.count)]];
            MetalTensorLayout *view=[flat slice:0 first:(NSInteger)input->first count:input->bytes step:1];
            [fields->reaches appendData:[view reaches:input->transfer->payload base:0 elementBytes:1 access:MetalReachRead binding:-1]];
            [fields->owners addObject:input->transfer->payload];
        }
    }
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
    result->source=[MetalPageEffectsCompose(command,declarations,before,after,observed,published) copy];
    return result;
}
