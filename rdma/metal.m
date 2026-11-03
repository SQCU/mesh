// metal.h: the library's Metal interface. 1 and 2 from the SDK (MTLDevice.h newBufferWithBytesNoCopy; MTL4Compiler.h,
// MTL4LibraryDescriptor.h, MTL4LibraryFunctionDescriptor.h, MTL4SpecializedFunctionDescriptor.h); 3, the run's effector. Sources: mesh_graph_realize for every decision (mesh-graph.h); Apple's Metal 4 API
// for encoding and submission (MTL4CommandBuffer.h, MTL4ComputeCommandEncoder.h: the argument table is snapshotted at
// each dispatch, MTL4CommandEncoder.h: barriers and fences, MTL4CommandQueue.h: queue events and commit feedback);
// libdispatch groups for host joins.
#import "metal.h"
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include "mesh-graph.h"
#include "mesh-statistics.h"
#include "nccl.h"

id<MTLDevice> MetalDevice(void) {
    static id<MTLDevice> device;
    static dispatch_once_t made;
    dispatch_once(&made,^{device=MTLCreateSystemDefaultDevice();});
    return device;
}

// A region span's buffer, hazard-tracked. Changed from untracked (since 2026-09, no reason recorded) by the FFN
// demonstration session: every MPS tensor of a mesh process is a lease's buffer (torch_mesh allocator.mm), and torch's
// stream is classic Metal, whose encoders are ordered against one another only over tracked resources, as torch's own
// allocator's are (its pools' HAZARD usage). Untracked, torch's cast and the copy after it raced in one command buffer:
// v.float().cpu() read y and w2 wrong on every rank while the CPU read their bytes right (metal-microbench
// docs/facts.md, ffn-2026-11-03 runs 11-13). A Metal 4 queue tracks nothing whatever the mode, so a run's encoding
// is unchanged. What must stay: tracked, while torch's stream runs over leases.
id<MTLBuffer> MetalWrap(void *address,size_t bytes) {
    return [MetalDevice() newBufferWithBytesNoCopy:address length:bytes
        options:MTLResourceStorageModeShared|MTLResourceHazardTrackingModeTracked deallocator:nil];
}

void MetalRelease(id object) { [object release]; }
void *MetalContents(id<MTLBuffer> buffer) { return buffer.contents; }

id<MTLComputePipelineState> MetalPipeline(NSString *source,NSString *function,NSArray<NSArray *> *constants,NSError **error) {
    static id<MTL4Compiler> compiler;
    static dispatch_once_t made;
    dispatch_once(&made,^{compiler=[MetalDevice() newCompilerWithDescriptor:[[MTL4CompilerDescriptor new] autorelease] error:nil];});
    MTLCompileOptions *options=[[MTLCompileOptions new] autorelease];
    options.languageVersion=MTLLanguageVersion4_0;options.mathMode=MTLMathModeSafe;
    MTL4LibraryDescriptor *text=[[MTL4LibraryDescriptor new] autorelease];
    text.source=source;text.options=options;
    id<MTLLibrary> library=[[compiler newLibraryWithDescriptor:text error:error] autorelease];
    if(!library)return nil;
    MTL4LibraryFunctionDescriptor *named=[[MTL4LibraryFunctionDescriptor new] autorelease];
    named.name=function;named.library=library;
    MTL4FunctionDescriptor *chosen=named;
    if(constants.count) {
        MTLFunctionConstantValues *values=[[MTLFunctionConstantValues new] autorelease];
        for(NSArray *constant in constants)
            [values setConstantValue:((NSData *)constant[2]).bytes type:(MTLDataType)[constant[1] unsignedIntegerValue]
                             atIndex:[constant[0] unsignedIntegerValue]];
        MTL4SpecializedFunctionDescriptor *specialized=[[MTL4SpecializedFunctionDescriptor new] autorelease];
        specialized.functionDescriptor=named;specialized.constantValues=values;
        chosen=specialized;
    }
    MTL4ComputePipelineDescriptor *pipeline=[[MTL4ComputePipelineDescriptor new] autorelease];
    pipeline.computeFunctionDescriptor=chosen;
    return [compiler newComputePipelineStateWithDescriptor:pipeline compilerTaskOptions:nil error:error];
}

enum { RunDispatch, RunCopy, RunHost, RunInput };

struct RunNode {
    uint32_t kind;
    uint64_t event;
    __unsafe_unretained id<MTLComputePipelineState> pipeline;
    size_t first, count;
    MTLSize groups, threads;
    BOOL threadGrid, ends, stamp, mark; // ends: the node ends its part (MetalRunCut); stamp: a timestamp before it
                                       // (MetalRunStamp); mark: a traced run's timestamp after it (MetalRunMark)
    __unsafe_unretained id<MTLBuffer> source, destination;
    NSUInteger from, to, bytes;
    __unsafe_unretained MetalHostOperation operation;
    __unsafe_unretained id<MTLSharedEvent> signal;
    uint64_t value;
};
struct RunResult { __unsafe_unretained id<MTLBuffer> buffer; NSUInteger offset; uint32_t bytes; uint64_t threshold; };
struct RunWait { __unsafe_unretained id<MTLSharedEvent> event; uint64_t value; };

// A landing's presence. The process's one watcher reads the bridge's `landed` words of enqueued runs' landings and
// sets each landing's event once its word reaches its threshold (or the link's loss, all-ones); it parks while none
// is outstanding. Records reach it by a lock-free stack; a freed run cancels its own. The run and the watcher each hold
// one reference.
enum { LandingWaiting, LandingPresent, LandingLost, LandingCancelled };
struct RunLanding {
    const _Atomic uint64_t *word;
    uint64_t threshold;
    id<MTLSharedEvent> event;
    _Atomic int state, references;
    double present; // when the watcher saw it (mesh_statistics_now)
    struct RunLanding *next;
};
static _Atomic(struct RunLanding *) landingStack;
static dispatch_semaphore_t landingPosted;
static void RunLandingRelease(struct RunLanding *landing) {
    if(atomic_fetch_sub_explicit(&landing->references,1,memory_order_acq_rel)==1){[landing->event release];free(landing);}
}
static void *RunLandingWatch(void *unused) {
    (void)unused;
    struct RunLanding **watched=NULL;size_t count=0,capacity=0;
    for(;;) {
        for(struct RunLanding *taken=atomic_exchange_explicit(&landingStack,NULL,memory_order_acquire);taken;) {
            struct RunLanding *next=taken->next;
            if(count==capacity){capacity=capacity?2*capacity:64;watched=realloc(watched,capacity*sizeof *watched);}
            watched[count++]=taken;taken=next;
        }
        if(!count){dispatch_semaphore_wait(landingPosted,DISPATCH_TIME_FOREVER);continue;}
        size_t kept=0;
        for(size_t k=0;k<count;k++) {
            struct RunLanding *landing=watched[k];
            const uint64_t landed=atomic_load_explicit(landing->word,memory_order_acquire);
            int state=atomic_load_explicit(&landing->state,memory_order_acquire);
            if(state==LandingWaiting&&(landed==UINT64_MAX||landed>=landing->threshold)) {
                state=landed==UINT64_MAX?LandingLost:LandingPresent;
                landing->present=mesh_statistics_now();
                atomic_store_explicit(&landing->state,state,memory_order_release);
                landing->event.signaledValue=1;
            }
            if(state==LandingWaiting)watched[kept++]=landing;else RunLandingRelease(landing);
        }
        count=kept;
    }
    return NULL;
}
static void RunLandingStart(void) {
    landingPosted=dispatch_semaphore_create(0);
    pthread_t watcher;
    pthread_attr_t attributes;pthread_attr_init(&attributes);
    pthread_attr_set_qos_class_np(&attributes,QOS_CLASS_USER_INTERACTIVE,0);
    pthread_attr_setdetachstate(&attributes,PTHREAD_CREATE_DETACHED);
    if(pthread_create(&watcher,&attributes,RunLandingWatch,NULL))[NSException raise:NSMallocException format:@"the landing watcher"];
    pthread_attr_destroy(&attributes);
}
static void RunLandingPost(struct RunLanding *landing) {
    static pthread_once_t started=PTHREAD_ONCE_INIT;
    pthread_once(&started,RunLandingStart);
    atomic_fetch_add_explicit(&landing->references,1,memory_order_relaxed); // the watcher's
    struct RunLanding *head=atomic_load_explicit(&landingStack,memory_order_relaxed);
    do landing->next=head;
    while(!atomic_compare_exchange_weak_explicit(&landingStack,&head,landing,memory_order_release,memory_order_relaxed));
    dispatch_semaphore_signal(landingPosted);
}

// a host node at run time: its prepared task, the join its producers leave, the joins it leaves on completion
struct RunHostState {
    struct MetalRun *run;
    MeshHostTask task;
    dispatch_group_t join;
    NSMutableData *leaves; // dispatch_group_t
    double ended;
};
// a part: one command buffer; `join` set where it commits after a host producer, NULL where it commits at enqueue
struct RunPart {
    id<MTL4CommandAllocator> allocator;
    id<MTL4CommandBuffer> buffer;
    MTL4CommitOptions *options;
    id<MTLFence> fence;
    id<MTLSharedEvent> done;
    dispatch_group_t join;
    NSMutableData *inputs, *leaves; // struct RunWait; dispatch_group_t
    double committed, started, ended;
};

struct MetalRun {
    id<MTL4CommandQueue> queue;
    id<MTLDevice> device;
    id<MTLResidencySet> residency;
    NSMutableArray *held;
    NSMutableData *nodes, *bindings, *edges, *results, *comms, *landings; // landings: struct RunLanding *
    uint64_t event;
    size_t input; // the last outside producer MetalRunInput appended
    uint32_t parts;
    struct RunPart *part;
    id<MTL4CounterHeap> stamps; // two a part: its compute's start (its first stamped node) and its end
    BOOL traced; // and, where traced (MetalRunTrace), one a node after it, from 2 x parts on
    struct RunHostState *host; // by node
    size_t nodeCount;
    struct mesh_graph_crossing *wait; // realize's rows, wired into joins at enqueue
    size_t waits;
    MTLSharedEventListener *listener;
    dispatch_queue_t commits;
    _Atomic long pending;
    _Atomic long references; // the caller's, each committed part's and each submitted host task's: the last frees
    _Atomic(uintptr_t) failure; // a retained NSError
    double observed;
};

static double RunHostSeconds(void) {
    static mach_timebase_info_data_t base;
    if(!base.denom)mach_timebase_info(&base);
    return (double)mach_absolute_time()*base.numer/base.denom*1e-9;
}
// MTL4CommitFeedback's GPU times are host time (mach_absolute_time) in seconds; the statistics clock is CLOCK_MONOTONIC
static double RunStatisticsTime(CFTimeInterval host) { return host+(mesh_statistics_now()-RunHostSeconds()); }

static void RunFail(struct MetalRun *run,NSString *message) {
    NSError *error=[[NSError alloc] initWithDomain:@"mesh.run" code:EINVAL userInfo:@{NSLocalizedDescriptionKey:message}];
    uintptr_t none=0;
    if(!atomic_compare_exchange_strong(&run->failure,&none,(uintptr_t)error))[error release];
}

static void RunRelease(struct MetalRun *run);
struct MetalRun *MetalRunCreate(id<MTL4CommandQueue> queue) {
    struct MetalRun *run=calloc(1,sizeof *run);
    atomic_init(&run->references,1);
    run->queue=[queue retain];run->device=[queue.device retain];
    MTLResidencySetDescriptor *descriptor=[[MTLResidencySetDescriptor new] autorelease];
    NSError *error=nil;
    run->residency=[run->device newResidencySetWithDescriptor:descriptor error:&error];
    if(!run->residency)[NSException raise:NSMallocException format:@"run residency: %@",error];
    run->held=[NSMutableArray new];
    run->nodes=[NSMutableData new];run->bindings=[NSMutableData new];
    run->edges=[NSMutableData new];run->results=[NSMutableData new];run->comms=[NSMutableData new];
    run->landings=[NSMutableData new];
    return run;
}
id<MTLDevice> MetalRunDevice(struct MetalRun *run) { return run->device; }
void MetalRunEvent(struct MetalRun *run,uint64_t event) { run->event=event; }

static void RunResident(struct MetalRun *run,id<MTLBuffer> buffer) {
    if(buffer&&![run->residency containsAllocation:buffer])[run->residency addAllocation:buffer];
}
static void RunAppend(struct MetalRun *run,struct RunNode node) {
    node.event=run->event;[run->nodes appendBytes:&node length:sizeof node];
}

void MetalRunDispatch(struct MetalRun *run,id<MTLComputePipelineState> pipeline,const struct MetalRunBinding *bindings,
                      size_t count,MTLSize groups,MTLSize threads,BOOL threadGrid) {
    [run->held addObject:pipeline];
    const size_t first=run->bindings.length/sizeof *bindings;
    [run->bindings appendBytes:bindings length:count*sizeof *bindings];
    for(size_t k=0;k<count;k++)if(bindings[k].buffer){[run->held addObject:bindings[k].buffer];RunResident(run,bindings[k].buffer);}
    RunAppend(run,(struct RunNode){.kind=RunDispatch,.pipeline=pipeline,.first=first,.count=count,.groups=groups,
        .threads=threads,.threadGrid=threadGrid});
}
void MetalRunCopy(struct MetalRun *run,id<MTLBuffer> source,NSUInteger from,id<MTLBuffer> destination,NSUInteger to,
                  NSUInteger bytes) {
    [run->held addObject:source];[run->held addObject:destination];
    RunResident(run,source);RunResident(run,destination);
    RunAppend(run,(struct RunNode){.kind=RunCopy,.source=source,.destination=destination,.from=from,.to=to,.bytes=bytes});
}
void MetalRunHost(struct MetalRun *run,MetalHostOperation operation) {
    operation=[[operation copy] autorelease];[run->held addObject:operation];
    RunAppend(run,(struct RunNode){.kind=RunHost,.operation=operation});
}
void MetalRunInput(struct MetalRun *run,id<MTLSharedEvent> event,uint64_t value) {
    [run->held addObject:event];
    run->input=run->nodes.length/sizeof(struct RunNode);
    RunAppend(run,(struct RunNode){.kind=RunInput,.signal=event,.value=value});
}
void MetalRunInputValue(struct MetalRun *run,uint64_t value) {
    ((struct RunNode *)run->nodes.mutableBytes)[run->input].value=value;
}
void MetalRunLanded(struct MetalRun *run,const void *word,uint64_t threshold) {
    struct RunLanding *landing=calloc(1,sizeof *landing);
    landing->word=word;landing->threshold=threshold;landing->event=[run->device newSharedEvent];
    atomic_init(&landing->state,LandingWaiting);atomic_init(&landing->references,1); // the run's
    [run->landings appendBytes:&landing length:sizeof landing];
    RunAppend(run,(struct RunNode){.kind=RunInput,.signal=landing->event,.value=1}); // no input of MetalRunInputValue's
}
void MetalRunMark(struct MetalRun *run) {
    if(run->nodes.length)((struct RunNode *)run->nodes.mutableBytes)[run->nodes.length/sizeof(struct RunNode)-1].mark=YES;
}
void MetalRunStamp(struct MetalRun *run) {
    if(run->nodes.length)((struct RunNode *)run->nodes.mutableBytes)[run->nodes.length/sizeof(struct RunNode)-1].stamp=YES;
}
size_t MetalRunSpans(struct MetalRun *run,double *us,size_t capacity) {
    if(!run->stamps)return 0;
    NSData *entries=[run->stamps resolveCounterRange:NSMakeRange(0,2*(NSUInteger)run->parts)];
    const MTL4TimestampHeapEntry *e=entries.bytes;
    const double frequency=(double)[run->device queryTimestampFrequency];
    for(uint32_t p=0;p<run->parts&&p<capacity;p++)
        us[p]=e&&e[2*p].timestamp&&e[2*p+1].timestamp>e[2*p].timestamp?(double)(e[2*p+1].timestamp-e[2*p].timestamp)/frequency*1e6:0;
    return run->parts;
}
void MetalRunTrace(struct MetalRun *run) { run->traced=YES; }
size_t MetalRunTimes(struct MetalRun *run,double *us,size_t capacity) {
    const size_t n=run->nodes.length/sizeof(struct RunNode);
    if(!run->stamps||!run->traced||run->stamps.count<2*(NSUInteger)run->parts+n)return 0;
    NSData *entries=[run->stamps resolveCounterRange:NSMakeRange(0,2*(NSUInteger)run->parts+n)];
    const MTL4TimestampHeapEntry *e=entries.bytes;
    uint64_t base=UINT64_MAX;
    for(size_t k=0;e&&k<2*(size_t)run->parts+n;k++)if(e[k].timestamp&&e[k].timestamp<base)base=e[k].timestamp;
    const double frequency=(double)[run->device queryTimestampFrequency];
    for(size_t v=0;v<n&&v<capacity;v++) {
        const uint64_t t=e?e[2*(size_t)run->parts+v].timestamp:0;
        us[v]=t&&base!=UINT64_MAX?(double)(t-base)/frequency*1e6:-1;
    }
    return n;
}
void MetalRunCut(struct MetalRun *run) {
    if(run->nodes.length)((struct RunNode *)run->nodes.mutableBytes)[run->nodes.length/sizeof(struct RunNode)-1].ends=YES;
}
void MetalRunEdges(struct MetalRun *run,const uint64_t (*edges)[2],size_t count) {
    [run->edges appendBytes:edges length:count*sizeof *edges];
}
void MetalRunResult(struct MetalRun *run,id<MTLBuffer> buffer,NSUInteger offset,uint32_t bytes,uint64_t threshold) {
    [run->held addObject:buffer];
    const struct RunResult result={buffer,offset,bytes,threshold};
    [run->results appendBytes:&result length:sizeof result];
}
void MetalRunHold(struct MetalRun *run,id object) {
    [run->held addObject:object];
    if([object conformsToProtocol:@protocol(MTLBuffer)])RunResident(run,object);
}
void MetalRunCommunicator(struct MetalRun *run,struct ncclComm *comm) { [run->comms appendBytes:&comm length:sizeof comm]; }

static void RunLeave(NSData *leaves) {
    const dispatch_group_t *join=leaves.bytes;
    for(size_t k=0;k<leaves.length/sizeof *join;k++)dispatch_group_leave(join[k]);
}
static void RunHostCompleted(void *context,const char *error) {
    struct RunHostState *host=context;
    host->ended=mesh_statistics_now();
    if(error)RunFail(host->run,[NSString stringWithUTF8String:error]);
    RunLeave(host->leaves);
    struct MetalRun *run=host->run;
    atomic_fetch_sub_explicit(&run->pending,1,memory_order_release);
    RunRelease(run);
}
static void RunCommit(struct MetalRun *run,uint32_t p) {
    struct RunPart *part=&run->part[p];
    const struct RunWait *wait=part->inputs.bytes;
    for(size_t k=0;k<part->inputs.length/sizeof *wait;k++)[run->queue waitForEvent:wait[k].event value:wait[k].value];
    id<MTL4CommandBuffer> buffer=part->buffer;
    atomic_fetch_add_explicit(&run->references,1,memory_order_relaxed); // its feedback's
    part->committed=mesh_statistics_now();
    [run->queue commit:&buffer count:1 options:part->options];
    if(part->done)[run->queue signalEvent:part->done value:1];
    RunLeave(part->leaves);
}

// the events' nodes: each event's nodes chained in appended order, an edge (e, f) from e's last node to f's first
static int RunEventOrder(void *context,const void *a,const void *b) {
    const struct RunNode *nodes=context;const uint32_t i=*(const uint32_t *)a,j=*(const uint32_t *)b;
    if(nodes[i].event!=nodes[j].event)return nodes[i].event<nodes[j].event?-1:1;
    return i<j?-1:i>j;
}
static const uint32_t *RunEventFind(const uint32_t *sorted,size_t n,const struct RunNode *nodes,uint64_t event,int last) {
    size_t low=0,high=n;
    while(low<high){const size_t mid=(low+high)/2;
        if(nodes[sorted[mid]].event<event||(last&&nodes[sorted[mid]].event==event))low=mid+1;else high=mid;}
    if(last)return low&&nodes[sorted[low-1]].event==event?&sorted[low-1]:NULL;
    return low<n&&nodes[sorted[low]].event==event?&sorted[low]:NULL;
}

const char *MetalRunFinish(struct MetalRun *run) {
    const struct RunNode *nodes=run->nodes.bytes;
    const uint32_t n=(uint32_t)(run->nodes.length/sizeof *nodes);
    run->nodeCount=n;
    const struct RunResult *result=run->results.bytes;
    for(size_t k=0;k<run->results.length/sizeof *result;k++)RunResident(run,result[k].buffer);
    [run->residency commit];
    const struct MetalRunBinding *binding=run->bindings.bytes;

    uint32_t *sorted=malloc((n+1)*sizeof *sorted);
    for(uint32_t v=0;v<n;v++)sorted[v]=v;
    qsort_r(sorted,n,sizeof *sorted,(void *)nodes,RunEventOrder);
    const uint64_t (*solved)[2]=run->edges.bytes;
    const size_t solvedCount=run->edges.length/sizeof *solved;
    uint32_t (*edge)[2]=malloc((n+solvedCount+1)*sizeof *edge);
    size_t m=0;
    for(uint32_t k=1;k<n;k++)if(nodes[sorted[k]].event==nodes[sorted[k-1]].event){edge[m][0]=sorted[k-1];edge[m][1]=sorted[k];m++;}
    for(size_t k=0;k<solvedCount;k++) {
        const uint32_t *from=RunEventFind(sorted,n,nodes,solved[k][0],1),*to=RunEventFind(sorted,n,nodes,solved[k][1],0);
        if(!from||!to){free(sorted);free(edge);return "a solve edge names an event with no node (the producer's defect)";}
        edge[m][0]=*from;edge[m][1]=*to;m++;
    }
    free(sorted);

    unsigned char *host=calloc(n+1,1),*barrier=calloc(n+1,1),*fence=calloc(n+1,1);
    for(uint32_t v=0;v<n;v++)host[v]=nodes[v].kind==RunHost?1:nodes[v].kind==RunInput?2:0;
    uint32_t *part=malloc((n+1)*sizeof *part);
    unsigned char *deferred=calloc(n+1,1),*ends=calloc(n+1,1);
    for(uint32_t v=0;v<n;v++)ends[v]=nodes[v].ends;
    size_t capacity=m+1,count=0;
    struct mesh_graph_crossing *wait=malloc(capacity*sizeof *wait);
    int status=mesh_graph_realize(n,host,m,(const uint32_t (*)[2])edge,part,&run->parts,barrier,fence,wait,capacity,&count,
                                  deferred,ends);
    if(!status&&count>capacity) {
        capacity=count;wait=realloc(wait,capacity*sizeof *wait);
        status=mesh_graph_realize(n,host,m,(const uint32_t (*)[2])edge,part,&run->parts,barrier,fence,wait,capacity,&count,
                                  deferred,ends);
    }
    free(edge);
    const char *defect=status==EINVAL?"a solve edge runs backward in the nodes' order (the producer's defect)":
                       status?"realizing the run: out of memory":NULL;
    run->part=calloc(run->parts+1,sizeof *run->part);
    run->host=calloc(n+1,sizeof *run->host);
    for(uint32_t p=0;p<run->parts;p++){run->part[p].inputs=[NSMutableData new];run->part[p].leaves=[NSMutableData new];}
    for(uint32_t v=0;v<n;v++)if(host[v]){run->host[v].run=run;run->host[v].leaves=[NSMutableData new];}

    // realize's deferred parts commit on a join of their producers; the others at enqueue
    for(uint32_t p=0;!defect&&p<run->parts;p++)if(deferred[p])run->part[p].join=dispatch_group_create();
    for(uint32_t v=0;!defect&&v<n;v++)if(nodes[v].kind==RunHost)run->host[v].join=dispatch_group_create();
    for(size_t k=0;!defect&&k<count;k++) // a host waiting on a part hears of its completion through its event
        if((wait[k].kind&1)&&!(wait[k].kind&4)&&!run->part[wait[k].producer].done)
            run->part[wait[k].producer].done=[run->device newSharedEvent];
    if(!defect) {
        run->listener=[[MTLSharedEventListener alloc] initWithDispatchQueue:dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE,0)];
        run->commits=dispatch_queue_create("mesh.run.commits",DISPATCH_QUEUE_SERIAL);
    }

    // a part's compute span: two timestamps a part (metal-microbench docs/facts.md: a count past 4096 with no error slot
    // faulted the driver; two a part with the slot)
    const NSUInteger traced=run->traced?(NSUInteger)n:0;
    if(!defect&&run->parts&&2*run->parts+traced<=4096) {
        MTL4CounterHeapDescriptor *stampsDescriptor=[[MTL4CounterHeapDescriptor new] autorelease];
        stampsDescriptor.type=MTL4CounterHeapTypeTimestamp;stampsDescriptor.count=2*run->parts+traced;
        NSError *error=nil;
        run->stamps=[run->device newCounterHeapWithDescriptor:stampsDescriptor error:&error];
        if(run->stamps)[run->held addObject:run->stamps];
    }
    unsigned char *begun=calloc(run->parts+1,1);
    // encode each part: fences its producers updated, then its nodes with realize's barriers
    for(uint32_t v=0;!defect&&v<n;v++)if(!host[v]&&fence[v]&&!run->part[part[v]].fence)run->part[part[v]].fence=[run->device newFence];
    uint32_t *lastFenced=malloc((run->parts+1)*sizeof *lastFenced);
    for(uint32_t p=0;p<run->parts;p++)lastFenced[p]=UINT32_MAX;
    for(uint32_t v=0;v<n;v++)if(!host[v]&&fence[v])lastFenced[part[v]]=v;
    MTL4ArgumentTableDescriptor *tableDescriptor=[[MTL4ArgumentTableDescriptor new] autorelease];
    tableDescriptor.maxBufferBindCount=31;
    const MTLStages stages=MTLStageDispatch|MTLStageBlit;
    id<MTL4ComputeCommandEncoder> encoder=nil;id<MTL4ArgumentTable> table=nil;uint32_t open=UINT32_MAX;
    for(uint32_t v=0;!defect&&v<=n;v++) {
        if(v<n&&host[v])continue;
        if(v==n||part[v]!=open) {
            if(encoder){
                if(run->stamps)[encoder writeTimestampWithGranularity:MTL4TimestampGranularityPrecise intoHeap:run->stamps atIndex:2*(NSUInteger)open+1];
                [encoder endEncoding];[run->part[open].buffer endCommandBuffer];encoder=nil;
            }
            if(v==n)break;
            open=part[v];struct RunPart *p=&run->part[open];
            p->allocator=[run->device newCommandAllocator];p->buffer=[run->device newCommandBuffer];
            [p->buffer beginCommandBufferWithAllocator:p->allocator];[p->buffer useResidencySet:run->residency];
            encoder=[p->buffer computeCommandEncoder];
            NSError *error=nil;table=[[run->device newArgumentTableWithDescriptor:tableDescriptor error:&error] autorelease];
            if(!table){defect="a run part's argument table";break;}
            [run->held addObject:table];[encoder setArgumentTable:table];
            for(size_t k=0;k<count;k++)if(!(wait[k].kind&1)&&!(wait[k].kind&4)&&wait[k].waiter==open&&run->part[wait[k].producer].fence)
                [encoder waitForFence:run->part[wait[k].producer].fence beforeEncoderStages:stages];
            const uint32_t at=open;struct MetalRun *owner=run;
            p->options=[MTL4CommitOptions new];
            [p->options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback){
                owner->part[at].started=RunStatisticsTime(feedback.GPUStartTime);
                owner->part[at].ended=RunStatisticsTime(feedback.GPUEndTime);
                if(feedback.error)RunFail(owner,feedback.error.localizedDescription);
                atomic_fetch_sub_explicit(&owner->pending,1,memory_order_release);
                RunRelease(owner);
            }];
        }
        if(barrier[v])[encoder barrierAfterEncoderStages:stages beforeEncoderStages:stages visibilityOptions:MTL4VisibilityOptionDevice];
        const struct RunNode *node=&nodes[v];
        if(run->stamps&&node->stamp&&!begun[open]){
            [encoder writeTimestampWithGranularity:MTL4TimestampGranularityPrecise intoHeap:run->stamps atIndex:2*(NSUInteger)open];begun[open]=1;
        }
        if(node->kind==RunCopy) {
            [encoder copyFromBuffer:node->source sourceOffset:node->from toBuffer:node->destination
                  destinationOffset:node->to size:node->bytes];
        } else {
            [encoder setComputePipelineState:node->pipeline];
            for(size_t k=node->first;k<node->first+node->count;k++)
                [table setAddress:binding[k].buffer.gpuAddress+binding[k].offset atIndex:binding[k].index];
            if(node->threadGrid)[encoder dispatchThreads:node->groups threadsPerThreadgroup:node->threads];
            else [encoder dispatchThreadgroups:node->groups threadsPerThreadgroup:node->threads];
        }
        if(traced&&run->stamps&&node->mark)
            [encoder writeTimestampWithGranularity:MTL4TimestampGranularityPrecise intoHeap:run->stamps atIndex:2*(NSUInteger)run->parts+v];
        if(lastFenced[open]==v)[encoder updateFence:run->part[open].fence afterEncoderStages:stages];
    }
    free(lastFenced);free(begun);

    // each host operation prepared once, its completion bound; deferred parts and host operations fire on their joins
    long pending=run->parts;
    for(uint32_t v=0;!defect&&v<n;v++)if(nodes[v].kind==RunHost) {
        struct RunHostState *state=&run->host[v];
        const char *error=nodes[v].operation(RunHostCompleted,state,&state->task);
        if(error){defect=error;break;}
        pending++;
    }
    atomic_store_explicit(&run->pending,pending,memory_order_relaxed);
    free(host);free(barrier);free(fence);free(part);free(deferred);free(ends);
    run->wait=wait;run->waits=count;
    return defect;
}

// The joins: each host operation's and each deferred part's producers, entered here so that a run freed unenqueued
// holds none; each fires its operation or commit once they have all left
void MetalRunEnqueue(struct MetalRun *run) {
    const struct RunNode *nodes=run->nodes.bytes;
    for(size_t v=0;v<run->nodeCount;v++)if(nodes[v].kind==RunHost)dispatch_group_enter(run->host[v].join); // the enqueue
    for(size_t k=0;k<run->waits;k++) {
        const struct mesh_graph_crossing c=run->wait[k];
        const BOOL hostWaiter=(c.kind&1)!=0,hostProducer=(c.kind&4)!=0;
        dispatch_group_t join=hostWaiter?run->host[c.waiter].join:run->part[c.waiter].join;
        if(hostProducer&&nodes[c.producer].kind==RunInput) {
            if(!hostWaiter){const struct RunWait input={nodes[c.producer].signal,nodes[c.producer].value};
                [run->part[c.waiter].inputs appendBytes:&input length:sizeof input];continue;}
            if(!join)continue;
            dispatch_group_enter(join);
            [nodes[c.producer].signal notifyListener:run->listener atValue:nodes[c.producer].value
                block:^(id<MTLSharedEvent> e,uint64_t value){(void)e;(void)value;dispatch_group_leave(join);}];
        } else if(hostProducer) {
            if(!join)continue;
            dispatch_group_enter(join);[run->host[c.producer].leaves appendBytes:&join length:sizeof join];
        } else if(hostWaiter) {
            if(!join)continue;
            dispatch_group_enter(join);
            [run->part[c.producer].done notifyListener:run->listener atValue:1
                block:^(id<MTLSharedEvent> e,uint64_t value){(void)e;(void)value;dispatch_group_leave(join);}];
        } else if(join&&run->part[c.producer].join) {
            dispatch_group_enter(join);[run->part[c.producer].leaves appendBytes:&join length:sizeof join];
        }
    }
    for(size_t v=0;v<run->nodeCount;v++)if(nodes[v].kind==RunHost) {
        const MeshHostTask task=run->host[v].task;
        atomic_fetch_add_explicit(&run->references,1,memory_order_relaxed); // its completion's
        dispatch_group_notify(run->host[v].join,dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE,0),^{task.submit(task.context);});
    }
    for(uint32_t p=0;p<run->parts;p++)if(run->part[p].join) {
        struct MetalRun *owner=run;const uint32_t at=p;
        dispatch_group_notify(run->part[p].join,run->commits,^{RunCommit(owner,at);});
    }
    struct RunLanding *const *landing=run->landings.bytes;
    for(size_t k=0;k<run->landings.length/sizeof *landing;k++)RunLandingPost(landing[k]);
    for(uint32_t p=0;p<run->parts;p++)if(!run->part[p].join)RunCommit(run,p);
    for(size_t v=0;v<run->nodeCount;v++)if(nodes[v].kind==RunHost)dispatch_group_leave(run->host[v].join);
}

int MetalRunEnd(struct MetalRun *run,double *ended,NSError **error) {
    *ended=0;
    const uintptr_t failure=atomic_load_explicit(&run->failure,memory_order_acquire);
    if(failure){if(error)*error=(NSError *)failure;return -1;}
    if(atomic_load_explicit(&run->pending,memory_order_acquire)>0)return 0;
    struct RunLanding *const *landing=run->landings.bytes;
    for(size_t k=0;k<run->landings.length/sizeof *landing;k++) {
        const int state=atomic_load_explicit(&landing[k]->state,memory_order_acquire);
        if(state==LandingWaiting)return 0;
        if(state==LandingLost) {
            RunFail(run,@"a landing's link was lost (its stream's landed is all-ones)");
            if(error)*error=(NSError *)atomic_load(&run->failure);
            return -1;
        }
    }
    struct ncclComm *const *comm=run->comms.bytes;
    for(size_t k=0;k<run->comms.length/sizeof *comm;k++) {
        ncclResult_t state=ncclSuccess;ncclCommGetAsyncError(comm[k],&state);
        if(state==ncclInProgress)return 0;
        if(state!=ncclSuccess) {
            RunFail(run,[NSString stringWithFormat:@"communicator error %d",(int)state]);
            if(error)*error=(NSError *)atomic_load(&run->failure);
            return -1;
        }
    }
    const struct RunResult *result=run->results.bytes;
    for(size_t k=0;k<run->results.length/sizeof *result;k++) {
        const void *word=(const char *)result[k].buffer.contents+result[k].offset;
        const uint64_t value=result[k].bytes==8?atomic_load_explicit((const _Atomic uint64_t *)word,memory_order_acquire):
                                                 atomic_load_explicit((const _Atomic uint32_t *)word,memory_order_acquire);
        if(value<result[k].threshold)return 0;
    }
    if(run->results.length&&!run->observed)run->observed=mesh_statistics_now();
    double latest=run->observed;
    for(uint32_t p=0;p<run->parts;p++)if(run->part[p].ended>latest)latest=run->part[p].ended;
    const struct RunNode *nodes=run->nodes.bytes;
    for(size_t v=0;v<run->nodeCount;v++)if(nodes[v].kind==RunHost&&run->host[v].ended>latest)latest=run->host[v].ended;
    *ended=latest;
    return 1;
}

void MetalRunFree(struct MetalRun *run) { if(run)RunRelease(run); }
size_t MetalRunLandings(struct MetalRun *run,double (*out)[2],size_t capacity) {
    struct RunLanding *const *landing=run->landings.bytes;
    const size_t count=run->landings.length/sizeof *landing;
    for(size_t k=0;k<count&&k<capacity;k++) {
        out[k][0]=(double)landing[k]->threshold;
        out[k][1]=atomic_load_explicit(&landing[k]->state,memory_order_acquire)==LandingWaiting?0:landing[k]->present;
    }
    return count;
}
size_t MetalRunParts(struct MetalRun *run,double (*times)[3],size_t capacity) {
    for(uint32_t p=0;p<run->parts&&p<capacity;p++)
        times[p][0]=run->part[p].committed,times[p][1]=run->part[p].started,times[p][2]=run->part[p].ended;
    return run->parts;
}
size_t MetalRunWaiting(struct MetalRun *run,uint64_t (*out)[3],size_t capacity,long *parts) {
    struct RunLanding *const *landing=run->landings.bytes;
    size_t count=0;
    for(size_t k=0;k<run->landings.length/sizeof *landing;k++) {
        if(atomic_load_explicit(&landing[k]->state,memory_order_acquire)!=LandingWaiting)continue;
        if(count<capacity) {
            out[count][0]=(uint64_t)(uintptr_t)landing[k]->word;
            out[count][1]=atomic_load_explicit(landing[k]->word,memory_order_acquire);
            out[count][2]=landing[k]->threshold;
        }
        count++;
    }
    *parts=atomic_load_explicit(&run->pending,memory_order_acquire);
    return count;
}
// the run's memory, once the caller and every part and host task it committed have let it go
static void RunRelease(struct MetalRun *run) {
    if(atomic_fetch_sub_explicit(&run->references,1,memory_order_acq_rel)!=1)return;
    for(uint32_t p=0;p<run->parts;p++) {
        struct RunPart *part=&run->part[p];
        [part->allocator release];[part->buffer release];[part->options release];[part->fence release];[part->done release];
        if(part->join)dispatch_release(part->join);
        [part->inputs release];[part->leaves release];
    }
    const struct RunNode *nodes=run->nodes.bytes;
    for(size_t v=0;v<run->nodeCount;v++)if(run->host&&run->host[v].leaves) {
        if(run->host[v].task.release)run->host[v].task.release(run->host[v].task.context);
        if(run->host[v].join)dispatch_release(run->host[v].join);
        [run->host[v].leaves release];
    }
    (void)nodes;
    free(run->part);free(run->host);free(run->wait);
    struct RunLanding *const *landing=run->landings.bytes;
    for(size_t k=0;k<run->landings.length/sizeof *landing;k++) {
        int waiting=LandingWaiting;
        atomic_compare_exchange_strong(&landing[k]->state,&waiting,LandingCancelled);
        RunLandingRelease(landing[k]);
    }
    [run->landings release];
    if(run->commits)dispatch_release(run->commits);
    [run->listener release];[(NSError *)atomic_load(&run->failure) release];
    [run->held release];[run->nodes release];[run->bindings release];[run->edges release];
    [run->results release];[run->comms release];[run->residency release];[run->device release];[run->queue release];
    free(run);
}
