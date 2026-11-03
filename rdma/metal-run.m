// metal-run.h: the run's effector. Sources: mesh_graph_realize for every decision (mesh-graph.h); Apple's Metal 4 API
// for encoding and submission (MTL4CommandBuffer.h, MTL4ComputeCommandEncoder.h: the argument table is snapshotted at
// each dispatch, MTL4CommandEncoder.h: barriers and fences, MTL4CommandQueue.h: queue events and commit feedback);
// libdispatch groups for host joins.
#import "metal-run.h"
#import "metal-recording.h"
#include <mach/mach_time.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include "mesh-graph.h"
#include "mesh-statistics.h"
#include "nccl.h"

enum { RunDispatch, RunCopy, RunHost, RunInput };

struct RunNode {
    uint32_t kind;
    uint64_t event;
    __unsafe_unretained id<MTLComputePipelineState> pipeline;
    size_t first, count;
    MTLSize groups, threads;
    BOOL threadGrid;
    __unsafe_unretained id<MTLBuffer> source, destination;
    NSUInteger from, to, bytes;
    __unsafe_unretained MetalHostOperation operation;
    __unsafe_unretained id<MTLSharedEvent> signal;
    uint64_t value;
};
struct RunResult { __unsafe_unretained id<MTLBuffer> buffer; NSUInteger offset; uint32_t bytes; uint64_t threshold; };
struct RunWait { __unsafe_unretained id<MTLSharedEvent> event; uint64_t value; };

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
    double ended;
};

struct MetalRun {
    id<MTL4CommandQueue> queue;
    id<MTLDevice> device;
    id<MTLResidencySet> residency;
    NSMutableArray *held;
    NSMutableData *nodes, *bindings, *constants, *edges, *results, *comms;
    uint64_t event;
    uint32_t parts;
    struct RunPart *part;
    struct RunHostState *host; // by node
    size_t nodeCount;
    struct mesh_graph_crossing *wait; // realize's rows, wired into joins at enqueue
    size_t waits;
    MTLSharedEventListener *listener;
    dispatch_queue_t commits;
    _Atomic long pending;
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

struct MetalRun *MetalRunCreate(id<MTL4CommandQueue> queue) {
    struct MetalRun *run=calloc(1,sizeof *run);
    run->queue=[queue retain];run->device=[queue.device retain];
    MTLResidencySetDescriptor *descriptor=[[MTLResidencySetDescriptor new] autorelease];
    NSError *error=nil;
    run->residency=[run->device newResidencySetWithDescriptor:descriptor error:&error];
    if(!run->residency)[NSException raise:NSMallocException format:@"run residency: %@",error];
    run->held=[NSMutableArray new];
    run->nodes=[NSMutableData new];run->bindings=[NSMutableData new];run->constants=[NSMutableData new];
    run->edges=[NSMutableData new];run->results=[NSMutableData new];run->comms=[NSMutableData new];
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
    RunAppend(run,(struct RunNode){.kind=RunInput,.signal=event,.value=value});
}
struct MetalRunBinding MetalRunConstant(struct MetalRun *run,const void *bytes,size_t length,uint32_t index) {
    const NSUInteger offset=(run->constants.length+15)&~(NSUInteger)15;
    [run->constants setLength:offset];[run->constants appendBytes:bytes length:length];
    return (struct MetalRunBinding){nil,offset,index};
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
    atomic_fetch_sub_explicit(&host->run->pending,1,memory_order_release);
}
static void RunCommit(struct MetalRun *run,uint32_t p) {
    struct RunPart *part=&run->part[p];
    const struct RunWait *wait=part->inputs.bytes;
    for(size_t k=0;k<part->inputs.length/sizeof *wait;k++)[run->queue waitForEvent:wait[k].event value:wait[k].value];
    id<MTL4CommandBuffer> buffer=part->buffer;
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
    // the constants lease, and every constants binding resolved to it
    if(run->constants.length) {
        id<MTLBuffer> constants=[MetalLeaseBuffer(run->device,run->constants.length) autorelease];
        memcpy(constants.contents,run->constants.bytes,run->constants.length);
        [run->held addObject:constants];RunResident(run,constants);
        struct MetalRunBinding *binding=run->bindings.mutableBytes;
        for(size_t k=0;k<run->bindings.length/sizeof *binding;k++)if(!binding[k].buffer)binding[k].buffer=constants;
    }
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
    unsigned char *deferred=calloc(n+1,1);
    size_t capacity=m+1,count=0;
    struct mesh_graph_crossing *wait=malloc(capacity*sizeof *wait);
    int status=mesh_graph_realize(n,host,m,(const uint32_t (*)[2])edge,part,&run->parts,barrier,fence,wait,capacity,&count,
                                  deferred);
    if(!status&&count>capacity) {
        capacity=count;wait=realloc(wait,capacity*sizeof *wait);
        status=mesh_graph_realize(n,host,m,(const uint32_t (*)[2])edge,part,&run->parts,barrier,fence,wait,capacity,&count,
                                  deferred);
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
            if(encoder){[encoder endEncoding];[run->part[open].buffer endCommandBuffer];encoder=nil;}
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
                owner->part[at].ended=RunStatisticsTime(feedback.GPUEndTime);
                if(feedback.error)RunFail(owner,feedback.error.localizedDescription);
                atomic_fetch_sub_explicit(&owner->pending,1,memory_order_release);
            }];
        }
        if(barrier[v])[encoder barrierAfterEncoderStages:stages beforeEncoderStages:stages visibilityOptions:MTL4VisibilityOptionDevice];
        const struct RunNode *node=&nodes[v];
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
        if(lastFenced[open]==v)[encoder updateFence:run->part[open].fence afterEncoderStages:stages];
    }
    free(lastFenced);

    // each host operation prepared once, its completion bound; deferred parts and host operations fire on their joins
    long pending=run->parts;
    for(uint32_t v=0;!defect&&v<n;v++)if(nodes[v].kind==RunHost) {
        struct RunHostState *state=&run->host[v];
        const char *error=nodes[v].operation(RunHostCompleted,state,&state->task);
        if(error){defect=error;break;}
        pending++;
    }
    atomic_store_explicit(&run->pending,pending,memory_order_relaxed);
    struct ncclComm *const *comm=run->comms.bytes;
    for(size_t k=0;!defect&&k<run->comms.length/sizeof *comm;k++)
        if(ncclMeshCommRealized(comm[k])!=ncclSuccess)defect="realizing a communicator's occurrences";
    free(host);free(barrier);free(fence);free(part);free(deferred);
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
        dispatch_group_notify(run->host[v].join,dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE,0),^{task.submit(task.context);});
    }
    for(uint32_t p=0;p<run->parts;p++)if(run->part[p].join) {
        struct MetalRun *owner=run;const uint32_t at=p;
        dispatch_group_notify(run->part[p].join,run->commits,^{RunCommit(owner,at);});
    }
    for(uint32_t p=0;p<run->parts;p++)if(!run->part[p].join)RunCommit(run,p);
    for(size_t v=0;v<run->nodeCount;v++)if(nodes[v].kind==RunHost)dispatch_group_leave(run->host[v].join);
}

int MetalRunEnd(struct MetalRun *run,double *ended,NSError **error) {
    *ended=0;
    const uintptr_t failure=atomic_load_explicit(&run->failure,memory_order_acquire);
    if(failure){if(error)*error=(NSError *)failure;return -1;}
    if(atomic_load_explicit(&run->pending,memory_order_acquire)>0)return 0;
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

void MetalRunFree(struct MetalRun *run) {
    if(!run)return;
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
    if(run->commits)dispatch_release(run->commits);
    [run->listener release];[(NSError *)atomic_load(&run->failure) release];
    [run->held release];[run->nodes release];[run->bindings release];[run->constants release];[run->edges release];
    [run->results release];[run->comms release];[run->residency release];[run->device release];[run->queue release];
    free(run);
}
