#import <Metal/Metal.h>
#import <objc/runtime.h>
#include "nccl-mesh-metal.h"
#include "nccl-mesh-kernels.h"
#include "mesh_metal.h"
#include "metal-capabilities.h"
#include "nccl.h"
#include "metal-recording.h"
#include "mesh-landing.h"
#include "mesh-fabric.h"
#include "mesh-graph.h"
#include "mesh-publication.h"
#import "metal-page-effects.h"
#import "mesh-index-relation.h"
#include "mesh-cpu-source.h"
#include <isl/ctx.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* a typed kernel's kind by its operation and its ncclDataType_t (int8 uint8 int32 uint32 int64 uint64 float16 float32
   float64 bfloat16 e4m3 e5m2, then the partials merge's rows, METAL_MERGE), KINDS where none is compiled: float64's,
   truncdiv's but the integers', premul's of the merge */
enum { OP_PREMUL, OP_TRUNCDIV, OPS };
#define INTEGERS(op) KIND_##op##_i8, KIND_##op##_u8, KIND_##op##_i32, KIND_##op##_u32, KIND_##op##_i64, KIND_##op##_u64
#define FLOATS(op) KIND_##op##_f16, KIND_##op##_f32, KINDS, KIND_##op##_bf16, KIND_##op##_e4m3, KIND_##op##_e5m2
static const uint8_t TYPED[OPS][METAL_MERGE + 1] = {
  {INTEGERS(premul), FLOATS(premul), KINDS},
  {INTEGERS(truncdiv), KINDS, KINDS, KINDS, KINDS, KINDS, KINDS, KINDS}};
/* The arithmetic operation is part of the compiled entry point, never a
   selector read for each payload element. */
#define REDUCTION_TYPES(op) {KIND_zelloth_i8_##op,KIND_zelloth_u8_##op,KIND_zelloth_i32_##op,KIND_zelloth_u32_##op, \
  KIND_zelloth_i64_##op,KIND_zelloth_u64_##op,KIND_zelloth_f16_##op,KIND_zelloth_f32_##op,KINDS, \
  KIND_zelloth_bf16_##op,KIND_zelloth_e4m3_##op,KIND_zelloth_e5m2_##op}
static const uint8_t REDUCTIONS[4][ncclNumTypes]={REDUCTION_TYPES(sum),REDUCTION_TYPES(prod),REDUCTION_TYPES(max),REDUCTION_TYPES(min)};
/* The width is a compiled copy operand; every source uses the page accessor. */
static const uint8_t COPY[2]={KIND_copy_4,KIND_copy_1};

ARGS(args, uint64_t, uint32_t);
/* Kernel lookup is shared; compiled argument pages, pieces of the region (nccl-mesh.c metal_piece_hold), belong to
   their function's residency set, are never recycled while that function can execute, and are given back once that
   set ends. */
#define RECORD MESH_METAL_ARGUMENT_BYTES
/* Prepared pipelines; launch extent comes from the compiled page cells. */
struct kernels { id<MTLDevice> device; id<MTLComputePipelineState> pipelines[KINDS]; };
@interface MeshArgumentPages : NSObject {
@public
  struct metal_bytes page;
  size_t bytes, used, count, room;
  void **held;
  NSMutableArray *sources;
}
@end
@implementation MeshArgumentPages
- (void)dealloc { for (size_t k = 0; k < count; k++) metal_piece_give(held[k]); free(held);[sources release];[super dealloc]; }
@end
static char argumentPagesKey;
_Static_assert(sizeof(struct args) <= RECORD, "an argument record");

/* Preparation retains the equation's input identities and fixed value cells.
   Coverage specializes their page operands; execution never searches transfer
   intervals or interprets this declaration. */
@interface MeshFoldActivity : NSObject {
@public
  NSData *transports,*origins,*cells;
  size_t width;
}
@end
@implementation MeshFoldActivity
- (void)dealloc { [transports release];[origins release];[cells release];[super dealloc]; }
@end

@interface MeshCollectiveCommand : NSObject {
@public
  MetalPageCommand *source;
  NSData *bindings,*reaches,*arguments;
  uint32_t kind;
  MTLSize grid,threads;
  BOOL threadGrid,sourceSpecialized;
  MeshFoldActivity *fold;
  size_t foldCell;
}
@end
@implementation MeshCollectiveCommand
- (void)dealloc { [source release];[bindings release];[reaches release];[arguments release];[fold release];[super dealloc]; }
@end
@interface MeshCollectiveHost : NSObject { @public void *program; }
@end
@implementation MeshCollectiveHost
- (void)dealloc { mesh_cpu_program_release(program);[super dealloc]; }
@end

static MetalPageAccess *group_access(struct metal_program *program,id<MTLBuffer> buffer,NSUInteger offset,
    NSUInteger bytes,NSInteger slot,uint32_t access,NSString *relation) {
  isl_ctx *ctx=program->relation_context;
  isl_map *groups=isl_map_read_from_str(ctx,relation.UTF8String);
  (void)offset;(void)bytes;
  isl_map *physical=MetalPageRegionRelation(groups,buffer);isl_map_free(groups);
  if(!physical)[NSException raise:NSInvalidArgumentException format:@"invalid collective group/page relation"];
  MetalPageAccess *made=[MetalPageAccess root:MetalPageRegionRoot() slot:slot access:access relation:physical];
  isl_map_free(physical);return made;
}
/* Ordinary cyclic element loops have one fixed relation from dispatch groups
   to bytes. This covers arbitrary widths without enumerating their elements. */
static void element_access(struct metal_program *program,void *buffer,size_t offset,size_t count,size_t width,
    NSInteger slot,uint32_t access,NSUInteger grid,NSUInteger threads) {
  if(!count)return;
  NSString *relation=[NSString stringWithFormat:
    @"{ [g] -> [b] : exists k : 0 <= k < %lu and %lu + %lu*k <= b < %lu + %lu*(k+1) and g = floor((k mod %lu)/%lu) }",
    (unsigned long)count,(unsigned long)offset,(unsigned long)width,(unsigned long)offset,(unsigned long)width,
    (unsigned long)grid,(unsigned long)threads];
  [(NSMutableArray *)program->group_accesses addObject:group_access(program,(id<MTLBuffer>)buffer,offset,count*width,slot,access,relation)];
}
static void loop_access(struct metal_program *program,void *buffer,size_t offset,size_t count,size_t width,
    NSInteger slot,uint32_t access) {
  [(NSMutableArray *)program->loop_accesses addObject:@[(id<MTLBuffer>)buffer,@(offset),@(count),@(width),@(slot),@(access)]];
}
static void cell_access(struct metal_program *program,void *buffer,size_t offset,size_t bytes,NSInteger slot,
    uint32_t access,const uint64_t *cells,size_t count,size_t stride,size_t width) {
  NSMutableString *relation=[NSMutableString stringWithString:@"{ [g] -> [b] : "];
  for(size_t k=0;k<count;k++) {
    [relation appendFormat:@"%@(g = %lu and %llu <= b < %llu)",k?@" or ":@"",(unsigned long)k,
      (unsigned long long)(offset+cells[k*stride]*width),(unsigned long long)(offset+(cells[k*stride]+cells[k*stride+1])*width)];
  }
  if(!count)[relation appendString:@"false"];
  [relation appendString:@" }"];
  [(NSMutableArray *)program->group_accesses addObject:group_access(program,(id<MTLBuffer>)buffer,offset,bytes,slot,access,relation)];
}
void *metal_page_selection(struct metal_program *program,const ncclMeshPageEffect *effect,void *payload,const ncclMeshSpan *spans,size_t count) {
  if(!count)return NULL;
  NSMutableArray *runs=[NSMutableArray array];
  for(size_t k=0;k<count;k++)[runs addObject:@[@(spans[k].offset),@(spans[k].bytes),@0,@1]];
  MetalTensorLayout *layout=[MetalTensorLayout runs:runs];
  isl_map *selection=MetalPageRegionSelection(program->relation_context,(id<MTLBuffer>)payload,layout);
  MetalPageTransfer *transfer=[MetalPageTransfer owner:[NSValue valueWithPointer:program] order:[(NSArray *)program->commands count]+(effect->kind==ncclMeshPagePublish)
    effect:*effect root:MetalPageRegionRoot() selection:selection payload:(id<MTLBuffer>)payload layout:layout];
  [(NSMutableArray *)program->transfers addObject:transfer];isl_map_free(selection);return transfer;
}
void metal_page_effect(struct metal_program *program,const ncclMeshPageEffect *effect,void *payload,size_t offset,size_t bytes) {
  if(bytes){const ncclMeshSpan span={offset,bytes};metal_page_selection(program,effect,payload,&span,1);}
}

static MetalPageCoverage *coverage_value(struct metal_program *program,ncclMeshPageCoverage values) {
  NSMutableDictionary *declarations=program->coverage_values;
  NSArray *key=@[[NSValue valueWithPointer:values.active.buffer],@(values.active.offset),
    [NSValue valueWithPointer:values.expected.buffer],@(values.expected.offset),@(values.cuts)];
  MetalPageCoverage *value=declarations[key];
  if(!value)declarations[key]=value=[MetalPageCoverage declaration:values];
  return value;
}

static struct metal_result fold_coverage(struct metal_program *,const struct metal_coverage_projection *,size_t);

/* Coverage refines retained transfer declarations, before their single common
   solve. The span table already supplies the exact packet boundaries. */
struct metal_result metal_coverage(struct metal_program *program,const struct metal_coverage_projection *bindings,size_t count) {
  const struct metal_result folded=fold_coverage(program,bindings,count);if(folded.status)return folded;
  NSMutableDictionary *ranges=[NSMutableDictionary dictionary];
  for(size_t first=0;first<count;) {
    size_t end=first+1;while(end<count&&bindings[end].transport==bindings[first].transport)end++;
    ranges[@(bindings[first].transport+1)]=[NSValue valueWithRange:NSMakeRange(first,end-first)];first=end;
  }
  NSMutableArray *made=[NSMutableArray new];
  for(MetalPageTransfer *prior in (NSArray *)program->transfers) {
    NSValue *selected=ranges[@(prior->effect.transport)];
    if(!selected){[made addObject:prior];continue;}
    const NSRange range=selected.rangeValue;
    const size_t begin=prior->effect.transport_first,end=begin+prior->effect.bytes;
    size_t low=range.location,high=NSMaxRange(range);
    while(low<high){const size_t mid=low+(high-low)/2;
      if(bindings[mid].first+bindings[mid].bytes<=begin)low=mid+1;else high=mid;}
    if(low==NSMaxRange(range)||bindings[low].first>=end){[made addObject:prior];continue;}
    const struct metal_coverage_projection *domain=bindings+range.location;
    for(size_t at=begin,k=low;at<end;) {
      const int covered=k<NSMaxRange(range)&&bindings[k].first<=at;
      const size_t until=MIN(end,k==NSMaxRange(range)?end:covered?bindings[k].first+bindings[k].bytes:bindings[k].first);
      ncclMeshPageEffect effect=prior->effect;
      effect.first+=at-begin;effect.bytes=until-at;effect.transport_first=at;
      size_t before=0,after=domain->units;
      while(before<after){const size_t mid=before+(after-before)/2;
        if(domain->ends[mid]<until)before=mid+1;else after=mid;}
      effect.threshold=domain->base+before;
      MetalTensorLayout *layout=[prior->layout slice:0 first:(NSInteger)(at-begin) count:until-at step:1];
      isl_map *selection=MetalPageRegionSelection(program->relation_context,prior->payload,layout);
      MetalPageTransfer *next=[MetalPageTransfer owner:prior->owner order:prior->order effect:effect
        root:prior->root selection:selection payload:prior->payload layout:layout];
      next.coverage=covered?coverage_value(program,bindings[k].values):prior->coverage;
      next->coverageCut=covered?bindings[k].cut:prior->coverageCut;
      [made addObject:next];isl_map_free(selection);at=until;
      if(covered)k++;
    }
  }
  [(id)program->transfers release];program->transfers=made;program->declaration.transfers=made;
  return METAL_DONE;
}

void metal_coverage_announcement(struct metal_program *program,ncclMeshPageCoverage values,size_t cut) {
  MeshCollectiveCommand *command=[(NSArray *)program->commands lastObject];
  MetalPageCoverage *coverage=coverage_value(program,values);
  NSString *text=[NSString stringWithFormat:@"{ [g] -> [c] : 0 <= g < %lu and c = %lu }",
    (unsigned long)command->source->groups,(unsigned long)cut];
  isl_map *relation=isl_map_read_from_str(program->relation_context,text.UTF8String);
  MetalPageAnnouncement *announcement=[MetalPageAnnouncement coverage:coverage phase:0 flags:4 relation:relation];
  [command->source->announcements release];command->source->announcements=[@[announcement] copy];
  isl_map_free(relation);
}

void metal_place_sources(struct metal_program *program,size_t previous,const size_t *positions,
    const void *const *effects,size_t count,size_t *relocation) {
  NSMutableArray *commands=program->commands;
  if(count) {
    NSMutableArray *made=[NSMutableArray arrayWithCapacity:commands.count];
    NSMutableDictionary *effectOrder=[NSMutableDictionary dictionaryWithCapacity:count];
    NSMutableDictionary *atBoundary=[NSMutableDictionary dictionary];
    for(size_t k=0;k<count;k++) {
      NSNumber *at=@(positions[k]);NSMutableArray *items=atBoundary[at];
      if(!items)atBoundary[at]=items=[NSMutableArray array];
      [items addObject:@(k)];
    }
    for(size_t boundary=0;boundary<=previous;boundary++) {
      for(NSNumber *item in atBoundary[@(boundary)]) {
        const NSUInteger k=item.unsignedIntegerValue;
        MeshCollectiveCommand *command=commands[previous+k];
        [made addObject:command];command->source->order=made.count;
        effectOrder[[NSValue valueWithPointer:effects[k]]]=@(made.count+1);
      }
      if(boundary<previous) {
        relocation[boundary]=made.count;
        MeshCollectiveCommand *command=commands[boundary];
        [made addObject:command];command->source->order=made.count;
      }
    }
    for(MetalPageTransfer *transfer in (NSArray *)program->transfers) {
      NSNumber *placed=effectOrder[[NSValue valueWithPointer:transfer]];
      if(placed)transfer->order=placed.unsignedIntegerValue;
      else if(transfer->order)transfer->order=transfer->order<=previous?
        relocation[transfer->order-1]+1:made.count+1;
    }
    [commands setArray:made];
  } else {
    for(size_t i=0;i<previous;i++)relocation[i]=i;
  }
  [(id)program->declaration.commands release];program->declaration.commands=NULL;
}

static id<MTLDevice> device_;
static NSMutableDictionary *kernels_;
/* the text of this thread's last failure here that is composed (say): a result's message (struct metal_result), its
   caller's to carry before the thread's next step */
static _Thread_local char said_[MESH_NCCL_DIAGNOSTIC_BYTES];

static const char *say(const char *format, ...) __attribute__((format(printf, 1, 2)));
static const char *say(const char *format, ...) {
  va_list arguments;
  va_start(arguments, format);
  vsnprintf(said_, sizeof said_, format, arguments);
  va_end(arguments);
  return said_;
}

void *metal_device(void) {
  static dispatch_once_t once;
  dispatch_once(&once, ^{ device_ = MTLCreateSystemDefaultDevice(); });
  return device_;
}

/* `device`'s kernels at its first use, none compiled here: the library from the AIR its text was compiled to when this
   library was built, and every kind's pipeline (NAME) from this node's binary archive of them, made by its install,
   each looked up with MTLPipelineOptionFailOnBinaryArchiveMiss.  Both files sit beside this library's image, named by
   the kernels' SHA-256 this image was compiled with (KERNELS_SHA256: of the text and its compile line, the
   Makefile's): libnccl-mesh.<SHA-256>.metallib and libnccl-mesh.<SHA-256>.binarchive.  NULL, *message the rebuild
   that makes them and what failed, where a file is absent (an install of other kernels, or none) or the archive lacks
   a kind (its GPU compiler not the system's: an OS update since the install); the caller's collective fails as a
   value, and nothing is kept, so a later call reads the files again (rdma/NCCL.md "The library's kernels") */
static struct kernels *kernels_of(id<MTLDevice> device, const char **message) {
  @autoreleasepool {
    if (!kernels_) kernels_ = [NSMutableDictionary new];
    NSValue *key = [NSValue valueWithPointer:(__bridge const void *)device];
    NSValue *held = kernels_[key];
    if (held) return held.pointerValue;
    Dl_info image;
    if (!dladdr((const void *)kernels_of, &image)) { *message = "dladdr: no image holds libnccl-mesh's code"; return NULL; }
    NSString *stem = [[@(image.dli_fname) stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"libnccl-mesh." KERNELS_SHA256];
    NSString *air = [stem stringByAppendingString:@".metallib"], *binaries = [stem stringByAppendingString:@".binarchive"];
    NSError *error = nil;
    id<MTLLibrary> library = [[device newLibraryWithURL:[NSURL fileURLWithPath:air] error:&error] autorelease];
    MTLBinaryArchiveDescriptor *archived = [[MTLBinaryArchiveDescriptor new] autorelease];
    archived.url = [NSURL fileURLWithPath:binaries];
    id<MTLBinaryArchive> archive = library ? [[device newBinaryArchiveWithDescriptor:archived error:&error] autorelease] : nil;
    if (!archive) {
      *message = say("make -C rdma install, on this node, installs %s: %s", (library ? binaries : air).UTF8String,
                     error ? error.localizedDescription.UTF8String : "absent");
      return NULL;
    }
    struct kernels *made = calloc(1, sizeof *made);
    if (!made) { *message = "out of memory"; return NULL; }
    MTLComputePipelineDescriptor *described = [[MTLComputePipelineDescriptor new] autorelease];
    described.binaryArchives = @[archive];
    for (uint32_t kind = 0; kind < KINDS; kind++) {
      described.computeFunction = [[library newFunctionWithName:@(NAME[kind])] autorelease];
      error = nil;
      made->pipelines[kind] = described.computeFunction ? [device newComputePipelineStateWithDescriptor:described
                                                                                            options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                                                         reflection:nil error:&error] : nil;
      if (made->pipelines[kind]) continue;
      *message = say("make -C rdma install, on this node, makes %s anew: kernel %s: %s", binaries.UTF8String, NAME[kind],
                     error ? error.localizedDescription.UTF8String : "absent from the library");
      for (uint32_t k = 0; k < kind; k++) [made->pipelines[k] release];
      free(made);
      return NULL;
    }
    made->device = [device retain];
    kernels_[key] = [NSValue valueWithPointer:made];
    return made;
  }
}

/* the program's device's kernels (kernels_of), found at its first step and kept with it */
static struct kernels *program_kernels(struct metal_program *program, const char **message) {
  if (!program->kernels)
    program->kernels = kernels_of(program->residency.device, message);
  return program->kernels;
}

void *metal_wrap(void *address, size_t bytes) {
  return [(id<MTLDevice>)metal_device() newBufferWithBytesNoCopy:address length:bytes options:MTLResourceStorageModeShared deallocator:nil];
}

void *metal_event(void) { return [(id<MTLDevice>)metal_device() newSharedEvent]; }
uint64_t metal_signaled(void *event) { return ((id<MTLSharedEvent>)event).signaledValue; }
int metal_wait(void *event, uint64_t value, uint64_t ns) { return [(id<MTLSharedEvent>)event waitUntilSignaledValue:value timeoutMS:ns / 1000000]; }
void metal_release(void *object) { [(id)object release]; }
void *metal_retain(void *object) { return [(id)object retain]; }
/* The occurrence already owns its effective observation budget. Binding does
   not reinterpret configuration or start any observation. */
struct metal_program *metal_prepare(struct metal_program *program,struct isl_ctx *context,void *violation,uint32_t polls) {
  *program=(struct metal_program){0};program->violation=[(id<MTLBuffer>)violation retain];program->polls=polls;
  program->accesses=[NSMutableData new];program->commands=[NSMutableArray new];program->transfers=[NSMutableArray new];
  program->coverage_values=[NSMutableDictionary new];
  program->group_accesses=[NSMutableArray new];program->loop_accesses=[NSMutableArray new];program->resources=[NSMutableSet new];
  program->relation_context=context;metal_resident(program,violation);return program;
}


void metal_resident(struct metal_program *program,void *buffer) {
  if(buffer)[(NSMutableSet *)program->resources addObject:(id<MTLBuffer>)buffer];
}

void metal_gate(struct metal_program *program, void *violation) {
  if ((id<MTLBuffer>)violation == program->violation) return;
  [(id<MTLBuffer>)violation retain];
  [program->violation release];
  program->violation = (id<MTLBuffer>)violation;
  metal_resident(program, violation);
}

/* All argument tables and records use these function-owned pages. */
static MeshArgumentPages *argument_owner(struct metal_program *program) {
  MeshArgumentPages *pages=program->arguments;
  if(!pages){pages=[MeshArgumentPages new];pages->sources=[NSMutableArray new];program->arguments=pages;}
  return pages;
}

static int argument_pages(struct metal_program *program,size_t bytes,struct metal_bytes *made) {
  MeshArgumentPages *pages=argument_owner(program);
  const size_t size=(bytes+RECORD-1)/RECORD*RECORD;
  if(!pages->count||size>pages->bytes-pages->used) {
    if(pages->count==pages->room) {
      const size_t room=pages->room?2*pages->room:1;
      void **held=realloc(pages->held,room*sizeof *held);
      if(!held)return ENOMEM;
      pages->held=held;pages->room=room;
    }
    void *piece=metal_piece_hold(&pages->page,&pages->bytes,size);
    if(!piece)return ENOMEM;
    pages->held[pages->count++]=piece;pages->used=0;
    metal_resident(program,pages->page.buffer);
  }
  *made=(struct metal_bytes){pages->page.buffer,pages->page.offset+pages->used,(char *)pages->page.contents+pages->used};
  pages->used+=size;
  return 0;
}

/* Activity is a normal input page. The prepared index column has one entry
   per existing (cell, contributor); it contains no transport descriptors and
   execution does not walk the coverage ranges retained here. Static folds
   compile the same predicate to true and need no index column. */
static struct metal_result fold_coverage(struct metal_program *program,
    const struct metal_coverage_projection *bindings,size_t count) {
  NSMutableDictionary *ranges=[NSMutableDictionary dictionary],*folds=[NSMutableDictionary dictionary];
  for(size_t first=0;first<count;) {
    size_t end=first+1;while(end<count&&bindings[end].transport==bindings[first].transport)end++;
    ranges[@(bindings[first].transport+1)]=[NSValue valueWithRange:NSMakeRange(first,end-first)];first=end;
  }
  for(MeshCollectiveCommand *command in (NSArray *)program->commands)if(command->fold) {
    NSValue *key=[NSValue valueWithPointer:command->fold];NSMutableArray *commands=folds[key];
    if(!commands)folds[key]=commands=[NSMutableArray array];[commands addObject:command];
  }
  for(NSArray *commands in folds.allValues) {
    MeshCollectiveCommand *first=commands[0];MeshFoldActivity *fold=first->fold;
    const size_t *transports=fold->transports.bytes,*origins=fold->origins.bytes,*cells=fold->cells.bytes;
    const size_t K=fold->transports.length/sizeof *transports,Q=fold->cells.length/(2*sizeof *cells);
    BOOL dynamic=NO;for(size_t k=0;k<K;k++)dynamic|=ranges[@(transports[k])]!=nil;
    if(!dynamic)continue;
    struct metal_bytes indices;
    if(argument_pages(program,(1+K*Q)*sizeof(uint64_t),&indices))return metal_failed(ENOMEM);
    *(uint64_t *)indices.contents=1;
    const BOOL gpu=first->source->backend==MetalPageMetal;
    const uint64_t initial=gpu?((id<MTLBuffer>)indices.buffer).gpuAddress+indices.offset:(uint64_t)(uintptr_t)indices.contents;
    uint64_t *selected=(uint64_t *)indices.contents+1;
    for(size_t i=0;i<K*Q;i++)selected[i]=initial;
    NSMutableArray *operands=[NSMutableArray arrayWithCapacity:K*Q];
    for(size_t i=0;i<K*Q;i++)[operands addObject:[NSNull null]];
    for(size_t k=0;k<K;k++) {
      NSValue *rangeValue=ranges[@(transports[k])];if(!rangeValue)continue;
      const NSRange range=rangeValue.rangeValue;size_t b=range.location;
      for(size_t q=0;q<Q;q++) {
        /* A numerical value may span physical fragments. Its declared
           activity is shared by those fragments; its final byte identifies
           the complete value's incoming cut, just like arrival readiness. */
        const size_t last=origins[k]+(cells[2*q]+cells[2*q+1])*fold->width-1;
        while(b<NSMaxRange(range)&&bindings[b].first+bindings[b].bytes<=last)b++;
        if(b==NSMaxRange(range)||last<bindings[b].first)continue;
        ncclMeshBuffer active=bindings[b].values.active;active.offset+=bindings[b].cut*sizeof(uint32_t);
        selected[q*K+k]=gpu?((id<MTLBuffer>)active.buffer).gpuAddress+active.offset:
          (uint64_t)(uintptr_t)((char *)metal_contents(active.buffer)+active.offset);
        operands[q*K+k]=[NSValue valueWithBytes:&active objCType:@encode(ncclMeshBuffer)];
        metal_resident(program,active.buffer);
      }
    }
    indices.offset+=sizeof(uint64_t);indices.contents=(uint64_t *)indices.contents+1;
    for(MeshCollectiveCommand *command in commands) {
      const size_t begin=gpu?0:command->foldCell,end=gpu?Q:begin+1;
      NSMutableArray *accesses=[command->source->accesses mutableCopy];
      NSMutableData *reaches=[command->reaches mutableCopy];
      for(size_t q=begin;q<end;q++)for(size_t k=0;k<K;k++) {
        id operand=operands[q*K+k];if(operand==[NSNull null])continue;
        ncclMeshBuffer active;[operand getValue:&active size:sizeof active];
        NSString *relation=[NSString stringWithFormat:@"{ [g] -> [b] : g=%lu and %lu<=b<%lu }",
          (unsigned long)(gpu?q:0),(unsigned long)active.offset,(unsigned long)(active.offset+sizeof(uint32_t))];
        [accesses addObject:group_access(program,active.buffer,active.offset,sizeof(uint32_t),-1,MetalReachRead,relation)];
        const struct MetalReach reach={.buffer=active.buffer,.offset=active.offset,.length=sizeof(uint32_t),.access=MetalReachRead};
        [reaches appendBytes:&reach length:sizeof reach];
      }
      const struct MetalReach indexReach={.buffer=indices.buffer,.offset=indices.offset+begin*K*sizeof(uint64_t),
        .length=(end-begin)*K*sizeof(uint64_t),.access=MetalReachRead};
      [reaches appendBytes:&indexReach length:sizeof indexReach];
      const struct MetalReach constantReach={.buffer=indices.buffer,.offset=indices.offset-sizeof(uint64_t),
        .length=sizeof(uint32_t),.access=MetalReachRead};
      [reaches appendBytes:&constantReach length:sizeof constantReach];
      [command->source->accesses release];command->source->accesses=accesses;
      [command->reaches release];command->reaches=reaches;
      NSString *predicate;
      if(gpu) {
        NSMutableData *bound=[command->bindings mutableCopy];
        struct MetalCommandBinding *arguments=bound.mutableBytes;
        arguments[3].buffer=indices.buffer;arguments[3].offset=indices.offset;
        [command->bindings release];command->bindings=bound;
        predicate=[NSString stringWithFormat:@"#define MESH_FOLD_ACTIVE(k,q) (mesh_step_read<uint>((device uchar *)mesh_step_read<ulong>(a,ulong(q)*%luul+(k)),0)!=0)\n#define MESH_FOLD_OWN(q) MESH_FOLD_ACTIVE(%luul,q)\n",(unsigned long)K,(unsigned long)(K-1)];
      } else {
        NSMutableData *arguments=[command->arguments mutableCopy];
        const uint32_t *const *activity=(const uint32_t *const *)indices.contents;
        memcpy((char *)arguments.mutableBytes+arguments.length-sizeof activity,&activity,sizeof activity);
        [command->arguments release];command->arguments=arguments;
        predicate=[NSString stringWithFormat:@"#define MESH_FOLD_ACTIVE(k,q) (__atomic_load_n(a.activity[(unsigned long)(q)*%luul+(k)],__ATOMIC_ACQUIRE)!=0)\n#define MESH_FOLD_OWN(q) MESH_FOLD_ACTIVE(%luul,q)\n",(unsigned long)K,(unsigned long)(K-1)];
      }
      NSString *source=[predicate stringByAppendingString:command->source->source];
      [command->source->source release];command->source->source=[source copy];
      command->sourceSpecialized=YES;
    }
  }
  return METAL_DONE;
}
static struct metal_result record4(struct metal_program *program,id<MTLBuffer> d,id<MTLBuffer> s,id<MTLBuffer> a,id<MTLBuffer> words,
    struct args p,struct MetalCommandBinding *bindings) {
  struct metal_bytes record;
  const int failed=argument_pages(program,sizeof p,&record);
  if(failed)return metal_failed(failed);
  id<MTLBuffer> v = words ? words : program->violation ? program->violation : d;
  p.polls = v != d ? program->polls : 0;
  p.sequence=0;
  memcpy(record.contents,&p,sizeof p);
  const id<MTLBuffer> used[4] = {d, s ? s : d, a ? a : d, v};
  for (int k = 0; k < 4; k++) metal_resident(program, used[k]);
  {
    const id<MTLBuffer> buffers[6]={d,s?s:d,(id<MTLBuffer>)record.buffer,a?a:d,v,(id<MTLBuffer>)record.buffer};
    for(uint32_t k=0;k<6;k++)bindings[k]=(struct MetalCommandBinding){.buffer=buffers[k],
      .offset=(k==2||k==5)?record.offset:0,.length=k==2?sizeof p:0,
      .index=k,.access=k==2?MetalReachRead:0};
  }
  return METAL_DONE;
}

/* Transport leaves enter the same prepared command graph as numerical leaves.
   Bindings carry storage; accesses name the exact data relation. */
static void command_emit(struct metal_program *program,uint32_t kind,
    const struct MetalCommandBinding *bindings,MTLSize grid,MTLSize threads,BOOL threadGrid) {
  NSMutableData *accesses=(NSMutableData *)program->accesses;
  NSMutableArray *relations=program->group_accesses;
  const NSUInteger groups=threadGrid?(grid.width+threads.width-1)/threads.width:grid.width*grid.height*grid.depth;
  const struct MetalReach *uses=accesses.bytes;
  for(NSUInteger k=0;!program->administrative&&k<accesses.length/sizeof *uses;k++) {
    const struct MetalReach use=uses[k];if(!use.length||!use.buffer)continue;
    for(uint32_t mode=MetalReachRead;mode<=MetalReachWritten;mode<<=1)if(use.access&mode) {
      NSString *relation=[NSString stringWithFormat:@"{ [g] -> [b] : 0 <= g < %lu and %lu <= b < %lu }",
        (unsigned long)groups,(unsigned long)use.offset,(unsigned long)(use.offset+use.length)];
      MetalPageAccess *all=group_access(program,use.buffer,use.offset,use.length,use.binding,mode,relation);
      isl_set *remaining=isl_map_range(isl_map_copy(all->relation));
      for(MetalPageAccess *known in relations)if(known->access&mode)
        remaining=isl_set_subtract(remaining,isl_map_range(isl_map_copy(known->relation)));
      isl_map *unassigned=isl_map_intersect_range(isl_map_copy(all->relation),remaining);
      if(!isl_map_is_empty(unassigned))[relations addObject:[MetalPageAccess root:MetalPageRegionRoot() slot:use.binding access:mode relation:unassigned]];
      isl_map_free(unassigned);
    }
  }
  /* Keep one numerical body. The explicit entry is the common source
     compiler's entry, including builtin kernels defined with macros. */
  NSString *source=[NSString stringWithFormat:@"%@%@\nkernel void mesh_prepared_leaf(device uchar *d [[buffer(0)]], device uchar *s [[buffer(1)]], constant mesh_args &p [[buffer(2)]], device uchar *a [[buffer(3)]], device uchar *vb [[buffer(4)]], device ulong *progress [[buffer(5)]], uint i [[thread_position_in_grid]], uint g [[threads_per_grid]], uint l [[thread_position_in_threadgroup]], uint t [[threads_per_threadgroup]]) { threadgroup ulong tg[4]; body_%s(d,s,p,a,vb,progress,i,g,l,t,tg); }\n",
    @(metal_prelude()),KERNELS,NAME[kind]];
  MeshCollectiveCommand *command=[[[MeshCollectiveCommand alloc] init] autorelease];
  command->source=[[MetalPageCommand order:[(NSArray *)program->commands count]+1 groups:groups backend:MetalPageMetal
    source:source function:@"mesh_prepared_leaf" grid:MTLSizeMake(groups,1,1) metadataSlot:6 accesses:relations] retain];
  command->source->coherentAccesses=YES;
  command->source->frame=program->administrative?UINT32_MAX:program->frame;
  command->bindings=[[NSData alloc] initWithBytes:bindings length:6*sizeof *bindings];
  command->reaches=[accesses copy];command->kind=kind;
  command->grid=grid;command->threads=threads;command->threadGrid=threadGrid;
  [(NSMutableArray *)program->commands addObject:command];
  [relations removeAllObjects];
  [accesses setLength:0];
}

static struct metal_result dispatch4(struct metal_program *program,uint32_t kind,id<MTLBuffer> d,id<MTLBuffer> s,
    id<MTLBuffer> a,id<MTLBuffer> words,struct args p,uint64_t units,int group) {
  struct MetalCommandBinding bindings[6];
  const struct metal_result recorded=record4(program,d,s,a,words,p,bindings);
  if(recorded.status)return recorded;
  const NSUInteger most=group?MESH_TRANSPORT_GROUP_THREADS:MESH_TRANSPORT_ELEMENT_THREADS;
  const NSUInteger width=most;
  for(NSArray *use in (NSArray *)program->loop_accesses)
    element_access(program,use[0],[use[1] unsignedLongValue],[use[2] unsignedLongValue],[use[3] unsignedLongValue],
      [use[4] integerValue],[use[5] unsignedIntValue],MIN(units,MESH_TRANSPORT_GRID_THREADS),width);
  [(NSMutableArray *)program->loop_accesses removeAllObjects];
  command_emit(program,kind,bindings,MTLSizeMake(group?1:MIN(units,MESH_TRANSPORT_GRID_THREADS),1,1),
    MTLSizeMake(width,1,1),!group);
  return METAL_DONE;
}

/* kernel `kind` (KINDS_OF) on buffers d, s, a and v (0, 1, 3 and 4; an absent one d, and v, where none is given, the
   program's failure words: metal_gate) and the arguments p (2, its polls the program's where v is such words), the
   progress word (5): a GROUP kernel (`group`: publish,
   spin) as one threadgroup of the configured width, any other over `units` threads */
static struct metal_result dispatch(struct metal_program *program, uint32_t kind, void *d, void *s, void *a, void *v, struct args p,
                                    uint64_t units, int group) {
  if (kind >= KINDS) return (struct metal_result){EINVAL, "no kernel of this operation and data type (nccl-mesh-metal.m TYPED)"};
  if (!units) return METAL_DONE;
  @autoreleasepool {
    return dispatch4(program,kind,(id<MTLBuffer>)d,(id<MTLBuffer>)s,(id<MTLBuffer>)a,(id<MTLBuffer>)v,p,units,group);
  }
}

struct metal_result metal_copy(struct metal_program *program,void *dst,size_t dst_offset,void *src,size_t src_offset,size_t bytes) {
  const int words=!((dst_offset|src_offset|bytes)&3);
  const ncclMeshAccess uses[]={{{src,src_offset},bytes,ncclMeshAccessRead},{{dst,dst_offset},bytes,ncclMeshAccessWritten}};
  metal_accesses(program,uses,2);
  loop_access(program,src,src_offset,words?bytes/4:bytes,words?4:1,1,MetalReachRead);
  loop_access(program,dst,dst_offset,words?bytes/4:bytes,words?4:1,0,MetalReachWritten);
  return dispatch(program,COPY[!words],dst,src,NULL,NULL,(struct args){.dst=dst_offset,.src=src_offset,.n=words?bytes/4:bytes},words?bytes/4:bytes,0);
}

struct metal_result metal_mask_spans(struct metal_program *program,ncclMeshBuffer table,
    const struct mesh_dma_span *spans,size_t count,ncclMeshBuffer active) {
  if(!count)return METAL_DONE;
  struct metal_bytes lengths;
  const int failed=argument_pages(program,count*sizeof(uint64_t),&lengths);
  if(failed)return metal_failed(failed);
  for(size_t k=0;k<count;k++)((uint64_t *)lengths.contents)[k]=spans[k].bytes;
  const ncclMeshAccess inputs[]={{{lengths.buffer,lengths.offset},count*sizeof(uint64_t),ncclMeshAccessRead},
    {active,sizeof(uint32_t),ncclMeshAccessRead}};
  metal_accesses(program,inputs,2);
  for(size_t k=0;k<count;k++) {
    const ncclMeshAccess output={{table.buffer,table.offset+k*sizeof(struct mesh_dma_span)+offsetof(struct mesh_dma_span,bytes)},
      sizeof(uint64_t),ncclMeshAccessWritten};
    metal_accesses(program,&output,1);
  }
  const NSUInteger grid=MIN(count,MESH_TRANSPORT_GRID_THREADS),threads=MESH_TRANSPORT_ELEMENT_THREADS;
  loop_access(program,lengths.buffer,lengths.offset,count,sizeof(uint64_t),1,MetalReachRead);
  NSString *writes=[NSString stringWithFormat:
    @"{ [g] -> [b] : exists k : 0 <= k < %lu and %lu + %lu*k <= b < %lu + %lu*k and g = floor((k mod %lu)/%lu) }",
    (unsigned long)count,(unsigned long)(table.offset+offsetof(struct mesh_dma_span,bytes)),
    (unsigned long)sizeof(struct mesh_dma_span),
    (unsigned long)(table.offset+offsetof(struct mesh_dma_span,bytes)+sizeof(uint64_t)),
    (unsigned long)sizeof(struct mesh_dma_span),
    (unsigned long)grid,(unsigned long)threads];
  [(NSMutableArray *)program->group_accesses addObject:group_access(program,(id<MTLBuffer>)table.buffer,
    table.offset,count*sizeof(struct mesh_dma_span),0,MetalReachWritten,writes)];
  NSString *reads=[NSString stringWithFormat:@"{ [g] -> [b] : 0 <= g < %lu and %lu <= b < %lu }",
    (unsigned long)((grid+threads-1)/threads),(unsigned long)active.offset,(unsigned long)(active.offset+sizeof(uint32_t))];
  [(NSMutableArray *)program->group_accesses addObject:group_access(program,(id<MTLBuffer>)active.buffer,
    active.offset,sizeof(uint32_t),3,MetalReachRead,reads)];
  return dispatch(program,KIND_mask_spans,table.buffer,lengths.buffer,active.buffer,NULL,
    (struct args){.dst=table.offset,.src=lengths.offset,.n=count,.aux=active.offset,
      .step=sizeof(struct mesh_dma_span)/sizeof(uint64_t),.phase=offsetof(struct mesh_dma_span,bytes)/sizeof(uint64_t)},count,0);
}

struct metal_result metal_fill_pattern(struct metal_program *program,void *dst,size_t dst_offset,
    size_t bytes,uint64_t scalar,size_t element_bytes,size_t unit_bytes) {
  const ncclMeshAccess output={{dst,dst_offset},bytes,ncclMeshAccessWritten};
  metal_accesses(program,&output,1);
  loop_access(program,dst,dst_offset,bytes,1,0,MetalReachWritten);
  return dispatch(program,KIND_fill_pattern,dst,NULL,NULL,NULL,
    (struct args){.dst=dst_offset,.n=bytes,.scalar=scalar,.step=unit_bytes,.input=element_bytes},bytes,0);
}

/* operation `op`'s kernel of `type` (TYPED), KINDS where none is compiled */
static uint32_t typed(int op, int type) { return (unsigned)type <= METAL_MERGE ? TYPED[op][type] : KINDS; }

/* CPU arithmetic is another prepared leaf over the same region pages. The
   recorder owns dependencies, submission and completion for every backend. */
static struct metal_result native_emit(struct metal_program *program,NSString *text,const void *arguments,size_t bytes) {
  NSData *reaches=program->accesses;const struct MetalReach *uses=reaches.bytes;
  NSMutableArray *relations=[NSMutableArray array];
  for(NSUInteger k=0;k<reaches.length/sizeof *uses;k++)if(uses[k].length&&(uses[k].access&(MetalReachRead|MetalReachWritten))) {
    NSString *relation=[NSString stringWithFormat:@"{ [g] -> [b] : g = 0 and %lu <= b < %lu }",
      (unsigned long)uses[k].offset,(unsigned long)(uses[k].offset+uses[k].length)];
    [relations addObject:group_access(program,uses[k].buffer,uses[k].offset,uses[k].length,uses[k].binding,uses[k].access,relation)];
  }
  MeshCollectiveCommand *command=[[[MeshCollectiveCommand alloc] init] autorelease];
  command->source=[[MetalPageCommand order:[(NSArray *)program->commands count]+1 groups:1 backend:MetalPageCPU
    source:text function:@"mesh_cpu_entry" grid:MTLSizeMake(1,1,1) metadataSlot:0 accesses:relations] retain];
  command->source->frame=program->frame;
  command->reaches=[reaches copy];command->arguments=[[NSData alloc] initWithBytes:arguments length:bytes];
  [(NSMutableArray *)program->commands addObject:command];[(NSMutableData *)program->accesses setLength:0];
  return METAL_DONE;
}

struct metal_result metal_premultiply(struct metal_program *program,int type,void *dst,size_t dst_offset,
    void *src,size_t src_offset,size_t n,size_t element_bytes,struct metal_scalar scalar) {
  if(!scalar.buffer) {
    struct metal_bytes constant;
    if(argument_pages(program,sizeof scalar.bits,&constant))return metal_failed(ENOMEM);
    memcpy(constant.contents,&scalar.bits,sizeof scalar.bits);
    scalar.buffer=constant.buffer;scalar.offset=constant.offset;
  }
  const ncclMeshAccess uses[]={{{src,src_offset},n*element_bytes,ncclMeshAccessRead},
    {{scalar.buffer,scalar.offset},element_bytes,ncclMeshAccessRead},{{dst,dst_offset},n*element_bytes,ncclMeshAccessWritten}};
  metal_accesses(program,uses,3);
  if(type==ncclFloat64) {
    double *out=(double *)((char *)metal_contents(dst)+dst_offset);
    const double *input=(const double *)((const char *)metal_contents(src)+src_offset);
    const double *factor=(const double *)((const char *)metal_contents(scalar.buffer)+scalar.offset);
    const struct { double *out;const double *input,*factor; } arguments={out,input,factor};
    NSString *source=[NSString stringWithFormat:@"#include <stdint.h>\nstruct MeshCpuArguments { double *out; const double *input,*factor; };\nextern \"C\" void mesh_cpu_entry(const void *arguments) { const auto &a=*static_cast<const MeshCpuArguments *>(arguments); const double value=*a.factor; for(unsigned long k=0;k<%luul;k++)a.out[k]=a.input[k]*value; }\n",(unsigned long)n];
    return native_emit(program,source,&arguments,sizeof arguments);
  }
  loop_access(program,src,src_offset,n,element_bytes,1,MetalReachRead);
  loop_access(program,dst,dst_offset,n,element_bytes,0,MetalReachWritten);
  return dispatch(program,typed(OP_PREMUL,type),dst,src,scalar.buffer,NULL,
    (struct args){.dst=dst_offset,.src=src_offset,.n=n,.aux=scalar.offset},n,0);
}

struct metal_result metal_truncdiv(struct metal_program *program, int type, void *dst, size_t dst_offset, void *src, size_t src_offset, size_t n,
                                   size_t element_bytes, uint64_t divisor) {
  const ncclMeshAccess uses[]={{{src,src_offset},n*element_bytes,ncclMeshAccessRead},{{dst,dst_offset},n*element_bytes,ncclMeshAccessWritten}};
  metal_accesses(program,uses,sizeof uses/sizeof *uses);
  loop_access(program,src,src_offset,n,element_bytes,1,MetalReachRead);
  loop_access(program,dst,dst_offset,n,element_bytes,0,MetalReachWritten);
  return dispatch(program, typed(OP_TRUNCDIV, type), dst, src, NULL, NULL, (struct args){.dst = dst_offset, .src = src_offset, .n = n, .scalar = divisor},
                  n, 0);
}

/* a wait's arguments (nccl-mesh-metal.h struct metal_wait): `landed` at aux until scalar,
   the landed bytes at src, dst the destination's offset, its entry and channel for its failure's record.  Its page
   counts become the record's words here and in release_args alone */
static struct args release_args(const struct metal_release *r, size_t dst_offset) {
  return (struct args){.dst = dst_offset, .scalar = pages_to_word(r->value), .aux = r->word, .stride = (uint32_t)pages_to_word(r->stride)};
}

/* An access is a compiler operand; no overlap decision occurs in execution. */
void metal_accesses(struct metal_program *program,const ncclMeshAccess *uses,size_t n) {
  NSMutableData *accesses=(NSMutableData *)program->accesses;
  for(size_t k=0;k<n;k++)if(uses[k].bytes) {
    metal_resident(program,uses[k].storage.buffer);
    const uint32_t mode=(uses[k].access&ncclMeshAccessRead?MetalReachRead:0)|
      (uses[k].access&ncclMeshAccessWritten?MetalReachWritten:0)|(uses[k].access&ncclMeshAccessObserve?MetalReachObserve:0)|(uses[k].access&ncclMeshAccessNext?MetalReachNext:0)|(uses[k].access&ncclMeshAccessAtomic?MetalReachAtomic:0);
    const struct MetalReach reach={(id<MTLBuffer>)uses[k].storage.buffer,uses[k].storage.offset,uses[k].bytes,mode,-1};
    [accesses appendBytes:&reach length:sizeof reach];
  }
}
struct metal_result metal_publish(struct metal_program *program, const struct metal_release *r) {
  const ncclMeshAccess use={{r->words,r->word},sizeof(uint64_t),ncclMeshAccessWritten|ncclMeshAccessAtomic};metal_accesses(program,&use,1);
  return dispatch(program, KIND_publish, r->words, NULL, NULL, NULL, release_args(r, 0), 1, 1);
}

/* Many overwritten packets may name one completion word. Preparation joins
   their exact output ranges into one threshold column for that word. Runtime
   scans words relevant to the cell, never all packets of the tensor. */
struct output_observation { struct metal_wait wait; size_t word; };
struct output_observations { const struct metal_wait *word; size_t count,spans;
  const struct mesh_graph_span *span; const uint64_t *reach; const struct output_observation *values; };
static int output_observation_order(const void *left,const void *right) {
  const struct output_observation *a=left,*b=right;
  return (a->wait.data_offset>b->wait.data_offset)-(a->wait.data_offset<b->wait.data_offset);
}
static struct output_observations output_observations(const struct metal_zelloth_sum *sum) {
  NSMutableData *words=[NSMutableData data],*values=[NSMutableData dataWithLength:sum->waits*sizeof(struct output_observation)];
  struct output_observation *v=values.mutableBytes;
  for(size_t i=0;i<sum->waits;i++) {
    const struct metal_wait wait=sum->wait[i];const struct metal_wait *known=words.bytes;
    size_t word=0,count=words.length/sizeof wait;
    while(word<count&&(known[word].words!=wait.words||known[word].word!=wait.word))word++;
    if(word==count)[words appendBytes:&wait length:sizeof wait];
    v[i]=(struct output_observation){wait,word};
  }
  qsort(v,sum->waits,sizeof *v,output_observation_order);
  NSMutableData *spans=[NSMutableData dataWithLength:sum->waits*sizeof(struct mesh_graph_span)],
    *reach=[NSMutableData dataWithLength:sum->waits*sizeof(uint64_t)];
  struct mesh_graph_span *span=spans.mutableBytes;
  for(size_t i=0;i<sum->waits;i++)span[i]=(struct mesh_graph_span){v[i].wait.data_offset-sum->out_offset,
    v[i].wait.data_offset-sum->out_offset+v[i].wait.bytes};
  mesh_graph_span_reach(span,sum->waits,reach.mutableBytes);
  return (struct output_observations){words.bytes,words.length/sizeof(struct metal_wait),sum->waits,
    spans.bytes,reach.bytes,values.bytes};
}
struct output_observation_meet { const struct output_observations *observations; uint64_t *threshold; };
static void output_observation_met(void *context,size_t i) {
  const struct output_observation_meet *meet=context;
  const struct output_observation value=meet->observations->values[i];
  const uint64_t target=pages_to_word(value.wait.target);
  if(target>meet->threshold[value.word])meet->threshold[value.word]=target;
}
struct landing_cells_output { uint64_t *at; size_t inputs; size_t element_bytes; const struct output_observations *observations; };
static void landing_cell(void *context,uint64_t first,uint64_t elements,const uint64_t *through) {
  struct landing_cells_output *out=context;
  out->at[0]=first;out->at[1]=elements;
  memcpy(out->at+2,through,out->inputs*sizeof *through);
  memset(out->at+2+out->inputs,0,out->observations->count*sizeof *out->at);
  struct output_observation_meet meet={out->observations,out->at+2+out->inputs};
  mesh_graph_meets(out->observations->span,out->observations->reach,out->observations->spans,
    (struct mesh_graph_span){first*out->element_bytes,(first+elements)*out->element_bytes},output_observation_met,&meet);
  out->at+=2+out->inputs+out->observations->count;
}
static MeshFoldActivity *fold_activity(const struct metal_zelloth_sum *sum,const uint64_t *cells,
    size_t count,size_t stride,size_t width) {
  MeshFoldActivity *fold=[[[MeshFoldActivity alloc] init] autorelease];
  NSMutableData *transport=[NSMutableData dataWithLength:(sum->inputs+1)*sizeof(size_t)],
    *origins=[NSMutableData dataWithLength:(sum->inputs+1)*sizeof(size_t)],
    *ranges=[NSMutableData dataWithLength:2*count*sizeof(size_t)];
  size_t *input=transport.mutableBytes,*range=ranges.mutableBytes;
  for(size_t k=0;k<sum->inputs;k++)input[k]=sum->in[k].transport;
  input[sum->inputs]=sum->initial.transport;
  ((size_t *)origins.mutableBytes)[sum->inputs]=sum->initial_first;
  for(size_t q=0;q<count;q++){range[2*q]=cells[q*stride];range[2*q+1]=cells[q*stride+1];}
  fold->transports=[transport copy];fold->origins=[origins copy];fold->cells=[ranges copy];fold->width=width;return fold;
}
static struct metal_result native_zelloth(struct metal_program *program,int op,const struct metal_zelloth_sum *sum,
    uint32_t channel,uint64_t entry) {
  const struct output_observations observations=output_observations(sum);
  const size_t K=sum->inputs,W=observations.count,O=K+W;
  const size_t map_inputs=K+(sum->initial.count!=0);
  struct mesh_landing_map maps[K+1];size_t cursor[K+1];uint64_t through[K+1];
  for(size_t k=0;k<K;k++)maps[k]=(struct mesh_landing_map){sum->in[k].runs,sum->in[k].count,
    pages_to_word(pages_sub(sum->in[k].target,sum->in[k].packets))};
  if(sum->initial.count)maps[K]=(struct mesh_landing_map){sum->initial.runs,sum->initial.count,
    pages_to_word(pages_sub(sum->initial.target,sum->initial.packets))};
  const uint32_t count=(uint32_t)mesh_landing_cells(maps,map_inputs,MESH_TRANSPORT_ELEMENT_THREADS*MESH_TRANSPORT_VALUES_PER_THREAD,cursor,through,NULL,NULL);
  if(!count)return METAL_DONE;
  const size_t words=(2+O)*count,states=words+(count+2)/2;
  struct metal_bytes table;
  if(argument_pages(program,states*sizeof(uint64_t),&table))return metal_failed(ENOMEM);
  uint64_t *cells=table.contents;
  struct landing_cells_output output={cells,K,sizeof(double),&observations};
  mesh_landing_cells(maps,map_inputs,MESH_TRANSPORT_ELEMENT_THREADS*MESH_TRANSPORT_VALUES_PER_THREAD,cursor,through,landing_cell,&output);
  MeshFoldActivity *activity=fold_activity(sum,cells,count,2+O,sizeof(double));
  uint32_t *state=(uint32_t *)(cells+words),*prefix=state+count;
  memset(state,0,(count+1)*sizeof *state);
  struct metal_bytes pointers;
  if(argument_pages(program,(3*K+W)*sizeof(void *),&pointers))return metal_failed(ENOMEM);
  const double **inputs=pointers.contents;
  uint64_t **progress=(uint64_t **)(inputs+K),**ready=progress+K;
  void *violation=metal_contents(program->violation);
  const uint32_t polls=program->polls;
  for(size_t k=0;k<K;k++) {
    inputs[k]=(const double *)((const char *)metal_contents(sum->in[k].data)+sum->in[k].data_offset);
    progress[k]=sum->in[k].summed?(uint64_t *)((char *)metal_contents(sum->in[k].summed)+sum->in[k].summed_offset):NULL;
    ready[k]=(uint64_t *)((char *)metal_contents(sum->in[k].words)+sum->in[k].word);
    metal_resident(program,sum->in[k].summed);metal_resident(program,sum->in[k].words);
  }
  for(size_t k=0;k<W;k++)ready[K+k]=(uint64_t *)((char *)metal_contents(observations.word[k].words)+observations.word[k].word);
  struct metal_bytes waits={0};
  if(W&&argument_pages(program,W*sizeof(struct metal_wait),&waits))return metal_failed(ENOMEM);
  if(W)memcpy(waits.contents,observations.word,W*sizeof(struct metal_wait));
  const struct metal_wait *overwrite=waits.contents;
  const double *own=(const double *)((const char *)metal_contents(sum->own)+sum->own_offset);
  double *out=(double *)((char *)metal_contents(sum->out)+sum->out_offset);
  metal_resident(program,table.buffer);
  for(uint32_t q=0;q<count;q++) {
    const uint64_t *cell=cells+(2+O)*q;const size_t first=cell[0],elements=cell[1],offset=first*sizeof(double),bytes=elements*sizeof(double);
    const ncclMeshAccess uses[]={{{sum->own,sum->own_offset+offset},bytes,ncclMeshAccessRead},
      {{sum->out,sum->out_offset+offset},bytes,ncclMeshAccessWritten}};
    metal_accesses(program,uses,2);
    for(size_t k=0;k<K;k++) {
      const ncclMeshAccess input={{sum->in[k].data,sum->in[k].data_offset+offset},bytes,ncclMeshAccessRead};
      metal_accesses(program,&input,1);
      const ncclMeshAccess words[]={
        {{sum->in[k].words,sum->in[k].word},sizeof(uint64_t),ncclMeshAccessRead|ncclMeshAccessObserve},
        {{sum->in[k].summed,sum->in[k].summed_offset},sum->in[k].summed?sizeof(uint64_t):0,ncclMeshAccessWritten|ncclMeshAccessAtomic}};
      metal_accesses(program,words,2);
    }
    for(size_t k=0;k<W;k++)if(cell[2+K+k]) {
      const ncclMeshAccess use={{observations.word[k].words,observations.word[k].word},sizeof(uint64_t),ncclMeshAccessRead|ncclMeshAccessObserve};
      metal_accesses(program,&use,1);
    }
    const ncclMeshAccess completion={
      {table.buffer,table.offset+words*sizeof(uint64_t)},(count+1)*sizeof(uint32_t),
      ncclMeshAccessRead|ncclMeshAccessWritten|ncclMeshAccessAtomic};
    metal_accesses(program,&completion,1);
    const struct { double *out;const double *own;const double *const *inputs;uint64_t **ready,**progress;
      const uint64_t *cells;uint32_t *state,*prefix;const struct metal_wait *overwrite;void *violation;
      const uint32_t *const *activity;
    } arguments={out,own,inputs,ready,progress,cells,state,prefix,overwrite,violation,NULL};
    static NSString *const expression[]={@"value+next",@"value*next",@"fmax(value,next)",@"fmin(value,next)"};
    NSString *text=[NSString stringWithFormat:
      @"#include <math.h>\n#include \"mesh-publication.h\"\n#include \"nccl-mesh-metal.h\"\n"
       "#ifndef MESH_FOLD_ACTIVE\n#define MESH_FOLD_ACTIVE(k,q) true\n#endif\n"
       "#ifndef MESH_FOLD_OWN\n#define MESH_FOLD_OWN(q) true\n#endif\n"
       "struct MeshCpuArguments { double *out; const double *own; const double *const *inputs; uint64_t **ready,**progress; const uint64_t *cells; uint32_t *state,*prefix; const struct metal_wait *overwrite; void *violation; const uint32_t *const *activity; };\n"
       "extern \"C\" void mesh_cpu_entry(const void *arguments) { const auto &a=*static_cast<const MeshCpuArguments *>(arguments); "
       "[[maybe_unused]] const unsigned long q=%uu;const uint64_t *cell=a.cells+%luul; "
       "for(unsigned long k=0;k<%luul;k++)if(MESH_FOLD_ACTIVE(k,q))mesh_step_observe(a.ready[k],cell[2+k],a.violation,%uu,%uu,%lluul); "
       "for(unsigned long k=0;k<%luul;k++)if(cell[2+%luul+k])mesh_step_observe(a.ready[%luul+k],cell[2+%luul+k],a.violation,%uu,a.overwrite[k].channel,a.overwrite[k].at); "
       "const uint64_t identity=%lluul;double empty;__builtin_memcpy(&empty,&identity,sizeof empty); "
       "for(unsigned long j=%luul;j<%luul;j++){double value=MESH_FOLD_OWN(q)?a.own[j]:empty;for(unsigned long k=0;k<%luul;k++)if(MESH_FOLD_ACTIVE(k,q)){const double next=a.inputs[k][j];value=(%@);}a.out[j]=value;} "
       "__atomic_store_n(a.state+%uu,2u,__ATOMIC_RELEASE);const uint32_t at=mesh_graph_prefix(a.state,0,%uu,a.prefix);"
       "if(at){const uint64_t *through=a.cells+%luul*(at-1)+2;for(unsigned long k=0;k<%luul;k++)if(a.progress[k])mesh_publish(a.progress[k],through[k]);} }\n",
      q,(unsigned long)((2+O)*q),(unsigned long)K,polls,channel,(unsigned long long)entry,
      (unsigned long)W,(unsigned long)K,(unsigned long)K,(unsigned long)K,polls,
      (unsigned long long)sum->identity,(unsigned long)first,(unsigned long)(first+elements),(unsigned long)K,expression[op],q,count,(unsigned long)(2+O),(unsigned long)K];
    struct metal_result made=native_emit(program,text,&arguments,sizeof arguments);
    if(made.status)return made;
    MeshCollectiveCommand *command=[(NSArray *)program->commands lastObject];
    command->fold=[activity retain];command->foldCell=q;
  }
  return METAL_DONE;
}

/* One ordinary command per value version. Each threadgroup has its prepared
   cell, with every input threshold already resolved. There is no task claim,
   run search or dependency interpreter in the numerical kernel. */
struct metal_result metal_zelloth(struct metal_program *program,int type,int op,size_t element_bytes,uint32_t S,
    const struct metal_zelloth_sum *sums,uint32_t channel,uint64_t at) {
  if(type==ncclFloat64) {
    for(uint32_t s=0;s<S;s++){struct metal_result made=native_zelloth(program,op,sums+s,channel,at);if(made.status)return made;}
    return METAL_DONE;
  }
  const uint32_t kind=type==METAL_MERGE?KIND_zelloth_merge:
    (unsigned)type<ncclNumTypes&&(unsigned)op<4?REDUCTIONS[op][type]:KINDS;
  if(kind>=KINDS)return metal_failed(EINVAL);
  const uint64_t width=MESH_TRANSPORT_GROUP_THREADS;
  const uint64_t block=(type==METAL_MERGE?1:MESH_TRANSPORT_VALUES_PER_THREAD)*width;
  for(uint32_t s=0;s<S;s++) {
    const struct metal_zelloth_sum *sum=sums+s;
    const struct output_observations observations=output_observations(sum);
    const size_t K=sum->inputs,W=observations.count,O=K+W;
    struct mesh_landing_map *maps=malloc((K+1)*sizeof *maps);
    size_t *cursor=malloc((K+1)*sizeof *cursor);
    uint64_t *through=malloc((K+1)*sizeof *through);
    if(!maps||!cursor||!through){free(maps);free(cursor);free(through);return metal_failed(ENOMEM);}
    for(size_t k=0;k<K;k++)maps[k]=(struct mesh_landing_map){sum->in[k].runs,sum->in[k].count,
      pages_to_word(pages_sub(sum->in[k].target,sum->in[k].packets))};
    const size_t map_inputs=K+(sum->initial.count!=0);
    if(sum->initial.count)maps[K]=(struct mesh_landing_map){sum->initial.runs,sum->initial.count,
      pages_to_word(pages_sub(sum->initial.target,sum->initial.packets))};
    const uint64_t blocks=mesh_landing_cells(maps,map_inputs,block,cursor,through,NULL,NULL);
    if(!blocks){free(maps);free(cursor);free(through);continue;}
    const size_t cells=8+3*O,state=cells+(2+O)*blocks,prefix=state+(blocks+1)/2,total=prefix+1;
    struct metal_bytes table;
    if(argument_pages(program,total*sizeof(uint64_t),&table)){free(maps);free(cursor);free(through);return metal_failed(ENOMEM);}
    uint64_t *z=table.contents;
    z[0]=K;z[1]=blocks;z[2]=((id<MTLBuffer>)sum->own).gpuAddress+sum->own_offset;
    z[3]=((id<MTLBuffer>)sum->out).gpuAddress+sum->out_offset;
    z[4]=cells;z[5]=state;z[6]=prefix;z[7]=W;
    struct landing_cells_output output={z+cells,K,element_bytes,&observations};
    mesh_landing_cells(maps,map_inputs,block,cursor,through,landing_cell,&output);
    free(maps);free(cursor);free(through);
    memset(z+state,0,(total-state)*sizeof *z);
    const struct mesh_landing_run *last=sum->in[0].runs+sum->in[0].count-1;
    const size_t bytes=(last->first+last->elements)*element_bytes;
    const ncclMeshAccess own[]={{{sum->own,sum->own_offset},bytes,ncclMeshAccessRead},
      {{sum->out,sum->out_offset},bytes,ncclMeshAccessWritten}};
    metal_accesses(program,own,2);
    cell_access(program,sum->own,sum->own_offset,bytes,-1,MetalReachRead,z+cells,blocks,2+O,element_bytes);
    cell_access(program,sum->out,sum->out_offset,bytes,0,MetalReachWritten,z+cells,blocks,2+O,element_bytes);
    for(size_t k=0;k<K;k++) {
      const struct metal_zelloth_input *input=sum->in+k;
      z[8+3*k]=((id<MTLBuffer>)input->data).gpuAddress+input->data_offset;
      z[9+3*k]=((id<MTLBuffer>)input->words).gpuAddress+input->word;
      z[10+3*k]=input->summed?((id<MTLBuffer>)input->summed).gpuAddress+input->summed_offset:0;
      const ncclMeshAccess uses[]={{{input->data,input->data_offset},bytes,ncclMeshAccessRead},
        {{input->words,input->word},sizeof(uint64_t),ncclMeshAccessRead|ncclMeshAccessObserve},
        {{input->summed,input->summed_offset},input->summed?sizeof(uint64_t):0,ncclMeshAccessWritten|ncclMeshAccessAtomic}};
      metal_accesses(program,uses,3);
      cell_access(program,input->data,input->data_offset,bytes,-1,MetalReachRead,z+cells,blocks,2+O,element_bytes);
    }
    for(size_t k=0;k<W;k++) {
      const struct metal_wait *wait=observations.word+k;
      z[8+3*(K+k)]=((id<MTLBuffer>)wait->words).gpuAddress+wait->word;
      z[9+3*(K+k)]=wait->channel;z[10+3*(K+k)]=wait->at;
      const ncclMeshAccess use={{wait->words,wait->word},sizeof(uint64_t),ncclMeshAccessRead|ncclMeshAccessObserve};
      metal_accesses(program,&use,1);
    }
    metal_resident(program,table.buffer);
    struct MetalCommandBinding bindings[6];
    const struct args args={.src=table.offset,.scalar=sum->identity,.op=(uint32_t)op,.channel=channel,.at=at,.polls=program->polls};
    const struct metal_result recorded=record4(program,(id<MTLBuffer>)sum->out,(id<MTLBuffer>)table.buffer,nil,
      program->violation,args,bindings);
    if(recorded.status)return recorded;
    command_emit(program,kind,bindings,MTLSizeMake(blocks,1,1),MTLSizeMake(width,1,1),NO);
    MeshCollectiveCommand *command=[(NSArray *)program->commands lastObject];
    command->fold=[fold_activity(sum,z+cells,blocks,2+O,element_bytes) retain];
  }
  return METAL_DONE;
}

struct metal_result metal_open_gate(struct metal_program *program, void *words, size_t word, uint64_t value) {
  const ncclMeshAccess use={{words,word},sizeof(uint64_t),ncclMeshAccessWritten|ncclMeshAccessAtomic};metal_accesses(program,&use,1);
  return dispatch(program, KIND_publish, words, NULL, NULL, NULL, (struct args){.scalar = value, .aux = word}, 1, 1);
}

void *metal_contents(void *buffer) { return buffer ? ((id<MTLBuffer>)buffer).contents : NULL; }

/* mesh.metal's text as this library was built (.build/mesh_metal.h, xxd of it), then STEPS: made once, kept */
const char *metal_prelude(void) {
  static NSString *text;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    NSString *words = [[[NSString alloc] initWithBytes:mesh_metal length:mesh_metal_len encoding:NSUTF8StringEncoding] autorelease];
    words = [@(MetalCompilationPrelude()) stringByAppendingString:words];
    text = [[NSString alloc] initWithFormat:@"%@\n%@", words, STEPS];
  });
  return text.UTF8String;
}

size_t metal_command_count(const struct metal_program *program) { return [(NSArray *)program->commands count]; }
const ncclMeshSources *metal_sources(struct metal_program *program) {
  if(!program->declaration.commands) {
    NSMutableArray *sources=[NSMutableArray array];
    for(MeshCollectiveCommand *command in (NSArray *)program->commands)[sources addObject:command->source];
    program->declaration.commands=[sources copy];program->declaration.transfers=program->transfers;
  }
  return &program->declaration;
}
void metal_command_accesses(struct metal_program *program,size_t index,const ncclMeshAccess *uses,size_t count) {
  MeshCollectiveCommand *command=((NSArray *)program->commands)[index];
  metal_accesses(program,uses,count);
  NSMutableData *reaches=[[command->reaches mutableCopy] autorelease];[reaches appendData:program->accesses];
  [command->reaches release];command->reaches=[reaches copy];[(NSMutableData *)program->accesses setLength:0];
}
struct metal_result metal_specialize(struct metal_program *program,void *sources) {
  NSArray *made=sources;
  if(program->sources)return (struct metal_result){EINVAL,"a numerical source program is specialized once"};
  if(made.count!=[(NSArray *)program->commands count])return (struct metal_result){EINVAL,"specialization must cover every retained numerical declaration"};
  program->sources=[made copy];
  /* Relation operands end at specialization. Runtime ownership retains the
     completed source and its page operands, never the enclosing ISL context. */
  for(MeshCollectiveCommand *command in (NSArray *)program->commands) {
    [command->source->accesses release];command->source->accesses=nil;
    [command->source->announcements release];command->source->announcements=nil;
    [command->source->contributions release];command->source->contributions=nil;
    [command->fold release];command->fold=nil;
  }
  [(id)program->declaration.commands release];program->declaration.commands=NULL;
  [(id)program->transfers release];program->transfers=NULL;program->declaration.transfers=NULL;
  [(id)program->coverage_values release];program->coverage_values=NULL;
  return METAL_DONE;
}
struct metal_result metal_record(struct metal_program *program,const size_t *selected,size_t count,const ncclMeshStream *on) {
  if(!program->sources)return (struct metal_result){EINVAL,"complete the whole-function source solve before recording"};
  program->encoder=(id<MTL4ComputeCommandEncoder>)on->commandEncoder;
  program->table=(id<MTL4ArgumentTable>)on->argumentTable;
  program->residency=(id<MTLResidencySet>)on->residencySet;
  NSMutableArray *held=objc_getAssociatedObject(program->residency,&argumentPagesKey);
  if(!held){held=[NSMutableArray array];objc_setAssociatedObject(program->residency,&argumentPagesKey,held,OBJC_ASSOCIATION_RETAIN_NONATOMIC);}
  [held addObject:argument_owner(program)];
  struct metal_result result=METAL_DONE;
  @try {
    NSArray<MeshCollectiveCommand *> *commands=program->commands;
    for(NSUInteger at=0;at<count&&!result.status;at++) {
      const NSUInteger i=selected[at];
      MeshCollectiveCommand *command=commands[i];
      MetalPageSource *source=((NSArray *)program->sources)[i];
      [argument_owner(program)->sources addObject:source];
      id<MTLComputePipelineState> pipeline=nil;
      NSMutableData *bindings=[[command->bindings mutableCopy] autorelease];
      NSMutableData *reaches=[[command->reaches mutableCopy] autorelease];
      [reaches appendData:source->reaches];
      if(command->source->backend==MetalPageCPU) {
        if(!program->cpu_compiler||!program->cpu_resources) {
          result=(struct metal_result){EINVAL,"CPU collective source requires its declared compiler and resource operands"};break;
        }
        const char *artifact=mesh_cpu_compile(source->source.UTF8String,program->cpu_compiler);
        if(!artifact){result=(struct metal_result){EINVAL,say("collective CPU source: %s",mesh_cpu_error())};break;}
        NSMutableData *arguments=[[command->arguments mutableCopy] autorelease];
        if(source->metadata) {
          void *metadata=(char *)source->metadata.contents+source->metadataOffset;
          [arguments appendBytes:&metadata length:sizeof metadata];
        }
        const struct MetalReach *uses=reaches.bytes;
        NSMutableData *hostAccesses=[NSMutableData dataWithLength:reaches.length/sizeof *uses*sizeof(MeshHostAccess)];
        MeshHostAccess *host=hostAccesses.mutableBytes;
        for(NSUInteger k=0;k<reaches.length/sizeof *uses;k++)host[k]=(MeshHostAccess){uses[k].buffer.contents,uses[k].offset,uses[k].length,uses[k].access};
        const MeshCpuInstruction instruction={.arguments=arguments.bytes,.argument_bytes=arguments.length,
          .library=artifact,.symbol="mesh_cpu_entry",.accesses=host,.access_count=hostAccesses.length/sizeof *host};
        MeshCollectiveHost *bound=[[[MeshCollectiveHost alloc] init] autorelease];
        bound->program=mesh_cpu_program_create(&instruction,1,program->cpu_resources);
        if(!bound->program){result=(struct metal_result){EINVAL,say("collective CPU binding: %s",mesh_cpu_error())};break;}
        [argument_owner(program)->sources addObject:bound];
        MetalEncoderHost(program->encoder,reaches.bytes,reaches.length/sizeof(struct MetalReach),
          ^const char *(MeshHostCompletion complete,void *context,MeshHostTask *out) {
            return mesh_cpu_program_prepare(bound->program,complete,context,out);
          });
        continue;
      }
      if(source->metadata||command->sourceSpecialized) {
        NSError *error=nil;
        id<MTLLibrary> library=[[program->residency.device newLibraryWithSource:source->source
          options:MetalDeviceCompilerOptions(program->residency.device) error:&error] autorelease];
        pipeline=library?[MetalPipelineMake(library,command->source->function,@[],@{},@[],&error) autorelease]:nil;
        if(!pipeline){result=(struct metal_result){EINVAL,say("collective source preparation: %s",error.localizedDescription.UTF8String)};break;}
        if(source->metadata) {
          const struct MetalCommandBinding metadata={.buffer=source->metadata,.offset=source->metadataOffset,
            .length=source->metadata.length-source->metadataOffset,.index=command->source->metadataSlot,.access=MetalReachRead};
          [bindings appendBytes:&metadata length:sizeof metadata];
        }
      }
      if(!pipeline) {
        const char *message=NULL;struct kernels *compiled=program_kernels(program,&message);
        if(!compiled){result=(struct metal_result){EINVAL,message};break;}
        pipeline=compiled->pipelines[command->kind];
      }
      const struct MetalCommandDescription emitted={.pipeline=pipeline,.bindings=bindings.bytes,
        .bindingCount=bindings.length/sizeof(struct MetalCommandBinding),.reaches=reaches.bytes,
        .reachCount=reaches.length/sizeof(struct MetalReach),.groups=command->grid,
        .threads=command->threads,.threadGrid=command->threadGrid};
      MetalEncoderCompute(program->encoder,&emitted);
    }
  } @catch(NSException *failure) { result=(struct metal_result){EINVAL,say("collective page lowering: %s",failure.reason.UTF8String)}; }
  for(id<MTLBuffer> buffer in (NSSet *)program->resources)
    if(![program->residency containsAllocation:buffer])[program->residency addAllocation:buffer];
  [program->residency commit];return result;
}
void metal_end(struct metal_program *program) {
  [(id)program->accesses release];[(id)program->commands release];[(id)program->transfers release];
  [(id)program->group_accesses release];[(id)program->loop_accesses release];
  [(id)program->arguments release];[(id)program->resources release];[(id)program->sources release];
  [(id)program->coverage_values release];
  [(id)program->declaration.commands release];[program->violation release];
}
