#import <Metal/Metal.h>
#include "nccl-mesh-kernels.h"
#include "mesh_metal.h"
#include <stdio.h>

static int failed(const char *what, NSError *error) {
  fprintf(stderr, "nccl-mesh-kernels: %s: %s\n", what, error ? error.localizedDescription.UTF8String : "absent");
  return 1;
}

static int text(const char *path) {
  NSString *primitives = [[[NSString alloc] initWithBytes:mesh_metal length:mesh_metal_len encoding:NSUTF8StringEncoding] autorelease];
  NSString *source = [@[primitives, STEPS, KERNELS, TRACKS, TRACK_KERNEL] componentsJoinedByString:@""];
  NSError *error = nil;
  return [source writeToFile:@(path) atomically:YES encoding:NSUTF8StringEncoding error:&error] ? 0 : failed(path, error);
}

static int archive(const char *library_path, const char *archive_path) {
  id<MTLDevice> device = [MTLCreateSystemDefaultDevice() autorelease];
  NSError *error = nil;
  id<MTLLibrary> library = [[device newLibraryWithURL:[NSURL fileURLWithPath:@(library_path)] error:&error] autorelease];
  if (!library) return failed(library_path, error);
  id<MTLBinaryArchive> made = [[device newBinaryArchiveWithDescriptor:[[MTLBinaryArchiveDescriptor new] autorelease] error:&error] autorelease];
  if (!made) return failed("a binary archive", error);
  MTLComputePipelineDescriptor *described = [[MTLComputePipelineDescriptor new] autorelease];
  for (uint32_t kind = 0; kind < PIPELINES; kind++) {
    described.computeFunction = [[library newFunctionWithName:@(NAME[kind])] autorelease];
    error = nil;
    if (!described.computeFunction || ![made addComputePipelineFunctionsWithDescriptor:described error:&error]) return failed(NAME[kind], error);
  }
  return [made serializeToURL:[NSURL fileURLWithPath:@(archive_path)] error:&error] ? 0 : failed(archive_path, error);
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc == 2) return text(argv[1]);
    if (argc == 3) return archive(argv[1], argv[2]);
    fprintf(stderr, "usage: nccl-mesh-kernels TEXT | nccl-mesh-kernels METALLIB ARCHIVE\n");
    return 2;
  }
}
