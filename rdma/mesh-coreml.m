/* mesh-coreml.m: a Core ML model's predictions on caller memory, for torch-mesh's Neural Engine engine
   (torch-mesh/torch_mesh/engines.py Ane).  A prediction's input is the caller's memory as a float16 MLMultiArray (no
   copy: on unified memory, torch's MPS tensors' shared storage), and its output is written into the caller's memory
   (Core ML's output backing), so the CPU does no copying on the prediction's critical path: through coremltools each
   prediction copied its input and output on the CPU, and once the CPU's performance cores clocked down beside a busy
   GPU (1.3-2.8 GHz from 4.3 on an M4 Pro), a 3 ms prediction took 9.  Called through ctypes, which releases the
   interpreter while it runs.

   mesh_coreml_load(package, input, output, units): compiles a model package (MLModel compileModelAtURL) and loads it
   on `units` (0 CPU and Neural Engine, 1 all, 2 CPU only); its handle, or -1 (mesh_coreml_error).
   mesh_coreml_predict(handle, in, in_shape, in_rank, out, out_shape, out_rank): one prediction, the input read from
   `in` and the output written to `out` (row-major float16 of those shapes); 1 where Core ML wrote the output into
   `out` itself, 0 where it was copied there from Core ML's own, -1 on an error. */
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static NSMutableArray *models_;
static NSMutableArray *names_;
static char error_[2048];

const char *mesh_coreml_error(void) { return error_; }

long mesh_coreml_load(const char *package, const char *input, const char *output, int units) {
  @autoreleasepool {
    if (!models_) { models_ = [NSMutableArray new]; names_ = [NSMutableArray new]; }
    NSError *error = nil;
    NSURL *compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:@(package)] error:&error];
    if (!compiled) { snprintf(error_, sizeof error_, "compile: %s", error.localizedDescription.UTF8String); return -1; }
    MLModelConfiguration *configuration = [MLModelConfiguration new];
    configuration.computeUnits = units == 1 ? MLComputeUnitsAll : units == 2 ? MLComputeUnitsCPUOnly : MLComputeUnitsCPUAndNeuralEngine;
    MLModel *model = [MLModel modelWithContentsOfURL:compiled configuration:configuration error:&error];
    if (!model) { snprintf(error_, sizeof error_, "load: %s", error.localizedDescription.UTF8String); return -1; }
    [models_ addObject:model];
    [names_ addObject:@[@(input), @(output)]];
    return (long)models_.count - 1;
  }
}

static MLMultiArray *wrap(void *data, const int64_t *shape, int rank, NSError **error) {
  NSMutableArray *dims = [NSMutableArray array], *strides = [NSMutableArray array];
  int64_t stride = 1;
  for (int i = rank - 1; i >= 0; i--) { [strides insertObject:@(stride) atIndex:0]; stride *= shape[i]; }
  for (int i = 0; i < rank; i++) [dims addObject:@(shape[i])];
  return [[MLMultiArray alloc] initWithDataPointer:data shape:dims dataType:MLMultiArrayDataTypeFloat16 strides:strides
                                       deallocator:nil error:error];
}

int mesh_coreml_predict(long handle, void *in, const int64_t *in_shape, int in_rank, void *out, const int64_t *out_shape,
                        int out_rank) {
  @autoreleasepool {
    if (handle < 0 || handle >= (long)models_.count) { snprintf(error_, sizeof error_, "no model %ld", handle); return -1; }
    MLModel *model = models_[handle];
    NSArray *names = names_[handle];
    NSError *error = nil;
    MLMultiArray *input = wrap(in, in_shape, in_rank, &error), *backing = wrap(out, out_shape, out_rank, &error);
    if (!input || !backing) { snprintf(error_, sizeof error_, "arrays: %s", error.localizedDescription.UTF8String); return -1; }
    MLDictionaryFeatureProvider *features = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{names[0]: [MLFeatureValue featureValueWithMultiArray:input]} error:&error];
    if (!features) { snprintf(error_, sizeof error_, "features: %s", error.localizedDescription.UTF8String); return -1; }
    MLPredictionOptions *options = [MLPredictionOptions new];
    options.outputBackings = @{names[1]: backing};
    id<MLFeatureProvider> result = [model predictionFromFeatures:features options:options error:&error];
    if (!result) { snprintf(error_, sizeof error_, "predict: %s", error.localizedDescription.UTF8String); return -1; }
    MLMultiArray *got = [result featureValueForName:names[1]].multiArrayValue;
    if (got.dataPointer == out) return 1;
    int64_t count = 1;
    for (int i = 0; i < out_rank; i++) count *= out_shape[i];
    __block int copied = 0;
    [got getBytesWithHandler:^(const void *bytes, NSInteger size) {
      if ((int64_t)size >= count * 2) { memcpy(out, bytes, (size_t)count * 2); copied = 1; }
    }];
    if (!copied) { snprintf(error_, sizeof error_, "output: %ld elements, %lld expected", (long)got.count, (long long)count); return -1; }
    return 0;
  }
}
