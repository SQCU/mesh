/* design/the-compiler.md, order 4 */
#import <CoreML/CoreML.h>
#include <stdint.h>
#include <string.h>

struct mesh_coreml_view {
    const char *name;
    void *pages;
    const int64_t *shape, *stride;
    int32_t rank, type;
};

static const MLMultiArrayDataType types[] = {MLMultiArrayDataTypeFloat16, MLMultiArrayDataTypeFloat32,
                                             MLMultiArrayDataTypeInt32};

static void report(NSError *error, char *message, size_t room) {
    if (room) strlcpy(message, error ? error.localizedDescription.UTF8String : "", room);
}

static MLModelConfiguration *configuration(void) {
    MLModelConfiguration *made = [MLModelConfiguration new];
    made.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    return made;
}

/* MLModel.h, compileModelAtURL:error: */
void *mesh_coreml_model(const char *package, const char *compiled, char *message, size_t room) {
    @autoreleasepool {
        NSError *error = nil;
        NSURL *destination = [NSURL fileURLWithPath:@(compiled)];
        NSURL *built = [MLModel compileModelAtURL:[NSURL fileURLWithPath:@(package)] error:&error];
        MLModel *model = nil;
        if (built && [NSFileManager.defaultManager moveItemAtURL:built toURL:destination error:&error])
            model = [MLModel modelWithContentsOfURL:destination configuration:configuration() error:&error];
        report(error, message, room);
        return (__bridge_retained void *)model;
    }
}

void mesh_coreml_release(void *model) {
    (void)(__bridge_transfer id)model;
}

static MLMultiArray *viewed(const struct mesh_coreml_view *view, NSError **error) {
    NSMutableArray *shape = [NSMutableArray array], *stride = [NSMutableArray array];
    for (int32_t d = 0; d < view->rank; d++) {
        [shape addObject:@(view->shape[d])];
        [stride addObject:@(view->stride[d])];
    }
    return [[MLMultiArray alloc] initWithDataPointer:view->pages shape:shape dataType:types[view->type]
                                             strides:stride deallocator:nil error:error];
}

/* MLMultiArray.h, initWithDataPointer:shape:dataType:strides:deallocator:error:; MLPredictionOptions.h, outputBackings */
int mesh_coreml_predict(void *model, const struct mesh_coreml_view *arguments, uint32_t argument_count,
                        const struct mesh_coreml_view *results, uint32_t result_count, char *message, size_t room) {
    @autoreleasepool {
        NSError *error = nil;
        NSMutableDictionary *given = [NSMutableDictionary dictionary], *backings = [NSMutableDictionary dictionary];
        for (uint32_t k = 0; k < argument_count && !error; k++)
            given[@(arguments[k].name)] = [MLFeatureValue featureValueWithMultiArray:viewed(&arguments[k], &error)];
        for (uint32_t k = 0; k < result_count && !error; k++)
            backings[@(results[k].name)] = viewed(&results[k], &error);
        id<MLFeatureProvider> made = nil;
        if (!error) {
            MLDictionaryFeatureProvider *features = [[MLDictionaryFeatureProvider alloc] initWithDictionary:given
                                                                                                    error:&error];
            MLPredictionOptions *options = [MLPredictionOptions new];
            options.outputBackings = backings;
            if (features)
                made = [(__bridge MLModel *)model predictionFromFeatures:features options:options error:&error];
        }
        report(error, message, room);
        if (!made) return -1;
        int written = 0;
        for (NSString *name in backings) {
            MLMultiArray *result = [made featureValueForName:name].multiArrayValue;
            if (result == backings[name]) written++;
            else [result transferToMultiArray:backings[name]];
        }
        return written;
    }
}

static NSString *device(id<MLComputeDeviceProtocol> used) {
    if (!used) return @"unknown";
    if ([used isKindOfClass:MLNeuralEngineComputeDevice.class]) return @"ane";
    return [used isKindOfClass:MLGPUComputeDevice.class] ? @"gpu" : @"cpu";
}

/* MLComputePlan.h, computeDeviceUsageForMLProgramOperation: */
int mesh_coreml_plan(const char *compiled, char *text, size_t room) {
    @autoreleasepool {
        __block MLComputePlan *plan = nil;
        __block NSError *failure = nil;
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        [MLComputePlan loadContentsOfURL:[NSURL fileURLWithPath:@(compiled)] configuration:configuration()
                       // mesh callback (preparation (completion handler); passed to completionHandler:): callback inversion of control is outside the working specification, and runtime callback overhead is inadmissible in performance-critical code
                       completionHandler:^(MLComputePlan *loaded, NSError *error) { void (*mesh_callback_disabled)(void) = &mesh_callbacks_are_outside_spec;
            plan = loaded;
            failure = error;
            dispatch_semaphore_signal(done);
        }];
        dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
        if (!plan) {
            report(failure, text, room);
            return -1;
        }
        NSMutableString *lines = [NSMutableString string];
        for (MLModelStructureProgramOperation *operation in plan.modelStructure.program.functions[@"main"].block.operations) {
            MLComputePlanDeviceUsage *usage = [plan computeDeviceUsageForMLProgramOperation:operation];
            NSMutableArray *supported = [NSMutableArray array];
            for (id<MLComputeDeviceProtocol> able in usage.supportedComputeDevices) [supported addObject:device(able)];
            [lines appendFormat:@"%@ %@ %@ %@\n", operation.operatorName, operation.outputs.firstObject.name,
                                usage ? device(usage.preferredComputeDevice) : @"none",
                                usage ? [supported componentsJoinedByString:@","] : @"none"];
        }
        if (room) strlcpy(text, lines.UTF8String, room);
        return 0;
    }
}
