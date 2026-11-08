# Metal 4 commit feedback: sources, quoted verbatim, and a measurement

Retrieved 2026-11-08. Each excerpt in §1 and §2 was copied by script from the cited lines of the file named; nothing
in a block is retyped. §3 is a measurement, not a source: the program and its output are copied by the same script,
which built and ran it.

- **Source:** Apple, Metal.framework headers, macOS SDK 26.4.1 (`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`),
  `System/Library/Frameworks/Metal.framework/Headers/`.
  - `MTL4CommandQueue.h` sha256 51bfecbd27ee6ba37891f42f0b75532dd915b9ccb6d6316f6f2966e8983bb881.
  - `MTL4CommitFeedback.h` sha256 e07d4a4e63680ad32664d0bb9c1e7710abb941efc96a8816e36e319b26f05188.
- **Source:** Apple, MetalValidation(1), `/usr/share/man/man1/MetalValidation.1`, as `man MetalValidation | col -b`
  prints it.

## 1. Commit options and their feedback handlers

`MTL4CommandQueue.h` L55–68:

```objc
/// Represents options to configure a commit operation on a command queue.
///
/// You pass these options as a parameter when you call ``MTL4CommandQueue/commit:count:options:``.
///
/// - Note Instances of this class are not thread-safe. If your app modifies a shared commit options instance from
/// multiple threads simultaneously, you are responsible for providing external synchronization.
MTL_EXPORT
API_AVAILABLE(macos(26.0), ios(26.0))
@interface MTL4CommitOptions : NSObject

/// Registers a commit feedback handler that Metal calls with feedback data when available.
///
/// - Parameter block: ``MTL4CommitFeedbackHandler`` that Metal invokes.
- (void)addFeedbackHandler:(MTL4CommitFeedbackHandler)block;
```

`MTL4CommandQueue.h` L237–259:

```objc
/// Enqueues an array of command buffer instances for execution with a set of options.
///
/// Provide an ``MTL4CommitOptions`` instance to configure the commit operation.
///
/// The order in which you sort the command buffers in the array is meaningful, especially when it contains suspending/resuming
/// render passes. A suspending/resuming render pass is a render pass you create by calling
/// ``MTL4CommandBuffer/renderCommandEncoderWithDescriptor:options:``,
/// and provide `MTL4RenderEncoderOptionSuspending` or `MTL4RenderEncoderOptionResuming` for the `options` parameter.
///
/// If your command buffers contain suspend/resume render passes, ensure that the first command buffer only suspends,
/// and the last one only resumes. Additionally, make sure that all intermediate command buffers are both suspending
/// and resuming.
///
/// When you commit work from multiple threads, modifying and reusing the same options instance,
/// you are responsible for externally synchronizing access to it.
///
/// - Parameters:
///   - commandBuffers: an array of ``MTL4CommandBuffer``.
///   - count: the number of ``MTL4CommandBuffer`` instances in the `commandBuffers` array.
///   - options: an instance of ``MTL4CommitOptions`` that configures the commit operation.
- (void)commit:(const id<MTL4CommandBuffer> _Nonnull[_Nonnull])commandBuffers
         count:(NSUInteger)count
       options:(MTL4CommitOptions *)options;
```

`MTL4CommandQueue.h` L261–267:

```objc
/// Schedules an operation to signal a GPU event with a specific value after all GPU work prior to this point is complete.
///
/// - Parameters:
///   - event: ``MTLEvent`` to signal.
///   - value: the value to signal the ``MTLEvent`` with.
- (void)signalEvent:(id<MTLEvent>)event
              value:(uint64_t)value;
```

`MTL4CommitFeedback.h` L19–27:

```objc
/// Defines the block signature for a callback Metal invokes to provide your app feedback after completing a workload.
///
/// You register a commit feedback block with Metal by providing an instance of ``MTL4CommitOptions`` to
/// the command queue's commit method, ``MTL4CommandQueue/commit:count:options:``. The commit options instance
/// references your commit feedback handler after you add it via its ``MTL4CommitOptions/addFeedbackHandler:``
/// method.
///
/// - Parameter commitFeedback: a commit feedback instance containing information about the workload.
typedef void (^ NS_SWIFT_SENDABLE MTL4CommitFeedbackHandler)(id<MTL4CommitFeedback> commitFeedback) API_AVAILABLE(macos(26.0), ios(26.0));
```

The headers do not say whether a handler that `addFeedbackHandler:` registers is called for one commit or for every
commit that carries the options.

## 2. API validation

MetalValidation(1), lines 25–48 of its `col -b` text:

```
DESCRIPTION
     Unless otherwise noted all environment variables listed must be set
     before any Metal device is created in the process.  Any changes to these
     values after device creation will not take effect.

     Metal provides two modes of validation, API and GPU.

     API validation validates CPU API usage correctness such as calling draw
     without a pipeline set.

     GPU validation validates GPU API correctness such as accessing invalid
     GPU memory.

ENVIRONMENT
     MTL_DEBUG_LAYER
	   Defaults to 0.
	   All environment variables prefixed with MTL_DEBUG_LAYER only apply
	   when MTL_DEBUG_LAYER is non-zero.
	   Any value other than 0 will enable API validation.

     MTL_DEBUG_LAYER_ERROR_MODE
	   Defaults to "assert"
	   Valid values are "assert", "ignore" and "nslog"
	   Sets the behavior for when a debug layer error occurs.
```

## 3. Measured: how many commits a registered handler serves

On this Mac: Apple M5 Max, macOS 26.3.2 (25D2150).
Mode bit 1 surrounds each commit with the host loop's events (an input event set by the host and waited for by the
queue, a step event the queue signals after the commit); bit 2 gives each commit its own `MTL4CommitOptions` with
its handler; bit 4 reuses one options instance and adds a handler to it before each commit.

`probe.m`:

```objc
#import <Metal/Metal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <unistd.h>
static _Atomic int calls;
int main(int argc, char **argv) {
  @autoreleasepool {
    int mode = argc > 1 ? atoi(argv[1]) : 0, n = argc > 2 ? atoi(argv[2]) : 4;
    id<MTLDevice> gpu = MTLCreateSystemDefaultDevice();
    id<MTL4CommandQueue> queue = [gpu newMTL4CommandQueue];
    id<MTL4CommandAllocator> pool[2] = {[gpu newCommandAllocator], [gpu newCommandAllocator]};
    id<MTL4CommandBuffer> own[2] = {[gpu newCommandBuffer], [gpu newCommandBuffer]};
    id<MTLSharedEvent> input = [gpu newSharedEvent], step = [gpu newSharedEvent];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    MTL4CommitOptions *shared = [MTL4CommitOptions new];
    [shared addFeedbackHandler:^(id<MTL4CommitFeedback> f) { atomic_fetch_add(&calls, 1); dispatch_semaphore_signal(done); }];
    for (int i = 0; i < n; i++) {
      if (i >= 2) [step waitUntilSignaledValue:i - 1 timeoutMS:5000];
      [pool[i % 2] reset];
      if (mode & 1) { input.signaledValue = i + 1; [queue waitForEvent:input value:i + 1]; }
      id<MTL4CommandBuffer> command = own[i % 2];
      [command beginCommandBufferWithAllocator:pool[i % 2]];
      id<MTL4ComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder endEncoding];
      [command endCommandBuffer];
      MTL4CommitOptions *options = shared;
      if (mode & 4) [shared addFeedbackHandler:^(id<MTL4CommitFeedback> f) { atomic_fetch_add(&calls, 1); dispatch_semaphore_signal(done); }];
      if (mode & 2) {
        options = [[MTL4CommitOptions new] autorelease];
        [options addFeedbackHandler:^(id<MTL4CommitFeedback> f) { atomic_fetch_add(&calls, 1); dispatch_semaphore_signal(done); }];
      }
      [queue commit:&command count:1 options:options];
      if (mode & 1) [queue signalEvent:step value:i + 1];
    }
    if (mode & 1) printf("step reached %d: %s\n", n, [step waitUntilSignaledValue:n timeoutMS:5000] ? "yes" : "no");
    int got = 0;
    for (int i = 0; i < n; i++) { if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 5000000000LL))) break; got++; }
    usleep(1000000);
    printf("mode %d (events %s, options %s): %d commits, %d feedbacks waited, %d handler calls\n", mode,
      mode & 1 ? "yes" : "no", mode & 2 ? "one a commit" : mode & 4 ? "one reused, a handler added before each commit" : "one reused", n, got, atomic_load(&calls));
  }
  return 0;
}
```

Built and run:

```
$ clang -fno-objc-arc -fblocks -O1 -framework Metal -framework Foundation probe.m -o probe
$ ./probe 0 8
mode 0 (events no, options one reused): 8 commits, 1 feedbacks waited, 1 handler calls
$ ./probe 1 8
step reached 8: yes
mode 1 (events yes, options one reused): 8 commits, 1 feedbacks waited, 1 handler calls
$ ./probe 2 8
mode 2 (events no, options one a commit): 8 commits, 8 feedbacks waited, 8 handler calls
$ ./probe 3 8
step reached 8: yes
mode 3 (events yes, options one a commit): 8 commits, 8 feedbacks waited, 8 handler calls
$ ./probe 4 8
mode 4 (events no, options one reused, a handler added before each commit): 8 commits, 8 feedbacks waited, 9 handler calls
```

With one options instance reused, eight commits called its handler once. With one options instance a commit, each of
the eight called its handler. With a handler added before each commit, the first commit called the two handlers it
carried and each later commit the one added before it: a handler serves the first commit that carries it, and no other.
