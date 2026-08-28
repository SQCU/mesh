#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <mach/mach_time.h>
static double now(void){static mach_timebase_info_data_t t;if(!t.denom)mach_timebase_info(&t);return mach_absolute_time()*(double)t.numer/t.denom/1e9;}
int main(int argc,char**argv){@autoreleasepool{
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
 id<MTLCommandQueue> q=[dev newCommandQueue];
 NSLog(@"device %@ unified=%d maxbuf=%.1fGB",dev.name,(int)dev.hasUnifiedMemory,dev.maxBufferLength/1e9);
 int N=4096; int Ks[]={1,4,16,32,64,128,256,512,1024,4096}; 
 for(int t32=0;t32<2;t32++){
  MPSDataType dt = t32? MPSDataTypeFloat32 : MPSDataTypeFloat16;
  int es = t32?4:2;
  id<MTLBuffer> A=[dev newBufferWithLength:(NSUInteger)N*N*es options:MTLResourceStorageModeShared];
  for(int ki=0;ki<10;ki++){int K=Ks[ki];
   id<MTLBuffer> B=[dev newBufferWithLength:(NSUInteger)N*K*es options:MTLResourceStorageModeShared];
   id<MTLBuffer> C=[dev newBufferWithLength:(NSUInteger)N*K*es options:MTLResourceStorageModeShared];
   MPSMatrixDescriptor*da=[MPSMatrixDescriptor matrixDescriptorWithRows:N columns:N rowBytes:N*es dataType:dt];
   MPSMatrixDescriptor*db=[MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:K*es dataType:dt];
   MPSMatrix*ma=[[MPSMatrix alloc]initWithBuffer:A descriptor:da];
   MPSMatrix*mb=[[MPSMatrix alloc]initWithBuffer:B descriptor:db];
   MPSMatrix*mc=[[MPSMatrix alloc]initWithBuffer:C descriptor:db];
   MPSMatrixMultiplication*mm=[[MPSMatrixMultiplication alloc]initWithDevice:dev transposeLeft:NO transposeRight:NO resultRows:N resultColumns:K interiorColumns:N alpha:1.0 beta:0.0];
   for(int w=0;w<2;w++){id<MTLCommandBuffer>cb=[q commandBuffer];[mm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];[cb commit];[cb waitUntilCompleted];}
   int reps=K<=64?50:(K<=512?10:3);
   double t0=now();
   for(int r=0;r<reps;r++){id<MTLCommandBuffer>cb=[q commandBuffer];[mm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];[cb commit];[cb waitUntilCompleted];}
   double d=(now()-t0)/reps; double fl=2.0*N*N*K;
   printf("%s N=4096 K=%5d  t=%8.3f ms  %8.1f GFLOP/s\n",t32?"fp32":"fp16",K,d*1e3,fl/d/1e9);
  }
 }
}return 0;}
