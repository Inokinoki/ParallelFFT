//
//  FFT.mm
//  InokiFFT
//
//  FFT implementation using Metal
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <complex>
#include <cmath>

#include "DFT_Metal_private.h"

static id<MTLFunction> initFFTFunction(NSString *functionName, id<MTLDevice> device) {
    // Load the shader files with a .metal file extension in the project
    id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
    if (defaultLibrary == nil)
    {
        NSLog(@"Failed to find the default library.");
        return nil;
    }
    id<MTLFunction> loadedFunction = [defaultLibrary newFunctionWithName:functionName];
    return loadedFunction;
}

void calculateFFTMetal(std::complex<float>* inBuffer, std::complex<float>* outBuffer, size_t num) {
    @autoreleasepool {
        if (num == 0) return;
        
        // Calculate log2(num)
        int log2n = 0;
        size_t temp = num;
        while (temp > 1) {
            log2n++;
            temp /= 2;
        }
        
        // Verify num is power of 2
        if ((1 << log2n) != num) {
            NSLog(@"FFT size must be a power of 2");
            return;
        }
        
        id<MTLDevice> device = GetMetalSystemDevice();
        
        // Load bit-reverse permutation kernel
        id<MTLFunction> bitReverseFunc = initFFTFunction(@"fftBitReversePermute", device);
        if (bitReverseFunc == nil)
        {
            NSLog(@"Failed to find the fftBitReversePermute function.");
            return;
        }
        
        // Load butterfly kernel
        id<MTLFunction> butterflyFunc = initFFTFunction(@"fftButterfly", device);
        if (butterflyFunc == nil)
        {
            NSLog(@"Failed to find the fftButterfly function.");
            return;
        }
        
        NSError* error = nil;
        
        id<MTLComputePipelineState> bitReversePSO = [device newComputePipelineStateWithFunction:bitReverseFunc error:&error];
        if (bitReversePSO == nil)
        {
            NSLog(@"Failed to create bit-reverse pipeline state object, error %@.", error);
            return;
        }
        
        id<MTLComputePipelineState> butterflyPSO = [device newComputePipelineStateWithFunction:butterflyFunc error:&error];
        if (butterflyPSO == nil)
        {
            NSLog(@"Failed to create butterfly pipeline state object, error %@.", error);
            return;
        }
        
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (commandQueue == nil)
        {
            NSLog(@"Failed to find the command queue.");
            return;
        }
        
        // Prepare input buffer
        id<MTLBuffer> bufferIn = [device newBufferWithLength:num * sizeof(float) * 2 options:MTLResourceStorageModeShared];
        float *inDataPtr = (float *)bufferIn.contents;
        
        for (unsigned long index = 0; index < num; index++)
        {
            inDataPtr[index * 2] = inBuffer[index].real();
            inDataPtr[index * 2 + 1] = inBuffer[index].imag();
        }
        
        // Prepare working buffer (for bit-reversed data and butterfly operations)
        id<MTLBuffer> bufferData = [device newBufferWithLength:num * sizeof(float) * 2 options:MTLResourceStorageModeShared];
        
        // Prepare num buffer
        id<MTLBuffer> bufferNum = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
        *((int *)bufferNum.contents) = (int)num;
        
        // Prepare log2n buffer
        id<MTLBuffer> bufferLog2n = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
        *((int *)bufferLog2n.contents) = log2n;
        
        // Prepare stage buffer
        id<MTLBuffer> bufferStage = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
        
        // Step 1: Bit-reverse permutation
        {
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            assert(commandBuffer != nil);
            
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            assert(computeEncoder != nil);
            
            [computeEncoder setComputePipelineState:bitReversePSO];
            [computeEncoder setBuffer:bufferIn offset:0 atIndex:0];
            [computeEncoder setBuffer:bufferData offset:0 atIndex:1];
            [computeEncoder setBuffer:bufferNum offset:0 atIndex:2];
            [computeEncoder setBuffer:bufferLog2n offset:0 atIndex:3];
            
            MTLSize gridSize = MTLSizeMake(num, 1, 1);
            NSUInteger threadGroupSize = bitReversePSO.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > num)
            {
                threadGroupSize = num;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
            
            [computeEncoder dispatchThreads:gridSize
                      threadsPerThreadgroup:threadgroupSize];
            
            [computeEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }
        
        // Step 2: FFT butterfly stages
        for (int s = 1; s <= log2n; s++) {
            *((int *)bufferStage.contents) = s;
            
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            assert(commandBuffer != nil);
            
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            assert(computeEncoder != nil);
            
            [computeEncoder setComputePipelineState:butterflyPSO];
            [computeEncoder setBuffer:bufferData offset:0 atIndex:0];
            [computeEncoder setBuffer:bufferNum offset:0 atIndex:1];
            [computeEncoder setBuffer:bufferStage offset:0 atIndex:2];
            
            // We need num/2 threads for butterfly operations
            MTLSize gridSize = MTLSizeMake(num / 2, 1, 1);
            NSUInteger threadGroupSize = butterflyPSO.maxTotalThreadsPerThreadgroup;
            if (threadGroupSize > num / 2)
            {
                threadGroupSize = num / 2;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
            
            [computeEncoder dispatchThreads:gridSize
                      threadsPerThreadgroup:threadgroupSize];
            
            [computeEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }
        
        // Copy results back
        float *outDataPtr = (float *)bufferData.contents;
        for (unsigned long index = 0; index < num; index++)
        {
            outBuffer[index].real(outDataPtr[index * 2]);
            outBuffer[index].imag(outDataPtr[index * 2 + 1]);
        }
    }
}
