//
//  DFT.m
//  InokiFFT
//
//  Created by inoki on 1/28/22.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <complex>

#include "DFT_Metal_private.h"

id<MTLFunction> initFunction(NSString *functionName, id<MTLDevice> device) {
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

void calculateDFTMetal(std::complex<float>* inBuffer, std::complex<float>* outBuffer, size_t num) {
    @autoreleasepool {
        id<MTLDevice> device = GetMetalSystemDevice();
        id<MTLFunction> func = initFunction(@"computeDFTMetal", device);
        if (func == nil)
        {
            NSLog(@"Failed to find the DFT function.");
            return;
        }

        NSError* error = nil;
        id<MTLComputePipelineState> funcPSO = [device newComputePipelineStateWithFunction: func error:&error];

        if (funcPSO == nil)
        {
            //  If the Metal API validation is enabled, you can find out more information about what
            //  went wrong.  (Metal API validation is enabled by default when a debug build is run
            //  from Xcode)
            NSLog(@"Failed to created pipeline state object, error %@.", error);
            return;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (commandQueue == nil)
        {
            NSLog(@"Failed to find the command queue.");
            return;
        }

        // Prepare data
        id<MTLBuffer> bufferIn = [device newBufferWithLength:num * sizeof(float) * 2 options:MTLResourceStorageModeShared];
        float *inDataPtr = (float *)bufferIn.contents;

        for (unsigned long index = 0; index < num; index++)
        {
            inDataPtr[index * 2] = inBuffer[index].real();
            inDataPtr[index * 2 + 1] = inBuffer[index].imag();
        }

        id<MTLBuffer> bufferOut = [device newBufferWithLength:num * sizeof(float) * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferNum = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
        *((int *)bufferNum.contents) = (int)num;
        NSLog(@"%d", *((int *)bufferNum.contents));

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        assert(commandBuffer != nil);

        // Start a compute pass.
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        assert(computeEncoder != nil);

        // Encode the pipeline state object and its parameters.
        [computeEncoder setComputePipelineState:funcPSO];
        [computeEncoder setBuffer:bufferIn offset:0 atIndex:0];
        [computeEncoder setBuffer:bufferOut offset:0 atIndex:1];
        [computeEncoder setBuffer:bufferNum offset:0 atIndex:2];

        MTLSize gridSize = MTLSizeMake(num, 1, 1);
        NSUInteger threadGroupSize = funcPSO.maxTotalThreadsPerThreadgroup;
        NSLog(@"My GPU has %lu thread groups.", (unsigned long)threadGroupSize);
        if (threadGroupSize > num)
        {
            threadGroupSize = num;
        }
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        // Encode the compute command.
        [computeEncoder dispatchThreads:gridSize
                  threadsPerThreadgroup:threadgroupSize];

        // End the compute pass.
        [computeEncoder endEncoding];

        // Execute the command.
        [commandBuffer commit];

        // Normally, you want to do other work in your app while the GPU is running,
        // but in this example, the code simply blocks until the calculation is complete.
        [commandBuffer waitUntilCompleted];

        float *outDataPtr = (float *)bufferOut.contents;
        // Write back
        for (unsigned long index = 0; index < num; index++)
        {
            outBuffer[index].real(outDataPtr[index * 2]);
            outBuffer[index].imag(outDataPtr[index * 2 + 1]);
        }
        // TODO: The result is buggy
    }
}
