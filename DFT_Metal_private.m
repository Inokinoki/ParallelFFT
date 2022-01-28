//
//  DFT.m
//  InokiFFT
//
//  Created by inoki on 1/28/22.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "DFT_Metal_private.h"

id<MTLDevice> GetMetalSystemDevice() {
    return MTLCreateSystemDefaultDevice();
}
