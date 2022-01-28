//
//  DFT_Metal_private.h
//  InokiFFT
//
//  Created by inoki on 1/28/22.
//

#ifndef DFT_Metal_private_h
#define DFT_Metal_private_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#ifdef __cplusplus
extern "C" {
#endif

id<MTLDevice> GetMetalSystemDevice();

#ifdef __cplusplus
}
#endif


#endif /* DFT_Metal_private_h */
