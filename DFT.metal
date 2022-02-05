//
//  DFT.metal
//  InokiFFT
//
//  Created by inoki on 1/28/22.
//

#include <metal_stdlib>
using namespace metal;

float2 complexAdd(const float2 inA, const float2 inB)
{
    float2 out;
    out[0] = inA[0] + inB[0];
    out[1] = inA[1] + inB[1];
    return out;
}

float2 complexMul(const float2 inA, const float2 inB)
{
    float2 out;
    out[0] = inA[0] * inB[0] - inA[1] * inB[1];
    out[1] = inA[0] * inB[1] + inA[1] * inB[0];
    return out;
}

float2 complexExp(const float2 in)
{
    float2 out;
    float t = exp(in[0]);
    out[0] = t * cos(in[1]);
    out[1] = t * sin(in[1]);
    return out;
}

/// This is a Metal Shading Language (MSL) function
kernel void computeDFTMetal(device const float *in,
                       device float *out,
                       device const int *num,
                       uint index [[thread_position_in_grid]])
{
    if (index < (uint)*num) {
        int i = index * 2;
        out[i] = 0;
        out[i + 1] = 0;
        float2 temp1, temp2;
        for (int j = 0; j < *num; j++)
        {
            temp2[0] = 0; temp2[1] = -2 * 3.14159265 * index * j / *num;
            temp2 = complexExp(temp2);

            temp1[0] = in[j * 2]; temp1[1] = in[j * 2 + 1];
            temp1 = complexMul(temp1, temp2);

            // Copy back
            out[i] += temp1[0];
            out[i + 1] += temp1[1];
        }
    }
}

kernel void computeDFTMetalWithPrecomputedRoot(device const float *in,
                                               device const float *roots,
                                               device float *out,
                                               device const int *num,
                                               uint index [[thread_position_in_grid]])
{
    if (index < (uint)*num) {
        int i = index * 2;
        out[i] = 0;
        out[i + 1] = 0;
        float2 temp1, temp2;
        for (int j = 0; j < *num; j++)
        {
            temp2[0] = roots[2 * ((i * j) % (*num))];
            temp2[1] = roots[2 * ((i * j) % (*num)) + 1];

            temp1[0] = in[i]; temp1[1] = in[i + 1];
            temp1 = complexMul(temp1, temp2);

            // Copy back
            out[i] += temp1[0];
            out[i + 1] += temp1[1];
        }
    }
}
