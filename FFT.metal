//
//  FFT.metal
//  InokiFFT
//
//  FFT implementation using Cooley-Tukey algorithm for Metal
//

#include <metal_stdlib>
using namespace metal;

// Complex number operations
float2 complexAdd(const float2 inA, const float2 inB)
{
    return float2(inA[0] + inB[0], inA[1] + inB[1]);
}

float2 complexSub(const float2 inA, const float2 inB)
{
    return float2(inA[0] - inB[0], inA[1] - inB[1]);
}

float2 complexMul(const float2 inA, const float2 inB)
{
    return float2(inA[0] * inB[0] - inA[1] * inB[1],
                  inA[0] * inB[1] + inA[1] * inB[0]);
}

// Bit reverse function
uint bitReverse(uint x, int log2n)
{
    uint n = 0;
    for (int i = 0; i < log2n; i++)
    {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

// Kernel for bit-reverse permutation
kernel void fftBitReversePermute(device const float *in [[buffer(0)]],
                                  device float *out [[buffer(1)]],
                                  device const int *num [[buffer(2)]],
                                  device const int *log2n [[buffer(3)]],
                                  uint index [[thread_position_in_grid]])
{
    int n = *num;
    if (index < (uint)n)
    {
        uint reversedIndex = bitReverse(index, *log2n);
        // Copy complex number (2 floats)
        out[reversedIndex * 2] = in[index * 2];
        out[reversedIndex * 2 + 1] = in[index * 2 + 1];
    }
}

// Kernel for FFT butterfly operation at a given stage
kernel void fftButterfly(device float *data [[buffer(0)]],
                         device const int *num [[buffer(1)]],
                         device const int *stage [[buffer(2)]],
                         uint index [[thread_position_in_grid]])
{
    int n = *num;
    int s = *stage;
    
    int m = 1 << s;      // 2^stage
    int m2 = m >> 1;     // m/2
    
    // Each thread handles one butterfly operation
    // We need n/2 butterflies per stage
    if (index < (uint)(n / 2))
    {
        // Find which group this thread belongs to and position within group
        int group = index / m2;
        int j = index % m2;
        
        // Calculate the two indices for butterfly
        int k = group * m + j;
        int k2 = k + m2;
        
        // Calculate twiddle factor: W_m^j = exp(-2*pi*i*j/m)
        float angle = -2.0f * 3.14159265358979323846f * j / m;
        float2 w = float2(cos(angle), sin(angle));
        
        // Load data
        float2 u = float2(data[k * 2], data[k * 2 + 1]);
        float2 t_val = float2(data[k2 * 2], data[k2 * 2 + 1]);
        
        // Butterfly operation
        float2 t = complexMul(w, t_val);
        
        float2 result1 = complexAdd(u, t);
        float2 result2 = complexSub(u, t);
        
        // Store results
        data[k * 2] = result1[0];
        data[k * 2 + 1] = result1[1];
        data[k2 * 2] = result2[0];
        data[k2 * 2 + 1] = result2[1];
    }
}
