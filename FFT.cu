#include <complex>

#define M_PI 3.14159265358979323846

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"

// Bit reverse function for index permutation
__device__ unsigned int bitReverseCUDA(unsigned int x, int log2n)
{
    unsigned int n = 0;
    for (int i = 0; i < log2n; i++)
    {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

// Kernel for bit-reverse permutation
__global__ void bitReversePermuteKernel(cuComplex* in, cuComplex* out, int num, int log2n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num)
    {
        out[bitReverseCUDA(i, log2n)] = in[i];
    }
}

// Kernel for FFT butterfly operation at a given stage
__global__ void fftButterflyKernel(cuComplex* data, int num, int stage)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int m = 1 << stage;      // 2^stage
    int m2 = m >> 1;         // m/2
    
    // Each thread handles one butterfly operation
    // We need num/2 butterflies per stage
    if (i < num / 2)
    {
        // Find which group this thread belongs to and position within group
        int group = i / m2;
        int j = i % m2;
        
        // Calculate the two indices for butterfly
        int k = group * m + j;
        int k2 = k + m2;
        
        // Calculate twiddle factor: W_m^j = exp(-2*pi*i*j/m)
        float angle = -2.0f * M_PI * j / m;
        cuComplex w;
        w.x = cosf(angle);
        w.y = sinf(angle);
        
        // Butterfly operation
        cuComplex t = cuCmulf(w, data[k2]);
        cuComplex u = data[k];
        
        data[k] = cuCaddf(u, t);
        data[k2] = cuCsubf(u, t);
    }
}

void calculateFFTCUDA(std::complex<float>* in, std::complex<float>* out, size_t num)
{
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
        fprintf(stderr, "FFT size must be a power of 2\n");
        return;
    }
    
    // Allocate device memory
    cuComplex* d_in;
    cuComplex* d_data;
    cudaMalloc(&d_in, num * sizeof(cuComplex));
    cudaMalloc(&d_data, num * sizeof(cuComplex));
    
    // Copy input to device
    cudaMemcpy(d_in, in, num * sizeof(cuComplex), cudaMemcpyHostToDevice);
    
    // Calculate grid and block dimensions
    int blockSize = 256;
    int numBlocks = (num + blockSize - 1) / blockSize;
    int numBlocksHalf = ((num / 2) + blockSize - 1) / blockSize;
    
    // Step 1: Bit-reverse permutation
    bitReversePermuteKernel<<<numBlocks, blockSize>>>(d_in, d_data, num, log2n);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Bit-reverse kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after bit-reverse kernel!\n", cudaStatus);
        goto Error;
    }
    
    // Step 2: FFT butterfly stages
    for (int s = 1; s <= log2n; s++) {
        fftButterflyKernel<<<numBlocksHalf, blockSize>>>(d_data, num, s);
        
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "FFT butterfly kernel launch failed at stage %d: %s\n", s, cudaGetErrorString(cudaStatus));
            goto Error;
        }
        
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after FFT stage %d!\n", cudaStatus, s);
            goto Error;
        }
    }
    
    // Copy result back to host
    cudaMemcpy(out, d_data, num * sizeof(cuComplex), cudaMemcpyDeviceToHost);

Error:
    cudaFree(d_in);
    cudaFree(d_data);
}
