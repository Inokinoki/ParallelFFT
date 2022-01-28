#include <complex>

#define M_PI 3.14159265

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"

__device__ __forceinline__ cuComplex cuComplexExp(cuComplex z)
{
    cuComplex res;
    float t = expf(z.x);
    sincosf(z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

// Kernel definition
__global__ void calculateDFTCUDAKernel(cuComplex* in, cuComplex* out, size_t num)
{
    int i = threadIdx.x;
    if (i < num)
    {
        out[i].x = 0;
        out[i].y = 0;
        for (int j = 0; j < num; j++)
        {
            out[i] = cuCaddf(out[i], 
                cuCmulf(in[j], cuComplexExp(make_cuComplex(0, -2 * M_PI * i * j / num)))
            );
        }
    }
}

void calculateDFTCUDA(std::complex<float>* in, std::complex<float>* out, size_t num)
{
    if (num == 0) return;

    // Allocate vectors in device memory
    cuComplex* d_in;
    cudaMalloc(&d_in, num * sizeof(cuComplex));
    cuComplex* d_out;
    cudaMalloc(&d_out, num * sizeof(cuComplex));

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_in, in, num * sizeof(cuComplex), cudaMemcpyHostToDevice);

    calculateDFTCUDAKernel<<<1, num>>>(d_in, d_out, num);
    cudaMemcpy(out, d_out, num * sizeof(cuComplex), cudaMemcpyDeviceToHost);

    cudaError_t cudaStatus;

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "DFT Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching DFT Kernel!\n", cudaStatus);
        goto Error;
    }

Error:
    cudaFree(d_in);
    cudaFree(d_out);
}

// Kernel definition
__global__ void calculateDFTCUDAKernelWithPrecomputedRoot(cuComplex* in, cuComplex* roots, cuComplex* out, size_t num)
{
    int i = threadIdx.x;
    if (i < num)
    {
        out[i].x = 0;
        out[i].y = 0;
        for (int j = 0; j < num; j++)
        {
            out[i] = cuCaddf(out[i], cuCmulf(in[j], roots[i * j % num]));
        }
    }
}

void calculateDFTCUDALargeMem(std::complex<float>* in, std::complex<float>* out, size_t num)
{
    if (num == 0) return;

    // Allocate vectors in device memory
    cuComplex* d_in;
    cudaMalloc(&d_in, num * sizeof(cuComplex));
    cuComplex* d_out;
    cudaMalloc(&d_out, num * sizeof(cuComplex));
    cuComplex* d_roots;
    cudaMalloc(&d_roots, num * sizeof(cuComplex));

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_in, in, num * sizeof(cuComplex), cudaMemcpyHostToDevice);

    // Precompute DFT
    cuComplex* roots = (cuComplex *)malloc(sizeof(cuComplex) * num);
    for (int i = 0; i < num; i++)
    {
        std::complex<float> root = std::exp(std::complex<float>(0, -2 * M_PI * i / num));
        roots[i] = make_cuComplex(root.real(), root.imag());
    }
    cudaMemcpy(d_roots, roots, num * sizeof(cuComplex), cudaMemcpyHostToDevice);
    free(roots);

    calculateDFTCUDAKernelWithPrecomputedRoot<<<1, num >>>(d_in, d_roots, d_out, num);

    cudaError_t cudaStatus;

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "DFT Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching DFT Kernel!\n", cudaStatus);
        goto Error;
    }

    // Copy back the results
    cudaMemcpy(out, d_out, num * sizeof(cuComplex), cudaMemcpyDeviceToHost);

Error:
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_roots);
}
