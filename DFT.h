#pragma once

#include <complex>

void calculateDFT(std::complex<float> *in, std::complex<float> *out, size_t num);
#ifdef __HAS_CUDA__
void calculateDFTCUDA(std::complex<float>* in, std::complex<float>* out, size_t num);
void calculateDFTCUDALargeMem(std::complex<float>* in, std::complex<float>* out, size_t num);
#endif
