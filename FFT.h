#pragma once

#include <complex>

void calculateFFT(std::complex<float>* in, std::complex<float>* out, size_t num);
#ifdef __HAS_CUDA__
void calculateFFTCUDA(std::complex<float>* in, std::complex<float>* out, size_t num);
#endif
