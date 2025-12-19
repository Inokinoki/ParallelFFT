#pragma once

#include <complex>

void calculateFFT(std::complex<float>* in, std::complex<float>* out, size_t num);

#ifdef __HAS_CUDA__
void calculateFFTCUDA(std::complex<float>* in, std::complex<float>* out, size_t num);
#endif

#ifdef HAS_METAL
void calculateFFTMetal(std::complex<float>* inBuffer, std::complex<float>* outBuffer, size_t num);
#endif
