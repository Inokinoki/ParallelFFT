// InokiFourier.cpp
//

#include <iostream>
#include "DFT.h"
#include "FFT.h"

#define FFT_LEN 128

int main()
{
	std::complex<float> inBuffer[128], outBuffer[128];
	for (int i = 0; i < FFT_LEN; i++)
	{
		switch (i % 8)
		{
		case 0:
			inBuffer[i] = 1;
			break;
		case 1: case 7:
			inBuffer[i] = std::sqrt(2) / 2;
			break;
		case 2: case 6:
			inBuffer[i] = 0;
			break;
		case 3: case 5:
			inBuffer[i] = -std::sqrt(2) / 2;
			break;
		case 4:
			inBuffer[i] = -1;
			break;
		default:
			break;
		}
	}

#ifdef __HAS_CUDA__
	std::cout << "Using CUDA implementation" << std::endl;
	std::cout << "DFT result:" << std::endl;
	calculateDFTCUDA(inBuffer, outBuffer, FFT_LEN);
	for (int i = 0; i < FFT_LEN; i++)
	{
		std::cout << std::abs(outBuffer[i]) << " ";
	}
	std::cout << std::endl;
	
	std::cout << "FFT result:" << std::endl;
	calculateFFTCUDA(inBuffer, outBuffer, FFT_LEN);
#elif defined(HAS_METAL)
	std::cout << "Using Metal implementation" << std::endl;
	std::cout << "DFT result:" << std::endl;
	calculateDFTMetal(inBuffer, outBuffer, FFT_LEN);
	for (int i = 0; i < FFT_LEN; i++)
	{
		std::cout << std::abs(outBuffer[i]) << " ";
	}
	std::cout << std::endl;
	
	std::cout << "FFT result:" << std::endl;
	calculateFFTMetal(inBuffer, outBuffer, FFT_LEN);
#else
	std::cout << "Using CPU implementation" << std::endl;
	std::cout << "DFT result:" << std::endl;
	calculateDFT(inBuffer, outBuffer, FFT_LEN);
	for (int i = 0; i < FFT_LEN; i++)
	{
		std::cout << std::abs(outBuffer[i]) << " ";
	}
	std::cout << std::endl;
	
	std::cout << "FFT result:" << std::endl;
	calculateFFT(inBuffer, outBuffer, FFT_LEN);
#endif

	float maxOut = -1;
	int maxI = 0;
	for (int i = 0; i < FFT_LEN; i++)
	{
		float outAbs = std::abs(outBuffer[i]);
		std::cout << outAbs << " ";
		if (outAbs > maxOut)
		{
			maxOut = outAbs;
			maxI = i;
		}
	}
	std::cout << std::endl << "Max: " << maxOut << " at " << maxI << std::endl;
	return 0;
}
