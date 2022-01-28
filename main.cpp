// InokiFourier.cpp
//

#include <iostream>
#include "DFT.h"

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
#ifndef __HAS_CUDA__
#ifndef HAS_METAL
	std::cout << "Using CPU implementation" << std::endl;
	calculateDFT(inBuffer, outBuffer, FFT_LEN);
#else
    std::cout << "Using Metal implementation" << std::endl;
    calculateDFTMetal(inBuffer, outBuffer, FFT_LEN);
#endif
#else
	std::cout << "Using customized CUDA implementation" << std::endl;
	calculateDFTCUDA(inBuffer, outBuffer, FFT_LEN);
#endif // !__NVCC__
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
