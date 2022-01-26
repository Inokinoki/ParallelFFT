
#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include <complex>

void calculateDFT(std::complex<float>* in, std::complex<float>* out, size_t num)
{
	for (int i = 0; i < num; i++)
	{
		out[i] = 0;
		for (int j = 0; j < num; j++)
		{
			out[i] += in[j] * std::exp(std::complex<float>(0, - 2 * M_PI * i * j / num));
		}
	}
}
