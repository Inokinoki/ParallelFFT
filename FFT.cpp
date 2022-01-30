
#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include <complex>
#include <vector>

unsigned int bitReverse(unsigned int x, int log2n)
{
	int n = 0;
	int mask = 0x1;
	for (int i = 0; i < log2n; i++)
	{
		n <<= 1;
		n |= (x & 1);
		x >>= 1;
	}
	return n;
}

void fft_impl(std::complex<float> *a, std::complex<float>* b, int num)
{
	std::complex<float> J(0, 1);
	int n = num;
	int log2n = 0;
	while (num > 1) {
		log2n++;
		num /= 2;
	}
	for (unsigned int i = 0; i < n; ++i) {
		b[bitReverse(i, log2n)] = a[i];
	}
	for (int s = 1; s <= log2n; ++s) {
		int m = 1 << s;
		int m2 = m >> 1;
		std::complex<float> w(1, 0);
		std::complex<float> wm = std::exp(-J * (float)(M_PI / m2));
		for (int j = 0; j < m2; ++j) {
			for (int k = j; k < n; k += m) {
				std::complex<float> t = w * b[k + m2];
				std::complex<float> u = b[k];
				b[k] = u + t;
				b[k + m2] = u - t;
			}
			w *= wm;
		}
	}
}

void calculateFFT(std::complex<float>* in, std::complex<float>* out, size_t num)
{
	fft_impl(in, out, num);
}
