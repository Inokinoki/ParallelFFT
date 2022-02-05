# Discrete Fourier Transform (DFT/FFT) implementations

This project has experimental implementations of DFT/FFT in CUDA and Apple Metal. Use it as your own risk (remember to check the array boarder if you would like to use them in your own project).

- `DFT.cu` has DFT implementations (with or without precomputed complex roots) in CUDA
- `DFT.metal` has DFT implementations (with or without precomputed complex roots) in Apple Metal
- `FFT.cpp` includes an FFT CPU implementation
- Parallel FFT is work in progress...
