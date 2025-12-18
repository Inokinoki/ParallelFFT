# Discrete Fourier Transform (DFT/FFT) implementations

This project has experimental implementations of DFT/FFT in CUDA and Apple Metal. Use it at your own risk (remember to check the array border if you would like to use them in your own project).

## DFT Implementations
- `DFT.cpp` - CPU DFT implementation
- `DFT.cu` - CUDA DFT implementations (with or without precomputed complex roots)
- `DFT.metal` / `DFT.mm` - Apple Metal DFT implementations (with or without precomputed complex roots)

## FFT Implementations (Cooley-Tukey Algorithm)
- `FFT.cpp` - CPU FFT implementation using iterative Cooley-Tukey algorithm
- `FFT.cu` - CUDA FFT implementation with parallel bit-reversal and butterfly operations
- `FFT.metal` / `FFT.mm` - Apple Metal FFT implementation with parallel bit-reversal and butterfly operations

## Building

### CPU only
```bash
g++ -o fourier main.cpp DFT.cpp FFT.cpp -std=c++11
```

### With CUDA
```bash
nvcc -o fourier main.cpp DFT.cpp FFT.cpp DFT.cu FFT.cu -D__HAS_CUDA__
```

### With Metal (macOS)
```bash
clang++ -o fourier main.cpp DFT.cpp FFT.cpp DFT.mm FFT.mm DFT_Metal_private.m \
    -framework Foundation -framework Metal -DHAS_METAL -std=c++11
```

## Usage

The FFT implementations require input sizes that are powers of 2 (e.g., 64, 128, 256, 512, etc.).
