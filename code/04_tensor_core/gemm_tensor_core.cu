#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include "../common/matrix_utils.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define USE_WMMA 1
#include <mma.h>
using namespace nvcuda;
#else
#define USE_WMMA 0
#endif

int main(int argc, char** argv)
{
    std::cout << "Tensor Core GEMM skeleton target. Please implement WMMA logic based on your GPU architecture." << std::endl;
    return 0;
}
