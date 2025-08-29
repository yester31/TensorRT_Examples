#include <cuda.h>
#include <cuda_fp16.h>
#include "preprocKernel.h"

using half = __half;

/*
PreprocForward(...)
    INPUT  = BGR[NHWC](0, 255) int8_t
    OUTPUT = RGB[NCHW](0.,1.) float or half
    This equation include 3 steps
    1. Scale Image to range [0., 1.], /255
    2. Shuffle form HWC to CHW
    3. BGR -> RGB
*/
template <typename T>
__global__ void PreprocForward(T *output,      // [N,C(RGB),H,W]
                               T const *input, // [N,H,W,C(BGR)]
                               int32_t const batchSize, int32_t const channel, int32_t const height, int32_t const width,
                               int32_t const nthreads) // nthreads
{
    size_t pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (pos >= nthreads)
        return;

    const int32_t w_idx = pos % width;
    int32_t idx = pos / width;
    const int32_t h_idx = idx % height;
    idx /= height;
    const int32_t c_idx = idx % channel;
    const int32_t b_idx = idx / channel;

    int32_t s_idx = b_idx * height * width * channel + h_idx * width * channel + w_idx * channel + (channel - 1) - c_idx;

    output[pos] = input[s_idx] / static_cast<T>(255.);
}

template <typename T>
cudaError_t PreprocImpl(cudaStream_t stream, int32_t const maxThreadsPerBlock, T const *input,
                        int32_t const batchSize, int32_t const channel, int32_t const height, int32_t const width,
                        T *output)
{
    int32_t const outputSize = batchSize * channel * height * width;
    int32_t blocksPerGrid = static_cast<int32_t>(ceil(static_cast<float>(outputSize) / maxThreadsPerBlock));

    PreprocForward<T><<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(output, input, batchSize, channel, height, width, outputSize);

    return cudaGetLastError();
}

#define SPECIALIZED_IMPL(T)                                                                                                        \
    template cudaError_t PreprocImpl<T>(cudaStream_t stream, int32_t const maxThreadsPerBlock, T const *input,                     \
                                        int32_t const batchSize, int32_t const channel, int32_t const height, int32_t const width, \
                                        T *output);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(half)