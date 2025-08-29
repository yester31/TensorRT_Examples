#ifndef TRT_PREPROC_KERNEL_H
#define TRT_PREPROC_KERNEL_H

#include <cuda_runtime.h>
#include <stdint.h>

template <typename T>
cudaError_t PreprocImpl(cudaStream_t stream, int32_t const maxThreadsPerBlock, T const *input,
                        int32_t const batchSize, int32_t const channel, int32_t const height, int32_t const width,
                        T *output);

#endif // TRT_PREPROC_KERNEL_H