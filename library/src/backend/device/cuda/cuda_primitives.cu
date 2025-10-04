//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025 James Sandham
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//********************************************************************************
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "common.cuh"
#include "primitive_kernels.cuh"

#include "cuda_primitives.h"

#include "../../../trace.h"

//-------------------------------------------------------------------------------
// find maximum
//-------------------------------------------------------------------------------
template <typename T>
T linalg::cuda_find_maximum(int size, const T* array)
{
    T* workspace = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&workspace, sizeof(T) * 256));

    find_max_kernel_part1<256><<<256, 256>>>(size, array, workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    find_max_kernel_part2<256><<<1, 256>>>(workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    T result;
    CHECK_CUDA(cudaMemcpy(&result, workspace, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(workspace));

    return result;
}

//-------------------------------------------------------------------------------
// find minimum
//-------------------------------------------------------------------------------
template <typename T>
T linalg::cuda_find_minimum(int size, const T* array)
{
    T* workspace = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&workspace, sizeof(T) * 256));

    find_min_kernel_part1<256><<<256, 256>>>(size, array, workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    find_min_kernel_part2<256><<<1, 256>>>(workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    T result;
    CHECK_CUDA(cudaMemcpy(&result, workspace, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(workspace));

    return result;
}

//-------------------------------------------------------------------------------
// exclusive scan
//-------------------------------------------------------------------------------
template <typename T>
void linalg::cuda_exclusive_scan(int size, T* array)
{
    ROUTINE_TRACE("linalg::cuda_exclusive_scan");

    assert(size > 0);

    int nblocks = (size - 1) / 256 + 1;

    assert(size >= nblocks);

    T* workspace = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&workspace, sizeof(T) * (size + 2 * nblocks)));
    CHECK_CUDA(cudaMemcpy(workspace, array, sizeof(T) * size, cudaMemcpyDeviceToDevice));

    T* workspace1 = workspace;
    T* workspace2 = workspace + size;
    T* workspace3 = workspace + size + nblocks;

    exclusive_scan_kernel_part1<256>
        <<<((size - 1) / 256 + 1), 256>>>(workspace1, array, workspace2, size);
    CHECK_CUDA_LAUNCH_ERROR();

    exclusive_scan_kernel_part1<256>
        <<<((size - 1) / 256 + 1), 256>>>(workspace2, workspace3, (T*)nullptr, size);
    CHECK_CUDA_LAUNCH_ERROR();

    exclusive_scan_kernel_part2<256><<<((size - 1) / 256 + 1), 256>>>(workspace3, array, size);
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaFree(workspace));
}

template int32_t linalg::cuda_find_minimum<int32_t>(int size, const int32_t* array);
template int64_t linalg::cuda_find_minimum<int64_t>(int size, const int64_t* array);
template double  linalg::cuda_find_minimum<double>(int size, const double* array);

template int32_t linalg::cuda_find_maximum<int32_t>(int size, const int32_t* array);
template int64_t linalg::cuda_find_maximum<int64_t>(int size, const int64_t* array);
template double  linalg::cuda_find_maximum<double>(int size, const double* array);

template void linalg::cuda_exclusive_scan<int32_t>(int size, int32_t* array);
template void linalg::cuda_exclusive_scan<int64_t>(int size, int64_t* array);
template void linalg::cuda_exclusive_scan<double>(int size, double* array);
