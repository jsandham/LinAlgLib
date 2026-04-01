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

    constexpr int blocksize = 256;

    // Build level sizes for hierarchical block sums:
    // level 0: block sums of the original input,
    // level k: block sums of level k-1.
    std::vector<int> level_sizes;
    int current_size = size;
    while(current_size > 1)
    {
        int blocks = (current_size - 1) / blocksize + 1;
        level_sizes.push_back(blocks);
        current_size = blocks;
    }

    size_t total_workspace_count = static_cast<size_t>(size);
    for(int level_size : level_sizes)
    {
        total_workspace_count += static_cast<size_t>(level_size);
    }

    T* workspace = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&workspace, sizeof(T) * total_workspace_count));
    CHECK_CUDA(cudaMemcpy(workspace, array, sizeof(T) * size, cudaMemcpyDeviceToDevice));

    // Partition workspace into [input copy][level sums...].
    std::vector<T*> level_ptrs(level_sizes.size(), nullptr);
    T* level_base = workspace + size;
    for(size_t i = 0; i < level_sizes.size(); ++i)
    {
        level_ptrs[i] = level_base;
        level_base += level_sizes[i];
    }

    // Forward pass:
    // 1) scan original input into array, producing level 0 block sums.
    // 2) scan each level's block sums in-place, producing next level sums.
    T* level0_workspace = level_ptrs.empty() ? nullptr : level_ptrs[0];
    exclusive_scan_kernel_part1<blocksize><<<((size - 1) / blocksize + 1), blocksize>>>(
        workspace,
        array,
        level0_workspace,
        size);
    CHECK_CUDA_LAUNCH_ERROR();

    for(size_t level = 0; level < level_ptrs.size(); ++level)
    {
        T* input_output = level_ptrs[level];
        T* next_workspace = (level + 1 < level_ptrs.size()) ? level_ptrs[level + 1] : nullptr;
        int n = level_sizes[level];

        exclusive_scan_kernel_part1<blocksize><<<((n - 1) / blocksize + 1), blocksize>>>(
            input_output,
            input_output,
            next_workspace,
            n);
        CHECK_CUDA_LAUNCH_ERROR();
    }

    // Backward pass:
    // add scanned block offsets from higher levels down to level 0,
    // then apply final level-0 offsets to the output array.
    if(!level_ptrs.empty())
    {
        for(size_t level = level_ptrs.size(); level-- > 1;)
        {
            int n = level_sizes[level - 1];
            exclusive_scan_kernel_part2<blocksize><<<((n - 1) / blocksize + 1), blocksize>>>(
                level_ptrs[level],
                level_ptrs[level - 1],
                n);
            CHECK_CUDA_LAUNCH_ERROR();
        }

        exclusive_scan_kernel_part2<blocksize><<<((size - 1) / blocksize + 1), blocksize>>>(
            level_ptrs[0],
            array,
            size);
        CHECK_CUDA_LAUNCH_ERROR();
    }

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
