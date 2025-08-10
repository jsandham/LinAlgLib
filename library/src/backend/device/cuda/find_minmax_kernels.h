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

#include "common.h"

#ifndef FIND_MINMAX_KERNELS_H
#define FIND_MINMAX_KERNELS_H

template <uint32_t BLOCKSIZE, typename T>
__global__ void find_max_kernel_part1(int size, const T* __restrict__ x, T* __restrict__ workspace)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    __shared__ T shared[BLOCKSIZE];

    T val = static_cast<T>(0);

    int index = gid;
    while(index < size)
    {
        val = linalg::max(linalg::abs(val), linalg::abs(x[index]));

        index += BLOCKSIZE * gridDim.x;
    }

    shared[tid] = val;

    __syncthreads();

    block_reduction_max<BLOCKSIZE>(shared, tid);

    if(tid == 0)
    {
        workspace[bid] = shared[0];
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void find_max_kernel_part2(T* __restrict__ workspace)
{
    int tid = threadIdx.x;

    __shared__ T shared[BLOCKSIZE];

    shared[tid] = workspace[tid];

    __syncthreads();

    block_reduction_max<BLOCKSIZE>(shared, tid);

    if(tid == 0)
    {
        workspace[0] = shared[0];
    }
}

#endif