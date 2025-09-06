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

#ifndef PRIMITIVE_KERNELS_H
#define PRIMITIVE_KERNELS_H

#include "common.cuh"

template <uint32_t BLOCKSIZE, typename T>
__global__ void exclusive_scan_kernel_part1(const T* d_in, T* d_out, T* workspace, int n)
{
    __shared__ T shared[BLOCKSIZE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    if(gid < n)
    {
        shared[tid] = d_in[gid];
    }
    else
    {
        shared[tid] = 0;
    }

    __syncthreads();

    // Up-Sweep Phase
    for(uint32_t stride = 1; stride < BLOCKSIZE; stride *= 2)
    {
        // Wait for all threads to finish loading data for the current stride
        __syncthreads();
        int index = (tid + 1) * 2 * stride - 1;
        if(index < BLOCKSIZE)
        {
            // Add the element from the left child of the tree node
            shared[index] += shared[index - stride];
        }
    }

    // 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  1 1 1 1
    // 1 2 1 2 1 2 1 2 1 2 1 2 1 2 1 2  1 2 1 2
    // 1 2 1 4 1 2 1 4 1 2 1 4 1 2 1 4  1 2 1 4
    // 1 2 1 4 1 2 1 8 1 2 1 4 1 2 1 8  1 2 1 4
    // 1 2 1 4 1 2 1 8 1 2 1 4 1 2 1 16 1 2 1 4

    __syncthreads();
    if(tid == 0)
    {
        if(workspace != nullptr)
        {
            workspace[bid] = shared[BLOCKSIZE - 1];
        }
        shared[BLOCKSIZE - 1] = 0;
    }

    // Down-Sweep Phase
    for(uint32_t stride = BLOCKSIZE / 2; stride > 0; stride /= 2)
    {
        __syncthreads();
        int index = (tid + 1) * 2 * stride - 1;
        if(index < BLOCKSIZE)
        {
            // Store the current value to a temporary variable
            T temp = shared[index];
            // Update the current value with the value from the left child
            shared[index] += shared[index - stride];
            // The value from the right child gets updated with the temporary
            shared[index - stride] = temp;
        }
    }

    __syncthreads();

    if(gid < n)
    {
        d_out[gid] = shared[tid];
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void exclusive_scan_kernel_part2(const T* workspace, T* x, int n)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    if(gid < n)
    {
        x[gid] += workspace[bid];
    }
}

#endif