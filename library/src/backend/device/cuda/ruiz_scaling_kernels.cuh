//********************************************************************************
//
// MIT License
//
// Copyright(c) 2026 James Sandham
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

#ifndef RUIZ_SCALING_KERNELS_KERNELS_H
#define RUIZ_SCALING_KERNELS_KERNELS_H

#include "common.cuh"

template <uint32_t BLOCKSIZE, typename T>
__global__ void set_D_to_one_kernel(int m, T* __restrict__ DR)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    if(gid < m)
    {
        DR[gid] = static_cast<T>(1);
    }
}

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, typename T>
__global__ void fill_DR_kernel(int m,
                               const int* __restrict__ csr_row_ptr,
                               const T* __restrict__ csr_val,
                               T* __restrict__ DR)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    int lid = tid & WARPSIZE - 1;
    int wid = tid / WARPSIZE;

    int row = gid / WARPSIZE;

    if(row < m)
    {
        const int row_start = csr_row_ptr[row];
        const int row_end   = csr_row_ptr[row + 1];

        T max = static_cast<T>(0);

        for(int j = row_start + lid; j < row_end; j += WARPSIZE)
        {
            const T val = csr_val[j];

            max = linalg::max(max, linalg::abs(val));
        }

        warp_reduction_max<WARPSIZE>(&max);

        if(lid == 0)
        {
            DR[row] = max;
        }
    }
}




template <uint32_t BLOCKSIZE, typename T>
__global__ void compute_max_divergence_part1(int m, T eps, T* __restrict__ DR, T* workspace)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    __shared__ T shared[BLOCKSIZE];

    int index = gid;

    T max_divergence = static_cast<T>(0);

    while(index < m)
    {
        max_divergence = linalg::max(max_divergence, linalg::abs(static_cast<T>(1) - DR[index]));

        DR[index] = (DR[index] > eps) ? std::sqrt(DR[index]) : static_cast<T>(1);

        index += BLOCKSIZE * gridDim.x;
    }

    shared[tid] = max_divergence;
    __syncthreads();

    block_reduction_max<BLOCKSIZE>(&shared[0], tid);

    if(tid == 0)
    {
        workspace[bid] = shared[0];
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void compute_max_divergence_part2(int m, const T* __restrict__ workspace, T* max_divergence)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    __shared__ T shared[BLOCKSIZE];

    shared[tid] = workspace[tid];
    __syncthreads();

    block_reduction_max<BLOCKSIZE>(&shared[0], tid);

    if(tid == 0)
    {
        *max_divergence = shared[0];
    }
}





template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, typename T>
__global__ void update_A(int m,
                         const int* __restrict__ csr_row_ptr,
                         const int* __restrict__ csr_col_ind,
                         T* __restrict__ csr_val,
                         T* __restrict__ DR)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    int lid = tid & WARPSIZE - 1;
    int wid = tid / WARPSIZE;

    int row = gid / WARPSIZE;

    if(row < m)
    {
        const int row_start = csr_row_ptr[row];
        const int row_end   = csr_row_ptr[row + 1];

        const T DR_row_val = DR[row];

        for(int j = row_start + lid; j < row_end; j += WARPSIZE)
        {
            csr_val[j] /= (DR_row_val * DR[csr_col_ind[j]]);
        }
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void update_D(int m, const T* __restrict__ DR, T* __restrict__ D)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    if(gid < m)
    {
        D[gid] = D[gid] / DR[gid];
    }
}

#endif
