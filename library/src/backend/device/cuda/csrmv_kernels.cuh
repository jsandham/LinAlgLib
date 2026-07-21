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

#ifndef CSRMV_KERNELS_H
#define CSRMV_KERNELS_H

#include "common.cuh"

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, typename T>
__global__ void csrmv_vector_kernel(int     m,
                                    int     n,
                                    int     nnz,
                                    const T alpha,
                                    const int* __restrict__ csr_row_ptr,
                                    const int* __restrict__ csr_col_ind,
                                    const T* __restrict__ csr_val,
                                    const T* __restrict__ x,
                                    const T beta,
                                    T* __restrict__ y)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    int lid = tid & WARPSIZE - 1;
    //int wid = tid / WARPSIZE;

    for(int row = gid / WARPSIZE; row < m; row += (BLOCKSIZE / WARPSIZE) * gridDim.x)
    {
        int row_start = csr_row_ptr[row];
        int row_end   = csr_row_ptr[row + 1];

        T sum = static_cast<T>(0);
        for(int j = row_start + lid; j < row_end; j += WARPSIZE)
        {
            int col = csr_col_ind[j];
            T   val = csr_val[j];

            sum = std::fma(x[col], val, sum);
        }

        warp_reduction_sum<WARPSIZE>(&sum);

        if(lid == 0)
        {
            if(beta == static_cast<T>(0))
            {
                y[row] = alpha * sum;
            }
            else
            {
                y[row] = std::fma(alpha, sum, beta * y[row]);
            }
        }
    }
}









__device__ inline int csr_row_from_index(const int* __restrict__ row_ptr,
                                            int row_ptr_size,
                                            int nnz_index)
{
    // row_ptr has size m+1 and is non-decreasing
    // find r such that row_ptr[r] <= nnz_index < row_ptr[r + 1]
    int lo = 0;
    int hi = row_ptr_size - 1;

    while(lo < hi)
    {
        int mid = lo + ((hi - lo) >> 1);
        if(row_ptr[mid] <= nnz_index)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }

    return lo - 1;
}

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, uint32_t NNZ_PER_THREAD,typename T>
__global__ void csrmv_stream_kernel(int     m,
                                    int     n,
                                    int     nnz,
                                    const T alpha,
                                    const int* __restrict__ csr_row_ptr,
                                    const int* __restrict__ csr_col_ind,
                                    const T* __restrict__ csr_val,
                                    const T* __restrict__ x,
                                    const T beta,
                                    T* __restrict__ y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int lid = tid & WARPSIZE - 1;
    const int wid = tid / WARPSIZE;

    const int start = NNZ_PER_THREAD * (BLOCKSIZE * bid + WARPSIZE * wid);

    int prev_row = -1;

    T sum = static_cast<T>(0);

    for(int i = 0; i < NNZ_PER_THREAD; ++i)
    {
        const int index = start + i * WARPSIZE + lid;

        const int row = (index < nnz) ? csr_row_from_index(csr_row_ptr, m + 1, index) : -1;
        const int col = csr_col_ind[index];
        const T   val = csr_val[index];

        const int left_row = __shfl_sync(0xffffffff, row, lid - 1);
        const int right_row = __shfl_sync(0xffffffff, row, lid + 1);

        const bool predicate = (row != -1 && row == left_row) || (row != -1 && row == right_row);

        if(__any_sync(0xffffffff, predicate))
        {
            // write out old values
            warp_reduction_sum<WARPSIZE>(&sum);

            if(lid == 0)
            {
                if(prev_row >= 0)
                {
                    atomicAdd(&y[prev_row], alpha * sum);
                }
            }

            sum = x[col] * val;

            // segmented reduction for current values
            sum = warp_segmented_reduction_sum(row, sum);

            if(lid < WARPSIZE - 1)
            {
                if(row != right_row && row != -1)
                {
                    atomicAdd(&y[row], alpha * sum);
                }
            }
            else
            {
                if(row != -1)
                {
                    atomicAdd(&y[row], alpha * sum);
                }
            }
        }
        else
        {
            sum = std::fma(x[col], val, sum);
        }

        prev_row = row;
    }
}





#endif
