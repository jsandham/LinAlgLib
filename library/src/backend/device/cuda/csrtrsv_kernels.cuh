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

#ifndef CSRTRSV_KERNELS_H
#define CSRTRSV_KERNELS_H

#include <cuda/atomic>

#include "common.cuh"
#include "linalg_enums.h"

// 1 0 0 0
// 1 1 0 0
// 1 1 1 0
// 1 1 1 1
//
// diag_ind = [0, 2, 5, 9]
// done_array = [1, 2, 3, 4]

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, typename T>
__global__ void csrtrsv_analysis_kernel(int                     m,
                                        linalg::triangular_type tri_type,
                                        const int* __restrict__ csr_row_ptr,
                                        const int* __restrict__ csr_col_ind,
                                        const T* __restrict__ csr_val,
                                        int* __restrict__ csr_diag_ind,
                                        int* __restrict__ done_array)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int lid = tid & (WARPSIZE - 1);
    const int wid = tid / WARPSIZE;

    if(bid * (BLOCKSIZE / WARPSIZE) + wid >= m)
    {
        return;
    }

    const int row = tri_type == linalg::triangular_type::lower
                        ? bid * (BLOCKSIZE / WARPSIZE) + wid
                        : (m - 1) - (bid * (BLOCKSIZE / WARPSIZE) + wid);
    assert(row >= 0 && row < m);

    const int start = csr_row_ptr[row];
    const int end   = csr_row_ptr[row + 1];

    int max_depth = 0;
    for(int jj = start + lid; jj < end; jj += WARPSIZE)
    {
        const int col = csr_col_ind[jj];
        assert(col >= 0 && col < m);

        if(tri_type == linalg::triangular_type::lower)
        {
            if(col > row)
            {
                continue;
            }
        }
        else
        {
            if(col < row)
            {
                continue;
            }
        }

        if(col == row)
        {
            csr_diag_ind[row] = jj;
            continue;
        }

        if(tri_type == linalg::triangular_type::lower ? col < row : col > row)
        {
            // 1. Create a temporary atomic reference to the element.
            // This provides an atomic view of the non-atomic data.
            cuda::atomic_ref<int, cuda::thread_scope_device> atomic_view(done_array[col]);

            // 2. Perform the explicit atomic load operation.
            // This is the functional equivalent to __hip_atomic_load with Acquire/Device scope.
            int loaded_value = atomic_view.load(cuda::memory_order_acquire);
            while(loaded_value == 0)
            {
                loaded_value = atomic_view.load(cuda::memory_order_acquire);
            }

            max_depth = max(max_depth, loaded_value);
        }
    }

    warp_reduction_max<WARPSIZE>(&max_depth, lid);

    if(lid == 0)
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> atomic_view(done_array[row]);

        // mark this row as done
        atomic_view.store(max_depth + 1, cuda::memory_order_release);
    }
}

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, typename T>
__global__ void csrtrsv_solve_kernel(int                     m,
                                     linalg::triangular_type tri_type,
                                     linalg::diagonal_type   diag_type,
                                     T                       alpha,
                                     const int* __restrict__ csr_row_ptr,
                                     const int* __restrict__ csr_col_ind,
                                     const T* __restrict__ csr_val,
                                     const int* __restrict__ csr_diag_ind,
                                     const T* __restrict__ b,
                                     T* __restrict__ x,
                                     int* __restrict__ done_array,
                                     int* __restrict__ row_perm)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int lid = tid & (WARPSIZE - 1);
    const int wid = tid / WARPSIZE;

    if(bid * (BLOCKSIZE / WARPSIZE) + wid >= m)
    {
        return;
    }

    const int row = row_perm[bid * (BLOCKSIZE / WARPSIZE) + wid];
    assert(row >= 0 && row < m);

    const int start = csr_row_ptr[row];
    const int end   = csr_row_ptr[row + 1];

    const T diag_val = (diag_type == linalg::diagonal_type::non_unit) ? csr_val[csr_diag_ind[row]]
                                                                      : static_cast<T>(1);

    T sum = static_cast<T>(0);
    for(int jj = start + lid; jj < end; jj += WARPSIZE)
    {
        const int col = csr_col_ind[jj];
        const T   val = csr_val[jj];

        assert(col >= 0 && col < m);

        if(tri_type == linalg::triangular_type::lower)
        {
            if(col >= row)
            {
                continue;
            }
        }
        else
        {
            if(col <= row)
            {
                continue;
            }
        }

        if(tri_type == linalg::triangular_type::lower ? col < row : col > row)
        {
            // 1. Create a temporary atomic reference to the element.
            // This provides an atomic view of the non-atomic data.
            cuda::atomic_ref<int, cuda::thread_scope_device> atomic_view(done_array[col]);

            // 2. Perform the explicit atomic load operation.
            // This is the functional equivalent to __hip_atomic_load with Acquire/Device scope.
            int loaded_value = atomic_view.load(cuda::memory_order_acquire);
            while(loaded_value == 0)
            {
                loaded_value = atomic_view.load(cuda::memory_order_acquire);
            }

            sum = std::fma(val, x[col], sum);
        }
    }

    warp_reduction_sum<WARPSIZE>(&sum, lid);

    if(lid == 0)
    {
        x[row] = (b[row] / alpha - sum) / diag_val;

        __threadfence();

        cuda::atomic_ref<int, cuda::thread_scope_device> atomic_view(done_array[row]);

        // mark this row as done
        atomic_view.store(1, cuda::memory_order_release);
    }
}

#endif // CSRTRSV_KERNELS_H