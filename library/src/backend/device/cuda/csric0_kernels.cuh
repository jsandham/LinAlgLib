//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025-2026 James Sandham
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

#ifndef CSRIC0_KERNELS_H
#define CSRIC0_KERNELS_H

#include <cuda/atomic>

#include "common.cuh"

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, uint32_t HASHSIZE, typename T>
__global__ void csric0_solve_kernel(int m,
                                    const int* __restrict__ csr_row_ptr,
                                    const int* __restrict__ csr_col_ind,
                                    T* __restrict__ csr_val,
                                    const int* __restrict__ csr_diag_ind,
                                    int* __restrict__ done_array,
                                    const int* __restrict__ row_perm)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int lid = tid & (WARPSIZE - 1);
    const int wid = tid / WARPSIZE;

    __shared__ int shared_key[(BLOCKSIZE / WARPSIZE) * HASHSIZE];
    __shared__ int shared_val[(BLOCKSIZE / WARPSIZE) * HASHSIZE];

    for(int i = tid; i < (BLOCKSIZE / WARPSIZE) * HASHSIZE; i += BLOCKSIZE)
    {
        shared_key[i] = -1;
        shared_val[i] = 0;
    }

    __syncthreads();

    if(bid * (BLOCKSIZE / WARPSIZE) + wid >= m)
    {
        return;
    }

    const int row = row_perm[bid * (BLOCKSIZE / WARPSIZE) + wid];
    assert(row >= 0 && row < m);

    const int start    = csr_row_ptr[row];
    const int diag_end = csr_diag_ind[row];

    assert(start >= 0);
    assert(diag_end >= 0);

    for(int i = start + lid; i < diag_end; i += WARPSIZE)
    {
        const int col = csr_col_ind[i];

        atomic_insert_key_value2<HASHSIZE>(&shared_key[(BLOCKSIZE / WARPSIZE) * wid],
                                           &shared_val[(BLOCKSIZE / WARPSIZE) * wid],
                                           col,
                                           i,
                                           -1);
    }

    __syncthreads();

    T diag_sum = static_cast<T>(0);
    for(int i = start; i < diag_end; i++)
    {
        const int col_i = csr_col_ind[i];
        assert(col_i >= 0 && col_i < row);

        // 1. Create a temporary atomic reference to the element.
        // This provides an atomic view of the non-atomic data.
        cuda::atomic_ref<int, cuda::thread_scope_device> atomic_view(done_array[col_i]);

        // 2. Perform the explicit atomic load operation.
        // This is the functional equivalent to __hip_atomic_load with Acquire/Device scope.
        int loaded_value = atomic_view.load(cuda::memory_order_acquire);
        while(loaded_value == 0)
        {
            loaded_value = atomic_view.load(cuda::memory_order_acquire);
        }

        const int local_start    = csr_row_ptr[col_i];
        const int local_diag_end = csr_diag_ind[col_i];

        assert(local_start >= 0);
        assert(local_diag_end >= 0);
        assert(csr_col_ind[local_diag_end] == col_i);

        T sum = static_cast<T>(0);
        for(int j = local_start + lid; j < local_diag_end; j += WARPSIZE)
        {
            const int col_j = csr_col_ind[j];
            assert(col_j >= 0 && col_j < col_i);

            int pos = atomic_find_val<HASHSIZE>(&shared_key[(BLOCKSIZE / WARPSIZE) * wid],
                                                &shared_val[(BLOCKSIZE / WARPSIZE) * wid],
                                                col_j,
                                                -1);

            if(pos != -1)
            {
                sum = std::fma(csr_val[j], csr_val[pos], sum);
            }
        }

        warp_reduction_sum<WARPSIZE>(&sum, lid);

        if(lid == 0)
        {
            T diag_val = csr_val[local_diag_end];
            if(diag_val == static_cast<T>(0))
            {
                // Handle zero diagonal element
                diag_val = static_cast<T>(1);
            }
            T val = (csr_val[i] - sum) / diag_val;

            // Accumulate squared L[i,k] values for diagonal computation
            diag_sum   = std::fma(val, val, diag_sum);
            csr_val[i] = val;
        }
    }

    if(lid == 0)
    {
        // Diagonal: L[i,i] = sqrt(A[i,i] - sum(L[i,k]^2))
        csr_val[diag_end] = sqrt(abs(csr_val[diag_end] - diag_sum));

        __threadfence();

        cuda::atomic_ref<int, cuda::thread_scope_device> atomic_view(done_array[row]);

        // mark this row as done
        atomic_view.store(1, cuda::memory_order_release);
    }
}

#endif // CSRIC0_KERNELS_H
