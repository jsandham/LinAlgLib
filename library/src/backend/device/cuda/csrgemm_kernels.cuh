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

#ifndef CSRGEMM_KERNELS_H
#define CSRGEMM_KERNELS_H

#include "common.cuh"

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, typename T>
__global__ void csrgemm_count_products_kernel(int     m,
                                              const T alpha,
                                              const int* __restrict__ csr_row_ptr_A,
                                              const int* __restrict__ csr_col_ind_A,
                                              const int* __restrict__ csr_row_ptr_B,
                                              const T beta,
                                              const int* __restrict__ csr_row_ptr_D,
                                              int* __restrict__ products_per_row_C)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    int lid = tid & WARPSIZE - 1;
    int wid = tid / WARPSIZE;

    int row = gid / WARPSIZE;

    if(row >= m)
    {
        return;
    }

    int product_count = 0;

    if(alpha != static_cast<T>(0))
    {
        int start_A = csr_row_ptr_A[row];
        int end_A   = csr_row_ptr_A[row + 1];

        for(int i = start_A + lid; i < end_A; i += WARPSIZE)
        {
            int col_A = csr_col_ind_A[i];

            int start_B = csr_row_ptr_B[col_A];
            int end_B   = csr_row_ptr_B[col_A + 1];

            product_count += end_B - start_B;
        }

        warp_reduction_sum<WARPSIZE>(&product_count, lid);
    }

    if(beta != static_cast<T>(0))
    {
        if(lid == 0)
        {
            int start_D = csr_row_ptr_D[row];
            int end_D   = csr_row_ptr_D[row + 1];

            product_count += end_D - start_D;
        }
    }

    if(lid == 0)
    {
        products_per_row_C[row] = product_count;
    }
}

template <uint32_t BLOCKSIZE>
__global__ void compute_rows_bin_number_kernel(int m, int* __restrict__ csr_row_ptr_C)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    if(gid >= m)
    {
        return;
    }

    if(csr_row_ptr_C[gid] > 0)
    {
        double log_2 = std::log2(static_cast<double>(csr_row_ptr_C[gid]));

        csr_row_ptr_C[gid] = max(0, (int)std::ceil(log_2) - (int)std::log2(32.0));
    }

    if(gid == m - 1)
    {
        csr_row_ptr_C[m] = -1;
    }
}

template <uint32_t BLOCKSIZE>
__global__ void
    fill_bin_offsets_kernel(int m, const int* __restrict__ bins_C, int* __restrict__ bin_offsets_C)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    if(gid >= m)
    {
        return;
    }

    int current = bins_C[gid];
    int next    = bins_C[gid + 1];

    if(current != next)
    {
        bin_offsets_C[current + 1] = gid + 1;
    }
}

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, uint32_t HASHSIZE, typename T>
__global__ void csrgemm_nnz_per_row_kernel(int        bin_count,
                                           const int* perm,
                                           const int* bin_offset,
                                           const T    alpha,
                                           const int* __restrict__ csr_row_ptr_A,
                                           const int* __restrict__ csr_col_ind_A,
                                           const int* __restrict__ csr_row_ptr_B,
                                           const int* __restrict__ csr_col_ind_B,
                                           const T beta,
                                           const int* __restrict__ csr_row_ptr_D,
                                           const int* __restrict__ csr_col_ind_D,
                                           int* __restrict__ nnz_per_row)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    int lid = tid & WARPSIZE - 1;
    int wid = tid / WARPSIZE;

    int row = gid / WARPSIZE;

    __shared__ int shared[HASHSIZE * (BLOCKSIZE / WARPSIZE)];

    for(int i = tid; i < HASHSIZE * (BLOCKSIZE / WARPSIZE); i += BLOCKSIZE)
    {
        shared[i] = -1;
    }
    __syncthreads();

    if(row >= bin_count)
    {
        return;
    }

    row = perm[row + *bin_offset];

    int row_nnz = 0;

    if(alpha != static_cast<T>(0))
    {
        const int start_A = csr_row_ptr_A[row];
        const int end_A   = csr_row_ptr_A[row + 1];

        for(int i = start_A + lid; i < end_A; i += WARPSIZE)
        {
            const int col_A = csr_col_ind_A[i];

            const int start_B = csr_row_ptr_B[col_A];
            const int end_B   = csr_row_ptr_B[col_A + 1];

            for(int j = start_B; j < end_B; j++)
            {
                const int col_B = csr_col_ind_B[j];

                row_nnz += atomic_insert_key<HASHSIZE>(&shared[HASHSIZE * wid], col_B, -1);
            }
        }
    }

    if(beta != static_cast<T>(0))
    {
        const int start_D = csr_row_ptr_D[row];
        const int end_D   = csr_row_ptr_D[row + 1];

        for(int i = start_D + lid; i < end_D; i += WARPSIZE)
        {
            const int col_D = csr_col_ind_D[i];

            row_nnz += atomic_insert_key<HASHSIZE>(&shared[HASHSIZE * wid], col_D, -1);
        }
    }

    warp_reduction_sum<WARPSIZE>(&row_nnz, lid);

    if(lid == 0)
    {
        nnz_per_row[row] = row_nnz;
    }
}

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, uint32_t HASHSIZE, typename T>
__global__ void csrgemm_fill_kernel(int        bin_count,
                                    const int* perm,
                                    const int* bin_offset,
                                    const T    alpha,
                                    const int* __restrict__ csr_row_ptr_A,
                                    const int* __restrict__ csr_col_ind_A,
                                    const T* __restrict__ csr_val_A,
                                    const int* __restrict__ csr_row_ptr_B,
                                    const int* __restrict__ csr_col_ind_B,
                                    const T* __restrict__ csr_val_B,
                                    const T beta,
                                    const int* __restrict__ csr_row_ptr_D,
                                    const int* __restrict__ csr_col_ind_D,
                                    const T* __restrict__ csr_val_D,
                                    const int* __restrict__ csr_row_ptr_C,
                                    int* __restrict__ csr_col_ind_C,
                                    T* __restrict__ csr_val_C)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    int lid = tid & WARPSIZE - 1;
    int wid = tid / WARPSIZE;

    int row = gid / WARPSIZE;

    __shared__ int shared[HASHSIZE * (BLOCKSIZE / WARPSIZE)];
    __shared__ T   shared_val[HASHSIZE * (BLOCKSIZE / WARPSIZE)];

    for(int i = tid; i < HASHSIZE * (BLOCKSIZE / WARPSIZE); i += BLOCKSIZE)
    {
        shared[i]     = -1;
        shared_val[i] = static_cast<T>(0);
    }
    __syncthreads();

    if(row >= bin_count)
    {
        return;
    }

    row = perm[row + *bin_offset];

    if(alpha != static_cast<T>(0))
    {
        const int start_A = csr_row_ptr_A[row];
        const int end_A   = csr_row_ptr_A[row + 1];

        for(int i = start_A + lid; i < end_A; i += WARPSIZE)
        {
            const int col_A = csr_col_ind_A[i];
            const T   val_A = csr_val_A[i];

            const int start_B = csr_row_ptr_B[col_A];
            const int end_B   = csr_row_ptr_B[col_A + 1];

            for(int j = start_B; j < end_B; j++)
            {
                const int col_B = csr_col_ind_B[j];
                const T   val_B = csr_val_B[j];

                atomic_insert_key_value<HASHSIZE>(
                    &shared[HASHSIZE * wid], &shared_val[HASHSIZE * wid], col_B, val_A * val_B, -1);
            }
        }
    }

    if(beta != static_cast<T>(0))
    {
        const int start_D = csr_row_ptr_D[row];
        const int end_D   = csr_row_ptr_D[row + 1];

        for(int i = start_D + lid; i < end_D; i += WARPSIZE)
        {
            const int col_D = csr_col_ind_D[i];
            const T   val_D = csr_val_D[i];

            atomic_insert_key_value<HASHSIZE>(
                &shared[HASHSIZE * wid], &shared_val[HASHSIZE * wid], col_D, val_D, -1);
        }
    }

    __syncthreads(); //warp sync

    if(lid == 0)
    {
        const int start_C = csr_row_ptr_C[row];

        int j = start_C;
        for(int i = 0; i < HASHSIZE; i++)
        {
            if(shared[HASHSIZE * wid + i] != -1)
            {
                csr_col_ind_C[j] = shared[HASHSIZE * wid + i];
                csr_val_C[j]     = shared_val[HASHSIZE * wid + i];
                j++;
            }
        }
    }
}

#endif