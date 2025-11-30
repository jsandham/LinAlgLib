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

#ifndef EXTRACT_DIAGONAL_KERNELS_H
#define EXTRACT_DIAGONAL_KERNELS_H

#include "common.cuh"

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE, typename T>
__global__ void extract_diagonal_kernel(int m,
                                        int n,
                                        int nnz,
                                        const int* __restrict__ csr_row_ptr,
                                        const int* __restrict__ csr_col_ind,
                                        const T* __restrict__ csr_val,
                                        T* __restrict__ diag)
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

        for(int j = row_start + lid; j < row_end; j += WARPSIZE)
        {
            int col = csr_col_ind[j];
            T   val = csr_val[j];

            if(col == row)
            {
                diag[row] = val;
                break;
            }
        }
    }
}

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE>
__global__ void extract_lower_triangular_nnz_kernel(int m_A,
                                                    int n_A,
                                                    int nnz_A,
                                                    const int* __restrict__ csr_row_ptr_A,
                                                    const int* __restrict__ csr_col_ind_A,
                                                    int* __restrict__ csr_row_ptr_L)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    int lid = tid & WARPSIZE - 1;
    //int wid = tid / WARPSIZE;

    for(int row = gid / WARPSIZE; row < m_A; row += (BLOCKSIZE / WARPSIZE) * gridDim.x)
    {
        int row_start = csr_row_ptr_A[row];
        int row_end   = csr_row_ptr_A[row + 1];

        int count = 0;

        for(int j = row_start + lid; j < row_end; j += WARPSIZE)
        {
            int col = csr_col_ind_A[j];

            if(col <= row)
            {
                count++;
            }
        }

        // Reduce within warp
        warp_reduction_sum<WARPSIZE>(&count, lid);

        // Write result
        if(lid == 0)
        {
            csr_row_ptr_L[row] = count;
        }
    }
}

template <uint32_t BLOCKSIZE, uint32_t WARPSIZE>
__global__ void extract_upper_triangular_nnz_kernel(int m_A,
                                                    int n_A,
                                                    int nnz_A,
                                                    const int* __restrict__ csr_row_ptr_A,
                                                    const int* __restrict__ csr_col_ind_A,
                                                    int* __restrict__ csr_row_ptr_U)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    int lid = tid & WARPSIZE - 1;
    //int wid = tid / WARPSIZE;

    for(int row = gid / WARPSIZE; row < m_A; row += (BLOCKSIZE / WARPSIZE) * gridDim.x)
    {
        int row_start = csr_row_ptr_A[row];
        int row_end   = csr_row_ptr_A[row + 1];

        int count = 0;

        for(int j = row_start + lid; j < row_end; j += WARPSIZE)
        {
            int col = csr_col_ind_A[j];

            if(col >= row)
            {
                count++;
            }
        }

        // Reduce within warp
        warp_reduction_sum<WARPSIZE>(&count, lid);

        // Write result
        if(lid == 0)
        {
            csr_row_ptr_U[row] = count;
        }
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void extract_lower_triangular_kernel(int m_A,
                                                int n_A,
                                                int nnz_A,
                                                const int* __restrict__ csr_row_ptr_A,
                                                const int* __restrict__ csr_col_ind_A,
                                                const T* __restrict__ csr_val_A,
                                                int m_L,
                                                int n_L,
                                                int nnz_L,
                                                const int* __restrict__ csr_row_ptr_L,
                                                int* __restrict__ csr_col_ind_L,
                                                T* __restrict__ csr_val_L)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    for(int row = gid; row < m_A; row += BLOCKSIZE * gridDim.x)
    {
        const int row_start_A = csr_row_ptr_A[row];
        const int row_end_A   = csr_row_ptr_A[row + 1];

        const int row_start_L = csr_row_ptr_L[row];

        int index = row_start_L;
        for(int j = row_start_A; j < row_end_A; j++)
        {
            int col_A = csr_col_ind_A[j];

            if(col_A <= row)
            {
                csr_col_ind_L[index] = col_A;
                csr_val_L[index]     = csr_val_A[j];
                index++;
            }
        }
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void extract_upper_triangular_kernel(int m_A,
                                                int n_A,
                                                int nnz_A,
                                                const int* __restrict__ csr_row_ptr_A,
                                                const int* __restrict__ csr_col_ind_A,
                                                const T* __restrict__ csr_val_A,
                                                int m_U,
                                                int n_U,
                                                int nnz_U,
                                                const int* __restrict__ csr_row_ptr_U,
                                                int* __restrict__ csr_col_ind_U,
                                                T* __restrict__ csr_val_U)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    for(int row = gid; row < m_A; row += BLOCKSIZE * gridDim.x)
    {
        const int row_start_A = csr_row_ptr_A[row];
        const int row_end_A   = csr_row_ptr_A[row + 1];

        const int row_start_U = csr_row_ptr_U[row];

        int index = row_start_U;
        for(int j = row_start_A; j < row_end_A; j++)
        {
            int col_A = csr_col_ind_A[j];

            if(col_A >= row)
            {
                csr_col_ind_U[index] = col_A;
                csr_val_U[index]     = csr_val_A[j];
                index++;
            }
        }
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void scale_diagonal_kernel(int m,
                                      const int* __restrict__ csr_row_ptr,
                                      const int* __restrict__ csr_col_ind,
                                      T* __restrict__ csr_val,
                                      T scalar)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    for(int row = gid; row < m; row += BLOCKSIZE * gridDim.x)
    {
        const int row_start = csr_row_ptr[row];
        const int row_end   = csr_row_ptr[row + 1];

        for(int j = row_start; j < row_end; j++)
        {
            const int col = csr_col_ind[j];

            if(col == row)
            {
                csr_val[j] *= scalar;
                break;
            }
        }
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void scale_by_inverse_diagonal_kernel(int m,
                                                 const int* __restrict__ csr_row_ptr,
                                                 const int* __restrict__ csr_col_ind,
                                                 T* __restrict__ csr_val,
                                                 const T* __restrict__ diag)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    for(int row = gid; row < m; row += BLOCKSIZE * gridDim.x)
    {
        const int row_start = csr_row_ptr[row];
        const int row_end   = csr_row_ptr[row + 1];

        for(int j = row_start; j < row_end; j++)
        {
            csr_val[j] *= static_cast<T>(1) / diag[csr_col_ind[j]];
        }
    }
}

#endif