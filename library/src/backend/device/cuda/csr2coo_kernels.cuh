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

#ifndef CSR2COO_KERNELS_H
#define CSR2COO_KERNELS_H

#include "common.cuh"

// 1 2 0 0
// 0 3 0 4
// 5 6 0 7
// 0 8 9 0

// ptr = [0, 2, 4, 7, 9]

static __device__ int get_row_index(int m, const int* csr_row_ptr, int gid)
{
    int left         = 0;
    int right        = m;
    int result_index = m;

    while(left < right)
    {
        // Calculate mid-point
        int mid = left + (right - left) / 2;

        if(csr_row_ptr[mid] > gid)
        {
            result_index = mid;
            right        = mid;
        }
        else
        {
            left = mid + 1;
        }
    }
    return result_index - 1;
}

template <uint32_t BLOCKSIZE>
__global__ void csr2coo_kernel(
    int m, int n, int nnz, const int* __restrict__ csr_row_ptr, int* __restrict__ coo_row_ind)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    if(gid < nnz)
    {
        coo_row_ind[gid] = get_row_index(m, csr_row_ptr, gid);
    }
}

// 1 2 0 0
// 0 3 0 4
// 5 6 0 7
// 0 8 9 0

// ind = [0, 0, 1,    1,    2,    2, 2, 3, 3]

// 3 + (4 - 3) / 2 = 3

static __device__ int get_row_ptr(int nnz, const int* coo_row_ind, int row)
{
    int left  = 0;
    int right = nnz;

    while(left < right)
    {
        // Calculate mid-point
        int mid = left + (right - left) / 2;

        if(coo_row_ind[mid] > row)
        {
            right = mid;
        }
        else
        {
            left = mid + 1;
        }
    }
    return left;
}

template <uint32_t BLOCKSIZE>
__global__ void coo2csr_kernel(
    int m, int n, int nnz, const int* __restrict__ coo_row_ind, int* __restrict__ csr_row_ptr)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int row = tid + BLOCKSIZE * bid;

    if(row < m)
    {
        if(row == 0)
        {
            csr_row_ptr[0] = 0;
        }

        csr_row_ptr[row + 1] = get_row_ptr(nnz, coo_row_ind, row);
    }
}

template <uint32_t BLOCKSIZE>
__global__ void fill_identity_permuation_kernel(int length, int* __restrict__ perm)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    if(gid < length)
    {
        perm[gid] = gid;
    }
}

template <uint32_t BLOCKSIZE>
__global__ void csr2csc_permute_colval_kernel(int m,
                                              int n,
                                              int nnz,
                                              const int* __restrict__ coo_row_ind,
                                              const double* __restrict__ csr_val,
                                              const int* __restrict__ perm,
                                              int* __restrict__ csc_row_ind,
                                              double* __restrict__ csc_val)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    if(gid < nnz)
    {
        int p            = perm[gid];
        csc_row_ind[gid] = coo_row_ind[p];
        csc_val[gid]     = csr_val[p];
    }
}

#endif