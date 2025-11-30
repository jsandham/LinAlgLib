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
#ifndef PRECONDITIONER_KERNELS_H
#define PRECONDITIONER_KERNELS_H

#include "common.cuh"

template <uint32_t BLOCKSIZE, typename T>
__global__ void jacobi_solve_kernel(int size,
                                    const T* __restrict__ rhs,
                                    const T* __restrict__ diag,
                                    T* __restrict__ x)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    if(gid < size)
    {
        x[gid] = rhs[gid] / diag[gid];
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void ssor_fill_lower_precond_kernel(int        m_A,
                                               int        n_A,
                                               int        nnz_A,
                                               T          omega,
                                               const int* csr_row_ptr_A,
                                               const int* csr_col_ind_A,
                                               const T*   csr_val_A,
                                               const T*   diag,
                                               const int* csr_row_ptr_L,
                                               int*       csr_col_ind_L,
                                               T*         csr_val_L)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    double beta = 1.0 / omega;

    // L = (beta * D - E) * D^-1, beta = 1 / omega
    // L = beta * I - E * D^-1
    for(int i = gid; i < m_A; i += BLOCKSIZE * gridDim.x)
    {
        const int row_start_A = csr_row_ptr_A[i];
        const int row_end_A   = csr_row_ptr_A[i + 1];

        const int row_start_L = csr_row_ptr_L[i];

        int index = row_start_L;
        for(int j = row_start_A; j < row_end_A; j++)
        {
            int col_A = csr_col_ind_A[j];

            if(col_A < i)
            {
                csr_col_ind_L[index] = col_A;
                csr_val_L[index]     = csr_val_A[j] / diag[col_A];
                index++;
            }
            else if(col_A == i)
            {
                csr_col_ind_L[index] = col_A;
                csr_val_L[index]     = beta;
                index++;
            }
        }
    }
}

template <uint32_t BLOCKSIZE, typename T>
__global__ void ssor_fill_upper_precond_kernel(int        m_A,
                                               int        n_A,
                                               int        nnz_A,
                                               T          omega,
                                               const int* csr_row_ptr_A,
                                               const int* csr_col_ind_A,
                                               const T*   csr_val_A,
                                               const int* csr_row_ptr_U,
                                               int*       csr_col_ind_U,
                                               T*         csr_val_U)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = tid + BLOCKSIZE * bid;

    double beta = 1.0 / omega;

    // U = (beta * D - F), beta = 1 / omega
    for(int i = gid; i < m_A; i += BLOCKSIZE * gridDim.x)
    {
        const int row_start_A = csr_row_ptr_A[i];
        const int row_end_A   = csr_row_ptr_A[i + 1];

        const int row_start_U = csr_row_ptr_U[i];

        int index = row_start_U;
        for(int j = row_start_A; j < row_end_A; j++)
        {
            int col_A = csr_col_ind_A[j];

            if(col_A > i)
            {
                csr_col_ind_U[index] = col_A;
                csr_val_U[index]     = csr_val_A[j];
                index++;
            }
            else if(col_A == i)
            {
                csr_col_ind_U[index] = col_A;
                csr_val_U[index]     = beta * csr_val_A[j];
                index++;
            }
        }
    }
}

#endif