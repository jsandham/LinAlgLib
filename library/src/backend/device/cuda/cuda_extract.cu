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

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "cuda_extract.h"

#include "extract_diagonal_kernels.cuh"

#include "../../../trace.h"

//-------------------------------------------------------------------------------
// diagonal d = diag(A)
//-------------------------------------------------------------------------------
void linalg::cuda_extract_diagonal(int           m,
                                   int           n,
                                   int           nnz,
                                   const int*    csr_row_ptr,
                                   const int*    csr_col_ind,
                                   const double* csr_val,
                                   double*       d)
{
    ROUTINE_TRACE("linalg::cuda_extract_diagonal_impl");
    extract_diagonal_kernel<256, 4>
        <<<((m - 1) / (256 / 4) + 1), 256>>>(m, n, nnz, csr_row_ptr, csr_col_ind, csr_val, d);
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// lower triangular L = tril(A)
//-------------------------------------------------------------------------------
void linalg::cuda_extract_lower_triangular_nnz(int        m_A,
                                               int        n_A,
                                               int        nnz_A,
                                               const int* csr_row_ptr_A,
                                               const int* csr_col_ind_A,
                                               int*       csr_row_ptr_L,
                                               int*       nnz_L)
{
    ROUTINE_TRACE("linalg::cuda_extract_lower_triangular_nnz");
    extract_lower_triangular_nnz_kernel<256, 4><<<((m_A - 1) / (256 / 4) + 1), 256>>>(
        m_A, n_A, nnz_A, csr_row_ptr_A, csr_col_ind_A, csr_row_ptr_L);
    CHECK_CUDA_LAUNCH_ERROR();

    // exclusive scan to get row pointers
    thrust::device_ptr<int> d_csr_row_ptr_L(csr_row_ptr_L);
    thrust::exclusive_scan(
        thrust::device, d_csr_row_ptr_L, d_csr_row_ptr_L + (m_A + 1), d_csr_row_ptr_L);

    CHECK_CUDA(cudaMemcpy(nnz_L, &csr_row_ptr_L[m_A], sizeof(int), cudaMemcpyDeviceToHost));
}

void linalg::cuda_extract_lower_triangular(int           m_A,
                                           int           n_A,
                                           int           nnz_A,
                                           const int*    csr_row_ptr_A,
                                           const int*    csr_col_ind_A,
                                           const double* csr_val_A,
                                           int           m_L,
                                           int           n_L,
                                           int           nnz_L,
                                           int*          csr_row_ptr_L,
                                           int*          csr_col_ind_L,
                                           double*       csr_val_L)
{
    ROUTINE_TRACE("linalg::cuda_extract_lower_triangular");
    extract_lower_triangular_kernel<256><<<((m_A - 1) / 256 + 1), 256>>>(m_A,
                                                                         n_A,
                                                                         nnz_A,
                                                                         csr_row_ptr_A,
                                                                         csr_col_ind_A,
                                                                         csr_val_A,
                                                                         m_L,
                                                                         n_L,
                                                                         nnz_L,
                                                                         csr_row_ptr_L,
                                                                         csr_col_ind_L,
                                                                         csr_val_L);
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// upper triangular U = triu(A)
//-------------------------------------------------------------------------------
void linalg::cuda_extract_upper_triangular_nnz(int        m_A,
                                               int        n_A,
                                               int        nnz_A,
                                               const int* csr_row_ptr_A,
                                               const int* csr_col_ind_A,
                                               int*       csr_row_ptr_U,
                                               int*       nnz_U)
{
    ROUTINE_TRACE("linalg::cuda_extract_upper_triangular_nnz");
    extract_upper_triangular_nnz_kernel<256, 4><<<((m_A - 1) / (256 / 4) + 1), 256>>>(
        m_A, n_A, nnz_A, csr_row_ptr_A, csr_col_ind_A, csr_row_ptr_U);
    CHECK_CUDA_LAUNCH_ERROR();

    // exclusive scan to get row pointers
    thrust::device_ptr<int> d_csr_row_ptr_U(csr_row_ptr_U);
    thrust::exclusive_scan(
        thrust::device, d_csr_row_ptr_U, d_csr_row_ptr_U + (m_A + 1), d_csr_row_ptr_U);

    CHECK_CUDA(cudaMemcpy(nnz_U, &csr_row_ptr_U[m_A], sizeof(int), cudaMemcpyDeviceToHost));
}

void linalg::cuda_extract_upper_triangular(int           m_A,
                                           int           n_A,
                                           int           nnz_A,
                                           const int*    csr_row_ptr_A,
                                           const int*    csr_col_ind_A,
                                           const double* csr_val_A,
                                           int           m_U,
                                           int           n_U,
                                           int           nnz_U,
                                           int*          csr_row_ptr_U,
                                           int*          csr_col_ind_U,
                                           double*       csr_val_U)
{
    ROUTINE_TRACE("linalg::cuda_extract_upper_triangular");
    extract_upper_triangular_kernel<256><<<((m_A - 1) / 256 + 1), 256>>>(m_A,
                                                                         n_A,
                                                                         nnz_A,
                                                                         csr_row_ptr_A,
                                                                         csr_col_ind_A,
                                                                         csr_val_A,
                                                                         m_U,
                                                                         n_U,
                                                                         nnz_U,
                                                                         csr_row_ptr_U,
                                                                         csr_col_ind_U,
                                                                         csr_val_U);
    CHECK_CUDA_LAUNCH_ERROR();
}
