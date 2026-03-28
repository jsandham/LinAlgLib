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

#include "cuda_matrix_vector.h"

#include "compute_residual_kernels.cuh"
#include "csrmv_kernels.cuh"

#include "../../../trace.h"

//-------------------------------------------------------------------------------
// sparse matrix-vector product y = A*x
//-------------------------------------------------------------------------------
void linalg::cuda_matrix_vector_product(int           m,
                                        int           n,
                                        int           nnz,
                                        const int*    csr_row_ptr,
                                        const int*    csr_col_ind,
                                        const double* csr_val,
                                        const double* x,
                                        double*       y)
{
    ROUTINE_TRACE("linalg::cuda_matrix_vector_product_impl");

    int avg_nnz_per_row = nnz / m;

    if(avg_nnz_per_row <= 8)
    {
        csrmv_vector_kernel<256, 4><<<((m - 1) / (256 / 4) + 1), 256>>>(
            m, n, nnz, 1.0, csr_row_ptr, csr_col_ind, csr_val, x, 0.0, y);
    }
    else if(avg_nnz_per_row <= 16)
    {
        csrmv_vector_kernel<256, 8><<<((m - 1) / (256 / 8) + 1), 256>>>(
            m, n, nnz, 1.0, csr_row_ptr, csr_col_ind, csr_val, x, 0.0, y);
    }
    else if(avg_nnz_per_row <= 32)
    {
        csrmv_vector_kernel<256, 16><<<((m - 1) / (256 / 16) + 1), 256>>>(
            m, n, nnz, 1.0, csr_row_ptr, csr_col_ind, csr_val, x, 0.0, y);
    }
    else
    {
        csrmv_vector_kernel<256, 32><<<((m - 1) / (256 / 32) + 1), 256>>>(
            m, n, nnz, 1.0, csr_row_ptr, csr_col_ind, csr_val, x, 0.0, y);
    }
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// Compute residual res = b - A * x
//-------------------------------------------------------------------------------
void linalg::cuda_compute_residual(int           m,
                                   int           n,
                                   int           nnz,
                                   const int*    csr_row_ptr,
                                   const int*    csr_col_ind,
                                   const double* csr_val,
                                   const double* x,
                                   const double* b,
                                   double*       res)
{
    ROUTINE_TRACE("linalg::cuda_compute_residual_impl");
    compute_residual_kernel<256, 4><<<((m - 1) / (256 / 4) + 1), 256>>>(
        m, n, nnz, csr_row_ptr, csr_col_ind, csr_val, x, b, res);
    CHECK_CUDA_LAUNCH_ERROR();
}

struct linalg::csrmv_descr
{
};

void linalg::allocate_csrmv_cuda_data(csrmv_descr* descr) {}

void linalg::free_csrmv_cuda_data(csrmv_descr* descr)
{
    if(descr != nullptr)
    {
    }
}

void linalg::cuda_csrmv_analysis(int             m,
                                 int             n,
                                 int             nnz,
                                 const int*      csr_row_ptr,
                                 const int*      csr_col_ind,
                                 const double*   csr_val,
                                 csrmv_algorithm alg,
                                 csrmv_descr*    descr)
{
}

void linalg::cuda_csrmv_solve(int                m,
                              int                n,
                              int                nnz,
                              double             alpha,
                              const int*         csr_row_ptr,
                              const int*         csr_col_ind,
                              const double*      csr_val,
                              const double*      x,
                              double             beta,
                              double*            y,
                              csrmv_algorithm    alg,
                              const csrmv_descr* descr)
{
    ROUTINE_TRACE("linalg::cuda_csrmv_solve");

    int avg_nnz_per_row = nnz / m;

    if(avg_nnz_per_row <= 8)
    {
        csrmv_vector_kernel<256, 4><<<((m - 1) / (256 / 4) + 1), 256>>>(
            m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
    }
    else if(avg_nnz_per_row <= 16)
    {
        csrmv_vector_kernel<256, 8><<<((m - 1) / (256 / 8) + 1), 256>>>(
            m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
    }
    else if(avg_nnz_per_row <= 32)
    {
        csrmv_vector_kernel<256, 16><<<((m - 1) / (256 / 16) + 1), 256>>>(
            m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
    }
    else
    {
        csrmv_vector_kernel<256, 32><<<((m - 1) / (256 / 32) + 1), 256>>>(
            m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
    }
}
