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
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

#include "cuda_math.h"

#include "axpby_kernels.cuh"
#include "compute_residual_kernels.cuh"
#include "csrmv_kernels.cuh"
#include "dot_product_kernels.cuh"
#include "extract_diagonal_kernels.cuh"
#include "find_minmax_kernels.cuh"
#include "jacobi_solve_kernels.cuh"

#include "../../../trace.h"

//-------------------------------------------------------------------------------
// Compute y = alpha * x + y
//-------------------------------------------------------------------------------
void linalg::cuda_axpy(int size, double alpha, const double* x, double* y)
{
    ROUTINE_TRACE("linalg::cuda_axpy_impl");
    axpy_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, alpha, x, y);
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// Compute y = alpha * x + beta * y
//-------------------------------------------------------------------------------
void linalg::cuda_axpby(int size, double alpha, const double* x, double beta, double* y)
{
    ROUTINE_TRACE("linalg::cuda_axpby_impl");
    axpby_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, alpha, x, beta, y);
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// Compute z = alpha * x + beta * y + gamma * z
//-------------------------------------------------------------------------------
void linalg::cuda_axpbypgz(
    int size, double alpha, const double* x, double beta, const double* y, double gamma, double* z)
{
    ROUTINE_TRACE("linalg::cuda_axpbypgz_impl");
    axpbypgz_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, alpha, x, beta, y, gamma, z);
    CHECK_CUDA_LAUNCH_ERROR();
}

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

    csrmv_vector_kernel<256, 4><<<((m - 1) / (256 / 4) + 1), 256>>>(
        m, n, nnz, 1.0, csr_row_ptr, csr_col_ind, csr_val, x, 0.0, y);
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// dot product z = x*y
//-------------------------------------------------------------------------------
double linalg::cuda_dot_product(const double* x, const double* y, int size)
{
    ROUTINE_TRACE("linalg::cuda_dot_product_impl");
    double* workspace = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&workspace, sizeof(double) * 256));

    dot_product_kernel_part1<256><<<256, 256>>>(size, x, y, workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    dot_product_kernel_part2<256><<<1, 256>>>(workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    double result;
    CHECK_CUDA(cudaMemcpy(&result, workspace, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(workspace));

    return result;
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

//-------------------------------------------------------------------------------
// exclusive scan
//-------------------------------------------------------------------------------
void linalg::cuda_exclusive_scan(int64_t* x, int n) {}

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
// infinity norm
//-------------------------------------------------------------------------------
double linalg::cuda_norm_inf(const double* array, int size)
{
    ROUTINE_TRACE("linalg::cuda_norm_inf_impl");
    double* workspace = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&workspace, sizeof(double) * 256));

    find_max_kernel_part1<256><<<256, 256>>>(size, array, workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    find_max_kernel_part2<256><<<1, 256>>>(workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    double result;
    CHECK_CUDA(cudaMemcpy(&result, workspace, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(workspace));

    return result;
}

//-------------------------------------------------------------------------------
// jacobi solve
//-------------------------------------------------------------------------------
void linalg::cuda_jacobi_solve(const double* rhs, const double* diag, double* x, size_t size)
{
    ROUTINE_TRACE("linalg::cuda_jacobi_solve_impl");
    jacobi_solve_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, rhs, diag, x);
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// Compute y = alpha * A * x + beta * y
//-------------------------------------------------------------------------------
void linalg::cuda_csrmv(int           m,
                        int           n,
                        int           nnz,
                        double        alpha,
                        const int*    csr_row_ptr,
                        const int*    csr_col_ind,
                        const double* csr_val,
                        const double* x,
                        double        beta,
                        double*       y)
{
    ROUTINE_TRACE("linalg::cuda_csrmv_impl");

    csrmv_vector_kernel<256, 4><<<((m - 1) / (256 / 4) + 1), 256>>>(
        m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// Compute C = alpha * A * B + beta * D
//-------------------------------------------------------------------------------
void linalg::cuda_csrgemm_nnz(int        m,
                              int        n,
                              int        k,
                              int        nnz_A,
                              int        nnz_B,
                              int        nnz_D,
                              double     alpha,
                              const int* csr_row_ptr_A,
                              const int* csr_col_ind_A,
                              const int* csr_row_ptr_B,
                              const int* csr_col_ind_B,
                              double     beta,
                              const int* csr_row_ptr_D,
                              const int* csr_col_ind_D,
                              int*       csr_row_ptr_C,
                              int*       nnz_C)
{
}

void linalg::cuda_csrgemm(int           m,
                          int           n,
                          int           k,
                          int           nnz_A,
                          int           nnz_B,
                          int           nnz_D,
                          double        alpha,
                          const int*    csr_row_ptr_A,
                          const int*    csr_col_ind_A,
                          const double* csr_val_A,
                          const int*    csr_row_ptr_B,
                          const int*    csr_col_ind_B,
                          const double* csr_val_B,
                          double        beta,
                          const int*    csr_row_ptr_D,
                          const int*    csr_col_ind_D,
                          const double* csr_val_D,
                          const int*    csr_row_ptr_C,
                          int*          csr_col_ind_C,
                          double*       csr_val_C)
{
}

//-------------------------------------------------------------------------------
// Compute C = alpha * A + beta * B
//-------------------------------------------------------------------------------
void linalg::cuda_csrgeam_nnz(int        m,
                              int        n,
                              int        nnz_A,
                              int        nnz_B,
                              double     alpha,
                              const int* csr_row_ptr_A,
                              const int* csr_col_ind_A,
                              double     beta,
                              const int* csr_row_ptr_B,
                              const int* csr_col_ind_B,
                              int*       csr_row_ptr_C,
                              int*       nnz_C)
{
}

void linalg::cuda_csrgeam(int           m,
                          int           n,
                          int           nnz_A,
                          int           nnz_B,
                          double        alpha,
                          const int*    csr_row_ptr_A,
                          const int*    csr_col_ind_A,
                          const double* csr_val_A,
                          double        beta,
                          const int*    csr_row_ptr_B,
                          const int*    csr_col_ind_B,
                          const double* csr_val_B,
                          const int*    csr_row_ptr_C,
                          int*          csr_col_ind_C,
                          double*       csr_val_C)
{
}

//-------------------------------------------------------------------------------
// Compute incomplete LU factorization inplace
//-------------------------------------------------------------------------------
void linalg::cuda_csrilu0(int        m,
                          int        n,
                          int        nnz,
                          const int* csr_row_ptr,
                          const int* csr_col_ind,
                          double*    csr_val,
                          int*       structural_zero,
                          int*       numeric_zero)
{
}

//----------------------------------------------------------------------------------------
// Compute incomplete Cholesky factorization inplace (only modifies lower triangular part)
//----------------------------------------------------------------------------------------
void linalg::cuda_csric0(int        m,
                         int        n,
                         int        nnz,
                         const int* csr_row_ptr,
                         const int* csr_col_ind,
                         double*    csr_val,
                         int*       structural_zero,
                         int*       numeric_zero)
{
}

//-------------------------------------------------------------------------------
// solve Lx = b where L is a lower triangular sparse matrix
//-------------------------------------------------------------------------------
void linalg::cuda_forward_solve(const int*    csr_row_ptr,
                                const int*    csr_col_ind,
                                const double* csr_val,
                                const double* b,
                                double*       x,
                                int           n,
                                bool          unit_diag)
{
}

//-------------------------------------------------------------------------------
// solve Ux = b where U is a upper triangular sparse matrix
//-------------------------------------------------------------------------------
void linalg::cuda_backward_solve(const int*    csr_row_ptr,
                                 const int*    csr_col_ind,
                                 const double* csr_val,
                                 const double* b,
                                 double*       x,
                                 int           n,
                                 bool          unit_diag)
{
}