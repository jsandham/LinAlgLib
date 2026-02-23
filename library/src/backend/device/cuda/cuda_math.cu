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

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h> // Required to specify thrust::device execution policy
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "cuda_math.h"
#include "cuda_primitives.h"

#include "axpby_kernels.cuh"
#include "compute_residual_kernels.cuh"
#include "csr2coo_kernels.cuh"
#include "csrgeam_kernels.cuh"
#include "csrgemm_kernels.cuh"
#include "csric0_kernels.cuh"
#include "csrmv_kernels.cuh"
#include "csrtrsv_kernels.cuh"
#include "dot_product_kernels.cuh"
#include "extract_diagonal_kernels.cuh"
#include "preconditioner_kernels.cuh"
#include "tridiagonal_solver_kernels.cuh"

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

//-------------------------------------------------------------------------------
// scale diagonal
//-------------------------------------------------------------------------------
void linalg::cuda_scale_diagonal(
    const int* csr_row_ptr, const int* csr_col_ind, double* csr_val, int m, double scalar)
{
    ROUTINE_TRACE("linalg::cuda_scale_diagonal_impl");
    scale_diagonal_kernel<256>
        <<<((m - 1) / 256 + 1), 256>>>(m, csr_row_ptr, csr_col_ind, csr_val, scalar);
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// scale by inverse diagonal
//-------------------------------------------------------------------------------
void linalg::cuda_scale_by_inverse_diagonal(
    const int* csr_row_ptr, const int* csr_col_ind, double* csr_val, int m, const double* diag)
{
    ROUTINE_TRACE("linalg::cuda_scale_by_inverse_diagonal_impl");
    scale_by_inverse_diagonal_kernel<256>
        <<<((m - 1) / 256 + 1), 256>>>(m, csr_row_ptr, csr_col_ind, csr_val, diag);
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// infinity norm
//-------------------------------------------------------------------------------
double linalg::cuda_norm_inf(const double* array, int size)
{
    ROUTINE_TRACE("linalg::cuda_norm_inf_impl");
    return cuda_find_maximum(size, array);
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
// SSOR fill lower preconditioner L = (beta * D - E) * D^-1, beta = 1 / omega
//-------------------------------------------------------------------------------
void linalg::cuda_ssor_fill_lower_precond(int           m_A,
                                          int           n_A,
                                          int           nnz_A,
                                          const int*    csr_row_ptr_A,
                                          const int*    csr_col_ind_A,
                                          const double* csr_val_A,
                                          int           m_L,
                                          int           n_L,
                                          int           nnz_L,
                                          const int*    csr_row_ptr_L,
                                          int*          csr_col_ind_L,
                                          double*       csr_val_L,
                                          double        omega)
{
    ROUTINE_TRACE("linalg::cuda_ssor_fill_lower_precond");

    double* diag = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&diag, sizeof(double) * m_A));

    cuda_extract_diagonal(m_A, n_A, nnz_A, csr_row_ptr_A, csr_col_ind_A, csr_val_A, diag);

    ssor_fill_lower_precond_kernel<256><<<((m_A - 1) / 256 + 1), 256>>>(m_A,
                                                                        n_A,
                                                                        nnz_A,
                                                                        omega,
                                                                        csr_row_ptr_A,
                                                                        csr_col_ind_A,
                                                                        csr_val_A,
                                                                        diag,
                                                                        csr_row_ptr_L,
                                                                        csr_col_ind_L,
                                                                        csr_val_L);
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaFree(diag));
}

//-------------------------------------------------------------------------------
// SSOR fill upper preconditioner U = (beta * D - F), beta = 1 / omega
//-------------------------------------------------------------------------------
void linalg::cuda_ssor_fill_upper_precond(int           m_A,
                                          int           n_A,
                                          int           nnz_A,
                                          const int*    csr_row_ptr_A,
                                          const int*    csr_col_ind_A,
                                          const double* csr_val_A,
                                          int           m_U,
                                          int           n_U,
                                          int           nnz_U,
                                          const int*    csr_row_ptr_U,
                                          int*          csr_col_ind_U,
                                          double*       csr_val_U,
                                          double        omega)
{
    ROUTINE_TRACE("linalg::cuda_ssor_fill_upper_precond");
    ssor_fill_upper_precond_kernel<256><<<((m_A - 1) / 256 + 1), 256>>>(m_A,
                                                                        n_A,
                                                                        nnz_A,
                                                                        omega,
                                                                        csr_row_ptr_A,
                                                                        csr_col_ind_A,
                                                                        csr_val_A,
                                                                        csr_row_ptr_U,
                                                                        csr_col_ind_U,
                                                                        csr_val_U);
    CHECK_CUDA_LAUNCH_ERROR();
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

void linalg::cuda_csr2csc_buffer_size(int           m,
                                      int           n,
                                      int           nnz,
                                      const int*    csr_row_ptr,
                                      const int*    csr_Col_ind,
                                      const double* csr_val,
                                      size_t*       buffer_size)
{
    *buffer_size = 0;
    *buffer_size += sizeof(int) * nnz; // perm
    *buffer_size += sizeof(int) * nnz; // coo_row_ind
}

void linalg::cuda_csr2csc(int           m,
                          int           n,
                          int           nnz,
                          const int*    csr_row_ptr,
                          const int*    csr_col_ind,
                          const double* csr_val,
                          int*          csc_col_ptr,
                          int*          csc_row_ind,
                          double*       csc_val,
                          void*         buffer)
{
    int* perm        = reinterpret_cast<int*>(buffer);
    int* coo_row_ind = reinterpret_cast<int*>(buffer) + nnz;

    fill_identity_permuation_kernel<256><<<((nnz - 1) / 256 + 1), 256>>>(nnz, perm);
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaMemcpy(csc_row_ind, csr_col_ind, sizeof(int) * nnz, cudaMemcpyDeviceToDevice));

    // Wrap Raw Pointers and Execute Thrust Algorithm
    // thrust::device_ptr allows us to treat a raw pointer like a Thrust iterator.
    thrust::device_ptr<int> d_keys(csc_row_ind);
    thrust::device_ptr<int> d_values(perm);

    // Use sort_by_key: sorts d_keys and applies the identical permutation to d_values
    thrust::sort_by_key(d_keys, d_keys + nnz, d_values);

    coo2csr_kernel<256><<<((nnz - 1) / 256 + 1), 256>>>(m, n, nnz, csc_row_ind, csc_col_ptr);
    CHECK_CUDA_LAUNCH_ERROR();

    csr2coo_kernel<256><<<((nnz - 1) / 256 + 1), 256>>>(m, n, nnz, csr_row_ptr, coo_row_ind);
    CHECK_CUDA_LAUNCH_ERROR();

    csr2csc_permute_colval_kernel<256><<<((nnz - 1) / 256 + 1), 256>>>(
        m, n, nnz, coo_row_ind, csr_val, perm, csc_row_ind, csc_val);
    CHECK_CUDA_LAUNCH_ERROR();
}

struct linalg::csrtrsv_descr
{
    int* done_array;
    int* row_perm;
    int* diag_ind;
};

void linalg::allocate_csrtrsv_cuda_data(csrtrsv_descr* descr)
{
    descr->done_array = nullptr;
    descr->row_perm   = nullptr;
    descr->diag_ind   = nullptr;
}

void linalg::free_csrtrsv_cuda_data(csrtrsv_descr* descr)
{
    if(descr != nullptr)
    {
        if(descr->done_array != nullptr)
        {
            std::cout << "Freeing done_array" << std::endl;
            CHECK_CUDA(cudaFree(descr->done_array));
        }

        if(descr->row_perm != nullptr)
        {
            std::cout << "Freeing row_perm" << std::endl;
            CHECK_CUDA(cudaFree(descr->row_perm));
        }

        if(descr->diag_ind != nullptr)
        {
            std::cout << "Freeing diag_ind" << std::endl;
            CHECK_CUDA(cudaFree(descr->diag_ind));
        }
    }
}

void linalg::cuda_csrtrsv_analysis(int             m,
                                   int             n,
                                   int             nnz,
                                   const int*      csr_row_ptr,
                                   const int*      csr_col_ind,
                                   const double*   csr_val,
                                   triangular_type tri_type,
                                   diagonal_type   diag_type,
                                   csrtrsv_descr*  descr)
{
    std::cout << "csrtrsv_analysis m: " << m << " n: " << n << " nnz: " << nnz << std::endl;

    // Free any previous allocations?
    assert(descr->done_array == nullptr);
    assert(descr->row_perm == nullptr);
    assert(descr->diag_ind == nullptr);

    descr->done_array = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&(descr->done_array), sizeof(int) * m));
    CHECK_CUDA(cudaMemset(descr->done_array, 0, sizeof(int) * m));

    descr->row_perm = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&(descr->row_perm), sizeof(int) * m));

    descr->diag_ind = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&(descr->diag_ind), sizeof(int) * m));

    csrtrsv_analysis_kernel<256, 32><<<((m - 1) / (256 / 32) + 1), 256>>>(
        m, tri_type, csr_row_ptr, csr_col_ind, csr_val, descr->diag_ind, descr->done_array);
    CHECK_CUDA_LAUNCH_ERROR();

    // std::vector<int> hdiag_ind(m, 10);
    // CHECK_CUDA(
    //     cudaMemcpy(hdiag_ind.data(), descr->diag_ind, sizeof(int) * m, cudaMemcpyDeviceToHost));

    // std::cout << "diag_ind" << std::endl;
    // for(int i = 0; i < m; i++)
    // {
    //     std::cout << hdiag_ind[i] << " ";
    // }
    // std::cout << "" << std::endl;

    // std::vector<int> hdone_array(m, 0);
    // CHECK_CUDA(
    //     cudaMemcpy(hdone_array.data(), descr->done_array, sizeof(int) * m, cudaMemcpyDeviceToHost));

    // std::cout << "done_array" << std::endl;
    // for(int i = 0; i < m; i++)
    // {
    //     std::cout << hdone_array[i] << " ";
    // }
    // std::cout << "" << std::endl;

    fill_identity_permuation_kernel<256><<<((m - 1) / 256 + 1), 256>>>(m, descr->row_perm);
    CHECK_CUDA_LAUNCH_ERROR();

    // Wrap Raw Pointers and Execute Thrust Algorithm
    // thrust::device_ptr allows us to treat a raw pointer like a Thrust iterator.
    thrust::device_ptr<int> d_keys(descr->done_array);
    thrust::device_ptr<int> d_values(descr->row_perm);

    // Use sort_by_key: sorts d_keys and applies the identical permutation to d_values
    thrust::sort_by_key(d_keys, d_keys + m, d_values);

    // std::vector<int> hrow_perm(m, 0);
    // CHECK_CUDA(
    //     cudaMemcpy(hrow_perm.data(), descr->row_perm, sizeof(int) * m, cudaMemcpyDeviceToHost));

    // std::cout << "row_perm" << std::endl;
    // for(int i = 0; i < m; i++)
    // {
    //     std::cout << hrow_perm[i] << " ";
    // }
    // std::cout << "" << std::endl;
}
void linalg::cuda_csrtrsv_solve(int                  m,
                                int                  n,
                                int                  nnz,
                                double               alpha,
                                const int*           csr_row_ptr,
                                const int*           csr_col_ind,
                                const double*        csr_val,
                                const double*        b,
                                double*              x,
                                triangular_type      tri_type,
                                diagonal_type        diag_type,
                                const csrtrsv_descr* descr)
{
    assert(descr->diag_ind != nullptr);
    assert(descr->done_array != nullptr);
    assert(descr->row_perm != nullptr);

    CHECK_CUDA(cudaMemset(descr->done_array, 0, sizeof(int) * m));

    csrtrsv_solve_kernel<256, 32><<<((m - 1) / (256 / 32) + 1), 256>>>(m,
                                                                       tri_type,
                                                                       diag_type,
                                                                       alpha,
                                                                       csr_row_ptr,
                                                                       csr_col_ind,
                                                                       csr_val,
                                                                       descr->diag_ind,
                                                                       b,
                                                                       x,
                                                                       descr->done_array,
                                                                       descr->row_perm);
    CHECK_CUDA_LAUNCH_ERROR();

    // std::vector<double> hx(m, 0.0);
    // CHECK_CUDA(cudaMemcpy(hx.data(), x, sizeof(double) * m, cudaMemcpyDeviceToHost));
    // std::cout << "x" << std::endl;
    // for(int i = 0; i < m; i++)
    // {
    //     std::cout << hx[i] << " ";
    // }
    // std::cout << std::endl;
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

//-------------------------------------------------------------------------------
// Compute C = alpha * A * B + beta * D
//-------------------------------------------------------------------------------
struct linalg::csrgemm_descr
{
    int* perm;
    int* bin_offsets;
};

void linalg::allocate_csrgemm_cuda_data(csrgemm_descr* descr)
{
    descr->perm        = nullptr;
    descr->bin_offsets = nullptr;
}

void linalg::free_csrgemm_cuda_data(csrgemm_descr* descr)
{
    if(descr != nullptr)
    {
        if(descr->perm != nullptr)
        {
            std::cout << "Freeing perm" << std::endl;
            CHECK_CUDA(cudaFree(descr->perm));
        }

        if(descr->bin_offsets != nullptr)
        {
            std::cout << "Freeing bin_offsets" << std::endl;
            CHECK_CUDA(cudaFree(descr->bin_offsets));
        }
    }
}

void linalg::cuda_csrgemm_nnz(int            m,
                              int            n,
                              int            k,
                              int            nnz_A,
                              int            nnz_B,
                              int            nnz_D,
                              csrgemm_descr* descr,
                              double         alpha,
                              const int*     csr_row_ptr_A,
                              const int*     csr_col_ind_A,
                              const int*     csr_row_ptr_B,
                              const int*     csr_col_ind_B,
                              double         beta,
                              const int*     csr_row_ptr_D,
                              const int*     csr_col_ind_D,
                              int*           csr_row_ptr_C,
                              int*           nnz_C)
{
    // Determine maximum hash table size
    csrgemm_count_products_kernel<256, 32><<<((m - 1) / (256 / 32) + 1), 256>>>(
        m, alpha, csr_row_ptr_A, csr_col_ind_A, csr_row_ptr_B, beta, csr_row_ptr_D, csr_row_ptr_C);
    CHECK_CUDA_LAUNCH_ERROR();

    std::vector<int> hcsr_row_ptr_C(m + 1);
    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    descr->perm = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&descr->perm, sizeof(int) * m));

    fill_identity_permuation_kernel<256><<<((m - 1) / 256 + 1), 256>>>(m, descr->perm);
    CHECK_CUDA_LAUNCH_ERROR();

    // Wrap Raw Pointers and Execute Thrust Algorithm
    // thrust::device_ptr allows us to treat a raw pointer like a Thrust iterator.
    thrust::device_ptr<int> d_keys(csr_row_ptr_C);
    thrust::device_ptr<int> d_values(descr->perm);

    // Use sort_by_key: sorts d_keys and applies the identical permutation to d_values
    thrust::sort_by_key(d_keys, d_keys + m, d_values);

    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    std::vector<int> hperm(m);
    CHECK_CUDA(cudaMemcpy(hperm.data(), descr->perm, sizeof(int) * m, cudaMemcpyDeviceToHost));

    std::cout << "perm" << std::endl;
    for(int i = 0; i < m; i++)
    {
        std::cout << hperm[i] << " ";
    }
    std::cout << "" << std::endl;

    // Bin rows based on required hash table size.
    // Bin sizes are: 32, 64, 128, 256, 512, 1024, 2048, 4096
    int bin_count      = 8;
    descr->bin_offsets = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&descr->bin_offsets, sizeof(int) * (bin_count + 1)));
    CHECK_CUDA(cudaMemset(descr->bin_offsets, 0, sizeof(int) * (bin_count + 1)));

    compute_rows_bin_number_kernel<256><<<((m - 1) / 256 + 1), 256>>>(m, csr_row_ptr_C);
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    fill_bin_offsets_kernel<256>
        <<<((m - 1) / 256 + 1), 256>>>(m, csr_row_ptr_C, descr->bin_offsets);
    CHECK_CUDA_LAUNCH_ERROR();

    std::vector<int> hbin_offsets(bin_count + 1, 0);
    CHECK_CUDA(cudaMemcpy(hbin_offsets.data(),
                          descr->bin_offsets,
                          sizeof(int) * (bin_count + 1),
                          cudaMemcpyDeviceToHost));

    std::cout << "hbin_offsets" << std::endl;
    for(int i = 0; i < bin_count + 1; i++)
    {
        std::cout << hbin_offsets[i] << " ";
    }
    std::cout << "" << std::endl;

    const int bin_0_count = hbin_offsets[1] - hbin_offsets[0];
    const int bin_1_count = hbin_offsets[2] - hbin_offsets[1];
    const int bin_2_count = hbin_offsets[3] - hbin_offsets[2];
    const int bin_3_count = hbin_offsets[4] - hbin_offsets[3];
    const int bin_4_count = hbin_offsets[5] - hbin_offsets[4];
    const int bin_5_count = hbin_offsets[6] - hbin_offsets[5];
    const int bin_6_count = hbin_offsets[7] - hbin_offsets[6];
    const int bin_7_count = hbin_offsets[8] - hbin_offsets[7];

    CHECK_CUDA(cudaMemset(csr_row_ptr_C, 0, sizeof(int) * (m + 1)));

    if(bin_0_count > 0)
    {
        std::cout << "bin_0_count: " << bin_0_count << std::endl;
        csrgemm_nnz_per_row_kernel<256, 32, 32>
            <<<((bin_0_count - 1) / (256 / 32) + 1), 256>>>(bin_0_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[0]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_row_ptr_C);
    }
    if(bin_1_count > 0)
    {
        std::cout << "bin_1_count: " << bin_1_count << std::endl;
        csrgemm_nnz_per_row_kernel<256, 32, 64>
            <<<((bin_1_count - 1) / (256 / 32) + 1), 256>>>(bin_1_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[1]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_row_ptr_C);
    }
    if(bin_2_count > 0)
    {
        std::cout << "bin_2_count: " << bin_2_count << std::endl;
        csrgemm_nnz_per_row_kernel<256, 32, 128>
            <<<((bin_2_count - 1) / (256 / 32) + 1), 256>>>(bin_2_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[2]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_row_ptr_C);
    }
    if(bin_3_count > 0)
    {
        std::cout << "bin_3_count: " << bin_3_count << std::endl;
        csrgemm_nnz_per_row_kernel<256, 32, 256>
            <<<((bin_3_count - 1) / (256 / 32) + 1), 256>>>(bin_3_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[3]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_row_ptr_C);
    }
    if(bin_4_count > 0)
    {
        std::cout << "bin_4_count: " << bin_4_count << std::endl;
        csrgemm_nnz_per_row_kernel<256, 32, 512>
            <<<((bin_4_count - 1) / (256 / 32) + 1), 256>>>(bin_4_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[4]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_row_ptr_C);
    }
    if(bin_5_count > 0)
    {
        std::cout << "bin_5_count: " << bin_5_count << std::endl;
        csrgemm_nnz_per_row_kernel<256, 32, 1024>
            <<<((bin_5_count - 1) / (256 / 32) + 1), 256>>>(bin_5_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[5]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_row_ptr_C);
    }
    if(bin_6_count > 0)
    {
        std::cout << "bin_6_count: " << bin_6_count << std::endl;
        csrgemm_nnz_per_row_kernel<128, 32, 2048>
            <<<((bin_6_count - 1) / (128 / 32) + 1), 128>>>(bin_6_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[6]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_row_ptr_C);
    }
    if(bin_7_count > 0)
    {
        std::cout << "bin_7_count: " << bin_7_count << std::endl;
        csrgemm_nnz_per_row_kernel<64, 32, 4096>
            <<<((bin_7_count - 1) / (64 / 32) + 1), 64>>>(bin_7_count,
                                                          descr->perm,
                                                          &(descr->bin_offsets[7]),
                                                          alpha,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          beta,
                                                          csr_row_ptr_D,
                                                          csr_col_ind_D,
                                                          csr_row_ptr_C);
    }
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    cuda_exclusive_scan(m + 1, csr_row_ptr_C);

    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    CHECK_CUDA(cudaMemcpy(nnz_C, csr_row_ptr_C + m, sizeof(int), cudaMemcpyDeviceToHost));

    // CHECK_CUDA(cudaFree(perm));
    // CHECK_CUDA(cudaFree(bin_offsets));
}

void linalg::cuda_csrgemm_solve(int                  m,
                                int                  n,
                                int                  k,
                                int                  nnz_A,
                                int                  nnz_B,
                                int                  nnz_D,
                                int                  nnz_C,
                                const csrgemm_descr* descr,
                                double               alpha,
                                const int*           csr_row_ptr_A,
                                const int*           csr_col_ind_A,
                                const double*        csr_val_A,
                                const int*           csr_row_ptr_B,
                                const int*           csr_col_ind_B,
                                const double*        csr_val_B,
                                double               beta,
                                const int*           csr_row_ptr_D,
                                const int*           csr_col_ind_D,
                                const double*        csr_val_D,
                                const int*           csr_row_ptr_C,
                                int*                 csr_col_ind_C,
                                double*              csr_val_C)
{
    std::cout << "nnz_C: " << nnz_C << std::endl;

    int              bin_count = 8;
    std::vector<int> hbin_offsets(bin_count + 1, 0);
    CHECK_CUDA(cudaMemcpy(hbin_offsets.data(),
                          descr->bin_offsets,
                          sizeof(int) * (bin_count + 1),
                          cudaMemcpyDeviceToHost));

    std::cout << "hbin_offsets" << std::endl;
    for(int i = 0; i < bin_count + 1; i++)
    {
        std::cout << hbin_offsets[i] << " ";
    }
    std::cout << "" << std::endl;

    const int bin_0_count = hbin_offsets[1] - hbin_offsets[0];
    const int bin_1_count = hbin_offsets[2] - hbin_offsets[1];
    const int bin_2_count = hbin_offsets[3] - hbin_offsets[2];
    const int bin_3_count = hbin_offsets[4] - hbin_offsets[3];
    const int bin_4_count = hbin_offsets[5] - hbin_offsets[4];
    const int bin_5_count = hbin_offsets[6] - hbin_offsets[5];
    const int bin_6_count = hbin_offsets[7] - hbin_offsets[6];
    const int bin_7_count = hbin_offsets[8] - hbin_offsets[7];

    if(bin_0_count > 0)
    {
        std::cout << "bin_0_count: " << bin_0_count << std::endl;
        csrgemm_fill_kernel<256, 32, 32>
            <<<((bin_0_count - 1) / (256 / 32) + 1), 256>>>(bin_0_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[0]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_val_D,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_1_count > 0)
    {
        std::cout << "bin_1_count: " << bin_1_count << std::endl;
        csrgemm_fill_kernel<256, 32, 64>
            <<<((bin_1_count - 1) / (256 / 32) + 1), 256>>>(bin_1_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[1]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_val_D,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_2_count > 0)
    {
        std::cout << "bin_2_count: " << bin_2_count << std::endl;
        csrgemm_fill_kernel<256, 32, 128>
            <<<((bin_2_count - 1) / (256 / 32) + 1), 256>>>(bin_2_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[2]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_val_D,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_3_count > 0)
    {
        std::cout << "bin_3_count: " << bin_3_count << std::endl;
        csrgemm_fill_kernel<256, 32, 256>
            <<<((bin_3_count - 1) / (256 / 32) + 1), 256>>>(bin_3_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[3]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_val_D,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_4_count > 0)
    {
        std::cout << "bin_4_count: " << bin_4_count << std::endl;
        csrgemm_fill_kernel<256, 32, 512>
            <<<((bin_4_count - 1) / (256 / 32) + 1), 256>>>(bin_4_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[4]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_val_D,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_5_count > 0)
    {
        std::cout << "bin_5_count: " << bin_5_count << std::endl;
        csrgemm_fill_kernel<128, 32, 1024>
            <<<((bin_5_count - 1) / (128 / 32) + 1), 128>>>(bin_5_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[5]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            beta,
                                                            csr_row_ptr_D,
                                                            csr_col_ind_D,
                                                            csr_val_D,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_6_count > 0)
    {
        std::cout << "bin_6_count: " << bin_6_count << std::endl;
        csrgemm_fill_kernel<64, 32, 2048>
            <<<((bin_6_count - 1) / (64 / 32) + 1), 64>>>(bin_6_count,
                                                          descr->perm,
                                                          &(descr->bin_offsets[6]),
                                                          alpha,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          csr_val_A,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          csr_val_B,
                                                          beta,
                                                          csr_row_ptr_D,
                                                          csr_col_ind_D,
                                                          csr_val_D,
                                                          csr_row_ptr_C,
                                                          csr_col_ind_C,
                                                          csr_val_C);
    }
    if(bin_7_count > 0)
    {
        std::cout << "bin_7_count: " << bin_7_count << std::endl;
        csrgemm_fill_kernel<32, 32, 4096>
            <<<((bin_7_count - 1) / (32 / 32) + 1), 32>>>(bin_7_count,
                                                          descr->perm,
                                                          &(descr->bin_offsets[7]),
                                                          alpha,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          csr_val_A,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          csr_val_B,
                                                          beta,
                                                          csr_row_ptr_D,
                                                          csr_col_ind_D,
                                                          csr_val_D,
                                                          csr_row_ptr_C,
                                                          csr_col_ind_C,
                                                          csr_val_C);
    }
    CHECK_CUDA_LAUNCH_ERROR();

    std::vector<int>    hcsr_row_ptr_C(m + 1, 0);
    std::vector<int>    hcsr_col_ind_C(nnz_C, 0);
    std::vector<double> hcsr_val_C(nnz_C);
    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(
        hcsr_col_ind_C.data(), csr_col_ind_C, sizeof(int) * nnz_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(hcsr_val_C.data(), csr_val_C, sizeof(double) * nnz_C, cudaMemcpyDeviceToHost));

    std::cout << "hcsr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "hcsr_col_ind_C" << std::endl;
    for(int i = 0; i < nnz_C; i++)
    {
        std::cout << hcsr_col_ind_C[i] << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "hcsr_val_C" << std::endl;
    for(int i = 0; i < nnz_C; i++)
    {
        std::cout << hcsr_val_C[i] << " ";
    }
    std::cout << "" << std::endl;
}

//-------------------------------------------------------------------------------
// Compute C = alpha * A + beta * B
//-------------------------------------------------------------------------------
struct linalg::csrgeam_descr
{
    int* perm;
    int* bin_offsets;
};

void linalg::allocate_csrgeam_cuda_data(csrgeam_descr* descr)
{
    descr->perm        = nullptr;
    descr->bin_offsets = nullptr;
}

void linalg::free_csrgeam_cuda_data(csrgeam_descr* descr)
{
    if(descr != nullptr)
    {
        if(descr->perm != nullptr)
        {
            std::cout << "Freeing perm" << std::endl;
            CHECK_CUDA(cudaFree(descr->perm));
        }

        if(descr->bin_offsets != nullptr)
        {
            std::cout << "Freeing bin_offsets" << std::endl;
            CHECK_CUDA(cudaFree(descr->bin_offsets));
        }
    }
}

void linalg::cuda_csrgeam_nnz(int            m,
                              int            n,
                              int            nnz_A,
                              int            nnz_B,
                              csrgeam_descr* descr,
                              double         alpha,
                              const int*     csr_row_ptr_A,
                              const int*     csr_col_ind_A,
                              double         beta,
                              const int*     csr_row_ptr_B,
                              const int*     csr_col_ind_B,
                              int*           csr_row_ptr_C,
                              int*           nnz_C)
{
    // Determine maximum hash table size
    csrgeam_count_additions_kernel<256><<<((m - 1) / 256 + 1), 256>>>(
        m, alpha, csr_row_ptr_A, csr_col_ind_A, beta, csr_row_ptr_B, csr_row_ptr_C);
    CHECK_CUDA_LAUNCH_ERROR();

    std::vector<int> hcsr_row_ptr_C(m + 1);
    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    descr->perm = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&descr->perm, sizeof(int) * m));

    fill_identity_permuation_kernel<256><<<((m - 1) / 256 + 1), 256>>>(m, descr->perm);
    CHECK_CUDA_LAUNCH_ERROR();

    // Wrap Raw Pointers and Execute Thrust Algorithm
    // thrust::device_ptr allows us to treat a raw pointer like a Thrust iterator.
    thrust::device_ptr<int> d_keys(csr_row_ptr_C);
    thrust::device_ptr<int> d_values(descr->perm);

    // Use sort_by_key: sorts d_keys and applies the identical permutation to d_values
    thrust::sort_by_key(d_keys, d_keys + m, d_values);

    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    std::vector<int> hperm(m);
    CHECK_CUDA(cudaMemcpy(hperm.data(), descr->perm, sizeof(int) * m, cudaMemcpyDeviceToHost));

    std::cout << "perm" << std::endl;
    for(int i = 0; i < m; i++)
    {
        std::cout << hperm[i] << " ";
    }
    std::cout << "" << std::endl;

    // Bin rows based on required hash table size.
    // Bin sizes are: 32, 64, 128, 256, 512, 1024, 2048, 4096
    int bin_count      = 8;
    descr->bin_offsets = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&descr->bin_offsets, sizeof(int) * (bin_count + 1)));
    CHECK_CUDA(cudaMemset(descr->bin_offsets, 0, sizeof(int) * (bin_count + 1)));

    compute_rows_bin_number_kernel<256><<<((m - 1) / 256 + 1), 256>>>(m, csr_row_ptr_C);
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    fill_bin_offsets_kernel<256>
        <<<((m - 1) / 256 + 1), 256>>>(m, csr_row_ptr_C, descr->bin_offsets);
    CHECK_CUDA_LAUNCH_ERROR();

    std::vector<int> hbin_offsets(bin_count + 1, 0);
    CHECK_CUDA(cudaMemcpy(hbin_offsets.data(),
                          descr->bin_offsets,
                          sizeof(int) * (bin_count + 1),
                          cudaMemcpyDeviceToHost));

    std::cout << "hbin_offsets" << std::endl;
    for(int i = 0; i < bin_count + 1; i++)
    {
        std::cout << hbin_offsets[i] << " ";
    }
    std::cout << "" << std::endl;

    const int bin_0_count = hbin_offsets[1] - hbin_offsets[0];
    const int bin_1_count = hbin_offsets[2] - hbin_offsets[1];
    const int bin_2_count = hbin_offsets[3] - hbin_offsets[2];
    const int bin_3_count = hbin_offsets[4] - hbin_offsets[3];
    const int bin_4_count = hbin_offsets[5] - hbin_offsets[4];
    const int bin_5_count = hbin_offsets[6] - hbin_offsets[5];
    const int bin_6_count = hbin_offsets[7] - hbin_offsets[6];
    const int bin_7_count = hbin_offsets[8] - hbin_offsets[7];

    CHECK_CUDA(cudaMemset(csr_row_ptr_C, 0, sizeof(int) * (m + 1)));

    if(bin_0_count > 0)
    {
        std::cout << "bin_0_count: " << bin_0_count << std::endl;
        csrgeam_nnz_per_row_kernel<256, 32, 32>
            <<<((bin_0_count - 1) / (256 / 32) + 1), 256>>>(bin_0_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[0]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            beta,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_row_ptr_C);
    }
    if(bin_1_count > 0)
    {
        std::cout << "bin_1_count: " << bin_1_count << std::endl;
        csrgeam_nnz_per_row_kernel<256, 32, 64>
            <<<((bin_1_count - 1) / (256 / 32) + 1), 256>>>(bin_1_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[1]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            beta,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_row_ptr_C);
    }
    if(bin_2_count > 0)
    {
        std::cout << "bin_2_count: " << bin_2_count << std::endl;
        csrgeam_nnz_per_row_kernel<256, 32, 128>
            <<<((bin_2_count - 1) / (256 / 32) + 1), 256>>>(bin_2_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[2]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            beta,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_row_ptr_C);
    }
    if(bin_3_count > 0)
    {
        std::cout << "bin_3_count: " << bin_3_count << std::endl;
        csrgeam_nnz_per_row_kernel<256, 32, 256>
            <<<((bin_3_count - 1) / (256 / 32) + 1), 256>>>(bin_3_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[3]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            beta,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_row_ptr_C);
    }
    if(bin_4_count > 0)
    {
        std::cout << "bin_4_count: " << bin_4_count << std::endl;
        csrgeam_nnz_per_row_kernel<256, 32, 512>
            <<<((bin_4_count - 1) / (256 / 32) + 1), 256>>>(bin_4_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[4]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            beta,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_row_ptr_C);
    }
    if(bin_5_count > 0)
    {
        std::cout << "bin_5_count: " << bin_5_count << std::endl;
        csrgeam_nnz_per_row_kernel<256, 32, 1024>
            <<<((bin_5_count - 1) / (256 / 32) + 1), 256>>>(bin_5_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[5]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            beta,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_row_ptr_C);
    }
    if(bin_6_count > 0)
    {
        std::cout << "bin_6_count: " << bin_6_count << std::endl;
        csrgeam_nnz_per_row_kernel<128, 32, 2048>
            <<<((bin_6_count - 1) / (128 / 32) + 1), 128>>>(bin_6_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[6]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            beta,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_row_ptr_C);
    }
    if(bin_7_count > 0)
    {
        std::cout << "bin_7_count: " << bin_7_count << std::endl;
        csrgeam_nnz_per_row_kernel<64, 32, 4096>
            <<<((bin_7_count - 1) / (64 / 32) + 1), 64>>>(bin_7_count,
                                                          descr->perm,
                                                          &(descr->bin_offsets[7]),
                                                          alpha,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          beta,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          csr_row_ptr_C);
    }
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    cuda_exclusive_scan(m + 1, csr_row_ptr_C);

    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));

    std::cout << "csr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;

    CHECK_CUDA(cudaMemcpy(nnz_C, csr_row_ptr_C + m, sizeof(int), cudaMemcpyDeviceToHost));
}

void linalg::cuda_csrgeam_solve(int                  m,
                                int                  n,
                                int                  nnz_A,
                                int                  nnz_B,
                                int                  nnz_C,
                                const csrgeam_descr* descr,
                                double               alpha,
                                const int*           csr_row_ptr_A,
                                const int*           csr_col_ind_A,
                                const double*        csr_val_A,
                                double               beta,
                                const int*           csr_row_ptr_B,
                                const int*           csr_col_ind_B,
                                const double*        csr_val_B,
                                const int*           csr_row_ptr_C,
                                int*                 csr_col_ind_C,
                                double*              csr_val_C)
{
    std::cout << "nnz_C: " << nnz_C << std::endl;

    int              bin_count = 8;
    std::vector<int> hbin_offsets(bin_count + 1, 0);
    CHECK_CUDA(cudaMemcpy(hbin_offsets.data(),
                          descr->bin_offsets,
                          sizeof(int) * (bin_count + 1),
                          cudaMemcpyDeviceToHost));

    std::cout << "hbin_offsets" << std::endl;
    for(int i = 0; i < bin_count + 1; i++)
    {
        std::cout << hbin_offsets[i] << " ";
    }
    std::cout << "" << std::endl;

    const int bin_0_count = hbin_offsets[1] - hbin_offsets[0];
    const int bin_1_count = hbin_offsets[2] - hbin_offsets[1];
    const int bin_2_count = hbin_offsets[3] - hbin_offsets[2];
    const int bin_3_count = hbin_offsets[4] - hbin_offsets[3];
    const int bin_4_count = hbin_offsets[5] - hbin_offsets[4];
    const int bin_5_count = hbin_offsets[6] - hbin_offsets[5];
    const int bin_6_count = hbin_offsets[7] - hbin_offsets[6];
    const int bin_7_count = hbin_offsets[8] - hbin_offsets[7];

    if(bin_0_count > 0)
    {
        std::cout << "bin_0_count: " << bin_0_count << std::endl;
        csrgeam_fill_kernel<256, 32, 32>
            <<<((bin_0_count - 1) / (256 / 32) + 1), 256>>>(bin_0_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[0]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            beta,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_1_count > 0)
    {
        std::cout << "bin_1_count: " << bin_1_count << std::endl;
        csrgeam_fill_kernel<256, 32, 64>
            <<<((bin_1_count - 1) / (256 / 32) + 1), 256>>>(bin_1_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[1]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            beta,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_2_count > 0)
    {
        std::cout << "bin_2_count: " << bin_2_count << std::endl;
        csrgeam_fill_kernel<256, 32, 128>
            <<<((bin_2_count - 1) / (256 / 32) + 1), 256>>>(bin_2_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[2]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            beta,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_3_count > 0)
    {
        std::cout << "bin_3_count: " << bin_3_count << std::endl;
        csrgeam_fill_kernel<256, 32, 256>
            <<<((bin_3_count - 1) / (256 / 32) + 1), 256>>>(bin_3_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[3]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            beta,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_4_count > 0)
    {
        std::cout << "bin_4_count: " << bin_4_count << std::endl;
        csrgeam_fill_kernel<256, 32, 512>
            <<<((bin_4_count - 1) / (256 / 32) + 1), 256>>>(bin_4_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[4]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            beta,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_5_count > 0)
    {
        std::cout << "bin_5_count: " << bin_5_count << std::endl;
        csrgeam_fill_kernel<128, 32, 1024>
            <<<((bin_5_count - 1) / (128 / 32) + 1), 128>>>(bin_5_count,
                                                            descr->perm,
                                                            &(descr->bin_offsets[5]),
                                                            alpha,
                                                            csr_row_ptr_A,
                                                            csr_col_ind_A,
                                                            csr_val_A,
                                                            beta,
                                                            csr_row_ptr_B,
                                                            csr_col_ind_B,
                                                            csr_val_B,
                                                            csr_row_ptr_C,
                                                            csr_col_ind_C,
                                                            csr_val_C);
    }
    if(bin_6_count > 0)
    {
        std::cout << "bin_6_count: " << bin_6_count << std::endl;
        csrgeam_fill_kernel<64, 32, 2048>
            <<<((bin_6_count - 1) / (64 / 32) + 1), 64>>>(bin_6_count,
                                                          descr->perm,
                                                          &(descr->bin_offsets[6]),
                                                          alpha,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          csr_val_A,
                                                          beta,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          csr_val_B,
                                                          csr_row_ptr_C,
                                                          csr_col_ind_C,
                                                          csr_val_C);
    }
    if(bin_7_count > 0)
    {
        std::cout << "bin_7_count: " << bin_7_count << std::endl;
        csrgeam_fill_kernel<32, 32, 4096>
            <<<((bin_7_count - 1) / (32 / 32) + 1), 32>>>(bin_7_count,
                                                          descr->perm,
                                                          &(descr->bin_offsets[7]),
                                                          alpha,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          csr_val_A,
                                                          beta,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          csr_val_B,
                                                          csr_row_ptr_C,
                                                          csr_col_ind_C,
                                                          csr_val_C);
    }
    CHECK_CUDA_LAUNCH_ERROR();

    std::vector<int>    hcsr_row_ptr_C(m + 1, 0);
    std::vector<int>    hcsr_col_ind_C(nnz_C, 0);
    std::vector<double> hcsr_val_C(nnz_C);
    CHECK_CUDA(cudaMemcpy(
        hcsr_row_ptr_C.data(), csr_row_ptr_C, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(
        hcsr_col_ind_C.data(), csr_col_ind_C, sizeof(int) * nnz_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(hcsr_val_C.data(), csr_val_C, sizeof(double) * nnz_C, cudaMemcpyDeviceToHost));

    std::cout << "hcsr_row_ptr_C" << std::endl;
    for(int i = 0; i < m + 1; i++)
    {
        std::cout << hcsr_row_ptr_C[i] << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "hcsr_col_ind_C" << std::endl;
    for(int i = 0; i < nnz_C; i++)
    {
        std::cout << hcsr_col_ind_C[i] << " ";
    }
    std::cout << "" << std::endl;
    std::cout << "hcsr_val_C" << std::endl;
    for(int i = 0; i < nnz_C; i++)
    {
        std::cout << hcsr_val_C[i] << " ";
    }
    std::cout << "" << std::endl;
}

//-------------------------------------------------------------------------------
// Compute Incomplete Cholesky IC0: A = L * L^T
//-------------------------------------------------------------------------------
struct linalg::csric0_descr
{
    int* done_array;
    int* row_perm;
    int* diag_ind;
};

void linalg::allocate_csric0_cuda_data(csric0_descr* descr)
{
    std::cout << "allocate_csric0_cuda_data" << std::endl;
    descr->done_array = nullptr;
    descr->row_perm   = nullptr;
    descr->diag_ind   = nullptr;
}

void linalg::free_csric0_cuda_data(csric0_descr* descr)
{
    if(descr != nullptr)
    {
        if(descr->done_array != nullptr)
        {
            std::cout << "Freeing done_array" << std::endl;
            CHECK_CUDA(cudaFree(descr->done_array));
        }

        if(descr->row_perm != nullptr)
        {
            std::cout << "Freeing row_perm" << std::endl;
            CHECK_CUDA(cudaFree(descr->row_perm));
        }

        if(descr->diag_ind != nullptr)
        {
            std::cout << "Freeing diag_ind" << std::endl;
            CHECK_CUDA(cudaFree(descr->diag_ind));
        }
    }
}

void linalg::cuda_csric0_analysis(int           m,
                                  int           n,
                                  int           nnz,
                                  const int*    csr_row_ptr,
                                  const int*    csr_col_ind,
                                  const double* csr_val,
                                  csric0_descr* descr)
{
    std::cout << "cuda_csric0_analysis m: " << m << " n: " << n << " nnz: " << nnz << std::endl;

    // Free any previous allocations?
    assert(descr->done_array == nullptr);
    assert(descr->row_perm == nullptr);
    assert(descr->diag_ind == nullptr);

    descr->done_array = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&(descr->done_array), sizeof(int) * m));
    CHECK_CUDA(cudaMemset(descr->done_array, 0, sizeof(int) * m));

    descr->row_perm = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&(descr->row_perm), sizeof(int) * m));

    descr->diag_ind = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&(descr->diag_ind), sizeof(int) * m));

    csrtrsv_analysis_kernel<256, 32><<<((m - 1) / (256 / 32) + 1), 256>>>(m,
                                                                          triangular_type::lower,
                                                                          csr_row_ptr,
                                                                          csr_col_ind,
                                                                          csr_val,
                                                                          descr->diag_ind,
                                                                          descr->done_array);
    CHECK_CUDA_LAUNCH_ERROR();

    std::vector<int> hdiag_ind(m, 10);
    CHECK_CUDA(
        cudaMemcpy(hdiag_ind.data(), descr->diag_ind, sizeof(int) * m, cudaMemcpyDeviceToHost));

    std::cout << "diag_ind" << std::endl;
    for(int i = 0; i < m; i++)
    {
        std::cout << hdiag_ind[i] << " ";
    }
    std::cout << "" << std::endl;

    std::vector<int> hdone_array(m, 0);
    CHECK_CUDA(
        cudaMemcpy(hdone_array.data(), descr->done_array, sizeof(int) * m, cudaMemcpyDeviceToHost));

    std::cout << "done_array" << std::endl;
    for(int i = 0; i < m; i++)
    {
        std::cout << hdone_array[i] << " ";
    }
    std::cout << "" << std::endl;

    fill_identity_permuation_kernel<256><<<((m - 1) / 256 + 1), 256>>>(m, descr->row_perm);
    CHECK_CUDA_LAUNCH_ERROR();

    // Wrap Raw Pointers and Execute Thrust Algorithm
    // thrust::device_ptr allows us to treat a raw pointer like a Thrust iterator.
    thrust::device_ptr<int> d_keys(descr->done_array);
    thrust::device_ptr<int> d_values(descr->row_perm);

    // Use sort_by_key: sorts d_keys and applies the identical permutation to d_values
    thrust::sort_by_key(d_keys, d_keys + m, d_values);

    std::vector<int> hrow_perm(m, 0);
    CHECK_CUDA(
        cudaMemcpy(hrow_perm.data(), descr->row_perm, sizeof(int) * m, cudaMemcpyDeviceToHost));

    std::cout << "row_perm" << std::endl;
    for(int i = 0; i < m; i++)
    {
        std::cout << hrow_perm[i] << " ";
    }
    std::cout << "" << std::endl;
}

void linalg::cuda_csric0_compute(int                 m,
                                 int                 n,
                                 int                 nnz,
                                 const int*          csr_row_ptr,
                                 const int*          csr_col_ind,
                                 double*             csr_val,
                                 const csric0_descr* descr)
{
    assert(descr->diag_ind != nullptr);
    assert(descr->done_array != nullptr);
    assert(descr->row_perm != nullptr);

    CHECK_CUDA(cudaMemset(descr->done_array, 0, sizeof(int) * m));

    std::cout << "cuda_csric0_compute m: " << m << " n: " << n << " nnz: " << nnz << std::endl;
    csric0_solve_kernel<256, 32, 128><<<((m - 1) / (256 / 32) + 1), 256>>>(
        m, csr_row_ptr, csr_col_ind, csr_val, descr->diag_ind, descr->done_array, descr->row_perm);
    CHECK_CUDA_LAUNCH_ERROR();
}

//-------------------------------------------------------------------------------
// Compute Incomplete LU ILU0: A = L * U
//-------------------------------------------------------------------------------
struct linalg::csrilu0_descr
{
    int* done_array;
    int* row_perm;
    int* diag_ind;
};

void linalg::allocate_csrilu0_cuda_data(csrilu0_descr* descr)
{
    descr->done_array = nullptr;
    descr->row_perm   = nullptr;
    descr->diag_ind   = nullptr;
}

void linalg::free_csrilu0_cuda_data(csrilu0_descr* descr)
{
    if(descr != nullptr)
    {
        if(descr->done_array != nullptr)
        {
            std::cout << "Freeing done_array" << std::endl;
            CHECK_CUDA(cudaFree(descr->done_array));
        }

        if(descr->row_perm != nullptr)
        {
            std::cout << "Freeing row_perm" << std::endl;
            CHECK_CUDA(cudaFree(descr->row_perm));
        }

        if(descr->diag_ind != nullptr)
        {
            std::cout << "Freeing diag_ind" << std::endl;
            CHECK_CUDA(cudaFree(descr->diag_ind));
        }
    }
}

void linalg::cuda_csrilu0_analysis(int            m,
                                   int            n,
                                   int            nnz,
                                   const int*     csr_row_ptr,
                                   const int*     csr_col_ind,
                                   const double*  csr_val,
                                   csrilu0_descr* descr)
{
    std::cout << "csrtrsv_analysis m: " << m << " n: " << n << " nnz: " << nnz << std::endl;

    // Free any previous allocations?
    assert(descr->done_array == nullptr);
    assert(descr->row_perm == nullptr);
    assert(descr->diag_ind == nullptr);

    descr->done_array = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&(descr->done_array), sizeof(int) * m));
    CHECK_CUDA(cudaMemset(descr->done_array, 0, sizeof(int) * m));

    descr->row_perm = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&(descr->row_perm), sizeof(int) * m));

    descr->diag_ind = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&(descr->diag_ind), sizeof(int) * m));

    csrtrsv_analysis_kernel<256, 32><<<((m - 1) / (256 / 32) + 1), 256>>>(m,
                                                                          triangular_type::lower,
                                                                          csr_row_ptr,
                                                                          csr_col_ind,
                                                                          csr_val,
                                                                          descr->diag_ind,
                                                                          descr->done_array);
    CHECK_CUDA_LAUNCH_ERROR();

    // std::vector<int> hdiag_ind(m, 10);
    // CHECK_CUDA(
    //     cudaMemcpy(hdiag_ind.data(), descr->diag_ind, sizeof(int) * m, cudaMemcpyDeviceToHost));

    // std::cout << "diag_ind" << std::endl;
    // for(int i = 0; i < m; i++)
    // {
    //     std::cout << hdiag_ind[i] << " ";
    // }
    // std::cout << "" << std::endl;

    // std::vector<int> hdone_array(m, 0);
    // CHECK_CUDA(
    //     cudaMemcpy(hdone_array.data(), descr->done_array, sizeof(int) * m, cudaMemcpyDeviceToHost));

    // std::cout << "done_array" << std::endl;
    // for(int i = 0; i < m; i++)
    // {
    //     std::cout << hdone_array[i] << " ";
    // }
    // std::cout << "" << std::endl;

    fill_identity_permuation_kernel<256><<<((m - 1) / 256 + 1), 256>>>(m, descr->row_perm);
    CHECK_CUDA_LAUNCH_ERROR();

    // Wrap Raw Pointers and Execute Thrust Algorithm
    // thrust::device_ptr allows us to treat a raw pointer like a Thrust iterator.
    thrust::device_ptr<int> d_keys(descr->done_array);
    thrust::device_ptr<int> d_values(descr->row_perm);

    // Use sort_by_key: sorts d_keys and applies the identical permutation to d_values
    thrust::sort_by_key(d_keys, d_keys + m, d_values);

    // std::vector<int> hrow_perm(m, 0);
    // CHECK_CUDA(
    //     cudaMemcpy(hrow_perm.data(), descr->row_perm, sizeof(int) * m, cudaMemcpyDeviceToHost));

    // std::cout << "row_perm" << std::endl;
    // for(int i = 0; i < m; i++)
    // {
    //     std::cout << hrow_perm[i] << " ";
    // }
    // std::cout << "" << std::endl;
}

void linalg::cuda_csrilu0_compute(int                  m,
                                  int                  n,
                                  int                  nnz,
                                  const int*           csr_row_ptr,
                                  const int*           csr_col_ind,
                                  double*              csr_val,
                                  const csrilu0_descr* descr)
{
    assert(descr->diag_ind != nullptr);
    assert(descr->done_array != nullptr);
    assert(descr->row_perm != nullptr);

    CHECK_CUDA(cudaMemset(descr->done_array, 0, sizeof(int) * m));
}

void linalg::cuda_tridiagonal_solver(int          m,
                                     int          n,
                                     const float* lower_diag,
                                     const float* main_diag,
                                     const float* upper_diag,
                                     const float* b,
                                     float*       x)
{
    if(m == 2)
    {
        thomas_algorithm_kernel<256, 2>
            <<<((n - 1) / 256 + 1), 256>>>(n, lower_diag, main_diag, upper_diag, b, x);
    }
    else if(m == 3)
    {
        thomas_algorithm_kernel<256, 3>
            <<<((n - 1) / 256 + 1), 256>>>(n, lower_diag, main_diag, upper_diag, b, x);
    }
    else if(m == 4)
    {
        thomas_algorithm_kernel<256, 4>
            <<<((n - 1) / 256 + 1), 256>>>(n, lower_diag, main_diag, upper_diag, b, x);
    }
    else if(m == 5)
    {
        thomas_algorithm_kernel<256, 5>
            <<<((n - 1) / 256 + 1), 256>>>(n, lower_diag, main_diag, upper_diag, b, x);
    }
    else if(m == 6)
    {
        thomas_algorithm_kernel<256, 6>
            <<<((n - 1) / 256 + 1), 256>>>(n, lower_diag, main_diag, upper_diag, b, x);
    }
    else if(m == 7)
    {
        thomas_algorithm_kernel<256, 7>
            <<<((n - 1) / 256 + 1), 256>>>(n, lower_diag, main_diag, upper_diag, b, x);
    }
    else if(m == 8)
    {
        thomas_shared_transpose_kernel2<256, 32, 8, 4>
            <<<((n - 1) / 256 + 1), 256>>>(m, n, lower_diag, main_diag, upper_diag, b, x);
        //thomas_algorithm_kernel<256, 8>
        //    <<<((n - 1) / 256 + 1), 256>>>(n, lower_diag, main_diag, upper_diag, b, x);
    }
    // else if(m == 65536)
    // {
    //     std::cout << "cuda_tridiagonal_solver m: " << m << " n: " << n << std::endl;
    //     // Configuration
    //     const int N          = 65536;
    //     const int BLOCKSIZE  = 256;
    //     const int num_blocks = N / BLOCKSIZE;
    //     const int num_spikes = 2 * num_blocks; // 512
    //     const int num_levels = 7; // log2(256) - 1

    //     float* d_l_pyramid         = nullptr;
    //     float* d_m_pyramid         = nullptr;
    //     float* d_u_pyramid         = nullptr;
    //     float* d_r_pyramid         = nullptr;
    //     float* d_l_spike           = nullptr;
    //     float* d_m_spike           = nullptr;
    //     float* d_u_spike           = nullptr;
    //     float* d_x_spike           = nullptr;
    //     float* d_x_boundary_solved = nullptr;

    //     CHECK_CUDA(cudaMalloc((void**)&d_l_pyramid, sizeof(float) * N * num_levels));
    //     CHECK_CUDA(cudaMalloc((void**)&d_m_pyramid, sizeof(float) * N * num_levels));
    //     CHECK_CUDA(cudaMalloc((void**)&d_u_pyramid, sizeof(float) * N * num_levels));
    //     CHECK_CUDA(cudaMalloc((void**)&d_r_pyramid, sizeof(float) * N * num_levels));
    //     CHECK_CUDA(cudaMalloc((void**)&d_l_spike, sizeof(float) * num_spikes));
    //     CHECK_CUDA(cudaMalloc((void**)&d_m_spike, sizeof(float) * num_spikes));
    //     CHECK_CUDA(cudaMalloc((void**)&d_u_spike, sizeof(float) * num_spikes));
    //     CHECK_CUDA(cudaMalloc((void**)&d_x_spike, sizeof(float) * num_spikes));
    //     CHECK_CUDA(cudaMalloc((void**)&d_x_boundary_solved, sizeof(float) * 2 * num_blocks));

    //     // 1. Forward Sweep
    //     // pyramid_idx uses (level * N), so ensure pyramid arrays are N * num_levels in size
    //     cr_forward_sweep_kernel<BLOCKSIZE, float><<<num_blocks, BLOCKSIZE>>>(N,
    //                                                                          lower_diag,
    //                                                                          main_diag,
    //                                                                          upper_diag,
    //                                                                          b,
    //                                                                          d_l_pyramid,
    //                                                                          d_m_pyramid,
    //                                                                          d_u_pyramid,
    //                                                                          d_r_pyramid,
    //                                                                          d_l_spike,
    //                                                                          d_m_spike,
    //                                                                          d_u_spike,
    //                                                                          d_x_spike);

    //     // 2. Spike Solver (PCR)
    //     // This solves the global boundary dependencies in a single block
    //     size_t spike_smem = 4 * num_spikes * sizeof(float);
    //     spike_solver_pcr_kernel<float><<<1, num_spikes, spike_smem>>>(
    //         num_spikes,
    //         d_l_spike,
    //         d_m_spike,
    //         d_u_spike,
    //         d_x_spike,
    //         d_x_boundary_solved // Output: corrected X values for 0, 255, 256...
    //     );

    //     // 3. Backward Sweep
    //     // Uses the solved boundaries and the pyramid to fill in the middle
    //     cr_backward_sweep_kernel<BLOCKSIZE, float><<<num_blocks, BLOCKSIZE>>>(
    //         N, d_l_pyramid, d_m_pyramid, d_u_pyramid, d_r_pyramid, d_x_boundary_solved, x);

    //     // Cleanup
    //     CHECK_CUDA(cudaFree(d_l_pyramid));
    //     CHECK_CUDA(cudaFree(d_m_pyramid));
    //     CHECK_CUDA(cudaFree(d_u_pyramid));
    //     CHECK_CUDA(cudaFree(d_r_pyramid));
    //     CHECK_CUDA(cudaFree(d_l_spike));
    //     CHECK_CUDA(cudaFree(d_m_spike));
    //     CHECK_CUDA(cudaFree(d_u_spike));
    //     CHECK_CUDA(cudaFree(d_x_spike));
    //     CHECK_CUDA(cudaFree(d_x_boundary_solved));
    // }
    else if(m % 8 == 0)
    {
        //std::cout << "cuda_tridiagonal_solver m: " << m << " n: " << n << std::endl;
        constexpr int BLOCKSIZE      = 128;
        constexpr int GROUPSIZE      = 128;
        constexpr int WAVEFRONT_SIZE = 32;
        constexpr int M              = 128;

        // std::vector<float> htemp_a(M);
        // std::vector<float> htemp_b(M);
        // std::vector<float> htemp_c(M);
        // std::vector<float> htemp_d(M);
        float* dtemp_a = nullptr;
        float* dtemp_b = nullptr;
        float* dtemp_c = nullptr;
        float* dtemp_d = nullptr;
        // CHECK_CUDA(cudaMalloc((void**)&dtemp_a, sizeof(float) * M));
        // CHECK_CUDA(cudaMalloc((void**)&dtemp_b, sizeof(float) * M));
        // CHECK_CUDA(cudaMalloc((void**)&dtemp_c, sizeof(float) * M));
        // CHECK_CUDA(cudaMalloc((void**)&dtemp_d, sizeof(float) * M));

        // thomas_pcr_wavefront_kernel<BLOCKSIZE, WAVEFRONT_SIZE, M><<<((n - 1) / (BLOCKSIZE / WAVEFRONT_SIZE) + 1), BLOCKSIZE>>>(
        //    m, n, lower_diag, main_diag, upper_diag, b, x, dtemp_a, dtemp_b, dtemp_c, dtemp_d);
        //thomas_pcr_wavefront_kernel2<BLOCKSIZE, WAVEFRONT_SIZE, M><<<((n - 1) / (BLOCKSIZE / WAVEFRONT_SIZE) + 1), BLOCKSIZE>>>(
        //    m, n, lower_diag, main_diag, upper_diag, b, x, dtemp_a, dtemp_b, dtemp_c, dtemp_d);
        // thomas_pcr_multiple_wavefront_kernel<BLOCKSIZE, GROUPSIZE, WAVEFRONT_SIZE, M>
        //     <<<((n - 1) / (BLOCKSIZE / GROUPSIZE) + 1), BLOCKSIZE>>>(
        //         m, n, lower_diag, main_diag, upper_diag, b, x, dtemp_a, dtemp_b, dtemp_c, dtemp_d);
        //pcr_shared_kernel<BLOCKSIZE, GROUPSIZE, WAVEFRONT_SIZE, M>
        //    <<<((n - 1) / (BLOCKSIZE / GROUPSIZE) + 1), BLOCKSIZE>>>(
        //        m, n, lower_diag, main_diag, upper_diag, b, x, dtemp_a, dtemp_b, dtemp_c, dtemp_d);

        pcr_shared_kernel2<BLOCKSIZE, WAVEFRONT_SIZE, M, 8><<<((n - 1) / 8 + 1), BLOCKSIZE>>>(
            m, n, lower_diag, main_diag, upper_diag, b, x, dtemp_a, dtemp_b, dtemp_c, dtemp_d);
        //thomas_shared_transpose_kernel2<256, 32, 32, 1>
        //    <<<((n - 1) / 256 + 1), 256>>>(m, n, lower_diag, main_diag, upper_diag, b, x);
        //thomas_algorithm_kernel<256, 16>
        //    <<<((n - 1) / 256 + 1), 256>>>(n, lower_diag, main_diag, upper_diag, b, x);

        // CHECK_CUDA(cudaMemcpy(htemp_a.data(), dtemp_a, sizeof(float) * M, cudaMemcpyDeviceToHost));
        // CHECK_CUDA(cudaMemcpy(htemp_b.data(), dtemp_b, sizeof(float) * M, cudaMemcpyDeviceToHost));
        // CHECK_CUDA(cudaMemcpy(htemp_c.data(), dtemp_c, sizeof(float) * M, cudaMemcpyDeviceToHost));
        // CHECK_CUDA(cudaMemcpy(htemp_d.data(), dtemp_d, sizeof(float) * M, cudaMemcpyDeviceToHost));

        // std::cout << "htemp_a" << std::endl;
        // for(int i = 0; i < M; i++)
        // {
        //     std::cout << htemp_a[i] << " ";
        // }
        // std::cout << "" << std::endl;
        // std::cout << "htemp_b" << std::endl;
        // for(int i = 0; i < M; i++)
        // {
        //     std::cout << htemp_b[i] << " ";
        // }
        // std::cout << "" << std::endl;
        // std::cout << "htemp_c" << std::endl;
        // for(int i = 0; i < M; i++)
        // {
        //     std::cout << htemp_c[i] << " ";
        // }
        // std::cout << "" << std::endl;
        // std::cout << "htemp_d" << std::endl;
        // for(int i = 0; i < M; i++)
        // {
        //     std::cout << htemp_d[i] << " ";
        // }
        // std::cout << "" << std::endl;

        // CHECK_CUDA(cudaFree(dtemp_a));
        // CHECK_CUDA(cudaFree(dtemp_b));
        // CHECK_CUDA(cudaFree(dtemp_c));
        // CHECK_CUDA(cudaFree(dtemp_d));
    }
    else
    {
        std::cerr << "Error: cuda_tridiagonal_solver only supports m = 2 to 8." << std::endl;
        return;
    }

    CHECK_CUDA_LAUNCH_ERROR();
}
