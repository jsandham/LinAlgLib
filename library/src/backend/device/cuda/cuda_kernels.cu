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

#include "axpby_kernels.h"
#include "compute_residual_kernels.h"
#include "csrmv_kernels.h"
#include "dot_product_kernels.h"
#include "extract_diagonal_kernels.h"
#include "find_minmax_kernels.h"
#include "jacobi_solve_kernels.h"

#include "cuda_kernels.h"

#include "../../../trace.h"

template <typename T>
void launch_cuda_fill_kernel(T* data, size_t size, T val)
{
    ROUTINE_TRACE("launch_cuda_fill_kernel");
    fill_kernel<256><<<((size - 1) / 256 + 1), 256>>>(data, size, val);

    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_copy_kernel(T* dest, const T* src, size_t size)
{
    ROUTINE_TRACE("launch_cuda_copy_kernel");
    copy_kernel<256><<<((size - 1) / 256 + 1), 256>>>(dest, src, size);

    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_dot_product_kernel(int size, const T* x, const T* y, T* result)
{
    ROUTINE_TRACE("launch_cuda_dot_product_kernel");
    T* workspace = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&workspace, sizeof(T) * 256));

    dot_product_kernel_part1<256><<<256, 256>>>(size, x, y, workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    dot_product_kernel_part2<256><<<1, 256>>>(workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaMemcpy(result, workspace, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(workspace));
}

template <typename T>
void launch_cuda_norm_inf_kernel(int size, const T* x, T* result)
{
    ROUTINE_TRACE("launch_cuda_norm_inf_kernel");
    T* workspace = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&workspace, sizeof(T) * 256));

    find_max_kernel_part1<256><<<256, 256>>>(size, x, workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    find_max_kernel_part2<256><<<1, 256>>>(workspace);
    CHECK_CUDA_LAUNCH_ERROR();

    CHECK_CUDA(cudaMemcpy(result, workspace, sizeof(T), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(workspace));
}

template <typename T>
void launch_cuda_csrmv_kernel(int        m,
                              int        n,
                              int        nnz,
                              const T    alpha,
                              const int* csr_row_ptr,
                              const int* csr_col_ind,
                              const T*   csr_val,
                              const T*   x,
                              const T    beta,
                              T*         y)
{
    ROUTINE_TRACE("launch_cuda_csrmv_kernel");

    csrmv_vector_kernel<256, 4><<<((m - 1) / (256 / 4) + 1), 256>>>(
        m, n, nnz, alpha, csr_row_ptr, csr_col_ind, csr_val, x, beta, y);
    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_extract_diagonal_kernel(int        m,
                                         int        n,
                                         int        nnz,
                                         const int* csr_row_ptr,
                                         const int* csr_col_ind,
                                         const T*   csr_val,
                                         T*         diag)
{
    ROUTINE_TRACE("launch_cuda_extract_diagonal_kernel");
    extract_diagonal_kernel<256, 4>
        <<<((m - 1) / (256 / 4) + 1), 256>>>(m, n, nnz, csr_row_ptr, csr_col_ind, csr_val, diag);
    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_compute_residual_kernel(int        m,
                                         int        n,
                                         int        nnz,
                                         const int* csr_row_ptr,
                                         const int* csr_col_ind,
                                         const T*   csr_val,
                                         const T*   x,
                                         const T*   b,
                                         T*         res)
{
    ROUTINE_TRACE("launch_cuda_compute_residual_kernel");
    compute_residual_kernel<256, 4><<<((m - 1) / (256 / 4) + 1), 256>>>(
        m, n, nnz, csr_row_ptr, csr_col_ind, csr_val, x, b, res);
    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_axpy_kernel(int size, T alpha, const T* x, T* y)
{
    ROUTINE_TRACE("launch_cuda_axpy_kernel");
    axpy_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, alpha, x, y);
    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_axpby_kernel(int size, T alpha, const T* x, T beta, T* y)
{
    ROUTINE_TRACE("launch_cuda_axpby_kernel");
    axpby_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, alpha, x, beta, y);
    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_axpbypgz_kernel(int size, T alpha, const T* x, T beta, const T* y, T gamma, T* z)
{
    ROUTINE_TRACE("launch_cuda_axpbypgz_kernel");
    axpbypgz_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, alpha, x, beta, y, gamma, z);
    CHECK_CUDA_LAUNCH_ERROR();
}

template <typename T>
void launch_cuda_jacobi_solve_kernel(int size, const T* rhs, const T* diag, T* x)
{
    ROUTINE_TRACE("launch_cuda_jacobi_solve_kernel");
    jacobi_solve_kernel<256><<<((size - 1) / 256 + 1), 256>>>(size, rhs, diag, x);
    CHECK_CUDA_LAUNCH_ERROR();
}

template void launch_cuda_fill_kernel<uint32_t>(uint32_t* data, size_t size, uint32_t val);
template void launch_cuda_fill_kernel<int32_t>(int32_t* data, size_t size, int32_t val);
template void launch_cuda_fill_kernel<int64_t>(int64_t* data, size_t size, int64_t val);
template void launch_cuda_fill_kernel<double>(double* data, size_t size, double val);

template void launch_cuda_copy_kernel<uint32_t>(uint32_t* dest, const uint32_t* src, size_t size);
template void launch_cuda_copy_kernel<int32_t>(int32_t* dest, const int32_t* src, size_t size);
template void launch_cuda_copy_kernel<int64_t>(int64_t* dest, const int64_t* src, size_t size);
template void launch_cuda_copy_kernel<double>(double* dest, const double* src, size_t size);

template void launch_cuda_dot_product_kernel<uint32_t>(int             size,
                                                       const uint32_t* x,
                                                       const uint32_t* y,
                                                       uint32_t*       result);
template void launch_cuda_dot_product_kernel<int32_t>(int            size,
                                                      const int32_t* x,
                                                      const int32_t* y,
                                                      int32_t*       result);
template void launch_cuda_dot_product_kernel<int64_t>(int            size,
                                                      const int64_t* x,
                                                      const int64_t* y,
                                                      int64_t*       result);
template void launch_cuda_dot_product_kernel<double>(int           size,
                                                     const double* x,
                                                     const double* y,
                                                     double*       result);

//template void launch_cuda_norm_inf_kernel<uint32_t>(int size, const uint32_t* x, uint32_t* result);
template void launch_cuda_norm_inf_kernel<int32_t>(int size, const int32_t* x, int32_t* result);
template void launch_cuda_norm_inf_kernel<int64_t>(int size, const int64_t* x, int64_t* result);
template void launch_cuda_norm_inf_kernel<double>(int size, const double* x, double* result);

template void launch_cuda_csrmv_kernel<double>(int           m,
                                               int           n,
                                               int           nnz,
                                               const double  alpha,
                                               const int*    csr_row_ptr,
                                               const int*    csr_col_ind,
                                               const double* csr_val,
                                               const double* x,
                                               const double  beta,
                                               double*       y);

template void launch_cuda_extract_diagonal_kernel<double>(int           m,
                                                          int           n,
                                                          int           nnz,
                                                          const int*    csr_row_ptr,
                                                          const int*    csr_col_ind,
                                                          const double* csr_val,
                                                          double*       diag);

template void launch_cuda_compute_residual_kernel<double>(int           m,
                                                          int           n,
                                                          int           nnz,
                                                          const int*    csr_row_ptr,
                                                          const int*    csr_col_ind,
                                                          const double* csr_val,
                                                          const double* x,
                                                          const double* b,
                                                          double*       res);

template void launch_cuda_axpy_kernel<double>(int size, double alpha, const double* x, double* y);
template void launch_cuda_axpby_kernel<double>(
    int size, double alpha, const double* x, double beta, double* y);
template void launch_cuda_axpbypgz_kernel<double>(
    int size, double alpha, const double* x, double beta, const double* y, double gamma, double* z);

template void launch_cuda_jacobi_solve_kernel<double>(int           size,
                                                      const double* rhs,
                                                      const double* diag,
                                                      double*       x);
