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

#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

template <typename T>
void launch_cuda_fill_kernel(T* data, size_t size, T val);

template <typename T>
void launch_cuda_copy_kernel(T* dest, const T* src, size_t size);

template <typename T>
void launch_cuda_dot_product_kernel(int size, const T* x, const T* y, T* result);

template <typename T>
void launch_cuda_norm_inf_kernel(int size, const T* array, T* norm);

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
                              T*         y);

template <typename T>
void launch_cuda_extract_diagonal_kernel(int        m,
                                         int        n,
                                         int        nnz,
                                         const int* csr_row_ptr,
                                         const int* csr_col_ind,
                                         const T*   csr_val,
                                         T*         diag);

template <typename T>
void launch_cuda_compute_residual_kernel(int        m,
                                         int        n,
                                         int        nnz,
                                         const int* csr_row_ptr,
                                         const int* csr_col_ind,
                                         const T*   csr_val,
                                         const T*   x,
                                         const T*   b,
                                         T*         res);

template <typename T>
void launch_cuda_axpy_kernel(int size, T alpha, const T* x, T* y);

template <typename T>
void launch_cuda_axpby_kernel(int size, T alpha, const T* x, T beta, T* y);

template <typename T>
void launch_cuda_axpbypgz_kernel(int size, T alpha, const T* x, T beta, const T* y, T gamma, T* z);

template <typename T>
void launch_cuda_jacobi_solve_kernel(int size, const T* rhs, const T* diag, T* x);

#endif
