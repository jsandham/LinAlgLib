//********************************************************************************
//
// MIT License
//
// Copyright(c) 2019-2025 James Sandham
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

#ifndef SLAF_H
#define SLAF_H

#include <string>

#include "../linalglib_export.h"

// Compute y = alpha * x + y
LINALGLIB_API void axpy(int n, double alpha, const double* x, double* y);

// Compute y = alpha * x + beta * y
LINALGLIB_API void axpby(int n, double alpha, const double* x, double beta, double* y);

// Compute y = alpha * A * x + beta * y
LINALGLIB_API void csrmv(int m, int n, int nnz, double alpha, const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val,
           const double *x, double beta, double *y);

// Compute C = alpha * A * B + beta * D
LINALGLIB_API void csrgemm_nnz(int m, int n, int k, int nnz_A, int nnz_B, int nnz_D, double alpha, const int *csr_row_ptr_A,
                 const int *csr_col_ind_A, const int *csr_row_ptr_B, const int *csr_col_ind_B, double beta,
                 const int *csr_row_ptr_D, const int *csr_col_ind_D, int *csr_row_ptr_C, int *nnz_C);

LINALGLIB_API void csrgemm(int m, int n, int k, int nnz_A, int nnz_B, int nnz_D, double alpha, const int *csr_row_ptr_A,
             const int *csr_col_ind_A, const double *csr_val_A, const int *csr_row_ptr_B, const int *csr_col_ind_B,
             const double *csr_val_B, double beta, const int *csr_row_ptr_D, const int *csr_col_ind_D,
             const double *csr_val_D, const int *csr_row_ptr_C, int *csr_col_ind_C, double *csr_val_C);

// Compute C = alpha * A + beta * B
LINALGLIB_API void csrgeam_nnz(int m, int n, int nnz_A, int nnz_B, double alpha, const int *csr_row_ptr_A, const int *csr_col_ind_A,
                 double beta, const int *csr_row_ptr_B, const int *csr_col_ind_B, int *csr_row_ptr_C, int *nnz_C);

LINALGLIB_API void csrgeam(int m, int n, int nnz_A, int nnz_B, double alpha, const int *csr_row_ptr_A, const int *csr_col_ind_A,
             const double *csr_val_A, double beta, const int *csr_row_ptr_B, const int *csr_col_ind_B,
             const double *csr_val_B, const int *csr_row_ptr_C, int *csr_col_ind_C, double *csr_val_C);

// Compute incomplete LU factorization inplace
LINALGLIB_API void csrilu0(int m, int n, int nnz, const int *csr_row_ptr, const int *csr_col_ind, double *csr_val,
             int *structural_zero, int *numeric_zero);

// Compute incomplete Cholesky factorization inplace (only modifies lower triangular part)
LINALGLIB_API void csric0(int m, int n, int nnz, const int *csr_row_ptr, const int *csr_col_ind, double *csr_val,
            int *structural_zero, int *numeric_zero);

// Compute y = A * x
LINALGLIB_API void matrix_vector_product(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
                           double *y, const int n);

// Compute result = x * y
LINALGLIB_API double dot_product(const double *x, const double *y, int n);

// Compute residual res = b - A * x
LINALGLIB_API void compute_residual(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
                      const double* b, double* res, int n);

// Copy array
LINALGLIB_API void copy(double* dest, const double* src, int n);

// Extract diagonal entries
LINALGLIB_API void diagonal(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *d, int n);

// Solve Lx = b where L is a lower triangular sparse matrix
LINALGLIB_API void forward_solve(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *b, double *x,
                   int n, bool unit_diag = false);

// Solve Ux = b where U is a upper triangular sparse matrix
LINALGLIB_API void backward_solve(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *b, double *x,
                    int n, bool unit_diag = false);

LINALGLIB_API double error(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x, const double *b,
             int n);

LINALGLIB_API double fast_error(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
                  const double *b, int n, double tol);

LINALGLIB_API double norm_euclid(const double *array, int n);
LINALGLIB_API double norm_inf(const double *array, int n);

LINALGLIB_API void print_matrix(const std::string name, const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n,
           int nnz);

#endif
