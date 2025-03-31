//********************************************************************************
//
// MIT License
//
// Copyright(c) 2019 James Sandham
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

// Compute y = alpha * x + y
void axpy(int n, double alpha, const double* x, double* y);

// Compute y = alpha * x + beta * y
void axpby(int n, double alpha, const double* x, double beta, double* y);

// Compute y = alpha * A * x + beta * y
void csrmv(int m, int n, int nnz, double alpha, const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val,
           const double *x, double beta, double *y);

// Compute C = alpha * A * B + beta * D
void csrgemm_nnz(int m, int n, int k, int nnz_A, int nnz_B, int nnz_D, double alpha, const int *csr_row_ptr_A,
                 const int *csr_col_ind_A, const int *csr_row_ptr_B, const int *csr_col_ind_B, double beta,
                 const int *csr_row_ptr_D, const int *csr_col_ind_D, int *csr_row_ptr_C, int *nnz_C);

void csrgemm(int m, int n, int k, int nnz_A, int nnz_B, int nnz_D, double alpha, const int *csr_row_ptr_A,
             const int *csr_col_ind_A, const double *csr_val_A, const int *csr_row_ptr_B, const int *csr_col_ind_B,
             const double *csr_val_B, double beta, const int *csr_row_ptr_D, const int *csr_col_ind_D,
             const double *csr_val_D, const int *csr_row_ptr_C, int *csr_col_ind_C, double *csr_val_C);

// Compute C = alpha * A + beta * B
void csrgeam_nnz(int m, int n, int nnz_A, int nnz_B, double alpha, const int *csr_row_ptr_A, const int *csr_col_ind_A,
                 double beta, const int *csr_row_ptr_B, const int *csr_col_ind_B, int *csr_row_ptr_C, int *nnz_C);

void csrgeam(int m, int n, int nnz_A, int nnz_B, double alpha, const int *csr_row_ptr_A, const int *csr_col_ind_A,
             const double *csr_val_A, double beta, const int *csr_row_ptr_B, const int *csr_col_ind_B,
             const double *csr_val_B, const int *csr_row_ptr_C, int *csr_col_ind_C, double *csr_val_C);

// Compute incomplete LU factorization inplace
void csrilu0(int m, int n, int nnz, const int *csr_row_ptr, const int *csr_col_ind, double *csr_val,
             int *structural_zero, int *numeric_zero);

// Compute incomplete Cholesky factorization inplace (only modifies lower triangular part)
void csric0(int m, int n, int nnz, const int *csr_row_ptr, const int *csr_col_ind, double *csr_val,
            int *structural_zero, int *numeric_zero);

// Compute y = A * x
void matrix_vector_product(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
                           double *y, const int n);

// Compute result = x * y
double dot_product(const double *x, const double *y, int n);

// Compute residual res = b - A * x
void compute_residual(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
                      const double* b, double* res, int n);

// Copy array
void copy(double* dest, const double* src, int n);

// Extract diagonal entries
void diagonal(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *d, int n);

// Solve Lx = b where L is a lower triangular sparse matrix
void forward_solve(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *b, double *x,
                   int n, bool unit_diag = false);

// Solve Ux = b where U is a upper triangular sparse matrix
void backward_solve(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *b, double *x,
                    int n, bool unit_diag = false);

double error(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x, const double *b,
             int n);

double fast_error(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, const double *x,
                  const double *b, int n, double tol);

double norm_euclid(const double *array, int n);
double norm_inf(const double *array, int n);

void print(const std::string name, const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n,
           int nnz);

#endif
