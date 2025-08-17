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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
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
#include "device_math.h"
#include "device_memory.h"
#include <iostream>

#include "device_math_impl.h"

#include "../../trace.h"

// Compute y = alpha * x + y
void linalg::device_axpy(double alpha, const vector<double>& x, vector<double>& y)
{
    ROUTINE_TRACE("linalg::device_axpy");
    device_axpy_impl(x.get_size(), alpha, x.get_vec(), y.get_vec());
}

// Compute y = alpha * x + beta * y
void linalg::device_axpby(double alpha, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::device_axpby");
    device_axpby_impl(x.get_size(), alpha, x.get_vec(), beta, y.get_vec());
}

// Compute z = alpha * x + beta * y + gamma * z
void linalg::device_axpbypgz(double                alpha,
                             const vector<double>& x,
                             double                beta,
                             const vector<double>& y,
                             double                gamma,
                             vector<double>&       z)
{
    ROUTINE_TRACE("linalg::device_axpbypgz");
    device_axpbypgz_impl(x.get_size(), alpha, x.get_vec(), beta, y.get_vec(), gamma, z.get_vec());
}

// Compute y = A * x
void linalg::device_matrix_vector_product(const csr_matrix&     A,
                                          const vector<double>& x,
                                          vector<double>&       y)
{
    ROUTINE_TRACE("linalg::device_multiply_by_vector");
    device_csrmv_impl(A.get_m(),
                      A.get_n(),
                      A.get_nnz(),
                      1.0,
                      A.get_row_ptr(),
                      A.get_col_ind(),
                      A.get_val(),
                      x.get_vec(),
                      0.0,
                      y.get_vec());
}

// Compute y = alpha * A * x + beta * y
void linalg::device_matrix_vector_product(
    double alpha, const csr_matrix& A, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::device_multiply_by_vector");
    device_csrmv_impl(A.get_m(),
                      A.get_n(),
                      A.get_nnz(),
                      alpha,
                      A.get_row_ptr(),
                      A.get_col_ind(),
                      A.get_val(),
                      x.get_vec(),
                      beta,
                      y.get_vec());
}

// Compute C = A * B
void linalg::device_matrix_matrix_product(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    std::cout << "Error: matrix_matrix_product on device not implemented" << std::endl;
}

// Compute C = A + B
void linalg::device_matrix_matrix_addition(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    std::cout << "Error: matrix_matrix_addition on device not implemented" << std::endl;
}

// Incomplete IC factorization
void linalg::device_csric0(csr_matrix& LL, int* structural_zero, int* numeric_zero)
{
    std::cout << "Error: csric0 on device not implemented" << std::endl;
}

// Incomplete LU factorization
void linalg::device_csrilu0(csr_matrix& LU, int* structural_zero, int* numeric_zero)
{
    std::cout << "Error: csrilu0 on device not implemented" << std::endl;
}

// Forward solve
void linalg::device_forward_solve(const csr_matrix&     A,
                                  const vector<double>& b,
                                  vector<double>&       x,
                                  bool                  unit_diag)
{
    std::cout << "Error: forward_solve on device not implemented" << std::endl;
}

// Backward solve
void linalg::device_backward_solve(const csr_matrix&     A,
                                   const vector<double>& b,
                                   vector<double>&       x,
                                   bool                  unit_diag)
{
    std::cout << "Error: backward_solve on device not implemented" << std::endl;
}

// Transpose matrix
void linalg::device_transpose_matrix(const csr_matrix& A, csr_matrix& transposeA)
{
    std::cout << "Error: transpose_matrix on device not implemented" << std::endl;
}

// Dot product
double linalg::device_dot_product(const vector<double>& x, const vector<double>& y)
{
    ROUTINE_TRACE("linalg::device_dot_product");
    return device_dot_product_impl(x.get_vec(), y.get_vec(), x.get_size());
}

// Compute residual
void linalg::device_compute_residual(const csr_matrix&     A,
                                     const vector<double>& x,
                                     const vector<double>& b,
                                     vector<double>&       res)
{
    ROUTINE_TRACE("linalg::device_compute_residual");
    device_compute_residual_impl(A.get_m(),
                                 A.get_n(),
                                 A.get_nnz(),
                                 A.get_row_ptr(),
                                 A.get_col_ind(),
                                 A.get_val(),
                                 x.get_vec(),
                                 b.get_vec(),
                                 res.get_vec());
}

// Exclusive scan
void linalg::device_exclusive_scan(vector<int64_t>& x)
{
    std::cout << "Error: exclusive_scan on device not implemented" << std::endl;
}

// Extract diagonal entries
void linalg::device_diagonal(const csr_matrix& A, vector<double>& d)
{
    ROUTINE_TRACE("linalg::device_diagonal");
    device_extract_diagonal_impl(A.get_m(),
                                 A.get_n(),
                                 A.get_nnz(),
                                 A.get_row_ptr(),
                                 A.get_col_ind(),
                                 A.get_val(),
                                 d.get_vec());
}

// Euclidean norm
double linalg::device_norm_euclid(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::device_norm_euclid");
    return std::sqrt(device_dot_product(array, array));
}

// Infinity norm
double linalg::device_norm_inf(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::device_norm_inf");
    return device_norm_inf_impl(array.get_vec(), array.get_size());
}

// Jacobi solve
void linalg::device_jacobi_solve(const vector<double>& rhs,
                                 const vector<double>& diag,
                                 vector<double>&       x)
{
    ROUTINE_TRACE("linalg::device_jacobi_solve");
    device_jacobi_solve_impl(rhs.get_vec(), diag.get_vec(), x.get_vec(), rhs.get_size());
}
