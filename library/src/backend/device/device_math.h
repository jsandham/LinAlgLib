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

#ifndef DEVICE_MATH_H
#define DEVICE_MATH_H

#include <string>

#include "csr_matrix.h"
#include "linalg_export.h"
#include "vector.h"

namespace linalg
{
    // Compute y = alpha * x + y
    void device_axpy(double alpha, const vector<double>& x, vector<double>& y);

    // Compute y = alpha * x + beta * y
    void device_axpby(double alpha, const vector<double>& x, double beta, vector<double>& y);

    // Compute z = alpha * x + beta * y + gamma * z
    void device_axpbypgz(double                alpha,
                         const vector<double>& x,
                         double                beta,
                         const vector<double>& y,
                         double                gamma,
                         vector<double>&       z);

    // Compute y = A * x
    void device_matrix_vector_product(const csr_matrix&     A,
                                      const vector<double>& x,
                                      vector<double>&       y);

    // Compute y = alpha * A * x + beta * y
    void device_matrix_vector_product(
        double alpha, const csr_matrix& A, const vector<double>& x, double beta, vector<double>& y);

    // Compute C = A * B
    void device_matrix_matrix_product(csr_matrix& C, const csr_matrix& A, const csr_matrix& B);

    // Compute C = A + B
    void device_matrix_matrix_addition(csr_matrix& C, const csr_matrix& A, const csr_matrix& B);

    // Incomplete IC factorization
    void device_csric0(csr_matrix& LL, int* structural_zero, int* numeric_zero);

    // Incomplete LU factorization
    void device_csrilu0(csr_matrix& LU, int* structural_zero, int* numeric_zero);

    // Forward solve
    void device_forward_solve(const csr_matrix&     A,
                              const vector<double>& b,
                              vector<double>&       x,
                              bool                  unit_diag);

    // Backward solve
    void device_backward_solve(const csr_matrix&     A,
                               const vector<double>& b,
                               vector<double>&       x,
                               bool                  unit_diag);

    // Transpose matrix
    void device_transpose_matrix(const csr_matrix& A, csr_matrix& transposeA);

    // Dot product
    double device_dot_product(const vector<double>& x, const vector<double>& y);

    // Compute residual
    void device_compute_residual(const csr_matrix&     A,
                                 const vector<double>& x,
                                 const vector<double>& b,
                                 vector<double>&       res);

    // Extract diagonal entries
    void device_diagonal(const csr_matrix& A, vector<double>& d);

    // Euclidean norm
    double device_norm_euclid(const vector<double>& array);

    // Infinity norm
    double device_norm_inf(const vector<double>& array);

    // Jacobi solve
    void device_jacobi_solve(const vector<double>& rhs,
                             const vector<double>& diag,
                             vector<double>&       x);

} // namespace linalg

#endif