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

#include "../include/linalg_math.h"
#include <assert.h>
#include <iostream>

#include "trace.h"
#include "utility.h"

#include "backend/device/device_math.h"
#include "backend/host/host_math.h"

// Compute y = alpha * x + y
void linalg::axpy(double alpha, const vector<double>& x, vector<double>& y)
{
    ROUTINE_TRACE("linalg::axpy");

    backend_dispatch("linalg::axpy", host_axpy, device_axpy, alpha, x, y);
}

// Compute y = alpha * x + beta * y
void linalg::axpby(double alpha, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::axpby");

    backend_dispatch("linalg::axpby", host_axpby, device_axpby, alpha, x, beta, y);
}

// Compute z = alpha * x + beta * y + gamma * z
void linalg::axpbypgz(double                alpha,
                      const vector<double>& x,
                      double                beta,
                      const vector<double>& y,
                      double                gamma,
                      vector<double>&       z)
{
    ROUTINE_TRACE("linalg::axpbypgz");

    backend_dispatch(
        "linalg::axpbypgz", host_axpbypgz, device_axpbypgz, alpha, x, beta, y, gamma, z);
}

// Compute y = A * x
void linalg::matrix_vector_product(const csr_matrix& A, const vector<double>& x, vector<double>& y)
{
    ROUTINE_TRACE("linalg::matrix_vector_product");

    auto host_function = [](const csr_matrix& A, const vector<double>& x, vector<double>& y) {
        return host_matrix_vector_product(A, x, y);
    };
    auto device_function = [](const csr_matrix& A, const vector<double>& x, vector<double>& y) {
        return device_matrix_vector_product(A, x, y);
    };

    backend_dispatch("linalg::matrix_vector_product", host_function, device_function, A, x, y);
}

// Compute y = alpha * A * x + beta * y
void linalg::matrix_vector_product(
    double alpha, const csr_matrix& A, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::matrix_vector_product");

    auto host_function
        = [](double                alpha,
             const csr_matrix&     A,
             const vector<double>& x,
             double                beta,
             vector<double>&       y) { return host_matrix_vector_product(alpha, A, x, beta, y); };
    auto device_function
        = [](double                alpha,
             const csr_matrix&     A,
             const vector<double>& x,
             double                beta,
             vector<double>& y) { return device_matrix_vector_product(alpha, A, x, beta, y); };

    backend_dispatch(
        "linalg::matrix_vector_product", host_function, device_function, alpha, A, x, beta, y);
}

// Compute C = A * B
void linalg::matrix_matrix_product(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    ROUTINE_TRACE("linalg::matrix_matrix_product");

    backend_dispatch("linalg::matrix_matrix_product",
                     host_matrix_matrix_product,
                     device_matrix_matrix_product,
                     C,
                     A,
                     B);
}

// Compute C = A + B
void linalg::matrix_matrix_addition(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    ROUTINE_TRACE("linalg::matrix_matrix_addition");

    std::cout << "Entering linalg::matrix_matrix_addition" << std::endl;
    backend_dispatch("linalg::matrix_matrix_addition",
                     host_matrix_matrix_addition,
                     device_matrix_matrix_addition,
                     C,
                     A,
                     B);
}

// Incomplete IC factorization
void linalg::csric0(csr_matrix& LL, int* structural_zero, int* numeric_zero)
{
    ROUTINE_TRACE("linalg::csric0");

    backend_dispatch(
        "linalg::csric0", host_csric0, device_csric0, LL, structural_zero, numeric_zero);
}

// Incomplete LU factorization
void linalg::csrilu0(csr_matrix& LU, int* structural_zero, int* numeric_zero)
{
    ROUTINE_TRACE("linalg::csrilu0");

    backend_dispatch(
        "linalg::csrilu0", host_csrilu0, device_csrilu0, LU, structural_zero, numeric_zero);
}

// Forward solve
void linalg::forward_solve(const csr_matrix&     A,
                           const vector<double>& b,
                           vector<double>&       x,
                           bool                  unit_diag)
{
    ROUTINE_TRACE("linalg::forward_solve");

    backend_dispatch(
        "linalg::forward_solve", host_forward_solve, device_forward_solve, A, b, x, unit_diag);
}

// Backward solve
void linalg::backward_solve(const csr_matrix&     A,
                            const vector<double>& b,
                            vector<double>&       x,
                            bool                  unit_diag)
{
    ROUTINE_TRACE("linalg::backward_solve");

    backend_dispatch(
        "linalg::backward_solve", host_backward_solve, device_backward_solve, A, b, x, unit_diag);
}

// Transpose matrix
void linalg::transpose_matrix(const csr_matrix& A, csr_matrix& transposeA)
{
    ROUTINE_TRACE("linalg::transpose_matrix");

    backend_dispatch(
        "linalg::transpose_matrix", host_transpose_matrix, device_transpose_matrix, A, transposeA);
}

// Dot product
double linalg::dot_product(const vector<double>& x, const vector<double>& y)
{
    ROUTINE_TRACE("linalg::dot_product");

    return backend_dispatch("linalg::dot_product", host_dot_product, device_dot_product, x, y);
}

// Compute residual
void linalg::compute_residual(const csr_matrix&     A,
                              const vector<double>& x,
                              const vector<double>& b,
                              vector<double>&       res)
{
    ROUTINE_TRACE("linalg::compute_residual");

    backend_dispatch(
        "linalg::compute_residual", host_compute_residual, device_compute_residual, A, x, b, res);
}

// Extract diagonal entries
void linalg::diagonal(const csr_matrix& A, vector<double>& d)
{
    ROUTINE_TRACE("linalg::diagonal");

    backend_dispatch("linalg::diagonal", host_diagonal, device_diagonal, A, d);
}

// Euclidean norm
double linalg::norm_euclid(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::norm_euclid");

    return backend_dispatch("linalg::norm_euclid", host_norm_euclid, device_norm_euclid, array);
}

// Infinity norm
double linalg::norm_inf(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::norm_inf");

    return backend_dispatch("linalg::norm_inf", host_norm_inf, device_norm_inf, array);
}

// Jacobi solve
void linalg::jacobi_solve(const vector<double>& rhs, const vector<double>& diag, vector<double>& x)
{
    ROUTINE_TRACE("linalg::jacobi_solve");

    return backend_dispatch(
        "linalg::jacobi_solve", host_jacobi_solve, device_jacobi_solve, rhs, diag, x);
}