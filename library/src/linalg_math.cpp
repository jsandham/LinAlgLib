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

    backend bend = determine_backend(x, y);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::axpy must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(x.is_on_host())
    {
        return host::axpy(alpha, x, y);
    }
    else
    {
        return device::axpy(alpha, x, y);
    }
}

// Compute y = alpha * x + beta * y
void linalg::axpby(double alpha, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::axpby");

    backend bend = determine_backend(x, y);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::axpby must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(x.is_on_host())
    {
        return host::axpby(alpha, x, beta, y);
    }
    else
    {
        return device::axpby(alpha, x, beta, y);
    }
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

    backend bend = determine_backend(x, y, z);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::axpbypgz must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(x.is_on_host())
    {
        return host::axpbypgz(alpha, x, beta, y, gamma, z);
    }
    else
    {
        return device::axpbypgz(alpha, x, beta, y, gamma, z);
    }
}

// Compute y = A * x
void linalg::matrix_vector_product(const csr_matrix& A, const vector<double>& x, vector<double>& y)
{
    ROUTINE_TRACE("linalg::matrix_vector_product");

    backend bend = determine_backend(A, x, y);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::matrix_vector_product must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(A.is_on_host())
    {
        return host::matrix_vector_product(A, x, y);
    }
    else
    {
        return device::matrix_vector_product(A, x, y);
    }
}

// Compute y = alpha * A * x + beta * y
void linalg::matrix_vector_product(
    double alpha, const csr_matrix& A, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::matrix_vector_product");

    backend bend = determine_backend(A, x, y);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::matrix_vector_product must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(A.is_on_host())
    {
        return host::matrix_vector_product(alpha, A, x, beta, y);
    }
    else
    {
        return device::matrix_vector_product(alpha, A, x, beta, y);
    }
}

// Compute C = A * B
void linalg::matrix_matrix_product(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    ROUTINE_TRACE("linalg::matrix_matrix_product");

    backend bend = determine_backend(C, A, B);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::matrix_matrix_product must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(C.is_on_host())
    {
        return host::matrix_matrix_product(C, A, B);
    }
    else
    {
        return device::matrix_matrix_product(C, A, B);
    }
}

// Compute C = A + B
void linalg::matrix_matrix_addition(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    ROUTINE_TRACE("linalg::matrix_matrix_addition");

    backend bend = determine_backend(C, A, B);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::matrix_matrix_addition must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(C.is_on_host())
    {
        return host::matrix_matrix_addition(C, A, B);
    }
    else
    {
        return device::matrix_matrix_addition(C, A, B);
    }
}

// Incomplete IC factorization
void linalg::csric0(csr_matrix& LL, int* structural_zero, int* numeric_zero)
{
    ROUTINE_TRACE("linalg::csric0");

    backend bend = determine_backend(LL);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::csric0 must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(LL.is_on_host())
    {
        return host::csric0(LL, structural_zero, numeric_zero);
    }
    else
    {
        return device::csric0(LL, structural_zero, numeric_zero);
    }
}

// Incomplete LU factorization
void linalg::csrilu0(csr_matrix& LU, int* structural_zero, int* numeric_zero)
{
    ROUTINE_TRACE("linalg::csrilu0");

    backend bend = determine_backend(LU);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::csrilu0 must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(LU.is_on_host())
    {
        return host::csrilu0(LU, structural_zero, numeric_zero);
    }
    else
    {
        return device::csrilu0(LU, structural_zero, numeric_zero);
    }
}

// Forward solve
void linalg::forward_solve(const csr_matrix&     A,
                           const vector<double>& b,
                           vector<double>&       x,
                           bool                  unit_diag)
{
    ROUTINE_TRACE("linalg::forward_solve");

    backend bend = determine_backend(A, b, x);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::forward_solve must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(A.is_on_host())
    {
        return host::forward_solve(A, b, x, unit_diag);
    }
    else
    {
        return device::forward_solve(A, b, x, unit_diag);
    }
}

// Backward solve
void linalg::backward_solve(const csr_matrix&     A,
                            const vector<double>& b,
                            vector<double>&       x,
                            bool                  unit_diag)
{
    ROUTINE_TRACE("linalg::backward_solve");

    backend bend = determine_backend(A, b, x);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::backward_solve must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(A.is_on_host())
    {
        return host::backward_solve(A, b, x, unit_diag);
    }
    else
    {
        return device::backward_solve(A, b, x, unit_diag);
    }
}

// Transpose matrix
void linalg::transpose_matrix(const csr_matrix& A, csr_matrix& transposeA)
{
    ROUTINE_TRACE("linalg::transpose_matrix");

    backend bend = determine_backend(A, transposeA);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::transpose_matrix must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(A.is_on_host())
    {
        return host::transpose_matrix(A, transposeA);
    }
    else
    {
        return device::transpose_matrix(A, transposeA);
    }
}

// Dot product
double linalg::dot_product(const vector<double>& x, const vector<double>& y)
{
    ROUTINE_TRACE("linalg::dot_product");

    backend bend = determine_backend(x, y);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::dot_product must all be on host or "
                     "all be on device"
                  << std::endl;
        return 0.0;
    }

    if(x.is_on_host())
    {
        return host::dot_product(x, y);
    }
    else
    {
        return device::dot_product(x, y);
    }
}

// Compute residual
void linalg::compute_residual(const csr_matrix&     A,
                              const vector<double>& x,
                              const vector<double>& b,
                              vector<double>&       res)
{
    ROUTINE_TRACE("linalg::compute_residual");

    backend bend = determine_backend(A, x, b, res);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::compute_residual must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(A.is_on_host())
    {
        return host::compute_residual(A, x, b, res);
    }
    else
    {
        return device::compute_residual(A, x, b, res);
    }
}

// Exclusive scan
void linalg::exclusive_scan(vector<double>& x)
{
    ROUTINE_TRACE("linalg::exclusive_scan");

    backend bend = determine_backend(x);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::exclusive_scan must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(x.is_on_host())
    {
        return host::exclusive_scan(x);
    }
    else
    {
        return device::exclusive_scan(x);
    }
}

// Extract diagonal entries
void linalg::diagonal(const csr_matrix& A, vector<double>& d)
{
    ROUTINE_TRACE("linalg::diagonal");

    backend bend = determine_backend(A, d);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::diagonal must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(A.is_on_host())
    {
        return host::diagonal(A, d);
    }
    else
    {
        return device::diagonal(A, d);
    }
}

// Euclidean norm
double linalg::norm_euclid(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::norm_euclid");

    backend bend = determine_backend(array);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::norm_euclid must all be on host or "
                     "all be on device"
                  << std::endl;
        return 0.0;
    }

    if(array.is_on_host())
    {
        return host::norm_euclid(array);
    }
    else
    {
        return device::norm_euclid(array);
    }
}

// Infinity norm
double linalg::norm_inf(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::norm_inf");

    backend bend = determine_backend(array);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::norm_inf must all be on host or "
                     "all be on device"
                  << std::endl;
        return 0.0;
    }

    if(array.is_on_host())
    {
        return host::norm_inf(array);
    }
    else
    {
        return device::norm_inf(array);
    }
}

// Fill array with value
template <typename T>
void linalg::fill(vector<T>& vec, T value)
{
    ROUTINE_TRACE("linalg::fill<T>");

    backend bend = determine_backend(vec);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::fill<T> must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(vec.is_on_host())
    {
        return host::fill(vec, value);
    }
    else
    {
        return device::fill(vec, value);
    }
}

template void linalg::fill<uint32_t>(vector<uint32_t>& vec, uint32_t value);
template void linalg::fill<int32_t>(vector<int32_t>& vec, int32_t value);
template void linalg::fill<int64_t>(vector<int64_t>& vec, int64_t value);
template void linalg::fill<double>(vector<double>& vec, double value);

// Copy array
template <typename T>
void linalg::copy(vector<T>& dest, const vector<T>& src)
{
    ROUTINE_TRACE("linalg::copy<T>");

    backend bend = determine_backend(dest, src);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::copy must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(dest.is_on_host())
    {
        return host::copy(dest, src);
    }
    else
    {
        return device::copy(dest, src);
    }
}

template void linalg::copy<uint32_t>(vector<uint32_t>& dest, const vector<uint32_t>& src);
template void linalg::copy<int32_t>(vector<int32_t>& dest, const vector<int32_t>& src);
template void linalg::copy<int64_t>(vector<int64_t>& dest, const vector<int64_t>& src);
template void linalg::copy<double>(vector<double>& dest, const vector<double>& src);

// Jacobi solve
void linalg::jacobi_solve(const vector<double>& rhs, const vector<double>& diag, vector<double>& x)
{
    ROUTINE_TRACE("linalg::jacobi_solve");

    backend bend = determine_backend(rhs, diag, x);

    if(bend != backend::host && bend != backend::device)
    {
        std::cout << "Error: parameters to linalg::jacobi_solve must all be on host or "
                     "all be on device"
                  << std::endl;
        return;
    }

    if(rhs.is_on_host())
    {
        return host::jacobi_solve(rhs, diag, x);
    }
    else
    {
        return device::jacobi_solve(rhs, diag, x);
    }
}