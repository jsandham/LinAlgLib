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

#include "backend/host/host_math.h"
#include "backend/device/device_math.h"

// Compute y = alpha * x + y
void linalg::axpy(double alpha, const vector<double>& x, vector<double>& y)
{
    if(x.is_on_host() != y.is_on_host())
    {
        std::cout << "Error (axpy): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(x.is_on_host())
    {
        host::axpy(alpha, x, y);
    }
    else
    {
        device::axpy(alpha, x, y);
    }
}

void linalg::axpy(const scalar<double>& alpha, const vector<double>& x, vector<double>& y)
{
    if(x.is_on_host() != y.is_on_host() || x.is_on_host() != alpha.is_on_host())
    {
        std::cout << "Error (axpy): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(x.is_on_host())
    {
        host::axpy(alpha, x, y);
    }
    else
    {
        device::axpy(alpha, x, y);
    }
}

// Compute y = alpha * x + beta * y
void linalg::axpby(double alpha, const vector<double>& x, double beta, vector<double>& y)
{
    if(x.is_on_host() != y.is_on_host())
    {
        std::cout << "Error (axpby): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(x.is_on_host())
    {
        host::axpby(alpha, x, beta, y);
    }
    else
    {
        device::axpby(alpha, x, beta, y);
    }
}
void linalg::axpby(const scalar<double>& alpha, const vector<double>& x, const scalar<double>& beta, vector<double>& y)
{
    if(x.is_on_host() != y.is_on_host() || x.is_on_host() != alpha.is_on_host() || x.is_on_host() != beta.is_on_host())
    {
        std::cout << "Error (axpby): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(x.is_on_host())
    {
        host::axpby(alpha, x, beta, y);
    }
    else
    {
        device::axpby(alpha, x, beta, y);
    }
}

// Compute z = alpha * x + beta * y + gamma * z
void linalg::axpbypgz(double alpha, const vector<double>& x, double beta, const vector<double>& y, double gamma, vector<double>& z)
{
    if(x.is_on_host() != y.is_on_host() || x.is_on_host() != z.is_on_host())
    {
        std::cout << "Error (axpbypgz): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(x.is_on_host())
    {
        host::axpbypgz(alpha, x, beta, y, gamma, z);
    }
    else
    {
        device::axpbypgz(alpha, x, beta, y, gamma, z);
    }
}
void linalg::axpbypgz(const scalar<double>& alpha, const vector<double>& x, const scalar<double>& beta, const vector<double>& y, const scalar<double>& gamma, vector<double>& z)
{
    if(x.is_on_host() != y.is_on_host() || 
       x.is_on_host() != z.is_on_host() ||
       x.is_on_host() != alpha.is_on_host() || 
       x.is_on_host() != beta.is_on_host() ||
       x.is_on_host() != gamma.is_on_host())
    {
        std::cout << "Error (axpbypgz): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(x.is_on_host())
    {
        host::axpbypgz(alpha, x, beta, y, gamma, z);
    }
    else
    {
        device::axpbypgz(alpha, x, beta, y, gamma, z);
    }
}

// Compute y = A * x
void linalg::matrix_vector_product(const csr_matrix& A, const vector<double>& x, vector<double>&y)
{
    if(A.is_on_host() != x.is_on_host() || A.is_on_host() != y.is_on_host())
    {
        std::cout << "Error (matrix_vector_product): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(A.is_on_host())
    {
        host::matrix_vector_product(A, x, y);
    }
    else
    {
        device::matrix_vector_product(A, x, y);
    }
}

// Compute y = alpha * A * x + beta * y
void linalg::matrix_vector_product(double alpha, const csr_matrix& A, const vector<double>& x, double beta, vector<double>&y)
{
    if(A.is_on_host() != x.is_on_host() || A.is_on_host() != y.is_on_host())
    {
        std::cout << "Error (matrix_vector_product): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(A.is_on_host())
    {
        host::matrix_vector_product(alpha, A, x, beta, y);
    }
    else
    {
        device::matrix_vector_product(alpha, A, x, beta, y);
    }
}

// Compute C = A * B
void linalg::matrix_matrix_product(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    if(C.is_on_host() != A.is_on_host() || C.is_on_host() != B.is_on_host())
    {
        std::cout << "Error (matrix_matrix_product): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(C.is_on_host())
    {
        host::matrix_matrix_product(C, A, B);
    }
    else
    {
        device::matrix_matrix_product(C, A, B);
    }
}

// Compute C = A + B
void linalg::matrix_matrix_addition(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    if(C.is_on_host() != A.is_on_host() || C.is_on_host() != B.is_on_host())
    {
        std::cout << "Error (matrix_matrix_addition): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(C.is_on_host())
    {
        host::matrix_matrix_addition(C, A, B);
    }
    else
    {
        device::matrix_matrix_addition(C, A, B);
    }
}

// Incomplete IC factorization
void linalg::csric0(csr_matrix& LL, int* structural_zero, int* numeric_zero)
{
    if(LL.is_on_host())
    {
        host::csric0(LL, structural_zero, numeric_zero);
    }
    else
    {
        device::csric0(LL, structural_zero, numeric_zero);
    }
}

// Incomplete LU factorization
void linalg::csrilu0(csr_matrix& LU, int* structural_zero, int* numeric_zero)
{
    if(LU.is_on_host())
    {
        host::csrilu0(LU, structural_zero, numeric_zero);
    }
    else
    {
        device::csrilu0(LU, structural_zero, numeric_zero);
    }
}

// Forward solve
void linalg::forward_solve(const csr_matrix& A, const vector<double>& b, vector<double>& x, bool unit_diag)
{
    if(A.is_on_host() != b.is_on_host() || A.is_on_host() != x.is_on_host())
    {
        std::cout << "Error (forward_solve): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(A.is_on_host())
    {
        host::forward_solve(A, b, x, unit_diag);
    }
    else
    {
        device::forward_solve(A, b, x, unit_diag);
    }
}

// Backward solve
void linalg::backward_solve(const csr_matrix& A, const vector<double>& b, vector<double>& x, bool unit_diag)
{
    if(A.is_on_host() != b.is_on_host() || A.is_on_host() != x.is_on_host())
    {
        std::cout << "Error (backward_solve): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(A.is_on_host())
    {
        host::backward_solve(A, b, x, unit_diag);
    }
    else
    {
        device::backward_solve(A, b, x, unit_diag);
    }
}

// Transpose matrix
void linalg::transpose_matrix(const csr_matrix &A, csr_matrix &transposeA)
{
    if(A.is_on_host() != transposeA.is_on_host())
    {
        std::cout << "Error (transpose_matrix): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(A.is_on_host())
    {
        host::transpose_matrix(A, transposeA);
    }
    else
    {
        device::transpose_matrix(A, transposeA);
    }
}

// Dot product
double linalg::dot_product(const vector<double>& x, const vector<double>& y)
{
    if(x.is_on_host() != y.is_on_host())
    {
        std::cout << "Error (dot_product): Mixing host and device inputs not supported. Skipping computation." << std::endl;
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



// Dot product
void linalg::dot_product(const vector<double>& x, const vector<double>& y, scalar<double>& result)
{
    if(x.is_on_host() != y.is_on_host() || x.is_on_host() != result.is_on_host())
    {
        std::cout << "Error (dot_product): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(x.is_on_host())
    {
        host::dot_product(x, y, result);
    }
    else
    {
        device::dot_product(x, y, result);
    }
}

// Compute residual
void linalg::compute_residual(const csr_matrix& A, const vector<double>& x, const vector<double>& b, vector<double>& res)
{
    if(A.is_on_host() != x.is_on_host() || A.is_on_host() != b.is_on_host() || A.is_on_host() != res.is_on_host())
    {
        std::cout << "Error (compute_residual): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(A.is_on_host())
    {
        host::compute_residual(A, x, b, res);
    }
    else
    {
        device::compute_residual(A, x, b, res);
    }
}

// Exclusive scan
void linalg::exclusize_scan(vector<double>& x)
{
    if(x.is_on_host())
    {
        host::exclusize_scan(x);
    }
    else
    {
        device::exclusize_scan(x);
    }
}

// Extract diagonal entries
void linalg::diagonal(const csr_matrix& A, vector<double>& d)
{
    if(A.is_on_host() != d.is_on_host())
    {
        std::cout << "Error (diagonal): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(A.is_on_host())
    {
        host::diagonal(A, d);
    }
    else
    {
        device::diagonal(A, d);
    }
}

// Euclidean norm
double linalg::norm_euclid(const vector<double>& array)
{
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
    if(array.is_on_host())
    {
        return host::norm_inf(array);
    }
    else
    {
        return device::norm_inf(array);
    }
}

// Fill array with zeros
void linalg::fill_with_zeros(vector<uint32_t> &vec)
{
    if(vec.is_on_host())
    {
        host::fill_with_zeros(vec);
    }
    else
    {
        device::fill_with_zeros(vec);
    }
}
void linalg::fill_with_zeros(vector<int32_t> &vec)
{
    if(vec.is_on_host())
    {
        host::fill_with_zeros(vec);
    }
    else
    {
        device::fill_with_zeros(vec);
    }
}
void linalg::fill_with_zeros(vector<int64_t> &vec)
{
    if(vec.is_on_host())
    {
        host::fill_with_zeros(vec);
    }
    else
    {
        device::fill_with_zeros(vec);
    }
}
void linalg::fill_with_zeros(vector<double> &vec)
{
    if(vec.is_on_host())
    {
        host::fill_with_zeros(vec);
    }
    else
    {
        device::fill_with_zeros(vec);
    }
}

// Fill array with ones
void linalg::fill_with_ones(vector<uint32_t> &vec)
{
    if(vec.is_on_host())
    {
        host::fill_with_ones(vec);
    }
    else
    {
        device::fill_with_ones(vec);
    }
}
void linalg::fill_with_ones(vector<int32_t> &vec)
{
    if(vec.is_on_host())
    {
        host::fill_with_ones(vec);
    }
    else
    {
        device::fill_with_ones(vec);
    }
}
void linalg::fill_with_ones(vector<int64_t> &vec)
{
    if(vec.is_on_host())
    {
        host::fill_with_ones(vec);
    }
    else
    {
        device::fill_with_ones(vec);
    }
}
void linalg::fill_with_ones(vector<double> &vec)
{
    if(vec.is_on_host())
    {
        host::fill_with_ones(vec);
    }
    else
    {
        device::fill_with_ones(vec);
    }
}

// Copy array
void linalg::copy(vector<uint32_t> &dest, const vector<uint32_t> &src)
{
    if(dest.is_on_host() != src.is_on_host())
    {
        std::cout << "Error (copy): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(dest.is_on_host())
    {
        host::copy(dest, src);
    }
    else
    {
        device::copy(dest, src);
    }
}
void linalg::copy(vector<int32_t> &dest, const vector<int32_t> &src)
{
    if(dest.is_on_host() != src.is_on_host())
    {
        std::cout << "Error (copy): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(dest.is_on_host())
    {
        host::copy(dest, src);
    }
    else
    {
        device::copy(dest, src);
    }
}
void linalg::copy(vector<int64_t> &dest, const vector<int64_t> &src)
{
    if(dest.is_on_host() != src.is_on_host())
    {
        std::cout << "Error (copy): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(dest.is_on_host())
    {
        host::copy(dest, src);
    }
    else
    {
        device::copy(dest, src);
    }
}
void linalg::copy(vector<double> &dest, const vector<double> &src)
{
    if(dest.is_on_host() != src.is_on_host())
    {
        std::cout << "Error (copy): Mixing host and device inputs not supported. Skipping computation." << std::endl;
        return;
    }

    if(dest.is_on_host())
    {
        host::copy(dest, src);
    }
    else
    {
        device::copy(dest, src);
    }
}