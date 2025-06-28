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
#include <iostream>
#include "device_math.h"

// Compute y = alpha * x + y
void linalg::device::axpy(double alpha, const vector<double>& x, vector<double>& y)
{
    std::cout << "Error: axpy on device not implemented" << std::endl;
}

// Compute y = alpha * x + y
void linalg::device::axpy(const scalar<double>& alpha, const vector<double>& x, vector<double>& y)
{
    std::cout << "Error: axpy on device not implemented" << std::endl;
}

// Compute y = alpha * x + beta * y
void linalg::device::axpby(double alpha, const vector<double>& x, double beta, vector<double>& y)
{
    std::cout << "Error: axpyby on device not implemented" << std::endl;
}

// Compute y = alpha * x + beta * y
void linalg::device::axpby(const scalar<double>& alpha, const vector<double>& x, const scalar<double>& beta, vector<double>& y)
{
    std::cout << "Error: axpyby on device not implemented" << std::endl;
}

// Compute z = alpha * x + beta * y + gamma * z
void linalg::device::axpbypgz(double alpha, const vector<double>& x, double beta, const vector<double>& y, double gamma, vector<double>& z)
{
    std::cout << "Error: axpbypgz on device not implemented" << std::endl;
}

// Compute z = alpha * x + beta * y + gamma * z
void linalg::device::axpbypgz(const scalar<double>& alpha, const vector<double>& x, const scalar<double>& beta, const vector<double>& y, const scalar<double>& gamma, vector<double>& z)
{
    std::cout << "Error: axpbypgz on device not implemented" << std::endl;
}

// Compute y = A * x
void linalg::device::matrix_vector_product(const csr_matrix& A, const vector<double>& x, vector<double>&y)
{
    std::cout << "Error: matrix_vector_product on device not implemented" << std::endl;
}

// Compute y = alpha * A * x + beta * y
void linalg::device::matrix_vector_product(double alpha, const csr_matrix& A, const vector<double>& x, double beta, vector<double>&y)
{
    std::cout << "Error: matrix_vector_product on device not implemented" << std::endl;
}

// Compute C = A * B
void linalg::device::matrix_matrix_product(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    std::cout << "Error: matrix_matrix_product on device not implemented" << std::endl;
}

// Compute C = A + B
void linalg::device::matrix_matrix_addition(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    std::cout << "Error: matrix_matrix_addition on device not implemented" << std::endl;
}

// Incomplete IC factorization
void linalg::device::csric0(csr_matrix& LL, int* structural_zero, int* numeric_zero)
{
    std::cout << "Error: csric0 on device not implemented" << std::endl;
}

// Incomplete LU factorization
void linalg::device::csrilu0(csr_matrix& LU, int* structural_zero, int* numeric_zero)
{
    std::cout << "Error: csrilu0 on device not implemented" << std::endl;
}

// Forward solve
void linalg::device::forward_solve(const csr_matrix& A, const vector<double>& b, vector<double>& x, bool unit_diag)
{
    std::cout << "Error: forward_solve on device not implemented" << std::endl;
}

// Backward solve
void linalg::device::backward_solve(const csr_matrix& A, const vector<double>& b, vector<double>& x, bool unit_diag)
{
    std::cout << "Error: backward_solve on device not implemented" << std::endl;
}

// Transpose matrix
void linalg::device::transpose_matrix(const csr_matrix &A, csr_matrix &transposeA)
{
    std::cout << "Error: transpose_matrix on device not implemented" << std::endl;
}

// Dot product
double linalg::device::dot_product(const vector<double>& x, const vector<double>& y)
{
    std::cout << "Error: dot_product on device not implemented" << std::endl;
    return 0.0;
}

// Dot product
void linalg::device::dot_product(const vector<double>& x, const vector<double>& y, scalar<double>& result)
{
    std::cout << "Error: dot_product on device not implemented" << std::endl;
    *(result.get_val()) = 0.0;
}

// Compute residual
void linalg::device::compute_residual(const csr_matrix& A, const vector<double>& x, const vector<double>& b, vector<double>& res)
{
    std::cout << "Error: compute_residual on device not implemented" << std::endl;
}

// Exclusive scan
void linalg::device::exclusize_scan(vector<double>& x)
{
    std::cout << "Error: exclusize_scan on device not implemented" << std::endl;
}

// Extract diagonal entries
void linalg::device::diagonal(const csr_matrix& A, vector<double>& d)
{
    std::cout << "Error: diagonal on device not implemented" << std::endl;
}

// Euclidean norm
double linalg::device::norm_euclid(const vector<double>& array)
{
    std::cout << "Error: norm_euclid on device not implemented" << std::endl;
    return 0.0;
}

// Infinity norm
double linalg::device::norm_inf(const vector<double>& array)
{
    std::cout << "Error: norm_inf on device not implemented" << std::endl;
    return 0.0;
}

// Fill array with value
template<typename T>
void linalg::device::fill(vector<T> &vec, T value)
{
    std::cout << "Error: fill on device not implemented" << std::endl;
}

template void linalg::device::fill<uint32_t>(vector<uint32_t> &vec, uint32_t value);
template void linalg::device::fill<int32_t>(vector<int32_t> &vec, int32_t value);
template void linalg::device::fill<int64_t>(vector<int64_t> &vec, int64_t value);
template void linalg::device::fill<double>(vector<double> &vec, double value);

// Copy array
template <typename T>
void linalg::device::copy(vector<T> &dest, const vector<T> &src)
{
    std::cout << "Error: copy on device not implemented" << std::endl;
}

template void linalg::device::copy<uint32_t>(vector<uint32_t> &dest, const vector<uint32_t> &src);
template void linalg::device::copy<int32_t>(vector<int32_t> &dest, const vector<int32_t> &src);
template void linalg::device::copy<int64_t>(vector<int64_t> &dest, const vector<int64_t> &src);
template void linalg::device::copy<double>(vector<double> &dest, const vector<double> &src);
