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

enum class backend
{
    host,
    device,
    invalid
};

template<typename T, typename... Rest>
backend determine_backend(const T& first, const Rest&... rest) 
{
    bool first_is_on_host = first.is_on_host();
    bool rest_equal_to_first = ((rest.is_on_host() == first_is_on_host) && ...);
    if(rest_equal_to_first)
    {
        return first_is_on_host ? backend::host : backend::device;
    }
    else
    {
        return backend::invalid;
    }
}

// Compute y = alpha * x + y
void linalg::axpy(double alpha, const vector<double>& x, vector<double>& y)
{
    switch(determine_backend(x, y))
    {
        case backend::host:
            return host::axpy(alpha, x, y);
        case backend::device:
            return device::axpy(alpha, x, y);
        case backend::invalid:
            return;
    }
}

void linalg::axpy(const scalar<double>& alpha, const vector<double>& x, vector<double>& y)
{
    switch(determine_backend(alpha, x, y))
    {
        case backend::host:
            return host::axpy(alpha, x, y);
        case backend::device:
            return device::axpy(alpha, x, y);
        case backend::invalid:
            return;
    }
}

// Compute y = alpha * x + beta * y
void linalg::axpby(double alpha, const vector<double>& x, double beta, vector<double>& y)
{
    switch(determine_backend(x, y))
    {
        case backend::host:
            return host::axpby(alpha, x, beta, y);
        case backend::device:
            return device::axpby(alpha, x, beta, y);
        case backend::invalid:
            return;
    }
}
void linalg::axpby(const scalar<double>& alpha, const vector<double>& x, const scalar<double>& beta, vector<double>& y)
{
    switch(determine_backend(alpha, x, beta, y))
    {
        case backend::host:
            return host::axpby(alpha, x, beta, y);
        case backend::device:
            return device::axpby(alpha, x, beta, y);
        case backend::invalid:
            return;
    }
}

// Compute z = alpha * x + beta * y + gamma * z
void linalg::axpbypgz(double alpha, const vector<double>& x, double beta, const vector<double>& y, double gamma, vector<double>& z)
{
    switch(determine_backend(x, y, z))
    {
        case backend::host:
            return host::axpbypgz(alpha, x, beta, y, gamma, z);
        case backend::device:
            return device::axpbypgz(alpha, x, beta, y, gamma, z);
        case backend::invalid:
            return;
    }
}
void linalg::axpbypgz(const scalar<double>& alpha, const vector<double>& x, const scalar<double>& beta, const vector<double>& y, const scalar<double>& gamma, vector<double>& z)
{
    switch(determine_backend(alpha, x, beta, y, gamma, z))
    {
        case backend::host:
            return host::axpbypgz(alpha, x, beta, y, gamma, z);
        case backend::device:
            return device::axpbypgz(alpha, x, beta, y, gamma, z);
        case backend::invalid:
            return;
    }
}

// Compute y = A * x
void linalg::matrix_vector_product(const csr_matrix& A, const vector<double>& x, vector<double>&y)
{
    switch(determine_backend(A, x, y))
    {
        case backend::host:
            return host::matrix_vector_product(A, x, y);
        case backend::device:
            return device::matrix_vector_product(A, x, y);
        case backend::invalid:
            return;
    }
}

// Compute y = alpha * A * x + beta * y
void linalg::matrix_vector_product(double alpha, const csr_matrix& A, const vector<double>& x, double beta, vector<double>&y)
{
    switch(determine_backend(A, x, y))
    {
        case backend::host:
            return host::matrix_vector_product(alpha, A, x, beta, y);
        case backend::device:
            return device::matrix_vector_product(alpha, A, x, beta, y);
        case backend::invalid:
            return;
    }
}

// Compute C = A * B
void linalg::matrix_matrix_product(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    switch(determine_backend(C, A, B))
    {
        case backend::host:
            return host::matrix_matrix_product(C, A, B);
        case backend::device:
            return device::matrix_matrix_product(C, A, B);
        case backend::invalid:
            return;
    }
}

// Compute C = A + B
void linalg::matrix_matrix_addition(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    switch(determine_backend(C, A, B))
    {
        case backend::host:
            return host::matrix_matrix_addition(C, A, B);
        case backend::device:
            return device::matrix_matrix_addition(C, A, B);
        case backend::invalid:
            return;
    }
}

// Incomplete IC factorization
void linalg::csric0(csr_matrix& LL, int* structural_zero, int* numeric_zero)
{
    switch(determine_backend(LL))
    {
        case backend::host:
            return host::csric0(LL, structural_zero, numeric_zero);
        case backend::device:
            return device::csric0(LL, structural_zero, numeric_zero);
        case backend::invalid:
            return;
    }
}

// Incomplete LU factorization
void linalg::csrilu0(csr_matrix& LU, int* structural_zero, int* numeric_zero)
{
    switch(determine_backend(LU))
    {
        case backend::host:
            return host::csrilu0(LU, structural_zero, numeric_zero);
        case backend::device:
            return device::csrilu0(LU, structural_zero, numeric_zero);
        case backend::invalid:
            return;
    }
}

// Forward solve
void linalg::forward_solve(const csr_matrix& A, const vector<double>& b, vector<double>& x, bool unit_diag)
{
    switch(determine_backend(A, b, x))
    {
        case backend::host:
            return host::forward_solve(A, b, x, unit_diag);
        case backend::device:
            return device::forward_solve(A, b, x, unit_diag);
        case backend::invalid:
            return;
    }
}

// Backward solve
void linalg::backward_solve(const csr_matrix& A, const vector<double>& b, vector<double>& x, bool unit_diag)
{
    switch(determine_backend(A, b, x))
    {
        case backend::host:
            return host::backward_solve(A, b, x, unit_diag);
        case backend::device:
            return device::backward_solve(A, b, x, unit_diag);
        case backend::invalid:
            return;
    }
}

// Transpose matrix
void linalg::transpose_matrix(const csr_matrix &A, csr_matrix &transposeA)
{
    switch(determine_backend(A, transposeA))
    {
        case backend::host:
            return host::transpose_matrix(A, transposeA);
        case backend::device:
            return device::transpose_matrix(A, transposeA);
        case backend::invalid:
            return;
    }
}

// Dot product
double linalg::dot_product(const vector<double>& x, const vector<double>& y)
{
    switch(determine_backend(x, y))
    {
        case backend::host:
            return host::dot_product(x, y);
        case backend::device:
            return device::dot_product(x, y);
        case backend::invalid:
            return 0.0;
    }

    return 0.0;
}

// Dot product
void linalg::dot_product(const vector<double>& x, const vector<double>& y, scalar<double>& result)
{
    switch(determine_backend(x, y, result))
    {
        case backend::host:
            return host::dot_product(x, y, result);
        case backend::device:
            return device::dot_product(x, y, result);
        case backend::invalid:
            return;
    }
}

// Compute residual
void linalg::compute_residual(const csr_matrix& A, const vector<double>& x, const vector<double>& b, vector<double>& res)
{
    switch(determine_backend(A, x, b, res))
    {
        case backend::host:
            return host::compute_residual(A, x, b, res);
        case backend::device:
            return device::compute_residual(A, x, b, res);
        case backend::invalid:
            return;
    }
}

// Exclusive scan
void linalg::exclusize_scan(vector<double>& x)
{
    switch(determine_backend(x))
    {
        case backend::host:
            return host::exclusize_scan(x);
        case backend::device:
            return device::exclusize_scan(x);
        case backend::invalid:
            return;
    }
}

// Extract diagonal entries
void linalg::diagonal(const csr_matrix& A, vector<double>& d)
{
    switch(determine_backend(A, d))
    {
        case backend::host:
            return host::diagonal(A, d);
        case backend::device:
            return device::diagonal(A, d);
        case backend::invalid:
            return;
    }
}

// Euclidean norm
double linalg::norm_euclid(const vector<double>& array)
{
    switch(determine_backend(array))
    {
        case backend::host:
            return host::norm_euclid(array);
        case backend::device:
            return device::norm_euclid(array);
        case backend::invalid:
            return 0.0;
    }
    return 0.0;
}

// Infinity norm
double linalg::norm_inf(const vector<double>& array)
{
    switch(determine_backend(array))
    {
        case backend::host:
            return host::norm_inf(array);
        case backend::device:
            return device::norm_inf(array);
        case backend::invalid:
            return 0.0;
    }

    return 0.0;
}

// Fill array with value
template<typename T>
void linalg::fill(vector<T> &vec, T value)
{
    switch(determine_backend(vec))
    {
        case backend::host:
            return host::fill(vec, value);
        case backend::device:
            return device::fill(vec, value);
        case backend::invalid:
            return;
    }
}

template void linalg::fill<uint32_t>(vector<uint32_t> &vec, uint32_t value);
template void linalg::fill<int32_t>(vector<int32_t> &vec, int32_t value);
template void linalg::fill<int64_t>(vector<int64_t> &vec, int64_t value);
template void linalg::fill<double>(vector<double> &vec, double value);

// Copy array
template <typename T>
void linalg::copy(vector<T> &dest, const vector<T> &src)
{
    switch(determine_backend(dest, src))
    {
        case backend::host:
            return host::copy(dest, src);
        case backend::device:
            return device::copy(dest, src);
        case backend::invalid:
            return;
    }
}

template void linalg::copy<uint32_t>(vector<uint32_t> &dest, const vector<uint32_t> &src);
template void linalg::copy<int32_t>(vector<int32_t> &dest, const vector<int32_t> &src);
template void linalg::copy<int64_t>(vector<int64_t> &dest, const vector<int64_t> &src);
template void linalg::copy<double>(vector<double> &dest, const vector<double> &src);
