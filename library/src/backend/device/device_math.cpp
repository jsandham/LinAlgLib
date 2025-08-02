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
#include <iostream>

#include "cuda/cuda_kernels.h"

#include "../../trace.h"

// Compute y = alpha * x + y
void linalg::device::axpy(double alpha, const vector<double>& x, vector<double>& y)
{
    ROUTINE_TRACE("linalg::device::axpy");
    launch_cuda_axpy_kernel(x.get_size(), alpha, x.get_vec(), y.get_vec());
}

// Compute y = alpha * x + beta * y
void linalg::device::axpby(double alpha, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::device::axpby");
    launch_cuda_axpby_kernel(x.get_size(), alpha, x.get_vec(), beta, y.get_vec());
}

// Compute z = alpha * x + beta * y + gamma * z
void linalg::device::axpbypgz(double                alpha,
                              const vector<double>& x,
                              double                beta,
                              const vector<double>& y,
                              double                gamma,
                              vector<double>&       z)
{
    ROUTINE_TRACE("linalg::device::axpbypgz");
    launch_cuda_axpbypgz_kernel(
        x.get_size(), alpha, x.get_vec(), beta, y.get_vec(), gamma, z.get_vec());
}

// Compute y = A * x
void linalg::device::matrix_vector_product(const csr_matrix&     A,
                                           const vector<double>& x,
                                           vector<double>&       y)
{
    ROUTINE_TRACE("linalg::device::multiply_by_vector");
    launch_cuda_csrmv_kernel(A.get_m(),
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
void linalg::device::matrix_vector_product(
    double alpha, const csr_matrix& A, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::device::multiply_by_vector");
    launch_cuda_csrmv_kernel(A.get_m(),
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
void linalg::device::forward_solve(const csr_matrix&     A,
                                   const vector<double>& b,
                                   vector<double>&       x,
                                   bool                  unit_diag)
{
    std::cout << "Error: forward_solve on device not implemented" << std::endl;
}

// Backward solve
void linalg::device::backward_solve(const csr_matrix&     A,
                                    const vector<double>& b,
                                    vector<double>&       x,
                                    bool                  unit_diag)
{
    std::cout << "Error: backward_solve on device not implemented" << std::endl;
}

// Transpose matrix
void linalg::device::transpose_matrix(const csr_matrix& A, csr_matrix& transposeA)
{
    std::cout << "Error: transpose_matrix on device not implemented" << std::endl;
}

// Dot product
double linalg::device::dot_product(const vector<double>& x, const vector<double>& y)
{
    ROUTINE_TRACE("linalg::device::dot_product");
    double result = 0.0;
    launch_cuda_dot_product_kernel(x.get_size(), x.get_vec(), y.get_vec(), &result);

    return result;
}

// Compute residual
void linalg::device::compute_residual(const csr_matrix&     A,
                                      const vector<double>& x,
                                      const vector<double>& b,
                                      vector<double>&       res)
{
    ROUTINE_TRACE("linalg::device::compute_residual");
    launch_cuda_compute_residual_kernel(A.get_m(),
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
void linalg::device::exclusive_scan(vector<double>& x)
{
    std::cout << "Error: exclusive_scan on device not implemented" << std::endl;
}

// Extract diagonal entries
void linalg::device::diagonal(const csr_matrix& A, vector<double>& d)
{
    ROUTINE_TRACE("linalg::device::diagonal");
    launch_cuda_extract_diagonal_kernel(A.get_m(),
                                        A.get_n(),
                                        A.get_nnz(),
                                        A.get_row_ptr(),
                                        A.get_col_ind(),
                                        A.get_val(),
                                        d.get_vec());
}

// Euclidean norm
double linalg::device::norm_euclid(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::device::norm_euclid");
    return std::sqrt(dot_product(array, array));
}

// Infinity norm
double linalg::device::norm_inf(const vector<double>& array)
{
    ROUTINE_TRACE("linalg::device::norm_inf");
    double norm = 0.0;
    launch_cuda_norm_inf_kernel(array.get_size(), array.get_vec(), &norm);
    return norm;
}

// Fill array with value
template <typename T>
void linalg::device::fill(vector<T>& vec, T value)
{
    ROUTINE_TRACE("linalg::device::fill");
    //in host_math.cpp fill calls host_fill...should this fill call device_fill??
    //device_fill can then call either CUDA or HIP fill...

    // Also should most of these "math functions" be removed and we can call host_fill
    // or device_fill directly from the vector.cpp methods? Or at least the non math ones like
    // fill and copy?

    // Maybe copy, fill, device memory allocation and free should be moved to device_memory.cpp??
    //void device_allocate(void** ptr, size_t size_in_bytes);
    //void device_free(void* ptr);
    //void copy_h2d(void* dest, const void* src, size_t size_in_bytes);
    //void copy_d2h(void* dest, const void* src, size_t size_in_bytes);
    //template <typename T>
    //void device_fill(T* data, size_t size, T val);
    //
    // or maybe make them all templated:
    // template <typename T>
    // void device_allocate(T** ptr, size_t size);
    // template <typename T>
    // void device_free(T* ptr); // if we pass with T** ptr, we could set to NULL after free'ing...
    // template <typename T>
    // void copy_h2d(T* dest, const T* src, size_t size);
    // template <typename T>
    // void copy_d2h(T* dest, const T* src, size_t size);
    // template <typename T>
    // void device_fill(T* data, size_t size, T val);

    launch_cuda_fill_kernel(vec.get_vec(), vec.get_size(), value);
}

template void linalg::device::fill<uint32_t>(vector<uint32_t>& vec, uint32_t value);
template void linalg::device::fill<int32_t>(vector<int32_t>& vec, int32_t value);
template void linalg::device::fill<int64_t>(vector<int64_t>& vec, int64_t value);
template void linalg::device::fill<double>(vector<double>& vec, double value);

// Copy array
template <typename T>
void linalg::device::copy(vector<T>& dest, const vector<T>& src)
{
    ROUTINE_TRACE("linalg::device::copy");
    launch_cuda_copy_kernel(dest.get_vec(), src.get_vec(), src.get_size());
}

template void linalg::device::copy<uint32_t>(vector<uint32_t>& dest, const vector<uint32_t>& src);
template void linalg::device::copy<int32_t>(vector<int32_t>& dest, const vector<int32_t>& src);
template void linalg::device::copy<int64_t>(vector<int64_t>& dest, const vector<int64_t>& src);
template void linalg::device::copy<double>(vector<double>& dest, const vector<double>& src);

// Jacobi solve
void linalg::device::jacobi_solve(const vector<double>& rhs,
                                  const vector<double>& diag,
                                  vector<double>&       x)
{
    ROUTINE_TRACE("linalg::device::jacobi_solve");
    launch_cuda_jacobi_solve_kernel(rhs.get_size(), rhs.get_vec(), diag.get_vec(), x.get_vec());
}
