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
#include <cmath>
#include <cstdint>
#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_math.h"
#include "cuda/cuda_memory.h"
#endif

// Compute y = alpha * x + y
void linalg::device_axpy(double alpha, const vector<double>& x, vector<double>& y)
{
    ROUTINE_TRACE("linalg::device_axpy");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_axpy(x.get_size(), alpha, x.get_vec(), y.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

// Compute y = alpha * x + beta * y
void linalg::device_axpby(double alpha, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::device_axpby");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_axpby(x.get_size(), alpha, x.get_vec(), beta, y.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
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

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(
            cuda_axpbypgz(x.get_size(), alpha, x.get_vec(), beta, y.get_vec(), gamma, z.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

// Compute y = A * x
void linalg::device_matrix_vector_product(const csr_matrix&     A,
                                          const vector<double>& x,
                                          vector<double>&       y)
{
    ROUTINE_TRACE("linalg::device_multiply_by_vector");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrmv(A.get_m(),
                             A.get_n(),
                             A.get_nnz(),
                             1.0,
                             A.get_row_ptr(),
                             A.get_col_ind(),
                             A.get_val(),
                             x.get_vec(),
                             0.0,
                             y.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

// Compute y = alpha * A * x + beta * y
void linalg::device_matrix_vector_product(
    double alpha, const csr_matrix& A, const vector<double>& x, double beta, vector<double>& y)
{
    ROUTINE_TRACE("linalg::device_multiply_by_vector");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrmv(A.get_m(),
                             A.get_n(),
                             A.get_nnz(),
                             alpha,
                             A.get_row_ptr(),
                             A.get_col_ind(),
                             A.get_val(),
                             x.get_vec(),
                             beta,
                             y.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

// Compute C = A * B
void linalg::device_matrix_matrix_product(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    ROUTINE_TRACE("linalg::device_matrix_matrix_product");

    if constexpr(is_cuda_available())
    {
        int nnz_C;
        //CALL_CUDA(cuda_allocate(&nnz_C, sizeof(int)));

        if(C.get_m() != A.get_m() || C.get_n() != B.get_n())
        {
            C.resize(A.get_m(), B.get_n(), 0);
        }

        std::cout << "C.get_m(): " << C.get_m() << " C.get_n(): " << C.get_n()
                  << " C.get_nnz(): " << C.get_nnz() << std::endl;

        csrgemm_descr* descr;
        CALL_CUDA(cuda_create_csrgemm_descr(&descr));

        CALL_CUDA(cuda_csrgemm_nnz(C.get_m(),
                                   C.get_n(),
                                   B.get_m(),
                                   A.get_nnz(),
                                   B.get_nnz(),
                                   0,
                                   descr,
                                   1.0,
                                   A.get_row_ptr(),
                                   A.get_col_ind(),
                                   B.get_row_ptr(),
                                   B.get_col_ind(),
                                   0.0,
                                   nullptr,
                                   nullptr,
                                   C.get_row_ptr(),
                                   &nnz_C));
        std::cout << "nnz_C: " << nnz_C << std::endl;

        C.resize(C.get_m(), C.get_n(), nnz_C);

        CALL_CUDA(cuda_csrgemm(C.get_m(),
                               C.get_n(),
                               B.get_m(),
                               A.get_nnz(),
                               B.get_nnz(),
                               0,
                               C.get_nnz(),
                               descr,
                               1.0,
                               A.get_row_ptr(),
                               A.get_col_ind(),
                               A.get_val(),
                               B.get_row_ptr(),
                               B.get_col_ind(),
                               B.get_val(),
                               0.0,
                               nullptr,
                               nullptr,
                               nullptr,
                               C.get_row_ptr(),
                               C.get_col_ind(),
                               C.get_val()));

        CALL_CUDA(cuda_destroy_csrgemm_descr(descr));

        //CALL_CUDA(cuda_free(nnz_C));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

// Compute C = A + B
void linalg::device_matrix_matrix_addition(csr_matrix& C, const csr_matrix& A, const csr_matrix& B)
{
    ROUTINE_TRACE("linalg::device_matrix_matrix_addition");

    if constexpr(is_cuda_available())
    {
        if(C.get_m() != A.get_m() || C.get_n() != A.get_n())
        {
            C.resize(A.get_m(), A.get_n(), 0);
        }

        std::cout << "C.get_m(): " << C.get_m() << " C.get_n(): " << C.get_n()
                  << " C.get_nnz(): " << C.get_nnz() << std::endl;

        int nnz_C;

        csrgeam_descr* descr;
        CALL_CUDA(cuda_create_csrgeam_descr(&descr));

        CALL_CUDA(cuda_csrgeam_nnz(C.get_m(),
                                   C.get_n(),
                                   A.get_nnz(),
                                   B.get_nnz(),
                                   descr,
                                   1.0,
                                   A.get_row_ptr(),
                                   A.get_col_ind(),
                                   0.0,
                                   B.get_row_ptr(),
                                   B.get_col_ind(),
                                   C.get_row_ptr(),
                                   &nnz_C));
        std::cout << "nnz_C: " << nnz_C << std::endl;

        C.resize(C.get_m(), C.get_n(), nnz_C);

        CALL_CUDA(cuda_csrgeam(C.get_m(),
                               C.get_n(),
                               A.get_nnz(),
                               B.get_nnz(),
                               C.get_nnz(),
                               descr,
                               1.0,
                               A.get_row_ptr(),
                               A.get_col_ind(),
                               A.get_val(),
                               1.0,
                               B.get_row_ptr(),
                               B.get_col_ind(),
                               B.get_val(),
                               C.get_row_ptr(),
                               C.get_col_ind(),
                               C.get_val()));

        CALL_CUDA(cuda_destroy_csrgeam_descr(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
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
    ROUTINE_TRACE("linalg::device_multiply_by_vector");

    if constexpr(is_cuda_available())
    {
        size_t buffer_size;
        CALL_CUDA(cuda_csr2csc_buffer_size(A.get_m(),
                                           A.get_n(),
                                           A.get_nnz(),
                                           A.get_row_ptr(),
                                           A.get_col_ind(),
                                           A.get_val(),
                                           &buffer_size));

        int32_t* buffer = nullptr;
        CALL_CUDA(cuda_allocate(&buffer, sizeof(int32_t) * A.get_nnz()));

        CALL_CUDA(cuda_csr2csc(A.get_m(),
                               A.get_n(),
                               A.get_nnz(),
                               A.get_row_ptr(),
                               A.get_col_ind(),
                               A.get_val(),
                               transposeA.get_row_ptr(),
                               transposeA.get_col_ind(),
                               transposeA.get_val(),
                               buffer));

        CALL_CUDA(cuda_free(buffer));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }

    // std::cout << "Error: transpose_matrix on device not implemented" << std::endl;
}

// Dot product
double linalg::device_dot_product(const vector<double>& x, const vector<double>& y)
{
    ROUTINE_TRACE("linalg::device_dot_product");

    if constexpr(is_cuda_available())
    {
        return RETURN_CALL_CUDA(cuda_dot_product(x.get_vec(), y.get_vec(), x.get_size()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return 0.0;
    }

    //return device_dot_product_impl(x.get_vec(), y.get_vec(), x.get_size());
}

// Compute residual
void linalg::device_compute_residual(const csr_matrix&     A,
                                     const vector<double>& x,
                                     const vector<double>& b,
                                     vector<double>&       res)
{
    ROUTINE_TRACE("linalg::device_compute_residual");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_compute_residual(A.get_m(),
                                        A.get_n(),
                                        A.get_nnz(),
                                        A.get_row_ptr(),
                                        A.get_col_ind(),
                                        A.get_val(),
                                        x.get_vec(),
                                        b.get_vec(),
                                        res.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

// Extract diagonal entries
void linalg::device_diagonal(const csr_matrix& A, vector<double>& d)
{
    ROUTINE_TRACE("linalg::device_diagonal");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_extract_diagonal(A.get_m(),
                                        A.get_n(),
                                        A.get_nnz(),
                                        A.get_row_ptr(),
                                        A.get_col_ind(),
                                        A.get_val(),
                                        d.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
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

    if constexpr(is_cuda_available())
    {
        return RETURN_CALL_CUDA(cuda_norm_inf(array.get_vec(), array.get_size()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return 0.0;
    }
    // return device_norm_inf_impl(array.get_vec(), array.get_size());
}

// Jacobi solve
void linalg::device_jacobi_solve(const vector<double>& rhs,
                                 const vector<double>& diag,
                                 vector<double>&       x)
{
    ROUTINE_TRACE("linalg::device_jacobi_solve");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_jacobi_solve(rhs.get_vec(), diag.get_vec(), x.get_vec(), rhs.get_size()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}
