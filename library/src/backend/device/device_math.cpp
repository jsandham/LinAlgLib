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

// Extract lower triangular entries
void linalg::device_extract_lower_triangular_nnz(const csr_matrix& A, csr_matrix& L, int& nnz_L)
{
    ROUTINE_TRACE("linalg::device_extract_lower_triangular_nnz");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_extract_lower_triangular_nnz(A.get_m(),
                                                    A.get_n(),
                                                    A.get_nnz(),
                                                    A.get_row_ptr(),
                                                    A.get_col_ind(),
                                                    L.get_row_ptr(),
                                                    &nnz_L));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}
void linalg::device_extract_lower_triangular(const csr_matrix& A, csr_matrix& L)
{
    ROUTINE_TRACE("linalg::device_extract_lower_triangular");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_extract_lower_triangular(A.get_m(),
                                                A.get_n(),
                                                A.get_nnz(),
                                                A.get_row_ptr(),
                                                A.get_col_ind(),
                                                A.get_val(),
                                                L.get_m(),
                                                L.get_n(),
                                                L.get_nnz(),
                                                L.get_row_ptr(),
                                                L.get_col_ind(),
                                                L.get_val()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

// Extract upper triangular entries
void linalg::device_extract_upper_triangular_nnz(const csr_matrix& A, csr_matrix& U, int& nnz_U)
{
    ROUTINE_TRACE("linalg::device_extract_upper_triangular_nnz_count");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_extract_upper_triangular_nnz(A.get_m(),
                                                    A.get_n(),
                                                    A.get_nnz(),
                                                    A.get_row_ptr(),
                                                    A.get_col_ind(),
                                                    U.get_row_ptr(),
                                                    &nnz_U));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}
void linalg::device_extract_upper_triangular(const csr_matrix& A, csr_matrix& U)
{
    ROUTINE_TRACE("linalg::device_extract_upper_triangular");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_extract_upper_triangular(A.get_m(),
                                                A.get_n(),
                                                A.get_nnz(),
                                                A.get_row_ptr(),
                                                A.get_col_ind(),
                                                A.get_val(),
                                                U.get_m(),
                                                U.get_n(),
                                                U.get_nnz(),
                                                U.get_row_ptr(),
                                                U.get_col_ind(),
                                                U.get_val()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

// Scale diagonal entries
void linalg::device_scale_diagonal(csr_matrix& A, double scalar)
{
    ROUTINE_TRACE("linalg::device_scale_diagonal");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(
            cuda_scale_diagonal(A.get_row_ptr(), A.get_col_ind(), A.get_val(), A.get_m(), scalar));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

// Scale by inverse of diagonal entries
void linalg::device_scale_by_inverse_diagonal(csr_matrix& A, const vector<double>& diag)
{
    ROUTINE_TRACE("linalg::device_scale_by_inverse_diagonal");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_scale_by_inverse_diagonal(
            A.get_row_ptr(), A.get_col_ind(), A.get_val(), A.get_m(), diag.get_vec()));
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

// SSOR fill lower preconditioner L = (beta * D - E) * D^-1, beta = 1 / omega
void linalg::device_ssor_fill_lower_precond(const csr_matrix& A, csr_matrix& L, double omega)
{
    ROUTINE_TRACE("linalg::device_ssor_fill_lower_precond");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_ssor_fill_lower_precond(A.get_m(),
                                               A.get_n(),
                                               A.get_nnz(),
                                               A.get_row_ptr(),
                                               A.get_col_ind(),
                                               A.get_val(),
                                               L.get_m(),
                                               L.get_n(),
                                               L.get_nnz(),
                                               L.get_row_ptr(),
                                               L.get_col_ind(),
                                               L.get_val(),
                                               omega));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

//SSOR fill upper preconditioner U = (beta * D - F), beta = 1 / omega
void linalg::device_ssor_fill_upper_precond(const csr_matrix& A, csr_matrix& U, double omega)
{
    ROUTINE_TRACE("linalg::device_ssor_fill_upper_precond");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_ssor_fill_upper_precond(A.get_m(),
                                               A.get_n(),
                                               A.get_nnz(),
                                               A.get_row_ptr(),
                                               A.get_col_ind(),
                                               A.get_val(),
                                               U.get_m(),
                                               U.get_n(),
                                               U.get_nnz(),
                                               U.get_row_ptr(),
                                               U.get_col_ind(),
                                               U.get_val(),
                                               omega));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::allocate_csrtrsv_device_data(csrtrsv_descr* descr)
{
    ROUTINE_TRACE("linalg::allocate_csrtrsv_device_data");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(allocate_csrtrsv_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}
void linalg::free_csrtrsv_device_data(csrtrsv_descr* descr)
{
    ROUTINE_TRACE("linalg::free_csrtrsv_device_data");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(free_csrtrsv_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::device_csrtrsv_analysis(const csr_matrix& A,
                                     triangular_type   tri_type,
                                     diagonal_type     diag_type,
                                     csrtrsv_descr*    descr)
{
    ROUTINE_TRACE("linalg::device_csrtrsv_analysis");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrtrsv_analysis(A.get_m(),
                                        A.get_n(),
                                        A.get_nnz(),
                                        A.get_row_ptr(),
                                        A.get_col_ind(),
                                        A.get_val(),
                                        tri_type,
                                        diag_type,
                                        descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}
void linalg::device_csrtrsv_solve(const csr_matrix&     A,
                                  const vector<double>& b,
                                  vector<double>&       x,
                                  double                alpha,
                                  triangular_type       tri_type,
                                  diagonal_type         diag_type,
                                  const csrtrsv_descr*  descr)
{
    ROUTINE_TRACE("linalg::device_csrtrsv_solve");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrtrsv_solve(A.get_m(),
                                     A.get_n(),
                                     A.get_nnz(),
                                     alpha,
                                     A.get_row_ptr(),
                                     A.get_col_ind(),
                                     A.get_val(),
                                     b.get_vec(),
                                     x.get_vec(),
                                     tri_type,
                                     diag_type,
                                     descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::allocate_csrmv_device_data(csrmv_descr* descr)
{
    ROUTINE_TRACE("linalg::allocate_csrmv_device_data");

    if constexpr(is_cuda_available())
    {
        //CALL_CUDA(allocate_csrmv_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}
void linalg::free_csrmv_device_data(csrmv_descr* descr)
{
    ROUTINE_TRACE("linalg::free_csrmv_device_data");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(free_csrmv_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::device_csrmv_analysis(const csr_matrix& A, csrmv_algorithm alg, csrmv_descr* descr)
{
    ROUTINE_TRACE("linalg::device_csrmv_analysis");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrmv_analysis(A.get_m(),
                                      A.get_n(),
                                      A.get_nnz(),
                                      A.get_row_ptr(),
                                      A.get_col_ind(),
                                      A.get_val(),
                                      alg,
                                      descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}
void linalg::device_csrmv_solve(double                alpha,
                                const csr_matrix&     A,
                                const vector<double>& x,
                                double                beta,
                                vector<double>&       y,
                                csrmv_algorithm       alg,
                                const csrmv_descr*    descr)
{
    ROUTINE_TRACE("linalg::device_csrmv_solve");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrmv_solve(A.get_m(),
                                   A.get_n(),
                                   A.get_nnz(),
                                   alpha,
                                   A.get_row_ptr(),
                                   A.get_col_ind(),
                                   A.get_val(),
                                   x.get_vec(),
                                   beta,
                                   y.get_vec(),
                                   alg,
                                   descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::allocate_csrgeam_device_data(csrgeam_descr* descr)
{
    ROUTINE_TRACE("linalg::allocate_csrgeam_device_data");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(allocate_csrgeam_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}
void linalg::free_csrgeam_device_data(csrgeam_descr* descr)
{
    ROUTINE_TRACE("linalg::free_csrgeam_device_data");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(free_csrgeam_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::device_csrgeam_nnz(const csr_matrix& A,
                                const csr_matrix& B,
                                csr_matrix&       C,
                                csrgeam_algorithm alg,
                                csrgeam_descr*    descr)
{
    ROUTINE_TRACE("linalg::device_csrgeam_nnz");

    if constexpr(is_cuda_available())
    {
        int nnz_C;
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
        C.resize(C.get_m(), C.get_n(), nnz_C);
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}
void linalg::device_csrgeam_solve(double               alpha,
                                  const csr_matrix&    A,
                                  double               beta,
                                  const csr_matrix&    B,
                                  csr_matrix&          C,
                                  csrgeam_algorithm    alg,
                                  const csrgeam_descr* descr)
{
    ROUTINE_TRACE("linalg::device_csrgeam_solve");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrgeam_solve(C.get_m(),
                                     C.get_n(),
                                     A.get_nnz(),
                                     B.get_nnz(),
                                     C.get_nnz(),
                                     descr,
                                     alpha,
                                     A.get_row_ptr(),
                                     A.get_col_ind(),
                                     A.get_val(),
                                     beta,
                                     B.get_row_ptr(),
                                     B.get_col_ind(),
                                     B.get_val(),
                                     C.get_row_ptr(),
                                     C.get_col_ind(),
                                     C.get_val()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::allocate_csrgemm_device_data(csrgemm_descr* descr)
{
    ROUTINE_TRACE("linalg::allocate_csrgemm_device_data");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(allocate_csrgemm_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}
void linalg::free_csrgemm_device_data(csrgemm_descr* descr)
{
    ROUTINE_TRACE("linalg::free_csrgemm_device_data");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(free_csrgemm_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::device_csrgemm_nnz(const csr_matrix& A,
                                const csr_matrix& B,
                                const csr_matrix& D,
                                csr_matrix&       C,
                                csrgemm_algorithm alg,
                                csrgemm_descr*    descr)
{
    ROUTINE_TRACE("linalg::device_csrgemm_nnz");

    if constexpr(is_cuda_available())
    {
        int nnz_C;
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

        C.resize(C.get_m(), C.get_n(), nnz_C);
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}
void linalg::device_csrgemm_solve(double               alpha,
                                  const csr_matrix&    A,
                                  const csr_matrix&    B,
                                  double               beta,
                                  const csr_matrix&    D,
                                  csr_matrix&          C,
                                  csrgemm_algorithm    alg,
                                  const csrgemm_descr* descr)
{
    ROUTINE_TRACE("linalg::device_csrgemm_solve");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrgemm_solve(C.get_m(),
                                     C.get_n(),
                                     B.get_m(),
                                     A.get_nnz(),
                                     B.get_nnz(),
                                     0,
                                     C.get_nnz(),
                                     descr,
                                     alpha,
                                     A.get_row_ptr(),
                                     A.get_col_ind(),
                                     A.get_val(),
                                     B.get_row_ptr(),
                                     B.get_col_ind(),
                                     B.get_val(),
                                     beta,
                                     nullptr,
                                     nullptr,
                                     nullptr,
                                     C.get_row_ptr(),
                                     C.get_col_ind(),
                                     C.get_val()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::allocate_csric0_device_data(csric0_descr* descr)
{
    ROUTINE_TRACE("linalg::allocate_csric0_device_data");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(allocate_csric0_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::free_csric0_device_data(csric0_descr* descr)
{
    ROUTINE_TRACE("linalg::free_csric0_device_data");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(free_csric0_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::device_csric0_analysis(const csr_matrix& A, csric0_descr* descr)
{
    ROUTINE_TRACE("linalg::device_csric0_analysis");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csric0_analysis(A.get_m(),
                                       A.get_n(),
                                       A.get_nnz(),
                                       A.get_row_ptr(),
                                       A.get_col_ind(),
                                       A.get_val(),
                                       descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}
void linalg::device_csric0_compute(csr_matrix& A, const csric0_descr* descr)
{
    ROUTINE_TRACE("linalg::device_csric0_compute");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csric0_compute(A.get_m(),
                                      A.get_n(),
                                      A.get_nnz(),
                                      A.get_row_ptr(),
                                      A.get_col_ind(),
                                      A.get_val(),
                                      descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::allocate_csrilu0_device_data(csrilu0_descr* descr)
{
    ROUTINE_TRACE("linalg::allocate_csrilu0_device_data");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(allocate_csrilu0_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::free_csrilu0_device_data(csrilu0_descr* descr)
{
    ROUTINE_TRACE("linalg::free_csrilu0_device_data");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(free_csrilu0_cuda_data(descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::device_csrilu0_analysis(const csr_matrix& A, csrilu0_descr* descr)
{
    ROUTINE_TRACE("linalg::device_csrilu0_analysis");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrilu0_analysis(A.get_m(),
                                        A.get_n(),
                                        A.get_nnz(),
                                        A.get_row_ptr(),
                                        A.get_col_ind(),
                                        A.get_val(),
                                        descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}
void linalg::device_csrilu0_compute(csr_matrix& A, const csrilu0_descr* descr)
{
    ROUTINE_TRACE("linalg::device_csrilu0_compute");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_csrilu0_compute(A.get_m(),
                                       A.get_n(),
                                       A.get_nnz(),
                                       A.get_row_ptr(),
                                       A.get_col_ind(),
                                       A.get_val(),
                                       descr));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}

void linalg::device_tridiagonal_solver(int                  m,
                                       int                  n,
                                       const vector<float>& lower_diag,
                                       const vector<float>& main_diag,
                                       const vector<float>& upper_diag,
                                       const vector<float>& b,
                                       vector<float>&       x)
{
    ROUTINE_TRACE("linalg::device_tridiagonal_solver");

    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_tridiagonal_solver(m,
                                          n,
                                          lower_diag.get_vec(),
                                          main_diag.get_vec(),
                                          upper_diag.get_vec(),
                                          b.get_vec(),
                                          x.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
        return;
    }
}
