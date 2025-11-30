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
#include "linalg_enums.h"
#include "linalg_export.h"
#include "vector.h"

/*! \file
 *  \brief device_math.h provides linear algebra APIs for device (GPU) backend
 */
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

    // Extract lower triangular entries
    void device_extract_lower_triangular_nnz(const csr_matrix& A, csr_matrix& L, int& nnz_L);
    void device_extract_lower_triangular(const csr_matrix& A, csr_matrix& L);

    // Extract upper triangular entries
    void device_extract_upper_triangular_nnz(const csr_matrix& A, csr_matrix& U, int& nnz_U);
    void device_extract_upper_triangular(const csr_matrix& A, csr_matrix& U);

    // Scale diagonal entries
    void device_scale_diagonal(csr_matrix& A, double scalar);

    // Scale by inverse diagonal entries
    void device_scale_by_inverse_diagonal(csr_matrix& A, const vector<double>& diag);

    // Euclidean norm
    double device_norm_euclid(const vector<double>& array);

    // Infinity norm
    double device_norm_inf(const vector<double>& array);

    // Jacobi solve
    void device_jacobi_solve(const vector<double>& rhs,
                             const vector<double>& diag,
                             vector<double>&       x);

    // SSOR fill lower preconditioner
    void device_ssor_fill_lower_precond(const csr_matrix& A, csr_matrix& L, double omega);
    // SSOR fill upper preconditioner
    void device_ssor_fill_upper_precond(const csr_matrix& A, csr_matrix& U, double omega);

    struct csrtrsv_descr;

    void allocate_csrtrsv_device_data(csrtrsv_descr* descr);
    void free_csrtrsv_device_data(csrtrsv_descr* descr);

    void device_csrtrsv_analysis(const csr_matrix& A,
                                 triangular_type   tri_type,
                                 diagonal_type     diag_type,
                                 csrtrsv_descr*    descr);
    void device_csrtrsv_solve(const csr_matrix&     A,
                              const vector<double>& b,
                              vector<double>&       x,
                              double                alpha,
                              triangular_type       tri_type,
                              diagonal_type         diag_type,
                              const csrtrsv_descr*  descr);

} // namespace linalg

#endif