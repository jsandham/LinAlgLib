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

#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <string>

#include "csr_matrix.h"
#include "linalg_enums.h"
#include "linalg_export.h"
#include "linalg_types.h"
#include "vector.h"

/*! \file
 *  \brief linalg_math.h provides linear algebra APIs
 */

namespace linalg
{
    /**
     * @brief Computes the DAXPY operation: \f$y = \alpha \cdot x + y\f$.
     *
     * @param alpha The scalar multiplier \f$\alpha\f$.
     * @param x The input vector \f$x\f$.
     * @param y The input/output vector \f$y\f$. On input, it contains the initial values; on output,
     * it contains the result of the DAXPY operation.
     */
    LINALGLIB_API void axpy(double alpha, const vector<double>& x, vector<double>& y);

    /**
     * @brief Computes the DAXPBY operation: \f$y = \alpha \cdot x + \beta \cdot y\f$.
     *
     * @param alpha The scalar multiplier \f$\alpha\f$.
     * @param x The input vector \f$x\f$.
     * @param beta The scalar multiplier \f$\beta\f$.
     * @param y The input/output vector \f$y\f$. On input, it contains the initial values; on output,
     * it contains the result of the DAXPBY operation.
     */
    LINALGLIB_API void axpby(double alpha, const vector<double>& x, double beta, vector<double>& y);

    /**
     * @brief Computes the DAXPBYPGZ operation: \f$z = \alpha \cdot x + \beta \cdot y + \gamma \cdot z\f$.
     *
     * @param alpha The scalar multiplier \f$\alpha\f$.
     * @param x The input vector \f$x\f$.
     * @param beta The scalar multiplier \f$\beta\f$.
     * @param y The input vector \f$y\f$.
     * @param gamma The scalar multiplier \f$\gamma\f$.
     * @param z The input/output vector \f$z\f$. On input, it contains the initial values; on output,
     * it contains the result of the DAXPBYPGZ operation.
     */
    LINALGLIB_API void axpbypgz(double                alpha,
                                const vector<double>& x,
                                double                beta,
                                const vector<double>& y,
                                double                gamma,
                                vector<double>&       z);

    /**
     * @brief Performs an Incomplete Cholesky (IC) factorization with zero fill-in (IC(0)).
     *
     * This function computes the incomplete Cholesky factorization \f$LL^T \approx A\f$ for a symmetric positive-definite matrix \f$A\f$.
     * Only the lower triangular part \f$L\f$ is stored.
     *
     * @param LL The output `csr_matrix` representing the incomplete Cholesky factor \f$L\f$.
     * @param structural_zero An optional pointer to an integer that will be set to 1 if a structural zero
     * is encountered during factorization, 0 otherwise. Can be `nullptr`.
     * @param numeric_zero An optional pointer to an integer that will be set to 1 if a numeric zero
     * (pivot element is zero) is encountered during factorization, 0 otherwise. Can be `nullptr`.
     */
    LINALGLIB_API void csric0(csr_matrix& LL, int* structural_zero, int* numeric_zero);

    /**
     * @brief Performs an Incomplete LU (ILU) factorization with zero fill-in (ILU(0)).
     *
     * This function computes the incomplete LU factorization \f$LU \approx A\f$ for a given matrix \f$A\f$.
     * The factors \f$L\f$ (lower triangular with unit diagonal) and \f$U\f$ (upper triangular) are stored in a single `csr_matrix` `LU`.
     *
     * @param LU The output `csr_matrix` representing the incomplete LU factors.
     * @param structural_zero An optional pointer to an integer that will be set to 1 if a structural zero
     * is encountered during factorization, 0 otherwise. Can be `nullptr`.
     * @param numeric_zero An optional pointer to an integer that will be set to 1 if a numeric zero
     * (pivot element is zero) is encountered during factorization, 0 otherwise. Can be `nullptr`.
     */
    LINALGLIB_API void csrilu0(csr_matrix& LU, int* structural_zero, int* numeric_zero);

    /**
     * @brief Computes the transpose of a CSR matrix.
     *
     * This function computes \f$transposeA = A^T\f$.
     *
     * @param A The input `csr_matrix` to transpose.
     * @param transposeA The output `csr_matrix` to store the transposed matrix.
     */
    LINALGLIB_API void transpose_matrix(const csr_matrix& A, csr_matrix& transposeA);

    /**
     * @brief Computes the dot product of two vectors.
     *
     * This function calculates the scalar dot product (inner product) of two vectors \f$x\f$ and \f$y\f$:
     * \f$x \cdot y = \sum_{i=0}^{n-1} x_i \cdot y_i\f$.
     *
     * @param x The first input vector.
     * @param y The second input vector.
     * @return The double-precision floating-point result of the dot product.
     */
    LINALGLIB_API double dot_product(const vector<double>& x, const vector<double>& y);

    /**
     * @brief Computes the residual vector for a linear system: \f$res = b - A \cdot x\f$.
     *
     * This function calculates the residual vector for a given matrix \f$A\f$, solution vector \f$x\f$, and right-hand side vector \f$b\f$.
     *
     * @param A The input `csr_matrix` \f$A\f$.
     * @param x The input solution vector \f$x\f$.
     * @param b The input right-hand side vector \f$b\f$.
     * @param res The output vector to store the computed residual $res$.
     */
    LINALGLIB_API void compute_residual(const csr_matrix&     A,
                                        const vector<double>& x,
                                        const vector<double>& b,
                                        vector<double>&       res);

    /**
     * @brief Extracts the diagonal entries of a CSR matrix.
     *
     * @param A The input `csr_matrix` from which to extract the diagonal.
     * @param d The output vector that will store the diagonal elements.
     */
    LINALGLIB_API void diagonal(const csr_matrix& A, vector<double>& d);

    /**
     * @brief Computes the Euclidean (L2) norm of a vector.
     *
     * The Euclidean norm is calculated as \f$\sqrt{\sum_{i=0}^{n-1} |array_i|^2}\f$.
     *
     * @param array The input vector.
     * @return The double-precision floating-point value of the Euclidean norm.
     */
    LINALGLIB_API double norm_euclid(const vector<double>& array);

    /**
     * @brief Computes the infinity (maximum absolute value) norm of a vector.
     *
     * The infinity norm is calculated as \f$\max_{i} |array_i|\f$.
     *
     * @param array The input vector.
     * @return The double-precision floating-point value of the infinity norm.
     */
    LINALGLIB_API double norm_inf(const vector<double>& array);

    /**
     * @brief Performs a Jacobi diagonal solve: x = rhs ./ diag (element-wise division).
     *
     * @details
     * This helper computes the solution for a diagonal system where the matrix is
     * represented by its diagonal entries. For each index i:
     * \f[ x_i = \frac{rhs_i}{diag_i} \f]
     *
     * The caller is responsible for ensuring that `diag` contains no zeros; division
     * by zero will lead to undefined behavior. Both input vectors (`rhs`, `diag`)
     * must have the same length as the output vector `x`.
     *
     * @param rhs Right-hand side vector b (input).
     * @param diag Diagonal entries of the matrix (input). Must be non-zero.
     * @param x Solution vector (output). On return, contains the element-wise
     *          division result rhs ./ diag.
     */
    LINALGLIB_API void
        jacobi_solve(const vector<double>& rhs, const vector<double>& diag, vector<double>& x);

    /**
     * @brief Creates an opaque descriptor for CSR triangular solve operations.
     *
     * @param descr A pointer to a csrtrsv_descr* that will be initialized to point
     * to a newly allocated descriptor.
     * @see destroy_csrtrsv_descr
     */
    LINALGLIB_API void create_csrtrsv_descr(csrtrsv_descr** descr);

    /**
     * @brief Destroys a CSR triangular solve descriptor.
     *
     * Frees all resources associated with the descriptor. After this call,
     * the descr pointer should not be used.
     *
     * @param descr The descriptor to destroy. Can be nullptr.
     * @see create_csrtrsv_descr
     */
    LINALGLIB_API void destroy_csrtrsv_descr(csrtrsv_descr* descr);

    /**
     * @brief Performs analysis for CSR triangular solve.
     *
     * Preprocesses a triangular matrix to prepare for subsequent solve operations.
     * This may compute reordering, factorization, or other data needed for efficient solving.
     *
     * @param A The input triangular `csr_matrix`.
     * @param tri_type Specifies whether the matrix is lower or upper triangular.
     * @param diag_type Specifies whether the diagonal is unit (1.0) or general.
     * @param descr The descriptor to populate with analysis results.
     * @see csrtrsv_solve
     */
    LINALGLIB_API void csrtrsv_analysis(const csr_matrix& A,
                                        triangular_type   tri_type,
                                        diagonal_type     diag_type,
                                        csrtrsv_descr*    descr);

    /**
     * @brief Solves a triangular system \f$A \cdot x = \alpha \cdot b\f$ using CSR format.
     *
     * Solves the equation using preprocessing information from a prior csrtrsv_analysis() call.
     *
     * @param A The triangular `csr_matrix`.
     * @param b The input right-hand side vector.
     * @param x The output solution vector.
     * @param alpha A scaling factor applied to the right-hand side.
     * @param tri_type Specifies whether the matrix is lower or upper triangular.
     * @param diag_type Specifies whether the diagonal is unit (1.0) or general.
     * @param descr The descriptor populated by a prior csrtrsv_analysis() call.
     * @see csrtrsv_analysis
     */
    LINALGLIB_API void csrtrsv_solve(const csr_matrix&     A,
                                     const vector<double>& b,
                                     vector<double>&       x,
                                     double                alpha,
                                     triangular_type       tri_type,
                                     diagonal_type         diag_type,
                                     const csrtrsv_descr*  descr);

    /**
     * @brief Creates an opaque descriptor for CSR matrix-vector product operations.
     *
     * @param descr A pointer to a csrmv_descr* that will be initialized to point
     * to a newly allocated descriptor.
     * @see destroy_csrmv_descr
     */
    LINALGLIB_API void create_csrmv_descr(csrmv_descr** descr);

    /**
     * @brief Destroys a CSR matrix-vector product descriptor.
     *
     * Frees all resources associated with the descriptor. After this call,
     * the descr pointer should not be used.
     *
     * @param descr The descriptor to destroy. Can be nullptr.
     * @see create_csrmv_descr
     */
    LINALGLIB_API void destroy_csrmv_descr(csrmv_descr* descr);

    /**
     * @brief Performs analysis for CSR matrix-vector product.
     *
     * Preprocesses a matrix to prepare for subsequent matrix-vector product operations.
     * May involve inspecting sparsity patterns or data layout optimization.
     *
     * @param A The input `csr_matrix`.
     * @param alg The algorithm selection for the multiplication.
     * @param descr The descriptor to populate with analysis results.
     * @see csrmv_solve
     */
    LINALGLIB_API void csrmv_analysis(const csr_matrix& A, csrmv_algorithm alg, csrmv_descr* descr);

    /**
     * @brief Computes a CSR matrix-vector product \f$y = \alpha A x + \beta y\f$.
     *
     * Performs matrix-vector multiplication using preprocessing from csrmv_analysis().
     *
     * @param alpha Scaling factor for \f$Ax\f$.
     * @param A The `csr_matrix`.
     * @param x The input vector.
     * @param beta Scaling factor for \f$y\f$.
     * @param y The output vector.
     * @param alg The algorithm selection for the multiplication.
     * @param descr The descriptor populated by a prior csrmv_analysis() call.
     * @see csrmv_analysis
     */
    LINALGLIB_API void csrmv_solve(double                alpha,
                                   const csr_matrix&     A,
                                   const vector<double>& x,
                                   double                beta,
                                   vector<double>&       y,
                                   csrmv_algorithm       alg,
                                   const csrmv_descr*    descr);

    /**
     * @brief Creates an opaque descriptor for CSR matrix-matrix addition operations.
     *
     * @param descr A pointer to a csrgeam_descr* that will be initialized to point
     * to a newly allocated descriptor.
     * @see destroy_csrgeam_descr
     */
    LINALGLIB_API void create_csrgeam_descr(csrgeam_descr** descr);

    /**
     * @brief Destroys a CSR matrix-matrix addition descriptor.
     *
     * Frees all resources associated with the descriptor. After this call,
     * the descr pointer should not be used.
     *
     * @param descr The descriptor to destroy. Can be nullptr.
     * @see create_csrgeam_descr
     */
    LINALGLIB_API void destroy_csrgeam_descr(csrgeam_descr* descr);

    /**
     * @brief Computes the sparsity pattern for CSR matrix-matrix addition \f$C = \alpha A + \beta B\f$.
     *
     * First stage: determines the number of nonzeros and sparsity structure of the result.
     * Must be called before csrgeam_solve().
     *
     * @param A The first input `csr_matrix`.
     * @param B The second input `csr_matrix`.
     * @param C The output `csr_matrix` (row and column indices allocated, values set by csrgeam_solve()).
     * @param alg The algorithm selection.
     * @param descr The descriptor to populate with analysis results.
     * @see csrgeam_solve
     */
    LINALGLIB_API void csrgeam_nnz(const csr_matrix& A,
                                   const csr_matrix& B,
                                   csr_matrix&       C,
                                   csrgeam_algorithm alg,
                                   csrgeam_descr*    descr);

    /**
     * @brief Computes CSR matrix-matrix addition \f$C = \alpha A + \beta B\f$.
     *
     * Second stage: computes the values of the result matrix.
     * csrgeam_nnz() must be called first to allocate the sparsity structure.
     *
     * @param alpha Scaling factor for matrix A.
     * @param A The first input `csr_matrix`.
     * @param beta Scaling factor for matrix B.
     * @param B The second input `csr_matrix`.
     * @param C The output `csr_matrix` with structure allocated by csrgeam_nnz().
     * @param alg The algorithm selection.
     * @param descr The descriptor populated by a prior csrgeam_nnz() call.
     * @see csrgeam_nnz
     */
    LINALGLIB_API void csrgeam_solve(double               alpha,
                                     const csr_matrix&    A,
                                     double               beta,
                                     const csr_matrix&    B,
                                     csr_matrix&          C,
                                     csrgeam_algorithm    alg,
                                     const csrgeam_descr* descr);

    /**
     * @brief Creates an opaque descriptor for CSR matrix-matrix multiplication operations.
     *
     * @param descr A pointer to a csrgemm_descr* that will be initialized to point
     * to a newly allocated descriptor.
     * @see destroy_csrgemm_descr
     */
    LINALGLIB_API void create_csrgemm_descr(csrgemm_descr** descr);

    /**
     * @brief Destroys a CSR matrix-matrix multiplication descriptor.
     *
     * Frees all resources associated with the descriptor. After this call,
     * the descr pointer should not be used.
     *
     * @param descr The descriptor to destroy. Can be nullptr.
     * @see create_csrgemm_descr
     */
    LINALGLIB_API void destroy_csrgemm_descr(csrgemm_descr* descr);

    /**
     * @brief Computes the sparsity pattern for CSR matrix-matrix multiplication \f$C = \alpha A \cdot B + \beta D\f$.
     *
     * First stage: determines the number of nonzeros and sparsity structure of the result.
     * Must be called before csrgemm_solve().
     *
     * @param A The first input `csr_matrix`.
     * @param B The second input `csr_matrix`.
     * @param D The third input `csr_matrix` (can represent \f$\beta D\f$ term or be empty).
     * @param C The output `csr_matrix` (row and column indices allocated, values set by csrgemm_solve()).
     * @param alg The algorithm selection.
     * @param descr The descriptor to populate with analysis results.
     * @see csrgemm_solve
     */
    LINALGLIB_API void csrgemm_nnz(const csr_matrix& A,
                                   const csr_matrix& B,
                                   const csr_matrix& D,
                                   csr_matrix&       C,
                                   csrgemm_algorithm alg,
                                   csrgemm_descr*    descr);

    /**
     * @brief Computes CSR matrix-matrix multiplication \f$C = \alpha A \cdot B + \beta D\f$.
     *
     * Second stage: computes the values of the result matrix.
     * csrgemm_nnz() must be called first to allocate the sparsity structure.
     *
     * @param alpha Scaling factor for the product \f$A \cdot B\f$.
     * @param A The first input `csr_matrix`.
     * @param B The second input `csr_matrix`.
     * @param beta Scaling factor for the matrix D.
     * @param D The third input `csr_matrix` (the \f$\beta D\f$ term).
     * @param C The output `csr_matrix` with structure allocated by csrgemm_nnz().
     * @param alg The algorithm selection.
     * @param descr The descriptor populated by a prior csrgemm_nnz() call.
     * @see csrgemm_nnz
     */
    LINALGLIB_API void csrgemm_solve(double               alpha,
                                     const csr_matrix&    A,
                                     const csr_matrix&    B,
                                     double               beta,
                                     const csr_matrix&    D,
                                     csr_matrix&          C,
                                     csrgemm_algorithm    alg,
                                     const csrgemm_descr* descr);

    /**
     * @brief Creates an opaque descriptor for CSR incomplete Cholesky (IC(0)) factorization.
     *
     * @param descr A pointer to a csric0_descr* that will be initialized to point
     * to a newly allocated descriptor.
     * @see destroy_csric0_descr
     */
    LINALGLIB_API void create_csric0_descr(csric0_descr** descr);

    /**
     * @brief Destroys a CSR incomplete Cholesky (IC(0)) factorization descriptor.
     *
     * Frees all resources associated with the descriptor. After this call,
     * the descr pointer should not be used.
     *
     * @param descr The descriptor to destroy. Can be nullptr.
     * @see create_csric0_descr
     */
    LINALGLIB_API void destroy_csric0_descr(csric0_descr* descr);

    /**
     * @brief Performs analysis for CSR incomplete Cholesky (IC(0)) factorization.
     *
     * Preprocesses a matrix to prepare for subsequent IC(0) factorization.
     *
     * @param A The input `csr_matrix`.
     * @param descr The descriptor to populate with analysis results.
     * @see csric0_compute
     */
    LINALGLIB_API void csric0_analysis(const csr_matrix& A, csric0_descr* descr);

    /**
     * @brief Compute an inplace CSR incomplete Cholesky (IC(0)) factorization.
     *
     * Uses preprocessing information from a prior csric0_analysis() call.
     * @param A The `csr_matrix` to factorize in place.
     * @param descr The descriptor populated by a prior csric0_analysis() call.
     * @see csric0_analysis
     */
    LINALGLIB_API void csric0_compute(csr_matrix& A, const csric0_descr* descr);

    /**
     * @brief Creates an opaque descriptor for CSR incomplete LU (ILU(0)) factorization.
     *
     * @param descr A pointer to a csrilu0_descr* that will be initialized to point
     * to a newly allocated descriptor.
     * @see destroy_csrilu0_descr
     */
    LINALGLIB_API void create_csrilu0_descr(csrilu0_descr** descr);

    /**
     * @brief Destroys a CSR incomplete LU (ILU(0)) factorization descriptor.
     *
     * Frees all resources associated with the descriptor. After this call,
     * the descr pointer should not be used.
     *
     * @param descr The descriptor to destroy. Can be nullptr.
     * @see create_csrilu0_descr
     */
    LINALGLIB_API void destroy_csrilu0_descr(csrilu0_descr* descr);

    /**
     * @brief Performs analysis for CSR incomplete LU (ILU(0)) factorization.
     *
     * Preprocesses a matrix to prepare for subsequent ILU(0) factorization.
     *
     * @param A The input `csr_matrix`.
     * @param descr The descriptor to populate with analysis results.
     * @see csrilu0_compute
     */
    LINALGLIB_API void csrilu0_analysis(const csr_matrix& A, csrilu0_descr* descr);

    /**
     * @brief Compute an inplace CSR incomplete LU (ILU(0)) factorization.
     *
     * Uses preprocessing information from a prior csrilu0_analysis() call.
     * @param A The `csr_matrix` to factorize in place.
     * @param descr The descriptor populated by a prior csrilu0_analysis() call.
     * @see csrilu0_analysis
     */
    LINALGLIB_API void csrilu0_compute(csr_matrix& A, const csrilu0_descr* descr);

    /**
     * @brief Solves a tridiagonal system of equations using the Thomas algorithm.
     *
     * This function solves the linear system defined by a tridiagonal matrix
     * represented by its lower diagonal, main diagonal, and upper diagonal vectors.
     *
     * @param m The number of rows in the tridiagonal matrix.
     * @param n The number of columns in the right-hand-side matrix.
     * @param lower_diag The vector representing the lower diagonal of the tridiagonal matrix.
     * @param main_diag The vector representing the main diagonal of the tridiagonal matrix.
     * @param upper_diag The vector representing the upper diagonal of the tridiagonal matrix.
     * @param rhs The right-hand side vector of the linear system.
     * @param solution The output vector that will contain the solution to the system.
     */
    LINALGLIB_API void tridiagonal_solver(int                  m,
                                          int                  n,
                                          const vector<float>& lower_diag,
                                          const vector<float>& main_diag,
                                          const vector<float>& upper_diag,
                                          const vector<float>& rhs,
                                          vector<float>&       solution);
}

#endif
