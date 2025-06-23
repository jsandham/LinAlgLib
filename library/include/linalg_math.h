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

#include "linalg_export.h"
#include "scalar.h"
#include "vector.h"
#include "csr_matrix.h"

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
    LINALGLIB_API void axpy(const scalar<double>& alpha, const vector<double>& x, vector<double>& y);

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
    LINALGLIB_API void axpby(const scalar<double>& alpha, const vector<double>& x, const scalar<double>& beta, vector<double>& y);



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
    LINALGLIB_API void axpbypgz(double alpha, const vector<double>& x, double beta, const vector<double>& y, double gamma, vector<double>& z);
    LINALGLIB_API void axpbypgz(const scalar<double>& alpha, const vector<double>& x, const scalar<double>& beta, const vector<double>& y, const scalar<double>& gamma, vector<double>& z);

    /**
     * @brief Computes the matrix-vector product: \f$y = A \cdot x\f$.
     *
     * This function performs a sparse matrix-vector multiplication where \f$A\f$ is a `csr_matrix`.
     *
     * @param A The input `csr_matrix` \f$A\f$.
     * @param x The input vector \f$x\f$.
     * @param y The output vector \f$y\f$ to store the result of the multiplication.
     */
    LINALGLIB_API void matrix_vector_product(const csr_matrix& A, const vector<double>& x, vector<double>& y);

    /**
     * @brief Computes the generalized matrix-vector product: \f$y = \alpha \cdot A \cdot x + \beta \cdot y\f$.
     *
     * This function performs a sparse matrix-vector multiplication, scales the result, and
     * accumulates it with a scaled version of vector `y`.
     *
     * @param alpha The scalar multiplier \f$\alpha\f$.
     * @param A The input `csr_matrix` \f$A\f$.
     * @param x The input vector \f$x\f$.
     * @param beta The scalar multiplier \f$\beta\f$.
     * @param y The input/output vector \f$y\f$. On input, it contains the initial values; on output,
     * it contains the result of the generalized matrix-vector product.
     */
    LINALGLIB_API void matrix_vector_product(double alpha, const csr_matrix& A, const vector<double>& x, double beta, vector<double>& y);

    /**
     * @brief Computes the matrix-matrix product: \f$C = A \cdot B\f$.
     *
     * This function performs a sparse matrix-matrix multiplication where \f$A\f$, \f$B\f$, and \f$C\f$ are `csr_matrix` objects.
     *
     * @param C The output `csr_matrix` to store the product \f$A \cdot B\f$.
     * @param A The left-hand side input `csr_matrix` \f$A\f$.
     * @param B The right-hand side input `csr_matrix` \f$B\f$.
     */
    LINALGLIB_API void matrix_matrix_product(csr_matrix& C, const csr_matrix& A, const csr_matrix& B);

    /**
     * @brief Computes the matrix-matrix addition: \f$C = A + B\f$.
     *
     * This function performs the element-wise addition of two sparse matrices \f$A\f$ and \f$B\f$,
     * storing the result in matrix \f$C\f$. All three matrices are in CSR format.
     * It is assumed that matrices \f$A\f$ and \f$B\f$ have the same dimensions.
     *
     * @param C The output `csr_matrix` to store the sum \f$A + B\f$.
     * @param A The left-hand side input `csr_matrix` \f$A\f$.
     * @param B The right-hand side input `csr_matrix` \f$B\f$.
     */
    LINALGLIB_API void matrix_matrix_addition(csr_matrix& C, const csr_matrix& A, const csr_matrix& B);

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
     * @brief Performs a forward substitution solve for \f$A \cdot x = b\f$.
     *
     * This function solves a lower triangular system \f$A \cdot x = b\f$, where \f$A\f$ is a `csr_matrix`.
     *
     * @param A The input lower triangular `csr_matrix` \f$A\f$.
     * @param b The input right-hand side vector \f$b\f$.
     * @param x The output vector \f$x\f$ to store the solution.
     * @param unit_diag A boolean flag indicating whether the diagonal of \f$A\f$ is assumed to be unit (1.0).
     * If `true`, division by diagonal elements is skipped.
     */
    LINALGLIB_API void forward_solve(const csr_matrix& A, const vector<double>& b, vector<double>& x, bool unit_diag);

    /**
     * @brief Performs a backward substitution solve for \f$A \cdot x = b\f$.
     *
     * This function solves an upper triangular system \f$A \cdot x = b\f$, where \f$A\f$ is a `csr_matrix`.
     *
     * @param A The input upper triangular `csr_matrix` \f$A\f$.
     * @param b The input right-hand side vector \f$b\f$.
     * @param x The output vector \f$x\f$ to store the solution.
     * @param unit_diag A boolean flag indicating whether the diagonal of \f$A\f$ is assumed to be unit (1.0).
     * If `true`, division by diagonal elements is skipped.
     */
    LINALGLIB_API void backward_solve(const csr_matrix& A, const vector<double>& b, vector<double>& x, bool unit_diag);

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
    LINALGLIB_API void dot_product(const vector<double>& x, const vector<double>& y, scalar<double>& result);

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
    LINALGLIB_API void compute_residual(const csr_matrix& A, const vector<double>& x, const vector<double>& b, vector<double>& res);

    /**
     * @brief Performs an exclusive scan (prefix sum) on a vector.
     *
     * This function replaces each element with the sum of all elements preceding it.
     * The first element of the output vector will be 0.
     * For example, if input `x` is `{a, b, c}`, output `x` will be `{0, a, a+b}`.
     *
     * @param x The input/output vector. On input, it contains the original values; on output,
     * it contains the exclusive scan results.
     */
    LINALGLIB_API void exclusize_scan(vector<double>& x);

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

    // Fill array with zeros
    LINALGLIB_API void fill_with_zeros(vector<uint32_t> &vec);
    LINALGLIB_API void fill_with_zeros(vector<int32_t> &vec);
    LINALGLIB_API void fill_with_zeros(vector<int64_t> &vec);
    LINALGLIB_API void fill_with_zeros(vector<double> &vec);

    // Fill array with ones
    LINALGLIB_API void fill_with_ones(vector<uint32_t> &vec);
    LINALGLIB_API void fill_with_ones(vector<int32_t> &vec);
    LINALGLIB_API void fill_with_ones(vector<int64_t> &vec);
    LINALGLIB_API void fill_with_ones(vector<double> &vec);

    // Copy array
    LINALGLIB_API void copy(vector<uint32_t> &dest, const vector<uint32_t> &src);
    LINALGLIB_API void copy(vector<int32_t> &dest, const vector<int32_t> &src);
    LINALGLIB_API void copy(vector<int64_t> &dest, const vector<int64_t> &src);
    LINALGLIB_API void copy(vector<double> &dest, const vector<double> &src);


    // /**
    //  * @brief Fills a vector with zeros.
    //  * @tparam T The type of elements in the vector.
    //  * @param vec The vector to be filled with zeros.
    //  */
    // template<typename T>
    // LINALGLIB_API void fill_with_zeros(vector<T>& vec);

    // /**
    //  * @brief Fills a vector with ones.
    //  * @tparam T The type of elements in the vector.
    //  * @param vec The vector to be filled with ones.
    //  */
    // template<typename T>
    // LINALGLIB_API void fill_with_ones(vector<T>& vec);

    // /**
    //  * @brief Copies the elements from a source vector to a destination vector.
    //  * @tparam T The type of elements in the vectors.
    //  * @param dest The destination vector.
    //  * @param src The source vector from which to copy.
    //  */
    // template<typename T>
    // LINALGLIB_API void copy(vector<T>& dest, const vector<T>& src);
}

#endif