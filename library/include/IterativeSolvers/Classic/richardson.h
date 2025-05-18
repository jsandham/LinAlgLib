//********************************************************************************
//
// MIT License
//
// Copyright(c) 2019 James Sandham
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

#ifndef RICHARDSON_H
#define RICHARDSON_H

#include "../../linalglib_export.h"

#include "../iter_control.h"

/*! \file
 *  \brief richardson.h provides interface for richardson solver
 */

/**
 * @brief Performs the Richardson iterative method for solving a linear system of equations.
 *
 * Solves the linear system Ax = b, where A is a sparse matrix represented in Compressed Sparse Row (CSR) format.
 *
 * @param[in] csr_row_ptr Array of size (n+1) containing row pointers for the CSR matrix.
 * csr_row_ptr[i] stores the index of the first non-zero element in row i.
 * csr_row_ptr[n] stores the number of non-zero elements in the matrix.
 * @param[in] csr_col_ind Array of size (number of non-zero elements) containing the column indices
 * for the non-zero elements of the CSR matrix.
 * @param[in] csr_val Array of size (number of non-zero elements) containing the values
 * of the non-zero elements of the CSR matrix.
 * @param[in,out] x Array of size n. On input, it contains the initial guess for the solution.
 * On output, it contains the computed solution.
 * @param[in] b Array of size n containing the right-hand side vector.
 * @param[in] n The order of the square matrix A.
 * @param[in] theta The relaxation parameter.
 * @param[in] control Structure containing iteration control parameters.
 *
 * @return Returns 0 if the iteration converges, and 1 otherwise.
 *
 * @details
 * The Richardson method is a basic iterative method for solving a system of linear equations.
 * For a matrix equation
 *
 * \f$ A\mathbf{x} = \mathbf{b}, \f$
 *
 * where \f$ A \f$ is a known square matrix of size \f$n \times n\f$, \f$\mathbf{b}\f$ is a known vector of length \f$n\f$,
 * and \f$\mathbf{x}\f$ is an unknown vector of length \f$n\f$ that we want to solve for, the Richardson method
 * iteratively refines an initial guess for \f$\mathbf{x}\f$ until convergence.
 *
 * **Derivation of the Richardson Iteration**
 *
 * We start with the linear system:
 *
 * \f$ A\mathbf{x} = \mathbf{b}. \f$
 *
 * We can rewrite this as:
 *
 * \f$ \mathbf{x} = \mathbf{x} + (\mathbf{b} - A\mathbf{x}). \f$
 *
 * Multiplying the residual by a relaxation parameter \f$\theta\f$, we get the iterative scheme:
 *
 * \f$ \mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \theta (\mathbf{b} - A\mathbf{x}^{(k)}), \f$
 *
 * where \f$\mathbf{x}^{(k)}\f$ is the \f$k\f$-th approximation of the solution \f$\mathbf{x}\f$, and \f$\theta\f$ is a parameter
 * that controls the rate of convergence.  The choice of \f$\theta\f$ is crucial for the convergence of the method.
 *
 * The iteration continues until a stopping criterion is met, such as the residual norm being sufficiently small
 * or the maximum number of iterations being reached.
 *
 * **CSR Implementation Details**
 *
 * The function operates on a matrix stored in CSR format, which is an efficient storage scheme for sparse matrices.
 * The CSR format uses three arrays to represent the matrix:
 *
 * - `csr_row_ptr`: Stores the starting index of each row in the `csr_col_ind` and `csr_val` arrays.
 * - `csr_col_ind`: Stores the column indices of the non-zero elements.
 * - `csr_val`: Stores the values of the non-zero elements.
 *
 * This implementation efficiently calculates the matrix-vector product \f$A\mathbf{x}^{(k)}\f$ using the CSR format.
 *
 * **Code Example**
 *
 * @code
 * #include <iostream>
 * #include <vector>
 * #include <cmath>
 *
 * int main() {
 * // Example usage:
 * // Define the CSR matrix A
 * int n = 4;
 * std::vector<int> csr_row_ptr = {0, 2, 5, 7, 9};
 * std::vector<int> csr_col_ind = {0, 1, 0, 1, 2, 1, 3, 2, 3};
 * std::vector<double> csr_val = {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, 4.0};
 *
 * // Define the right-hand side vector b
 * std::vector<double> b = {3.0, 3.0, 3.0, 3.0};
 *
 * // Define the initial guess for x
 * std::vector<double> x(n, 0.0);
 *
 * // Define the iteration control parameters
 * iter_control control;
 * double theta = 0.1; // Example value for theta
 *
 * // Solve the system using Richardson iteration
 * int result = rich(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), n, theta, control);
 *
 * if (result == 0) {
 * std::cout << "Richardson iteration converged. Solution x = ";
 * for (double val : x) {
 * std::cout << val << " ";
 * }
 * std::cout << std::endl;
 * } else {
 * std::cout << "Richardson iteration did not converge." << std::endl;
 * }
 *
 * return 0;
 * }
 * @endcode
 *
 */
LINALGLIB_API int rich(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
         double theta, iter_control control);

#endif
