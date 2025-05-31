//********************************************************************************
//
// MIT License
//
// Copyright(c) 2024 James Sandham
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

#ifndef SOR_H
#define SOR_H

#include "../../linalglib_export.h"

#include "../iter_control.h"

#include "../../vector.h"
#include "../../csr_matrix.h"

/*! \file
 *  \brief sor.h provides interface for sor solver
 */

/**
 * @brief Performs the Successive Over-Relaxation (SOR) iterative method for solving a linear system of equations.
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
 * @param[in] omega The relaxation parameter. 1 < omega < 2 for over-relaxation, 0 < omega < 1 for under-relaxation.
 * @param[in] control Structure containing iteration control parameters.
 *
 * @return Returns 0 if the iteration converges, and 1 otherwise.
 *
 * @details
 * The Successive Over-Relaxation (SOR) method is a variant of the Gauss-Seidel method for solving a system of linear
 * equations.  It introduces a relaxation parameter, \f$\omega\f$, to accelerate convergence.
 * For a matrix equation
 *
 * \f$ A\mathbf{x} = \mathbf{b}, \f$
 *
 * where \f$ A \f$ is a known square matrix of size \f$n \times n\f$, \f$\mathbf{b}\f$ is a known vector of length \f$n\f$,
 * and \f$\mathbf{x}\f$ is an unknown vector of length \f$n\f$ that we want to solve for, the SOR method
 * iteratively refines an initial guess for \f$\mathbf{x}\f$ until convergence.
 *
 * **Derivation of the SOR Iteration**
 *
 * We start with the Gauss-Seidel iteration:
 *
 * \f$ x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right), \quad i = 1, 2, \dots, n. \f$
 *
 * The SOR method introduces a weighted average between the previous iteration \f$x_i^{(k)}\f$ and the Gauss-Seidel result:
 *
 * \f$ x_i^{(k+1)} = (1 - \omega) x_i^{(k)} + \omega \frac{1}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right), \quad i = 1, 2, \dots, n. \f$
 *
 * Where \f$\omega\f$ is the relaxation parameter.
 * - If \f$\omega = 1\f$, the SOR method reduces to the Gauss-Seidel method.
 * - If \f$1 < \omega < 2\f$, the method is called over-relaxation.
 * - If \f$0 < \omega < 1\f$, the method is called under-relaxation.
 *
 * The optimal value of \f$\omega\f$ depends on the properties of the matrix \f$A\f$.
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
 * This implementation efficiently calculates the matrix-vector products using the CSR format.
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
 * double omega = 1.2; // Example value for omega
 *
 * // Solve the system using SOR iteration
 * int result = sor(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), n, omega, control);
 *
 * if (result == 0) {
 * std::cout << "SOR iteration converged. Solution x = ";
 * for (double val : x) {
 * std::cout << val << " ";
 * }
 * std::cout << std::endl;
 * } else {
 * std::cout << "SOR iteration did not converge." << std::endl;
 * }
 *
 * return 0;
 * }
 * @endcode
 *
 */

class sor_solver
{
private:
    vector res;

public:
    sor_solver();
    ~sor_solver();

    sor_solver (const sor_solver&) = delete;
    sor_solver& operator= (const sor_solver&) = delete;

    void build(const csr_matrix& A);
    int solve(const csr_matrix& A, vector& x, const vector& b, iter_control control, double omega);
};

#endif