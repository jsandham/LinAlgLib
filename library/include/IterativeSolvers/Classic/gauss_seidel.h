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

#ifndef GAUSS_SEIDEL_H
#define GAUSS_SEIDEL_H

#include "../../linalglib_export.h"

#include "../iter_control.h"

#include "../../vector.h"
#include "../../csr_matrix.h"

/*! \file
 *  \brief gauss_seidel.h provides interface for gauss seidel solver
 */

/**
 * @brief Performs the Gauss-Seidel iterative method for solving a linear system of equations.
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
 * @param[in] control Structure containing iteration control parameters.
 *
 * @return Returns 0 if the iteration converges, and 1 otherwise.
 *
 * @details
 * The Gauss-Seidel method is an iterative technique for solving a system of linear equations.  It is similar to the Jacobi method,
 * but it uses updated values of the solution vector \f$\mathbf{x}\f$ as soon as they are computed.
 * For a matrix equation
 *
 * \f$ A\mathbf{x} = \mathbf{b}, \f$
 *
 * where \f$ A \f$ is a known square matrix of size \f$n \times n\f$, \f$\mathbf{b}\f$ is a known vector of length \f$n\f$,
 * and \f$\mathbf{x}\f$ is an unknown vector of length \f$n\f$ that we want to solve for, the Gauss-Seidel method
 * iteratively refines an initial guess for \f$\mathbf{x}\f$ until convergence.
 *
 * **Derivation of the Gauss-Seidel Iteration**
 *
 * We can decompose the matrix \f$A\f$ into its diagonal, lower triangular, and upper triangular parts:
 *
 * \f$ A = D + L + U, \f$
 *
 * where
 *
 * \f$
 * D = \begin{pmatrix}
 * a_{11} & 0      & \cdots & 0      \\
 * 0      & a_{22} & \cdots & 0      \\
 * \vdots & \vdots & \ddots & \vdots \\
 * 0      & 0      & \cdots & a_{nn}
 * \end{pmatrix},
 * \f$
 *
 * \f$
 * L = \begin{pmatrix}
 * 0      & 0      & \cdots & 0      \\
 * a_{21} & 0      & \cdots & 0      \\
 * \vdots & \vdots & \ddots & \vdots \\
 * a_{n1} & a_{n2} & \cdots & 0
 * \end{pmatrix},
 * \f$
 *
 * \f$
 * U = \begin{pmatrix}
 * 0      & a_{12} & \cdots & a_{1n} \\
 * 0      & 0      & \cdots & a_{2n} \\
 * \vdots & \vdots & \ddots & \vdots \\
 * 0      & 0      & \cdots & 0
 * \end{pmatrix}.
 * \f$
 *
 * Substituting this decomposition into the equation \f$ A\mathbf{x} = \mathbf{b} \f$, we get
 *
 * \f$ (D + L + U)\mathbf{x} = \mathbf{b}. \f$
 *
 * Rearranging, we have
 *
 * \f$ (D + L)\mathbf{x} = \mathbf{b} - U\mathbf{x}. \f$
 *
 * Assuming that \f$(D + L)\f$ is invertible, we can solve for \f$\mathbf{x}\f$:
 *
 * \f$ \mathbf{x} = (D + L)^{-1}(\mathbf{b} - U\mathbf{x}). \f$
 *
 * This equation suggests an iterative scheme:
 *
 * \f$ \mathbf{x}^{(k+1)} = (D + L)^{-1}(\mathbf{b} - U\mathbf{x}^{(k)}), \f$
 *
 * where \f$\mathbf{x}^{(k)}\f$ is the \f$k\f$-th approximation of the solution \f$\mathbf{x}\f$.
 *
 * In component form, the Gauss-Seidel update is:
 *
 * \f$ x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right), \quad i = 1, 2, \dots, n. \f$
 *
 * Notice that the updated values \f$x_j^{(k+1)}\f$ for \f$j < i\f$ are used in the calculation of \f$x_i^{(k+1)}\f$,
 * which is the key difference from the Jacobi method.
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
 *
 * // Solve the system using Gauss-Seidel
 * int result = gs(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), x.data(), b.data(), n, control);
 *
 * if (result == 0) {
 * std::cout << "Gauss-Seidel converged. Solution x = ";
 * for (double val : x) {
 * std::cout << val << " ";
 * }
 * std::cout << std::endl;
 * } else {
 * std::cout << "Gauss-Seidel did not converge." << std::endl;
 * }
 *
 * return 0;
 * }
 * @endcode
 *
 */
class gs_solver
{
private:
    vector res;

public:
    gs_solver();
    ~gs_solver();

    gs_solver (const gs_solver&) = delete;
    gs_solver& operator= (const gs_solver&) = delete;

    void build(const csr_matrix& A);
    int solve(const csr_matrix& A, vector& x, const vector& b, iter_control control);
};


#endif