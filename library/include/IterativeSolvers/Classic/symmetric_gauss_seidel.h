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

#ifndef SYMMETRIC_GAUSS_SEIDEL_H
#define SYMMETRIC_GAUSS_SEIDEL_H

#include "../../linalg_export.h"

#include "../iter_control.h"

#include "../../vector.h"
#include "../../csr_matrix.h"

/*! \file
 *  \brief symmetric_gauss_seidel.h provides interface for symmetric gauss
 * seidel solver
 */

namespace linalg
{
/*! \brief A solver class implementing the Symmetric Gauss-Seidel (SGS) iterative method.
 *
 * \details
 * This class provides functionality to solve linear systems of equations of the form
 * \f$A \cdot x = b\f$ using the Symmetric Gauss-Seidel (SGS) iterative method.
 * SGS applies a forward Gauss-Seidel sweep followed by a backward Gauss-Seidel sweep
 * as a single iteration. This method often exhibits better convergence properties
 * than the standard Gauss-Seidel method, particularly for symmetric matrices, and
 * is frequently used as a preconditioner for Krylov subspace methods like Conjugate Gradient.
 *
 * \section sgs_derivation Derivation of the SGS Method
 *
 * The Symmetric Gauss-Seidel (SGS) method is constructed by combining a forward Gauss-Seidel
 * sweep and a backward Gauss-Seidel sweep.
 *
 * We decompose the matrix \f$A\f$ into its diagonal (\f$D\f$), strictly lower triangular (\f$L\f$),
 * and strictly upper triangular (\f$U\f$) parts: \f$A = L + D + U\f$.
 *
 * A single SGS iteration from \f$\mathbf{x}^{(k)}\f$ to \f$\mathbf{x}^{(k+1)}\f$ consists of two steps:
 *
 * \subsection sgs_forward_sweep Forward Gauss-Seidel Sweep
 * In the first half-step, we perform a standard Gauss-Seidel iteration to compute an intermediate
 * vector \f$\mathbf{x}^{(k+1/2)}\f$:
 * \f$ (L + D) \mathbf{x}^{(k+1/2)} = \mathbf{b} - U \mathbf{x}^{(k)} \f$
 * This can be written component-wise as:
 * \f$ x_i^{(k+1/2)} = \frac{1}{A_{ii}} \left( b_i - \sum_{j=1}^{i-1} A_{ij} x_j^{(k+1/2)} - \sum_{j=i+1}^{n} A_{ij} x_j^{(k)} \right) \f$
 *
 * \subsection sgs_backward_sweep Backward Gauss-Seidel Sweep
 * In the second half-step, we use the intermediate \f$\mathbf{x}^{(k+1/2)}\f$ to compute the
 * new iterate \f$\mathbf{x}^{(k+1)}\f$ using a backward Gauss-Seidel sweep. This means
 * iterating from \f$i=n\f$ down to \f$1\f$:
 * \f$ (U + D) \mathbf{x}^{(k+1)} = \mathbf{b} - L \mathbf{x}^{(k+1/2)} \f$
 * Component-wise, this is:
 * \f$ x_i^{(k+1)} = \frac{1}{A_{ii}} \left( b_i - \sum_{j=i+1}^{n} A_{ij} x_j^{(k+1)} - \sum_{j=1}^{i-1} A_{ij} x_j^{(k+1/2)} \right) \f$
 *
 * When used as a preconditioner, the action of the SGS preconditioner \f$M_{SGS}\f$ is equivalent to:
 * \f$ M_{SGS}^{-1} = (D+L)^{-1} + (D+U)^{-1} - D^{-1} \f$
 * Or, if \f$A\f$ is symmetric, it can be seen as:
 * \f$ M_{SGS}^{-1} = (D+L)^{-1} D (D+L)^T \f$
 *
 * \section sgs_convergence Convergence Criteria
 *
 * The Symmetric Gauss-Seidel method is guaranteed to converge if the matrix \f$A\f$ is
 * symmetric and positive definite.
 *
 * \section sgs_example Example Usage
 * Below is a simplified example demonstrating how to use the `sgs_solver` class.
 * This assumes the `csr_matrix`, `vector`, and `iter_control` classes are properly defined
 * and functional.
 *
 * \code
 * #include "linalglib.h"
 * #include <iostream>
 * #include <vector>
 *
 * int main() {
 * // Define a sample sparse matrix A (e.g., a symmetric tridiagonal matrix)
 * // For simplicity, let's create a 3x3 matrix:
 * // [ 4 -1  0 ]
 * // [-1  4 -1 ]
 * // [ 0 -1  4 ]
 *
 * std::vector<int> row_ptr = {0, 2, 5, 7}; // For 3 rows
 * std::vector<int> col_ind = {0, 1, 0, 1, 2, 1, 2};
 * std::vector<double> val = {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0};
 *
 * int m = 3; // Number of rows
 * int n = 3; // Number of columns
 * int nnz = 7; // Number of non-zeros
 *
 * csr_matrix A(row_ptr, col_ind, val, m, n, nnz);
 *
 * // Define the right-hand side vector b
 * std::vector<double> b_data = {5.0, 3.0, 10.0};
 * vector b(b_data);
 *
 * // Define an initial guess for the solution vector x (e.g., all zeros)
 * vector x(m);
 * x.zeros();
 *
 * // Set up iteration control
 * iter_control control;
 * control.max_iterations = 100;
 * control.tolerance = 1e-6;
 *
 * // Create an SGS solver instance
 * sgs_solver solver;
 *
 * // Build the solver (e.g., pre-process the matrix)
 * solver.build(A);
 *
 * // Solve the system
 * std::cout << "Starting Symmetric Gauss-Seidel solver..." << std::endl;
 * int status = solver.solve(A, x, b, control);
 *
 * if (status == 0) {
 * std::cout << "Symmetric Gauss-Seidel converged successfully!" << std::endl;
 * } else {
 * std::cout << "Symmetric Gauss-Seidel did NOT converge. Status code: " << status << std::endl;
 * }
 *
 * std::cout << "Approximate solution x:" << std::endl;
 * for (int i = 0; i < x.get_size(); ++i) {
 * std::cout << "x[" << i << "] = " << x[i] << std::endl;
 * }
 *
 * return 0;
 * }
 * \endcode
 */
class sgs_solver
{
private:
    /*! \brief Internal vector to store the residual during the solve process. */
    vector<double> res;

public:
    /*! \brief Default constructor.
     * Initializes a new `sgs_solver` object.
     */
    sgs_solver();

    /*! \brief Destructor.
     * Cleans up any resources allocated by the `sgs_solver` object.
     */
    ~sgs_solver();

    /*! \brief Deleted copy constructor.
     * Prevents direct copying of `sgs_solver` objects to avoid shallow copies and
     * ensure proper memory management.
     */
    sgs_solver (const sgs_solver&) = delete;

    /*! \brief Deleted copy assignment operator.
     * Prevents direct assignment of one `sgs_solver` object to another to avoid shallow copies
     * and ensure proper memory management.
     */
    sgs_solver& operator= (const sgs_solver&) = delete;

    /*! \brief Builds necessary data structures for the Symmetric Gauss-Seidel solver.
     * \details
     * This method typically involves pre-processing the input matrix `A` to
     * facilitate efficient forward and backward Gauss-Seidel sweeps. This may
     * include extracting the diagonal elements and setting up for triangular solves.
     * \param A The sparse matrix in CSR format for which the solver is being built.
     */
    void build(const csr_matrix& A);

    /*! \brief Solves the linear system \f$A \cdot x = b\f$ using the Symmetric Gauss-Seidel (SGS) method.
     *
     * This method iteratively updates the solution vector `x` by performing a
     * forward Gauss-Seidel sweep followed by a backward Gauss-Seidel sweep,
     * until the convergence criteria specified by `control` are met or the
     * maximum number of iterations is reached.
     *
     * \param A The sparse coefficient matrix in CSR format.
     * \param x On input, an initial guess for the solution vector; on output, the computed solution vector.
     * \param b The right-hand side vector.
     * \param control An `iter_control` object that manages the iteration process,
     * including convergence tolerance and maximum iterations.
     * \return An integer status code:
     * - `0` if the solver converged successfully within the specified tolerance.
     * - `1` if the maximum number of iterations was reached without convergence.
     * - Other negative values for errors (e.g., singular diagonal element).
     */
    int solve(const csr_matrix& A, vector<double>& x, const vector<double>& b, iter_control control);
};
}

#endif