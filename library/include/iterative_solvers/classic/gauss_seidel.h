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

#include "../../linalg_export.h"

#include "../iter_control.h"

#include "../../vector.h"
#include "../../csr_matrix.h"

/*! \file
 *  \brief gauss_seidel.h provides interface for gauss seidel solver
 */

namespace linalg
{
/*! \brief A solver class implementing the Gauss-Seidel iterative method.
 *
 * \details
 * This class provides functionality to solve linear systems of equations of the form
 * \f$A \cdot x = b\f$ using the Gauss-Seidel iterative method. It supports building
 * necessary data structures from a `csr_matrix` and performing the iterative solution
 * process controlled by an `iter_control` object.
 *
 * \section gs_derivation Derivation of the Gauss-Seidel Method
 *
 * Consider a system of \f$n\f$ linear equations with \f$n\f$ unknowns:
 * \f$ A \mathbf{x} = \mathbf{b} \f$
 * where \f$A\f$ is an \f$n \times n\f$ matrix, \f$\mathbf{x}\f$ is the vector of unknowns,
 * and \f$\mathbf{b}\f$ is the known right-hand side vector.
 *
 * We can write the matrix \f$A\f$ as a sum of three components:
 * \f$ A = L + D + U \f$
 * where:
 * - \f$D\f$ is a diagonal matrix containing the diagonal elements of \f$A\f$.
 * - \f$L\f$ is a strictly lower triangular matrix containing the lower triangular elements of \f$A\f$ (with zeros on the diagonal).
 * - \f$U\f$ is a strictly upper triangular matrix containing the upper triangular elements of \f$A\f$ (with zeros on the diagonal).
 *
 * Substituting this decomposition into the original equation:
 * \f$ (L + D + U) \mathbf{x} = \mathbf{b} \f$
 *
 * Rearranging the terms to isolate the diagonal part and the lower triangular part on the left-hand side:
 * \f$ (L + D) \mathbf{x} = \mathbf{b} - U \mathbf{x} \f$
 *
 * Now, we introduce the iterative nature of the Gauss-Seidel method. Let \f$\mathbf{x}^{(k)}\f$ be the approximation
 * of the solution vector at iteration \f$k\f$. The Gauss-Seidel method updates the components of \f$\mathbf{x}\f$
 * sequentially, using the *most recently computed* values. This is the key difference from the Jacobi method,
 * which uses values from the *previous* iteration entirely.
 *
 * The iterative formula for the Gauss-Seidel method is derived by expressing \f$\mathbf{x}^{(k+1)}\f$:
 * \f$ (L + D) \mathbf{x}^{(k+1)} = \mathbf{b} - U \mathbf{x}^{(k)} \f$
 *
 * To obtain the elements of \f$\mathbf{x}^{(k+1)}\f$ individually, we can write out the \f$i\f$-th equation:
 * \f$ \sum_{j=1}^{i-1} A_{ij} x_j^{(k+1)} + A_{ii} x_i^{(k+1)} + \sum_{j=i+1}^{n} A_{ij} x_j^{(k)} = b_i \f$
 *
 * Solving for $x_i^{(k+1)}$:
 * \f$ A_{ii} x_i^{(k+1)} = b_i - \sum_{j=1}^{i-1} A_{ij} x_j^{(k+1)} - \sum_{j=i+1}^{n} A_{ij} x_j^{(k)} \f$
 *
 * And finally, the explicit iterative formula for each component \f$x_i\f$ at iteration \f$k+1\f$:
 * \f$ x_i^{(k+1)} = \frac{1}{A_{ii}} \left( b_i - \sum_{j=1}^{i-1} A_{ij} x_j^{(k+1)} - \sum_{j=i+1}^{n} A_{ij} x_j^{(k)} \right) \f$
 *
 * This formula shows that when computing \f$x_i^{(k+1)}\f$, we use the newly updated values \f$x_j^{(k+1)}\f$
 * for \f$j < i\f$ and the values from the previous iteration \f$x_j^{(k)}\f$ for \f$j > i\f$.
 *
 * \section gs_convergence Convergence Criteria
 * The Gauss-Seidel method is guaranteed to converge if the matrix \f$A\f$ is either:
 * - Strictly diagonally dominant: \f$|A_{ii}| > \sum_{j=1, j \neq i}^{n} |A_{ij}|\f$ for all \f$i\f$.
 * - Symmetric and positive definite.
 *
 * \section gs_example Example Usage
 * Below is a simplified example demonstrating how to use the `gs_solver` class.
 * This assumes the `csr_matrix`, `vector`, and `iter_control` classes are properly defined
 * and functional.
 *
 * \code
 * #include "linalglib.h"
 * #include <iostream>
 * #include <vector>
 *
 * int main() {
 * // Define a sample sparse matrix A (e.g., a tridiagonal matrix)
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
 * // Create a Gauss-Seidel solver instance
 * gs_solver solver;
 *
 * // Build the solver (e.g., pre-process the matrix)
 * solver.build(A);
 *
 * // Solve the system
 * std::cout << "Starting Gauss-Seidel solver..." << std::endl;
 * int status = solver.solve(A, x, b, control);
 *
 * if (status == 0) {
 * std::cout << "Gauss-Seidel converged successfully!" << std::endl;
 * } else {
 * std::cout << "Gauss-Seidel did NOT converge. Status code: " << status << std::endl;
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
class gs_solver
{
private:
    /*! \brief Internal vector to store the residual during the solve process. */
    vector<double> res;

public:
    /*! \brief Default constructor.
     * Initializes a new `gs_solver` object.
     */
    gs_solver();

    /*! \brief Destructor.
     * Cleans up any resources allocated by the `gs_solver` object.
     */
    ~gs_solver();

    /*! \brief Deleted copy constructor.
     * Prevents direct copying of `gs_solver` objects to avoid shallow copies and
     * ensure proper memory management.
     */
    gs_solver (const gs_solver&) = delete;

    /*! \brief Deleted copy assignment operator.
     * Prevents direct assignment of one `gs_solver` object to another to avoid shallow copies
     * and ensure proper memory management.
     */
    gs_solver& operator= (const gs_solver&) = delete;

    /*! \brief Builds necessary data structures for the Gauss-Seidel solver.
     * \details
     * This method might pre-process the input matrix `A` to optimize the
     * Gauss-Seidel iterations. For example, it might extract the diagonal elements
     * of `A` for efficient division in the iterative step.
     * \param A The sparse matrix in CSR format for which the solver is being built.
     */
    void build(const csr_matrix& A);

    /*! \brief Solves the linear system \f$A \cdot x = b\f$ using the Gauss-Seidel method.
     *
     * This method iteratively updates the solution vector `x` until the convergence
     * criteria specified by `control` are met or the maximum number of iterations
     * is reached.
     *
     * \param A The sparse coefficient matrix in CSR format.
     * \param x On input, an initial guess for the solution vector; on output, the computed solution vector.
     * \param b The right-hand side vector.
     * \param control An `iter_control` object that manages the iteration process,
     * including convergence tolerance and maximum iterations.
     * \return An integer status code:
     * - `0` if the solver converged successfully within the specified tolerance.
     * - `1` if the maximum number of iterations was reached without convergence.
     * - Other negative values for errors (e.g., matrix not diagonally dominant, singular diagonal element).
     */
    int solve(const csr_matrix& A, vector<double>& x, const vector<double>& b, iter_control control);
};
}

#endif