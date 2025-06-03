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

namespace linalg
{
/*! \brief A solver class implementing the Successive Over-Relaxation (SOR) iterative method.
 *
 * \details
 * This class provides functionality to solve linear systems of equations of the form
 * \f$A \cdot x = b\f$ using the Successive Over-Relaxation (SOR) iterative method. SOR
 * is an acceleration of the Gauss-Seidel method, introducing a relaxation parameter
 * \f$\omega\f$ to potentially speed up convergence. It supports building necessary data
 * structures from a `csr_matrix` and performing the iterative solution process
 * controlled by an `iter_control` object and the relaxation parameter \f$\omega\f$.
 *
 * \section sor_derivation Derivation of the SOR Method
 *
 * The SOR method is an extension of the Gauss-Seidel method that aims to accelerate
 * its convergence. It applies a relaxation factor \f$\omega\f$ to the Gauss-Seidel update.
 *
 * Recall the Gauss-Seidel iterative formula for each component \f$x_i\f$:
 * \f$ x_i^{(k+1, \text{GS})} = \frac{1}{A_{ii}} \left( b_i - \sum_{j=1}^{i-1} A_{ij} x_j^{(k+1)} - \sum_{j=i+1}^{n} A_{ij} x_j^{(k)} \right) \f$
 *
 * The SOR method introduces a relaxation parameter \f$\omega \in (0, 2)\f$.
 * The update for \f$x_i^{(k+1)}\f$ is then a weighted average of the previous iteration's
 * value \f$x_i^{(k)}\f$ and the full Gauss-Seidel update \f$x_i^{(k+1, \text{GS})}\f$:
 * \f$ x_i^{(k+1)} = (1 - \omega) x_i^{(k)} + \omega x_i^{(k+1, \text{GS})} \f$
 *
 * Substituting the expression for \f$x_i^{(k+1, \text{GS})}\f$:
 * \f$ x_i^{(k+1)} = (1 - \omega) x_i^{(k)} + \frac{\omega}{A_{ii}} \left( b_i - \sum_{j=1}^{i-1} A_{ij} x_j^{(k+1)} - \sum_{j=i+1}^{n} A_{ij} x_j^{(k)} \right) \f$
 *
 * Rearranging the terms to express the update directly:
 * \f$ x_i^{(k+1)} = x_i^{(k)} + \frac{\omega}{A_{ii}} \left( b_i - \sum_{j=1}^{i-1} A_{ij} x_j^{(k+1)} - A_{ii} x_i^{(k)} - \sum_{j=i+1}^{n} A_{ij} x_j^{(k)} \right) \f$
 *
 * The term inside the parenthesis is essentially the negative of the residual
 * associated with the \f$i\f$-th equation, where the current values are used for \f$j<i\f$
 * and previous values for \f$j \ge i\f$.
 *
 * \subsection sor_matrix_form Matrix Form
 *
 * In matrix form, SOR can be written as:
 * \f$ (D + \omega L) \mathbf{x}^{(k+1)} = [\omega \mathbf{b} - (\omega U + (\omega - 1) D) \mathbf{x}^{(k)}] \f$
 *
 * where \f$D\f$, \f$L\f$, and \f$U\f$ are the diagonal, strictly lower, and strictly upper triangular parts of \f$A\f$, respectively.
 *
 * \section sor_convergence Convergence Criteria
 *
 * For a symmetric positive definite matrix \f$A\f$, the SOR method is guaranteed to converge
 * if and only if \f$0 < \omega < 2\f$.
 *
 * The optimal choice of \f$\omega\f$ is crucial for the performance of SOR. For certain classes
 * of matrices (e.g., those arising from finite difference discretizations of elliptic PDEs),
 * there are theoretical formulas for the optimal \f$\omega\f$. In general, finding the optimal
 * \f$\omega\f$ often requires numerical experimentation.
 *
 * \section sor_example Example Usage
 * Below is a simplified example demonstrating how to use the `sor_solver` class.
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
 * // Choose a relaxation parameter omega. A common starting point is between 1.0 and 1.5.
 * // Optimal omega depends on the matrix properties.
 * double omega = 1.2;
 *
 * // Create an SOR solver instance
 * sor_solver solver;
 *
 * // Build the solver (e.g., pre-process the matrix)
 * solver.build(A);
 *
 * // Solve the system
 * std::cout << "Starting SOR solver with omega = " << omega << "..." << std::endl;
 * int status = solver.solve(A, x, b, control, omega);
 *
 * if (status == 0) {
 * std::cout << "SOR converged successfully!" << std::endl;
 * } else {
 * std::cout << "SOR did NOT converge. Status code: " << status << std::endl;
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
class sor_solver
{
private:
    /*! \brief Internal vector to store the residual during the solve process. */
    vector res;

public:
    /*! \brief Default constructor.
     * Initializes a new `sor_solver` object.
     */
    sor_solver();

    /*! \brief Destructor.
     * Cleans up any resources allocated by the `sor_solver` object.
     */
    ~sor_solver();

    /*! \brief Deleted copy constructor.
     * Prevents direct copying of `sor_solver` objects to avoid shallow copies and
     * ensure proper memory management.
     */
    sor_solver (const sor_solver&) = delete;

    /*! \brief Deleted copy assignment operator.
     * Prevents direct assignment of one `sor_solver` object to another to avoid shallow copies
     * and ensure proper memory management.
     */
    sor_solver& operator= (const sor_solver&) = delete;

    /*! \brief Builds necessary data structures for the SOR solver.
     * \details
     * This method might involve pre-processing the input matrix `A`, such as
     * extracting the diagonal elements or preparing for efficient forward sweeps
     * in the SOR iteration.
     * \param A The sparse matrix in CSR format for which the solver is being built.
     */
    void build(const csr_matrix& A);

    /*! \brief Solves the linear system \f$A \cdot x = b\f$ using the Successive Over-Relaxation (SOR) method.
     *
     * This method iteratively updates the solution vector `x` using the SOR
     * iteration formula until the convergence criteria specified by `control` are
     * met or the maximum number of iterations is reached.
     *
     * \param A The sparse coefficient matrix in CSR format.
     * \param x On input, an initial guess for the solution vector; on output, the computed solution vector.
     * \param b The right-hand side vector.
     * \param control An `iter_control` object that manages the iteration process,
     * including convergence tolerance and maximum iterations.
     * \param omega The relaxation parameter (\f$\omega\f$) for the SOR method. Must be in the range \f$(0, 2)\f$.
     * \return An integer status code:
     * - `0` if the solver converged successfully within the specified tolerance.
     * - `1` if the maximum number of iterations was reached without convergence.
     * - Other negative values for errors (e.g., invalid omega, singular diagonal element).
     */
    int solve(const csr_matrix& A, vector& x, const vector& b, iter_control control, double omega);
};
}

#endif