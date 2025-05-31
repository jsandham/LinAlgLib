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

#include "../../vector.h"
#include "../../csr_matrix.h"

/*! \file
 *  \brief richardson.h provides interface for richardson solver
 */

/*! \brief A solver class implementing the Richardson iterative method.
 *
 * \details
 * This class provides functionality to solve linear systems of equations of the form
 * \f$A \cdot x = b\f$ using the Richardson iterative method. It supports building
 * necessary data structures from a `csr_matrix` and performing the iterative solution
 * process controlled by an `iter_control` object and a relaxation parameter \f$\theta\f$.
 *
 * \section rich_derivation Derivation of the Richardson Method
 *
 * The Richardson iteration is one of the simplest stationary iterative methods for solving
 * a system of linear equations \f$A \mathbf{x} = \mathbf{b}\f$. It can be derived by
 * rearranging the original system.
 *
 * We start with the system:
 * \f$ A \mathbf{x} = \mathbf{b} \f$
 *
 * We can rewrite this by adding and subtracting \f$\mathbf{x}\f$ to the left side and
 * introducing a scaling parameter \f$\theta\f$:
 * \f$ \mathbf{x} + \theta (\mathbf{b} - A \mathbf{x}) = \mathbf{x} \f$
 *
 * This rearrangement suggests an iterative scheme. Let \f$\mathbf{x}^{(k)}\f$ be the
 * approximation of the solution vector at iteration \f$k\f$. The Richardson iterative
 * formula is given by:
 * \f$ \mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \theta (\mathbf{b} - A \mathbf{x}^{(k)}) \f$
 *
 * Here, \f$\mathbf{r}^{(k)} = \mathbf{b} - A \mathbf{x}^{(k)}\f$ is the residual vector
 * at iteration \f$k\f$. The method attempts to reduce this residual by taking a step
 * proportional to it.
 *
 * The iterative formula can also be viewed as a fixed-point iteration for the equation
 * \f$\mathbf{x} = \mathbf{x} + \theta (\mathbf{b} - A \mathbf{x})\f$.
 *
 * \subsection rich_matrix_form Matrix Form
 *
 * To express it in matrix form, we can rewrite the iteration as:
 * \f$ \mathbf{x}^{(k+1)} = (I - \theta A) \mathbf{x}^{(k)} + \theta \mathbf{b} \f$
 *
 * The iteration matrix for the Richardson method is \f$M_{rich} = I - \theta A\f$.
 *
 * \section rich_convergence Convergence Criteria
 *
 * The Richardson method converges if and only if the spectral radius of the iteration
 * matrix \f$\rho(I - \theta A) < 1\f$.
 *
 * For a symmetric positive definite matrix \f$A\f$, the method converges if and only if
 * \f$0 < \theta < \frac{2}{\lambda_{max}}\f$, where \f$\lambda_{max}\f$ is the largest eigenvalue of \f$A\f$.
 * The optimal value for \f$\theta\f$ is typically chosen as \f$\frac{2}{\lambda_{min} + \lambda_{max}}\f$.
 *
 * \section rich_example Example Usage
 * Below is a simplified example demonstrating how to use the `rich_solver` class.
 * This assumes the `csr_matrix`, `vector`, and `iter_control` classes are properly defined
 * and functional.
 *
 * \code
 * #include "linalglib.h"
 * #include <iostream>
 * #include <vector>
 *
 * int main() {
 * // Define a sample sparse matrix A (e.g., a simple diagonal matrix for demonstration)
 * // [ 2  0  0 ]
 * // [ 0  3  0 ]
 * // [ 0  0  4 ]
 *
 * std::vector<int> row_ptr = {0, 1, 2, 3};
 * std::vector<int> col_ind = {0, 1, 2};
 * std::vector<double> val = {2.0, 3.0, 4.0};
 *
 * int m = 3; // Number of rows
 * int n = 3; // Number of columns
 * int nnz = 3; // Number of non-zeros
 *
 * csr_matrix A(row_ptr, col_ind, val, m, n, nnz);
 *
 * // Define the right-hand side vector b
 * std::vector<double> b_data = {6.0, 9.0, 12.0};
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
 * // Choose a relaxation parameter theta. For a diagonal matrix with A_ii between 2 and 4,
 * // an optimal theta might be around 2 / (2 + 4) = 1/3 for this simple case.
 * // In general, estimating optimal theta can be complex.
 * double theta = 0.3;
 *
 * // Create a Richardson solver instance
 * rich_solver solver;
 *
 * // Build the solver (e.g., pre-process the matrix)
 * solver.build(A);
 *
 * // Solve the system
 * std::cout << "Starting Richardson solver..." << std::endl;
 * int status = solver.solve(A, x, b, control, theta);
 *
 * if (status == 0) {
 * std::cout << "Richardson converged successfully!" << std::endl;
 * } else {
 * std::cout << "Richardson did NOT converge. Status code: " << status << std::endl;
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
class rich_solver
{
private:
    /*! \brief Internal vector to store the residual during the solve process. */
    vector res;

public:
    /*! \brief Default constructor.
     * Initializes a new `rich_solver` object.
     */
    rich_solver();

    /*! \brief Destructor.
     * Cleans up any resources allocated by the `rich_solver` object.
     */
    ~rich_solver();

    /*! \brief Deleted copy constructor.
     * Prevents direct copying of `rich_solver` objects to avoid shallow copies and
     * ensure proper memory management.
     */
    rich_solver (const rich_solver&) = delete;

    /*! \brief Deleted copy assignment operator.
     * Prevents direct assignment of one `rich_solver` object to another to avoid shallow copies
     * and ensure proper memory management.
     */
    rich_solver& operator= (const rich_solver&) = delete;

    /*! \brief Builds necessary data structures for the Richardson solver.
     * \details
     * For the Richardson method, this might involve simply storing a reference
     * or a copy of the matrix `A` if needed for residual computation.
     * \param A The sparse matrix in CSR format for which the solver is being built.
     */
    void build(const csr_matrix& A);

    /*! \brief Solves the linear system \f$A \cdot x = b\f$ using the Richardson method.
     *
     * This method iteratively updates the solution vector `x` using the Richardson
     * iteration formula until the convergence criteria specified by `control` are
     * met or the maximum number of iterations is reached.
     *
     * \param A The sparse coefficient matrix in CSR format.
     * \param x On input, an initial guess for the solution vector; on output, the computed solution vector.
     * \param b The right-hand side vector.
     * \param control An `iter_control` object that manages the iteration process,
     * including convergence tolerance and maximum iterations.
     * \param theta The relaxation parameter (\f$\theta\f$) for the Richardson iteration.
     * \return An integer status code:
     * - `0` if the solver converged successfully within the specified tolerance.
     * - `1` if the maximum number of iterations was reached without convergence.
     * - Other negative values for errors (e.g., invalid input).
     */
    int solve(const csr_matrix& A, vector& x, const vector& b, iter_control control, double theta);
};



#endif
