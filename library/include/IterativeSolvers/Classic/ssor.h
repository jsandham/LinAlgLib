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

#ifndef SSOR_H
#define SSOR_H

#include "../../linalglib_export.h"

#include "../iter_control.h"

#include "../../vector.h"
#include "../../csr_matrix.h"

/*! \file
 *  \brief ssor.h provides interface for ssor solver
 */

/*! \brief A solver class implementing the Symmetric Successive Over-Relaxation (SSOR) iterative method.
 *
 * \details
 * This class provides functionality to solve linear systems of equations of the form
 * \f$A \cdot x = b\f$ using the Symmetric Successive Over-Relaxation (SSOR) iterative method.
 * SSOR is a preconditioning technique or a standalone iterative method that applies
 * a forward SOR sweep followed by a backward SOR sweep. It aims to improve the
 * symmetry of the iteration matrix, which can be beneficial for convergence, especially
 * when used as a preconditioner for methods like Conjugate Gradient.
 *
 * \section ssor_derivation Derivation of the SSOR Method
 *
 * The SSOR method combines two sweeps of the SOR iteration: a forward sweep and a backward sweep.
 * It's often viewed as applying the SOR preconditioner \f$M_{SOR}\f$ and then its transpose
 * \f$M_{SOR}^T\f$.
 *
 * Recall the SOR iteration matrix for a given \f$\omega\f$:
 * The SOR update for \f$\mathbf{x}^{(k+1)}\f$ from \f$\mathbf{x}^{(k)}\f$ can be written as:
 * \f$ \mathbf{x}^{(k+1)} = (D + \omega L)^{-1} [(\omega - 1)D - \omega U] \mathbf{x}^{(k)} + \omega (D + \omega L)^{-1} \mathbf{b} \f$
 *
 * For the SSOR method, one full iteration consists of two half-steps:
 *
 * \subsection ssor_forward_sweep Forward Sweep
 * The first half-step is a standard forward SOR iteration, where we compute an intermediate vector \f$\mathbf{x}^{(k+1/2)}\f$:
 * \f$ (D + \omega L) \mathbf{x}^{(k+1/2)} = \omega \mathbf{b} - \omega U \mathbf{x}^{(k)} - (\omega - 1) D \mathbf{x}^{(k)} \f$
 * Or equivalently, in terms of the residual:
 * \f$ \mathbf{x}^{(k+1/2)} = \mathbf{x}^{(k)} + \omega D^{-1} (\mathbf{b} - L \mathbf{x}^{(k+1/2)} - D \mathbf{x}^{(k)} - U \mathbf{x}^{(k)}) \f$
 *
 * \subsection ssor_backward_sweep Backward Sweep
 * The second half-step is a backward SOR iteration, using the intermediate \f$\mathbf{x}^{(k+1/2)}\f$ to compute \f$\mathbf{x}^{(k+1)}\f$:
 * \f$ (D + \omega U) \mathbf{x}^{(k+1)} = \omega \mathbf{b} - \omega L \mathbf{x}^{(k+1/2)} - (\omega - 1) D \mathbf{x}^{(k+1/2)} \f$
 * Or equivalently:
 * \f$ \mathbf{x}^{(k+1)} = \mathbf{x}^{(k+1/2)} + \omega D^{-1} (\mathbf{b} - U \mathbf{x}^{(k+1)} - D \mathbf{x}^{(k+1/2)} - L \mathbf{x}^{(k+1/2)}) \f$
 *
 * Combining these two steps gives the SSOR iteration. The matrix representation of the SSOR preconditioning matrix \f$M_{SSOR}\f$ is:
 * \f$ M_{SSOR} = \frac{1}{\omega(2-\omega)}(D + \omega L)D^{-1}(D + \omega U) \f$
 *
 * \section ssor_convergence Convergence Criteria
 *
 * If the matrix \f$A\f$ is symmetric and positive definite, then the SSOR method is guaranteed to converge for any \f$\omega \in (0, 2)\f$.
 * Similar to SOR, the optimal choice of \f$\omega\f$ is crucial for the convergence rate and often determined through numerical experimentation.
 *
 * \section ssor_example Example Usage
 * Below is a simplified example demonstrating how to use the `ssor_solver` class.
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
 * // Create an SSOR solver instance
 * ssor_solver solver;
 *
 * // Build the solver (e.g., pre-process the matrix)
 * solver.build(A);
 *
 * // Solve the system
 * std::cout << "Starting SSOR solver with omega = " << omega << "..." << std::endl;
 * int status = solver.solve(A, x, b, control, omega);
 *
 * if (status == 0) {
 * std::cout << "SSOR converged successfully!" << std::endl;
 * } else {
 * std::cout << "SSOR did NOT converge. Status code: " << status << std::endl;
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
class ssor_solver
{
private:
    /*! \brief Internal vector to store the residual during the solve process. */
    vector res;

public:
    /*! \brief Default constructor.
     * Initializes a new `ssor_solver` object.
     */
    ssor_solver();

    /*! \brief Destructor.
     * Cleans up any resources allocated by the `ssor_solver` object.
     */
    ~ssor_solver();

    /*! \brief Deleted copy constructor.
     * Prevents direct copying of `ssor_solver` objects to avoid shallow copies and
     * ensure proper memory management.
     */
    ssor_solver (const ssor_solver&) = delete;

    /*! \brief Deleted copy assignment operator.
     * Prevents direct assignment of one `ssor_solver` object to another to avoid shallow copies
     * and ensure proper memory management.
     */
    ssor_solver& operator= (const ssor_solver&) = delete;

    /*! \brief Builds necessary data structures for the SSOR solver.
     * \details
     * This method typically involves pre-processing the input matrix `A` to
     * prepare for both forward and backward SOR sweeps. This may include
     * extracting diagonal elements or other matrix properties needed for the
     * SSOR iteration.
     * \param A The sparse matrix in CSR format for which the solver is being built.
     */
    void build(const csr_matrix& A);

    /*! \brief Solves the linear system \f$A \cdot x = b\f$ using the Symmetric Successive Over-Relaxation (SSOR) method.
     *
     * This method iteratively updates the solution vector `x` by performing a
     * forward SOR sweep followed by a backward SOR sweep, until the convergence
     * criteria specified by `control` are met or the maximum number of iterations
     * is reached.
     *
     * \param A The sparse coefficient matrix in CSR format.
     * \param x On input, an initial guess for the solution vector; on output, the computed solution vector.
     * \param b The right-hand side vector.
     * \param control An `iter_control` object that manages the iteration process,
     * including convergence tolerance and maximum iterations.
     * \param omega The relaxation parameter (\f$\omega\f$) for the SSOR method. Must be in the range \f$(0, 2)\f$.
     * \return An integer status code:
     * - `0` if the solver converged successfully within the specified tolerance.
     * - `1` if the maximum number of iterations was reached without convergence.
     * - Other negative values for errors (e.g., invalid omega, singular diagonal element).
     */
    int solve(const csr_matrix& A, vector& x, const vector& b, iter_control control, double omega);
};


#endif