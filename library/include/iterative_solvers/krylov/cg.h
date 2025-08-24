//********************************************************************************
//
// MIT License
//
// Copyright(c) 2024-2025 James Sandham
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

#ifndef CG_H
#define CG_H

#include "../../linalg_export.h"

#include "../iter_control.h"
#include "../preconditioner/preconditioner.h"

#include "../../csr_matrix.h"
#include "../../vector.h"

/*! \file
 *  \brief cg.h provides interface for conjugate gradient solvers
 */

namespace linalg
{
    /*! \brief A solver class implementing the Conjugate Gradient (CG) method.
 *
 * \details
 * This class provides functionality to solve large, sparse, **symmetric positive definite (SPD)**
 * systems of linear equations of the form \f$A \cdot x = b\f$ using the Conjugate Gradient (CG)
 * iterative method. CG is an extremely powerful and widely used krylov subspace method
 * for SPD systems, known for its rapid convergence. It can be used with or without a preconditioner
 * to further accelerate convergence.
 *
 * \section cg_algorithm The Conjugate Gradient Algorithm
 *
 * The Conjugate Gradient algorithm aims to find an approximate solution \f$\mathbf{x}\f$ to \f$A\mathbf{x} = \mathbf{b}\f$
 * by iteratively minimizing the quadratic function \f$\phi(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T A \mathbf{x} - \mathbf{x}^T \mathbf{b}\f$.
 * The method generates a sequence of search directions that are A-conjugate (or A-orthogonal),
 * meaning \f$\mathbf{p}_i^T A \mathbf{p}_j = 0\f$ for \f$i \neq j\f$. This ensures that each new search
 * direction is independent of previous directions in the A-inner product sense.
 *
 * The non-preconditioned Conjugate Gradient algorithm steps are:
 *
 * 1. **Initialize:**
 * Set initial guess \f$\mathbf{x}^{(0)}\f$.
 * Compute initial residual \f$\mathbf{r}^{(0)} = \mathbf{b} - A \mathbf{x}^{(0)}\f$.
 * Set initial search direction \f$\mathbf{p}^{(0)} = \mathbf{r}^{(0)}\f$.
 *
 * 2. **Iterate** for \f$k = 0, 1, 2, \dots\f$ until convergence:
 * a. Compute \f$\alpha_k = \frac{{\mathbf{r}^{(k)}}^T \mathbf{r}^{(k)}}{{\mathbf{p}^{(k)}}^T A \mathbf{p}^{(k)}}\f$
 * b. Update solution: \f$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \alpha_k \mathbf{p}^{(k)}\f$
 * c. Update residual: \f$\mathbf{r}^{(k+1)} = \mathbf{r}^{(k)} - \alpha_k A \mathbf{p}^{(k)}\f$
 * d. Check for convergence: If \f$\| \mathbf{r}^{(k+1)} \|_2 < \text{tolerance}\f$, then stop.
 * e. Compute \f$\beta_k = \frac{{\mathbf{r}^{(k+1)}}^T \mathbf{r}^{(k+1)}}{{\mathbf{r}^{(k)}}^T \mathbf{r}^{(k)}}\f$
 * f. Update search direction: \f$\mathbf{p}^{(k+1)} = \mathbf{r}^{(k+1)} + \beta_k \mathbf{p}^{(k)}\f$
 *
 * \subsection cg_preconditioned Preconditioned Conjugate Gradient (PCG)
 *
 * For the preconditioned version, a preconditioner \f$M \approx A^{-1}\f$ is used. The preconditioner
 * must be symmetric and positive definite for the PCG algorithm to maintain its theoretical
 * convergence properties. The algorithm steps are modified by applying the preconditioner:
 *
 * 1. **Initialize:**
 * Set initial guess \f$\mathbf{x}^{(0)}\f$.
 * Compute initial residual \f$\mathbf{r}^{(0)} = \mathbf{b} - A \mathbf{x}^{(0)}\f$.
 * Solve \f$M \mathbf{z}^{(0)} = \mathbf{r}^{(0)}\f$ (preconditioning step).
 * Set initial search direction \f$\mathbf{p}^{(0)} = \mathbf{z}^{(0)}\f$.
 *
 * 2. **Iterate** for \f$k = 0, 1, 2, \dots\f$ until convergence:
 * a. Compute \f$\alpha_k = \frac{{\mathbf{r}^{(k)}}^T \mathbf{z}^{(k)}}{{\mathbf{p}^{(k)}}^T A \mathbf{p}^{(k)}}\f$
 * b. Update solution: \f$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \alpha_k \mathbf{p}^{(k)}\f$
 * c. Update residual: \f$\mathbf{r}^{(k+1)} = \mathbf{r}^{(k)} - \alpha_k A \mathbf{p}^{(k)}\f$
 * d. Check for convergence: If \f$\| \mathbf{r}^{(k+1)} \|_2 < \text{tolerance}\f$, then stop.
 * e. Solve \f$M \mathbf{z}^{(k+1)} = \mathbf{r}^{(k+1)}\f$ (preconditioning step).
 * f. Compute \f$\beta_k = \frac{{\mathbf{r}^{(k+1)}}^T \mathbf{z}^{(k+1)}}{{\mathbf{r}^{(k)}}^T \mathbf{z}^{(k)}}\f$
 * g. Update search direction: \f$\mathbf{p}^{(k+1)} = \mathbf{z}^{(k+1)} + \beta_k \mathbf{p}^{(k)}\f$
 *
 * \section cg_convergence Convergence and Applicability
 *
 * The CG method is guaranteed to converge for any symmetric positive definite matrix \f$A\f$.
 * Its convergence rate depends on the condition number of \f$A\f$ (or \f$M^{-1}A\f$ for PCG).
 * A lower condition number leads to faster convergence. Preconditioners are crucial for
 * reducing the effective condition number and thus accelerating convergence, especially
 * for ill-conditioned problems.
 * The parameter `restart_iter` allows for restarting the CG algorithm every few iterations.
 * This can sometimes improve robustness or manage memory for very large problems,
 * though it generally slows down convergence for well-conditioned systems.
 *
 * \section cg_example Example Usage
 * Below is a simplified example demonstrating how to use the `cg_solver` class
 * with and without a preconditioner. This assumes `csr_matrix`, `vector`, `preconditioner`,
 * and `iter_control` classes are properly defined and functional.
 *
 * \code
 * #include "linalglib.h"
 * #include <iostream>
 * #include <vector>
 *
 * int main() {
 * // Define a sample sparse SPD matrix A (e.g., a symmetric tridiagonal matrix)
 * // A = [ 4 -1  0 ]
 * //     [-1  4 -1 ]
 * //     [ 0 -1  4 ]
 * std::vector<int> row_ptr = {0, 2, 5, 7};
 * std::vector<int> col_ind = {0, 1, 0, 1, 2, 1, 2};
 * std::vector<double> val = {4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0};
 *
 * int m = 3; // Number of rows
 * int n = 3; // Number of columns
 * int nnz = 7; // Number of non-zeros
 *
 * csr_matrix A(row_ptr, col_ind, val, m, n, nnz);
 * // A.make_diagonally_dominant(); // Ensure it's SPD if not inherently
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
 * control.max_iterations = 200;
 * control.tolerance = 1e-7;
 *
 * // Create a CG solver instance
 * cg_solver solver;
 *
 * // --- Non-preconditioned solve ---
 * std::cout << "--- Starting non-preconditioned CG solver ---" << std::endl;
 * vector x_nonprecond = x; // Copy initial guess
 * int status_nonprecond = solver.solve_nonprecond(A, x_nonprecond, b, control);
 *
 * if (status_nonprecond == 0) {
 * std::cout << "Non-preconditioned CG converged successfully!" << std::endl;
 * } else {
 * std::cout << "Non-preconditioned CG did NOT converge. Status code: " << status_nonprecond << std::endl;
 * }
 * std::cout << "Approximate solution x (non-preconditioned):" << std::endl;
 * for (int i = 0; i < x_nonprecond.get_size(); ++i) {
 * std::cout << "x[" << i << "] = " << x_nonprecond[i] << std::endl;
 * }
 *
 * // --- Preconditioned solve (e.g., with Jacobi preconditioner) ---
 * std::cout << "\n--- Starting preconditioned CG solver (Jacobi) ---" << std::endl;
 * jacobi_precond jacobi_prec;
 * jacobi_prec.build(A); // Build the preconditioner for matrix A
 *
 * vector x_precond = x; // Copy initial guess
 * int status_precond = solver.solve_precond(A, x_precond, b, &jacobi_prec, control);
 *
 * if (status_precond == 0) {
 * std::cout << "Preconditioned CG converged successfully!" << std::endl;
 * } else {
 * std::cout << "Preconditioned CG did NOT converge. Status code: " << status_precond << std::endl;
 * }
 * std::cout << "Approximate solution x (preconditioned):" << std::endl;
 * for (int i = 0; i < x_precond.get_size(); ++i) {
 * std::cout << "x[" << i << "] = " << x_precond[i] << std::endl;
 * }
 *
 * return 0;
 * }
 * \endcode
 */
    class cg_solver
    {
    private:
        /*! \brief Intermediate vector for preconditioning: \f$M^{-1} \mathbf{r}\f$. */
        vector<double> z;
        /*! \brief Search direction vector. */
        vector<double> p;
        /*! \brief Residual vector in the CG algorithm. */
        vector<double> res;

        /*! \brief Number of iterations after which the solver should restart.
     * A value of 0 or a very large number typically means no restart.
     * For CG, restarts are usually not needed for exact arithmetic but can
     * be beneficial for large problems or in the presence of floating-point errors.
     */
        int restart_iter;

        /*! \brief Flag indicating if the solver data is currently on the host (CPU) or device (GPU). */
        bool on_host;

    public:
        /*! \brief Default constructor.
     * Initializes a new `cg_solver` object.
     */
        cg_solver();

        /*! \brief Destructor.
     * Cleans up any resources allocated by the `cg_solver` object.
     */
        ~cg_solver();

        /*! \brief Deleted copy constructor.
     * Prevents direct copying of `cg_solver` objects to ensure proper memory management.
     */
        cg_solver(const cg_solver&) = delete;

        /*! \brief Deleted copy assignment operator.
     * Prevents direct assignment of one `cg_solver` object to another to ensure proper memory management.
     */
        cg_solver& operator=(const cg_solver&) = delete;

        /*! \brief Builds necessary data structures for the Conjugate Gradient solver.
     * \details
     * For CG, this typically involves allocating and resizing the internal
     * work vectors (`z`, `p`, `res`) to match the dimensions of the matrix `A`.
     * \param A The sparse matrix in CSR format for which the solver is being built.
     */
        void build(const csr_matrix& A);

        /*! \brief Solves the linear system \f$A \cdot x = b\f$ using the non-preconditioned Conjugate Gradient method.
     *
     * This method implements the CG algorithm without any explicit preconditioning.
     * It iteratively refines the solution `x` until the convergence criteria from `control`
     * are met or the maximum number of iterations is reached.
     *
     * \note This method assumes the input matrix `A` is symmetric positive definite.
     *
     * \param A The sparse coefficient matrix in CSR format.
     * \param x On input, an initial guess for the solution vector; on output, the computed solution vector.
     * \param b The right-hand side vector.
     * \param control An `iter_control` object that manages the iteration process,
     * including convergence tolerance and maximum iterations.
     * \return An integer status code:
     * - `0` if the solver converged successfully within the specified tolerance.
     * - `1` if the maximum number of iterations was reached without convergence.
     * - Negative values might indicate issues like a non-positive definite matrix or division by zero.
     */
        int solve_nonprecond(const csr_matrix&     A,
                             vector<double>&       x,
                             const vector<double>& b,
                             iter_control          control);

        /*! \brief Solves the linear system \f$A \cdot x = b\f$ using the preconditioned Conjugate Gradient method.
     *
     * This method implements the CG algorithm with the provided preconditioner.
     * It iteratively refines the solution `x` until the convergence criteria from `control`
     * are met or the maximum number of iterations is reached.
     *
     * \note This method assumes the input matrix `A` is symmetric positive definite,
     * and the provided `preconditioner` is also symmetric positive definite.
     *
     * \param A The sparse coefficient matrix in CSR format.
     * \param x On input, an initial guess for the solution vector; on output, the computed solution vector.
     * \param b The right-hand side vector.
     * \param precond A pointer to a `preconditioner` object to be used. The `build` method
     * of the preconditioner should have been called previously for matrix `A`.
     * \param control An `iter_control` object that manages the iteration process,
     * including convergence tolerance and maximum iterations.
     * \return An integer status code:
     * - `0` if the solver converged successfully within the specified tolerance.
     * - `1` if the maximum number of iterations was reached without convergence.
     * - Negative values might indicate issues with the matrix, preconditioner, or numerical stability.
     */
        int solve_precond(const csr_matrix&     A,
                          vector<double>&       x,
                          const vector<double>& b,
                          const preconditioner* precond,
                          iter_control          control);

        /*! \brief Generic solve method for the Conjugate Gradient solver (delegates to non-preconditioned or preconditioned).
     *
     * This method acts as a convenience wrapper. If `precond` is `nullptr`, it calls
     * `solve_nonprecond`. Otherwise, it calls `solve_precond`.
     *
     * \note This method assumes the input matrix `A` is symmetric positive definite.
     *
     * \param A The sparse coefficient matrix in CSR format.
     * \param x On input, an initial guess for the solution vector; on output, the computed solution vector.
     * \param b The right-hand side vector.
     * \param precond A pointer to a `preconditioner` object to be used. If `nullptr`, no preconditioning is applied.
     * \param control An `iter_control` object that manages the iteration process,
     * including convergence tolerance and maximum iterations.
     * \return An integer status code, consistent with `solve_nonprecond` or `solve_precond`.
     */
        int solve(const csr_matrix&     A,
                  vector<double>&       x,
                  const vector<double>& b,
                  const preconditioner* precond,
                  iter_control          control);

        /**
     * @brief Moves data from device memory to host memory.
     */
        void move_to_host();

        /**
     * @brief Moves data from host memory to device memory.
     */
        void move_to_device();

        /*! \brief Checks if the solver data is currently stored on the host (CPU).
     * \return `true` if the solver data is on the host, `false` otherwise (e.g., on a device).
     */
        bool is_on_host() const;
    };
}

#endif