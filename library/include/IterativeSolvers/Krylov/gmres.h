//********************************************************************************
//
// MIT License
//
// Copyright(c) 2019-2025 James Sandham
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

#ifndef GMRES_H
#define GMRES_H

#include "../../linalg_export.h"

#include "../iter_control.h"
#include "../Preconditioner/preconditioner.h"

#include "../../vector.h"
#include "../../csr_matrix.h"

/*! \file
 *  \brief gmres.h provides interface for generalized minimum residual solver
 */

namespace linalg
{
/*! \brief A solver class implementing the Generalized Minimum Residual (GMRES) method.
 *
 * \details
 * This class provides functionality to solve large, sparse, **non-symmetric** systems
 * of linear equations of the form \f$A \cdot x = b\f$ using the Generalized Minimum Residual (GMRES)
 * iterative method. GMRES is a robust and widely used Krylov subspace method that
 * minimizes the Euclidean norm of the residual over a Krylov subspace. It is particularly
 * effective for non-symmetric problems where Conjugate Gradient (CG) or BiCGSTAB might struggle.
 * GMRES is typically implemented with restarts (GMRES(m)) to manage memory usage, as
 * the cost and memory requirements grow with each iteration.
 *
 * \section gmres_algorithm The GMRES(m) Algorithm
 *
 * The GMRES(m) algorithm seeks to find an approximate solution \f$\mathbf{x}\f$ to \f$A\mathbf{x} = \mathbf{b}\f$
 * by minimizing the residual \f$\|\mathbf{b} - A\mathbf{x}\|_2\f$ over the \f$m\f$-th Krylov subspace
 * \f$\mathcal{K}_m(A, \mathbf{r}_0) = \text{span}\{\mathbf{r}_0, A\mathbf{r}_0, \dots, A^{m-1}\mathbf{r}_0\}\f$,
 * where \f$\mathbf{r}_0 = \mathbf{b} - A\mathbf{x}^{(0)}\f$ is the initial residual.
 *
 * The restarted GMRES(m) algorithm steps are (for each restart cycle):
 *
 * 1. **Start Iteration (Restart Cycle \f$j\f$):**
 * Compute initial residual \f$\mathbf{r}_0 = \mathbf{b} - A \mathbf{x}^{(j-1)}\f$.
 * Set \f$\beta = \|\mathbf{r}_0\|_2\f$.
 * Set \f$\mathbf{v}_1 = \mathbf{r}_0 / \beta\f$.
 * Initialize the \f$m \times m\f$ Hessenberg matrix \f$\bar{H}_m\f$ (or \f$H_{m+1, m}\f$ in some notations) with zeros.
 * Initialize the \f$(m+1)\f$-length residual vector \f$\mathbf{g}\f$ with \f$\mathbf{g}_1 = \beta\f$ and zeros elsewhere.
 *
 * 2. **Arnoldi Iteration (Build Orthonormal Basis for Krylov Subspace):**
 * For \f$i = 1, \dots, m\f$:
 * a. Compute \f$\mathbf{w} = A \mathbf{v}_i\f$.
 * b. For \f$j = 1, \dots, i\f$:
 * \f$h_{ji} = \mathbf{w}^T \mathbf{v}_j\f$
 * \f$\mathbf{w} = \mathbf{w} - h_{ji} \mathbf{v}_j\f$
 * c. \f$h_{i+1, i} = \|\mathbf{w}\|_2\f$
 * d. If \f$h_{i+1, i} = 0\f$ (lucky breakdown), set \f$m=i\f$ and exit Arnoldi.
 * e. \f$\mathbf{v}_{i+1} = \mathbf{w} / h_{i+1, i}\f$
 *
 * 3. **Form Least Squares Problem:**
 * At the end of \f$m\f$ iterations (or a breakdown), we have an orthonormal basis \f$V_m = [\mathbf{v}_1, \dots, \mathbf{v}_m]\f$
 * and an \f$(m+1) \times m\f$ Hessenberg matrix \f$\bar{H}_m\f$ such that \f$A V_m = V_{m+1} \bar{H}_m\f$.
 * The approximate solution \f$\mathbf{x}^{(j)} = \mathbf{x}^{(j-1)} + V_m \mathbf{y}_m\f$ is sought by minimizing:
 * \f$ \|\mathbf{b} - A \mathbf{x}^{(j)}\|_2 = \|\mathbf{r}_0 - A V_m \mathbf{y}_m\|_2 = \|\mathbf{r}_0 - V_{m+1} \bar{H}_m \mathbf{y}_m\|_2 \f$
 * Since \f$V_{m+1}\f$ is orthonormal, this simplifies to:
 * \f$ \|\beta \mathbf{e}_1 - \bar{H}_m \mathbf{y}_m\|_2 \f$
 * where \f$\mathbf{e}_1\f$ is the first standard basis vector of length \f$m+1\f$.
 *
 * 4. **Solve Least Squares Problem:**
 * Solve this \f$m+1 \times m\f$ least squares problem using Givens rotations (or QR factorization) to triangularize \f$\bar{H}_m\f$
 * and apply the rotations to \f$\mathbf{g}\f$. This results in an upper triangular matrix \f$R_m\f$ and modified vector \f$\hat{\mathbf{g}}\f$:
 * \f$ \min \|\hat{\mathbf{g}} - R_m \mathbf{y}_m\|_2 \f$
 * The minimum residual is \f$|\hat{\mathbf{g}}_{m+1}|\f$. If this is below tolerance, stop.
 * Solve for \f$\mathbf{y}_m\f$ using back substitution on the upper \f$m \times m\f$ part of \f$R_m\f$ and \f$\hat{\mathbf{g}}\f$.
 *
 * 5. **Update Solution:**
 * \f$\mathbf{x}^{(j)} = \mathbf{x}^{(j-1)} + V_m \mathbf{y}_m\f$.
 *
 * 6. **Check Convergence and Restart:**
 * If converged, exit. Otherwise, if \f$m\f$ iterations are completed, restart from step 1 with \f$\mathbf{x}^{(j)}\f$ as the new initial guess.
 *
 * \subsection gmres_preconditioned Preconditioned GMRES (PGMRES)
 *
 * For the preconditioned version, a preconditioner \f$M \approx A^{-1}\f$ is used. The core idea is to solve
 * \f$M^{-1}A \mathbf{x} = M^{-1}\mathbf{b}\f$ (left preconditioning) or \f$A M^{-1} (M \mathbf{x}) = \mathbf{b}\f$ (right preconditioning).
 * Here, we generally consider left preconditioning:
 *
 * The algorithm steps are modified by applying the preconditioner to the residual before constructing the Krylov subspace:
 *
 * 1. **Start Iteration (Restart Cycle \f$j\f$):**
 * Compute initial residual \f$\mathbf{r}_0 = \mathbf{b} - A \mathbf{x}^{(j-1)}\f$.
 * Solve \f$M \tilde{\mathbf{r}}_0 = \mathbf{r}_0\f$ (preconditioning step).
 * Set \f$\beta = \|\tilde{\mathbf{r}}_0\|_2\f$.
 * Set \f$\mathbf{v}_1 = \tilde{\mathbf{r}}_0 / \beta\f$.
 * Initialize the \f$m \times m\f$ Hessenberg matrix \f$\bar{H}_m\f$ with zeros.
 * Initialize the \f$(m+1)\f$-length residual vector \f$\mathbf{g}\f$ with \f$\mathbf{g}_1 = \beta\f$ and zeros elsewhere.
 *
 * 2. **Arnoldi Iteration (Build Orthonormal Basis for Preconditioned Krylov Subspace):**
 * For \f$i = 1, \dots, m\f$:
 * a. Compute \f$\mathbf{w}' = A \mathbf{v}_i\f$.
 * b. Solve \f$M \mathbf{w} = \mathbf{w}'\f$ (preconditioning step).
 * c. For \f$j = 1, \dots, i\f$:
 * \f$h_{ji} = \mathbf{w}^T \mathbf{v}_j\f$
 * \f$\mathbf{w} = \mathbf{w} - h_{ji} \mathbf{v}_j\f$
 * d. \f$h_{i+1, i} = \|\mathbf{w}\|_2\f$
 * e. If \f$h_{i+1, i} = 0\f$, set \f$m=i\f$ and exit Arnoldi.
 * f. \f$\mathbf{v}_{i+1} = \mathbf{w} / h_{i+1, i}\f$
 *
 * 3. **Form Least Squares Problem and Solve & Update Solution** (Steps 3-5 are the same as non-preconditioned GMRES).
 *
 * \section gmres_convergence Convergence and Memory
 *
 * GMRES is guaranteed to converge for any non-singular matrix \f$A\f$. The number of iterations
 * for convergence can be high for ill-conditioned or highly non-symmetric problems.
 *
 * A major characteristic of GMRES is that its memory requirements and computational cost per
 * iteration grow linearly with the number of iterations (\f$m\f$). To manage this, GMRES is
 * often restarted every \f$m\f$ iterations (GMRES(m)). The parameter `restart` in this class
 * specifies this maximum subspace dimension. A larger `restart` value generally leads to
 * faster convergence (fewer restarts), but increases memory usage and cost per iteration.
 *
 * \section gmres_example Example Usage
 * Below is a simplified example demonstrating how to use the `gmres_solver` class
 * with and without a preconditioner. This assumes `csr_matrix`, `vector`, `preconditioner`,
 * and `iter_control` classes are properly defined and functional.
 *
 * \code
 * #include "linalglib.h"
 * #include <iostream>
 * #include <vector>
 *
 * int main() {
 * // Define a sample sparse non-symmetric matrix A
 * // A = [ 4 -2  0 ]
 * //     [ 1  4 -1 ]
 * //     [ 0  1  5 ]
 * std::vector<int> row_ptr = {0, 2, 5, 7};
 * std::vector<int> col_ind = {0, 1, 0, 1, 2, 1, 2};
 * std::vector<double> val = {4.0, -2.0, 1.0, 4.0, -1.0, 1.0, 5.0};
 *
 * int m_dim = 3; // Number of rows/cols
 * int n_dim = 3;
 * int nnz = 7;
 *
 * csr_matrix A(row_ptr, col_ind, val, m_dim, n_dim, nnz);
 *
 * // Define the right-hand side vector b
 * std::vector<double> b_data = {6.0, 8.0, 11.0};
 * vector b(b_data);
 *
 * // Define an initial guess for the solution vector x (e.g., all zeros)
 * vector x(m_dim);
 * x.zeros();
 *
 * // Set up iteration control
 * iter_control control;
 * control.max_iterations = 200;
 * control.tolerance = 1e-7;
 *
 * // Set GMRES restart parameter (e.g., 30)
 * int gmres_restart = 30;
 *
 * // Create a GMRES solver instance
 * gmres_solver solver;
 *
 * // Build the solver (allocates memory based on matrix size and restart parameter)
 * solver.build(A, gmres_restart);
 *
 * // --- Non-preconditioned solve ---
 * std::cout << "--- Starting non-preconditioned GMRES(" << gmres_restart << ") solver ---" << std::endl;
 * vector x_nonprecond = x; // Copy initial guess
 * int status_nonprecond = solver.solve_nonprecond(A, x_nonprecond, b, control);
 *
 * if (status_nonprecond == 0) {
 * std::cout << "Non-preconditioned GMRES converged successfully!" << std::endl;
 * } else {
 * std::cout << "Non-preconditioned GMRES did NOT converge. Status code: " << status_nonprecond << std::endl;
 * }
 * std::cout << "Approximate solution x (non-preconditioned):" << std::endl;
 * for (int i = 0; i < x_nonprecond.get_size(); ++i) {
 * std::cout << "x[" << i << "] = " << x_nonprecond[i] << std::endl;
 * }
 *
 * // --- Preconditioned solve (e.g., with Jacobi preconditioner) ---
 * std::cout << "\n--- Starting preconditioned GMRES(" << gmres_restart << ") solver (Jacobi) ---" << std::endl;
 * jacobi_precond jacobi_prec;
 * jacobi_prec.build(A); // Build the preconditioner for matrix A
 *
 * vector x_precond = x; // Copy initial guess
 * int status_precond = solver.solve_precond(A, x_precond, b, &jacobi_prec, control);
 *
 * if (status_precond == 0) {
 * std::cout << "Preconditioned GMRES converged successfully!" << std::endl;
 * } else {
 * std::cout << "Preconditioned GMRES did NOT converge. Status code: " << status_precond << std::endl;
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
class gmres_solver
{
private:
    /*! \brief Upper Hessenberg matrix `H` (or \f$\bar{H}_m\f$), used in the Arnoldi process. */
    vector<double> H; // Stores coefficients of Hessenberg matrix, often flattened
    /*! \brief Orthonormal basis vectors `Q` (or `V`), storing the Krylov subspace basis.
     * \details This is typically a collection of `m+1` vectors, often stored contiguously
     * in a larger vector or an array of vectors.
     */
    vector<double> Q; // Stores orthonormal basis vectors v_1, ..., v_{m+1}
    /*! \brief Cosine rotation values from Givens rotations. */
    vector<double> c;
    /*! \brief Sine rotation values from Givens rotations. */
    vector<double> s;
    /*! \brief Residual vector or temporary storage for vector operations. */
    vector<double> res;
    /*! \brief Intermediate vector for preconditioning or other operations. */
    vector<double> z;

    /*! \brief The restart parameter `m` for GMRES(m).
     * \details This defines the maximum dimension of the Krylov subspace before restarting.
     * A smaller `restart` value means less memory usage but potentially more restarts.
     */
    int restart;

public:
    /*! \brief Default constructor.
     * Initializes a new `gmres_solver` object.
     */
    gmres_solver();

    /*! \brief Destructor.
     * Cleans up any resources allocated by the `gmres_solver` object.
     */
    ~gmres_solver();

    /*! \brief Deleted copy constructor.
     * Prevents direct copying of `gmres_solver` objects to ensure proper memory management.
     */
    gmres_solver (const gmres_solver&) = delete;

    /*! \brief Deleted copy assignment operator.
     * Prevents direct assignment of one `gmres_solver` object to another to ensure proper memory management.
     */
    gmres_solver& operator= (const gmres_solver&) = delete;

    /*! \brief Builds necessary data structures for the GMRES solver.
     * \details
     * This method allocates and resizes the internal work vectors (`H`, `Q`, `c`, `s`, `res`, `z`)
     * based on the matrix dimensions and the `restart` parameter.
     * \param A The sparse matrix in CSR format for which the solver is being built.
     * \param restart The restart parameter `m` for GMRES(m).
     */
    void build(const csr_matrix& A, int restart);

    /*! \brief Solves the linear system \f$A \cdot x = b\f$ using the non-preconditioned GMRES method.
     *
     * This method implements the GMRES(m) algorithm without any explicit preconditioning.
     * It iteratively refines the solution `x` until the convergence criteria from `control`
     * are met or the maximum number of iterations (or restarts) is reached.
     *
     * \param A The sparse coefficient matrix in CSR format.
     * \param x On input, an initial guess for the solution vector; on output, the computed solution vector.
     * \param b The right-hand side vector.
     * \param control An `iter_control` object that manages the iteration process,
     * including convergence tolerance and total maximum iterations.
     * \return An integer status code:
     * - `0` if the solver converged successfully within the specified tolerance.
     * - `1` if the maximum number of iterations was reached without convergence.
     * - Negative values indicate potential breakdowns (e.g., null vectors).
     */
    int solve_nonprecond(const csr_matrix& A, vector<double>& x, const vector<double>& b, iter_control control);

    /*! \brief Solves the linear system \f$A \cdot x = b\f$ using the preconditioned GMRES method.
     *
     * This method implements the GMRES(m) algorithm with the provided preconditioner.
     * It iteratively refines the solution `x` until the convergence criteria from `control`
     * are met or the maximum number of iterations (or restarts) is reached.
     *
     * \param A The sparse coefficient matrix in CSR format.
     * \param x On input, an initial guess for the solution vector; on output, the computed solution vector.
     * \param b The right-hand side vector.
     * \param precond A pointer to a `preconditioner` object to be used. The `build` method
     * of the preconditioner should have been called previously for matrix `A`.
     * \param control An `iter_control` object that manages the iteration process,
     * including convergence tolerance and total maximum iterations.
     * \return An integer status code:
     * - `0` if the solver converged successfully within the specified tolerance.
     * - `1` if the maximum number of iterations was reached without convergence.
     * - Negative values indicate potential breakdowns or issues with the preconditioner.
     */
    int solve_precond(const csr_matrix& A, vector<double>& x, const vector<double>& b, const preconditioner *precond, iter_control control);

    /*! \brief Generic solve method for the GMRES solver (delegates to non-preconditioned or preconditioned).
     *
     * This method acts as a convenience wrapper. If `precond` is `nullptr`, it calls
     * `solve_nonprecond`. Otherwise, it calls `solve_precond`.
     *
     * \param A The sparse coefficient matrix in CSR format.
     * \param x On input, an initial guess for the solution vector; on output, the computed solution vector.
     * \param b The right-hand side vector.
     * \param precond A pointer to a `preconditioner` object to be used. If `nullptr`, no preconditioning is applied.
     * \param control An `iter_control` object that manages the iteration process,
     * including convergence tolerance and total maximum iterations.
     * \return An integer status code, consistent with `solve_nonprecond` or `solve_precond`.
     */
    int solve(const csr_matrix& A, vector<double>& x, const vector<double>& b, const preconditioner *precond, iter_control control);
};
}

#endif
