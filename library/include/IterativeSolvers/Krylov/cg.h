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

#include "../../linalglib_export.h"

#include "../iter_control.h"
#include "../Preconditioner/preconditioner.h"

#include "../../vector.h"
#include "../../csr_matrix.h"

/*! \file
 *  \brief cg.h provides interface for conjugate gradient solvers
 */

/*! \ingroup iterative_solvers
 * \brief Preconditioned conjugate gradient iterative linear solver.
 *
 * \details
 * \p cg solves the sparse linear system \f$A x = b\f$ using the preconditioned
 * conjugate gradient (PCG) iterative solver. The conjugate gradient method
 * is an efficient algorithm for solving symmetric positive-definite linear
 * systems. Preconditioning is often used to accelerate the convergence of
 * the CG method by transforming the original system into one with more
 * favorable spectral properties.
 *
 * **Algorithm Overview:**
 *
 * The Preconditioned Conjugate Gradient method iteratively refines an initial
 * guess for the solution vector \f$x\f$. At each iteration, it computes a
 * search direction (\f$p\f$) that is conjugate to the previous search
 * directions with respect to the preconditioned system. The step length
 * (\f$\alpha\f$) along this direction is chosen to minimize the preconditioned
 * residual. The solution and the residual are then updated. The preconditioning
 * is applied by solving a related system involving the preconditioner matrix
 * (\f$M\f$) to obtain a preconditioned residual (\f$z\f$).
 *
 * The algorithm typically proceeds as follows:
 *
 * 1. Initialize: Set the initial guess \f$x_0\f$, compute the initial residual
 * \f$r_0 = b - A x_0\f$. If the initial residual is sufficiently small,
 * return \f$x_0\f$.
 * 2. Precondition: Solve \f$M z_0 = r_0\f$ for \f$z_0\f$.
 * 3. Initialize the search direction: \f$p_0 = z_0\f$.
 * 4. For \f$k = 0, 1, 2, ...\f$ until convergence:
 * a. Compute the matrix-vector product: \f$w_k = A p_k\f$.
 * b. Compute the step length: \f$\alpha_k = (z_k^T r_k) / (p_k^T w_k)\f$.
 * c. Update the solution: \f$x_{k+1} = x_k + \alpha_k p_k\f$.
 * d. Update the residual: \f$r_{k+1} = r_k - \alpha_k w_k\f$.
 * e. Check for convergence: If \f$||r_{k+1}||<\f$ tolerance, return \f$x_{k+1}\f$.
 * f. Precondition: Solve \f$M z_{k+1} = r_{k+1}\f$ for \f$z_{k+1}\f$.
 * g. Compute the scalar \f$\beta_k = (z_{k+1}^T r_{k+1}) / (z_k^T r_k)\f$.
 * h. Update the search direction: \f$p_{k+1} = z_{k+1} + \beta_k p_k\f$.
 *
 * The \p restart_iter parameter allows for restarting the CG algorithm periodically
 * to potentially improve convergence, especially when the eigenvalue distribution
 * of the preconditioned matrix is unfavorable. After \p restart_iter iterations,
 * the search direction is reset to the preconditioned residual.
 *
 * \note Requires the sparse matrix \f$A\f$ to be symmetric. While the standard
 * Conjugate Gradient method requires the matrix to be symmetric positive-definite,
 * the preconditioned version can sometimes be applied to symmetric indefinite
 * systems, although convergence is not guaranteed. The preconditioner \f$M\f$
 * should ideally be symmetric and positive-definite as well to preserve the
 * properties of the CG method.
 *
 * @param[in] csr_row_ptr
 * Array of \p n+1 elements that point to the start of every row of
 * the sparse CSR matrix \f$A\f$.
 * @param[in] csr_col_ind
 * Array of \p nnz elements containing the column indices of the
 * non-zero entries in the sparse CSR matrix \f$A\f$.
 * @param[in] csr_val
 * Array of \p nnz elements containing the numerical values of the
 * non-zero entries in the sparse CSR matrix \f$A\f$.
 * @param[inout] x
 * Array of \p n elements containing the initial guess for the solution
 * vector \f$x\f$ of the system \f$A x = b\f$. On output, if the solver
 * converges, this array will contain the computed solution.
 * @param[in] b
 * Array of \p n elements containing the right-hand side vector \f$b\f$ of
 * the linear system \f$A x = b\f$.
 * @param[in] n
 * The dimension of the square sparse CSR matrix \f$A\f$ (number of rows or columns)
 * and the size of the vectors \f$x\f$ and \f$b\f$.
 * @param[in] precond
 * Pointer to a preconditioner object. This object must provide a method to
 * solve a linear system of the form \f$M z = r\f$, where \f$M\f$ is the
 * preconditioner matrix and \f$r\f$ is the residual vector. The specific
 * type of the preconditioner (e.g., Jacobi, ILU, ICC) is determined by the
 * actual object pointed to by \p precond. If no preconditioning is desired,
 * a null pointer or an identity preconditioner can be used.
 * @param[in] control
 * Structure of type \ref iter_control specifying the convergence criteria
 * (relative tolerance, absolute tolerance) and the maximum number of iterations
 * for the CG solver. The solver will stop if either the relative or absolute
 * tolerance is met, or if the maximum number of iterations is reached.
 * @param[in] restart_iter
 * The number of iterations after which the conjugate gradient algorithm is
 * restarted. A restart involves resetting the search direction to the
 * preconditioned residual. Setting \p restart_iter to a large value (greater
 * than the expected number of iterations) effectively disables restarting.
 *
 * @retval int
 * The number of iterations performed by the CG solver. If the solver converges
 * within the specified tolerance and maximum iterations, the return value is
 * the number of iterations taken. If the solver does not converge, -1 is
 * returned.
 *
 * \par Example
 * \code{.cpp}
 * #include <vector>
 * #include <iostream>
 * #include "linalg.h"
 *
 * int main() {
 * int m, n, nnz;
 * std::vector<int> csr_row_ptr;
 * std::vector<int> csr_col_ind;
 * std::vector<double> csr_val;
 * const char* matrix_file = "my_matrix.mtx";
 * load_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);
 *
 * // Solution vector
 * std::vector<double> x(m, 0.0);
 *
 * // Righthand side vector
 * std::vector<double> b(m, 1.0);
 *
 * // ILU preconditioner
 * ilu_precond precond;
 * precond.build(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), m, n, nnz);
 *
 * iter_control control;
 * control.rel_tol = 1e-8;
 * control.max_iter = 1000;
 *
 * int it = cg(csr_row_ptr.data(),
 * csr_col_ind.data(),
 * csr_val.data(),
 * x.data(),
 * b.data(),
 * m,
 * &precond,
 * control,
 * 100);
 *
 * std::cout << "Number of iterations: " << it << std::endl;
 *
 * return 0;
 * }
 * \endcode
 */
class cg_solver
{
private:
    vector z;
    vector p;
    vector res;

    int restart_iter;

public:
    cg_solver();
    ~cg_solver();

    cg_solver (const cg_solver&) = delete;
    cg_solver& operator= (const cg_solver&) = delete;

    void build(const csr_matrix& A);
    int solve_nonprecond(const csr_matrix& A, vector& x, const vector& b, iter_control control);
    int solve_precond(const csr_matrix& A, vector& x, const vector& b, const preconditioner *precond, iter_control control);
    int solve(const csr_matrix& A, vector& x, const vector& b, const preconditioner *precond, iter_control control);
};

#endif