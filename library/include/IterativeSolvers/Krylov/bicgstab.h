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

#ifndef BICGSTAB_H
#define BICGSTAB_H

#include "../../linalglib_export.h"

#include "../iter_control.h"
#include "../Preconditioner/preconditioner.h"

/*! \file
 *  \brief bicgstab.h provides interface for stabilized bi-conjugate gradient solver
 */

/*! \ingroup iterative_solvers
 * \brief Preconditioned Stabilized Bi-conjugate gradient iterative linear solver.
 *
 * \details
 * \p bicgstab solves the sparse linear system \f$A x = b\f$ using the
 * preconditioned stabilized bi-conjugate gradient (BiCGStab) iterative solver.
 * BiCGStab is an efficient method for solving general non-symmetric linear
 * systems, often exhibiting smoother convergence behavior than the standard
 * Bi-Conjugate Gradient (BiCG) method. Preconditioning is crucial for
 * accelerating the convergence, especially for ill-conditioned systems.
 *
 * **Algorithm Overview:**
 *
 * The BiCGStab algorithm is a variant of BiCG that uses a stabilization
 * technique to avoid the irregular convergence patterns that can occur with BiCG.
 * It involves the use of a "shadow" residual and a polynomial to minimize the
 * residual in each iteration.
 *
 * The preconditioned BiCGStab algorithm typically proceeds as follows:
 *
 * 1. Initialize: Set the initial guess \f$x_0\f$, compute the initial residual
 * \f$r_0 = b - A x_0\f$. If \f$||r_0||_2\f$ is small enough, return \f$x_0\f$.
 * 2. Precondition the initial residual: Solve \f$M z_0 = r_0\f$ for \f$z_0\f$.
 * 3. Initialize: \f$p_0 = z_0\f$, \f$r_{hat, 0} = r_0\f$ (arbitrary, often \f$r_0\f$), \f$\rho_0 = \alpha = \omega_0 = 1\f$.
 * 4. For \f$i = 0, 1, 2, ..., \text{max\_iter}\f$:
 * a. Compute \f$\rho_i = (r_{hat, 0}^T r_i)\f$. If \f$\rho_i = 0\f$, the method breaks down.
 * b. Compute \f$\beta = (\rho_i / \rho_{i-1}) (\alpha / \omega_{i-1})\f$ (with \f$\beta_0 = 0\f$).
 * c. Update the search direction: \f$p_i = z_i + \beta (p_{i-1} - \omega_{i-1} v_{i-1})\f$ (with \f$p_{-1} = v_{-1} = 0\f$).
 * d. Compute \f$v_i = A p_i\f$.
 * e. Compute \f$\alpha = (\rho_i) / (r_{hat, 0}^T v_i)\f$.
 * f. Update the intermediate solution: \f$s = r_i - \alpha v_i\f$.
 * g. Precondition the intermediate residual: Solve \f$M t = s\f$ for \f$t\f$.
 * h. Compute \f$\omega_i = (t^T s) / (t^T t)\f$.
 * i. Update the solution: \f$x_{i+1} = x_i + \alpha p_i + \omega_i t\f$.
 * j. Update the residual: \f$r_{i+1} = s - \omega_i t\f$.
 * k. Check for convergence: If \f$||r_{i+1}||<\f$ tolerance, return \f$x_{i+1}\f$.
 * l. Precondition the residual: Solve \f$M z_{i+1} = r_{i+1}\f$ for \f$z_{i+1}\f$.
 *
 * **Note:** The original documentation incorrectly states that the matrix \f$A\f$
 * needs to be symmetric. BiCGStab is specifically designed for non-symmetric
 * linear systems. The preconditioner \f$M\f$ is still beneficial to be
 * symmetric and positive-definite if \f$A\f$ has such properties, but it's not a
 * strict requirement for the applicability of BiCGStab.
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
 * preconditioner matrix and \f$r\f$ is a vector (typically the residual or
 * an intermediate vector). If no preconditioning is desired, a null pointer
 * or an identity preconditioner can be used.
 * @param[in] control
 * Structure of type \ref iter_control specifying the convergence criteria
 * (relative tolerance, absolute tolerance) and the maximum number of iterations
 * for the BiCGStab solver. The solver will stop if either the relative or
 * absolute tolerance is met, or if the maximum number of iterations is reached.
 *
 * @retval int
 * The number of iterations performed by the BiCGStab solver. If the solver
 * converges within the specified tolerance and maximum iterations, the return
 * value is the number of iterations taken. If the solver does not converge,
 * -1 is returned.
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
 * int it = bicgstab(csr_row_ptr.data(),
 * csr_col_ind.data(),
 * csr_val.data(),
 * x.data(),
 * b.data(),
 * m,
 * &precond,
 * control);
 *
 * std::cout << "Number of iterations: " << it << std::endl;
 *
 * return 0;
 * }
 * \endcode
 */
LINALGLIB_API int bicgstab(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
    const preconditioner *precond, iter_control control);

#endif
