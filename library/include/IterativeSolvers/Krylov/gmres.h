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

#include "../../linalglib_export.h"

#include "../iter_control.h"
#include "../Preconditioner/preconditioner.h"

#include "../../vector.h"
#include "../../csr_matrix.h"

/*! \file
 *  \brief gmres.h provides interface for generalized minimum residual solver
 */

/*! \ingroup iterative_solvers
 * \brief Generalized minimum residual iterative linear solver.
 *
 * \details
 * \p gmres solves the sparse linear system \f$A x = b\f$ using the Generalized
 * Minimum RESidual (GMRES) iterative solver. GMRES is a powerful method for
 * solving general non-symmetric linear systems. It works by iteratively
 * building an orthonormal basis for the Krylov subspace and finding the
 * solution that minimizes the residual norm over this subspace.
 *
 * **Algorithm Overview:**
 *
 * The Generalized Minimum Residual method constructs a sequence of orthonormal
 * vectors that span the Krylov subspace \f$K_k(A, r_0) = span\{r_0, Ar_0, ..., A^{k-1}r_0\}\f$,
 * where \f$r_0 = b - A x_0\f$ is the initial residual. At each iteration \f$k\f$,
 * it finds the approximate solution \f$x_k = x_0 + y_k\f$, where \f$y_k\f$ lies in
 * the Krylov subspace and minimizes the Euclidean norm of the residual \f$||b - A x_k||_2\f$.
 *
 * The algorithm typically proceeds as follows:
 *
 * 1. Initialize: Set the initial guess \f$x_0\f$, compute the initial residual
 * \f$r_0 = b - A x_0\f$. If \f$||r_0||_2\f$ is small enough, return \f$x_0\f$.
 * 2. Initialize the first basis vector: \f$v_1 = r_0 / ||r_0||_2\f$.
 * 3. For \f$k = 1, 2, ..., \text{max\_iter}\f$:
 * a. Compute the next Krylov vector: \f$w = A v_k\f$.
 * b. Orthogonalize \f$w\f$ against the previous basis vectors \f$v_1, ..., v_k\f$
 * using Gram-Schmidt or Householder orthogonalization. This yields a new
 * orthonormal vector \f$v_{k+1}\f$ and a set of coefficients stored in a
 * Hessenberg matrix \f$H_k\f$.
 * c. Form an upper Hessenberg matrix \f$\bar{H}_k\f$ by adding the row corresponding
 * to the norm of the orthogonalized vector.
 * d. Find the vector \f$y_k\f$ that minimizes \f$||\beta e_1 - \bar{H}_k y_k||_2\f$,
 * where \f$\beta = ||r_0||_2\f$ and \f$e_1 = [1, 0, ..., 0]^T\f$. This is a
 * small least-squares problem.
 * e. Update the approximate solution: \f$x_k = x_0 + V_k y_k\f$, where \f$V_k = [v_1, ..., v_k]\f$.
 * f. Compute the residual norm. Check for convergence based on the tolerance
 * specified in \p control. If converged, return \f$x_k\f$.
 *
 * **Restarting:**
 *
 * The cost of the GMRES method increases with the number of iterations due to the
 * growing size of the Krylov subspace. To limit this cost, a common strategy is
 * to restart the algorithm after a certain number of iterations (\p restart).
 * When restarted, the current approximate solution is taken as the new initial
 * guess, and the Krylov subspace is built from the new residual. This is often
 * referred to as GMRES(\p restart).
 *
 * **Preconditioning:**
 *
 * Preconditioning can significantly improve the convergence rate of GMRES, especially
 * for ill-conditioned systems. A preconditioner \f$M\f$ is used to transform the
 * original system into a system that is easier to solve. This can be done by
 * either left-preconditioning (\f$M^{-1} A x = M^{-1} b\f$) or right-preconditioning
 * (\f$A M^{-1} y = b, x = M^{-1} y\f$). The choice of preconditioning can affect
 * the implementation details of the GMRES algorithm. The \p precond parameter
 * allows you to provide a preconditioner object that can be applied during the
 * GMRES iterations.
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
 * apply the preconditioner (e.g., solve \f$M z = r\f$ for left preconditioning
 * or apply \f$M^{-1}\f$ for right preconditioning). If no preconditioning is
 * desired, a null pointer or an identity preconditioner can be used.
 * @param[in] control
 * Structure of type \ref iter_control specifying the convergence criteria
 * (relative tolerance, absolute tolerance) and the maximum number of iterations
 * (before potential restart, if \p restart > 0) for the GMRES solver. The
 * solver will stop if either the relative or absolute tolerance is met, or if
 * the maximum number of iterations is reached within a restart cycle.
 * @param[in] restart
 * The number of iterations after which the GMRES algorithm is restarted.
 * Setting \p restart to a positive integer value (e.g., 30) limits the size of
 * the Krylov subspace built in each cycle. If \p restart is 0 or negative, the
 * algorithm will not restart until convergence or the maximum number of
 * iterations is reached.
 *
 * @retval int
 * The total number of iterations performed by the GMRES solver until convergence
 * or the maximum number of iterations is reached. Returns -1 if the solver did
 * not converge within the specified tolerance and maximum number of iterations.
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
 * control.max_iter = 100;
 *
 * int it = gmres(csr_row_ptr.data(),
 * csr_col_ind.data(),
 * csr_val.data(),
 * x.data(),
 * b.data(),
 * m,
 * &precond,
 * control,
 * 30);
 *
 * std::cout << "Number of iterations: " << it << std::endl;
 *
 * return 0;
 * }
 * \endcode
 */
LINALGLIB_API int gmres(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
    const preconditioner *precond, iter_control control, int restart);

LINALGLIB_API int gmres(const csr_matrix2& A, vector2& x, const vector2& b, const preconditioner *precond, iter_control control, 
       int restart);













class gmres_solver
{
private:
    vector2 H;
    vector2 Q;
    vector2 c;
    vector2 s;
    vector2 res;
    vector2 z;

    int restart;

public:
    gmres_solver();
    ~gmres_solver();

    gmres_solver (const gmres_solver&) = delete;
    gmres_solver& operator= (const gmres_solver&) = delete;

    void build(const csr_matrix2& A, int restart);
    int solve_nonprecond(const csr_matrix2& A, vector2& x, const vector2& b, iter_control control);
    int solve_precond(const csr_matrix2& A, vector2& x, const vector2& b, const preconditioner *precond, iter_control control);
    int solve(const csr_matrix2& A, vector2& x, const vector2& b, const preconditioner *precond, iter_control control);
};


#endif
