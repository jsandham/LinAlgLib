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
// The above copyright noticeand this permission notice shall be included in all
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

/*! \file
 *  \brief bicgstab.h provides interface for stabilized bi-conjugate gradient solver
 */

/*! \ingroup linear_solvers
 *  \brief Stabilized Bi-conjugate gradient iterative linear solver
 *
 *  \details
 *  \p bicgstab solves the sparse linear system A*x = b using the stabilized
 *  bi-conjugate gradient iterative solver.
 *
 *  \note Requires the sparse matrix A to be symmetric
 *
 *  @param[in]
 *  csr_row_ptr array of \p n+1 elements that point to the start of every row of
 * the sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the
 * sparse CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements containing the values of the sparse
 *              CSR matrix.
 *  @param[inout]
 *  x           array of \p n elements containing the solution values of A*x=b
 *  @param[in]
 *  b           array of \p n elements containing the righthad side values of
 * A*x=b.
 *
 *  @param[in]
 *  n           size of the sparse CSR matrix
 *  @param[in]
 *  omega       omega
 *  @param[in]
 *  tol         stopping tolerance
 *  @param[in]
 *  max_iter    maximum iterations allowed
 *
 *  \retval number of iterations actually used in the solver. If -1 is returned,
 * the solver did not converge to a solution with the given input tolerance \p
 * tol.
 *
 *  \par Example
 *  \code{.c}
 *  int m, n, nnz;
 *  std::vector<int> csr_row_ptr;
 *  std::vector<int> csr_col_ind;
 *  std::vector<double> csr_val;
 *  load_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);
 *
 *  // Solution vector
 *  std::vector<double> x(m, 0.0);
 *
 *  // Righthand side vector
 *  std::vector<double> b(m, 1.0);
 *
 *  int it = bicgstab(csr_row_ptr.data(),
 *               csr_col_ind.data(),
 *               csr_val.data(),
 *               x.data(),
 *               b.data(),
 *               m,
 *               0.00001,
 *               1000);
 *  \endcode
 */
/**@{*/
int bicgstab(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, const int n,
        const double tol, const int max_iter);

#endif