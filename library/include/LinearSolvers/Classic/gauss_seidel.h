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

#include "../../linalglib_export.h"

/*! \file
 *  \brief gauss_seidel.h provides interface for gauss seidel solver
 */

/*! \ingroup linear_solvers
 *  \brief Gauss-Siedel iterative linear solver
 *
 *  \details
 *  \p gs solves the sparse linear system \f$A\f$ * \f$x\f$ = \f$b\f$ using the
 *  Gauss-Siedel iterative solver.
 *
 *  \note Convergence is only guaranteed if \f$A\f$ is either strictly
 *  diagonally dominant or symmetric and positive definite
 *
 *  @param[in]
 *  csr_row_ptr array of \p n+1 elements that point to the start of every row of
 *              the sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the
 *              sparse CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements containing the values of the sparse
 *              CSR matrix.
 *  @param[inout]
 *  x           array of \p n elements containing the solution values of
 *              \f$A\f$ * \f$x\f$ = \f$b\f$
 *  @param[in]
 *  b           array of \p n elements containing the righthad side values of
 *              \f$A\f$ * \f$x\f$ = \f$b\f$.
 *
 *  @param[in]
 *  n           size of the sparse CSR matrix
 *  @param[in]
 *  tol         stopping tolerance
 *  @param[in]
 *  max_iter    maximum iterations allowed
 *
 *  \retval number of iterations actually used in the solver. If -1 is returned,
 *  the solver did not converge to a solution with the given input tolerance \p
 *  tol.
 *
 *  \par Example
 *  \code{.c}
 *  \endcode
 */
/**@{*/
LINALGLIB_API int gs(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
       double tol, int max_iter);
/**@}*/

#endif