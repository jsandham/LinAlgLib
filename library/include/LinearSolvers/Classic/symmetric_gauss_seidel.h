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

#ifndef SYMMETRIC_GAUSS_SEIDEL_H
#define SYMMETRIC_GAUSS_SEIDEL_H

#include "../../linalglib_export.h"

#include "../iter_control.h"

/*! \file
 *  \brief symmetric_gauss_seidel.h provides interface for symmetric gauss
 * seidel solver
 */

/*! \ingroup linear_solvers
 *  \brief Symmetric Gauss-Seidel iterative linear solver
 *
 *  \details
 *  \p sgs solves the sparse linear system \f$A\f$ * \f$x\f$ = \f$b\f$ using the
 *  symmetric Gauss-Seidel iterative solver.
 *
 *  \note Requires the sparse matrix \f$A\f$ to be symmetric
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
 *  control     iteration control struct specifying relative and absolut tolerence 
 *              as well as maximum iterations
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
LINALGLIB_API int sgs(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
        iter_control control);
/**@}*/

#endif