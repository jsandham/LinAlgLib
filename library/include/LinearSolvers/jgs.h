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

#ifndef JGS_H
#define JGS_H

/*! \file
*  \brief JGS.h provides interface for jacobi, Gauss-Siedel, SOR, symmetric Gauss-Siedel, and symmetrix SOR solvers
*/

/*! \ingroup linear_solvers
*  \brief Jacobi iterative linear solver
*
*  \details
*  \p jac solves the sparse linear system A*x = b using the Jacobi iterative solver.
*
*  \note Convergence is only guaranteed if A is strictly diagonally dominant
* 
*  @param[in]
*  csr_row_ptr array of \p n+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  csr_val     array of \p nnz elements containing the values of the sparse
*              CSR matrix.
*  @param[inout]
*  x           array of \p n elements containing the solution values of A*x=b
*  @param[in]
*  b           array of \p n elements containing the righthad side values of A*x=b.
*
*  @param[in]
*  n           size of the sparse CSR matrix
*  @param[in]
*  tol         stopping tolerance
*  @param[in]
*  max_iter    maximum iterations allowed
*
*  \retval number of iterations actually used in the solver
*
*  \par Example
*  \code{.c}
*  \endcode
*/
/**@{*/
int jac(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, 
        const int n, const double tol, const int max_iter);

/*! \ingroup linear_solvers
*  \brief Gauss-Siedel iterative linear solver
*
*  \details
*  \p gs solves the sparse linear system A*x = b using the Gauss-Siedel iterative solver.
* 
*  \note Convergence is only guaranteed if A is either strictly diagonally dominant or symmetric and positive definite
*
*  @param[in]
*  csr_row_ptr array of \p n+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  csr_val     array of \p nnz elements containing the values of the sparse
*              CSR matrix.
*  @param[inout]
*  x           array of \p n elements containing the solution values of A*x=b
*  @param[in]
*  b           array of \p n elements containing the righthad side values of A*x=b.
*
*  @param[in]
*  n           size of the sparse CSR matrix
*  @param[in]
*  tol         stopping tolerance
*  @param[in]
*  max_iter    maximum iterations allowed
*
*  \retval number of iterations actually used in the solver
*
*  \par Example
*  \code{.c}
*  \endcode
*/
/**@{*/
int gs(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, 
       const int n, const double tol, const int max_iter);

/*! \ingroup linear_solvers
*  \brief SOR iterative linear solver
*
*  \details
*  \p sor solves the sparse linear system A*x = b using the successsive overrelaxation iterative solver.
*
*  @param[in]
*  csr_row_ptr array of \p n+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  csr_val     array of \p nnz elements containing the values of the sparse
*              CSR matrix.
*  @param[inout]
*  x           array of \p n elements containing the solution values of A*x=b
*  @param[in]
*  b           array of \p n elements containing the righthad side values of A*x=b.
*
*  @param[in]
*  n           size of the sparse CSR matrix
*  @param[in]
*  omega       relaxation factor between 0 and 2
*  @param[in]
*  tol         stopping tolerance
*  @param[in]
*  max_iter    maximum iterations allowed
*
*  \retval number of iterations actually used in the solver
*
*  \par Example
*  \code{.c}
*  \endcode
*/
/**@{*/
int sor(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, 
        const int n, const double omega, const double tol, const int max_iter);

/*! \ingroup linear_solvers
*  \brief Symmetric Gauss-Siedel iterative linear solver
*
*  \details
*  \p sgs solves the sparse linear system A*x = b using the symmetrix Gauss-Siedel iterative solver.
*
*  \note Requires the sparse matrix A to be symmetric
* 
*  @param[in]
*  csr_row_ptr array of \p n+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  csr_val     array of \p nnz elements containing the values of the sparse
*              CSR matrix.
*  @param[inout]
*  x           array of \p n elements containing the solution values of A*x=b
*  @param[in]
*  b           array of \p n elements containing the righthad side values of A*x=b.
*
*  @param[in]
*  n           size of the sparse CSR matrix
*  @param[in]
*  tol         stopping tolerance
*  @param[in]
*  max_iter    maximum iterations allowed
*
*  \retval number of iterations actually used in the solver
*
*  \par Example
*  \code{.c}
*  \endcode
*/
/**@{*/
int sgs(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, 
        const int n, const double tol, const int max_iter);

/*! \ingroup linear_solvers
*  \brief SSOR iterative linear solver
*
*  \details
*  \p ssor solves the sparse linear system A*x = b using the symmetric successive overrelaxation iterative solver.
*
*  \note Requires the sparse matrix A to be symmetric
* 
*  @param[in]
*  csr_row_ptr array of \p n+1 elements that point to the start of every row of the
*              sparse CSR matrix.
*  @param[in]
*  csr_col_ind array of \p nnz elements containing the column indices of the sparse
*              CSR matrix.
*  @param[in]
*  csr_val     array of \p nnz elements containing the values of the sparse
*              CSR matrix.
*  @param[inout]
*  x           array of \p n elements containing the solution values of A*x=b
*  @param[in]
*  b           array of \p n elements containing the righthad side values of A*x=b.
*
*  @param[in]
*  n           size of the sparse CSR matrix
*  @param[in]
*  omega       relaxation factor between 0 and 2
*  @param[in]
*  tol         stopping tolerance
*  @param[in]
*  max_iter    maximum iterations allowed
*
*  \retval number of iterations actually used in the solver
*
*  \par Example
*  \code{.c}
*  \endcode
*/
/**@{*/
int ssor(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* x, const double* b, 
         const int n, const double omega, const double tol, const int max_iter);

#endif
