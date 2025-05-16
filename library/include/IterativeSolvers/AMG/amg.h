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

#ifndef AMG_H
#define AMG_H

#include <vector>

#include "amg_util.h"

#include "../../linalglib_export.h"

#include "../iter_control.h"

/*! \file
 *  \brief amg.h provides interface for algebraic multigrid solver
 */

/*! \ingroup iterative_solvers
 *  \brief Data structure for storing matrix heirarchy used in algrebric multigrid solver
 */
struct heirarchy
{
    /**< Prolongation matrices. */
    std::vector<csr_matrix> prolongations;

    /**< Restriction matrices. */
    std::vector<csr_matrix> restrictions;

    /**< Coarse level matrices. */
    std::vector<csr_matrix> A_cs;

    /**< Number of levels in the heirarchy. */
    int total_levels;
};

/*! \ingroup iterative_solvers
 *  \brief Cycle type used in algrebric multigrid solver
 *
 *  \details
 *  The type of cycle used in algebraic multigrid solver.
 */
enum class Cycle
{
    Vcycle,
    Wcycle,
    Fcycle
};

/*! \ingroup iterative_solvers
 *  \brief Smoother type used in algebraic multigrid solver.
 *
 *  \details
 *  The type of smoother used in algebraic multigrid solver.
 */
enum class Smoother
{
    Jacobi,
    Gauss_Seidel,
    Symm_Gauss_Seidel,
    SOR,
    SSOR
};

/*! \ingroup iterative_solvers
 *  \brief Algebraic Mulitgrid solver
 *
 *  \details
 *  \p amg_solve solves a linear system using algebraic multigrid
 *
 *  @param[in]
 *  heirarchy   Structure holding the heirarchy of restriction, prolongation,
 *              and coarse grid operators. Must be filled by calling
 *              \ref saamg_setup or \ref rsamg_setup prior to calling
 *              \p amg_solve.
 *  @param[inout]
 *  x           Array of \p m elements containing the solution values of
 *              \f$A\f$ * \f$x\f$ = \f$b\f$.
 *  @param[in]
 *  b           Array of \p m elements containing the righthad side values of
 *              \f$A\f$ * \f$x\f$ = \f$b\f$.
 *  @param[in]
 *  n1          Number pre-smoothing steps
 *  @param[in]
 *  n2          Number post-smoothing steps
 *  @param[in]
 *  tol         Tolerance used to determine stopping iteration.
 *  @param[in]
 *  cycle       Algebaric multigrid cycle type. Can be \ref Vcycle, \ref Wcycle,
 *              \ref Fcycle.
 *  @param[in]
 *  smoother    Smoother type used. Can be \ref Jacobi, Gauss_Seidel, \ref
 *              Symm_Gauss_Seidel, \ref SOR, or \ref SSOR.
 *  @param[in]
 *  control     iteration control struct specifying relative and absolute tolerence 
 *              as well as maximum iterations
 *
 *  \retval number of cycles actually used in the solver. If -1 is returned, the
 *  solver did not converge to a solution with the given input tolerance \p tol.
 *
 *  \par Example
 *  \code{.c}
 *   int m, n, nnz;
 *   std::vector<int> csr_row_ptr;
 *	 std::vector<int> csr_col_ind;
 *	 std::vector<double> csr_val;
 *	 load_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n,
 *   nnz);
 *
 *	 // Solution vector
 *	 std::vector<double> x(m, 0.0);
 *
 *	 // Righthand side vector
 *	 std::vector<double> b(m, 1.0);
 *
 *	 heirarchy hierachy;
 *	 saamg_setup(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), m,
 *   m, nnz, 100, hierachy);
 *
 *	 int cycles = amg_solve(hierachy, x.data(), b.data(), 10, 10, 1e-8,
 *   Cycle::Vcycle, Smoother::Gauss_Seidel);
 *  \endcode
 */
/**@{*/
LINALGLIB_API int amg_solve(const heirarchy &hierarchy, double *x, const double *b, int n1, int n2, Cycle cycle,
              Smoother smoother, iter_control control);
/**@}*/

#endif