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


#ifndef AMG_H
#define AMG_H

#include <vector>

/*! \file
*  \brief amg.h provides interface for algebraic multigrid solver
*/
struct csr_matrix
{
	int m; /**< Number of rows in the CSR matrix. */
	int n; /**< Number of columns in the CSR matrix. */
	int nnz; /**< Number of non-zeros in the CSR matrix. */
	std::vector<int> csr_row_ptr; /**< Row pointer array of CSR format. */
	std::vector<int> csr_col_ind; /**< Column indices array of CSR format. */
	std::vector<double> csr_val; /**< Values array of CSR format. */
};

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

/*! \ingroup linear_solvers
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

/*! \ingroup linear_solvers
 *  \brief Smoother type used in algebraic multigrid solver.
 *
 *  \details
 *  The type of smoother used in algebraic multigrid solver.
 */
enum class Smoother
{
	Jacobi,
	Gauss_Siedel,
	Symm_Gauss_Siedel,
	SOR,
	SSOR
};

/*! \ingroup linear_solvers
*  \brief Algebraic Mulitgrid solver
*
*  \details
*  \p amg_solve solves a linear system using algebraic multigrid
*
*  @param[in]
*  heirarchy   Structure holding the heirarchy of restriction, prolongation, and coarse grid operators. Must be
*              filled by calling \ref saamg_setup or \ref rsamg_setup prior to calling \p amg_solve. 
*  @param[inout]
*  x           Array of \p m elements containing the solution values of A*x=b
*  @param[in]
*  b           Array of \p m elements containing the righthad side values of A*x=b.
*  @param[in]
*  n1          Number pre-smoothing steps
*  @param[in]
*  n2          Number post-smoothing steps
*  @param[in]
*  tol         Tolerance used to determine stopping iteration.
*  @param[in]
*  cycle       Algebaric multigrid cycle type. Can be \ref Vcycle, \ref Wcycle, \ref Fcycle.
*  @param[in]
*  smoother    Smoother type used. Can be \ref Jacobi, Gauss_Siedel, \ref Symm_Gauss_Siedel, \ref SOR, or \ref SSOR.
*
*  \retval number of cycles actually used in the solver. If -1 is returned, the solver did not converge to a solution
*  with the given input tolerance \p tol.
*
*  \par Example
*  \code{.c}
*   int m, n, nnz;
*   std::vector<int> csr_row_ptr;
*	std::vector<int> csr_col_ind;
*	std::vector<double> csr_val;
*	load_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);
*
*	// Solution vector
*	std::vector<double> x(m, 0.0);
*
*	// Righthand side vector
*	std::vector<double> b(m, 1.0);
*
*	heirarchy hierachy;
*	saamg_setup(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), m, m, nnz, 100, hierachy);
*
*	int cycles = amg_solve(hierachy, x.data(), b.data(), 10, 10, 0.00001, Cycle::Vcycle, Smoother::Gauss_Siedel);
*  \endcode
*/
/**@{*/
int amg_solve(const heirarchy& hierarchy, double* x, const double* b, int n1, int n2, double tol, Cycle cycle, Smoother smoother);

#endif