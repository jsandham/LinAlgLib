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

#ifndef UAAMG_H
#define UAAMG_H

#include "amg.h"

/*! \file
 *  \brief uaamg.h provides interface for unsmoothed aggregation used in
 * algebraic multigrid
 */

/*! \ingroup linear_solvers
 *  \brief Unsmoothed Aggregation setup
 *
 *  \details
 *  \p uaamg_setup generates the hierarchy of restriction, prolongation, and
 *  coarse grid operators using Unsmoothed Aggregation
 *
 *  @param[in]
 *  csr_row_ptr array of \p m+1 elements that point to the start of every row of
 *              the sparse CSR matrix.
 *  @param[in]
 *  csr_col_ind array of \p nnz elements containing the column indices of the
 *              sparse CSR matrix.
 *  @param[in]
 *  csr_val     array of \p nnz elements containing the values of the sparse
 *              CSR matrix.
 *  @param[in]
 *  m           number of rows in the sparse CSR matrix
 *  @param[in]
 *  n           number of columns in the sparse CSR matrix
 *  @param[in]
 *  nnz         number of non-zeros in the sparse CSR matrix
 *  @param[in]
 *  max_level   maximum number of levels in the hierarchy of coarse grids
 *  @param[out]
 *  heirarchy   structure holding the heirarchy of restriction, prolongation,
 *              and coarse grid operators
 *
 *  \par Example
 *  \code{.c}
 *   int m, n, nnz;
 *   std::vector<int> csr_row_ptr;
 *	std::vector<int> csr_col_ind;
 *	std::vector<double> csr_val;
 *	load_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n,
 *nnz);
 *
 *	// Solution vector
 *	std::vector<double> x(m, 0.0);
 *
 *	// Righthand side vector
 *	std::vector<double> b(m, 1.0);
 *
 *	heirarchy hierachy;
 *	uaamg_setup(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), m,
 *m, nnz, 100, hierachy);
 *
 *	int cycles = amg_solve(hierachy, x.data(), b.data(), 10, 10, 0.00001,
 *Cycle::Vcycle, Smoother::Gauss_Siedel);
 *  \endcode
 */
/**@{*/
void uaamg_setup(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz,
                 int max_level, heirarchy &hierarchy);

#endif