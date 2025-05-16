//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025 James Sandham
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

#ifndef AMG_UTIL_H
#define AMG_UTIL_H

#include <vector>

#include "../../linalglib_export.h"

/*! \file
 *  \brief amg_util.h provides interface for algebraic multigrid solver
 */

/*! \ingroup iterative_solvers
 *  \brief Data structure for storing sparse CSR matrices
 */
struct csr_matrix
{
    int m;                        /**< Number of rows in the CSR matrix. */
    int n;                        /**< Number of columns in the CSR matrix. */
    int nnz;                      /**< Number of non-zeros in the CSR matrix. */
    std::vector<int> csr_row_ptr; /**< Row pointer array of CSR format. */
    std::vector<int> csr_col_ind; /**< Column indices array of CSR format. */
    std::vector<double> csr_val;  /**< Values array of CSR format. */
};

/*! \ingroup iterative_solvers
 *  \brief Transpose CSR matrix
 */
LINALGLIB_API void transpose(const csr_matrix &prolongation, csr_matrix &restriction);

/*! \ingroup iterative_solvers
 *  \brief Compute Galarkin triple product
 */
LINALGLIB_API void galarkin_triple_product(const csr_matrix &R, const csr_matrix &A, const csr_matrix &P, csr_matrix &A_coarse);

#endif
