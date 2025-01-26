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

#ifndef __POWER_ITERATION_H__
#define __POWER_ITERATION_H__

#include "../linalglib_export.h"

/*! \ingroup linear_solvers
 *  \brief Power iteration to find largest eigenvalue/eigenvector pair
 *
 *  \details
 *  \p power_iteration finds the largest eigenvalue/eigenvector pair
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
 *  eigenVec    array of \p n elements containing the eigenvector
 *  @param[in]
 *  tol         stopping tolerance
 *  @param[in]
 *  n           size of the sparse CSR matrix
 *  @param[in]
 *  max_iter    maximum iterations allowed
 *
 *  \retval eigenvalue
 */
/**@{*/
double power_iteration(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, double* eigenVec, const double tol, const int n, const int max_iter);
/**@}*/

#endif
