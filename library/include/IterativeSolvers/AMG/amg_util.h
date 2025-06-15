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

#include "../../linalg_export.h"
#include "../../csr_matrix.h"

/*! \file
 *  \brief amg_util.h provides interface for algebraic multigrid solver
 */

namespace linalg
{
/*! \ingroup iterative_solvers
* \brief Compute the Galerkin triple product for algebraic multigrid.
*
* \param R The restriction operator (CSR matrix).
* \param A The system matrix on the finer level (CSR matrix).
* \param P The prolongation operator (CSR matrix).
* \param A_coarse The output CSR matrix that will store the resulting coarse-level matrix (\f$A_c = R A P\f$).
*
* \details
* This function computes the Galerkin triple product, which is a fundamental operation
* in algebraic multigrid (AMG) methods for constructing the coarse-level system matrix.
* The coarse-level matrix \f$A_c\f$ is computed as the product of the restriction operator (\f$R\f$),
* the fine-level system matrix (\f$A\f$), and the prolongation operator (\f$P\f$). This projection
* ensures that the coarse-level problem accurately reflects the behavior of the fine-level
* problem on a coarser grid. The dimensions of the matrices must be compatible for
* the matrix multiplications to be valid. Specifically, if \f$R\f$ is \f$n_c \times n_f\f$,
* \f$A\f$ is \f$n_f \times n_f\f$, and \f$P\f$ is \f$n_f \times n_c\f$, then \f$A_c\f$ will be \f$n_c \times n_c\f$,
* where \f$n_f\f$ is the number of degrees of freedom on the fine level and \f$n_c\f$ is the
* number of degrees of freedom on the coarse level.
*/
LINALGLIB_API void galarkin_triple_product(const csr_matrix &R, const csr_matrix &A, const csr_matrix &P, csr_matrix &A_coarse);
}

#endif
