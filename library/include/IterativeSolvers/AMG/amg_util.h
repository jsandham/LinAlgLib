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
 * \brief Data structure for storing sparse CSR matrices
 *
 * \details
 * This struct represents a sparse matrix stored in the Compressed Sparse Row (CSR) format.
 * The CSR format is efficient for storing and operating on sparse matrices, as it only
 * stores the non-zero elements along with information about their row and column indices.
 */
struct csr_matrix
{
    /**
    * @brief Number of rows in the CSR matrix.
    */
    int m;

    /**
    * @brief Number of columns in the CSR matrix.
    */
    int n;

    /**
    * @brief Number of non-zeros in the CSR matrix.
    */
    int nnz;

    /**
    * @brief Row pointer array of CSR format.
    *
    * `csr_row_ptr` is an array of length `m + 1`. `csr_row_ptr[i]` stores the index
    * in the `csr_col_ind` and `csr_val` arrays where the non-zero elements of the
    * `i`-th row start. The last element, `csr_row_ptr[m]`, stores the total number
    * of non-zero elements (`nnz`).
    */
    std::vector<int> csr_row_ptr;

    /**
    * @brief Column indices array of CSR format.
    *
    * `csr_col_ind` is an array of length `nnz` that stores the column index of each
    * non-zero element. The column indices for the non-zero elements in row `i` are
    * stored in `csr_col_ind[csr_row_ptr[i] : csr_row_ptr[i+1] - 1]`.
    */
    std::vector<int> csr_col_ind;

    /**
    * @brief Values array of CSR format.
    *
    * `csr_val` is an array of length `nnz` that stores the numerical value of each
    * non-zero element. The values for the non-zero elements in row `i` are stored
    * in `csr_val[csr_row_ptr[i] : csr_row_ptr[i+1] - 1]`.
    */
    std::vector<double> csr_val;
};

/*! \ingroup iterative_solvers
* \brief Transpose a CSR matrix.
*
* \param prolongation The input CSR matrix to be transposed. Typically represents a prolongation operator.
* \param restriction The output CSR matrix that will store the transpose of the input matrix. Typically represents a restriction operator.
*
* \details
* This function computes the transpose of a given CSR matrix. If the input matrix
* represents a prolongation operator (mapping from a coarser to a finer grid), its
* transpose typically serves as a restriction operator (mapping from a finer to a
* coarser grid) in multigrid methods. The dimensions of the output `restriction`
* matrix will be swapped compared to the input `prolongation` matrix (i.e., if
* `prolongation` is \f$m \times n\f$, then `restriction` will be \f$n \times m\f$).
*/
LINALGLIB_API void transpose(const csr_matrix &prolongation, csr_matrix &restriction);

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

#endif
