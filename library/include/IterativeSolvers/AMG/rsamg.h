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

#ifndef RSAMG_H
#define RSAMG_H

#include "../../linalglib_export.h"
#include "../../csr_matrix.h"

#include "amg.h"

/*! \file
 *  \brief rsamg.h provides interface for classical algebraic multigrid
 */

/*! \ingroup iterative_solvers
 *  \brief Legacy Ruge Steuben algebraic multigrid setup
 */
// LINALGLIB_API void rsamg_setup_legacy(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz,
//                  int max_level, heirarchy &hierarchy);

/*! \ingroup iterative_solvers
 * \brief Ruge-Stüben Algebraic Multigrid setup.
 *
 * \details
 * \p rsamg_setup generates the hierarchy of restriction, prolongation, and
 * coarse grid operators using the Ruge-Stüben Algebraic Multigrid (AMG) method.
 * Ruge-Stüben is a classical AMG approach that relies on the concept of
 * "strong connections" within the system matrix to determine the coarse grid
 * structure and the interpolation operators.
 *
 * The Ruge-Stüben setup typically involves the following steps:
 *
 * 1.  **Strong/Weak Connections:** Determining the strength of connection
 * between pairs of degrees of freedom based on the magnitude of the off-diagonal
 * entries in the matrix. A common criterion is to compare the off-diagonal
 * entry with some fraction of the magnitude of the diagonal entry.
 *
 * 2.  **Independent Set (C/F Splitting):** Partitioning the degrees of freedom
 * into two sets: C (coarse) points and F (fine) points. C points will be
 * represented on the coarser grid, while F points will be interpolated from
 * the C points. This splitting is often done greedily, prioritizing points
 * that strongly influence many other points to be in the C set, while ensuring
 * that the C set is a maximal independent set with respect to strong connections.
 *
 * 3.  **Prolongation (Interpolation) Operator:** Constructing an interpolation
 * operator that defines how to transfer values from the coarse grid (C points)
 * to the fine grid (F points). The interpolation weights are typically based
 * on the strong connections between F points and their neighboring C points.
 *
 * 4.  **Restriction (Projection) Operator:** Defining a restriction operator that
 * transfers residuals from the fine grid to the coarse grid. A common choice
 * is the transpose of the prolongation operator (or a scaled transpose),
 * leading to a Galerkin projection.
 *
 * 5.  **Coarse-Level Operator:** Forming the system matrix on the coarser level
 * using a Galerkin projection: \f$A_c = R A P\f$, where \f$A\f$ is the matrix
 * on the finer level, \f$R\f$ is the restriction operator, and \f$P\f$ is the
 * prolongation operator.
 *
 * These steps are applied recursively until a sufficiently small coarse-level
 * problem is obtained. The resulting hierarchy of operators is stored in the
 * \ref heirarchy structure and can then be used by an AMG solver (e.g.,
 * \ref amg_solve) to efficiently solve the linear system.
 *
 * @param[in] csr_row_ptr
 * Array of \p m+1 elements that point to the start of every row of
 * the input sparse matrix in CSR format.
 * @param[in] csr_col_ind
 * Array of \p nnz elements containing the column indices of the
 * non-zero entries in the input sparse matrix (CSR format).
 * @param[in] csr_val
 * Array of \p nnz elements containing the numerical values of the
 * non-zero entries in the input sparse matrix (CSR format).
 * @param[in] m
 * Number of rows in the input sparse CSR matrix.
 * @param[in] n
 * Number of columns in the input sparse CSR matrix.
 * @param[in] nnz
 * Number of non-zero elements in the input sparse CSR matrix.
 * @param[in] max_level
 * Maximum number of levels to be generated in the multigrid hierarchy.
 * The actual number of levels generated might be less than \p max_level
 * depending on the coarsening process.
 * @param[out] hierarchy
 * Structure of type \ref heirarchy that will be populated with the
 * generated hierarchy of restriction operators, prolongation operators,
 * and coarse-level system matrices.
 *
 * \par Example
 * \code{.cpp}
 * #include <vector>
 * #include <iostream>
 * #include "linalg.h"
 *
 * int main() {
 * int m, n, nnz;
 * std::vector<int> csr_row_ptr;
 * std::vector<int> csr_col_ind;
 * std::vector<double> csr_val;
 * const char* matrix_file = "my_matrix.mtx";
 * load_mtx_file(matrix_file, csr_row_ptr, csr_col_ind, csr_val, m, n, nnz);
 *
 * heirarchy hierarchy;
 * rsamg_setup(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), m, m, nnz, 10, hierarchy);
 *
 * std::cout << "Ruge-Stüben AMG hierarchy setup complete." << std::endl;
 *
 * return 0;
 * }
 * \endcode
 */
LINALGLIB_API void rsamg_setup(const csr_matrix& A, int max_level, heirarchy &hierarchy);


#endif