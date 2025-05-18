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

#include "../../linalglib_export.h"

#include "amg.h"

/*! \file
 *  \brief uaamg.h provides interface for unsmoothed aggregation used in
 * algebraic multigrid
 */

/*! \ingroup iterative_solvers
 * \brief Unsmoothed Aggregation setup for Algebraic Multigrid.
 *
 * \details
 * \p uaamg_setup generates the multigrid hierarchy required by Algebraic
 * Multigrid (AMG) solvers using the Unsmoothed Aggregation (UA) method.
 * Similar to Smoothed Aggregation, UA constructs coarse grid operators and
 * transfer operators (prolongation and restriction) directly from the system
 * matrix without relying on geometric information. However, unlike SA,
 * Unsmoothed Aggregation uses a simpler approach for defining the prolongation
 * operator, typically without an explicit smoothing step.
 *
 * **How Unsmoothed Aggregation Works:**
 *
 * The Unsmoothed Aggregation method generally involves the following steps:
 *
 * 1.  **Aggregation:** The degrees of freedom (rows/columns of the matrix) on the
 * current (finer) level are partitioned into disjoint aggregates. This
 * aggregation is usually based on the sparsity pattern and the strength of
 * connections in the matrix, with strongly connected degrees of freedom
 * being grouped together. Various graph-based algorithms can be employed
 * for this purpose. Each aggregate on level \f$l\f$ will correspond to a
 * single degree of freedom on the next coarser level \f$l+1\f$.
 *
 * 2.  **Prolongation (Tentative):** A prolongation operator (\f$P\f$) is directly
 * defined based on the aggregation. A common approach is to use a piecewise
 * constant interpolation. For each aggregate, a basis vector is created that
 * has a value of 1 for all the fine-level degrees of freedom belonging to that
 * aggregate and 0 elsewhere. These basis vectors form the columns of the
 * prolongation matrix. This operator interpolates values from the coarse
 * level (representing the aggregates) back to the fine level by assigning
 * the coarse-level value uniformly to all fine-level nodes within the
 * corresponding aggregate. This prolongation is "unsmoothed" as no explicit
 * smoothing operation is applied to improve its properties.
 *
 * 3.  **Restriction Operator:** The restriction operator (\f$R\f$) is typically
 * chosen as the transpose (or a scaled transpose) of the prolongation
 * operator (\f$R = P^T\f$). This choice is often made for simplicity and to
 * ensure a Galerkin framework. The restriction operator transfers the
 * residual from the fine level to the coarse level by summing or averaging
 * the residuals corresponding to the degrees of freedom within each aggregate.
 *
 * 4.  **Coarse-Level Operator:** The system matrix on the coarser level
 * (\f$A_c\f$) is computed using the Galerkin triple product: \f$A_c = R A P\f$,
 * where \f$A\f$ is the system matrix on the finer level. This projection
 * yields a coarse-level problem that approximates the fine-level problem
 * in a Galerkin sense.
 *
 * These steps are repeated recursively to build a hierarchy of coarser levels
 * until a sufficiently small coarse-level problem is obtained. The resulting
 * hierarchy (prolongation operators, restriction operators, and coarse-level
 * matrices) is stored in the \ref heirarchy structure and is used in the
 * multigrid solve phase with appropriate cycling strategies and smoothers.
 * Unsmoothed Aggregation is often simpler to implement than Smoothed
 * Aggregation but might exhibit slower convergence for some problems,
 * especially those where the algebraically smooth errors are not well
 * represented by piecewise constant interpolation.
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
 * // Solution vector
 * std::vector<double> x(m, 0.0);
 *
 * // Righthand side vector
 * std::vector<double> b(m, 1.0);
 *
 * heirarchy hierarchy;
 * uaamg_setup(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), m, m, nnz, 10, hierarchy);
 *
 * int cycles = amg_solve(hierarchy, x.data(), b.data(), 10, 10, 1e-8,
 * Cycle::Vcycle, Smoother::Gauss_Seidel);
 * std::cout << "Number of cycles: " << cycles << std::endl;
 *
 * return 0;
 * }
 * \endcode
 */
LINALGLIB_API void uaamg_setup(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz,
int max_level, heirarchy &hierarchy);

#endif