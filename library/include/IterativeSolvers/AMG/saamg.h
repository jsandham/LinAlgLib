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

#ifndef SAAMG_H
#define SAAMG_H

#include "../../linalglib_export.h"
#include "../../csr_matrix.h"

#include "amg.h"

/*! \file
 *  \brief saamg.h provides interface for smoothed aggregation used in algebraic
 * multigrid
 */

/*! \ingroup iterative_solvers
 * \brief Smoothed Aggregation setup for Algebraic Multigrid.
 *
 * \details
 * \p saamg_setup is the core function responsible for generating the multigrid
 * hierarchy required by Algebraic Multigrid (AMG) solvers using the Smoothed
 * Aggregation (SA) method. This involves constructing a series of coarser
 * grid operators, along with the prolongation (interpolation) operators that
 * transfer corrections from coarser to finer levels, and the restriction
 * (projection) operators that transfer residuals from finer to coarser levels.
 * The hierarchy is built based on the input sparse matrix representing the
 * discretized problem on the finest level.
 *
 * **How Smoothed Aggregation Works:**
 *
 * Smoothed Aggregation is an algebraic multigrid method that builds the
 * coarse levels and transfer operators directly from the system matrix, without
 * relying on geometric information. The general process involves the following key steps:
 *
 * 1.  **Aggregation:** The degrees of freedom (rows/columns of the matrix) on the
 * current (finer) level are grouped into disjoint sets called "aggregates."
 * This aggregation is typically done based on the strength of connections
 * between the degrees of freedom, as represented by the entries in the matrix.
 * Strongly connected degrees of freedom are more likely to be grouped
 * together. Various algorithms exist for performing this aggregation, often
 * employing graph-based approaches on the sparsity pattern of the matrix.
 * Each aggregate on level \f$l\f$ will correspond to a single degree of
 * freedom on the next coarser level \f$l+1\f$.
 *
 * 2.  **Tentative Prolongation:** An initial "tentative" prolongation operator
 * (\f$P_{tent}\f$) is constructed based on the aggregation. This operator
 * defines how to interpolate values from the coarse level back to the fine
 * level based on the aggregates. A simple approach is to define a piecewise
 * constant interpolation within each aggregate. If a coarse-level degree
 * of freedom (representing an aggregate) has a value of 1, then all the
 * fine-level degrees of freedom belonging to that aggregate receive a value
 * of 1. Degrees of freedom in other aggregates receive 0. This tentative
 * prolongation is generally not smooth enough.
 *
 * 3.  **Prolongation Smoothing:** The tentative prolongation operator is then
 * "smoothed" using an iterative method, typically a few steps of a
 * point-wise smoother like weighted Jacobi applied to the original system
 * matrix (or a related matrix). This smoothing step aims to improve the
 * interpolation by making it better at representing the "algebraically
 * smooth" error components that standard smoothers like Jacobi or
 * Gauss-Seidel struggle to reduce. The resulting "smoothed" prolongation
 * operator (\f$P\f$) is then used to transfer corrections from the coarse
 * level to the fine level.
 *
 * 4.  **Restriction Operator:** The restriction operator (\f$R\f$) is typically
 * chosen as the transpose (or a scaled transpose) of the prolongation
 * operator (\f$R = P^T\f$). This variational approach ensures that the
 * coarse-level problem is a Galerkin projection of the fine-level problem,
 * leading to good energy properties. The restriction operator transfers the
 * residual from the fine level to the coarse level.
 *
 * 5.  **Coarse-Level Operator:** The system matrix on the coarser level
 * (\f$A_c\f$) is computed using a Galerkin triple product: \f$A_c = R A P\f$,
 * where \f$A\f$ is the system matrix on the finer level. This step ensures
 * that the coarse-level problem accurately represents the residual equation
 * on the coarser grid.
 *
 * These steps are repeated recursively to build a hierarchy of coarser levels
 * until a sufficiently small (and thus easily solvable) coarse-level problem
 * is obtained. The resulting hierarchy (prolongation operators, restriction
 * operators, and coarse-level matrices) is stored in the \ref heirarchy
 * structure and is then used in the multigrid solve phase with appropriate
 * cycling strategies (like V-cycle, W-cycle, F-cycle) and smoothers at each level.
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
 * saamg_setup(csr_row_ptr.data(), csr_col_ind.data(), csr_val.data(), m, m, nnz, 10, hierarchy);
 *
 * int cycles = amg_solve(hierarchy, x.data(), b.data(), 10, 10, 1e-8,
 * Cycle::Vcycle, Smoother::Gauss_Seidel);
 * std::cout << "Number of cycles: " << cycles << std::endl;
 *
 * return 0;
 * }
 * \endcode
 */
LINALGLIB_API void saamg_setup(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz,
    int max_level, heirarchy &hierarchy);

LINALGLIB_API void saamg_setup(const csr_matrix2& A, int max_level, heirarchy &hierarchy);


#endif