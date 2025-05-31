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
#include "../../csr_matrix.h"

#include "amg.h"

/*! \file
 *  \brief uaamg.h provides interface for unsmoothed aggregation used in
 * algebraic multigrid
 */

/*! \brief Sets up the hierarchy for an Unsmoothed Aggregation Algebraic Multigrid (UAAMG) solver.
 *
 * \details
 * This function constructs the multigrid hierarchy required for the Unsmoothed Aggregation
 * Algebraic Multigrid (UAAMG) method. UAAMG is a powerful and robust type of algebraic
 * multigrid (AMG) method, particularly well-suited for solving large sparse linear systems
 * that arise from the discretization of partial differential equations, especially
 * those with anisotropic or discontinuous coefficients where geometric multigrid is difficult to apply.
 *
 * \section uaamg_method Unsmoothed Aggregation Method
 *
 * Unsmoothed Aggregation AMG builds a hierarchy of coarser matrices by grouping (aggregating)
 * fine-grid unknowns into larger "aggregates" that form the unknowns on the coarser levels.
 * Unlike classical AMG methods which construct interpolation operators based on a "smoothed"
 * version of the fine-grid unknowns (e.g., Jacobi or Gauss-Seidel smoothing of an initial guess),
 * UAAMG constructs interpolation (prolongation) operators directly from the aggregates,
 * often using simple piecewise constant interpolation.
 *
 * The general setup procedure for UAAMG involves the following steps for each level,
 * starting from the finest (level 0) matrix \f$A_0 = A\f$:
 *
 * 1.  \b Aggregation: Partition the fine-grid nodes into disjoint sets called aggregates.
 * This step is crucial and influences the efficiency and robustness of the method.
 * Algorithms like Ruge-Stuben (RS) coarsening, or simpler greedy aggregation algorithms
 * based on graph properties of the matrix, can be used. Each aggregate forms a single
 * coarse-grid degree of freedom.
 *
 * 2.  \b Prolongation \b Operator \b Construction: Construct the prolongation (interpolation)
 * operator \f$P\f$. For unsmoothed aggregation, \f$P\f$ is typically a matrix where each column
 * corresponds to a coarse-grid aggregate, and its entries are non-zero only for fine-grid
 * nodes belonging to that aggregate. A common choice is piecewise constant interpolation,
 * where \f$P_{ij} = 1\f$ if fine-grid node \f$i\f$ belongs to aggregate \f$j\f$, and \f$0\f$ otherwise.
 * This results in a boolean (0-1) prolongation operator.
 *
 * 3.  \b Restriction \b Operator \b Construction: The restriction operator \f$R\f$ is often chosen
 * as the transpose of the prolongation operator, i.e., \f$R = P^T\f$. This ensures a Galerkin
 * coarse-grid approximation.
 *
 * 4.  \b Coarse-Grid \b Operator \b Construction: The coarse-grid matrix \f$A_c\f$ (or \f$A_{level+1}\f$)
 * is formed using the Galerkin product:
 * \f$ A_{level+1} = R A_{level} P = P^T A_{level} P \f$
 *
 * These steps are repeated recursively until the coarse-grid problem is small enough to be
 * solved directly (e.g., by direct factorization) or a maximum number of levels is reached.
 *
 * \param A The fine-grid (finest level) sparse matrix in CSR format. This is the matrix
 * for which the multigrid hierarchy is to be constructed.
 * \param max_level The maximum number of levels to build in the hierarchy. The hierarchy
 * will contain `max_level + 1` levels (level 0 to `max_level`).
 * The coarsest level will be `max_level`.
 * \param hierarchy A reference to a `hierarchy` object that will be populated with the
 * constructed multigrid levels (matrices, prolongation/restriction operators).
 * This object should be capable of storing `max_level + 1` levels of data.
 *
 * \section uaamg_example Example Usage
 * Below is a simplified example demonstrating how to use the `uaamg_setup` function.
 * This assumes `csr_matrix`, `vector`, `hierarchy`, and `iter_control` classes are
 * properly defined and functional.
 *
 * \code
 * #include "linalglib.h"
 * #include <iostream>
 * #include <vector>
 *
 * int main() {
 * // 1. Create a sample sparse matrix (e.g., from a 1D Poisson problem for simplicity)
 * // A 5x5 matrix for demonstration:
 * // [ 2 -1  0  0  0 ]
 * // [-1  2 -1  0  0 ]
 * // [ 0 -1  2 -1  0 ]
 * // [ 0  0 -1  2 -1 ]
 * // [ 0  0  0 -1  2 ]
 *
 * int N = 5; // Size of the matrix
 * std::vector<int> row_ptr(N + 1);
 * std::vector<int> col_ind;
 * std::vector<double> val;
 *
 * row_ptr[0] = 0;
 * int nnz_count = 0;
 * for (int i = 0; i < N; ++i) {
 * // Diagonal element
 * col_ind.push_back(i);
 * val.push_back(2.0);
 * nnz_count++;
 *
 * // Off-diagonal (left)
 * if (i > 0) {
 * col_ind.push_back(i - 1);
 * val.push_back(-1.0);
 * nnz_count++;
 * }
 * // Off-diagonal (right)
 * if (i < N - 1) {
 * col_ind.push_back(i + 1);
 * val.push_back(-1.0);
 * nnz_count++;
 * }
 * row_ptr[i+1] = nnz_count;
 * }
 *
 * // Sort elements by column index within each row to ensure CSR format
 * // (This is a simplified example; a real CSR builder would handle this)
 * for (int i = 0; i < N; ++i) {
 * std::vector<std::pair<int, double>> row_elements;
 * for (int k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
 * row_elements.push_back({col_ind[k], val[k]});
 * }
 * std::sort(row_elements.begin(), row_elements.end());
 * for (int k = 0; k < row_elements.size(); ++k) {
 * col_ind[row_ptr[i] + k] = row_elements[k].first;
 * val[row_ptr[i] + k] = row_elements[k].second;
 * }
 * }
 *
 * csr_matrix A(row_ptr, col_ind, val, N, N, nnz_count);
 *
 * // 2. Define the maximum number of multigrid levels
 * int max_levels = 2; // Including level 0, this means 3 levels (0, 1, 2)
 *
 * // 3. Create a hierarchy object
 * hierarchy mg_hierarchy;
 *
 * // 4. Call the UAAMG setup function
 * std::cout << "Setting up UAAMG hierarchy with max_level = " << max_levels << "..." << std::endl;
 * uaamg_setup(A, max_levels, mg_hierarchy);
 * std::cout << "UAAMG hierarchy setup complete." << std::endl;
 *
 * // You can now inspect the hierarchy, e.g., print sizes of matrices at each level
 * std::cout << "\nHierarchy Levels:" << std::endl;
 * for (int level = 0; level <= max_levels; ++level) {
 * // Assuming hierarchy has a method to get matrix at a level
 * // And a method to get prolongation/restriction operators
 * // This part is illustrative as hierarchy class structure is not provided
 * const csr_matrix& A_level = mg_hierarchy.get_matrix(level);
 * std::cout << "Level " << level << ": Matrix size = " << A_level.get_num_rows() << "x" << A_level.get_num_cols() << std::endl;
 * }
 *
 * // Further usage would involve passing this hierarchy to an AMG solver
 * // For instance:
 * // amg_solver solver;
 * // solver.solve(mg_hierarchy, x, b, control);
 *
 * return 0;
 * }
 * \endcode
 */
void uaamg_setup(const csr_matrix& A, int max_level, hierarchy &hierarchy);

#endif