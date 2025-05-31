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

/*! \brief Sets up the hierarchy for a Smoothed Aggregation Algebraic Multigrid (SAAMG) solver.
 *
 * \details
 * This function constructs the multigrid hierarchy required for the Smoothed Aggregation
 * Algebraic Multigrid (SAAMG) method. SAAMG is an advanced type of algebraic multigrid (AMG)
 * method, building upon unsmoothed aggregation by incorporating a "smoothing" step into
 * the prolongation operator. This smoothing typically improves the robustness and convergence
 * of the AMG solver, especially for more complex problems or when simple aggregation might
 * lead to oscillatory error components that are not well-interpolated.
 *
 * \section saamag_method Smoothed Aggregation Method
 *
 * Smoothed Aggregation AMG follows a similar philosophy to Unsmoothed Aggregation, in that
 * it groups fine-grid unknowns into aggregates to form coarse-grid degrees of freedom.
 * However, SAAMG introduces an additional "smoothing" step in the construction of the
 * prolongation (interpolation) operator.
 *
 * The general setup procedure for SAAMG for each level, starting from the finest (level 0) matrix \f$A_0 = \text{mat_A}\f$:
 *
 * 1.  \b Aggregation: Partition the fine-grid nodes into disjoint sets called aggregates.
 * This step is identical to that in UAAMG. Algorithms like Ruge-Stuben (RS) coarsening,
 * or simpler greedy aggregation algorithms based on graph properties of the matrix, can be used.
 * Each aggregate forms a single coarse-grid degree of freedom. This step yields an initial
 * unsmoothed prolongation operator \f$P_0\f$ (e.g., piecewise constant interpolation).
 *
 * 2.  \b Smoothing \b of \b Prolongation \b Operator: The key difference from UAAMG.
 * The initial prolongation operator \f$P_0\f$ is "smoothed" to obtain the final prolongation operator \f$P\f$.
 * This smoothing step often involves applying a few iterations of a simple stationary
 * relaxation method (like weighted Jacobi or a polynomial filter) to the columns of \f$P_0\f$.
 * A common smoothing formula is:
 * \f$ P = (I - \omega D^{-1} A) P_0 \f$
 * where \f$D\f$ is the diagonal of \f$A\f$, \f$I\f$ is the identity matrix, and \f$\omega\f$ is a relaxation
 * parameter (similar to SOR or Jacobi damping). This step helps to ensure that highly
 * oscillatory error components (which are poorly represented by piecewise constant interpolation)
 * are effectively interpolated to the coarse grid.
 *
 * 3.  \b Restriction \b Operator \b Construction: The restriction operator \f$R\f$ is chosen
 * as the transpose of the smoothed prolongation operator, i.e., \f$R = P^T\f$. This ensures a Galerkin
 * coarse-grid approximation.
 *
 * 4.  \b Coarse-Grid \b Operator \b Construction: The coarse-grid matrix \f$A_c\f$ (or \f$A_{level+1}\f$)
 * is formed using the Galerkin product:
 * \f$ A_{level+1} = R A_{level} P = P^T A_{level} P \f$
 *
 * These steps are repeated recursively until the coarse-grid problem is small enough to be
 * solved directly or a maximum number of levels is reached.
 *
 * \param mat_A The fine-grid (finest level) sparse matrix in CSR format. This is the matrix
 * for which the multigrid hierarchy is to be constructed.
 * \param max_level The maximum number of levels to build in the hierarchy. The hierarchy
 * will contain `max_level + 1` levels (level 0 to `max_level`).
 * The coarsest level will be `max_level`.
 * \param hierarchy A reference to a `hierarchy` object that will be populated with the
 * constructed multigrid levels (matrices, prolongation/restriction operators).
 * This object should be capable of storing `max_level + 1` levels of data.
 *
 * \section saamag_example Example Usage
 * Below is a simplified example demonstrating how to use the `saamg_setup` function.
 * This assumes `csr_matrix`, `vector`, `hierarchy`, and `iter_control` classes are
 * properly defined and functional.
 *
 * \code
 * #include "linalglib.h"
 * #include <iostream>
 * #include <vector>
 * #include <algorithm> // For std::sort
 * #include <utility>   // For std::pair
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
 * // 4. Call the SAAMG setup function
 * std::cout << "Setting up SAAMG hierarchy with max_level = " << max_levels << "..." << std::endl;
 * saamg_setup(A, max_levels, mg_hierarchy);
 * std::cout << "SAAMG hierarchy setup complete." << std::endl;
 *
 * // You can now inspect the hierarchy, e.g., print sizes of matrices at each level
 * std::cout << "\nHierarchy Levels:" << std::endl;
 * for (int level = 0; level <= max_levels; ++level) {
 * // Assuming hierarchy has a method to get matrix at a level
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
LINALGLIB_API void saamg_setup(const csr_matrix& mat_A, int max_level, hierarchy &hierarchy);


#endif