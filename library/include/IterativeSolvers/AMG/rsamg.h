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

#include "../../linalg_export.h"
#include "../../csr_matrix.h"

#include "amg.h"

/*! \file
 *  \brief rsamg.h provides interface for classical algebraic multigrid
 */

namespace linalg
{
/*! \ingroup iterative_solvers
 *  \brief Legacy Ruge Steuben algebraic multigrid setup
 */
// LINALGLIB_API void rsamg_setup_legacy(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, int m, int n, int nnz,
//                  int max_level, hierarchy &hierarchy);

/*! \brief Sets up the hierarchy for a Ruge-Stuben Algebraic Multigrid (RSAMG) solver.
 *
 * \details
 * This function constructs the multigrid hierarchy required for the Ruge-Stuben Algebraic Multigrid (RSAMG) method.
 * RSAMG is one of the foundational and most widely used classical AMG methods. It is particularly effective
 * for solving linear systems arising from elliptic partial differential equations (PDEs), especially
 * when the matrix is symmetric positive definite (SPD) or M-matrix.
 * The core idea of RSAMG lies in partitioning the unknowns (nodes) into "C-nodes" (coarse-grid nodes)
 * and "F-nodes" (fine-grid nodes) based on the strength of their connections.
 *
 * \section rsamg_method Ruge-Stuben Method
 *
 * The Ruge-Stuben (RS) approach for building the multigrid hierarchy is based on measuring the
 * "strength" of connections between degrees of freedom. A connection \f$A_{ij}\f$ is considered strong
 * if its magnitude is large relative to other off-diagonal entries in row \f$i\f$.
 *
 * The setup procedure for RSAMG for each level, starting from the finest (level 0) matrix \f$A_0 = A\f$:
 *
 * 1.  \b Strength \b of \b Connection: Define a measure for the strength of connection between
 * node \f$i\f$ and node \f$j\f$. A common definition is that \f$j\f$ strongly depends on \f$i\f$ if:
 * \f$ |A_{ij}| \ge \alpha \cdot \max_{k \neq i} |A_{ik}| \f$
 * for some parameter \f$\alpha \in [0, 1)\f$ (e.g., \f$\alpha = 0.25\f$ or \f$0.5\f$).
 * Let \f$S_i\f$ be the set of nodes \f$j\f$ that strongly influence node \f$i\f$.
 *
 * 2.  \b C/F \b Point \b Selection (Coarsening): Partition the set of all nodes \f$\Omega\f$ into
 * a coarse set \f$C\f$ (C-nodes) and a fine set \f$F\f$ (F-nodes). This is typically done greedily:
 * a. Initialize \f$C = \emptyset\f$, \f$F = \emptyset\f$, and all nodes are "unassigned".
 * b. Iteratively select an unassigned node that is strongly influenced by the most unassigned nodes.
 * This node becomes a C-node. Add it to \f$C\f$. All unassigned neighbors strongly influenced by this
 * new C-node are designated as F-nodes and added to \f$F\f$.
 * c. Repeat until all nodes are assigned.
 * This strategy aims to ensure that F-nodes are strongly connected to at least one C-node.
 *
 * 3.  \b Prolongation \b Operator \b Construction: Construct the prolongation (interpolation) operator \f$P\f$.
 * The interpolation operator maps coarse-grid vectors to fine-grid vectors. For RSAMG, \f$P\f$ is constructed
 * such that F-nodes are interpolated from their C-node neighbors.
 * For an F-node \f$i\f$, its value is interpolated from a weighted sum of its C-node neighbors:
 * \f$ x_i = \sum_{j \in C \cap S_i} P_{ij} x_j \f$
 * The weights \f$P_{ij}\f$ are derived from the entries of the matrix \f$A\f$. A common formula for \f$P_{ij}\f$ for \f$i \in F, j \in C\f$:
 * \f$ P_{ij} = \frac{-A_{ij}}{A_{ii} - \sum_{k \in F \cap S_i} A_{ik}} \f$
 * (This is a simplified representation; the full formula can be more complex, involving sums over strong F-F connections as well).
 * For C-nodes, \f$P_{ii} = 1\f$ if \f$i \in C\f$, and \f$P_{ij} = 0\f$ for \f$i \in C, i \neq j\f$.
 *
 * 4.  \b Restriction \b Operator \b Construction: The restriction operator \f$R\f$ is often chosen
 * as the transpose of the prolongation operator, i.e., \f$R = P^T\f$. This ensures a Galerkin
 * coarse-grid approximation and preserves symmetry if \f$A\f$ is symmetric.
 *
 * 5.  \b Coarse-Grid \b Operator \b Construction: The coarse-grid matrix \f$A_c\f$ (or \f$A_{level+1}\f$)
 * is formed using the Galerkin product:
 * \f$ A_{level+1} = R A_{level} P = P^T A_{level} P \f$
 *
 * These steps are repeated recursively until the coarse-grid problem is small enough to be
 * solved directly (e.g., by direct factorization) or a maximum number of levels is reached.
 *
 * \param A The fine-grid (finest level) sparse matrix in CSR format. This is the matrix
 * for which the multigrid hierarchy is to be constructed. It is generally assumed to be
 * a square matrix.
 * \param max_level The maximum number of levels to build in the hierarchy. The hierarchy
 * will contain `max_level + 1` levels (level 0 to `max_level`).
 * The coarsest level will be `max_level`.
 * \param hierarchy A reference to a `hierarchy` object that will be populated with the
 * constructed multigrid levels (matrices, prolongation/restriction operators).
 * This object should be capable of storing `max_level + 1` levels of data.
 *
 * \section rsamg_example Example Usage
 * Below is a simplified example demonstrating how to use the `rsamg_setup` function.
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
 * // 4. Call the RSAMG setup function
 * std::cout << "Setting up RSAMG hierarchy with max_level = " << max_levels << "..." << std::endl;
 * rsamg_setup(A, max_levels, mg_hierarchy);
 * std::cout << "RSAMG hierarchy setup complete." << std::endl;
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
LINALGLIB_API void rsamg_setup(const csr_matrix& A, int max_level, hierarchy &hierarchy);
}

#endif