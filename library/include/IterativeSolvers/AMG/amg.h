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

#ifndef AMG_H
#define AMG_H

#include <vector>

#include "amg_util.h"

#include "../../linalg_export.h"
#include "../../vector.h"
#include "../../csr_matrix.h"

#include "../iter_control.h"

/*! \file
 *  \brief amg.h provides interface for algebraic multigrid solver
 */

namespace linalg
{
/*! \ingroup iterative_solvers
 * \brief Data structure for storing the matrix hierarchy used in algebraic multigrid solver
 *
 * \details
 * This struct holds the essential components that define the multigrid hierarchy
 * used in an algebraic multigrid (AMG) solver. It includes the prolongation
 * operators (for interpolating corrections from coarser to finer levels),
 * restriction operators (for projecting residuals from finer to coarser levels),
 * the sequence of coarse-level matrices, and the total number of levels in the hierarchy.
 */
struct hierarchy
{
    /**
    * @brief Prolongation matrices.
    *
    * These matrices are used to interpolate the correction computed on a coarser
    * level back to the next finer level. The vector stores the prolongation
    * operators between consecutive levels, where `prolongations[i]` maps from
    * level `i+1` to level `i`.
    */
    std::vector<csr_matrix> prolongations;

    /**
    * @brief Restriction matrices.
    *
    * These matrices are used to transfer the residual from a finer level to the
    * next coarser level. The vector stores the restriction operators between
    * consecutive levels, where `restrictions[i]` maps from level `i` to level `i+1`.
    * Typically, the restriction operator is related to the transpose of the
    * prolongation operator.
    */
    std::vector<csr_matrix> restrictions;

    /**
    * @brief Coarse level matrices.
    *
    * These matrices represent the discretized problem on the coarser levels of
    * the multigrid hierarchy. `A_cs[i]` is the system matrix at level `i` of the
    * hierarchy, with `A_cs[0]` being the original system matrix on the finest level.
    * The size of these matrices decreases as the level number increases.
    */
    std::vector<csr_matrix> A_cs;

    /**
    * @brief Number of levels in the hierarchy.
    *
    * This integer indicates the total number of levels in the multigrid hierarchy,
    * including the finest level and all the coarser levels. A hierarchy with
    * `total_levels = 1` would represent a direct solver without any multigrid
    * coarsening.
    */
    int total_levels;
};

/*! \ingroup iterative_solvers
 * \brief Cycle type used in algebraic multigrid solver
 *
 * \details
 * The type of cycle used in algebraic multigrid solver. Algebraic multigrid (AMG)
 * methods employ different cycling strategies to efficiently reduce error components
 * at various levels of the multigrid hierarchy. This enumeration defines the
 * commonly used cycle types.
 */
enum class Cycle
{
    /*! \brief V-cycle.
    *
    * The V-cycle proceeds by recursively applying smoothing on the current level,
    * then restricting the residual to the coarser level, solving the coarse-level
    * problem (recursively or directly), and finally interpolating the correction
    * back to the finer level followed by another smoothing step. The process resembles
    * the shape of the letter "V".
    */
    Vcycle,

    /*! \brief W-cycle.
    *
    * The W-cycle differs from the V-cycle by performing more than one recursive
    * call to the coarser level. Typically, two or more V-cycles are performed on
    * the coarser level before interpolating back to the finer level. This strategy
    * can be more effective at reducing low-frequency errors but is also more computationally
    * expensive. The process resembles the shape of the letter "W".
    */
    Wcycle,

    /*! \brief F-cycle (or Full Multigrid cycle).
    *
    * The F-cycle starts by recursively solving the problem on the coarsest level.
    * Then, it interpolates the solution to the next finer level and performs one or
    * more V-cycles. This process is repeated until the finest level is reached. The
    * F-cycle aims to provide a good initial guess for the solution on the finest level
    * and often leads to faster convergence overall. The process resembles the shape
    * of the letter "F" if visualized across multiple levels.
    */
    Fcycle
};

/*! \ingroup iterative_solvers
 * \brief Smoother type used in algebraic multigrid solver.
 *
 * \details
 * The type of smoother used in the algebraic multigrid (AMG) solver. Smoothers
 * are iterative methods applied at each level of the multigrid hierarchy to
 * reduce high-frequency errors in the approximate solution. The choice of
 * smoother can significantly impact the convergence rate and efficiency of the
 * AMG method.
 */
enum class Smoother
{
    /*! \brief Jacobi smoother.
    *
    * The Jacobi method is a simple iterative method where each unknown is updated
    * based on the values of all other unknowns from the previous iteration. It is
    * easy to parallelize but often converges slowly.
    */
    Jacobi,

    /*! \brief Gauss-Seidel smoother.
    *
    * The Gauss-Seidel method is similar to Jacobi, but it updates each unknown
    * using the most recently computed values of other unknowns within the same
    * iteration. This sequential dependency can lead to faster convergence than Jacobi
    * in many cases.
    */
    Gauss_Seidel,

    /*! \brief Symmetric Gauss-Seidel (SGS) smoother.
    *
    * The Symmetric Gauss-Seidel method consists of performing a standard
    * Gauss-Seidel sweep followed by a backward Gauss-Seidel sweep. This symmetric
    * application can improve the smoothing properties, particularly for symmetric
    * positive definite systems.
    */
    Symm_Gauss_Seidel,

    /*! \brief Successive Over-Relaxation (SOR) smoother.
    *
    * The Successive Over-Relaxation method is an extension of the Gauss-Seidel
    * method that introduces a relaxation parameter (omega) to accelerate convergence.
    * The optimal choice of omega depends on the properties of the system matrix.
    */
    SOR,

    /*! \brief Symmetric Successive Over-Relaxation (SSOR) smoother.
    *
    * The Symmetric Successive Over-Relaxation method applies SOR in a forward sweep
    * followed by SOR in a backward sweep (often with the same relaxation parameter).
    * Similar to SGS, SSOR can offer improved smoothing properties, especially for
    * symmetric positive definite problems.
    */
    SSOR
};
/*! \brief Solves a linear system using the Algebraic Multigrid (AMG) method.
 *
 * \details
 * This function implements the core multigrid cycle (V-cycle, W-cycle, or F-cycle, specified by `cycle`)
 * to solve a linear system \f$A \cdot x = b\f$ using a pre-constructed AMG hierarchy.
 * AMG is an iterative method specifically designed for solving large sparse linear systems,
 * particularly effective for those arising from discretized partial differential equations.
 * It works by recursively solving the problem on a sequence of coarser grids,
 * where errors that are "smooth" (low-frequency) on finer grids appear "rough" (high-frequency)
 * on coarser grids and can be effectively damped by simple iterative smoothers.
 *
 * \section amg_cycle_description AMG Cycle Description
 *
 * The `amg_solve` function orchestrates an AMG cycle to reduce the error in the solution.
 * Each cycle typically involves the following steps:
 *
 * 1.  \b Pre-smoothing: Apply `n1` iterations of a chosen `smoother` to the current
 * approximate solution \f$\mathbf{x}\f$ on the current level. This step aims to damp
 * high-frequency error components.
 *
 * 2.  \b Coarse-grid \b Correction:
 * a. Compute the residual \f$\mathbf{r} = \mathbf{b} - A \mathbf{x}\f$.
 * b. Restrict the residual to the next coarser grid: \f$\mathbf{r}_c = R \mathbf{r}\f$,
 * where \f$R\f$ is the restriction operator from the `hierarchy`.
 * c. Solve the coarse-grid error equation \f$A_c \mathbf{e}_c = \mathbf{r}_c\f$ recursively.
 * If the coarsest level is reached, the problem is solved directly (e.g., using a direct solver).
 * The exact recursive call depends on the `Cycle` type.
 * d. Prolongate the coarse-grid error back to the fine grid: \f$\mathbf{e} = P \mathbf{e}_c\f$,
 * where \f$P\f$ is the prolongation operator from the `hierarchy`.
 * e. Update the fine-grid solution: \f$\mathbf{x} = \mathbf{x} + \mathbf{e}\f$.
 *
 * 3.  \b Post-smoothing: Apply `n2` iterations of the chosen `smoother` to the updated
 * approximate solution \f$\mathbf{x}\f$ on the current level. This further dampens remaining
 * high-frequency errors.
 *
 * The `Cycle` enumeration defines how the coarse-grid problem (step 2c) is handled recursively:
 * - \b V-Cycle (`Cycle::Vcycle`): The coarse-grid problem is solved by recursively applying
 * a single AMG V-cycle. This is the simplest and most common cycle type.
 * - \b W-Cycle (`Cycle::Wcycle`): The coarse-grid problem is solved by recursively applying
 * *more than one* V-cycle (typically two) on the coarser level before interpolating back.
 * This can lead to more thorough error reduction at the cost of increased computation per cycle.
 * - \b F-Cycle (`Cycle::Fcycle`): The F-cycle starts by recursively solving the problem on the
 * coarsest level, then interpolates the solution to the next finer level and performs
 * one or more V-cycles. This process is repeated, gradually moving to finer levels.
 * It's often used to provide a good initial guess and can lead to faster overall convergence.
 *
 * \param hierarchy A constant reference to the `hierarchy` object, which contains all the
 * necessary matrices (`A_cs`), prolongation operators (`prolongations`), and restriction
 * operators (`restrictions`) for all levels of the multigrid hierarchy. This hierarchy
 * must have been previously set up by a function like `uaamg_setup`, `saamg_setup`,
 * or `rsamg_setup`.
 * \param x On input, an initial guess for the solution vector; on output, the computed solution vector.
 * \param b The right-hand side vector of the linear system.
 * \param n1 The number of pre-smoothing iterations to perform at each level.
 * \param n2 The number of post-smoothing iterations to perform at each level.
 * \param cycle An enumeration value (`Cycle::Vcycle`, `Cycle::Wcycle`, or `Cycle::Fcycle`)
 * specifying the type of multigrid cycle to be performed.
 * \param smoother An enumeration value (`Smoother::Jacobi`, `Smoother::Gauss_Seidel`, `Smoother::Symm_Gauss_Seidel`, `Smoother::SOR`, `Smoother::SSOR`)
 * specifying the type of stationary iterative method to be used as a smoother at each level.
 * \param control An `iter_control` object that manages the overall iterative process,
 * including convergence tolerance (`rel_tol`, `abs_tol`) and maximum number of cycles (`max_cycle`).
 *
 * \return An integer status code:
 * - `0` if the AMG solver converged successfully within the specified tolerance.
 * - `1` if the maximum number of cycles (`control.max_cycle`) was reached without convergence.
 * - Negative values indicate errors (e.g., singular matrix on coarsest level, hierarchy not properly built).
 *
 * \section amg_example Example Usage
 * Below is a simplified example demonstrating how to use the `amg_solve` function.
 * This assumes `csr_matrix`, `vector`, `hierarchy`, `iter_control`, `Cycle` enum,
 * and `Smoother` enum are properly defined and functional.
 *
 * \code
 * #include "linalglib.h"
 * #include <iostream>
 * #include <vector>
 * #include <algorithm> // For std::min
 * #include <utility>   // For std::pair
 *
 * int main() {
 * // 1. Create a sample sparse matrix (e.g., from a 1D Poisson problem)
 * int N = 100; // Size of the matrix
 * std::vector<int> row_ptr(N + 1);
 * std::vector<int> col_ind;
 * std::vector<double> val;
 *
 * row_ptr[0] = 0;
 * int nnz_count = 0;
 * for (int i = 0; i < N; ++i) {
 * col_ind.push_back(i);
 * val.push_back(2.0);
 * nnz_count++;
 * if (i > 0) {
 * col_ind.push_back(i - 1);
 * val.push_back(-1.0);
 * nnz_count++;
 * }
 * if (i < N - 1) {
 * col_ind.push_back(i + 1);
 * val.push_back(-1.0);
 * nnz_count++;
 * }
 * row_ptr[i+1] = nnz_count;
 * }
 * // Sort elements by column index within each row to ensure proper CSR format
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
 * csr_matrix A(row_ptr, col_ind, val, N, N, nnz_count);
 *
 * // Define the right-hand side vector b (e.g., all ones)
 * vector b(N);
 * for (int i = 0; i < N; ++i) {
 * b[i] = 1.0;
 * }
 *
 * // Define an initial guess for the solution vector x (e.g., all zeros)
 * vector x(N);
 * x.zeros();
 *
 * // 2. Setup the AMG hierarchy (using Unsmoothed Aggregation as an example)
 * int max_levels = 5; // Maximum number of levels
 * hierarchy mg_hierarchy;
 * std::cout << "Setting up UAAMG hierarchy..." << std::endl;
 * uaamg_setup(A, max_levels, mg_hierarchy);
 * std::cout << "Hierarchy setup complete." << std::endl;
 *
 * // 3. Set AMG parameters
 * int n1 = 2; // Number of pre-smoothing iterations
 * int n2 = 2; // Number of post-smoothing iterations
 * Cycle cycle_type = Cycle::Vcycle; // Use V-cycle
 * Smoother smoother_type = Smoother::Jacobi; // Use Jacobi smoother
 *
 * // 4. Set up iteration control
 * iter_control control;
 * control.rel_tol = 1e-7;
 * control.abs_tol = 1e-10;
 * control.max_cycle = 50; // Max 50 AMG cycles
 *
 * // 5. Call the AMG solve function
 * std::cout << "\nStarting AMG solver (V-cycle, Jacobi smoother)..." << std::endl;
 * int status = amg_solve(mg_hierarchy, x, b, n1, n2, cycle_type, smoother_type, control);
 *
 * if (status == 0) {
 * std::cout << "AMG converged successfully!" << std::endl;
 * } else {
 * std::cout << "AMG did NOT converge. Status code: " << status << std::endl;
 * }
 *
 * // Optional: Print a few elements of the solution
 * std::cout << "Approximate solution x (first 5 elements):" << std::endl;
 * for (int i = 0; i < std::min(5, N); ++i) {
 * std::cout << "x[" << i << "] = " << x[i] << std::endl;
 * }
 *
 * return 0;
 * }
 * \endcode
 */
LINALGLIB_API int amg_solve(const hierarchy &hierarchy, vector<double>& x, const vector<double>& b, int n1, int n2, Cycle cycle,
              Smoother smoother, iter_control control);
}

#endif