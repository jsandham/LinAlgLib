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

#include "../../linalglib_export.h"
#include "../../vector.h"

#include "../iter_control.h"

/*! \file
 *  \brief amg.h provides interface for algebraic multigrid solver
 */

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
struct heirarchy
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







struct heirarchy2
{
    /**
    * @brief Prolongation matrices.
    *
    * These matrices are used to interpolate the correction computed on a coarser
    * level back to the next finer level. The vector stores the prolongation
    * operators between consecutive levels, where `prolongations[i]` maps from
    * level `i+1` to level `i`.
    */
    std::vector<csr_matrix2> prolongations;

    /**
    * @brief Restriction matrices.
    *
    * These matrices are used to transfer the residual from a finer level to the
    * next coarser level. The vector stores the restriction operators between
    * consecutive levels, where `restrictions[i]` maps from level `i` to level `i+1`.
    * Typically, the restriction operator is related to the transpose of the
    * prolongation operator.
    */
    std::vector<csr_matrix2> restrictions;

    /**
    * @brief Coarse level matrices.
    *
    * These matrices represent the discretized problem on the coarser levels of
    * the multigrid hierarchy. `A_cs[i]` is the system matrix at level `i` of the
    * hierarchy, with `A_cs[0]` being the original system matrix on the finest level.
    * The size of these matrices decreases as the level number increases.
    */
    std::vector<csr_matrix2> A_cs;

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

/*! \ingroup iterative_solvers
 * \brief Algebraic Multigrid solver.
 *
 * \details
 * \p amg_solve iteratively solves a linear system of equations \f$A x = b\f$
 * using an Algebraic Multigrid (AMG) method. This function assumes that a
 * multigrid hierarchy (containing restriction operators, prolongation operators,
 * and coarse-level matrices) has already been constructed and is provided
 * as input. The specific cycling strategy and smoother to be used during the
 * solve phase are also specified as input parameters.
 *
 * The AMG solve process typically involves recursively applying the following steps:
 *
 * 1.  **Pre-smoothing:** Apply a few iterations (\p n1) of a chosen smoother
 * (e.g., Jacobi, Gauss-Seidel) to the current solution on the current level
 * to reduce high-frequency errors.
 *
 * 2.  **Residual Calculation:** Compute the residual \f$r = b - A x\f$.
 *
 * 3.  **Restriction:** Project the residual to the next coarser level using the
 * restriction operator.
 *
 * 4.  **Coarse-Level Correction:** Solve the residual equation on the coarser
 * level (either recursively using AMG or directly if the coarsest level
 * is small enough) to obtain a coarse-level correction.
 *
 * 5.  **Prolongation:** Interpolate the coarse-level correction back to the current
 * (finer) level using the prolongation operator.
 *
 * 6.  **Correction Update:** Add the interpolated correction to the current solution.
 *
 * 7.  **Post-smoothing:** Apply a few more iterations (\p n2) of the chosen
 * smoother to the updated solution on the current level to further reduce
 * high-frequency errors.
 *
 * The order and number of recursive calls to the coarse level depend on the
 * chosen cycle type (\ref Vcycle, \ref Wcycle, or \ref Fcycle). The iteration
 * continues until a specified convergence criterion (defined by the tolerance
 * \p tol or the \ref iter_control struct) is met or the maximum number of
 * iterations is reached.
 *
 * **Note:** Before calling \p amg_solve, the \p hierarchy structure must be
 * properly initialized by calling a setup function such as \ref saamg_setup
 * or \ref uaamg_setup (or another AMG setup routine).
 *
 * @param[in] hierarchy
 * Structure of type \ref heirarchy containing the hierarchy of restriction
 * operators, prolongation operators, and coarse-level system matrices. This
 * structure must have been populated by a prior call to an AMG setup function
 * (e.g., \ref saamg_setup or \ref uaamg_setup).
 * @param[inout] x
 * Array of \p m elements serving as both the initial guess and the output
 * solution vector for the linear system \f$A x = b\f$. On input, it should
 * contain an initial guess for the solution. On output, it will contain the
 * converged solution (if convergence is achieved).
 * @param[in] b
 * Array of \p m elements containing the right-hand side vector of the linear
 * system \f$A x = b\f$.
 * @param[in] n1
 * Number of pre-smoothing iterations to perform at each level (except the
 * coarsest) before the residual is restricted to the coarser level.
 * @param[in] n2
 * Number of post-smoothing iterations to perform at each level (except the
 * coarsest) after the correction is interpolated from the coarser level.
 * @param[in] cycle
 * Enumeration value of type \ref Cycle specifying the multigrid cycle type to
 * be used during the solve phase (e.g., \ref Vcycle, \ref Wcycle, \ref Fcycle).
 * @param[in] smoother
 * Enumeration value of type \ref Smoother specifying the iterative method to
 * be used as a smoother at each level of the multigrid hierarchy (e.g.,
 * \ref Jacobi, \ref Gauss_Seidel).
 * @param[in] control
 * Structure of type \ref iter_control specifying the convergence criteria
 * (relative tolerance, absolute tolerance) and the maximum number of iterations
 * for the AMG solve.
 *
 * @retval int
 * The total number of multigrid cycles performed during the solve phase.
 * Returns -1 if the solver did not converge to a solution within the specified
 * tolerance and maximum number of iterations.
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
 * iter_control control;
 * control.max_iter = 20;
 * int cycles = amg_solve(hierarchy, x.data(), b.data(), 2, 2, Cycle::Vcycle, Smoother::Gauss_Seidel, control);
 * std::cout << "Number of cycles: " << cycles << std::endl;
 *
 * return 0;
 * }
 * \endcode
 */
/**@}*/
LINALGLIB_API int amg_solve(const heirarchy &hierarchy, double *x, const double *b, int n1, int n2, Cycle cycle,
    Smoother smoother, iter_control control);

LINALGLIB_API int amg_solve(const heirarchy2 &hierarchy, vector2& x, const vector2& b, int n1, int n2, Cycle cycle,
              Smoother smoother, iter_control control);

#endif