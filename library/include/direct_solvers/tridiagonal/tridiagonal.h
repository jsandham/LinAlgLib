//********************************************************************************
//
// MIT License
//
// Copyright(c) 2026 James Sandham
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

#ifndef TRIDIAGONAL_H
#define TRIDIAGONAL_H

#include "linalg_enums.h"
#include "linalg_export.h"
#include "linalg_types.h"
#include "vector.h"

/*! \file
 *  \brief tridiagonal.h provides tridiagonal solver APIs
 */

/*! \defgroup tridiagonal_solvers Tridiagonal
 *  \brief Tridiagonal system solver APIs.
 *  \ingroup direct_solvers
 */

namespace linalg
{
    /*! \ingroup tridiagonal_solvers
     * @brief Creates an opaque descriptor for tridiagonal analysis/solve operations.
     *
     * Allocates internal workspace and backend-specific metadata used by
        * tridiagonal_analysis() and tridiagonal_solver(). The descriptor also stores
        * solver configuration such as the selected pivoting strategy.
     *
     * @param descr A pointer to a tridiagonal_descr* that will be initialized
     * to point to a newly allocated descriptor.
        * @see destroy_tridiagonal_descr, set_pivoting_strategy
     */
    LINALGLIB_API void create_tridiagonal_descr(tridiagonal_descr** descr);

    /*! \ingroup tridiagonal_solvers
     * @brief Destroys a tridiagonal analysis/solve descriptor.
     *
     * Frees all resources associated with the descriptor. After this call,
     * the descr pointer should not be used.
     *
     * @param descr The descriptor to destroy. Can be nullptr.
     * @see create_tridiagonal_descr
     */
    LINALGLIB_API void destroy_tridiagonal_descr(tridiagonal_descr* descr);

    /*! \ingroup tridiagonal_solvers
     * @brief Sets the pivoting strategy used by the tridiagonal solver.
     *
     * Configures whether the solver applies pivoting during the tridiagonal solve.
     * This must be called after create_tridiagonal_descr() and before
        * tridiagonal_analysis(), as both analysis and solve dispatch may depend on
        * the selected mode. This provides an explicit way to choose between the
        * non-pivoting and pivoting variants of the tridiagonal solver.
     *
     * @param descr The tridiagonal descriptor to configure.
     * @param strategy The pivoting strategy to apply. Use pivoting_strategy::none
     *                 for maximum performance on diagonally dominant systems, or
     *                 pivoting_strategy::partial for improved numerical stability.
     * @see create_tridiagonal_descr, tridiagonal_analysis, pivoting_strategy
     */
    LINALGLIB_API void set_pivoting_strategy(tridiagonal_descr* descr, pivoting_strategy strategy);

    /*! \ingroup tridiagonal_solvers
     * @brief Performs analysis for tridiagonal solves.
     *
        * Preprocesses the tridiagonal system layout and caches data in descr for
        * subsequent tridiagonal_solver() calls. Depending on the problem size,
        * backend, and selected pivoting strategy, this may prepare workspace for
        * Thomas, parallel cyclic reduction (PCR), or tiled PCR-SPIKE based solves.
     *
     * @param m The number of rows in the tridiagonal system.
     * @param n The number of right-hand-side columns/vectors to solve.
     * @param lower_diag The lower diagonal entries (subdiagonal).
     * @param main_diag The main diagonal entries.
     * @param upper_diag The upper diagonal entries (superdiagonal).
        * @param descr The descriptor to populate with analysis data and solver configuration.
        * @see tridiagonal_solver, set_pivoting_strategy
     */
    LINALGLIB_API void tridiagonal_analysis(int                  m,
                                            int                  n,
                                            const vector<float>& lower_diag,
                                            const vector<float>& main_diag,
                                            const vector<float>& upper_diag,
                                            tridiagonal_descr*   descr);

    /*! \ingroup tridiagonal_solvers
        * @brief Solves a tridiagonal system using a specialized tridiagonal algorithm.
     *
     * This function solves the linear system defined by a tridiagonal matrix
        * represented by its lower diagonal, main diagonal, and upper diagonal vectors.
        * The implementation may dispatch to different solver families depending on
        * the backend, matrix size, and descriptor configuration, including Thomas,
        * PCR, and tiled PCR-SPIKE based methods. The descriptor also determines
        * whether a pivoting strategy is requested for the solve.
     *
     * @param m The number of rows in the tridiagonal matrix.
     * @param n The number of columns in the right-hand-side matrix.
     * @param lower_diag The vector representing the lower diagonal of the tridiagonal matrix.
     * @param main_diag The vector representing the main diagonal of the tridiagonal matrix.
     * @param upper_diag The vector representing the upper diagonal of the tridiagonal matrix.
     * @param rhs The right-hand side vector of the linear system.
     * @param solution The output vector that will contain the solution to the system.
        * @param descr The descriptor containing analysis information and the selected
        * pivoting strategy for the tridiagonal system.
     */
    LINALGLIB_API void tridiagonal_solver(int                      m,
                                          int                      n,
                                          const vector<float>&     lower_diag,
                                          const vector<float>&     main_diag,
                                          const vector<float>&     upper_diag,
                                          const vector<float>&     rhs,
                                          vector<float>&           solution,
                                          const tridiagonal_descr* descr);
}

#endif
