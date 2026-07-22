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
    template <typename T>
    /*! \ingroup tridiagonal_solvers
     *  \brief Non-pivoting recursion buffers for the tridiagonal solver.
     */
    struct non_pivoting_data
    {
        /*! \brief Maximum number of recursion levels used by the tridiagonal solver. */
        constexpr static int tridiagonal_max_recursion_levels = 3;

        /*! Modified lower-diagonal values for each recursion level. */
        vector<T> lower_modified[tridiagonal_max_recursion_levels];
        /*! Modified main-diagonal values for each recursion level. */
        vector<T> main_modified[tridiagonal_max_recursion_levels];
        /*! Modified upper-diagonal values for each recursion level. */
        vector<T> upper_modified[tridiagonal_max_recursion_levels];
        /*! Modified right-hand side values for each recursion level. */
        vector<T> B_modified[tridiagonal_max_recursion_levels];

        /*! Spike lower diagonal values used in non-pivoting recursion. */
        vector<T> spike_lower[tridiagonal_max_recursion_levels];
        /*! Spike main diagonal values used in non-pivoting recursion. */
        vector<T> spike_main[tridiagonal_max_recursion_levels];
        /*! Spike upper diagonal values used in non-pivoting recursion. */
        vector<T> spike_upper[tridiagonal_max_recursion_levels];
        /*! Spike right-hand side block values used in the solver. */
        vector<T> spike_B[tridiagonal_max_recursion_levels];
        /*! Solution blocks corresponding to the spike matrices. */
        vector<T> spike_X[tridiagonal_max_recursion_levels];
    };

    template <typename T>
    /*! \ingroup tridiagonal_solvers
     *  \brief Pivoting recursion buffers for the tridiagonal solver.
     */
    struct pivoting_data
    {
        /*! \brief Maximum number of recursion levels used by the tridiagonal solver. */
        constexpr static int tridiagonal_max_recursion_levels = 3;
        /*! \brief Block dimension for pivoting operations and recursion. */
        constexpr static int block_dim = 256;

        /*! Padded lower-diagonal values for pivoting recursion. */
        vector<T> lower_pad[tridiagonal_max_recursion_levels];
        /*! Padded main-diagonal values for pivoting recursion. */
        vector<T> main_pad[tridiagonal_max_recursion_levels];
        /*! Padded upper-diagonal values for pivoting recursion. */
        vector<T> upper_pad[tridiagonal_max_recursion_levels];
        /*! Padded right-hand side values for pivoting recursion. */
        vector<T> B_pad[tridiagonal_max_recursion_levels];

        /*! Working vector used during pivoting factorization. */
        vector<T> w[tridiagonal_max_recursion_levels];
        /*! Working vector used during pivoting factorization. */
        vector<T> v[tridiagonal_max_recursion_levels];
        /*! Working vector used during pivoting factorization. */
        vector<T> mt[tridiagonal_max_recursion_levels];

        /*! Lower block of the Schur complement for each recursion level. */
        vector<T> S_lower[tridiagonal_max_recursion_levels];
        /*! Main block of the Schur complement for each recursion level. */
        vector<T> S_main[tridiagonal_max_recursion_levels];
        /*! Upper block of the Schur complement for each recursion level. */
        vector<T> S_upper[tridiagonal_max_recursion_levels];
        /*! Right-hand side block of the Schur complement for each recursion level. */
        vector<T> S_B[tridiagonal_max_recursion_levels];
    };

    /*! \ingroup tridiagonal_solvers
     *  \brief Tridiagonal system solver with optional pivoting support.
     */
    class tridiagonal_solver
    {
    private:
        int               m;
        int               n;
        pivoting_strategy strategy;

        bool on_host;

        // Non-pivoting data
        non_pivoting_data<double> non_pivot_data;

        // Pivoting data
        pivoting_data<double> pivot_data;

    public:
        /*! Construct a tridiagonal solver for an m-by-n system using the given pivot strategy. */
        tridiagonal_solver(int m, int n, pivoting_strategy strategy);
        /*! Destroy the tridiagonal solver and release any internal buffers. */
        ~tridiagonal_solver();

        tridiagonal_solver(const tridiagonal_solver&)            = delete;
        tridiagonal_solver& operator=(const tridiagonal_solver&) = delete;

        /*! Move owned solver data to the device backend. */
        void move_to_device();

        /*! Move owned solver data back to the host backend. */
        void move_to_host();

        /*! Solve a tridiagonal system from its diagonals and right-hand side. */
        void solve(const vector<double>& lower_diag,
                   const vector<double>& main_diag,
                   const vector<double>& upper_diag,
                   const vector<double>& rhs,
                   vector<double>&       solution);
    };
}

#endif
