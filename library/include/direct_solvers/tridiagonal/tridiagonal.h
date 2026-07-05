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
    struct non_pivoting_data
    {
        constexpr static int tridiagonal_max_recursion_levels = 3;

        vector<T> lower_modified[tridiagonal_max_recursion_levels];
        vector<T> main_modified[tridiagonal_max_recursion_levels];
        vector<T> upper_modified[tridiagonal_max_recursion_levels];
        vector<T> B_modified[tridiagonal_max_recursion_levels];

        vector<T> spike_lower[tridiagonal_max_recursion_levels];
        vector<T> spike_main[tridiagonal_max_recursion_levels];
        vector<T> spike_upper[tridiagonal_max_recursion_levels];
        vector<T> spike_B[tridiagonal_max_recursion_levels];
        vector<T> spike_X[tridiagonal_max_recursion_levels];
    };

    template <typename T>
    struct pivoting_data
    {
        constexpr static int tridiagonal_max_recursion_levels = 3;
        constexpr static int block_dim                        = 256;

        vector<T> lower_pad[tridiagonal_max_recursion_levels];
        vector<T> main_pad[tridiagonal_max_recursion_levels];
        vector<T> upper_pad[tridiagonal_max_recursion_levels];
        vector<T> B_pad[tridiagonal_max_recursion_levels];

        vector<T> w[tridiagonal_max_recursion_levels];
        vector<T> v[tridiagonal_max_recursion_levels];
        vector<T> mt[tridiagonal_max_recursion_levels];

        vector<T> S_lower[tridiagonal_max_recursion_levels];
        vector<T> S_main[tridiagonal_max_recursion_levels];
        vector<T> S_upper[tridiagonal_max_recursion_levels];
        vector<T> S_B[tridiagonal_max_recursion_levels];
    };

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
        tridiagonal_solver(int m, int n, pivoting_strategy strategy);
        ~tridiagonal_solver();

        tridiagonal_solver(const tridiagonal_solver&)            = delete;
        tridiagonal_solver& operator=(const tridiagonal_solver&) = delete;

        void move_to_device();
        void move_to_host();

        void solve(const vector<double>& lower_diag,
                   const vector<double>& main_diag,
                   const vector<double>& upper_diag,
                   const vector<double>& rhs,
                   vector<double>&       solution);
    };
}

#endif
