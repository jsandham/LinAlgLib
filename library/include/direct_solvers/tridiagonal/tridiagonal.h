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
    struct non_pivoting_data
    {
        constexpr static int tridiagonal_max_recursion_levels = 3;

        vector<double> lower_modified[tridiagonal_max_recursion_levels];
        vector<double> main_modified[tridiagonal_max_recursion_levels];
        vector<double> upper_modified[tridiagonal_max_recursion_levels];
        vector<double> B_modified[tridiagonal_max_recursion_levels];

        vector<double> spike_lower[tridiagonal_max_recursion_levels];
        vector<double> spike_main[tridiagonal_max_recursion_levels];
        vector<double> spike_upper[tridiagonal_max_recursion_levels];
        vector<double> spike_B[tridiagonal_max_recursion_levels];
        vector<double> spike_X[tridiagonal_max_recursion_levels];
    };

    struct pivoting_data
    {
        vector<double> lower_pad;
        vector<double> main_pad;
        vector<double> upper_pad;
        vector<double> B_pad;

        vector<double> w;
        vector<double> v;
        vector<double> mt;

        vector<double> S_lower;
        vector<double> S_main;
        vector<double> S_upper;
        vector<double> S_B;
    };

    class tridiagonal_solver
    {
    private:
        int               m;
        int               n;
        pivoting_strategy strategy;

        bool on_host;

        // Non-pivoting data
        non_pivoting_data non_pivot_data;

        // Pivoting data
        pivoting_data pivot_data;

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
