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

#ifndef TRIDIAGONAL_DESCR_INTERNAL_H
#define TRIDIAGONAL_DESCR_INTERNAL_H

#include "../../include/linalg_enums.h"

namespace linalg
{
    constexpr int tridiagonal_max_recursion_levels = 3;

    struct tridiagonal_descr
    {
        pivoting_strategy pivoting_strategy;

        // Buffers for non-pivoting approach (one per recursion level)
        double* lower_modified[tridiagonal_max_recursion_levels];
        double* main_modified[tridiagonal_max_recursion_levels];
        double* upper_modified[tridiagonal_max_recursion_levels];
        double* B_modified[tridiagonal_max_recursion_levels];

        double* spike_lower[tridiagonal_max_recursion_levels];
        double* spike_main[tridiagonal_max_recursion_levels];
        double* spike_upper[tridiagonal_max_recursion_levels];
        double* spike_B[tridiagonal_max_recursion_levels];
        double* spike_X[tridiagonal_max_recursion_levels];

        // Buffers for partial pivoting approach (to be implemented)
        double* lower_pad;
        double* main_pad;
        double* upper_pad;
        double* B_pad;

        double* w_pad;
        double* v_pad;

        double* mt;

        double* S_lower;
        double* S_main;
        double* S_upper;
        double* S_B;
    };
}

#endif // TRIDIAGONAL_DESCR_INTERNAL_H
