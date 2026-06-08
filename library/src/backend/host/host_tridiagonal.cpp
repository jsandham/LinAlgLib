//********************************************************************************
//
// MIT License
//
// Copyright(c) 2025-2026 James Sandham
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

#include <cassert>
#include <iostream>

#include "../../trace.h"

#include "host_tridiagonal.h"
#include "spike_algorithm.h"
#include "thomas_algorithm.h"

void linalg::host_partial_pivoting_algorithm(int                   m,
                                             int                   n,
                                             const vector<double>& lower_diag,
                                             const vector<double>& main_diag,
                                             const vector<double>& upper_diag,
                                             const vector<double>& rhs,
                                             vector<double>&       solution,
                                             pivoting_data&        pivot_data)
{
    ROUTINE_TRACE("linalg::host_partial_pivoting_algorithm");
    spike_algorithm_template<double>(m,
                                     n,
                                     lower_diag.get_vec(),
                                     main_diag.get_vec(),
                                     upper_diag.get_vec(),
                                     rhs.get_vec(),
                                     solution.get_vec(),
                                     pivot_data.lower_pad.get_vec(),
                                     pivot_data.main_pad.get_vec(),
                                     pivot_data.upper_pad.get_vec(),
                                     pivot_data.B_pad.get_vec(),
                                     pivot_data.w.get_vec(),
                                     pivot_data.v.get_vec(),
                                     pivot_data.mt.get_vec(),
                                     pivot_data.S_lower.get_vec(),
                                     pivot_data.S_main.get_vec(),
                                     pivot_data.S_upper.get_vec(),
                                     pivot_data.S_B.get_vec());
}

void linalg::host_non_pivoting_algorithm(int                   m,
                                         int                   n,
                                         const vector<double>& lower_diag,
                                         const vector<double>& main_diag,
                                         const vector<double>& upper_diag,
                                         const vector<double>& rhs,
                                         vector<double>&       solution,
                                         non_pivoting_data&    non_pivot_data)
{
    ROUTINE_TRACE("linalg::host_non_pivoting_algorithm");
    thomas_algorithm_template<double>(m,
                                      n,
                                      lower_diag.get_vec(),
                                      main_diag.get_vec(),
                                      upper_diag.get_vec(),
                                      rhs.get_vec(),
                                      solution.get_vec());
}
