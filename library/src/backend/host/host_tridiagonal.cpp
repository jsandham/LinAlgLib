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

template <typename T>
void linalg::host_partial_pivoting_algorithm(int               m,
                                             int               n,
                                             const vector<T>&  lower_diag,
                                             const vector<T>&  main_diag,
                                             const vector<T>&  upper_diag,
                                             const vector<T>&  rhs,
                                             vector<T>&        solution,
                                             pivoting_data<T>& pivot_data)
{
    ROUTINE_TRACE("linalg::host_partial_pivoting_algorithm");

    const int L = pivoting_data<T>::tridiagonal_max_recursion_levels;
    T*        lower_pad_ptrs[L];
    T*        main_pad_ptrs[L];
    T*        upper_pad_ptrs[L];
    T*        B_pad_ptrs[L];
    T*        w_ptrs[L];
    T*        v_ptrs[L];
    T*        mt_ptrs[L];
    T*        S_lower_ptrs[L];
    T*        S_main_ptrs[L];
    T*        S_upper_ptrs[L];
    T*        S_B_ptrs[L];

    for(int i = 0; i < L; i++)
    {
        lower_pad_ptrs[i] = pivot_data.lower_pad[i].get_vec();
        main_pad_ptrs[i]  = pivot_data.main_pad[i].get_vec();
        upper_pad_ptrs[i] = pivot_data.upper_pad[i].get_vec();
        B_pad_ptrs[i]     = pivot_data.B_pad[i].get_vec();
        w_ptrs[i]         = pivot_data.w[i].get_vec();
        v_ptrs[i]         = pivot_data.v[i].get_vec();
        mt_ptrs[i]        = pivot_data.mt[i].get_vec();
        S_lower_ptrs[i]   = pivot_data.S_lower[i].get_vec();
        S_main_ptrs[i]    = pivot_data.S_main[i].get_vec();
        S_upper_ptrs[i]   = pivot_data.S_upper[i].get_vec();
        S_B_ptrs[i]       = pivot_data.S_B[i].get_vec();
    }

    spike_algorithm_template<T>(m,
                                n,
                                lower_diag.get_vec(),
                                main_diag.get_vec(),
                                upper_diag.get_vec(),
                                rhs.get_vec(),
                                solution.get_vec(),
                                lower_pad_ptrs,
                                main_pad_ptrs,
                                upper_pad_ptrs,
                                B_pad_ptrs,
                                w_ptrs,
                                v_ptrs,
                                mt_ptrs,
                                S_lower_ptrs,
                                S_main_ptrs,
                                S_upper_ptrs,
                                S_B_ptrs);
}

template <typename T>
void linalg::host_non_pivoting_algorithm(int                   m,
                                         int                   n,
                                         const vector<T>&      lower_diag,
                                         const vector<T>&      main_diag,
                                         const vector<T>&      upper_diag,
                                         const vector<T>&      rhs,
                                         vector<T>&            solution,
                                         non_pivoting_data<T>& non_pivot_data)
{
    ROUTINE_TRACE("linalg::host_non_pivoting_algorithm");
    thomas_algorithm_template<T>(m,
                                 n,
                                 lower_diag.get_vec(),
                                 main_diag.get_vec(),
                                 upper_diag.get_vec(),
                                 rhs.get_vec(),
                                 solution.get_vec());
}

template void linalg::host_partial_pivoting_algorithm<float>(int                   m,
                                                             int                   n,
                                                             const vector<float>&  lower_diag,
                                                             const vector<float>&  main_diag,
                                                             const vector<float>&  upper_diag,
                                                             const vector<float>&  rhs,
                                                             vector<float>&        solution,
                                                             pivoting_data<float>& pivot_data);

template void linalg::host_non_pivoting_algorithm<float>(int                       m,
                                                         int                       n,
                                                         const vector<float>&      lower_diag,
                                                         const vector<float>&      main_diag,
                                                         const vector<float>&      upper_diag,
                                                         const vector<float>&      rhs,
                                                         vector<float>&            solution,
                                                         non_pivoting_data<float>& non_pivot_data);

template void linalg::host_partial_pivoting_algorithm<double>(int                    m,
                                                              int                    n,
                                                              const vector<double>&  lower_diag,
                                                              const vector<double>&  main_diag,
                                                              const vector<double>&  upper_diag,
                                                              const vector<double>&  rhs,
                                                              vector<double>&        solution,
                                                              pivoting_data<double>& pivot_data);

template void
    linalg::host_non_pivoting_algorithm<double>(int                        m,
                                                int                        n,
                                                const vector<double>&      lower_diag,
                                                const vector<double>&      main_diag,
                                                const vector<double>&      upper_diag,
                                                const vector<double>&      rhs,
                                                vector<double>&            solution,
                                                non_pivoting_data<double>& non_pivot_data);
