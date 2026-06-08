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

#include "device_tridiagonal.h"

#include <iostream>

#include "../../trace.h"
#include "../../utility.h"

#if defined(LINALGLIB_HAS_CUDA)
#include "cuda/cuda_tridiagonal.h"
#endif

void linalg::device_partial_pivoting_algorithm(int                   m,
                                               int                   n,
                                               const vector<double>& lower_diag,
                                               const vector<double>& main_diag,
                                               const vector<double>& upper_diag,
                                               const vector<double>& rhs,
                                               vector<double>&       solution,
                                               pivoting_data&        pivot_data)
{
    ROUTINE_TRACE("linalg::device_partial_pivoting_algorithm");
    if constexpr(is_cuda_available())
    {
        CALL_CUDA(cuda_partial_pivoting_solver(m,
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
                                               pivot_data.S_B.get_vec()));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}

void linalg::device_non_pivoting_algorithm(int                   m,
                                           int                   n,
                                           const vector<double>& lower_diag,
                                           const vector<double>& main_diag,
                                           const vector<double>& upper_diag,
                                           const vector<double>& rhs,
                                           vector<double>&       solution,
                                           non_pivoting_data&    non_pivot_data)
{
    ROUTINE_TRACE("linalg::device_non_pivoting_algorithm");
    if constexpr(is_cuda_available())
    {
        constexpr int L = non_pivoting_data::tridiagonal_max_recursion_levels;

        double* lower_modified_ptrs[L];
        double* main_modified_ptrs[L];
        double* upper_modified_ptrs[L];
        double* B_modified_ptrs[L];
        double* spike_lower_ptrs[L];
        double* spike_main_ptrs[L];
        double* spike_upper_ptrs[L];
        double* spike_B_ptrs[L];
        double* spike_X_ptrs[L];

        for(int i = 0; i < L; i++)
        {
            lower_modified_ptrs[i] = non_pivot_data.lower_modified[i].get_vec();
            main_modified_ptrs[i]  = non_pivot_data.main_modified[i].get_vec();
            upper_modified_ptrs[i] = non_pivot_data.upper_modified[i].get_vec();
            B_modified_ptrs[i]     = non_pivot_data.B_modified[i].get_vec();
            spike_lower_ptrs[i]    = non_pivot_data.spike_lower[i].get_vec();
            spike_main_ptrs[i]     = non_pivot_data.spike_main[i].get_vec();
            spike_upper_ptrs[i]    = non_pivot_data.spike_upper[i].get_vec();
            spike_B_ptrs[i]        = non_pivot_data.spike_B[i].get_vec();
            spike_X_ptrs[i]        = non_pivot_data.spike_X[i].get_vec();
        }

        CALL_CUDA(cuda_non_pivoting_solver(m,
                                           n,
                                           lower_diag.get_vec(),
                                           main_diag.get_vec(),
                                           upper_diag.get_vec(),
                                           rhs.get_vec(),
                                           solution.get_vec(),
                                           lower_modified_ptrs,
                                           main_modified_ptrs,
                                           upper_modified_ptrs,
                                           B_modified_ptrs,
                                           spike_lower_ptrs,
                                           spike_main_ptrs,
                                           spike_upper_ptrs,
                                           spike_B_ptrs,
                                           spike_X_ptrs));
    }
    else
    {
        std::cout << "Error: Not device backend available for the function " << __func__
                  << std::endl;
    }
}
