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

#include "../../../include/direct_solvers/tridiagonal/tridiagonal.h"

#include "../../trace.h"
#include "../../utility.h"

#include "../../backend/device/device_math.h"
#include "../../backend/host/host_math.h"

static constexpr int MAX_RECURSION_LEVELS = 3;

struct linalg::tridiagonal_descr
{
    pivoting_strategy pivoting_strategy;

    // Buffers for non-pivoting approach (one per recursion level)
    float* lower_modified[MAX_RECURSION_LEVELS];
    float* main_modified[MAX_RECURSION_LEVELS];
    float* upper_modified[MAX_RECURSION_LEVELS];
    float* b_modified[MAX_RECURSION_LEVELS];

    float* spike_lower[MAX_RECURSION_LEVELS];
    float* spike_main[MAX_RECURSION_LEVELS];
    float* spike_upper[MAX_RECURSION_LEVELS];
    float* spike_b[MAX_RECURSION_LEVELS];
    float* spike_x[MAX_RECURSION_LEVELS];

    // Buffers for partial pivoting approach (to be implemented)
    float* lower_pad;
    float* main_pad;
    float* upper_pad;

    float* w_pad;
    float* v_pad;
};

void linalg::create_tridiagonal_descr(tridiagonal_descr** descr)
{
    ROUTINE_TRACE("linalg::create_tridiagonal_descr");

    *descr = new tridiagonal_descr;

    (*descr)->pivoting_strategy = pivoting_strategy::none;

    for(int level = 0; level < MAX_RECURSION_LEVELS; level++)
    {
        (*descr)->lower_modified[level] = nullptr;
        (*descr)->main_modified[level]  = nullptr;
        (*descr)->upper_modified[level] = nullptr;
        (*descr)->b_modified[level]     = nullptr;

        (*descr)->spike_lower[level] = nullptr;
        (*descr)->spike_main[level]  = nullptr;
        (*descr)->spike_upper[level] = nullptr;
        (*descr)->spike_b[level]     = nullptr;
        (*descr)->spike_x[level]     = nullptr;
    }
}

void linalg::destroy_tridiagonal_descr(tridiagonal_descr* descr)
{
    ROUTINE_TRACE("linalg::destroy_tridiagonal_descr");

    if(descr != nullptr)
    {
        free_tridiagonal_device_data(descr);
        delete descr;
    }
}

void linalg::set_pivoting_strategy(tridiagonal_descr* descr, pivoting_strategy strategy)
{
    ROUTINE_TRACE("linalg::set_pivoting_strategy");

    if(descr != nullptr)
    {
        descr->pivoting_strategy = strategy;
    }
}

void linalg::tridiagonal_analysis(int                  m,
                                  int                  n,
                                  const vector<float>& lower_diag,
                                  const vector<float>& main_diag,
                                  const vector<float>& upper_diag,
                                  tridiagonal_descr*   descr)
{
    ROUTINE_TRACE("linalg::tridiagonal_analysis");

    backend_dispatch("linalg::tridiagonal_analysis",
                     host_tridiagonal_analysis,
                     device_tridiagonal_analysis,
                     m,
                     n,
                     lower_diag,
                     main_diag,
                     upper_diag,
                     descr);
}

void linalg::tridiagonal_solver(int                      m,
                                int                      n,
                                const vector<float>&     lower_diag,
                                const vector<float>&     main_diag,
                                const vector<float>&     upper_diag,
                                const vector<float>&     rhs,
                                vector<float>&           solution,
                                const tridiagonal_descr* descr)
{
    ROUTINE_TRACE("linalg::tridiagonal_solver");

    return backend_dispatch("linalg::tridiagonal_solver",
                            host_tridiagonal_solver,
                            device_tridiagonal_solver,
                            m,
                            n,
                            lower_diag,
                            main_diag,
                            upper_diag,
                            rhs,
                            solution,
                            descr);
}
