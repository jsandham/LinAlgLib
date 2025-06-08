//********************************************************************************
//
// MIT License
//
// Copyright(c) 2019 James Sandham
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

#include "../../../include/IterativeSolvers/Classic/richardson.h"
#include "../../../include/slaf.h"
#include <iostream>
#include <math.h>
#include <vector>
#include <chrono>

#include "../../trace.h"

using namespace linalg;

//-------------------------------------------------------------------------------
// richardson method
//-------------------------------------------------------------------------------
namespace linalg
{
void richardson_iteration(const csr_matrix& A, vector<double>& x, vector<double>& res, double theta)
{
    ROUTINE_TRACE("richardson_iteration");

    double* x_ptr = x.get_vec();
    double* res_ptr = res.get_vec();

    // update approximation
#if defined(_OPENMP)
#pragma omp parallel for schedule(static, 1024)
#endif
    for (int j = 0; j < A.get_m(); j++)
    {
        x_ptr[j] = x_ptr[j] + theta * res_ptr[j];
    }
}
}

rich_solver::rich_solver(){}

rich_solver::~rich_solver(){}

void rich_solver::build(const csr_matrix& A)
{
    res.resize(A.get_m());
}

int rich_solver::solve(const csr_matrix& A, vector<double>& x, const vector<double>& b, iter_control control, double theta)
{
    ROUTINE_TRACE("rich_solver::solve");

    // res = b - A * x
    compute_residual(A, x, b, res);

    double initial_res_norm = norm_inf(res);

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        richardson_iteration(A, x, res, theta);

        compute_residual(A, x, b, res);

        double res_norm = norm_inf(res);
        
        if (control.residual_converges(res_norm, initial_res_norm))
        {
            break;
        }

        iter++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Richardson time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}
