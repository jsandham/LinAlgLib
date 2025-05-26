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

//-------------------------------------------------------------------------------
// richardson method
//-------------------------------------------------------------------------------
void richardson_iteration(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x,
                            double *res, const double *b, int n, double theta)
{
    ROUTINE_TRACE("richardson_iteration");

    // update approximation
#if defined(_OPENMP)
#pragma omp parallel for schedule(static, 1024)
#endif
    for (int j = 0; j < n; j++)
    {
        x[j] = x[j] + theta * res[j];
    }
}

int rich(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
         double theta, iter_control control)
{
    ROUTINE_TRACE("rich");

    // res = b - A * x
    std::vector<double> res(n);
    compute_residual(csr_row_ptr, csr_col_ind, csr_val, x, b, res.data(), n);

    double initial_res_norm = norm_inf(res.data(), n);

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        richardson_iteration(csr_row_ptr, csr_col_ind, csr_val, x, res.data(), b, n, theta);

        compute_residual(csr_row_ptr, csr_col_ind, csr_val, x, b, res.data(), n);

        double res_norm = norm_inf(res.data(), n);

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

int rich(const csr_matrix2& A, vector2& x, const vector2& b, double theta, iter_control control)
{
    if(!A.is_on_host() || !x.is_on_host() || !b.is_on_host())
    {
        std::cout << "Error: A matrix, x vector, and b vector must on host for jacobi iteration" << std::endl;
        return -1;
    }

    return rich(A.get_row_ptr(), A.get_col_ind(), A.get_val(), x.get_vec(), b.get_vec(), A.get_m(), theta, control);
}





















rich_solver::rich_solver(){}

rich_solver::~rich_solver(){}

void rich_solver::build(const csr_matrix2& A)
{
    res.resize(A.get_m());
}

int rich_solver::solve(const csr_matrix2& A, vector2& x, const vector2& b, iter_control control, double theta)
{
    ROUTINE_TRACE("rich_solver::solve");

    // res = b - A * x
    compute_residual(A, x, b, res);

    double initial_res_norm = res.norm_inf2();

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        richardson_iteration(A.get_row_ptr(), A.get_col_ind(), A.get_val(), x.get_vec(), res.get_vec(), b.get_vec(), A.get_m(), theta);

        compute_residual(A, x, b, res);

        double res_norm = res.norm_inf2();

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
