//********************************************************************************
//
// MIT License
//
// Copyright(c) 2024 James Sandham
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

#include "../../../include/IterativeSolvers/Classic/ssor.h"
#include "../../../include/slaf.h"
#include <iostream>
#include <chrono>
#include <vector>

#include "../../trace.h"

//-------------------------------------------------------------------------------
// symmetric successive over-relaxation method
//-------------------------------------------------------------------------------
void ssor_iteration(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b,
                      int n, double omega)
{
    ROUTINE_TRACE("ssor_iteration");

    // forward pass
    for (int j = 0; j < n; j++)
    {
        double sigma = 0.0;
        double ajj = 0.0;

        int row_start = csr_row_ptr[j];
        int row_end = csr_row_ptr[j + 1];

        for (int k = row_start; k < row_end; k++)
        {
            int col = csr_col_ind[k];
            double val = csr_val[k];

            if (col != j)
            {
                sigma = sigma + val * x[col];
            }
            else
            {
                ajj = val;
            }
        }
        x[j] = x[j] + omega * ((b[j] - sigma) / ajj - x[j]);
    }

    // backward pass
    for (int j = n - 1; j > -1; j--)
    {
        double sigma = 0.0;
        double ajj = 0.0;
        
        int row_start = csr_row_ptr[j];
        int row_end = csr_row_ptr[j + 1];

        for (int k = row_start; k < row_end; k++)
        {
            int col = csr_col_ind[k];
            double val = csr_val[k];

            if (col != j)
            {
                sigma = sigma + val * x[col];
            }
            else
            {
                ajj = val;
            }
        }
        
        x[j] = x[j] + omega * ((b[j] - sigma) / ajj - x[j]);
    }
}

int ssor(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
         double omega, iter_control control)
{
    ROUTINE_TRACE("ssor");

    // res = b - A * x
    std::vector<double> res(n);
    compute_residual(csr_row_ptr, csr_col_ind, csr_val, x, b, res.data(), n);

    double initial_res_norm = norm_inf(res.data(), n);

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        ssor_iteration(csr_row_ptr, csr_col_ind, csr_val, x, b, n, omega);

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
    std::cout << "SSOR time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}

int ssor(const csr_matrix2& A, vector2& x, const vector2& b, double omega, iter_control control)
{
    if(!A.is_on_host() || !x.is_on_host() || !b.is_on_host())
    {
        std::cout << "Error: A matrix, x vector, and b vector must on host for jacobi iteration" << std::endl;
        return -1;
    }

    return ssor(A.get_row_ptr(), A.get_col_ind(), A.get_val(), x.get_vec(), b.get_vec(), A.get_m(), omega, control);
}








ssor_solver::ssor_solver(){}

ssor_solver::~ssor_solver(){}

void ssor_solver::build(const csr_matrix2& A)
{
    res.resize(A.get_m());
}

int ssor_solver::solve(const csr_matrix2& A, vector2& x, const vector2& b, iter_control control, double omega)
{
    ROUTINE_TRACE("ssor_solver::solve");

    // res = b - A * x
    compute_residual(A, x, b, res);

    double initial_res_norm = res.norm_inf2();

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        // SSOR iteration
        ssor_iteration(A.get_row_ptr(), A.get_col_ind(), A.get_val(), x.get_vec(), b.get_vec(), A.get_m(), omega);

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
    std::cout << "SSOR time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}
