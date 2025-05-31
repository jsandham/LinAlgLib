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

#include "../../../include/IterativeSolvers/Classic/sor.h"
#include "../../../include/slaf.h"
#include <iostream>
#include <chrono>
#include <vector>

#include "../../trace.h"

//-------------------------------------------------------------------------------
// successive over-relaxation method
//-------------------------------------------------------------------------------
void sor_iteration(const csr_matrix& A, vector& x, const vector& b, double omega)
{
    ROUTINE_TRACE("sor_iteration");

    const int* csr_row_ptr = A.get_row_ptr();
    const int* csr_col_ind = A.get_col_ind();
    const double* csr_val = A.get_val();

    double* x_ptr = x.get_vec();
    const double* b_ptr = b.get_vec();

    for (int j = 0; j < A.get_m(); j++)
    {
        double sigma = 0.0;
        double ajj = 0.0; // diagonal entry a_jj

        int row_start = csr_row_ptr[j];
        int row_end = csr_row_ptr[j + 1];

        for (int k = row_start; k < row_end; k++)
        {
            int col = csr_col_ind[k];
            double val = csr_val[k];

            if (col != j)
            {
                sigma = sigma + val * x_ptr[col];
            }
            else
            {
                ajj = val;
            }
        }
    
        x_ptr[j] = x_ptr[j] + omega * ((b_ptr[j] - sigma) / ajj - x_ptr[j]);
    }
}

sor_solver::sor_solver(){}

sor_solver::~sor_solver(){}

void sor_solver::build(const csr_matrix& A)
{
    res.resize(A.get_m());
}

int sor_solver::solve(const csr_matrix& A, vector& x, const vector& b, iter_control control, double omega)
{
    ROUTINE_TRACE("sor_solver::solve");

    // res = b - A * x
    compute_residual(A, x, b, res);

    double initial_res_norm = res.norm_inf2();

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        // SOR iteration
        sor_iteration(A, x, b, omega);

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
    std::cout << "SOR time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}
