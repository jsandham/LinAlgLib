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

#include "../../../include/IterativeSolvers/Classic/jacobi.h"
#include "../../../include/linalg_math.h"

#include <iostream>
#include <vector>
#include <chrono>

#include "../../trace.h"

using namespace linalg;

//-------------------------------------------------------------------------------
// jacobi method
//-------------------------------------------------------------------------------
namespace linalg
{
void jacobi_iteration(const csr_matrix& A, vector<double>& x, const vector<double>& xold, const vector<double>& b)
{
    ROUTINE_TRACE("jacobi_iteration");

    const int* csr_row_ptr = A.get_row_ptr();
    const int* csr_col_ind = A.get_col_ind();
    const double* csr_val = A.get_val();

    double* x_ptr = x.get_vec();
    const double* x_old_ptr = xold.get_vec();
    const double* b_ptr = b.get_vec();

    for (int j = 0; j < A.get_m(); j++)
    {
        double sigma = 0.0;
        double ajj = 0.0; // diagonal entry a_jj
        for (int k = csr_row_ptr[j]; k < csr_row_ptr[j + 1]; k++)
        {
            if (csr_col_ind[k] != j)
            {
                sigma = sigma + csr_val[k] * x_old_ptr[csr_col_ind[k]];
            }
            else
            {
                ajj = csr_val[k];
            }
        }
        x_ptr[j] = (b_ptr[j] - sigma) / ajj;
    }
}
}

jacobi_solver::jacobi_solver(){}

jacobi_solver::~jacobi_solver(){}

void jacobi_solver::build(const csr_matrix& A)
{
    xold.resize(A.get_m());
    res.resize(A.get_m());
}

int jacobi_solver::solve(const csr_matrix& A, vector<double>& x, const vector<double>& b, iter_control control)
{
    ROUTINE_TRACE("jacobi_solver::solve");

    // res = b - A * x
    compute_residual(A, x, b, res);

    double initial_res_norm = norm_inf(res);

    // copy of x
    xold.copy_from(x);

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        // Jacobi iteration
        jacobi_iteration(A, x, xold, b);

        compute_residual(A, x, b, res);

        double res_norm = norm_inf(res);

        if (control.residual_converges(res_norm, initial_res_norm))
        {
            break;
        }

        xold.copy_from(x);

        iter++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Jacobi time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}