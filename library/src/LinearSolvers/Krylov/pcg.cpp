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

#include "../../../include/LinearSolvers/Krylov/pcg.h"
#include "../../../include/LinearSolvers/slaf.h"
#include "math.h"
#include <iostream>
#include <vector>
#include <chrono>

//****************************************************************************
//
// Preconditioned Conjugate Gradient
//
//****************************************************************************

//-------------------------------------------------------------------------------
// preconditioned conjugate gradient
//-------------------------------------------------------------------------------
int pcg(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
        const preconditioner *precond, iter_control control, int restart_iter)
{
    // create z and p vector
    std::vector<double> z(n);
    std::vector<double> p(n);

    // res = b - A * x and initial error
    std::vector<double> res(n);

    double gamma = 0.0;
    double initial_res_norm = 0.0;

    // start algorithm
    {
        // res = b - A * x
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, x, res.data(), n);
        for (int i = 0; i < n; i++)
        {
            res[i] = b[i] - res[i];
        }
        
        initial_res_norm = norm_inf(res.data(), n);

        // z = (M^-1) * res
        precond->solve(res.data(), z.data(), n);

        // p = z
        for (int i = 0; i < n; i++)
        {
            p[i] = z[i];
        }

        gamma = dot_product(z.data(), res.data(), n);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        // restart algorithm to better handle round off error
        if (iter > 0 && iter % restart_iter == 0)
        {
            // res = b - A * x
            matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, x, res.data(), n);
            for (int i = 0; i < n; i++)
            {
                res[i] = b[i] - res[i];
            }

            // z = (M^-1) * res
            precond->solve(res.data(), z.data(), n);

            // p = z
            for (int i = 0; i < n; i++)
            {
                p[i] = z[i];
            }

            gamma = dot_product(z.data(), res.data(), n);
        }

        // z = A * p and alpha = (z, r) / (Ap, p)
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, p.data(), z.data(), n);
        double alpha = gamma / dot_product(z.data(), p.data(), n);

        // update x = x + alpha * p
        for (int i = 0; i < n; i++)
        {
            x[i] += alpha * p[i];
        }

        // update res = res - alpha * z
        for (int i = 0; i < n; i++)
        {
            res[i] -= alpha * z[i];
        }

        double res_norm = norm_inf(res.data(), n);

        if (control.residual_converges(res_norm, initial_res_norm))
        {
            break;
        }

        // z = (M^-1) * res
        precond->solve(res.data(), z.data(), n);

        // find beta
        double old_gamma = gamma;
        gamma = dot_product(z.data(), res.data(), n);
        double beta = gamma / old_gamma;

        // update p = z + beta * p
        for (int i = 0; i < n; i++)
        {
            p[i] = z[i] + beta * p[i];
        }

        iter++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Preconditioned Conjugate Gradient time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}
