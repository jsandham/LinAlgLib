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

#include "../../../include/LinearSolvers/Krylov/cg.h"
#include "../../../include/LinearSolvers/slaf.h"
#include "math.h"
#include <iostream>
#include <vector>

//****************************************************************************
//
// Conjugate Gradient
//
//****************************************************************************

#define DEBUG 1

//-------------------------------------------------------------------------------
// Conjugate gradient
//-------------------------------------------------------------------------------
int cg(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
       double tol, int max_iter, int restart_iter)
{
    // create z and p vector
    std::vector<double> z(n);
    std::vector<double> p(n);

    // res = b - A * x and initial error
    std::vector<double> res(n);

    double gamma = 0.0;

    // start algorithm
    {
        // res = b - A * x
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, x, res.data(), n);
        for (int i = 0; i < n; i++)
        {
            res[i] = b[i] - res[i];
        }

        // p = res
        for (int i = 0; i < n; i++)
        {
            p[i] = res[i];
        }

        gamma = dot_product(res.data(), res.data(), n);
    }

    int iter = 0;
    while (iter < max_iter)
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

            // p = res
            for (int i = 0; i < n; i++)
            {
                p[i] = res[i];
            }

            gamma = dot_product(res.data(), res.data(), n);
        }

        // z = A * p and alpha = (r, r) / (Ap, p)
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

        double err = norm_inf(res.data(), n);

#if (DEBUG)
        std::cout << "error: " << err << std::endl;
#endif

        if (err <= tol)
        {
            break;
        }

        // find beta
        double old_gamma = gamma;
        gamma = dot_product(res.data(), res.data(), n);
        double beta = gamma / old_gamma;

        // update p = res + beta * p
        for (int i = 0; i < n; i++)
        {
            p[i] = res[i] + beta * p[i];
        }

        iter++;
    }

    return iter;
}