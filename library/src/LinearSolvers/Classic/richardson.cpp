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

#include "../../../include/LinearSolvers/Classic/richardson.h"
#include "../../../include/LinearSolvers/slaf.h"
#include <iostream>
#include <math.h>
#include <vector>
#include <chrono>

//********************************************************************************
//
// Richardson Iteration
//
//********************************************************************************

#define DEBUG 1

//-------------------------------------------------------------------------------
// richardson method
//-------------------------------------------------------------------------------
double richardson_iteration(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x,
                            double *res, const double *b, int n, double theta)
{
    double err = 0.0;

    // find res = A*x
    matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, x, res, n);

    // update approximation
    for (int j = 0; j < n; j++)
    {
        double xold = x[j];
        x[j] = x[j] + theta * (b[j] - res[j]);
        err = std::max(err, std::abs((x[j] - xold) / x[j]) * 100);
    }

    return err;
}

int rich(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
         double theta, double tol, int max_iter)
{
    // res = b - A * x and initial error
    std::vector<double> res(n);
    matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, x, res.data(), n);

    for (int i = 0; i < n; i++)
    {
        res[i] = b[i] - res[i];
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (iter < max_iter)
    {
        double err = richardson_iteration(csr_row_ptr, csr_col_ind, csr_val, x, res.data(), b, n, theta);

#if (DEBUG)
        std::cout << "error: " << err << std::endl;
#endif

        if (err <= tol)
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
