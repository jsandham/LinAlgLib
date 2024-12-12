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
// The above copyright noticeand this permission notice shall be included in all
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

#include "../../../include/LinearSolvers/Krylov/bicgstab.h"
#include "../../../include/LinearSolvers/slaf.h"
#include "math.h"
#include <iostream>

//****************************************************************************
//
// Stabilized Bi-Conjugate Gradient
//
//****************************************************************************

#define DEBUG 1

//-------------------------------------------------------------------------------
// stabilized bi-conjugate gradient
//-------------------------------------------------------------------------------
int bicgstab(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, const int n,
        const double tol, const int max_iter)
{
    // r = b - A * x and initial error
    double *r = new double[n];
    matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, x, r, n);
    for (int i = 0; i < n; i++)
    {
        r[i] = b[i] - r[i];
    }
    double err = error(csr_row_ptr, csr_col_ind, csr_val, x, b, n);

    double *r0 = new double[n];
    for (int i = 0; i < n; i++)
    {
        r0[i] = r[i];
    }

    double rho = dot_product(r0, r, n);

    // create p vector
    double *p = new double[n];

    // p = r
    for (int i = 0; i < n; i++)
    {
        p[i] = r[i];
    }

    // create v, h, s, t vectors
    double *v = new double[n];
    double *h = new double[n];
    double *s = new double[n];
    double *t = new double[n];    

    int iter = 0;
    while (iter < max_iter && err > tol)
    {
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, p, v, n);

        double alpha = rho / dot_product(r0, v, n);

        for(int i = 0; i < n; i++)
        {
            h[i] = x[i] + alpha * p[i];
            s[i] = r[i] - alpha * v[i];
        }

        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, s, t, n);

        double omega = dot_product(t, s, n) / dot_product(t, t, n);

        for(int i = 0; i < n; i++)
        {
            x[i] = h[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }

        // if norm(r) < tol
        //     break;

        double rho_prev = rho;
        rho = dot_product(r0, r, n);
        double beta = (rho / rho_prev) / (alpha / omega);

        for(int i = 0; i < n; i++)
        {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // calculate error
        err = error(csr_row_ptr, csr_col_ind, csr_val, x, b, n);
#if (DEBUG)
        std::cout << "error: " << err << std::endl;
#endif
        iter++;
    }

    delete[] r;
    delete[] r0;
    delete[] p;
    delete[] v;
    delete[] h;
    delete[] s;
    delete[] t;

    return iter;
}