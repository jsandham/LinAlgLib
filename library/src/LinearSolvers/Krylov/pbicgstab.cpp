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

#include "../../../include/LinearSolvers/Krylov/pbicgstab.h"
#include "../../../include/LinearSolvers/slaf.h"
#include "math.h"
#include <iostream>
#include <vector>

//****************************************************************************
//
// Preconditioned Stabilized Bi-Conjugate Gradient
//
//****************************************************************************

#define DEBUG 1

//-------------------------------------------------------------------------------
// preconditioned stabilized bi-conjugate gradient
//-------------------------------------------------------------------------------
int pbicgstab(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
             const preconditioner *precond, double tol, int max_iter)
{
    // r = b - A * x and initial error
    std::vector<double> r(n);
    matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, x, r.data(), n);
    for (int i = 0; i < n; i++)
    {
        r[i] = b[i] - r[i];
    }
   
    // r0 = r
    std::vector<double> r0(n);
    for (int i = 0; i < n; i++)
    {
        r0[i] = r[i];
    }

    double rho = dot_product(r0.data(), r.data(), n);

    // create p vector
    std::vector<double> p(n);

    // p = r
    for (int i = 0; i < n; i++)
    {
        p[i] = r[i];
    }

    // create v, t vectors
    std::vector<double> v(n);
    std::vector<double> t(n);

    std::vector<double> z(n);
    std::vector<double> q(n);
    
    // M*z = r
    precond->solve(r.data(), z.data(), n);

    int iter = 0;
    while (iter < max_iter)
    {
        // q = A*z
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, z.data(), q.data(), n);

        double alpha = rho / dot_product(r0.data(), q.data(), n);

        // r = r - alpha * q
        for (int i = 0; i < n; i++)
        {
            r[i] = r[i] - alpha * q[i];
        }

        // M*v = r
        precond->solve(r.data(), v.data(), n);

        // t = A*v
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, v.data(), t.data(), n);

        double omega1 = dot_product(t.data(), r.data(), n);
        double omega2 = dot_product(t.data(), t.data(), n);

        if(omega1 == 0.0 || omega2 == 0.0)
        {
            for (int i = 0; i < n; i++)
            {
                x[i] =  x[i] + alpha * p[i];
            }
            break;
        }
        double omega = omega1 / omega2;

        // x = x + alpha * z + omega * v
        for (int i = 0; i < n; i++)
        {
            x[i] = x[i] + alpha * z[i] + omega * v[i];
        }

        // r = r - omega * t
        for (int i = 0; i < n; i++)
        {
            r[i] = r[i] - omega * t[i];
        }

        double err = norm_inf(r.data(), n);
#if (DEBUG)
        std::cout << "error: " << err << std::endl;
#endif
        if (err < tol)
        {
            break;
        }

        double rho_prev = rho;
        rho = dot_product(r0.data(), r.data(), n);
        double beta = (rho / rho_prev) * (alpha / omega);

        // p = r + beta * (p - omega * q)
        for (int i = 0; i < n; i++)
        {
            p[i] = r[i] + beta * (p[i] - omega * q[i]);
        }

        // M*z = p
        precond->solve(p.data(), z.data(), n);

        iter++;
    }

    return iter;
}