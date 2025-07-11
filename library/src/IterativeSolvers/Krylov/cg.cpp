//********************************************************************************
//
// MIT License
//
// Copyright(c) 2024-2025 James Sandham
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

#include "../../../include/IterativeSolvers/Krylov/cg.h"
#include "../../../include/IterativeSolvers/slaf.h"
#include "math.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <assert.h>

#include "../../trace.h"

//****************************************************************************
//
// Conjugate Gradient
//
//****************************************************************************

static int nonpreconditioned_conjugate_gradient(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
       iter_control control, int restart_iter)
{
    ROUTINE_TRACE("nonpreconditioned_conjugate_gradient");

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
        compute_residual(csr_row_ptr, csr_col_ind, csr_val, x, b, res.data(), n);

        initial_res_norm = norm_inf(res.data(), n);

        // p = res
        copy(p.data(), res.data(), n);

        gamma = dot_product(res.data(), res.data(), n);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        // restart algorithm to better handle round off error
        if (iter > 0 && iter % restart_iter == 0)
        {
            // res = b - A * x
            compute_residual(csr_row_ptr, csr_col_ind, csr_val, x, b, res.data(), n);

            // p = res
            copy(p.data(), res.data(), n);

            gamma = dot_product(res.data(), res.data(), n);
        }

        // z = A * p and alpha = (r, r) / (A * p, p)
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, p.data(), z.data(), n);
        double alpha = gamma / dot_product(z.data(), p.data(), n);

        // update x = x + alpha * p
        axpy(n, alpha, p.data(), x);

        // update res = res - alpha * z
        axpy(n, -alpha, z.data(), res.data());

        double res_norm = norm_inf(res.data(), n);

        if (control.residual_converges(res_norm, initial_res_norm))
        {
            break;
        }

        // find beta
        double old_gamma = gamma;
        gamma = dot_product(res.data(), res.data(), n);
        double beta = gamma / old_gamma;

        // update p = res + beta * p
        axpby(n, 1.0, res.data(), beta, p.data());

        iter++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Conjugate Gradient time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}

static int preconditioned_conjugate_gradient(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
    const preconditioner *precond, iter_control control, int restart_iter)
{
    ROUTINE_TRACE("preconditioned_conjugate_gradient");

    assert(precond != nullptr);

    // create z and p vector
    std::vector<double> z(n);
    std::vector<double> p(n);

    // res = b - A * x
    std::vector<double> res(n);

    double gamma = 0.0;
    double initial_res_norm = 0.0;

    // start algorithm
    {
        // res = b - A * x
        compute_residual(csr_row_ptr, csr_col_ind, csr_val, x, b, res.data(), n);
        
        initial_res_norm = norm_inf(res.data(), n);

        // z = (M^-1) * res
        precond->solve(res.data(), z.data(), n);

        // p = z
        copy(p.data(), z.data(), n);

        gamma = dot_product(z.data(), res.data(), n);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        // restart algorithm to better handle round off error
        if (restart_iter != -1 && iter > 0 && iter % restart_iter == 0)
        {
            // res = b - A * x
            compute_residual(csr_row_ptr, csr_col_ind, csr_val, x, b, res.data(), n);

            // z = (M^-1) * res
            precond->solve(res.data(), z.data(), n);

            // p = z
            copy(p.data(), z.data(), n);

            gamma = dot_product(z.data(), res.data(), n);
        }

        // z = A * p and alpha = (z, r) / (Ap, p)
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, p.data(), z.data(), n);
        double alpha = gamma / dot_product(z.data(), p.data(), n);

        // update x = x + alpha * p
        axpy(n, alpha, p.data(), x);

        // update res = res - alpha * z
        axpy(n, -alpha, z.data(), res.data());

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
        axpby(n, 1.0, z.data(), beta, p.data());

        iter++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Preconditioned Conjugate Gradient time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}

//-------------------------------------------------------------------------------
// conjugate gradient
//-------------------------------------------------------------------------------
int cg(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
    const preconditioner *precond, iter_control control, int restart_iter)
{
    ROUTINE_TRACE("cg");

    if(precond == nullptr)
    {
        return nonpreconditioned_conjugate_gradient(csr_row_ptr, csr_col_ind, csr_val, x, b, n, control, restart_iter);
    }
    else
    {
        return preconditioned_conjugate_gradient(csr_row_ptr, csr_col_ind, csr_val, x, b, n, precond, control, restart_iter);
    }
}