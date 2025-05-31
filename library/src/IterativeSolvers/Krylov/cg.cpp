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
#include "../../../include/slaf.h"
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

cg_solver::cg_solver() : restart_iter(-1){}

cg_solver::~cg_solver(){}

void cg_solver::build(const csr_matrix& A)
{
    p.resize(A.get_m());
    z.resize(A.get_m());
    res.resize(A.get_m());
}

int cg_solver::solve_nonprecond(const csr_matrix& A, vector& x, const vector& b, iter_control control)
{
    ROUTINE_TRACE("cg_solver::solve_nonprecond");

    double gamma = 0.0;
    double initial_res_norm = 0.0;

    // start algorithm
    {
        // res = b - A * x
        compute_residual(A, x, b, res);

        initial_res_norm = res.norm_inf2();

        // p = res
        p.copy_from(res);

        gamma = res.dot(res);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        // restart algorithm to better handle round off error
        if (iter > 0 && iter % restart_iter == 0)
        {
            // res = b - A * x
            compute_residual(A, x, b, res);

            // p = res
            p.copy_from(res);

            gamma = res.dot(res);
        }

        // z = A * p and alpha = (r, r) / (A * p, p)
        A.multiply_vector(z, p);
        double alpha = gamma / z.dot(p);

        // update x = x + alpha * p
        axpy(alpha, p, x);

        // update res = res - alpha * z
        axpy(-alpha, z, res);

        double res_norm = res.norm_inf2();

        if (control.residual_converges(res_norm, initial_res_norm))
        {
            break;
        }

        // find beta
        double old_gamma = gamma;
        gamma = res.dot(res);
        double beta = gamma / old_gamma;

        // update p = res + beta * p
        axpby(1.0, res, beta, p);

        iter++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Conjugate Gradient time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}

int cg_solver::solve_precond(const csr_matrix& A, vector& x, const vector& b, const preconditioner* precond, iter_control control)
{
    ROUTINE_TRACE("cg_solver::solve_precond");

    assert(precond != nullptr);

    double gamma = 0.0;
    double initial_res_norm = 0.0;

    // start algorithm
    {
        // res = b - A * x
        compute_residual(A, x, b, res);
        
        initial_res_norm = res.norm_inf2();

        // z = (M^-1) * res
        precond->solve(res.get_vec(), z.get_vec(), A.get_m());

        // p = z
        p.copy_from(z);

        gamma = z.dot(res);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        // restart algorithm to better handle round off error
        if (restart_iter != -1 && iter > 0 && iter % restart_iter == 0)
        {
            // res = b - A * x
            compute_residual(A, x, b, res);

            // z = (M^-1) * res
            precond->solve(res.get_vec(), z.get_vec(), A.get_m());

            // p = z
            p.copy_from(z);

            gamma = z.dot(res);
        }

        // z = A * p and alpha = (z, r) / (Ap, p)
        A.multiply_vector(z, p);
        double alpha = gamma / z.dot(p);

        // update x = x + alpha * p
        axpy(alpha, p, x);

        // update res = res - alpha * z
        axpy(-alpha, z, res);

        double res_norm = res.norm_inf2();

        if (control.residual_converges(res_norm, initial_res_norm))
        {
            break;
        }

        // z = (M^-1) * res
        precond->solve(res.get_vec(), z.get_vec(), A.get_m());

        // find beta
        double old_gamma = gamma;
        gamma = z.dot(res);
        double beta = gamma / old_gamma;

        // update p = z + beta * p
        axpby(1.0, z, beta, p);

        iter++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Preconditioned Conjugate Gradient time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}

int cg_solver::solve(const csr_matrix& A, vector& x, const vector& b, const preconditioner* precond, iter_control control)
{
    ROUTINE_TRACE("cg_solver::solve");

    if(precond == nullptr)
    {
        return solve_nonprecond(A, x, b, control);
    }
    else
    {
        return solve_precond(A, x, b, precond, control);
    }
}