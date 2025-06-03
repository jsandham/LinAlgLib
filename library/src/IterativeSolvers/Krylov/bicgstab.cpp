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

#include "../../../include/IterativeSolvers/Krylov/bicgstab.h"
#include "../../../include/slaf.h"
#include "math.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <assert.h>

#include "../../trace.h"

using namespace linalg;

//****************************************************************************
//
// Stabilized Bi-Conjugate Gradient
//
//****************************************************************************

bicgstab_solver::bicgstab_solver() : restart_iter(-1){}

bicgstab_solver::~bicgstab_solver(){}

void bicgstab_solver::build(const csr_matrix& A)
{
    r.resize(A.get_m());
    r0.resize(A.get_m());
    p.resize(A.get_m());
    v.resize(A.get_m());
    t.resize(A.get_m());
    z.resize(A.get_m());
    q.resize(A.get_m());
}

int bicgstab_solver::solve_nonprecond(const csr_matrix& A, vector& x, const vector& b, iter_control control)
{
    ROUTINE_TRACE("bicgstab_solver::solve_nonprecond");

    // r = b - A * x and initial error
    compute_residual(A, x, b, r);

    double initial_res_norm = r.norm_inf2();
   
    // r0 = r
    r0.copy_from(r);

    double rho = r0.dot(r);

    // p = r
    p.copy_from(r);

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        // v = Ap
        A.multiply_by_vector(v, p);

        double alpha = rho / r0.dot(v);

        // r = r - alpha * v
        axpy(-alpha, v, r);

        // t = A * r
        A.multiply_by_vector(t, r);

        double omega1 = t.dot(r);
        double omega2 = t.dot(t);

        if(omega1 == 0.0 || omega2 == 0.0)
        {
            // x = x + alpha * p
            axpy(alpha, p, x);
            break;
        }
        double omega = omega1 / omega2;

        // x = x + alpha * p + omega * r
        for (int i = 0; i < A.get_m(); i++)
        {
            x[i] = x[i] + alpha * p[i] + omega * r[i];
        }

        // r = r - omega * t
        axpy(-omega, t, r);

        double res_norm = r.norm_inf2();

        if (control.residual_converges(res_norm, initial_res_norm))
        {
            break;
        }

        double rho_prev = rho;
        rho = r0.dot(r);
        double beta = (rho / rho_prev) * (alpha / omega);

        // p = r + beta * (p - omega * v)
        for (int i = 0; i < A.get_m(); i++)
        {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        iter++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "BICGSTAB time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}

int bicgstab_solver::solve_precond(const csr_matrix& A, vector& x, const vector& b, const preconditioner* precond, iter_control control)
{
    ROUTINE_TRACE("bicgstab_solver::solve_precond");

    assert(precond != nullptr);

    // r = b - A * x and initial error
    compute_residual(A, x, b, r);

    double initial_res_norm = r.norm_inf2();

    // r0 = r
    r0.copy_from(r);

    double rho = r0.dot(r);

    // p = r
    p.copy_from(r);

    // M*z = r
    precond->solve(r, z);

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        // q = A*z
        A.multiply_by_vector(q, z);

        double alpha = rho / r0.dot(q);

        // r = r - alpha * q
        axpy(-alpha, q, r);

        // M * v = r
        precond->solve(r, v);

        // t = A * v
        A.multiply_by_vector(t, v);

        double omega1 = t.dot(r);
        double omega2 = t.dot(t);

        if(omega1 == 0.0 || omega2 == 0.0)
        {
            // x = x + alpha * p
            axpy(alpha, p, x);
            break;
        }
        double omega = omega1 / omega2;

        // x = x + alpha * z + omega * v
        for (int i = 0; i < A.get_m(); i++)
        {
            x[i] = x[i] + alpha * z[i] + omega * v[i];
        }

        // r = r - omega * t
        axpy(-omega, t, r);

        double res_norm = r.norm_inf2();

        if (control.residual_converges(res_norm, initial_res_norm))
        {
            break;
        }

        double rho_prev = rho;
        rho = r0.dot(r);
        double beta = (rho / rho_prev) * (alpha / omega);

        // p = r + beta * (p - omega * q)
        for (int i = 0; i < A.get_m(); i++)
        {
            p[i] = r[i] + beta * (p[i] - omega * q[i]);
        }

        // M * z = p
        precond->solve(p, z);

        iter++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Preconditioned BICGSTAB time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}

int bicgstab_solver::solve(const csr_matrix& A, vector& x, const vector& b, const preconditioner* precond, iter_control control)
{
    ROUTINE_TRACE("bicgstab_solver::solve");

    if(precond == nullptr)
    {
        return solve_nonprecond(A, x, b, control);
    }
    else
    {
        return solve_precond(A, x, b, precond, control);
    }
}
