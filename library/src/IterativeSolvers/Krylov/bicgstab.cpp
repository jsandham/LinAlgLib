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

//****************************************************************************
//
// Stabilized Bi-Conjugate Gradient
//
//****************************************************************************

static int nonpreconditioned_bicgstab(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
             iter_control control)
{
    ROUTINE_TRACE("nonpreconditioned_bicgstab");

    // r = b - A * x and initial error
    std::vector<double> r(n);
    compute_residual(csr_row_ptr, csr_col_ind, csr_val, x, b, r.data(), n);

    double initial_res_norm = norm_inf(r.data(), n);
   
    // r0 = r
    std::vector<double> r0(n);
    copy(r0.data(), r.data(), n);

    double rho = dot_product(r0.data(), r.data(), n);

    // create p vector
    std::vector<double> p(n);

    // p = r
    copy(p.data(), r.data(), n);

    // create v, t vectors
    std::vector<double> v(n);
    std::vector<double> t(n);

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        // v = Ap
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, p.data(), v.data(), n);

        double alpha = rho / dot_product(r0.data(), v.data(), n);

        // r = r - alpha * v
        axpy(n, -alpha, v.data(), r.data());

        // t = A * r
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, r.data(), t.data(), n);

        double omega1 = dot_product(t.data(), r.data(), n);
        double omega2 = dot_product(t.data(), t.data(), n);

        if(omega1 == 0.0 || omega2 == 0.0)
        {
            // x = x + alpha * p
            axpy(n, alpha, p.data(), x);
            break;
        }
        double omega = omega1 / omega2;

        // x = x + alpha * p + omega * r
        for (int i = 0; i < n; i++)
        {
            x[i] = x[i] + alpha * p[i] + omega * r[i];
        }

        // r = r - omega * t
        axpy(n, -omega, t.data(), r.data());

        double res_norm = norm_inf(r.data(), n);

        if (control.residual_converges(res_norm, initial_res_norm))
        {
            break;
        }

        double rho_prev = rho;
        rho = dot_product(r0.data(), r.data(), n);
        double beta = (rho / rho_prev) * (alpha / omega);

        // p = r + beta * (p - omega * v)
        for (int i = 0; i < n; i++)
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

static int preconditioned_bicgstab(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
    const preconditioner *precond, iter_control control)
{
    ROUTINE_TRACE("preconditioned_bicgstab");

    assert(precond != nullptr);

    // r = b - A * x and initial error
    std::vector<double> r(n);
    compute_residual(csr_row_ptr, csr_col_ind, csr_val, x, b, r.data(), n);

    double initial_res_norm = norm_inf(r.data(), n);

    // r0 = r
    std::vector<double> r0(n);
    copy(r0.data(), r.data(), n);

    double rho = dot_product(r0.data(), r.data(), n);

    // create p vector
    std::vector<double> p(n);

    // p = r
    copy(p.data(), r.data(), n);

    // create v, t vectors
    std::vector<double> v(n);
    std::vector<double> t(n);

    std::vector<double> z(n);
    std::vector<double> q(n);

    // M*z = r
    precond->solve(r.data(), z.data(), n);

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        // q = A*z
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, z.data(), q.data(), n);

        double alpha = rho / dot_product(r0.data(), q.data(), n);

        // r = r - alpha * q
        axpy(n, -alpha, q.data(), r.data());

        // M * v = r
        precond->solve(r.data(), v.data(), n);

        // t = A * v
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, v.data(), t.data(), n);

        double omega1 = dot_product(t.data(), r.data(), n);
        double omega2 = dot_product(t.data(), t.data(), n);

        if(omega1 == 0.0 || omega2 == 0.0)
        {
            // x = x + alpha * p
            axpy(n, alpha, p.data(), x);
            break;
        }
        double omega = omega1 / omega2;

        // x = x + alpha * z + omega * v
        for (int i = 0; i < n; i++)
        {
            x[i] = x[i] + alpha * z[i] + omega * v[i];
        }

        // r = r - omega * t
        axpy(n, -omega, t.data(), r.data());

        double res_norm = norm_inf(r.data(), n);

        if (control.residual_converges(res_norm, initial_res_norm))
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

        // M * z = p
        precond->solve(p.data(), z.data(), n);

        iter++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Preconditioned BICGSTAB time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}

//-------------------------------------------------------------------------------
// stabilized bi-conjugate gradient
//-------------------------------------------------------------------------------
int bicgstab(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
    const preconditioner *precond, iter_control control)
{
    ROUTINE_TRACE("bicgstab");

    if(precond == nullptr)
    {
        return nonpreconditioned_bicgstab(csr_row_ptr, csr_col_ind, csr_val, x, b, n, control);
    }
    else
    {
        return preconditioned_bicgstab(csr_row_ptr, csr_col_ind, csr_val, x, b, n, precond, control);
    }
}

int bicgstab(const csr_matrix2& A, vector2& x, const vector2& b, const preconditioner *precond, iter_control control)
{
    if(!A.is_on_host() || !x.is_on_host() || !b.is_on_host())
    {
        std::cout << "Error: A matrix, x vector, and b vector must on host for jacobi iteration" << std::endl;
        return -1;
    }

    return bicgstab(A.get_row_ptr(), A.get_col_ind(), A.get_val(), x.get_vec(), b.get_vec(), A.get_m(), precond, control);
}




















bicgstab_solver::bicgstab_solver() : restart_iter(-1){}

bicgstab_solver::~bicgstab_solver(){}

void bicgstab_solver::build(const csr_matrix2& A)
{
    r.resize(A.get_m());
    r0.resize(A.get_m());
    p.resize(A.get_m());
    v.resize(A.get_m());
    t.resize(A.get_m());
    z.resize(A.get_m());
    q.resize(A.get_m());
}

int bicgstab_solver::solve_nonprecond(const csr_matrix2& A, vector2& x, const vector2& b, iter_control control)
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
        A.multiply_vector(v, p);

        double alpha = rho / r0.dot(v);

        // r = r - alpha * v
        axpy(-alpha, v, r);

        // t = A * r
        A.multiply_vector(t, r);

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

int bicgstab_solver::solve_precond(const csr_matrix2& A, vector2& x, const vector2& b, const preconditioner* precond, iter_control control)
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
    precond->solve(r.get_vec(), z.get_vec(), A.get_m());

    auto t1 = std::chrono::high_resolution_clock::now();

    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        // q = A*z
        A.multiply_vector(q, z);

        double alpha = rho / r0.dot(q);

        // r = r - alpha * q
        axpy(-alpha, q, r);

        // M * v = r
        precond->solve(r.get_vec(), v.get_vec(), A.get_m());

        // t = A * v
        A.multiply_vector(t, v);

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
        precond->solve(p.get_vec(), z.get_vec(), A.get_m());

        iter++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Preconditioned BICGSTAB time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}

int bicgstab_solver::solve(const csr_matrix2& A, vector2& x, const vector2& b, const preconditioner* precond, iter_control control)
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
