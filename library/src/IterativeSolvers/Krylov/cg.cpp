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

int cg(const csr_matrix2& A, vector2& x, const vector2& b, const preconditioner *precond, iter_control control, int restart_iter)
{
    if(!A.is_on_host() || !x.is_on_host() || !b.is_on_host())
    {
        std::cout << "Error: A matrix, x vector, and b vector must on host for jacobi iteration" << std::endl;
        return -1;
    }

    return cg(A.get_row_ptr(), A.get_col_ind(), A.get_val(), x.get_vec(), b.get_vec(), A.get_m(), precond, control, restart_iter);
}



























cg_solver::cg_solver() : restart_iter(-1){}

cg_solver::~cg_solver(){}

void cg_solver::build(const csr_matrix2& A)
{
    p.resize(A.get_m());
    z.resize(A.get_m());
    res.resize(A.get_m());
}

int cg_solver::solve_nonprecond(const csr_matrix2& A, vector2& x, const vector2& b, iter_control control)
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

int cg_solver::solve_precond(const csr_matrix2& A, vector2& x, const vector2& b, const preconditioner* precond, iter_control control)
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

int cg_solver::solve(const csr_matrix2& A, vector2& x, const vector2& b, const preconditioner* precond, iter_control control)
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