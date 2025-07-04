//********************************************************************************
//
// MIT License
//
// Copyright(c) 2019-2025 James Sandham
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

#include "../../../include/IterativeSolvers/Krylov/gmres.h"
#include "../../../include/IterativeSolvers/slaf.h"
#include "math.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <assert.h>

#include "../../trace.h"

//****************************************************************************
//
// Generalised Minimum Residual
//
//****************************************************************************

// A : n x n
// Q : n x (restart + 1)
// H : (restart + 1) x restart
// where k = 1....restart
static void arnoldi(const int* csr_row_ptr, const int* csr_col_ind, const double* csr_val, const preconditioner* precond, double* z, double* Q, double* H, int n, int k, int restart)
{
    ROUTINE_TRACE("arnoldi");

    // Column k-1 of Q matrix
    const double* qkm1 = &Q[(k - 1) * n];

    // Column k of Q matrix
    double* qk = &Q[k * n];

    if(precond != nullptr)
    {
        assert(z != nullptr);
        // z = A * q_(k-1)
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, qkm1, z, n);

        // qk = (M^-1) * z
        precond->solve(z, qk, n);
    }
    else
    {
        // q_k = A * q_(k-1)
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, qkm1, qk, n);
    }

    for (int i = 0; i < k; i++)
    {
        const double* qi = &Q[i * n];

        H[i + (k - 1) * (restart + 1)] = dot_product(qi, qk, n);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int j = 0; j < n; j++)
        {
            qk[j] = qk[j] - H[i + (k - 1) * (restart + 1)] * qi[j];
        }
    }

    double vv = dot_product(qk, qk, n);
    H[k + (k - 1) * (restart + 1)] = std::sqrt(vv);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static, 1024)
#endif
    for (int j = 0; j < n; j++)
    {
        qk[j] = qk[j] / H[k + (k - 1) * (restart + 1)];
    }
}

static void apply_givens_rotation(double c, double s, double* H, int i, int k, int restart)
{
    ROUTINE_TRACE("apply_givens_rotation");

    double temp1 = H[i + (k - 1) * (restart + 1)];
    double temp2 = H[i + 1 + (k - 1) * (restart + 1)];

    H[i + (k - 1) * (restart + 1)] = c * temp1 + s * temp2;
    H[i + 1 + (k - 1) * (restart + 1)] = -s * temp1 + c * temp2;
}

static void compute_givens_rotation(double* c, double* s, const double* H, int k, int restart)
{
    ROUTINE_TRACE("compute_givens_rotation");

    double xi = H[k - 1 + (k - 1) * (restart + 1)], xj = H[k + (k - 1) * (restart + 1)];

    if (xi == 0.0)
    {
        c[k - 1] = 0.0;
        s[k - 1] = 1.0;
    }
    else if(xj == 0.0)
    {
        c[k - 1] = 1.0;
        s[k - 1] = 0.0;
    }
    else if(std::abs(xi) > std::abs(xj))
    {
        double temp = xj / xi;
        c[k - 1] = 1.0 / std::sqrt(1.0 + temp * temp);
        s[k - 1] = temp * c[k - 1];
    }
    else
    {
        double temp = xi / xj;
        s[k - 1] = 1.0 / std::sqrt(1.0 + temp * temp);
        c[k - 1] = temp * s[k - 1];
    }
}

static int nonpreconditioned_gmres(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
    iter_control control, int restart)
{
    ROUTINE_TRACE("nonpreconditioned_gmres");

    restart = std::min(restart, n - 1);

    // create H and Q matrices (which are dense and stored as vectors columnwise)
    // H has size (restart + 1) x restart
    // Q has size n x (restart + 1)
    std::vector<double> H;
    std::vector<double> Q;
    std::vector<double> c;
    std::vector<double> s;
    H.resize((restart + 1) * restart);
    Q.resize(n * (restart + 1));
    c.resize(restart);
    s.resize(restart);

    std::vector<double> res(n);

    // res = b - A * x
    compute_residual(csr_row_ptr, csr_col_ind, csr_val, x, b, res.data(), n);

    double res_norm = norm_euclid(res.data(), n);
    double initial_res_norm = res_norm;

    // Check norm of residual against tolerance
    if(control.residual_converges(res_norm, initial_res_norm))
    {
        return 0;
    }

    for (int i = 0; i < n; i++)
    {
        Q[i] = res[i] / res_norm;
    }

    // Re-set residual vector to zero
    for (int i = 0; i < n; i++)
    {
        res[i] = 0.0;
    }
    res[0] = res_norm;

    auto t1 = std::chrono::high_resolution_clock::now();

    // gmres
    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        int k = 1;
        for(k = 1; k < restart + 1; k++)
        {
            // Arnoldi iteration
            arnoldi(csr_row_ptr, csr_col_ind, csr_val, nullptr, nullptr, Q.data(), H.data(), n, k, restart);

            // Solve least squares problem H(1:k+1,1:k) * y = sqrt(r'*r) * eye(k+1,1)
            // since H is hessenberg, use givens rotations
            
            // Apply previous cached givens rotations
            // Givens 2 by 2 rotation matrix:
            //  G =  [c s
            //       -s c]
            for (int i = 0; i < k - 1; i++)
            {
                apply_givens_rotation(c[i], s[i], H.data(), i, k, restart);
            }

            // calculate new givens rotation
            compute_givens_rotation(c.data(), s.data(), H.data(), k, restart);

            // apply newest givens rotation to eliminate off diagonal entry
            apply_givens_rotation(c[k - 1], s[k - 1], H.data(), k - 1, k, restart);

            // update residual vector
            res[k] = -s[k - 1] * res[k - 1];
            res[k - 1] = c[k - 1] * res[k - 1];

            if(control.residual_converges(std::abs(res[k]), initial_res_norm) || k == restart)
            {
                break;
            }
        }

        // std::cout << "H" << std::endl;
        // for(int i = 0; i < k + 1; i++)
        // {
        //     for(int j = 0; j < k; j++)
        //     {
        //         std::cout << H[(restart + 1) * j + i] << " ";
        //     }
        //     std::cout << "" << std::endl;
        // }
        // std::cout << "" << std::endl;

        // std::cout << "res" << std::endl;
        // for(size_t i = 0; i < res.size(); i++)
        // {
        //     std::cout << res[i] << " ";
        // }
        // std::cout << "" << std::endl;

        // backward solve
        for (int i = k - 1; i >= 0; i--)
        {
            for(int j = i + 1; j < k; j++)
            {
                res[i] = res[i] - H[i + (restart + 1) * j] * res[j];
            }

            res[i] = res[i] / H[i + (restart + 1) * i];
        }

        // update solution vector
        for (int j = 0; j < k; j++)
        {
            for (int i = 0; i < n; i++)
            {
                x[i] = x[i] + Q[j * n + i] * res[j];
            }
        }

        // res = b - A * x
        compute_residual(csr_row_ptr, csr_col_ind, csr_val, x, b, res.data(), n);

        res_norm = norm_euclid(res.data(), n);

        // Check norm of residual against tolerance
        if(control.residual_converges(res_norm, initial_res_norm))
        {
            break;
        }
        
        for (int i = 0; i < n; i++)
        {
            Q[i] = res[i] / res_norm;
        }

        // Re-set residual vector to zero
        for (int i = 0; i < n; i++)
        {
            res[i] = 0.0;
        }
        res[0] = res_norm;

        iter++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "GMRES time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}

static int preconditioned_gmres(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
    const preconditioner* precond, iter_control control, int restart)
{
    ROUTINE_TRACE("preconditioned_gmres");

    assert(precond != nullptr);

    restart = std::min(restart, n - 1);

    // create H and Q matrices (which are dense and stored as vectors columnwise)
    // H has size (restart + 1) x restart
    // Q has size n x (restart + 1)
    std::vector<double> H;
    std::vector<double> Q;
    std::vector<double> c;
    std::vector<double> s;
    H.resize((restart + 1) * restart);
    Q.resize(n * (restart + 1));
    c.resize(restart);
    s.resize(restart);

    std::vector<double> res(n);
    std::vector<double> z(n);

    // res = b - A * x
    compute_residual(csr_row_ptr, csr_col_ind, csr_val, x, b, res.data(), n);

    // z = (M^-1) * res
    precond->solve(res.data(), z.data(), n);

    double res_norm = norm_euclid(z.data(), n);
    double initial_res_norm = res_norm;

    // Check norm of residual against tolerance
    if(control.residual_converges(res_norm, initial_res_norm))
    {
        return 0;
    }

    for (int i = 0; i < n; i++)
    {
        Q[i] = z[i] / res_norm;
    }

    // Re-set res vector to zero
    for (int i = 0; i < n; i++)
    {
        res[i] = 0.0;
    }
    res[0] = res_norm;

    auto t1 = std::chrono::high_resolution_clock::now();

    // gmres
    int iter = 0;
    while (!control.exceed_max_iter(iter))
    {
        int k = 1;
        for(k = 1; k < restart + 1; k++)
        {
            // Arnoldi iteration
            arnoldi(csr_row_ptr, csr_col_ind, csr_val, precond, z.data(), Q.data(), H.data(), n, k, restart);

            // Solve least squares problem H(1:k+1,1:k) * y = sqrt(r'*r) * eye(k+1,1)
            // since H is hessenberg, use givens rotations
            
            // Apply previous cached givens rotations
            // Givens 2 by 2 rotation matrix:
            //  G =  [c s
            //       -s c]
            for (int i = 0; i < k - 1; i++)
            {
                apply_givens_rotation(c[i], s[i], H.data(), i, k, restart);
            }

            // calculate new givens rotation
            compute_givens_rotation(c.data(), s.data(), H.data(), k, restart);

            // apply newest givens rotation to eliminate off diagonal entry
            apply_givens_rotation(c[k - 1], s[k - 1], H.data(), k - 1, k, restart);

            // update residual vector
            res[k] = -s[k - 1] * res[k - 1];
            res[k - 1] = c[k - 1] * res[k - 1];

            if(control.residual_converges(std::abs(res[k]), initial_res_norm) || k == restart)
            {
                break;
            }
        }

        // std::cout << "H" << std::endl;
        // for(int i = 0; i < k + 1; i++)
        // {
        //     for(int j = 0; j < k; j++)
        //     {
        //         std::cout << H[(restart + 1) * j + i] << " ";
        //     }
        //     std::cout << "" << std::endl;
        // }
        // std::cout << "" << std::endl;

        // std::cout << "res" << std::endl;
        // for(size_t i = 0; i < res.size(); i++)
        // {
        //     std::cout << res[i] << " ";
        // }
        // std::cout << "" << std::endl;

        // backward solve
        for (int i = k - 1; i >= 0; i--)
        {
            for(int j = i + 1; j < k; j++)
            {
                res[i] = res[i] - H[i + (restart + 1) * j] * res[j];
            }

            res[i] = res[i] / H[i + (restart + 1) * i];
        }

        // update solution vector
        for (int j = 0; j < k; j++)
        {
            for (int i = 0; i < n; i++)
            {
                x[i] = x[i] + Q[j * n + i] * res[j];
            }
        }

        // res = b - A * x
        compute_residual(csr_row_ptr, csr_col_ind, csr_val, x, b, res.data(), n);

        // z = (M^-1) * res
        precond->solve(res.data(), z.data(), n);

        res_norm = norm_euclid(z.data(), n);

        // Check norm of residual against tolerance
        if(control.residual_converges(res_norm, initial_res_norm))
        {
            break;
        }
        
        for (int i = 0; i < n; i++)
        {
            Q[i] = z[i] / res_norm;
        }

        // Re-set residual vector to zero
        for (int i = 0; i < n; i++)
        {
            res[i] = 0.0;
        }
        res[0] = res_norm;

        iter++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "GMRES time: " << ms_double.count() << "ms" << std::endl;

    return iter;
}

//-------------------------------------------------------------------------------
// generalised minimum residual
//-------------------------------------------------------------------------------
int gmres(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
    const preconditioner *precond, iter_control control,  int restart)
{
    ROUTINE_TRACE("gmres");

    if(precond == nullptr)
    {
        return nonpreconditioned_gmres(csr_row_ptr, csr_col_ind, csr_val, x, b, n, control, restart);
    }
    else
    {
        return preconditioned_gmres(csr_row_ptr, csr_col_ind, csr_val, x, b, n, precond, control, restart);
    }
}