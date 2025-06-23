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

#include "../../../include/iterative_solvers/amg/amg.h"
#include "../../../include/linalg_math.h"

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <chrono>

#include "../../trace.h"

namespace linalg
{
void jacobi_iteration(const csr_matrix& A, vector<double>& x, const vector<double>& xold, const vector<double>& b);
void gauss_seidel_iteration(const csr_matrix& A, vector<double>& x, const vector<double>& b);
void symm_gauss_seidel_iteration(const csr_matrix& A, vector<double>& x, const vector<double>& b);
void sor_iteration(const csr_matrix& A, vector<double>& x, const vector<double>& b, const double omega);
void ssor_iteration(const csr_matrix& A, vector<double>& x, const vector<double>& b, const double omega);

static void apply_smoother(const csr_matrix& A, vector<double>& x, const vector<double>& b, Smoother smoother)
{
    ROUTINE_TRACE("apply_smoother");

    switch (smoother)
    {
    case Smoother::Jacobi: 
    {
        vector<double> xold(A.get_m());
        xold.copy_from(x);
       
        jacobi_iteration(A, x, xold, b);
        break;
    }
    case Smoother::Gauss_Seidel:
        gauss_seidel_iteration(A, x, b);
        break;
    case Smoother::Symm_Gauss_Seidel:
        gauss_seidel_iteration(A, x, b);
        break;
    case Smoother::SOR:
        sor_iteration(A, x, b, 0.5f);
        break;
    case Smoother::SSOR:
        ssor_iteration(A, x, b, 0.5f);
        break;
    default:
        break;
    }
}

static void vcycle(const hierarchy &hierarchy, vector<double>& x, const vector<double>& b, int n1, int n2, int currentLevel,
                   Smoother smoother)
{
    ROUTINE_TRACE("vcycle");

    // A_coarse = R*A*P
    const csr_matrix &A = hierarchy.A_cs[currentLevel];

    int N = A.get_m(); // size of A at current level

    if (currentLevel < hierarchy.total_levels)
    {
        const csr_matrix &R = hierarchy.restrictions[currentLevel];
        const csr_matrix &P = hierarchy.prolongations[currentLevel];
        const csr_matrix &A_coarse = hierarchy.A_cs[currentLevel + 1];

        int Nc = A_coarse.get_m(); // size of A at next course level

        assert(Nc == R.get_m());
        assert(N == R.get_n());
        assert(N == P.get_m());
        assert(Nc == P.get_n());

        // do n1 smoothing steps on A*x=b
        for (int i = 0; i < n1; i++)
        {
            apply_smoother(A, x, b, smoother);
        }

        // compute residual r = b - A*x = A*e
        vector<double> r(N);
        compute_residual(A, x, b, r);

        // compute wres = R*r
        vector<double> wres(Nc);
        wres.zeros();
        R.multiply_by_vector(wres, r);

        // set e = 0
        vector<double> ec(Nc);
        ec.zeros();

        // recursively solve Ac*ec = R*r = wres
        vcycle(hierarchy, ec, wres, n1, n2, currentLevel + 1, smoother);

        // correct x = x + P*ec
        P.multiply_by_vector_and_add(x, ec);

        // do n2 smoothing steps on A*x=b
        for (int i = 0; i < n2; i++)
        {
            apply_smoother(A, x, b, smoother);
        }
    }
    else
    {
        //std::cout << "coarsest level matrix size: A.m: " << A.m << "A.nnz: " << A.nnz << " N: " << N << std::endl;

        const int* csr_row_ptr_A = A.get_row_ptr();
        const int* csr_col_ind_A = A.get_col_ind();
        const double* csr_val_A = A.get_val();

        if(A.get_m() == 1)
        {
            assert(A.get_nnz() == 1);

            x[0] = b[0] / csr_val_A[0];
        }
        // else if(A.m == 2)
        // {
        //     assert(A.nnz == 4);
        //     // A = | a00 a01 |
        //     //     | a10 a11 |
        //     //
        //     // A^-1 = (1 / det) | a11 -a01 |
        //     //                  |-a10  a00 |
        //     double a00 = A.csr_val[0];
        //     double a01 = A.csr_val[1];
        //     double a10 = A.csr_val[2];
        //     double a11 = A.csr_val[3];

        //     double det = a00 * a11 - a01 * a10;

        //     x[0] = (1.0 / det) * (a11 * b[0] - a01 * b[1]);
        //     x[1] = (1.0 / det) * (-a10 * b[0] + a00 * b[1]);
        // }
        else
        {
            //  solve A*x=b exactly
            for (int i = 0; i < 1000; i++)
            {
                // Gauss-Seidel iteration
                for (int j = 0; j < N; j++)
                {
                    double sigma = 0.0;
                    double ajj = 0.0; // diagonal entry a_jj

                    int row_start = csr_row_ptr_A[j];
                    int row_end = csr_row_ptr_A[j + 1];

                    for (int k = row_start; k < row_end; k++)
                    {
                        int col = csr_col_ind_A[k];
                        double val = csr_val_A[k];
                        if (col != j)
                        {
                            sigma = sigma + val * x[col];
                        }
                        else
                        {
                            ajj = val;
                        }
                    }
                    x[j] = (b[j] - sigma) / ajj;
                }
            }
        }
    }
}

static void wcycle(const hierarchy &hierarchy, vector<double>& x, const vector<double>& b, int n1, int n2, int n3, int currentLevel,
                   Smoother smoother)
{
    ROUTINE_TRACE("wcycle");

    // A_coarse = R*A*P
    const csr_matrix &A = hierarchy.A_cs[currentLevel];

    int N = A.get_m(); // size of A at current level

    if (currentLevel < hierarchy.total_levels)
    {
        const csr_matrix &R = hierarchy.restrictions[currentLevel];
        const csr_matrix &P = hierarchy.prolongations[currentLevel];
        const csr_matrix &A_coarse = hierarchy.A_cs[currentLevel + 1];

        int Nc = A_coarse.get_m(); // size of A at next course level

        assert(Nc == R.get_m());
        assert(N == R.get_n());
        assert(N == P.get_m());
        assert(Nc == P.get_n());

        // do n1 smoothing steps on A*x=b
        for (int i = 0; i < n1; i++)
        {
            apply_smoother(A, x, b, smoother);
        }

        // compute residual r = b - A*x = A*e
        vector<double> r(N);
        compute_residual(A, x, b, r);

        // compute wres = R*r
        vector<double> wres(Nc);
        wres.zeros();
        R.multiply_by_vector(wres, r);

        // set ec = 0
        vector<double> ec(Nc);
        ec.zeros();

        // recursively solve Ac*ec = R*r = wres
        // std::cout << "currentLevel: " << currentLevel << " N: " << N <<
        // std::endl;
        wcycle(hierarchy, ec, wres, n1, n2, n3, currentLevel + 1, smoother);
        // std::cout << "currentLevel: " << currentLevel << " N: " << N <<
        // std::endl;

        // correct x = x + P*ec
        P.multiply_by_vector_and_add(x, ec);

        // do n2 smoothing steps on A*x=b
        for (int i = 0; i < n2; i++)
        {
            apply_smoother(A, x, b, smoother);
        }

        // compute residual r = b - A*x = A*e
        compute_residual(A, x, b, r);

        // compute wres = R*r
        R.multiply_by_vector(wres, r);

        // recursively solve Ac*ec = R*r = wres
        wcycle(hierarchy, ec, wres, n1, n2, n3, currentLevel + 1, smoother);
        // std::cout << "currentLevel: " << currentLevel << " N: " << N <<
        // std::endl;

        // correct x = x + P*ec
        P.multiply_by_vector_and_add(x, ec);

        // do n3 smoothing steps on A*x=b
        for (int i = 0; i < n3; i++)
        {
            apply_smoother(A, x, b, smoother);
        }
    }
    else
    {
        const int* csr_row_ptr_A = A.get_row_ptr();
        const int* csr_col_ind_A = A.get_col_ind();
        const double* csr_val_A = A.get_val();

        // std::cout << "currentLevel: " << currentLevel << " N: " << N <<
        // std::endl;
        //  solve A*x=b exactly
        for (int i = 0; i < 100; i++)
        {
            // Gauss-Seidel iteration
            for (int j = 0; j < N; j++)
            {
                double sigma = 0.0;
                double ajj = 0.0; // diagonal entry a_jj

                int row_start = csr_row_ptr_A[j];
                int row_end = csr_row_ptr_A[j + 1];

                for (int k = row_start; k < row_end; k++)
                {
                    int col = csr_col_ind_A[k];
                    double val = csr_val_A[k];
                    if (col != j)
                    {
                        sigma = sigma + val * x[col];
                    }
                    else
                    {
                        ajj = val;
                    }
                }
                x[j] = (b[j] - sigma) / ajj;
            }
        }
    }
}

static void fcycle(const hierarchy &hierarchy, vector<double>& x, const vector<double>& b, int n1, int n2, int n3, int currentLevel,
                   Smoother smoother)
{
    ROUTINE_TRACE("fcycle");

    // A_coarse = R*A*P
    const csr_matrix &A = hierarchy.A_cs[currentLevel];

    int N = A.get_m(); // size of A at current level

    if (currentLevel < hierarchy.total_levels)
    {
        const csr_matrix &R = hierarchy.restrictions[currentLevel];
        const csr_matrix &P = hierarchy.prolongations[currentLevel];
        const csr_matrix &A_coarse = hierarchy.A_cs[currentLevel + 1];

        int Nc = A_coarse.get_m(); // size of A at next course level

        assert(Nc == R.get_m());
        assert(N == R.get_n());
        assert(N == P.get_m());
        assert(Nc == P.get_n());

        // do n1 smoothing steps on A*x=b
        for (int i = 0; i < n1; i++)
        {
            apply_smoother(A, x, b, smoother);
        }

        // compute residual r = b - A*x = A*e
        vector<double> r(N);
        compute_residual(A, x, b, r);

        // compute wres = R*r
        vector<double> wres(Nc);
        wres.zeros();
        R.multiply_by_vector(wres, r);

        // set ec = 0
        vector<double> ec(Nc);
        ec.zeros();

        // recursively solve Ac*ec = R*r = wres
        // std::cout << "currentLevel: " << currentLevel << " N: " << N <<
        // std::endl;
        fcycle(hierarchy, ec, wres, n1, n2, n3, currentLevel + 1, smoother);
        // std::cout << "currentLevel: " << currentLevel << " N: " << N <<
        // std::endl;

        // correct x = x + P*ec
        P.multiply_by_vector_and_add(x, ec);

        // do n2 smoothing steps on A*x=b
        for (int i = 0; i < n2; i++)
        {
            apply_smoother(A, x, b, smoother);
        }

        // compute residual r = b - A*x = A*e
        compute_residual(A, x, b, r);

        // compute wres = R*r
        R.multiply_by_vector(wres, r);

        // recursively solve Ac*ec = R*r = wres
        vcycle(hierarchy, ec, wres, n1, n2, currentLevel + 1, smoother);
        // std::cout << "currentLevel: " << currentLevel << " N: " << N <<
        // std::endl;

        // correct x = x + P*ec
        P.multiply_by_vector_and_add(x, ec);

        // do n3 smoothing steps on A*x=b
        for (int i = 0; i < n3; i++)
        {
            apply_smoother(A, x, b, smoother);
        }
    }
    else
    {
        const int* csr_row_ptr_A = A.get_row_ptr();
        const int* csr_col_ind_A = A.get_col_ind();
        const double* csr_val_A = A.get_val();

        // std::cout << "currentLevel: " << currentLevel << " N: " << N <<
        // std::endl;
        //  solve A*x=b exactly
        for (int i = 0; i < 100; i++)
        {
            // Gauss-Seidel iteration
            for (int j = 0; j < N; j++)
            {
                double sigma = 0.0;
                double ajj = 0.0; // diagonal entry a_jj

                int row_start = csr_row_ptr_A[j];
                int row_end = csr_row_ptr_A[j + 1];

                for (int k = row_start; k < row_end; k++)
                {
                    int col = csr_col_ind_A[k];
                    double val = csr_val_A[k];
                    if (col != j)
                    {
                        sigma = sigma + val * x[col];
                    }
                    else
                    {
                        ajj = val;
                    }
                }
                x[j] = (b[j] - sigma) / ajj;
            }
        }
    }
}
}

int linalg::amg_solve(const hierarchy &hierarchy, vector<double>& x, const vector<double>& b, int n1, int n2, Cycle cycle,
              Smoother smoother, iter_control control)
{           
    ROUTINE_TRACE("amg_solve");

    auto t1 = std::chrono::high_resolution_clock::now();

    const csr_matrix &A = hierarchy.A_cs[0];

    vector<double> residual(A.get_m());
    compute_residual(A, x, b, residual);

    double initial_res_norm = norm_inf(residual);

    // amg recursive solve
    int n3 = n2;
    int cycle_count = 0;
    while (!control.exceed_max_cycle(cycle_count))
    {
        switch (cycle)
        {
        case Cycle::Vcycle:
            vcycle(hierarchy, x, b, n1, n2, 0, smoother);
            break;
        case Cycle::Wcycle:
            wcycle(hierarchy, x, b, n1, n2, n3, 0, smoother);
            break;
        case Cycle::Fcycle:
            fcycle(hierarchy, x, b, n1, n2, n3, 0, smoother);
            break;
        }

        compute_residual(A, x, b, residual);
        double res_norm = norm_inf(residual);

        if (control.residual_converges(res_norm, initial_res_norm))
        {
            break;
        }

        cycle_count++;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "amg solve time: " << ms_double.count() << "ms" << std::endl;

    return cycle_count;
}