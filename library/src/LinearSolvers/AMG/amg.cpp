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

#include "../../../include/LinearSolvers/AMG/amg.h"
#include "../../../include/LinearSolvers/slaf.h"
#include "math.h"
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

#define DEBUG 1
#define FAST_ERROR 0
#define MAX_VCYCLES 1000

double jacobi_iteration(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x,
                        const double *xold, const double *b, const int n);
double gauss_siedel_iteration(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x,
                              const double *b, const int n);
double symm_gauss_siedel_iteration(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x,
                                   const double *b, const int n);
double sor_iteration(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b,
                     const int n, const double omega);
double ssor_iteration(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b,
                      const int n, const double omega);

static void apply_smoother(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x,
                           const double *b, const int n, Smoother smoother)
{
    switch (smoother)
    {
    case Smoother::Jacobi: {
        std::vector<double> xold(n);
        for (int i = 0; i < n; i++)
        {
            xold[i] = x[i];
        }
        jacobi_iteration(csr_row_ptr, csr_col_ind, csr_val, x, xold.data(), b, n);
        break;
    }
    case Smoother::Gauss_Siedel:
        gauss_siedel_iteration(csr_row_ptr, csr_col_ind, csr_val, x, b, n);
        break;
    case Smoother::Symm_Gauss_Siedel:
        gauss_siedel_iteration(csr_row_ptr, csr_col_ind, csr_val, x, b, n);
        break;
    case Smoother::SOR:
        sor_iteration(csr_row_ptr, csr_col_ind, csr_val, x, b, n, 0.5f);
        break;
    case Smoother::SSOR:
        ssor_iteration(csr_row_ptr, csr_col_ind, csr_val, x, b, n, 0.5f);
        break;
    default:
        break;
    }
}

static void vcycle(const heirarchy &hierarchy, double *x, const double *b, int n1, int n2, int currentLevel,
                   Smoother smoother)
{
    // A_coarse = R*A*P
    const csr_matrix &A = hierarchy.A_cs[currentLevel];

    int N = A.m; // size of A at current level

    if (currentLevel < hierarchy.total_levels)
    {
        const csr_matrix &R = hierarchy.restrictions[currentLevel];
        const csr_matrix &P = hierarchy.prolongations[currentLevel];
        const csr_matrix &A_coarse = hierarchy.A_cs[currentLevel + 1];

        int Nc = A_coarse.m; // size of A at next course level

        assert(Nc == R.m);
        assert(N == R.n);
        assert(N == P.m);
        assert(Nc == P.n);

        // do n1 smoothing steps on A*x=b
        for (int i = 0; i < n1; i++)
        {
            apply_smoother(A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, b, N, smoother);
        }

        // compute residual r = b - A*x = A*e
        std::vector<double> r(N);
        for (int i = 0; i < N; i++)
        {
            r[i] = b[i];
        }
        csrmv(A.m, A.n, A.nnz, -1.0, A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, 1.0, r.data());

        // compute wres = R*r
        std::vector<double> wres(Nc, 0.0);
        csrmv(R.m, R.n, R.nnz, 1.0, R.csr_row_ptr.data(), R.csr_col_ind.data(), R.csr_val.data(), r.data(), 0.0,
              wres.data());

        // set e = 0
        std::vector<double> ec(Nc, 0.0);

        // recursively solve Ac*ec = R*r = wres
        vcycle(hierarchy, ec.data(), wres.data(), n1, n2, currentLevel + 1, smoother);

        // correct x = x + P*ec
        csrmv(P.m, P.n, P.nnz, 1.0, P.csr_row_ptr.data(), P.csr_col_ind.data(), P.csr_val.data(), ec.data(), 1.0, x);

        // do n2 smoothing steps on A*x=b
        for (int i = 0; i < n2; i++)
        {
            apply_smoother(A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, b, N, smoother);
        }
    }
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

                int row_start = A.csr_row_ptr[j];
                int row_end = A.csr_row_ptr[j + 1];

                for (int k = row_start; k < row_end; k++)
                {
                    int col = A.csr_col_ind[k];
                    double val = A.csr_val[k];
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

static void wcycle(const heirarchy &hierarchy, double *x, const double *b, int n1, int n2, int n3, int currentLevel,
                   Smoother smoother)
{
    // A_coarse = R*A*P
    const csr_matrix &A = hierarchy.A_cs[currentLevel];

    int N = A.m; // size of A at current level

    if (currentLevel < hierarchy.total_levels)
    {
        const csr_matrix &R = hierarchy.restrictions[currentLevel];
        const csr_matrix &P = hierarchy.prolongations[currentLevel];
        const csr_matrix &A_coarse = hierarchy.A_cs[currentLevel + 1];

        int Nc = A_coarse.m; // size of A at next course level

        assert(Nc == R.m);
        assert(N == R.n);
        assert(N == P.m);
        assert(Nc == P.n);

        // do n1 smoothing steps on A*x=b
        for (int i = 0; i < n1; i++)
        {
            apply_smoother(A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, b, N, smoother);
        }

        // compute residual r = b - A*x = A*e
        std::vector<double> r(N);
        for (int i = 0; i < N; i++)
        {
            r[i] = b[i];
        }
        csrmv(A.m, A.n, A.nnz, -1.0, A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, 1.0, r.data());

        // compute wres = R*r
        std::vector<double> wres(Nc, 0.0);
        csrmv(R.m, R.n, R.nnz, 1.0, R.csr_row_ptr.data(), R.csr_col_ind.data(), R.csr_val.data(), r.data(), 0.0,
              wres.data());

        // set ec = 0
        std::vector<double> ec(Nc, 0.0);

        // recursively solve Ac*ec = R*r = wres
        // std::cout << "currentLevel: " << currentLevel << " N: " << N <<
        // std::endl;
        wcycle(hierarchy, ec.data(), wres.data(), n1, n2, n3, currentLevel + 1, smoother);
        // std::cout << "currentLevel: " << currentLevel << " N: " << N <<
        // std::endl;

        // correct x = x + P*ec
        csrmv(P.m, P.n, P.nnz, 1.0, P.csr_row_ptr.data(), P.csr_col_ind.data(), P.csr_val.data(), ec.data(), 1.0, x);

        // do n2 smoothing steps on A*x=b
        for (int i = 0; i < n2; i++)
        {
            apply_smoother(A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, b, N, smoother);
        }

        // compute residual r = b - A*x = A*e
        for (int i = 0; i < N; i++)
        {
            r[i] = b[i];
        }
        csrmv(A.m, A.n, A.nnz, -1.0, A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, 1.0, r.data());

        // compute wres = R*r
        csrmv(R.m, R.n, R.nnz, 1.0, R.csr_row_ptr.data(), R.csr_col_ind.data(), R.csr_val.data(), r.data(), 0.0,
              wres.data());

        // recursively solve Ac*ec = R*r = wres
        wcycle(hierarchy, ec.data(), wres.data(), n1, n2, n3, currentLevel + 1, smoother);
        // std::cout << "currentLevel: " << currentLevel << " N: " << N <<
        // std::endl;

        // correct x = x + P*ec
        csrmv(P.m, P.n, P.nnz, 1.0, P.csr_row_ptr.data(), P.csr_col_ind.data(), P.csr_val.data(), ec.data(), 1.0, x);

        // do n3 smoothing steps on A*x=b
        for (int i = 0; i < n3; i++)
        {
            apply_smoother(A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, b, N, smoother);
        }
    }
    else
    {
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

                int row_start = A.csr_row_ptr[j];
                int row_end = A.csr_row_ptr[j + 1];

                for (int k = row_start; k < row_end; k++)
                {
                    int col = A.csr_col_ind[k];
                    double val = A.csr_val[k];
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

static void fcycle(const heirarchy &hierarchy, double *x, const double *b, int n1, int n2, int n3, int currentLevel,
                   Smoother smoother)
{
    // A_coarse = R*A*P
    const csr_matrix &A = hierarchy.A_cs[currentLevel];

    int N = A.m; // size of A at current level

    if (currentLevel < hierarchy.total_levels)
    {
        const csr_matrix &R = hierarchy.restrictions[currentLevel];
        const csr_matrix &P = hierarchy.prolongations[currentLevel];
        const csr_matrix &A_coarse = hierarchy.A_cs[currentLevel + 1];

        int Nc = A_coarse.m; // size of A at next course level

        assert(Nc == R.m);
        assert(N == R.n);
        assert(N == P.m);
        assert(Nc == P.n);

        // do n1 smoothing steps on A*x=b
        for (int i = 0; i < n1; i++)
        {
            apply_smoother(A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, b, N, smoother);
        }

        // compute residual r = b - A*x = A*e
        std::vector<double> r(N);
        for (int i = 0; i < N; i++)
        {
            r[i] = b[i];
        }
        csrmv(A.m, A.n, A.nnz, -1.0, A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, 1.0, r.data());

        // compute wres = R*r
        std::vector<double> wres(Nc, 0.0);
        csrmv(R.m, R.n, R.nnz, 1.0, R.csr_row_ptr.data(), R.csr_col_ind.data(), R.csr_val.data(), r.data(), 0.0,
              wres.data());

        // set ec = 0
        std::vector<double> ec(Nc, 0.0);

        // recursively solve Ac*ec = R*r = wres
        // std::cout << "currentLevel: " << currentLevel << " N: " << N <<
        // std::endl;
        fcycle(hierarchy, ec.data(), wres.data(), n1, n2, n3, currentLevel + 1, smoother);
        // std::cout << "currentLevel: " << currentLevel << " N: " << N <<
        // std::endl;

        // correct x = x + P*ec
        csrmv(P.m, P.n, P.nnz, 1.0, P.csr_row_ptr.data(), P.csr_col_ind.data(), P.csr_val.data(), ec.data(), 1.0, x);

        // do n2 smoothing steps on A*x=b
        for (int i = 0; i < n2; i++)
        {
            apply_smoother(A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, b, N, smoother);
        }

        // compute residual r = b - A*x = A*e
        for (int i = 0; i < N; i++)
        {
            r[i] = b[i];
        }
        csrmv(A.m, A.n, A.nnz, -1.0, A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, 1.0, r.data());

        // compute wres = R*r
        csrmv(R.m, R.n, R.nnz, 1.0, R.csr_row_ptr.data(), R.csr_col_ind.data(), R.csr_val.data(), r.data(), 0.0,
              wres.data());

        // recursively solve Ac*ec = R*r = wres
        vcycle(hierarchy, ec.data(), wres.data(), n1, n2, currentLevel + 1, smoother);
        // std::cout << "currentLevel: " << currentLevel << " N: " << N <<
        // std::endl;

        // correct x = x + P*ec
        csrmv(P.m, P.n, P.nnz, 1.0, P.csr_row_ptr.data(), P.csr_col_ind.data(), P.csr_val.data(), ec.data(), 1.0, x);

        // do n3 smoothing steps on A*x=b
        for (int i = 0; i < n3; i++)
        {
            apply_smoother(A.csr_row_ptr.data(), A.csr_col_ind.data(), A.csr_val.data(), x, b, N, smoother);
        }
    }
    else
    {
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

                int row_start = A.csr_row_ptr[j];
                int row_end = A.csr_row_ptr[j + 1];

                for (int k = row_start; k < row_end; k++)
                {
                    int col = A.csr_col_ind[k];
                    double val = A.csr_val[k];
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

int amg_solve(const heirarchy &hierarchy, double *x, const double *b, int n1, int n2, double tol, Cycle cycle,
              Smoother smoother)
{
    // AMG recursive solve
    int n3 = n2;
    int cycle_count = 0;
    double err = 1.0;
    while (err > tol && cycle_count < MAX_VCYCLES)
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
#if (FAST_ERROR)
        err = fast_error(hierarchy.A_cs[0].csr_row_ptr.data(), hierarchy.A_cs[0].csr_col_ind.data(),
                         hierarchy.A_cs[0].csr_val.data(), x, b, hierarchy.A_cs[0].m, tol);
#else
        err = error(hierarchy.A_cs[0].csr_row_ptr.data(), hierarchy.A_cs[0].csr_col_ind.data(),
                    hierarchy.A_cs[0].csr_val.data(), x, b, hierarchy.A_cs[0].m);
#endif
#if (DEBUG)
        std::cout << "error: " << err << std::endl;
#endif
        cycle_count++;
    }

    std::cout << "cycles: " << cycle_count << std::endl;

    return err > tol ? -1 : cycle_count;
}