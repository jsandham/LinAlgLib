//********************************************************************************
//
// MIT License
//
// Copyright(c) 2019 James Sandham
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

#include "../../../include/LinearSolvers/Krylov/pcg.h"
#include "../../../include/LinearSolvers/slaf.h"
#include "math.h"
#include <iostream>
#include <vector>

//****************************************************************************
//
// Preconditioned Conjugate Gradient
//
//****************************************************************************

#define DEBUG 1

//-------------------------------------------------------------------------------
// preconditioned conjugate gradient
//-------------------------------------------------------------------------------
int pcg(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, int n,
        const preconditioner *precond, double tol, int max_iter, int restart_iter)
{
    // jacobi preconditioner
    std::vector<double> diag(n);
    diagonal(csr_row_ptr, csr_col_ind, csr_val, diag.data(), n);

    // create z and p vector
    std::vector<double> z(n);
    std::vector<double> p(n);

    // res = b - A * x and initial error
    std::vector<double> res(n);

    double gamma = 0.0;

    // start algorithm
    {
        // res = b - A * x
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, x, res.data(), n);
        for (int i = 0; i < n; i++)
        {
            res[i] = b[i] - res[i];
        }

        // z = (M^-1) * res
        precond->solve(res.data(), z.data(), n);

        // p = z
        for (int i = 0; i < n; i++)
        {
            p[i] = z[i];
        }

        gamma = dot_product(z.data(), res.data(), n);
    }

    int iter = 0;
    while (iter < max_iter)
    {
        // restart algorithm to better handle round off error
        if (iter > 0 && iter % restart_iter == 0)
        {
            // res = b - A * x
            matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, x, res.data(), n);
            for (int i = 0; i < n; i++)
            {
                res[i] = b[i] - res[i];
            }

            // z = (M^-1) * res
            precond->solve(res.data(), z.data(), n);

            // p = z
            for (int i = 0; i < n; i++)
            {
                p[i] = z[i];
            }

            gamma = dot_product(z.data(), res.data(), n);
        }

        // z = A * p and alpha = (z, r) / (Ap, p)
        matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, p.data(), z.data(), n);
        double alpha = gamma / dot_product(z.data(), p.data(), n);

        // update x = x + alpha * p
        for (int i = 0; i < n; i++)
        {
            x[i] += alpha * p[i];
        }

        // update res = res - alpha * z
        for (int i = 0; i < n; i++)
        {
            res[i] -= alpha * z[i];
        }

        double err = norm_inf(res.data(), n);

#if (DEBUG)
        std::cout << "error: " << err << std::endl;
#endif

        if (err <= tol)
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
        for (int i = 0; i < n; i++)
        {
            p[i] = z[i] + beta * p[i];
        }

        iter++;
    }

    return iter;
}

// //-------------------------------------------------------------------------------
// // preconditioned conjugate gradient
// //-------------------------------------------------------------------------------
// int pcg2(const int *csr_row_ptr, const int *csr_col_ind, const double *csr_val, double *x, const double *b, const int
// n,
//          const double tol, const int max_iter)
// {
//     double err = error(csr_row_ptr, csr_col_ind, csr_val, x, b, n);
//     if (err < tol)
//     {
//         return 1;
//     }

//     // create w and p vector
//     std::vector<double> w(n);
//     std::vector<double> p(n);

//     // res = b - A * x and initial error
//     std::vector<double> res(n);

//     matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, x, res.data(), n);
//     for (int i = 0; i < n; i++)
//     {
//         res[i] = b[i] - res[i];
//     }

//     // AMG preconditioner setup
//     // int level = 10;
//     // int *matrixSizes = new int[level+1];   // size of A matrix at each level
//     // int **ar = new int*[level+1];          //
//     // int **ac = new int*[level+1];          // pointers to A-matrix at each
//     // level double **av = new double*[level+1];    // and diagonal entries of
//     // A-matrix double **ad = new double*[level+1];    // int **wr = new
//     // int*[level];            // int **wc = new int*[level];            //
//     // pointers to W-matrix at each level double **wv = new double*[level]; //

//     // amg_init(r,c,v,ar,ac,av,ad,matrixSizes,n);
//     // level = amg_setup(ar,ac,av,ad,wr,wc,wv,matrixSizes,level,0.25);

//     int iter = 0, inner_iter = 0;
//     double gamma0 = 1.0, gammai = 1.0;
//     double omega = 1.1, omega2 = 2.0 - omega;
//     while (iter < max_iter && err > tol)
//     {
//         // w = (M^-1)*r
//         for (int i = 0; i < n; i++)
//         {
//             // w[i] = md[i]*r[i];
//             w[i] = res[i];
//         }
//         // amg_solve(ar,ac,av,ad,wr,wc,wv,w,res,matrixSizes,2,2,level,0);

//         // gam = (r,w)
//         double gammai1 = gammai;
//         gammai = 0;
//         for (int i = 0; i < n; i++)
//         {
//             gammai += res[i] * w[i];
//         }
//         if (iter == 0)
//         {
//             gamma0 = gammai;
//             for (int i = 0; i < n; i++)
//             {
//                 p[i] = w[i];
//             }
//         }
//         else
//         {
//             double rg = gammai / gammai1;
//             for (int i = 0; i < n; i++)
//             {
//                 p[i] = w[i] + rg * p[i];
//             }
//         }

//         // w = A*p
//         matrix_vector_product(csr_row_ptr, csr_col_ind, csr_val, p.data(), w.data(), n);
//         double beta = 0;
//         for (int i = 0; i < n; i++)
//         {
//             beta += p[i] * w[i];
//         }
//         double alpha = gammai / beta;

//         // update x and res
//         for (int i = 0; i < n; i++)
//         {
//             x[i] += alpha * p[i];
//             res[i] -= alpha * w[i];
//         }

//         // calculate error
//         if (inner_iter == 1)
//         {
//             err = error(csr_row_ptr, csr_col_ind, csr_val, x, b, n);
//             inner_iter = 0;
// #if (DEBUG)
//             std::cout << "error: " << err << std::endl;
// #endif
//         }
//         iter++;
//         inner_iter++;
//     }

//     return iter;
// }

//-------------------------------------------------------------------------------
// diagonal preconditioner md = diag(A)^-1
//-------------------------------------------------------------------------------
// void diagonalPreconditioner()
//{
//  for(int j=0;j<neq;j++){
//    int i;
//    for(i=nrow[j];i<nrow[j+1];i++)
//      if(ncol[i]==j) break;
//    md[j] = 1.0/A[i];
//    if(A[i]==0){std::cout<<"i: "<<i<<std::endl;}
//  }
//}
